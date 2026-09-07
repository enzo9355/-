"""Pre-market overlay pipeline that never mutates post-close core evidence."""

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zoneinfo

from stock_papi.batch.runtime import acquire_job_lock, job_namespace
from stock_papi.batch.status import PipelineStatusWriter


NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
OVERNIGHT_SYMBOLS = ("SPY", "QQQ", "TSM", "UMC", "ASX")


def _valid_overlay(value):
    if not isinstance(value, dict):
        return False
    symbols = value.get("symbols")
    return (
        value.get("status") in {"risk_on", "risk_off", "neutral"}
        and isinstance(value.get("message"), str)
        and isinstance(value.get("as_of"), str)
        and isinstance(value.get("previous_as_of"), str)
        and re.fullmatch(
            r"quant/v1/manifests/US-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json",
            str(value.get("source_manifest") or ""),
        )
        is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("source_manifest_sha256") or ""))
        is not None
        and isinstance(symbols, list)
        and len(symbols) == len(OVERNIGHT_SYMBOLS)
        and {item.get("symbol") for item in symbols if isinstance(item, dict)}
        == set(OVERNIGHT_SYMBOLS)
        and all(
            isinstance(item, dict)
            and type(item.get("return_pct")) in (int, float)
            and item.get("direction") in {"up", "down", "unchanged"}
            and item.get("as_of") == value.get("as_of")
            for item in symbols
        )
    )


class PreMarketPipelineError(RuntimeError):
    """盤後 base、overlay source 或 checkpoint 不合法。"""


def _canonical(document):
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreMarketPipelineError("pre-market document is invalid") from exc


def _write_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical(document))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class PreMarketPipeline:
    def __init__(
        self,
        root,
        *,
        applicable_trading_date,
        load_base,
        source_loaders,
        publish,
        notify,
        us_source_loader=None,
        us_calendars=None,
    ):
        if (
            type(applicable_trading_date) is not datetime.date
            or not callable(load_base)
            or not isinstance(source_loaders, (list, tuple))
            or not all(callable(loader) for loader in source_loaders)
            or not callable(publish)
            or not callable(notify)
            or (us_source_loader is not None and not callable(us_source_loader))
        ):
            raise ValueError("invalid pre-market pipeline configuration")
        self.root = Path(root)
        self.applicable_trading_date = applicable_trading_date
        self.load_base = load_base
        self.source_loaders = tuple(source_loaders)
        self.publish = publish
        self.notify = notify
        self.us_source_loader = us_source_loader
        self.us_calendars = us_calendars

    def _base(self):
        receipt = self.load_base()
        if not isinstance(receipt, dict) or not isinstance(receipt.get("metadata"), dict):
            raise PreMarketPipelineError("verified post-close base is missing")
        metadata = receipt["metadata"]
        digest = receipt.get("metadata_sha256")
        capability = metadata.get("prediction_capability")
        if (
            metadata.get("schema_version") != 2
            or metadata.get("kind") not in {"absorb-report", "stock-papi-report"}
            or metadata.get("product_mode") != "observation"
            or metadata.get("report_type") != "post_close"
            or metadata.get("market") != "TW"
            or metadata.get("applicable_trading_date")
            != self.applicable_trading_date.isoformat()
            or hashlib.sha256(_canonical(metadata)).hexdigest() != digest
            or not isinstance(metadata.get("content"), dict)
            or metadata.get("model_versions") != {}
            or metadata.get("backtest_as_of") is not None
            or metadata.get("observation_start_date")
            != metadata.get("source_market_date")
            or metadata.get("observation_end_date")
            != metadata.get("applicable_trading_date")
            or not isinstance(capability, dict)
            or capability.get("mode") != "research"
            or capability.get("observation_enabled") is not True
            or capability.get("probability_allowed") is not False
            or capability.get("ranking_allowed") is not False
            or capability.get("strong_action_allowed") is not False
            or capability.get("performance_endorsement_allowed") is not False
        ):
            raise PreMarketPipelineError("verified post-close base is invalid")
        return receipt

    def _overnight_observation(self, now):
        if self.us_source_loader is None or self.us_calendars is None:
            raise PreMarketPipelineError("verified US quant source is missing")
        try:
            source = self.us_source_loader()
            manifest = source.manifest
            if (
                manifest.market != "US"
                or not re.fullmatch(
                    r"manifests/US-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json",
                    str(manifest.manifest_path),
                )
                or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.manifest_sha256))
            ):
                raise ValueError("US manifest identity is invalid")
            ny_now = now.astimezone(NEW_YORK)
            completed = self.us_calendars.latest_session_on_or_before(
                ny_now.date() if ny_now.time() >= datetime.time(16) else ny_now.date() - datetime.timedelta(days=1)
            )
            previous = self.us_calendars.session_offset(completed, -1)
            if manifest.market_as_of != completed:
                raise ValueError("US quant source is not the latest completed session")
            by_symbol = {stock.symbol: stock for stock in source.stocks}
            if not set(OVERNIGHT_SYMBOLS).issubset(by_symbol):
                raise ValueError("US overnight universe is incomplete")
            rows = {}
            for symbol in OVERNIGHT_SYMBOLS:
                stock = by_symbol[symbol]
                if stock.observation_kind != "regular_price" or stock.as_of != completed:
                    raise ValueError("US overnight row date is invalid")
                points = {}
                for row in stock.daily:
                    date_text = str(row.get("Date") or "").split("T", 1)[0]
                    try:
                        date = datetime.date.fromisoformat(date_text)
                    except ValueError:
                        continue
                    close = row.get("Close")
                    if isinstance(close, (int, float)) and not isinstance(close, bool) and close > 0:
                        points[date] = float(close)
                if completed not in points or previous not in points:
                    raise ValueError("US overnight rows do not contain two valid sessions")
                change = (points[completed] / points[previous] - 1) * 100
                if not math.isfinite(change):
                    raise ValueError("US overnight return is invalid")
                rows[symbol] = {
                    "symbol": symbol,
                    "return_pct": round(change, 4),
                    "direction": "up" if change > 0 else "down" if change < 0 else "unchanged",
                    "as_of": completed.isoformat(),
                }
            rising = sum(item["direction"] == "up" for item in rows.values())
            falling = sum(item["direction"] == "down" for item in rows.values())
            status = "risk_on" if rising >= 4 else "risk_off" if falling >= 4 else "neutral"
            message = {
                "risk_on": "隔夜觀察偏正向",
                "risk_off": "隔夜觀察偏保守",
                "neutral": "隔夜觀察中性",
            }[status]
            return {
                "status": status,
                "message": message,
                "symbols": [rows[symbol] for symbol in OVERNIGHT_SYMBOLS],
                "as_of": completed.isoformat(),
                "previous_as_of": previous.isoformat(),
                "source_manifest": f"quant/v1/{manifest.manifest_path}",
                "source_manifest_sha256": manifest.manifest_sha256,
            }
        except Exception as exc:
            raise PreMarketPipelineError("verified US overnight observation is unavailable") from exc

    def run(self, *, now=None):
        checked_at = now or datetime.datetime.now(datetime.timezone.utc)
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        base = self._base()
        base_metadata = base["metadata"]
        identity = {
            "applicable_trading_date": self.applicable_trading_date.isoformat(),
            "base_metadata_sha256": base["metadata_sha256"],
        }
        digest = hashlib.sha256(_canonical(identity)).hexdigest()[:8]
        run_id = f"{self.applicable_trading_date.strftime('%Y%m%d')}T000000Z-{digest}"
        current = job_namespace(self.root, "pre_market_update").checkpoint
        checkpoint_path = current.with_name(f"{run_id}.json")
        writer = PipelineStatusWriter(
            self.root,
            job_type="pre_market_update",
            run_id=run_id,
            target_date=self.applicable_trading_date,
        )
        with acquire_job_lock(
            self.root,
            "pre_market_update",
            self.applicable_trading_date,
            now=checked_at,
        ):
            if checkpoint_path.exists():
                try:
                    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise PreMarketPipelineError("pre-market checkpoint is invalid") from exc
                if any(state.get(key) != value for key, value in identity.items()):
                    raise PreMarketPipelineError("pre-market checkpoint identity mismatch")
                if state.get("status") == "completed":
                    overlay = (
                        state.get("outputs", {}).get("metadata", {})
                        .get("content", {})
                        .get("overnight_overlay")
                    )
                    if _valid_overlay(overlay):
                        return state
                    raise PreMarketPipelineError("pre-market checkpoint is from an obsolete contract")
            else:
                state = {
                    "schema_version": 1,
                    "job_type": "pre_market_update",
                    "run_id": run_id,
                    **identity,
                    "completed_stages": [],
                    "outputs": {},
                    "status": "running",
                }

            def save():
                state["updated_at"] = checked_at.astimezone(
                    datetime.timezone.utc
                ).isoformat().replace("+00:00", "Z")
                _write_atomic(checkpoint_path, state)

            try:
                if "metadata" not in state["completed_stages"]:
                    overnight = self._overnight_observation(checked_at)
                    base_core = base_metadata["content"]
                    market = base_core.get("market_observation")
                    quality = base_core.get("data_quality", {})
                    if not isinstance(market, dict):
                        raise PreMarketPipelineError("TW benchmark summary is missing")
                    core = {
                        "market_observation": dict(market),
                        "data_quality": dict(quality) if isinstance(quality, dict) else {},
                        "daily_focus": list(base_core.get("daily_focus") or [])[:1],
                    }
                    metadata = {
                        "schema_version": 2,
                        "product_mode": "observation",
                        "report_type": "pre_market",
                        "market": "TW",
                        "source_market_date": base_metadata["source_market_date"],
                        "applicable_trading_date": base_metadata[
                            "applicable_trading_date"
                        ],
                        "published_at": checked_at.astimezone(
                            datetime.timezone.utc
                        ).isoformat().replace("+00:00", "Z"),
                        "forecast_start_date": base_metadata["forecast_start_date"],
                        "forecast_end_date": base_metadata["forecast_end_date"],
                        "observation_start_date": base_metadata[
                            "observation_start_date"
                        ],
                        "observation_end_date": base_metadata[
                            "observation_end_date"
                        ],
                        "backtest_as_of": None,
                        "data_as_of": base_metadata["data_as_of"],
                        "source_manifest": base_metadata["source_manifest"],
                        "source_manifest_sha256": base_metadata[
                            "source_manifest_sha256"
                        ],
                        "model_versions": {},
                        "prediction_capability": dict(
                            base_metadata["prediction_capability"]
                        ),
                        "title": "ABSORB 盤前風險更新",
                        "summary": [overnight["message"]],
                        "warnings": [],
                        "content": {
                            "core": core,
                            "base_metadata_sha256": base["metadata_sha256"],
                            "overnight_overlay": {
                                **overnight,
                            },
                        },
                    }
                    state["outputs"]["metadata"] = metadata
                    state["completed_stages"].append("metadata")
                    save()
                    writer.record("aggregation", now=checked_at)
                metadata = state["outputs"]["metadata"]
                overlay = (
                    metadata.get("content", {}).get("overnight_overlay")
                    if isinstance(metadata, dict) and isinstance(metadata.get("content"), dict)
                    else None
                )
                if not _valid_overlay(overlay):
                    raise PreMarketPipelineError("pre-market checkpoint is from an obsolete contract")
                if "publish" not in state["completed_stages"]:
                    receipt = self.publish(metadata)
                    _canonical(receipt)
                    state["outputs"]["publish"] = receipt
                    state["completed_stages"].append("publish")
                    save()
                    writer.record("publish", now=checked_at)
                receipt = state["outputs"]["publish"]
                if "notify" not in state["completed_stages"]:
                    notification = self.notify(receipt)
                    _canonical(notification)
                    state["outputs"]["notify"] = notification
                    state["completed_stages"].append("notify")
                    save()
                    writer.record("notify", now=checked_at)
                state["status"] = "completed"
                save()
                writer.record("completed", now=checked_at)
                return state
            except Exception as exc:
                state["status"] = "failed"
                state["last_error_type"] = type(exc).__name__
                save()
                writer.record("failed", now=checked_at, error=exc)
                raise
