import argparse
import contextlib
import datetime
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import time
import csv
from pathlib import Path

def safe_print(*args, **kwargs):
    file = kwargs.get("file", sys.stdout)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")

    text = sep.join(str(arg) for arg in args) + end
    try:
        encoding = getattr(file, "encoding", None) or "utf-8"
        file.write(text.encode(encoding, errors="replace").decode(encoding))
        file.flush()
    except Exception:
        try:
            file.write(text)
            file.flush()
        except Exception:
            pass

from market_insights import (
    ETF_CATALOG,
    SUPPLY_CHAINS,
    build_industries,
    build_supply_chains,
    normalize_etf_holdings,
    parse_mops_items,
)
from stock_papi.integrations.market_data.provider import FinMindFetchError
from stock_papi.quant.features import CALCULATED_COLUMNS


TAIPEI = datetime.timezone(datetime.timedelta(hours=8), "Asia/Taipei")
RUN_START = datetime.time(2, 30)
US_RUN_START = datetime.time(5, 30)
DRAIN_START = datetime.time(9, 20)
CHECKPOINT_START = datetime.time(9, 25)
RUN_END = datetime.time(9, 30)
LAYOUT_DIRS = (
    "raw", "cache", "checkpoints", "artifacts", "publish", "logs", "secrets",
)
RETENTION_DAYS = {
    "cache/tmp": 1,
    "cache/pycache": 30,
    "raw": 30,
    "logs": 30,
    "publish": 30,
}
SEC_US_UNIVERSE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_US_UNIVERSE_MAX_BYTES = 5 * 1024 * 1024
NASDAQ_US_UNIVERSE_URLS = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
)
STOCK_ARTIFACT_MAX_COMPRESSED_BYTES = 5 * 1024 * 1024
STOCK_ARTIFACT_MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
US_EXCHANGES = {"Nasdaq", "NYSE", "CBOE"}
CRYPTO_SECURITY_TERMS = (
    "bitcoin", "ethereum", "crypto", "solana", "litecoin", "dogecoin",
)
OBSERVATION_SOURCE_VERSION = "observation-source-v1"
OBSERVATION_MODEL_COLUMNS = ("AI_P", "FUTURE_RET_5", "T")
MOPS_SOURCES = (
    ("TWSE", "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"),
    ("TPEx", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"),
)
MARKET_INSIGHTS_MAX_BYTES = 5 * 1024 * 1024


def validate_data_root(path):
    path = Path(path).expanduser()
    if (
        path.drive.upper() != "D:"
        or path.parent != Path("D:/")
        or path.name.lower() not in {"absorbdata", "stockpapidata"}
    ):
        raise ValueError(r"data root must be D:\AbsorbData or the deprecated D:\StockPapiData")
    return path


def window_phase(now=None):
    current = (now or datetime.datetime.now(TAIPEI)).astimezone(TAIPEI).time()
    if RUN_START <= current < DRAIN_START:
        return "run"
    if DRAIN_START <= current < CHECKPOINT_START:
        return "drain"
    if CHECKPOINT_START <= current < RUN_END:
        return "checkpoint"
    return "closed"


def market_run_allowed(market, now):
    local_time = now.astimezone(TAIPEI).time()
    return window_phase(now) == "run" and (
        market != "US" or local_time >= US_RUN_START
    )


def ensure_layout(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in LAYOUT_DIRS:
        (root / name).mkdir(exist_ok=True)
    return root


def check_free_space(root, min_free_gb=100.0, free_bytes=None):
    free_bytes = shutil.disk_usage(Path(root)).free if free_bytes is None else free_bytes
    if free_bytes < min_free_gb * 1024**3:
        raise RuntimeError(f"D drive requires at least {min_free_gb:g} GB free")
    return free_bytes


def _is_reparse_point(entry, metadata):
    return entry.is_symlink() or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _cleanup_tree(directory, cutoff, root, summary):
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return
    except OSError:
        summary["failed"] += 1
        return
    for entry in entries:
        path = Path(entry.path)
        try:
            metadata = entry.stat(follow_symlinks=False)
            if _is_reparse_point(entry, metadata):
                summary["skipped_reparse_points"] += 1
                continue
            if not path.resolve(strict=False).is_relative_to(root):
                raise RuntimeError("cleanup path escaped data root")
            if entry.is_dir(follow_symlinks=False):
                _cleanup_tree(path, cutoff, root, summary)
                try:
                    path.rmdir()
                except OSError:
                    pass
            elif entry.is_file(follow_symlinks=False) and metadata.st_mtime < cutoff:
                path.unlink()
                summary["deleted_files"] += 1
                summary["reclaimed_bytes"] += metadata.st_size
        except RuntimeError:
            raise
        except OSError:
            summary["failed"] += 1


def cleanup_expired_data(root, now=None):
    root = validate_data_root(root)
    root_resolved = root.resolve(strict=True)
    checked_at = now or datetime.datetime.now(TAIPEI)
    summary = {
        "deleted_files": 0,
        "reclaimed_bytes": 0,
        "failed": 0,
        "skipped_reparse_points": 0,
    }
    for relative, days in RETENTION_DAYS.items():
        cutoff = (checked_at - datetime.timedelta(days=days)).timestamp()
        _cleanup_tree(root / Path(relative), cutoff, root_resolved, summary)

    checkpoints = root / "checkpoints"
    cutoff = (checked_at - datetime.timedelta(days=7)).timestamp()
    try:
        entries = list(os.scandir(checkpoints))
    except OSError:
        summary["failed"] += 1
        return summary
    for entry in entries:
        if not re.fullmatch(r"runner\.lock\.stale\.\d{8}T\d{6}", entry.name):
            continue
        try:
            metadata = entry.stat(follow_symlinks=False)
            if _is_reparse_point(entry, metadata):
                summary["skipped_reparse_points"] += 1
            elif entry.is_file(follow_symlinks=False) and metadata.st_mtime < cutoff:
                Path(entry.path).unlink()
                summary["deleted_files"] += 1
                summary["reclaimed_bytes"] += metadata.st_size
        except OSError:
            summary["failed"] += 1
    return summary


class RunnerLock:
    def __init__(self, path, token):
        self.path = Path(path)
        self.token = token

    def release(self):
        if not self.path.exists():
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("runner lock is unreadable") from exc
        if current.get("token") != self.token:
            raise RuntimeError("runner lock ownership changed")
        self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.release()


def acquire_lock(root, now=None, stale_after=datetime.timedelta(hours=6)):
    now = now or datetime.datetime.now(TAIPEI)
    lock_path = Path(root) / "checkpoints" / "runner.lock"
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            started_at = datetime.datetime.fromisoformat(existing["started_at"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("existing runner lock is invalid") from exc
        if now - started_at <= stale_after:
            raise RuntimeError("local quant runner is already active")
        archive = lock_path.with_name(
            f"runner.lock.stale.{now.strftime('%Y%m%dT%H%M%S')}"
        )
        os.replace(lock_path, archive)

    token = secrets.token_hex(16)
    payload = json.dumps(
        {"pid": os.getpid(), "token": token, "started_at": now.isoformat()},
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError as exc:
        raise RuntimeError("local quant runner is already active") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return RunnerLock(lock_path, token)


def _checkpoint_path(root, market="TW"):
    if market == "TW":
        filename = "progress.json"
    elif market == "US":
        filename = "progress-US.json"
    else:
        raise ValueError("unsupported market")
    return Path(root) / "checkpoints" / filename


def save_checkpoint(root, state, market="TW"):
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a dictionary")
    checkpoint = _checkpoint_path(root, market)
    _write_json_atomic(checkpoint, state)


def prepare_daily_checkpoint(root, requested):
    from stock_papi.batch.contracts import ContractError, DailyRunCheckpoint

    desired = DailyRunCheckpoint.from_dict(requested)
    directory = Path(root) / "checkpoints" / "daily_prediction"
    directory.mkdir(parents=True, exist_ok=True)
    current_path = directory / "current.json"
    if not current_path.exists():
        _write_json_atomic(current_path, desired.to_dict())
        return desired.to_dict()

    try:
        current_document = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("daily checkpoint 無法讀取") from exc
    current = DailyRunCheckpoint.from_dict(current_document)
    try:
        current.assert_resume_compatible(
            target_market_date=desired.target_market_date,
            source_manifest_sha256=desired.source_manifest_sha256,
            model_version=desired.model_version,
        )
    except ContractError:
        archive_dir = directory / "archive"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"{current.run_id}.json"
        if archive_path.exists():
            try:
                archived = json.loads(archive_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ContractError("daily checkpoint 封存檔無法讀取") from exc
            if archived != current.to_dict():
                raise ContractError("daily checkpoint 封存衝突")
        else:
            _write_json_atomic(archive_path, current.to_dict())
        _write_json_atomic(current_path, desired.to_dict())
        return desired.to_dict()
    return current.to_dict()


def _write_json_atomic(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_bytes_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_gzip_json_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def validate_market_symbol(market, symbol):
    symbol = str(symbol)
    valid = (
        market == "TW" and bool(re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", symbol.upper()))
    ) or (
        market == "US"
        and len(symbol) <= 10
        and bool(re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?", symbol))
    )
    if not valid:
        raise ValueError("invalid market symbol")
    return symbol


def write_stock_artifact(root, market, symbol, payload):
    symbol = validate_market_symbol(market, symbol)
    if not isinstance(payload, dict):
        raise TypeError("stock artifact payload must be a dictionary")
    document = dict(payload)
    schema_version = document.get("schema_version", 1)
    if schema_version not in {1, 2} or (schema_version == 2 and market not in {"TW", "US"}):
        raise ValueError("stock artifact schema version is invalid")
    document.update(
        schema_version=schema_version, market=market, symbol=symbol
    )
    _validate_json_value(document)
    target = Path(root) / "artifacts" / "stocks" / market / f"{symbol}.json.gz"
    _write_gzip_json_atomic(target, document)
    return target


def _validated_artifact(
    root, market, symbol, generated_at, target_market_date=None
):
    symbol = validate_market_symbol(market, symbol)
    path = Path(root) / "artifacts" / "stocks" / market / f"{symbol}.json.gz"
    if not path.is_file():
        raise RuntimeError(f"artifact is missing for {market}:{symbol}")
    compressed_size = path.stat().st_size
    if not 0 < compressed_size <= STOCK_ARTIFACT_MAX_COMPRESSED_BYTES:
        raise RuntimeError(f"artifact is invalid for {market}:{symbol}")
    compressed = path.read_bytes()
    if len(compressed) != compressed_size:
        raise RuntimeError(f"artifact is invalid for {market}:{symbol}")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            decoded = stream.read(STOCK_ARTIFACT_MAX_UNCOMPRESSED_BYTES + 1)
        if len(decoded) > STOCK_ARTIFACT_MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError(f"artifact is too large for {market}:{symbol}")
        document = json.loads(decoded.decode("utf-8"))
        schema_version = document.get("schema_version")
        if target_market_date is None:
            if (
                schema_version != 1
                or document.get("market") != market
                or document.get("symbol") != symbol
            ):
                raise ValueError("artifact schema mismatch")
            _validate_json_value(document)
            as_of = datetime.date.fromisoformat(str(document["as_of"]))
            if as_of > generated_at.astimezone(TAIPEI).date():
                raise ValueError("artifact date is in the future")
            if any(
                key in document
                for key in (
                    "target_market_date",
                    "observation_as_of",
                    "latest_regular_price_date",
                    "observation_kind",
                    "trading_status_evidence",
                )
            ):
                raise ValueError("schema v1 artifact carries observation status")
        else:
            if (
                schema_version != 2
                or document.get("market") != market
                or document.get("symbol") != symbol
                or not isinstance(document.get("daily"), list)
                or not isinstance(document.get("backtest"), dict)
            ):
                raise ValueError("artifact schema mismatch")
            _validate_json_value(document)
            if market == "TW":
                from stock_papi.integrations.market_data.tw_trading_status import (
                    evidence_sha256,
                    validate_status_evidence,
                )
            else:
                from stock_papi.integrations.market_data.us_trading_status import (
                    evidence_sha256,
                    validate_us_status_evidence as validate_status_evidence,
                )

            as_of = str(document.get("as_of") or "")
            daily = document["daily"]
            if not daily or not isinstance(daily[-1], dict):
                raise ValueError("artifact daily rows are invalid")
            latest_date = str(daily[-1].get("Date") or "").split("T", 1)[0]
            target_text = target_market_date.isoformat()
            observation_kind = document.get("observation_kind")
            lineage = document.get("source_lineage") or document.get("lineage")
            status = document.get("trading_status_evidence")
            latest = document.get("latest")
            latest_daily = (
                str(daily[-1].get("Date") or "").split("T", 1)[0]
                if isinstance(daily, list) and daily and isinstance(daily[-1], dict)
                else ""
            )
            latest_summary = (
                str(latest.get("Date") or "").split("T", 1)[0]
                if isinstance(latest, dict)
                else ""
            )
            expected_lineage_version = (
                "tw-official-historical-v3"
                if market == "TW"
                else lineage.get("source_schema_version")
                if isinstance(lineage, dict)
                else ""
            )
            if (
                market not in ("TW", "US")
                or target_market_date > generated_at.astimezone(TAIPEI).date()
                or document.get("target_market_date") != target_text
                or document.get("observation_as_of") != target_text
                or latest_date != as_of
                or latest_daily != document.get("as_of")
                or latest_summary != document.get("as_of")
                or not isinstance(lineage, dict)
                or lineage.get("source_schema_version") != expected_lineage_version
                or lineage.get("observation_as_of") != target_text
                or lineage.get("latest_regular_price_date")
                != document.get("as_of")
                or lineage.get("observation_kind") != observation_kind
            ):
                raise ValueError("observation artifact mismatch")
            if observation_kind == "regular_price":
                if as_of != target_text or status is not None:
                    raise ValueError("regular price observation mismatch")
            elif observation_kind in {
                "official_no_regular_trade",
                "officially_suspended",
            }:
                if (
                    datetime.date.fromisoformat(as_of) >= target_market_date
                    or not isinstance(status, dict)
                    or status.get("schema_version") != 1
                    or status.get("status") != observation_kind
                    or status.get("market") != market
                    or status.get("symbol") != symbol
                    or status.get("target_market_date") != target_text
                    or status.get("evidence_sha256") != evidence_sha256(status)
                    or validate_status_evidence(
                        status,
                        symbol=symbol,
                        target_date=target_market_date,
                    )
                    != status
                    or lineage.get("trading_status_evidence_sha256")
                    != status.get("evidence_sha256")
                ):
                    raise ValueError("trading status observation mismatch")
            else:
                raise ValueError(f"unknown {market} observation kind")
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"artifact is invalid for {market}:{symbol}") from exc
    return path, compressed, document, len(decoded)


def publish_market_snapshot(
    root,
    market,
    symbols,
    generated_at=None,
    failed_symbols=(),
    *,
    target_market_date=None,
    unavailable_symbols=(),
):
    if market not in ("TW", "US"):
        raise ValueError("unsupported market")
    if target_market_date is not None:
        if not isinstance(target_market_date, datetime.date):
            target_market_date = datetime.date.fromisoformat(
                str(target_market_date)
            )
    symbols = sorted({validate_market_symbol(market, symbol) for symbol in symbols})
    if not symbols:
        raise ValueError("market universe is empty")
    failed = {
        validate_market_symbol(market, symbol) for symbol in failed_symbols
    }
    unavailable = {
        validate_market_symbol(market, symbol) for symbol in unavailable_symbols
    }
    if not failed.issubset(symbols) or not unavailable.issubset(symbols):
        raise ValueError("failed symbols must belong to market universe")
    if target_market_date is not None and failed - unavailable:
        raise RuntimeError(
            "operational symbol failures are not publishable: "
            + ", ".join(sorted(failed - unavailable))
        )
    excluded = failed | unavailable
    generated_at = generated_at or datetime.datetime.now(TAIPEI)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=TAIPEI)
    generated_text = generated_at.astimezone(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    publish_root = Path(root) / "publish" / "quant" / "v1"
    object_root = publish_root / "objects"
    candidates = {}
    errors = []
    for symbol in symbols:
        if symbol in excluded:
            continue
        try:
            path, compressed, document, uncompressed_size = _validated_artifact(
                root, market, symbol, generated_at, target_market_date
            )
        except RuntimeError as exc:
            excluded.add(symbol)
            errors.append(str(exc))
            continue
        candidates[symbol] = {
            "source": path,
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "size": len(compressed),
            "uncompressed_size": uncompressed_size,
            "as_of": document["as_of"],
            "model_version": (
                str(document["model_version"])
                if document.get("model_version") is not None
                else None
            ),
        }
        if target_market_date is not None:
            candidates[symbol].update(
                observation_as_of=document["observation_as_of"],
                latest_regular_price_date=document["latest_regular_price_date"],
                observation_kind=document["observation_kind"],
                trading_status_evidence=document.get(
                    "trading_status_evidence"
                ),
            )

    if target_market_date is not None and errors:
        raise RuntimeError(errors[0])
    if target_market_date is not None:
        market_as_of = None
    elif candidates:
        market_as_of = max(item["as_of"] for item in candidates.values())
        if market == "TW":
            for symbol in list(candidates):
                if candidates[symbol]["as_of"] != market_as_of:
                    excluded.add(symbol)
                    del candidates[symbol]
        else: # US market - allow rolling 3 days
            target_date = generated_at.astimezone(TAIPEI).date()
            for symbol in list(candidates):
                candidate_date = datetime.date.fromisoformat(candidates[symbol]["as_of"])
                if (target_date - candidate_date).days > 3:
                    excluded.add(symbol)
                    del candidates[symbol]
    else:
        market_as_of = None
    failure_rate = len(excluded) / len(symbols)
    threshold_percent = 5
    if len(excluded) * 100 >= len(symbols) * threshold_percent or not candidates:
        detail = f"; {errors[0]}" if errors else ""
        raise RuntimeError(
            f"market failure rate {failure_rate:.2%} is not publishable{detail}"
        )

    entries = {}
    for symbol, candidate in candidates.items():
        compressed = candidate["source"].read_bytes()
        digest = candidate["sha256"]
        if len(compressed) != candidate["size"] or hashlib.sha256(compressed).hexdigest() != digest:
            raise RuntimeError(f"artifact changed during publish for {market}:{symbol}")
        object_path = object_root / f"{digest}.json.gz"
        if object_path.exists():
            if (
                object_path.stat().st_size != len(compressed)
                or _sha256_path(object_path) != digest
            ):
                raise RuntimeError("published object hash mismatch")
            os.utime(object_path, None)
        else:
            _write_bytes_atomic(object_path, compressed)
        entries[symbol] = {
            "path": f"objects/{digest}.json.gz",
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": candidate["uncompressed_size"],
            "as_of": candidate["as_of"],
            "model_version": candidate["model_version"],
        }
        if target_market_date is not None:
            entries[symbol].update(
                observation_as_of=candidate["observation_as_of"],
                latest_regular_price_date=candidate[
                    "latest_regular_price_date"
                ],
                observation_kind=candidate["observation_kind"],
            )
            status = candidate["trading_status_evidence"]
            if status is not None:
                entries[symbol]["evidence_sha256"] = status[
                    "evidence_sha256"
                ]

    if target_market_date is None:
        manifest_schema = 2
        manifest = {
            "schema_version": manifest_schema,
            "market": market,
            "generated_at": generated_text,
            "universe_count": len(symbols),
            "symbol_count": len(entries),
            "failure_count": len(excluded),
            "failure_rate": failure_rate,
            "coverage": len(entries) / len(symbols),
            "failed_symbols": sorted(excluded),
            "unavailable_symbols": sorted(unavailable),
            "unavailable_count": len(unavailable),
            "market_as_of": market_as_of,
            "symbols": entries,
        }
    else:
        manifest_schema = 4
        expected_non_price = {}
        for symbol, candidate in candidates.items():
            status = candidate["trading_status_evidence"]
            if status is None:
                continue
            expected_non_price[symbol] = {
                "status": status["status"],
                "evidence_sha256": status["evidence_sha256"],
                "artifact_sha256": entries[symbol]["sha256"],
                "latest_regular_price_date": candidate[
                    "latest_regular_price_date"
                ],
            }
        regular_count = len(entries) - len(expected_non_price)
        denominator = len(entries) - len(expected_non_price)
        if denominator <= 0:
            raise RuntimeError("regular price denominator is invalid")
        operational_failed = sorted(failed - unavailable)
        manifest = {
            "schema_version": manifest_schema,
            "market": market,
            "generated_at": generated_text,
            "target_market_date": target_market_date.isoformat(),
            "observation_as_of": target_market_date.isoformat(),
            "active_universe_count": len(symbols),
            "observation_count": len(entries),
            "regular_price_symbol_count": regular_count,
            "verified_non_price_symbol_count": len(expected_non_price),
            "unavailable_count": len(unavailable),
            "unavailable_symbols": sorted(unavailable),
            "operational_failure_count": len(operational_failed),
            "operational_failed_symbols": operational_failed,
            "operational_failure_rate": len(operational_failed) / len(symbols),
            "observation_coverage": len(entries) / len(symbols),
            "regular_price_denominator": denominator,
            "regular_price_coverage": regular_count / denominator,
            "expected_non_price_symbols": expected_non_price,
            "symbols": entries,
        }
    latest_path = publish_root / f"latest-{market}.json"
    try:
        current_latest = json.loads(latest_path.read_text(encoding="utf-8"))
        current_relative = str(current_latest.get("manifest") or "")
        if (
            current_latest.get("schema_version") == manifest_schema
            and current_latest.get("market") == market
            and re.fullmatch(
                rf"manifests/{market}-[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{12}}\.json",
                current_relative,
            )
        ):
            current_path = publish_root / current_relative
            current_bytes = current_path.read_bytes()
            current_digest = hashlib.sha256(current_bytes).hexdigest()
            current_manifest = json.loads(current_bytes)
            if (
                secrets.compare_digest(
                    current_digest,
                    str(current_latest.get("manifest_sha256") or ""),
                )
                and current_latest.get("generated_at")
                == current_manifest.get("generated_at")
            ):
                current_identity = dict(current_manifest)
                candidate_identity = dict(manifest)
                current_identity.pop("generated_at", None)
                candidate_identity.pop("generated_at", None)
                if current_identity == candidate_identity:
                    return latest_path
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    run_id = generated_at.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_name = f"{market}-{run_id}-{manifest_digest[:12]}.json"
    manifest_path = publish_root / "manifests" / manifest_name
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("immutable manifest conflict")
    else:
        _write_bytes_atomic(manifest_path, manifest_bytes)

    _write_json_atomic(
        latest_path,
        {
            "schema_version": manifest_schema,
            "market": market,
            "generated_at": generated_text,
            "manifest": f"manifests/{manifest_name}",
            "manifest_sha256": manifest_digest,
        },
    )
    return latest_path


def publish_market_insights(root, document, generated_at=None):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid market insights document")
    as_of = datetime.date.fromisoformat(str(document.get("as_of")))
    generated_at = generated_at or datetime.datetime.now(TAIPEI)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=TAIPEI)
    if as_of > generated_at.astimezone(TAIPEI).date():
        raise ValueError("market insights date is in the future")
    _validate_json_value(document)
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
    if len(compressed) > STOCK_ARTIFACT_MAX_COMPRESSED_BYTES:
        raise RuntimeError("market insights snapshot is too large")
    digest = hashlib.sha256(compressed).hexdigest()
    publish_root = Path(root) / "publish" / "quant" / "v1"
    object_path = publish_root / "objects" / f"{digest}.json.gz"
    if object_path.exists():
        if object_path.stat().st_size != len(compressed) or _sha256_path(object_path) != digest:
            raise RuntimeError("published insights object hash mismatch")
        os.utime(object_path, None)
    else:
        _write_bytes_atomic(object_path, compressed)
    generated_text = generated_at.astimezone(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    latest_path = publish_root / "latest-insights.json"
    _write_json_atomic(latest_path, {
        "schema_version": 1,
        "kind": "market-insights",
        "generated_at": generated_text,
        "as_of": document["as_of"],
        "path": f"objects/{digest}.json.gz",
        "sha256": digest,
        "size": len(compressed),
    })
    return latest_path


def _fetch_json_list(url):
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": "ABSORB/1.0 (public market research)"},
        timeout=15,
    )
    response.raise_for_status()
    if int(response.headers.get("Content-Length") or 0) > MARKET_INSIGHTS_MAX_BYTES:
        raise RuntimeError("market insights response is too large")
    content = response.content
    if len(content) > MARKET_INSIGHTS_MAX_BYTES:
        raise RuntimeError("market insights response is too large")
    document = json.loads(content)
    if not isinstance(document, list):
        raise ValueError("market insights response must be a list")
    return document


def _fetch_yfinance_holdings(etf):
    import yfinance as yf

    frame = yf.Ticker(etf["ticker"]).funds_data.top_holdings
    if frame is None or getattr(frame, "empty", True):
        return []
    rows = []
    for symbol, row in frame.iterrows():
        rows.append({
            "symbol": str(symbol),
            "name": row.get("Name") or row.get("name") or str(symbol),
            "weight": row.get("Holding Percent") if "Holding Percent" in row else row.get("holdingPercent"),
        })
    return rows


def _read_insights_metric(root, symbol):
    symbol = str(symbol).upper()
    market = "TW" if re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", symbol.upper()) else "US"
    if market == "US" and not re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?", symbol):
        return None
    try:
        _path, _compressed, document = _validated_artifact(
            root, market, symbol, datetime.datetime.now(TAIPEI)
        )
        latest = document["latest"]
        probability = max(0, min(100, int(round(float(latest["AI_P"])))))
        close = float(latest["Close"])
        ma20 = float(latest["MA20"])
        def optional_number(key, scale=1.0):
            value = latest.get(key)
            if value is None:
                return None
            try:
                value = float(value) * scale
            except (TypeError, ValueError):
                return None
            return round(value, 4) if math.isfinite(value) else None
        return {
            "name": str(document.get("name") or symbol),
            "prob": probability,
            "trend": "多頭" if close > ma20 else "空頭",
            "as_of": str(document["as_of"]),
            "close": round(close, 4),
            "return_1d": optional_number("RET_1", 100),
            "inst_ratio": optional_number("INST_NET_RATIO"),
            "margin_change": optional_number("MARGIN_CHG"),
            "short_change": optional_number("SHORT_CHG"),
            "volume_ratio": optional_number("VOL_RATIO"),
        }
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None


def _load_local_market_insights(root):
    try:
        publish_root = Path(root) / "publish" / "quant" / "v1"
        latest = json.loads((publish_root / "latest-insights.json").read_text(encoding="utf-8"))
        path = str(latest["path"])
        if (
            latest.get("schema_version") != 1
            or latest.get("kind") != "market-insights"
            or re.fullmatch(r"objects/[0-9a-f]{64}\.json\.gz", path) is None
        ):
            return None
        compressed = (publish_root / path).read_bytes()
        if len(compressed) != latest["size"] or hashlib.sha256(compressed).hexdigest() != latest["sha256"]:
            return None
        decoded = gzip.decompress(compressed)
        if len(decoded) > STOCK_ARTIFACT_MAX_UNCOMPRESSED_BYTES:
            return None
        document = json.loads(decoded)
        return document if document.get("schema_version") == 1 else None
    except (KeyError, OSError, TypeError, ValueError):
        return None


def build_market_insights_document(root, pipeline, now=None, fetch_json=None, fetch_etf=None):
    checked_at = now or datetime.datetime.now(TAIPEI)
    previous = _load_local_market_insights(root) or {}
    fetch_json = fetch_json or _fetch_json_list
    fetch_etf = fetch_etf or _fetch_yfinance_holdings
    name_for_date = getattr(pipeline, "get_stock_name_for_date", None)

    def resolve_name(symbol):
        if callable(name_for_date):
            return name_for_date(
                symbol,
                checked_at.astimezone(TAIPEI).date(),
                require_authoritative=True,
            )
        return pipeline.get_stock_name(symbol)

    mops = []
    successful_mops_sources = 0
    for source, url in MOPS_SOURCES:
        try:
            mops.extend(parse_mops_items(fetch_json(url), source))
            successful_mops_sources += 1
        except Exception:
            continue
    if not successful_mops_sources:
        mops = list(previous.get("mops") or [])
    mops.sort(key=lambda row: row.get("published_at", ""), reverse=True)

    previous_etfs = {item.get("ticker"): item for item in previous.get("etfs") or []}
    etfs = []
    for etf in ETF_CATALOG:
        try:
            normalized = normalize_etf_holdings(fetch_etf(etf), etf)
        except Exception:
            normalized = previous_etfs.get(etf["ticker"])
        if normalized and normalized.get("holdings"):
            etfs.append(normalized)

    symbols = {
        str(symbol).upper()
        for category, codes in pipeline.industry_map.items()
        if category not in {"全市場", "ETF專區"}
        for symbol in codes
    }
    symbols.update(
        symbol
        for chain in SUPPLY_CHAINS
        for _stage, nodes in chain["stages"]
        for symbol, _name, _market in nodes
    )
    metrics = {
        str(symbol).upper(): {
            "name": resolve_name(symbol),
            "prob": None,
            "trend": "資料待更新",
            "as_of": "",
        }
        for category, codes in pipeline.industry_map.items()
        if category not in {"全市場", "ETF專區"}
        for symbol in codes
    }
    for symbol in symbols:
        metric = _read_insights_metric(root, symbol)
        if metric:
            if callable(name_for_date):
                metric["name"] = resolve_name(symbol)
            metrics[symbol] = metric

    return {
        "schema_version": 1,
        "as_of": checked_at.astimezone(TAIPEI).date().isoformat(),
        "industries": build_industries(pipeline.industry_map, metrics),
        "mops": mops[:200],
        "etfs": etfs,
        "supply_chains": build_supply_chains(metrics),
        "sources": [source for source, _url in MOPS_SOURCES] + ["yfinance", "ABSORB local artifacts"],
    }


def load_exclusion_list(root, market):
    csv_path = Path(root) / "checkpoints" / f"exclusion_list-{market}.csv"
    pending = set()
    excluded = set()
    rows = []
    invalid_actions_count = 0

    if not csv_path.is_file():
        return pending, excluded, rows, invalid_actions_count

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get("Symbol", "").strip()
                if not symbol:
                    continue
                action = row.get("OperatorAction", "").strip()
                state = row.get("State", "").strip()

                norm_action = action.strip().capitalize() if action else ""
                if norm_action not in ("", "Approve", "Reinstate"):
                    safe_print(f"Warning: Invalid action '{action}' for symbol '{symbol}' ignored.")
                    invalid_actions_count += 1
                    norm_action = ""

                if norm_action == "Reinstate":
                    continue
                elif norm_action == "Approve":
                    state = "Excluded"
                    row["State"] = state
                    row["OperatorAction"] = ""

                if state == "Excluded":
                    excluded.add(symbol)
                else:
                    pending.add(symbol)

                rows.append(row)
    except Exception as e:
        safe_print(f"Warning: Failed to read exclusion list: {e}")

    return pending, excluded, rows, invalid_actions_count

def save_exclusion_list(root, market, rows):
    csv_dir = Path(root) / "checkpoints"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"exclusion_list-{market}.csv"
    tmp_path = csv_dir / f"exclusion_list-{market}.csv.tmp"

    fields = ["Symbol", "Name", "ExclusionDate", "ConsecutiveFailures", "State", "Type", "Reason", "OperatorAction"]

    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                filtered_row = {k: row.get(k, "") for k in fields}
                writer.writerow(filtered_row)
        os.replace(tmp_path, csv_path)
        return "SUCCESS"
    except (PermissionError, OSError) as e:
        safe_print(f"Warning: Exclusion list write failed (file may be locked): {type(e).__name__}")
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return "FAILED (file locked)"
    except Exception as e:
        safe_print(f"Warning: Exclusion list write failed due to unknown error.")
        return "FAILED"

def load_consecutive_failures(root, market):
    path = Path(root) / "checkpoints" / f"consecutive_failures-{market}.json"
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_consecutive_failures(root, market, data):
    path = Path(root) / "checkpoints" / f"consecutive_failures-{market}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        pass

def _assert_no_pending_exclusion_operator_actions(root, market):
    csv_path = Path(root) / "checkpoints" / f"exclusion_list-{market}.csv"
    if not csv_path.is_file():
        return

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "OperatorAction" not in reader.fieldnames:
                raise RuntimeError("observation-only exclusion preflight failed")
            for row in reader:
                if str(row.get("OperatorAction") or "").strip():
                    raise RuntimeError(
                        "observation-only refuses pending exclusion operator actions"
                    )
    except RuntimeError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError("observation-only exclusion preflight failed") from exc


def run_market_batch(
    root,
    market,
    symbols,
    analyze_symbol,
    limit=200,
    now_fn=lambda: datetime.datetime.now(TAIPEI),
    delay=0.5,
    sleep_fn=time.sleep,
    enforce_window=True,
    batch_identity=None,
    mutate_exclusions=True,
):
    if (
        market not in ("TW", "US")
        or limit < 1
        or delay < 0
        or type(enforce_window) is not bool
        or (batch_identity is not None and not isinstance(batch_identity, dict))
        or type(mutate_exclusions) is not bool
    ):
        raise ValueError("invalid market batch settings")
    if not mutate_exclusions:
        _assert_no_pending_exclusion_operator_actions(root, market)

    checkpoint = load_checkpoint(root, market=market)
    checked_at = now_fn()
    same_batch = (
        checkpoint.get("stage") == "market_batch"
        and checkpoint.get("market") == market
        and (
            batch_identity is None
            or checkpoint.get("batch_identity") == batch_identity
        )
    )
    start = (
        checkpoint.get("next_index", 0)
        if same_batch
        else 0
    )
    symbol_set = {str(symbol) for symbol in symbols}
    failures = []
    seen_failures = set()
    if same_batch:
        for item in checkpoint.get("failed", []):
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", ""))
            if symbol not in symbol_set or symbol in seen_failures:
                continue
            failures.append({"symbol": symbol, "error": str(item.get("error") or "Error")})
            seen_failures.add(symbol)

    is_new_day = False
    if checkpoint.get("updated_at"):
        try:
            updated_date_str = checkpoint["updated_at"].split("T", 1)[0]
            updated_date = datetime.date.fromisoformat(updated_date_str)
            if checked_at.date() > updated_date:
                is_new_day = True
        except (ValueError, IndexError, TypeError):
            pass

    pending_symbols, excluded_symbols, exclusion_rows, invalid_actions = load_exclusion_list(root, market)
    csv_write_status = "SKIPPED" if not mutate_exclusions else "SUCCESS"
    if mutate_exclusions and exclusion_rows:
        csv_write_status = save_exclusion_list(root, market, exclusion_rows)
        pending_symbols, excluded_symbols, exclusion_rows, invalid_actions = load_exclusion_list(root, market)


    is_published = (
        checkpoint.get("published_cycle_on") == checkpoint.get("cycle_completed_on")
        or bool(checkpoint.get("published_at"))
    )

    if (
        start >= len(symbols)
        and (not failures or is_new_day)
        and checkpoint.get("cycle_completed_on") != checked_at.date().isoformat()
        and is_published
    ):
        start = 0
        failures = []
    next_index = start
    attempted = completed = 0
    consecutive_failures = load_consecutive_failures(root, market)

    def save_state():
        state = {
            "stage": "market_batch",
            "market": market,
            "next_index": next_index,
            "failed": failures,
            "updated_at": checked_at.isoformat(),
        }
        if batch_identity is not None:
            state["batch_identity"] = batch_identity
        if next_index >= len(symbols) and not failures:
            state["cycle_completed_on"] = checked_at.date().isoformat()
        save_checkpoint(root, state, market=market)

    def fail_provider(symbol, error, *, checkpoint_next_index):
        failed_items = []
        failed_symbols = set()
        for item in failures:
            failed_symbol = str(item.get("symbol", ""))
            if (
                not failed_symbol
                or failed_symbol == str(symbol)
                or failed_symbol in failed_symbols
            ):
                continue
            failed_items.append(item)
            failed_symbols.add(failed_symbol)
        failed_items.append(
            {
                "symbol": str(symbol),
                "error": type(error).__name__,
                "category": error.category,
            }
        )
        state = {
            "stage": "market_batch",
            "market": market,
            "next_index": checkpoint_next_index,
            "failed": failed_items,
            "updated_at": checked_at.isoformat(),
            "provider": "FinMind",
            "dataset": error.dataset,
            "category": error.category,
            "http_status": error.http_status,
            "blocked_until": error.blocked_until,
            "first_failed_symbol": symbol,
            "attempted_count": attempted,
            "successful_count": completed,
            "failed_count": len(failed_items),
            "provider_failure_count": 1,
            "timestamp": checked_at.isoformat(),
            "safe_message": error.safe_message,
        }
        if batch_identity is not None:
            state["batch_identity"] = batch_identity
        save_checkpoint(root, state, market=market)
        raise error

    # Retry failures
    for retry in list(failures):
        if attempted:
            checked_at = now_fn()
        if enforce_window and window_phase(checked_at) != "run":
            break
        symbol = retry["symbol"]
        if symbol in pending_symbols or symbol in excluded_symbols:
            failures = [item for item in failures if item["symbol"] != symbol]
            continue
        attempted += 1
        failures = [item for item in failures if item["symbol"] != symbol]
        try:
            payload = analyze_symbol(symbol)
        except FinMindFetchError as exc:
            fail_provider(
                symbol,
                exc,
                checkpoint_next_index=next_index,
            )
        except Exception as exc:
            exc_str = str(exc)
            import yfinance as yf
            yf_shared = getattr(yf, "shared", None)
            yf_err = getattr(yf_shared, "_ERRORS", {}).get(symbol) if yf_shared else None
            if yf_err:
                exc_str += f" | yfinance: {yf_err}"

            exc_str_lower = exc_str.lower()
            if "delisted" in exc_str_lower or "not found" in exc_str_lower or "no data found" in exc_str_lower:
                err_type = "delisted"
            elif "429" in exc_str_lower or "rate limit" in exc_str_lower or "too many requests" in exc_str_lower:
                err_type = "rate_limited"
            elif "timeout" in exc_str_lower or "timed out" in exc_str_lower:
                err_type = "timeout"
            else:
                err_type = "unknown"

            failures.append({"symbol": symbol, "error": type(exc).__name__})

            if err_type == "delisted":
                consecutive_failures[symbol] = consecutive_failures.get(symbol, 0) + 1
                if mutate_exclusions and consecutive_failures[symbol] >= 5:
                    if symbol not in pending_symbols and symbol not in excluded_symbols:
                        new_row = {
                            "Symbol": symbol,
                            "Name": symbol,
                            "ExclusionDate": checked_at.date().isoformat(),
                            "ConsecutiveFailures": consecutive_failures[symbol],
                            "State": "Pending",
                            "Type": "delisted",
                            "Reason": exc_str,
                            "OperatorAction": ""
                        }
                        exclusion_rows.append(new_row)
                        save_exclusion_list(root, market, exclusion_rows)
                        pending_symbols.add(symbol)
        else:
            write_stock_artifact(root, market, symbol, payload)
            completed += 1
            consecutive_failures[symbol] = 0
        save_state()
        if delay:
            sleep_fn(delay)

    # Main loop
    while attempted < limit and next_index < len(symbols):
        if attempted:
            checked_at = now_fn()
        if enforce_window and window_phase(checked_at) != "run":
            break
        symbol = str(symbols[next_index])
        if symbol in pending_symbols or symbol in excluded_symbols:
            next_index += 1
            continue
        attempted += 1
        try:
            payload = analyze_symbol(symbol)
        except FinMindFetchError as exc:
            fail_provider(
                symbol,
                exc,
                checkpoint_next_index=next_index + 1,
            )
        except Exception as exc:
            exc_str = str(exc)
            import yfinance as yf
            yf_shared = getattr(yf, "shared", None)
            yf_err = getattr(yf_shared, "_ERRORS", {}).get(symbol) if yf_shared else None
            if yf_err:
                exc_str += f" | yfinance: {yf_err}"

            exc_str_lower = exc_str.lower()
            if "delisted" in exc_str_lower or "not found" in exc_str_lower or "no data found" in exc_str_lower:
                err_type = "delisted"
            elif "429" in exc_str_lower or "rate limit" in exc_str_lower or "too many requests" in exc_str_lower:
                err_type = "rate_limited"
            elif "timeout" in exc_str_lower or "timed out" in exc_str_lower:
                err_type = "timeout"
            else:
                err_type = "unknown"

            failures = [item for item in failures if item["symbol"] != symbol]
            failures.append({"symbol": symbol, "error": type(exc).__name__})

            if err_type == "delisted":
                consecutive_failures[symbol] = consecutive_failures.get(symbol, 0) + 1
                if mutate_exclusions and consecutive_failures[symbol] >= 5:
                    if symbol not in pending_symbols and symbol not in excluded_symbols:
                        new_row = {
                            "Symbol": symbol,
                            "Name": symbol,
                            "ExclusionDate": checked_at.date().isoformat(),
                            "ConsecutiveFailures": consecutive_failures[symbol],
                            "State": "Pending",
                            "Type": "delisted",
                            "Reason": exc_str,
                            "OperatorAction": ""
                        }
                        exclusion_rows.append(new_row)
                        save_exclusion_list(root, market, exclusion_rows)
                        pending_symbols.add(symbol)
        else:
            write_stock_artifact(root, market, symbol, payload)
            completed += 1
            consecutive_failures[symbol] = 0
        next_index += 1
        save_state()
        if delay:
            sleep_fn(delay)


    save_consecutive_failures(root, market, consecutive_failures)
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failures,
        "next_index": next_index,
        "pending": list(pending_symbols),
        "excluded": list(excluded_symbols),
        "invalid_actions": invalid_actions,
        "csv_write_status": csv_write_status,
    }


def load_stock_pipeline(root):
    cache = Path(root) / "cache" / "yfinance"
    cache.mkdir(parents=True, exist_ok=True)
    import yfinance as yf

    yf.set_tz_cache_location(str(cache))
    os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "local-only")
    os.environ.setdefault("LINE_CHANNEL_SECRET", "local-only")
    removed = {
        key: os.environ.pop(key)
        for key in ("GEMINI_API_KEY", "GCP_PROJECT_ID")
        if key in os.environ
    }
    try:
        return importlib.import_module("app")
    finally:
        os.environ.update(removed)


def get_taiwan_symbols(pipeline):
    resolver = getattr(pipeline, "taiwan_security_master", None)
    if callable(getattr(resolver, "get_master", None)):
        master = resolver.get_master(required=True)
        return sorted(
            symbol
            for symbol in master.entries
            if re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", symbol)
        )
    return sorted(
        {
            str(symbol)
            for symbol in pipeline.industry_map.get("全市場", [])
            if re.fullmatch(r"[0-9]{4,5}[0-9A-Z]?", str(symbol).upper())
        }
    )


def parse_sec_us_universe(document):
    if not isinstance(document, dict):
        raise ValueError("invalid SEC universe document")
    fields = document.get("fields")
    rows = document.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("invalid SEC universe schema")
    required = {"name", "ticker", "exchange"}
    if not required.issubset(fields):
        raise ValueError("SEC universe fields are incomplete")
    positions = {name: fields.index(name) for name in required}
    symbols = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        exchange = row[positions["exchange"]]
        name = str(row[positions["name"]] or "").lower()
        symbol = str(row[positions["ticker"]] or "").strip().upper()
        if exchange not in US_EXCHANGES or any(term in name for term in CRYPTO_SECURITY_TERMS):
            continue
        try:
            symbols.add(validate_market_symbol("US", symbol))
        except ValueError:
            continue
    if not symbols:
        raise ValueError("SEC universe contains no supported US symbols")
    return sorted(symbols)


def fetch_sec_us_universe_json():
    import requests

    response = requests.get(
        SEC_US_UNIVERSE_URL,
        headers={
            "User-Agent": "ABSORB/1.0"
        },
        timeout=15,
    )
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > SEC_US_UNIVERSE_MAX_BYTES:
        raise RuntimeError("SEC universe response is too large")
    content = response.content
    if len(content) > SEC_US_UNIVERSE_MAX_BYTES:
        raise RuntimeError("SEC universe response is too large")
    return json.loads(content)


def parse_nasdaq_us_universe(listed_text, other_text):
    symbols = set()
    for text, fields in (
        (listed_text, ("Symbol",)),
        (other_text, ("NASDAQ Symbol", "ACT Symbol")),
    ):
        lines = str(text or "").splitlines()
        if not lines:
            raise ValueError("Nasdaq universe document is empty")
        header = [item.strip() for item in lines[0].split("|")]
        if "Security Name" not in header or "Test Issue" not in header:
            raise ValueError("Nasdaq universe fields are incomplete")
        symbol_field = next((field for field in fields if field in header), None)
        if symbol_field is None:
            raise ValueError("Nasdaq universe symbol field is missing")
        positions = {name: header.index(name) for name in (symbol_field, "Security Name", "Test Issue")}
        for line in lines[1:]:
            values = [item.strip() for item in line.split("|")]
            if len(values) != len(header) or values[positions["Test Issue"]] != "N":
                continue
            name = values[positions["Security Name"]].lower()
            if any(term in name for term in CRYPTO_SECURITY_TERMS):
                continue
            symbol = values[positions[symbol_field]].upper().replace(".", "-")
            try:
                symbols.add(validate_market_symbol("US", symbol))
            except ValueError:
                continue
    if not symbols:
        raise ValueError("Nasdaq universe is empty")
    return sorted(symbols)


def fetch_nasdaq_us_universe_texts():
    import requests

    documents = []
    for url in NASDAQ_US_UNIVERSE_URLS:
        response = requests.get(
            url,
            headers={"User-Agent": "ABSORB/1.0"},
            timeout=15,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError("Nasdaq universe request failed")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > SEC_US_UNIVERSE_MAX_BYTES:
            raise RuntimeError("Nasdaq universe response is too large")
        content = response.content
        if len(content) > SEC_US_UNIVERSE_MAX_BYTES:
            raise RuntimeError("Nasdaq universe response is too large")
        documents.append(content.decode("utf-8-sig"))
    return tuple(documents)


def _read_us_universe_cache(path):
    try:
        cached = json.loads(Path(path).read_text(encoding="utf-8"))
        symbols = [validate_market_symbol("US", item) for item in cached["symbols"]]
        if not symbols or not isinstance(cached.get("as_of"), str):
            return None
        return {"as_of": cached["as_of"], "symbols": sorted(set(symbols))}
    except (KeyError, OSError, TypeError, ValueError):
        return None


def get_us_symbols(root, fetch_json=None, now=None, fetch_nasdaq=None):
    checked_at = now or datetime.datetime.now(TAIPEI)
    cache_path = Path(root) / "raw" / "us-universe.json"
    cached = _read_us_universe_cache(cache_path) if cache_path.exists() else None
    if cached and cached["as_of"] == checked_at.date().isoformat():
        return cached["symbols"]
    source = SEC_US_UNIVERSE_URL
    try:
        symbols = parse_sec_us_universe((fetch_json or fetch_sec_us_universe_json)())
    except Exception:
        try:
            symbols = parse_nasdaq_us_universe(
                *(fetch_nasdaq or fetch_nasdaq_us_universe_texts)()
            )
            source = "|".join(NASDAQ_US_UNIVERSE_URLS)
        except Exception as nasdaq_exc:
            if cached:
                return cached["symbols"]
            raise RuntimeError("US universe is unavailable") from nasdaq_exc
    _write_json_atomic(
        cache_path,
        {
            "as_of": checked_at.date().isoformat(),
            "source": source,
            "symbols": symbols,
        },
    )
    return symbols


def _market_date(value):
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError("historical market date is invalid") from exc


def _canonical_history_frame(frame):
    result = frame.copy()
    dated_positions = [
        (_market_date(value), position)
        for position, value in enumerate(result.index)
    ]
    dates = [value for value, _ in dated_positions]
    if len(dates) != len(set(dates)):
        raise ValueError("historical market dates are duplicated")
    dated_positions.sort()
    result = result.iloc[[position for _, position in dated_positions]].copy()
    result.index = [
        datetime.datetime.combine(value, datetime.time.min)
        for value, _ in dated_positions
    ]
    result.index.name = frame.index.name or "Date"
    return result


def _persisted_history_frame(canonical_frame, calculated_frame):
    if calculated_frame.index.has_duplicates:
        raise ValueError("calculated market dates are duplicated")
    if not calculated_frame.index.isin(canonical_frame.index).all():
        raise ValueError("calculated market date is not canonical")
    derived = CALCULATED_COLUMNS + OBSERVATION_MODEL_COLUMNS
    source_columns = [
        name for name in canonical_frame.columns if name not in derived
    ]
    result = canonical_frame.loc[:, source_columns].copy()
    join_columns = [
        name for name in derived if name in calculated_frame.columns
    ]
    for name in join_columns:
        result[name] = math.nan
        if calculated_frame[name].dtype == object:
            result[name] = result[name].astype(object)
        result.loc[calculated_frame.index, name] = calculated_frame[name]
    return result


def build_stock_snapshot(
    pipeline,
    market,
    symbol,
    target_market_date=None,
    promoted_backtest=None,
    degraded_bootstrap=False,
    observation_only=False,
    trading_status=None,
):
    if target_market_date is not None and type(target_market_date) is not datetime.date:
        raise TypeError("target_market_date must be a date")
    symbol = validate_market_symbol(market, symbol)
    if trading_status is not None:
        from stock_papi.integrations.market_data.tw_trading_status import (
            evidence_sha256,
        )

        trading_status = dict(trading_status)
        if (
            market != "TW"
            or not observation_only
            or target_market_date is None
            or trading_status.get("status") not in {
                "official_no_regular_trade", "officially_suspended"
            }
            or trading_status.get("market") != "TW"
            or trading_status.get("symbol") != symbol
            or trading_status.get("target_market_date")
            != target_market_date.isoformat()
            or trading_status.get("evidence_sha256")
            != evidence_sha256(trading_status)
        ):
            raise ValueError("trading status evidence is invalid")
    frame = pipeline.get_data(symbol, 730)
    if frame is None or frame.empty:
        raise ValueError("price history is unavailable")
    if target_market_date is not None:
        frame = frame[frame.index.date <= target_market_date]
        if frame.empty:
            raise ValueError("point-in-time price history is unavailable")
    canonical_frame = _canonical_history_frame(frame)
    calculated_frame = pipeline.calc_all(canonical_frame.copy())
    if calculated_frame is None or calculated_frame.empty:
        calculated_frame = canonical_frame.iloc[0:0].copy()
    elif canonical_frame.index[-1] not in calculated_frame.index:
        raise ValueError("calculated history is unavailable")
    compatibility = None
    if observation_only:
        calculated_frame = calculated_frame.drop(
            columns=list(OBSERVATION_MODEL_COLUMNS),
            errors="ignore",
        )
        backtest = {}
        model_version = OBSERVATION_SOURCE_VERSION
    else:
        from stock_papi.quant.model import FEATURE_SCHEMA_VERSION
        from stock_papi.services.recommendation_engine import (
            RECOMMENDATION_POLICY_VERSION,
        )

        if promoted_backtest is None and not degraded_bootstrap:
            backtest = pipeline.run_ai_engine(calculated_frame)
            if not isinstance(backtest, dict):
                raise ValueError("backtest is unavailable")
            model_version = (
                f"lgbm-{int(getattr(pipeline, 'PREDICTION_HORIZON', 5))}d-v1"
            )
        else:
            if promoted_backtest is not None and not isinstance(
                promoted_backtest, dict
            ):
                raise TypeError("promoted_backtest must be a dictionary")
            infer = getattr(pipeline, "run_latest_inference", None)
            if not callable(infer):
                raise ValueError("latest inference is unavailable")
            inference = infer(calculated_frame)
            if not isinstance(inference, dict) or not isinstance(
                inference.get("model_version"), str
            ):
                raise ValueError("latest inference is unavailable")
            model_version = inference["model_version"]
            if promoted_backtest is None:
                compatibility = {
                    "compatible": False,
                    "confidence_cap": "low",
                    "strong_action_allowed": False,
                    "reason": "initial_backtest_bootstrap",
                    "mismatch_fields": ["validated_backtest_baseline"],
                }
                backtest = {}
            else:
                from stock_papi.batch.backtest_store import (
                    assess_backtest_compatibility,
                )

                compatibility = assess_backtest_compatibility(
                    promoted_backtest,
                    expected_model_version=model_version,
                    expected_feature_schema_version=FEATURE_SCHEMA_VERSION,
                    expected_recommendation_policy_version=(
                        RECOMMENDATION_POLICY_VERSION
                    ),
                )
                if not compatibility["compatible"]:
                    raise ValueError(
                        "backtest baseline is incompatible: "
                        f"{compatibility['reason']}"
                    )
                backtest = promoted_backtest

    persisted_frame = _persisted_history_frame(
        canonical_frame, calculated_frame
    )
    daily = json.loads(
        persisted_frame.reset_index().to_json(
            orient="records",
            date_format="iso",
            date_unit="ms",
        )
    )
    latest = daily[-1]
    as_of = str(latest.get("Date", "")).split("T", 1)[0]
    if not as_of:
        raise ValueError("latest market date is unavailable")
    if target_market_date is not None:
        if trading_status is None and as_of != target_market_date.isoformat():
            raise ValueError("target market date mismatch")
        if trading_status is not None and as_of >= target_market_date.isoformat():
            raise ValueError("trading status conflicts with regular price date")
    name_resolver = getattr(pipeline, "get_stock_name_for_date", None)
    if callable(name_resolver):
        resolved_name = name_resolver(
            symbol,
            target_market_date,
            require_authoritative=(market == "TW" and target_market_date is not None),
        )
    else:
        resolved_name = pipeline.get_stock_name(symbol)
    security_master_evidence = None
    security_master = getattr(pipeline, "taiwan_security_master", None)
    if market == "TW" and target_market_date is not None and callable(
        getattr(security_master, "get_master", None)
    ):
        master = security_master.get_master(required=True)
        security_master_evidence = {
            "schema_version": master.schema_version,
            "as_of": master.as_of.isoformat(),
            "source_hashes": dict(master.source_hashes),
        }
    result = {
        "as_of": as_of,
        "name": resolved_name,
        "rows": len(daily),
        "model_version": model_version,
        "latest": latest,
        "backtest": backtest,
        "daily": daily,
    }
    if market in ("TW", "US") and target_market_date is not None:
        observation_kind = (
            trading_status["status"]
            if trading_status is not None
            else "regular_price"
        )
        lineage_schema = (
            "tw-official-historical-v3"
            if market == "TW"
            else "us-market-data-v1"
        )
        lineage = {
            "source_schema_version": lineage_schema,
            "observation_as_of": target_market_date.isoformat(),
            "latest_regular_price_date": as_of,
            "observation_kind": observation_kind,
        }
        if security_master_evidence is not None:
            lineage["security_master"] = security_master_evidence
        if trading_status is not None:
            lineage["trading_status_evidence_sha256"] = (
                trading_status.get("evidence_sha256")
            )
        result.update(
            schema_version=2,
            target_market_date=target_market_date.isoformat(),
            observation_as_of=target_market_date.isoformat(),
            latest_regular_price_date=as_of,
            observation_kind=observation_kind,
            lineage=lineage,
        )
        if trading_status is not None:
            result["trading_status_evidence"] = trading_status
    if not observation_only:
        result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        result["recommendation_policy_version"] = (
            RECOMMENDATION_POLICY_VERSION
        )
    if compatibility is not None:
        result["backtest_compatibility"] = compatibility
        result["backtest_as_of"] = compatibility.get("backtest_as_of")
        result["backtest_version"] = compatibility.get("backtest_version")
    return result


def load_checkpoint(root, market="TW"):
    checkpoint = _checkpoint_path(root, market)
    if not checkpoint.exists():
        return {}
    try:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("checkpoint is invalid") from exc
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint must contain an object")
    return state


def main(argv=None, now=None, free_bytes=None):
    parser = argparse.ArgumentParser(description="ABSORB local quant runner")
    parser.add_argument("--root", default=r"D:\AbsorbData")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--post-close", action="store_true")
    parser.add_argument("--insights", action="store_true")
    parser.add_argument("--market", choices=("TW", "US", "ALL"), default="TW")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--target-market-date", type=datetime.date.fromisoformat)
    parser.add_argument("--allow-degraded-bootstrap", action="store_true")
    parser.add_argument("--observation-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = validate_data_root(Path(args.root))
        if args.init:
            ensure_layout(root)
        elif not root.is_dir():
            raise RuntimeError("data root is not initialized")
        available = check_free_space(root, args.min_free_gb, free_bytes)
        checked_at = now or datetime.datetime.now(TAIPEI)
        phase = window_phase(checked_at)
        if args.post_close:
            if args.market != "TW" or args.target_market_date is None:
                raise ValueError("--post-close requires TW and --target-market-date")
            if args.observation_only and args.allow_degraded_bootstrap:
                raise ValueError(
                    "--observation-only cannot use degraded bootstrap"
                )
            phase = "run"
        elif args.observation_only:
            raise ValueError("--observation-only requires --post-close")
        if args.run and args.market == "US" and not market_run_allowed(
            "US", checked_at
        ):
            phase = "closed"
        status = {
            "checked_at": checked_at.isoformat(),
            "dry_run": bool(args.dry_run),
            "run": bool(args.run),
            "post_close": bool(args.post_close),
            "insights": bool(args.insights),
            "free_gb": round(available / 1024**3, 1),
            "phase": phase,
            "root": str(root),
        }
        status_path = root / "logs" / "runner-status.json"
        _write_json_atomic(status_path, status)
        if sum(
            (
                bool(args.dry_run),
                bool(args.run),
                bool(args.post_close),
                bool(args.insights),
            )
        ) != 1:
            raise ValueError("choose one of --dry-run, --run, --post-close or --insights")
        if args.insights and phase == "run":
            with acquire_lock(root, now=checked_at):
                status["cleanup"] = cleanup_expired_data(root, now=checked_at)
                pipeline = load_stock_pipeline(root)
                document = build_market_insights_document(root, pipeline, now=checked_at)
                publish_market_insights(root, document, generated_at=checked_at)
                status["market_insights"] = {
                    "as_of": document["as_of"],
                    "mops": len(document["mops"]),
                    "etfs": len(document["etfs"]),
                }
                _write_json_atomic(status_path, status)
                safe_print(json.dumps(status["market_insights"], ensure_ascii=False, separators=(",", ":")))
        if (args.run or args.post_close) and phase == "run":
            promoted_backtest = None
            daily_lock = contextlib.nullcontext()
            if args.post_close:
                from stock_papi.batch.runtime import acquire_job_lock
                if args.observation_only:
                    status["product_mode"] = "observation"
                    batch_identity = {
                        "target_market_date": (
                            args.target_market_date.isoformat()
                        ),
                        "product_mode": "observation",
                        "source_version": OBSERVATION_SOURCE_VERSION,
                    }
                    daily_lock = acquire_job_lock(
                        root,
                        "daily_observation",
                        args.target_market_date,
                        now=checked_at,
                    )
                else:
                    from stock_papi.batch.backtest_store import (
                        BacktestStore,
                        assess_backtest_compatibility,
                    )
                    from stock_papi.quant.model import (
                        FEATURE_SCHEMA_VERSION,
                        MODEL_VERSION,
                    )
                    from stock_papi.services.recommendation_engine import (
                        RECOMMENDATION_POLICY_VERSION,
                    )

                    promoted_backtest = BacktestStore(root, "TW").load_latest()
                    compatibility = (
                        assess_backtest_compatibility(
                            promoted_backtest,
                            expected_model_version=MODEL_VERSION,
                            expected_feature_schema_version=(
                                FEATURE_SCHEMA_VERSION
                            ),
                            expected_recommendation_policy_version=(
                                RECOMMENDATION_POLICY_VERSION
                            ),
                        )
                        if promoted_backtest is not None
                        else None
                    )
                    if compatibility is None or not compatibility["compatible"]:
                        if not args.allow_degraded_bootstrap:
                            reason = (
                                compatibility["reason"]
                                if compatibility is not None
                                else "validated_backtest_baseline_unavailable"
                            )
                            raise ValueError(
                                f"daily baseline is unavailable: {reason}"
                            )
                        promoted_backtest = None
                        status["baseline_status"] = (
                            "initial_backtest_bootstrap"
                        )
                    else:
                        status["baseline_status"] = "validated_compatible"
                    batch_identity = {
                        "target_market_date": (
                            args.target_market_date.isoformat()
                        ),
                        "model_version": MODEL_VERSION,
                        "backtest_candidate_sha256": (
                            promoted_backtest["candidate_sha256"]
                            if promoted_backtest is not None
                            else "initial_backtest_bootstrap"
                        ),
                    }
                    daily_lock = acquire_job_lock(
                        root,
                        "daily_prediction",
                        args.target_market_date,
                        now=checked_at,
                    )
            with daily_lock, acquire_lock(root, now=checked_at):
                if args.observation_only:
                    status["cleanup"] = {
                        "deleted_files": 0,
                        "reclaimed_bytes": 0,
                        "failed": 0,
                        "skipped_reparse_points": 0,
                        "skipped": True,
                        "reason": "observation_only",
                    }
                else:
                    status["cleanup"] = cleanup_expired_data(root, now=checked_at)
                _write_json_atomic(status_path, status)
                pipeline = load_stock_pipeline(root)
                now_fn = (
                    (lambda: checked_at)
                    if now is not None
                    else (lambda: datetime.datetime.now(TAIPEI))
                )
                summaries = {}
                if not args.post_close:
                    batch_identity = None
                markets = ("TW", "US") if args.market == "ALL" else (args.market,)
                for market in markets:
                    market_now = now_fn()
                    if not args.post_close and not market_run_allowed(market, market_now):
                        break
                    symbols = (
                        get_taiwan_symbols(pipeline)
                        if market == "TW"
                        else get_us_symbols(root, now=market_now)
                    )
                    summary = run_market_batch(
                        root,
                        market,
                        symbols,
                        lambda symbol, selected=market: build_stock_snapshot(
                            pipeline,
                            selected,
                            symbol,
                            target_market_date=args.target_market_date,
                            promoted_backtest=promoted_backtest,
                            degraded_bootstrap=(
                                args.post_close and promoted_backtest is None
                                and not args.observation_only
                            ),
                            observation_only=args.observation_only,
                        ),
                        limit=args.limit,
                        now_fn=now_fn,
                        delay=args.delay,
                        enforce_window=not args.post_close,
                        batch_identity=batch_identity,
                        mutate_exclusions=not args.observation_only,
                    )
                    if summary.get("next_index", 0) >= len(symbols):
                        failed_symbols = (
                            [item["symbol"] for item in summary.get("failed", [])]
                            + summary.get("pending", [])
                            + summary.get("excluded", [])
                        )
                        if args.observation_only:
                            summary["published"] = False
                            summary["publish_skipped_reason"] = "observation_only"
                        else:
                            try:
                                publish_market_snapshot(
                                    root,
                                    market,
                                    symbols,
                                    generated_at=market_now,
                                    failed_symbols=failed_symbols,
                                )
                            except RuntimeError as exc:
                                summary["published"] = False
                                summary["publish_error"] = str(exc)
                            else:
                                checkpoint = load_checkpoint(root, market=market)
                                checkpoint["published_cycle_on"] = (
                                    checkpoint.get("cycle_completed_on")
                                    or market_now.date().isoformat()
                                )
                                checkpoint["published_at"] = market_now.isoformat()
                                checkpoint["published_failure_count"] = len(failed_symbols)
                                save_checkpoint(root, checkpoint, market=market)
                                summary["published"] = True


                    published_count = len(symbols) - len(failed_symbols) if summary.get("published", False) else 0
                    stale_count = 0
                    if summary.get("published", False):
                        latest_path = Path(root) / "publish" / "quant" / "v1" / f"latest-{market}.json"
                        if latest_path.is_file():
                            try:
                                with open(latest_path, "r", encoding="utf-8") as f:
                                    lat = json.load(f)
                                with open(Path(root) / "publish" / "quant" / "v1" / lat["manifest"], "r", encoding="utf-8") as f:
                                    man = json.load(f)
                                published_count = man.get("symbol_count", 0)
                                stale_count = max(0, man.get("failure_count", 0) - len(failed_symbols))
                            except Exception:
                                pass

                    delisted_candidates = 0
                    invalid_candidates = 0
                    for item in summary.get("failed", []):
                        err_str = str(item.get("error", "")).lower()
                        if "delisted" in err_str:
                            delisted_candidates += 1
                        elif "not found" in err_str or "404" in err_str:
                            invalid_candidates += 1

                    summary_report = f"""
==================================================
📊 {market} MARKET QUANT SNAPSHOT SUMMARY
==================================================
* Execution Time: {market_now.strftime('%Y-%m-%d %H:%M:%S')}
* Total Symbols Scanned: {len(symbols)}
* Published Symbols Count: {published_count}
* Stale Filtered Count: {stale_count}
* Pending Exclusion Count: {len(summary.get("pending", []))}
* Permanently Excluded Count: {len(summary.get("excluded", []))}
* Temporary Failure Count: {len(summary.get("failed", [])) - delisted_candidates - invalid_candidates}
* Delisted Candidates Count: {delisted_candidates}
* Invalid Symbols Candidates Count: {invalid_candidates}
* Failure Threshold: {"5%" if market == "TW" else "25%"}
* Final Publish Ratio: {published_count / len(symbols):.2%}
* Publish Success: {summary.get("published", False)}
* CSV Write Status: {summary.get("csv_write_status", "SUCCESS")}
* Invalid Operator Actions: {summary.get("invalid_actions", 0)}
==================================================
"""
                    safe_print(summary_report)
                    summaries[market] = summary
                safe_print(json.dumps(summaries, ensure_ascii=False, separators=(",", ":")))
        safe_print(f"local quant phase={phase} free_gb={available / 1024**3:.1f}")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        safe_print(f"local quant refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
