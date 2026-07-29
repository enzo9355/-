import datetime
import gzip
import hashlib
import hmac
import io
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

from .config import ReportConfig
from .exceptions import ReportSourceError
from .schemas import LoadedReportSource, ReportSourceManifest, StockSnapshot
from stock_papi.integrations.market_data.tw_trading_status import (
    evidence_sha256,
    validate_status_evidence,
)

StockSnapshot = StockSnapshot


def _read_limited(path: Path, limit: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if not 0 < size <= limit:
            raise ReportSourceError(f"{label} size is invalid")
        content = path.read_bytes()
    except OSError as exc:
        raise ReportSourceError(f"{label} is unavailable") from exc
    if len(content) != size:
        raise ReportSourceError(f"{label} changed while reading")
    return content


def _safe_child(root: Path, relative: str, label: str) -> Path:
    """拒絕跳脫發布根目錄或經過 symlink/junction 的來源路徑。"""
    try:
        root_resolved = root.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root_resolved):
            raise ReportSourceError(f"{label} escaped publish root")
        current = root / relative
        while current != root:
            metadata = current.lstat()
            if current.is_symlink() or (
                getattr(metadata, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ):
                raise ReportSourceError(f"{label} uses a reparse point")
            current = current.parent
    except OSError as exc:
        raise ReportSourceError(f"{label} is unavailable") from exc
    return path


def _validate_finite_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportSourceError("source contains non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReportSourceError("source contains non-string key")
            _validate_finite_json(item)
        return
    raise ReportSourceError("source contains unsupported JSON value")


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ReportSourceError(f"{label} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ReportSourceError(f"{label} must contain an object")
    _validate_finite_json(document)
    return document


def _validate_manifest_v2(document: dict[str, Any], market: str) -> None:
    try:
        universe = document["universe_count"]
        count = document["symbol_count"]
        failures = document["failure_count"]
        coverage = document["coverage"]
        failure_rate = document["failure_rate"]
        symbols = document["symbols"]
        failed_symbols = document["failed_symbols"]
        market_as_of = datetime.date.fromisoformat(str(document["market_as_of"]))
        datetime.datetime.fromisoformat(str(document["generated_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportSourceError("manifest fields are invalid") from exc
    if (
        document.get("schema_version") != 2
        or document.get("market") != market
        or type(universe) is not int
        or type(count) is not int
        or type(failures) is not int
        or not isinstance(symbols, dict)
        or not isinstance(failed_symbols, list)
        or type(coverage) not in (int, float)
        or type(failure_rate) not in (int, float)
        or universe < 1
        or count != len(symbols)
        or failures != universe - count
        or len(failed_symbols) != failures
        or not math.isclose(float(coverage), count / universe)
        or not math.isclose(float(failure_rate), failures / universe)
        or not 0 < float(coverage) <= 1
        or not 0 <= float(failure_rate) < 0.05
        or market_as_of > datetime.date.today()
    ):
        raise ReportSourceError("manifest consistency check failed")


def _validate_manifest_v3(document: dict[str, Any], market: str) -> None:
    try:
        target = datetime.date.fromisoformat(str(document["target_market_date"]))
        observation = datetime.date.fromisoformat(str(document["observation_as_of"]))
        datetime.datetime.fromisoformat(
            str(document["generated_at"]).replace("Z", "+00:00")
        )
        universe = document["universe_count"]
        observation_count = document["observation_count"]
        regular_count = document["regular_price_symbol_count"]
        status_count = document["expected_non_price_symbol_count"]
        failure_count = document["operational_failure_count"]
        denominator = document["regular_price_denominator"]
        regular_coverage = document["regular_price_coverage"]
        observation_coverage = document["observation_coverage"]
        failure_rate = document["operational_failure_rate"]
        symbols = document["symbols"]
        expected = document["expected_non_price_symbols"]
        failed = document["operational_failed_symbols"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportSourceError("manifest v3 fields are invalid") from exc
    numeric_counts = (
        universe,
        observation_count,
        regular_count,
        status_count,
        failure_count,
        denominator,
    )
    if (
        document.get("schema_version") != 3
        or market != "TW"
        or document.get("market") != market
        or "market_as_of" in document
        or observation != target
        or target > datetime.date.today()
        or any(type(value) is not int or value < 0 for value in numeric_counts)
        or universe < 1
        or denominator < 1
        or not isinstance(symbols, dict)
        or not isinstance(expected, dict)
        or not isinstance(failed, list)
        or len(set(failed)) != len(failed)
        or any(re.fullmatch(r"[0-9]{4,6}", str(item)) is None for item in failed)
        or len(symbols) != observation_count
        or len(expected) != status_count
        or len(failed) != failure_count
        or regular_count + status_count != observation_count
        or observation_count + failure_count != universe
        or denominator != universe - status_count
        or set(expected) - set(symbols)
        or set(failed) & set(symbols)
        or type(regular_coverage) not in (int, float)
        or type(observation_coverage) not in (int, float)
        or type(failure_rate) not in (int, float)
        or not math.isclose(float(regular_coverage), regular_count / denominator)
        or not math.isclose(float(observation_coverage), observation_count / universe)
        or not math.isclose(float(failure_rate), failure_count / universe)
        or not 0 < float(regular_coverage) <= 1
        or not 0 < float(observation_coverage) <= 1
        or not 0 <= float(failure_rate) < 0.05
    ):
        raise ReportSourceError("manifest v3 consistency check failed")
    for symbol, status in expected.items():
        if (
            re.fullmatch(r"[0-9]{4,6}", str(symbol)) is None
            or not isinstance(status, dict)
            or status.get("status")
            not in {"official_no_regular_trade", "officially_suspended"}
            or re.fullmatch(
                r"[0-9a-f]{64}", str(status.get("evidence_sha256") or "")
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(status.get("artifact_sha256") or "")
            )
            is None
        ):
            raise ReportSourceError("manifest v3 status entry is invalid")
        try:
            latest = datetime.date.fromisoformat(
                str(status["latest_regular_price_date"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReportSourceError("manifest v3 status date is invalid") from exc
        if latest >= target:
            raise ReportSourceError("manifest v3 status date is invalid")


def _validate_manifest(document: dict[str, Any], market: str) -> None:
    if document.get("schema_version") == 2:
        _validate_manifest_v2(document, market)
    elif document.get("schema_version") == 3:
        _validate_manifest_v3(document, market)
    else:
        raise ReportSourceError("manifest schema is unsupported")


def _manifest_as_of(document: dict[str, Any]) -> datetime.date:
    key = "market_as_of" if document.get("schema_version") == 2 else "observation_as_of"
    return datetime.date.fromisoformat(str(document[key]))


def _decompress_object(content: bytes, limit: int) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as stream:
            decoded = stream.read(limit + 1)
    except OSError as exc:
        raise ReportSourceError("stock object gzip is invalid") from exc
    if len(decoded) > limit:
        raise ReportSourceError("stock object expands beyond limit")
    return decoded


def _load_manifest_source(
    publish: Path,
    market: str,
    manifest_relative: str,
    manifest_sha: str,
    settings: ReportConfig,
    report_date: datetime.date | None = None,
    expected_schema: int | None = None,
    expected_generated_at: str | None = None,
) -> LoadedReportSource:
    """載入一份已指定 SHA-256 的 immutable manifest。"""
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(manifest_sha)) is None
        or not manifest_relative.endswith(f"-{manifest_sha[:12]}.json")
    ):
        raise ReportSourceError("manifest hash content address mismatch")
    manifest_path = _safe_child(publish, manifest_relative, "manifest")
    manifest_bytes = _read_limited(manifest_path, 5_000_000, "manifest")
    if not hmac.compare_digest(hashlib.sha256(manifest_bytes).hexdigest(), manifest_sha):
        raise ReportSourceError("manifest hash mismatch")
    manifest = _json_object(manifest_bytes, "manifest")
    _validate_manifest(manifest, market)
    if expected_schema is not None and manifest.get("schema_version") != expected_schema:
        raise ReportSourceError("pointer and manifest schema mismatch")
    if (
        expected_generated_at is not None
        and manifest.get("generated_at") != expected_generated_at
    ):
        raise ReportSourceError("pointer and manifest generation mismatch")
    schema_version = int(manifest["schema_version"])
    as_of = _manifest_as_of(manifest)
    if report_date is not None and as_of != report_date:
        raise ReportSourceError("requested report date does not match manifest")

    stocks = []
    for symbol in sorted(manifest["symbols"]):
        entry = manifest["symbols"][symbol]
        if not isinstance(entry, dict) or re.fullmatch(r"[0-9]{4,6}", str(symbol)) is None:
            raise ReportSourceError("manifest symbol entry is invalid")
        relative = str(entry.get("path") or "")
        digest = str(entry.get("sha256") or "")
        size = entry.get("size")
        if (
            re.fullmatch(r"objects/[0-9a-f]{64}\.json\.gz", relative) is None
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative != f"objects/{digest}.json.gz"
            or type(size) is not int
            or not 0 < size <= settings.max_gzip_bytes
        ):
            raise ReportSourceError("stock object path or metadata is invalid")
        if schema_version == 2 and entry.get("as_of") != manifest["market_as_of"]:
            raise ReportSourceError("stock object v2 date metadata is invalid")
        if schema_version == 2 and any(
            key in entry
            for key in (
                "observation_as_of",
                "latest_regular_price_date",
                "observation_kind",
                "evidence_sha256",
            )
        ):
            raise ReportSourceError("stock object v2 status metadata is invalid")
        if schema_version == 3 and (
            entry.get("observation_as_of") != manifest["observation_as_of"]
            or entry.get("latest_regular_price_date") != entry.get("as_of")
            or entry.get("observation_kind")
            not in {
                "regular_price",
                "official_no_regular_trade",
                "officially_suspended",
            }
        ):
            raise ReportSourceError("stock object v3 date metadata is invalid")
        object_path = _safe_child(publish, relative, "stock object")
        object_bytes = _read_limited(object_path, settings.max_gzip_bytes, "stock object")
        if len(object_bytes) != size or not hmac.compare_digest(
            hashlib.sha256(object_bytes).hexdigest(), digest
        ):
            raise ReportSourceError("stock object size or hash mismatch")
        decoded = _decompress_object(object_bytes, settings.max_uncompressed_bytes)
        uncompressed_size = entry.get("uncompressed_size")
        if type(uncompressed_size) is not int or uncompressed_size != len(decoded):
            raise ReportSourceError("stock object uncompressed size mismatch")
        document = _json_object(decoded, "stock object")
        expected_object_schema = 1 if schema_version == 2 else 2
        if (
            document.get("schema_version") != expected_object_schema
            or document.get("market") != market
            or document.get("symbol") != symbol
            or document.get("as_of") != entry.get("as_of")
            or document.get("model_version") != entry.get("model_version")
            or not isinstance(document.get("daily"), list)
            or not document["daily"]
            or not all(isinstance(row, dict) for row in document["daily"])
            or not isinstance(document.get("backtest"), dict)
        ):
            raise ReportSourceError("stock object schema mismatch")
        latest_date = str(document["daily"][-1].get("Date") or "").split("T", 1)[0]
        if latest_date != entry.get("as_of"):
            raise ReportSourceError("stock object daily as_of mismatch")
        if schema_version == 3:
            latest_summary = document.get("latest")
            expected = manifest["expected_non_price_symbols"].get(symbol)
            status = document.get("trading_status_evidence")
            if (
                not isinstance(latest_summary, dict)
                or str(latest_summary.get("Date") or "").split("T", 1)[0]
                != entry.get("as_of")
                or
                document.get("target_market_date")
                != manifest["target_market_date"]
                or document.get("observation_as_of")
                != manifest["observation_as_of"]
                or document.get("latest_regular_price_date")
                != entry.get("latest_regular_price_date")
                or document.get("observation_kind")
                != entry.get("observation_kind")
            ):
                raise ReportSourceError("stock object v3 observation mismatch")
            if expected is None:
                if (
                    entry.get("observation_kind") != "regular_price"
                    or entry.get("as_of") != manifest["target_market_date"]
                    or status is not None
                    or "evidence_sha256" in entry
                ):
                    raise ReportSourceError("regular price object is invalid")
            elif (
                not isinstance(status, dict)
                or status.get("schema_version") != 1
                or status.get("status") != expected.get("status")
                or status.get("market") != market
                or status.get("symbol") != symbol
                or status.get("target_market_date")
                != manifest["target_market_date"]
                or status.get("evidence_sha256") != evidence_sha256(status)
                or entry.get("evidence_sha256")
                != status.get("evidence_sha256")
                or expected.get("evidence_sha256")
                != status.get("evidence_sha256")
                or expected.get("artifact_sha256") != digest
                or expected.get("latest_regular_price_date")
                != entry.get("latest_regular_price_date")
                or entry.get("observation_kind") != expected.get("status")
            ):
                raise ReportSourceError("status object evidence mismatch")
            if expected is not None:
                try:
                    validate_status_evidence(
                        status,
                        symbol=symbol,
                        target_date=as_of,
                    )
                except (TypeError, ValueError) as exc:
                    raise ReportSourceError(
                        "status object evidence mismatch"
                    ) from exc
        elif any(
            key in document
            for key in (
                "target_market_date",
                "observation_as_of",
                "latest_regular_price_date",
                "observation_kind",
                "trading_status_evidence",
            )
        ):
            raise ReportSourceError("stock object v2 carries status evidence")
        stocks.append(StockSnapshot.from_document(document, digest, size))

    expected_stock_count = (
        manifest["symbol_count"]
        if schema_version == 2
        else manifest["observation_count"]
    )
    if len(stocks) != expected_stock_count:
        raise ReportSourceError("manifest symbol count mismatch")
    source_manifest = ReportSourceManifest(
        schema_version=schema_version,
        market=market,
        generated_at=str(manifest["generated_at"]),
        market_as_of=as_of,
        universe_count=manifest["universe_count"],
        symbol_count=expected_stock_count,
        failure_count=(
            manifest["failure_count"]
            if schema_version == 2
            else manifest["operational_failure_count"]
        ),
        failure_rate=float(
            manifest["failure_rate"]
            if schema_version == 2
            else manifest["operational_failure_rate"]
        ),
        coverage=float(
            manifest["coverage"]
            if schema_version == 2
            else manifest["observation_coverage"]
        ),
        failed_symbols=[
            str(item)
            for item in (
                manifest["failed_symbols"]
                if schema_version == 2
                else manifest["operational_failed_symbols"]
            )
        ],
        manifest_path=manifest_relative,
        manifest_sha256=manifest_sha,
        target_market_date=(as_of if schema_version == 3 else None),
        observation_as_of=(as_of if schema_version == 3 else None),
        regular_price_symbol_count=manifest.get("regular_price_symbol_count"),
        expected_non_price_symbol_count=manifest.get(
            "expected_non_price_symbol_count"
        ),
        operational_failure_count=manifest.get("operational_failure_count"),
        regular_price_denominator=manifest.get("regular_price_denominator"),
        regular_price_coverage=(
            float(manifest["regular_price_coverage"])
            if schema_version == 3
            else None
        ),
        observation_coverage=(
            float(manifest["observation_coverage"])
            if schema_version == 3
            else None
        ),
        expected_non_price_symbols={
            str(key): dict(value)
            for key, value in manifest.get(
                "expected_non_price_symbols", {}
            ).items()
        },
        operational_failed_symbols=[
            str(item)
            for item in manifest.get("operational_failed_symbols", [])
        ],
    )
    return LoadedReportSource(source_manifest, stocks)


def load_report_source(
    root: Path,
    market: str = "TW",
    *,
    report_date: datetime.date | None = None,
    config: ReportConfig | None = None,
) -> LoadedReportSource:
    """從 latest 指標安全載入 manifest 所列的台股快照。"""
    settings = config or ReportConfig(root=Path(root), market=market)
    if market != "TW":
        raise ReportSourceError("第一階段只支援 TW 日報")
    publish = Path(root) / "publish" / "quant" / "v1"
    latest = _json_object(_read_limited(publish / "latest-TW.json", 100_000, "latest"), "latest")
    manifest_relative = str(latest.get("manifest") or "")
    manifest_sha = str(latest.get("manifest_sha256") or "")
    latest_schema = latest.get("schema_version")
    if (
        latest_schema not in {2, 3}
        or latest.get("market") != market
        or re.fullmatch(r"manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json", manifest_relative) is None
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None
    ):
        raise ReportSourceError("latest pointer is invalid")
    return _load_manifest_source(
        publish,
        market,
        manifest_relative,
        manifest_sha,
        settings,
        report_date,
        expected_schema=int(latest_schema),
        expected_generated_at=str(latest.get("generated_at") or ""),
    )


def load_report_source_manifest(
    root: Path,
    manifest_path: str,
    manifest_sha256: str,
    *,
    market: str = "TW",
    report_date: datetime.date | None = None,
    config: ReportConfig | None = None,
) -> LoadedReportSource:
    """以明確 path + SHA 載入 immutable manifest，不讀取可變 latest pointer。"""
    prefix = "quant/v1/"
    if not isinstance(manifest_path, str) or not manifest_path.startswith(prefix):
        raise ReportSourceError("explicit manifest path is invalid")
    relative = manifest_path[len(prefix) :]
    if (
        re.fullmatch(
            r"manifests/TW-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\.json", relative
        )
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(manifest_sha256)) is None
    ):
        raise ReportSourceError("explicit manifest identity is invalid")
    settings = config or ReportConfig(root=Path(root), market=market)
    if market != "TW":
        raise ReportSourceError("第一階段只支援 TW 日報")
    publish = Path(root) / "publish" / "quant" / "v1"
    return _load_manifest_source(
        publish,
        market,
        relative,
        manifest_sha256,
        settings,
        report_date,
    )


def load_previous_report_source(
    root: Path,
    before: datetime.date,
    market: str = "TW",
    *,
    config: ReportConfig | None = None,
) -> LoadedReportSource | None:
    """找出目前交易日前最新一份可完整驗證的 immutable manifest。"""
    settings = config or ReportConfig(root=Path(root), market=market)
    if market != "TW":
        raise ReportSourceError("第一階段只支援 TW 日報")
    publish = Path(root) / "publish" / "quant" / "v1"
    candidates = []
    for path in (publish / "manifests").glob("TW-*.json"):
        match = re.fullmatch(
            r"TW-[0-9]{8}T[0-9]{6}Z-([0-9a-f]{12})\.json", path.name
        )
        if match is None:
            continue
        try:
            relative = f"manifests/{path.name}"
            safe_path = _safe_child(publish, relative, "previous manifest")
            content = _read_limited(safe_path, 5_000_000, "previous manifest")
            digest = hashlib.sha256(content).hexdigest()
            if not hmac.compare_digest(match.group(1), digest[:12]):
                continue
            document = _json_object(content, "previous manifest")
            _validate_manifest(document, market)
            as_of = _manifest_as_of(document)
            if as_of < before:
                candidates.append((as_of, str(document["generated_at"]), relative, digest))
        except (OSError, ReportSourceError, TypeError, ValueError):
            continue
    for _as_of, _generated_at, relative, digest in sorted(candidates, reverse=True):
        try:
            return _load_manifest_source(publish, market, relative, digest, settings)
        except ReportSourceError:
            continue
    return None
