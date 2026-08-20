import datetime
import gzip
import hashlib
import hmac
import io
import json
import math
import re
import time

from stock_papi.integrations.market_data.tw_trading_status import (
    evidence_sha256,
    validate_status_evidence,
)


QUANT_MANIFEST_CACHE_SECONDS = 300
MAX_QUANT_ARTIFACT_COMPRESSED_BYTES = 5 * 1024 * 1024
MAX_QUANT_ARTIFACT_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
QUANT_MANIFEST_CACHE = {}


def published_quant_manifest(market, today=None, *, load_object, cache=QUANT_MANIFEST_CACHE):
    now = time.time()
    latest_bytes = load_object(f"quant/v1/latest-{market}.json", 100_000)
    if latest_bytes is None:
        return None
    try:
        latest = json.loads(latest_bytes.decode("utf-8"))
        manifest_path = str(latest["manifest"])
        schema_version = latest.get("schema_version")
        if (
            schema_version not in {2, 3, 4}
            or latest.get("market") != market
            or re.fullmatch(
                rf"manifests/{market}-[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{12}}\.json",
                manifest_path,
            ) is None
        ):
            return None
        manifest_sha = str(latest["manifest_sha256"])
        if (
            re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None
            or not manifest_path.endswith(f"-{manifest_sha[:12]}.json")
        ):
            return None
        cache_key = (market, schema_version, manifest_sha)
        cached = cache.get(cache_key)
        if cached and now - cached[1] < QUANT_MANIFEST_CACHE_SECONDS:
            return cached[0]
        manifest_bytes = load_object(f"quant/v1/{manifest_path}", 5_000_000)
        if manifest_bytes is None or not hmac.compare_digest(
            hashlib.sha256(manifest_bytes).hexdigest(),
            manifest_sha,
        ):
            return None
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if (
            manifest.get("schema_version") != schema_version
            or manifest.get("market") != market
            or manifest.get("generated_at") != latest.get("generated_at")
        ):
            return None
        symbols = manifest.get("symbols")
        universe_count = (
            manifest.get("universe_count")
            if schema_version in (2, 3)
            else manifest.get("active_universe_count")
        )
        if schema_version == 2:
            market_date = datetime.date.fromisoformat(str(manifest["market_as_of"]))
            symbol_count = manifest.get("symbol_count")
            failure_count = manifest.get("failure_count")
            coverage = manifest.get("coverage")
            failure_rate = manifest.get("failure_rate")
            failed_symbols = manifest.get("failed_symbols")
            if (
                not isinstance(symbols, dict)
                or type(universe_count) is not int
                or type(symbol_count) is not int
                or type(failure_count) is not int
                or universe_count < 1
                or symbol_count != len(symbols)
                or failure_count != universe_count - symbol_count
                or not isinstance(failed_symbols, list)
                or len(failed_symbols) != failure_count
                or type(coverage) not in (int, float)
                or type(failure_rate) not in (int, float)
                or coverage <= 0.95
                or failure_rate >= 0.05
                or not math.isclose(coverage, symbol_count / universe_count)
                or not math.isclose(failure_rate, failure_count / universe_count)
            ):
                return None
        elif schema_version == 3:
            target = datetime.date.fromisoformat(str(manifest["target_market_date"]))
            market_date = datetime.date.fromisoformat(str(manifest["observation_as_of"]))
            observation_count = manifest.get("observation_count")
            regular_count = manifest.get("regular_price_symbol_count")
            status_count = manifest.get("expected_non_price_symbol_count")
            failure_count = manifest.get("operational_failure_count")
            denominator = manifest.get("regular_price_denominator")
            regular_coverage = manifest.get("regular_price_coverage")
            observation_coverage = manifest.get("observation_coverage")
            failure_rate = manifest.get("operational_failure_rate")
            expected = manifest.get("expected_non_price_symbols")
            failed_symbols = manifest.get("operational_failed_symbols")
            counts = (
                universe_count,
                observation_count,
                regular_count,
                status_count,
                failure_count,
                denominator,
            )
            if (
                market != "TW"
                or "market_as_of" in manifest
                or target != market_date
                or any(type(value) is not int or value < 0 for value in counts)
                or universe_count < 1
                or denominator < 1
                or not isinstance(symbols, dict)
                or not isinstance(expected, dict)
                or not isinstance(failed_symbols, list)
                or len(set(failed_symbols)) != len(failed_symbols)
                or observation_count != len(symbols)
                or status_count != len(expected)
                or failure_count != len(failed_symbols)
                or regular_count + status_count != observation_count
                or observation_count + failure_count != universe_count
                or denominator != universe_count - status_count
                or set(expected) - set(symbols)
                or set(failed_symbols) & set(symbols)
                or type(regular_coverage) not in (int, float)
                or type(observation_coverage) not in (int, float)
                or type(failure_rate) not in (int, float)
                or not math.isclose(regular_coverage, regular_count / denominator)
                or not math.isclose(observation_coverage, observation_count / universe_count)
                or not math.isclose(failure_rate, failure_count / universe_count)
                or failure_rate >= 0.05
            ):
                return None
        elif schema_version == 4:
            target = datetime.date.fromisoformat(str(manifest["target_market_date"]))
            market_date = datetime.date.fromisoformat(str(manifest["observation_as_of"]))
            observation_count = manifest.get("observation_count")
            regular_count = manifest.get("regular_price_symbol_count")
            status_count = manifest.get("verified_non_price_symbol_count")
            unavailable_count = manifest.get("unavailable_count")
            unavailable_symbols = manifest.get("unavailable_symbols")
            operational_count = manifest.get("operational_failure_count")
            operational_symbols = manifest.get("operational_failed_symbols")
            denominator = manifest.get("regular_price_denominator")
            regular_coverage = manifest.get("regular_price_coverage")
            observation_coverage = manifest.get("observation_coverage")
            expected = manifest.get("expected_non_price_symbols")
            failed_symbols = (unavailable_symbols or []) + (operational_symbols or [])
            counts = (
                universe_count,
                observation_count,
                regular_count,
                status_count,
                unavailable_count,
                operational_count,
                denominator,
            )
            if (
                market not in ("TW", "US")
                or "market_as_of" in manifest
                or target != market_date
                or any(type(value) is not int or value < 0 for value in counts)
                or universe_count < 1
                or denominator < 1
                or not isinstance(symbols, dict)
                or not isinstance(expected, dict)
                or not isinstance(unavailable_symbols, list)
                or not isinstance(operational_symbols, list)
                or len(set(unavailable_symbols)) != len(unavailable_symbols)
                or len(set(operational_symbols)) != len(operational_symbols)
                or unavailable_count != len(unavailable_symbols)
                or operational_count != 0
                or operational_count != len(operational_symbols)
                or observation_count != len(symbols)
                or status_count != len(expected)
                or regular_count + status_count != observation_count
                or observation_count + unavailable_count != universe_count
                or denominator != observation_count - status_count
                or set(expected) - set(symbols)
                or set(unavailable_symbols) & set(symbols)
                or type(regular_coverage) not in (int, float)
                or type(observation_coverage) not in (int, float)
                or not math.isclose(regular_coverage, regular_count / denominator)
                or not math.isclose(observation_coverage, observation_count / universe_count)
                or observation_count * 100 <= universe_count * 95
            ):
                return None
            for symbol, status in expected.items():
                entry = symbols.get(symbol)
                valid_symbol = (
                    bool(re.fullmatch(r"[0-9]{4,6}", str(symbol)))
                    if market == "TW"
                    else (len(str(symbol)) <= 10 and bool(re.fullmatch(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?$", str(symbol))))
                )
                if (
                    not valid_symbol
                    or not isinstance(status, dict)
                    or not isinstance(entry, dict)
                    or status.get("status")
                    not in {"official_no_regular_trade", "officially_suspended"}
                    or status.get("artifact_sha256") != entry.get("sha256")
                    or status.get("evidence_sha256")
                    != entry.get("evidence_sha256")
                    or status.get("latest_regular_price_date")
                    != entry.get("latest_regular_price_date")
                    or entry.get("observation_kind") != status.get("status")
                ):
                    return None
        else:
            return None
        age = (today or datetime.date.today()) - market_date
        if not 0 <= age.days <= 7:
            return None
    except (KeyError, TypeError, UnicodeError, ValueError):
        return None
    result = dict(manifest)
    result["symbols"] = dict(symbols)
    cache[cache_key] = (result, now)
    return result


def fetch_quant_snapshot(
    market_or_code,
    code=None,
    today=None,
    *,
    is_us_ticker_fn=None,
    load_manifest,
    load_object,
):
    if code is not None:
        market = market_or_code
        symbol = code
    else:
        symbol = market_or_code
        market = "US" if (is_us_ticker_fn and is_us_ticker_fn(symbol)) else "TW"
    if market not in ("TW", "US"):
        return None
    manifest = load_manifest(market, today=today)
    entry = (manifest or {}).get("symbols", {}).get(symbol)
    if not isinstance(entry, dict):
        return None
    try:
        manifest_schema = manifest.get("schema_version")
        path = str(entry["path"])
        digest = str(entry["sha256"])
        size = entry["size"]
        uncompressed_size = entry["uncompressed_size"]
        if (
            re.fullmatch(r"objects/[0-9a-f]{64}\.json\.gz", path) is None
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or path != f"objects/{digest}.json.gz"
            or type(size) is not int
            or not 0 < size <= MAX_QUANT_ARTIFACT_COMPRESSED_BYTES
            or type(uncompressed_size) is not int
            or not 0 < uncompressed_size <= MAX_QUANT_ARTIFACT_UNCOMPRESSED_BYTES
        ):
            return None
        if manifest_schema == 2:
            if entry.get("as_of") != manifest.get("market_as_of") or any(
                key in entry
                for key in (
                    "observation_as_of",
                    "latest_regular_price_date",
                    "observation_kind",
                    "evidence_sha256",
                )
            ):
                return None
        elif manifest_schema in (3, 4):
            if (
                market not in ("TW", "US")
                or entry.get("observation_as_of")
                != manifest.get("observation_as_of")
                or entry.get("latest_regular_price_date") != entry.get("as_of")
            ):
                return None
        else:
            return None
        compressed = load_object(f"quant/v1/{path}", size)
        if (
            compressed is None
            or len(compressed) != size
            or not hmac.compare_digest(hashlib.sha256(compressed).hexdigest(), digest)
        ):
            return None
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            decoded = stream.read(MAX_QUANT_ARTIFACT_UNCOMPRESSED_BYTES + 1)
        if len(decoded) != uncompressed_size:
            return None
        document = json.loads(decoded.decode("utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != (1 if manifest_schema == 2 else 2)
            or document.get("market") != market
            or document.get("symbol") != symbol
            or document.get("as_of") != entry.get("as_of")
            or not isinstance(document.get("backtest"), dict)
            or not isinstance(document.get("daily"), list)
        ):
            return None
        if manifest_schema in (3, 4):
            if not document["daily"] or not isinstance(document["daily"][-1], dict):
                return None
            latest_date = str(document["daily"][-1].get("Date") or "").split("T", 1)[0]
            if latest_date != entry.get("as_of"):
                return None
            latest_summary = document.get("latest")
            if (
                not isinstance(latest_summary, dict)
                or str(latest_summary.get("Date") or "").split("T", 1)[0]
                != entry.get("as_of")
            ):
                return None
            expected = manifest["expected_non_price_symbols"].get(symbol)
            status = document.get("trading_status_evidence")
            if (
                document.get("target_market_date")
                != manifest.get("target_market_date")
                or document.get("observation_as_of")
                != manifest.get("observation_as_of")
                or document.get("latest_regular_price_date")
                != entry.get("latest_regular_price_date")
                or document.get("observation_kind")
                != entry.get("observation_kind")
            ):
                return None
            if expected is None:
                if (
                    entry.get("observation_kind") != "regular_price"
                    or entry.get("as_of") != manifest.get("target_market_date")
                    or status is not None
                    or "evidence_sha256" in entry
                ):
                    return None
            elif (
                not isinstance(status, dict)
                or status.get("schema_version") != 1
                or status.get("status") != expected.get("status")
                or status.get("market") != market
                or status.get("symbol") != symbol
                or status.get("target_market_date")
                != manifest.get("target_market_date")
                or status.get("evidence_sha256") != evidence_sha256(status)
                or status.get("evidence_sha256")
                != entry.get("evidence_sha256")
                or status.get("evidence_sha256")
                != expected.get("evidence_sha256")
                or validate_status_evidence(
                    status,
                    symbol=symbol,
                    target_date=datetime.date.fromisoformat(
                        str(manifest["target_market_date"])
                    ),
                )
                != status
                or digest != expected.get("artifact_sha256")
                or entry.get("observation_kind") != expected.get("status")
            ):
                return None
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
            return None
        return document
    except (KeyError, OSError, TypeError, UnicodeError, ValueError):
        return None


published_stock_artifact = fetch_quant_snapshot
