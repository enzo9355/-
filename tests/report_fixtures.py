import datetime
import gzip
import hashlib
import json
from pathlib import Path

from stock_papi.integrations.market_data.tw_trading_status import evidence_sha256
from stock_papi.quant.features import CALCULATED_COLUMNS


def stock_document(
    symbol: str,
    *,
    start_price: float = 100.0,
    rows: int = 70,
    as_of: str = "2026-07-03",
    ai_probability: float = 70.0,
) -> dict:
    """建立明確標示為測試用途的股票快照。"""
    end = datetime.date.fromisoformat(as_of)
    dates = [end - datetime.timedelta(days=rows - 1 - index) for index in range(rows)]
    daily = []
    for index, day in enumerate(dates):
        close = start_price + index
        daily.append({
            "Date": day.isoformat() + "T00:00:00.000",
            "Close": close,
            "MA20": close - 1,
            "MA60": close - 2,
            "AI_P": ai_probability if index >= 5 else None,
            "RSI": 55.0,
            "RET_1": 0.01,
            "RET_5": 0.05,
            "RET_20": 0.20,
            "VOL_RATIO": 1.2,
            "INST_NET_RATIO": 0.02,
            "ForeignNet": 1000.0,
            "MARKET_RET_1": 0.005,
            "MARKET_RET_5": 0.025,
            "MARKET_RET_20": 0.10,
            "MARKET_VOL_20": 0.012,
            "DATA_PRICE_WARNING": 0.0,
            "OPTION_DATA_MISSING": 0.0,
        })
    return {
        "schema_version": 1,
        "market": "TW",
        "symbol": symbol,
        "name": f"測試股票 {symbol}",
        "as_of": as_of,
        "model_version": "lgbm-5d-v1",
        "latest": daily[-1],
        "backtest": {"accuracy": 55.0, "top_features": ["月線趨勢支撐"]},
        "daily": daily,
        "sample_data": True,
    }


def warmup_stock_document(
    symbol: str,
    *,
    rows: int = 70,
    warmup_rows: int = 20,
    as_of: str = "2026-07-03",
) -> dict:
    document = stock_document(symbol, rows=rows, as_of=as_of)
    document["sample_data"] = False
    for index, row in enumerate(document["daily"]):
        close = float(row["Close"])
        row.update(
            Open=close - 0.5,
            High=close + 1.0,
            Low=close - 1.0,
            Volume=float(1000 + index),
        )
        for offset, name in enumerate(CALCULATED_COLUMNS, 1):
            row[name] = None if index < warmup_rows else float(index + offset)
        row["AI_P"] = None if index < warmup_rows else 60.0
    document["latest"] = dict(document["daily"][-1])
    return document


def write_quant_publish(root: Path, documents: list[dict]) -> Path:
    """建立 content-addressed 測試快照，不代表正式資料。"""
    publish = root / "publish" / "quant" / "v1"
    objects = publish / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    entries = {}
    for document in documents:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        compressed = gzip.compress(encoded, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        relative = f"objects/{digest}.json.gz"
        (publish / relative).write_bytes(compressed)
        entries[document["symbol"]] = {
            "path": relative,
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": len(encoded),
            "as_of": document["as_of"],
            "model_version": document["model_version"],
        }
    manifest = {
        "schema_version": 2,
        "market": "TW",
        "generated_at": "2026-07-03T10:00:00Z",
        "universe_count": len(entries),
        "symbol_count": len(entries),
        "failure_count": 0,
        "failure_rate": 0.0,
        "coverage": 1.0,
        "failed_symbols": [],
        "market_as_of": max(document["as_of"] for document in documents),
        "symbols": entries,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_relative = f"manifests/TW-20260703T100000Z-{manifest_sha[:12]}.json"
    manifest_path = publish / manifest_relative
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    latest = {
        "schema_version": 2,
        "market": "TW",
        "generated_at": "2026-07-03T10:00:00Z",
        "manifest": manifest_relative,
        "manifest_sha256": manifest_sha,
    }
    (publish / "latest-TW.json").write_text(
        json.dumps(latest, separators=(",", ":")), encoding="utf-8"
    )
    return publish


def status_stock_document(symbol: str = "2303") -> dict:
    target = "2026-07-29"
    latest = "2026-07-16"
    status = {
        "schema_version": 1,
        "status": "official_no_regular_trade",
        "market": "TW",
        "exchange": "TWSE",
        "symbol": symbol,
        "target_market_date": target,
        "source_id": "twse_price",
        "payload_sha256": "a" * 64,
        "raw_row_sha256": "b" * 64,
        "raw_fields": {"symbol": symbol, "name": f"測試股票 {symbol}", "open": "--", "high": "--", "low": "--", "close": "--", "volume": "0"},
        "parser_version": "tw-official-historical-parser-v3",
    }
    status["evidence_sha256"] = evidence_sha256(status)
    return {
        "schema_version": 2,
        "market": "TW",
        "symbol": symbol,
        "name": f"測試股票 {symbol}",
        "as_of": latest,
        "target_market_date": target,
        "observation_as_of": target,
        "latest_regular_price_date": latest,
        "observation_kind": status["status"],
        "trading_status_evidence": status,
        "model_version": "observation-source-v1",
        "latest": {"Date": latest + "T00:00:00.000", "Close": 100.0},
        "backtest": {},
        "daily": [{"Date": latest + "T00:00:00.000", "Close": 100.0}],
    }


def regular_v3_stock_document(symbol: str = "2330") -> dict:
    target = "2026-07-29"
    return {
        "schema_version": 2,
        "market": "TW",
        "symbol": symbol,
        "name": f"測試股票 {symbol}",
        "as_of": target,
        "target_market_date": target,
        "observation_as_of": target,
        "latest_regular_price_date": target,
        "observation_kind": "regular_price",
        "trading_status_evidence": None,
        "model_version": "observation-source-v1",
        "latest": {"Date": target + "T00:00:00.000", "Close": 100.0},
        "backtest": {},
        "daily": [{"Date": target + "T00:00:00.000", "Close": 100.0}],
    }


def write_quant_publish_v3(root: Path) -> Path:
    publish = root / "publish" / "quant" / "v1"
    (publish / "objects").mkdir(parents=True, exist_ok=True)
    documents = [regular_v3_stock_document(), status_stock_document()]
    entries = {}
    expected = {}
    for document in documents:
        encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        compressed = gzip.compress(encoded, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        relative = f"objects/{digest}.json.gz"
        (publish / relative).write_bytes(compressed)
        entry = {
            "path": relative,
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": len(encoded),
            "as_of": document["as_of"],
            "observation_as_of": document["observation_as_of"],
            "latest_regular_price_date": document["latest_regular_price_date"],
            "observation_kind": document["observation_kind"],
            "model_version": document["model_version"],
        }
        status = document["trading_status_evidence"]
        if status is not None:
            entry["evidence_sha256"] = status["evidence_sha256"]
            expected[document["symbol"]] = {
                "status": status["status"],
                "evidence_sha256": status["evidence_sha256"],
                "artifact_sha256": digest,
                "latest_regular_price_date": document["latest_regular_price_date"],
            }
        entries[document["symbol"]] = entry
    manifest = {
        "schema_version": 3,
        "market": "TW",
        "generated_at": "2026-07-30T01:00:00Z",
        "target_market_date": "2026-07-29",
        "observation_as_of": "2026-07-29",
        "universe_count": 2,
        "observation_count": 2,
        "regular_price_symbol_count": 1,
        "expected_non_price_symbol_count": 1,
        "operational_failure_count": 0,
        "regular_price_denominator": 1,
        "regular_price_coverage": 1.0,
        "observation_coverage": 1.0,
        "operational_failure_rate": 0.0,
        "expected_non_price_symbols": expected,
        "operational_failed_symbols": [],
        "symbols": entries,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    relative = f"manifests/TW-20260730T010000Z-{digest[:12]}.json"
    path = publish / relative
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(manifest_bytes)
    (publish / "latest-TW.json").write_text(json.dumps({
        "schema_version": 3,
        "market": "TW",
        "generated_at": "2026-07-30T01:00:00Z",
        "manifest": relative,
        "manifest_sha256": digest,
    }, separators=(",", ":")), encoding="utf-8")
    return publish


def write_quant_publish_v4(root, *, status_symbol="2303"):
    """Write a v4 status-aware quant publish with a 20/21 unavailable partition."""
    publish = Path(root) / "publish" / "quant" / "v1"
    publish.mkdir(parents=True, exist_ok=True)
    (publish / "objects").mkdir(exist_ok=True)
    regular_symbols = [f"{3000 + index:04d}" for index in range(19)]
    documents = [
        regular_v3_stock_document(symbol) for symbol in regular_symbols
    ] + [status_stock_document(status_symbol)]
    entries = {}
    expected = {}
    for document in documents:
        encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        compressed = gzip.compress(encoded, mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        relative = f"objects/{digest}.json.gz"
        (publish / relative).write_bytes(compressed)
        entry = {
            "path": relative,
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": len(encoded),
            "as_of": document["as_of"],
            "observation_as_of": document["observation_as_of"],
            "latest_regular_price_date": document["latest_regular_price_date"],
            "observation_kind": document["observation_kind"],
            "model_version": document["model_version"],
        }
        status = document["trading_status_evidence"]
        if status is not None:
            entry["evidence_sha256"] = status["evidence_sha256"]
            expected[document["symbol"]] = {
                "status": status["status"],
                "evidence_sha256": status["evidence_sha256"],
                "artifact_sha256": digest,
                "latest_regular_price_date": document["latest_regular_price_date"],
            }
        entries[document["symbol"]] = entry
    manifest = {
        "schema_version": 4,
        "market": "TW",
        "generated_at": "2026-07-30T01:00:00Z",
        "target_market_date": "2026-07-29",
        "observation_as_of": "2026-07-29",
        "active_universe_count": 21,
        "observation_count": 20,
        "regular_price_symbol_count": 19,
        "verified_non_price_symbol_count": 1,
        "unavailable_count": 1,
        "unavailable_symbols": ["6001"],
        "operational_failure_count": 0,
        "operational_failed_symbols": [],
        "operational_failure_rate": 0.0,
        "observation_coverage": 20 / 21,
        "regular_price_denominator": 19,
        "regular_price_coverage": 1.0,
        "expected_non_price_symbols": expected,
        "symbols": entries,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    relative = f"manifests/TW-20260730T010000Z-{digest[:12]}.json"
    path = publish / relative
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(manifest_bytes)
    (publish / "latest-TW.json").write_text(json.dumps({
        "schema_version": 4,
        "market": "TW",
        "generated_at": "2026-07-30T01:00:00Z",
        "manifest": relative,
        "manifest_sha256": digest,
    }, separators=(",", ":")), encoding="utf-8")
    return publish
