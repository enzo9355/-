"""Authoritative US market post-close observation batch pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zoneinfo

from local_quant import publish_market_snapshot, write_stock_artifact
from reporting.source_loader import load_report_source
from stock_papi.batch.observation_products import (
    build_observation_dashboard,
    promote_observation_candidate,
    write_observation_candidate,
)
from stock_papi.config.capabilities import PredictionCapabilityState
from stock_papi.integrations.market_data.us_calendar import (
    generate_us_calendar_document,
    get_us_exchange_holidays,
)
from stock_papi.integrations.market_data.us_market_data import fetch_us_stock_history
from stock_papi.integrations.market_data.us_universe import get_us_symbols

US_ETF_SYMBOLS = [
    "SPY", "QQQ", "DIA", "IWM", "VOO", "IVV", "SOXX", "SMH",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLB", "VNQ", "GLD", "TLT", "VTI", "VEA", "VWO", "BND",
]

US_INDUSTRY_MAP = {
    "ETF專區": US_ETF_SYMBOLS,
    "科技": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "CSCO", "ADBE", "CRM", "AMD", "INTC", "TXN", "QCOM"],
    "通訊服務": ["GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T"],
    "非必需消費": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG", "LOW", "TJX"],
    "必需消費": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "MO", "CL"],
    "金融": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "BRK-B"],
    "醫療保健": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "DHR", "ISRG", "BMY"],
    "工業": ["GE", "CAT", "UNP", "HON", "BA", "RTX", "LMT", "DE", "UPS", "FDX"],
    "能源": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
    "原物料": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM"],
    "公用事業": ["NEE", "SO", "DUK", "CEG", "SRE", "AEP"],
    "房地產": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O"],
}

CORE_US_UNIVERSE = sorted(
    {sym for syms in US_INDUSTRY_MAP.values() for sym in syms}
)


def _fetch_and_write_symbol(
    root: Path,
    symbol: str,
    target_market_date: datetime.date,
) -> tuple[str, bool]:
    try:
        df = fetch_us_stock_history(symbol, target_market_date=target_market_date)
        daily = json.loads(
            df.reset_index().to_json(orient="records", date_format="iso", date_unit="ms")
        )
        if not daily:
            return symbol, False
        latest = daily[-1]
        as_of = str(latest.get("Date", "")).split("T", 1)[0]
        if as_of != target_market_date.isoformat():
            return symbol, False
        payload = {
            "schema_version": 2,
            "market": "US",
            "symbol": symbol,
            "as_of": as_of,
            "target_market_date": as_of,
            "observation_as_of": as_of,
            "latest_regular_price_date": as_of,
            "observation_kind": "regular_price",
            "lineage": {
                "source_schema_version": "us-market-data-v1",
                "observation_as_of": as_of,
                "latest_regular_price_date": as_of,
                "observation_kind": "regular_price",
            },
            "rows": len(daily),
            "latest": latest,
            "backtest": {},
            "daily": daily,
        }
        write_stock_artifact(root, "US", symbol, payload)
        return symbol, True
    except Exception as exc:
        return symbol, False


def run_us_post_close(
    root: Path | str,
    target_market_date: datetime.date,
    *,
    now: datetime.datetime | None = None,
    coverage_threshold: float = 0.95,
    max_workers: int = 20,
) -> Path:
    root = Path(root)
    if now is None:
        index_file = root / "publish" / "reports" / "v2" / "index-US.json"
        if index_file.is_file():
            try:
                index_doc = json.loads(index_file.read_text(encoding="utf-8"))
                for rep in index_doc.get("reports", []):
                    if rep.get("report_type") == "post_close" and rep.get("source_market_date") == target_market_date.isoformat():
                        pub = rep.get("published_at")
                        if pub:
                            now = datetime.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                            break
            except Exception:
                pass
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Ensure calendar exists
    year = target_market_date.year
    cal_dir = root / "publish" / "calendars" / "v1"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_file = cal_dir / f"US-{year}.json"
    if not cal_file.is_file():
        doc = generate_us_calendar_document(year)
        cal_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Get active US universe
    symbols = CORE_US_UNIVERSE
    print(f"US Active Core Universe: {len(symbols)} symbols")

    # 3. Collect regular price observation artifacts concurrently
    completed: list[str] = []
    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_and_write_symbol, root, sym, target_market_date): sym
            for sym in symbols
        }
        for fut in concurrent.futures.as_completed(futures):
            sym, success = fut.result()
            if success:
                completed.append(sym)
            else:
                failed.append(sym)

    print(f"US stock artifacts: completed={len(completed)}, failed={len(failed)}")

    # 4. Strict Observation Coverage Gate check (>95%)
    if len(completed) * 100 <= len(symbols) * 95:
        raise RuntimeError(
            f"US observation coverage {len(completed)}/{len(symbols)} ({len(completed)/len(symbols):.2%}) "
            f"fails strict >95% publishable threshold."
        )

    # 5. Publish Manifest v4 with unavailable isolation
    manifest_path = publish_market_snapshot(
        root,
        "US",
        symbols,
        generated_at=now,
        failed_symbols=failed,
        target_market_date=target_market_date,
        unavailable_symbols=failed,
    )
    print(f"Published US Manifest v4: {manifest_path}")

    # 6. Build Observation Dashboard
    source = load_report_source(root, market="US")
    pred_cap = PredictionCapabilityState.from_environment()
    dashboard = build_observation_dashboard(
        source, US_INDUSTRY_MAP, pred_cap, generated_at=now, today=target_market_date
    )

    dashboard_bytes = json.dumps(
        dashboard, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    content = {
        "dashboard_sha256": hashlib.sha256(dashboard_bytes).hexdigest(),
        "market_observation": dashboard["market_observation"],
        "industry_observations": dashboard["industry_observations"],
        "heatmap": dashboard.get("heatmap", []),
        "stock_events": dashboard["stock_events"],
        "trading_status_observations": dashboard.get("trading_status_observations", []),
        "etf_observations": dashboard["etf_observations"],
        "daily_focus": dashboard["daily_focus"],
        "data_quality": dashboard["data_quality"],
    }
    content_bytes = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    content_sha256 = hashlib.sha256(content_bytes).hexdigest()

    report_metadata = {
        "schema_version": 2,
        "kind": "absorb-report",
        "product_mode": "observation",
        "market": "US",
        "report_type": "post_close",
        "source_market_date": target_market_date.isoformat(),
        "applicable_trading_date": target_market_date.isoformat(),
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "data_as_of": target_market_date.isoformat(),
        "forecast_start_date": target_market_date.isoformat(),
        "forecast_end_date": target_market_date.isoformat(),
        "observation_start_date": target_market_date.isoformat(),
        "observation_end_date": target_market_date.isoformat(),
        "source_manifest": f"quant/v1/{source.manifest.manifest_path}",
        "source_manifest_sha256": source.manifest.manifest_sha256,
        "model_versions": {},
        "title": f"ABSORB 美股盤後市場觀察報告 ({target_market_date})",
        "summary": [f"美股 {target_market_date} 交易日收盤觀察與市場結構概況。"],
        "warnings": [],
        "content": content,
        "content_sha256": content_sha256,
        "prediction_capability": pred_cap.to_document(),
    }

    cand_dir = write_observation_candidate(
        root,
        report_metadata,
        dashboard,
    )
    promoted = promote_observation_candidate(root, cand_dir)
    print(f"Successfully promoted US observation candidate: {promoted}")
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run US official post-close observation batch pipeline")
    parser.add_argument("--root", required=True, help="Data root path")
    parser.add_argument("--target-market-date", required=True, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    target_date = datetime.date.fromisoformat(args.target_market_date)
    run_us_post_close(Path(args.root), target_date)


if __name__ == "__main__":
    main()
