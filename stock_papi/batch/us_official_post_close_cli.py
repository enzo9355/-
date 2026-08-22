"""Authoritative US market post-close observation batch pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zoneinfo

from local_quant import publish_market_snapshot, write_stock_artifact
from reporting.source_loader import load_report_source
from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.batch.observation_products import (
    build_observation_dashboard,
    promote_observation_candidate,
    write_observation_candidate,
)
from stock_papi.config.capabilities import PredictionCapabilityState
from stock_papi.integrations.market_data.us_calendar import (
    generate_us_calendar_document,
    get_us_calendar_documents,
)
from stock_papi.integrations.market_data.us_market_data import (
    fetch_us_stock_history,
    USObservationError,
    USObservationUnavailable,
    USProviderOperationalError,
    USSchemaError,
    USIntegrityError,
    USRateLimitError,
)
from stock_papi.integrations.market_data.us_trading_status import (
    get_us_trading_status_snapshot,
)
from stock_papi.integrations.market_data.us_universe import (
    get_us_universe_breakdown,
    USUniverseBreakdown,
)

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")

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


@dataclass(frozen=True)
class ObservationResult:
    symbol: str
    kind: str  # "R" (regular price), "N" (verified non-price), "M" (unavailable), "OP_FAIL" (operational failure)
    detail: Any = None
    error_type: str | None = None
    reason_code: str = ""


def _fetch_and_classify_symbol(
    root: Path,
    symbol: str,
    target_market_date: datetime.date,
    halt_evidence_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> ObservationResult:
    """Fetch and classify a single symbol into R, N, M, or OP_FAIL using typed exception semantics."""
    target_iso = target_market_date.isoformat()
    try:
        df = fetch_us_stock_history(symbol, target_market_date=target_market_date)
        if df.empty:
            # Empty dataframe from provider: check if authoritative halt evidence exists (N) or unavailable (M)
            if halt_evidence_by_symbol and symbol in halt_evidence_by_symbol:
                halt_doc = halt_evidence_by_symbol[symbol]
                payload = {
                    "schema_version": 2,
                    "market": "US",
                    "symbol": symbol,
                    "as_of": target_iso,
                    "target_market_date": target_iso,
                    "observation_as_of": target_iso,
                    "latest_regular_price_date": target_iso,
                    "observation_kind": halt_doc.get("status", "officially_suspended"),
                    "trading_status_evidence": halt_doc,
                    "lineage": {
                        "source_schema_version": "us-official-status-v1",
                        "observation_as_of": target_iso,
                        "latest_regular_price_date": target_iso,
                        "observation_kind": halt_doc.get("status", "officially_suspended"),
                        "trading_status_evidence_sha256": halt_doc.get("evidence_sha256"),
                    },
                    "rows": 0,
                    "latest": {},
                    "backtest": {},
                    "daily": [],
                }
                write_stock_artifact(root, "US", symbol, payload)
                return ObservationResult(symbol=symbol, kind="N", detail=halt_doc, reason_code="verified_halt")
            return ObservationResult(symbol=symbol, kind="M", detail="no_regular_trades_and_no_halt", reason_code="verified_no_regular_trade")

        daily = json.loads(
            df.reset_index().to_json(orient="records", date_format="iso", date_unit="ms")
        )
        if not daily:
            return ObservationResult(symbol=symbol, kind="M", detail="empty_daily_records", reason_code="provider_success_no_target_observation")

        latest = daily[-1]
        as_of = str(latest.get("Date", "")).split("T", 1)[0]
        if as_of == target_iso:
            # Valid regular price observation on target date (R)
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
            return ObservationResult(symbol=symbol, kind="R", detail=latest, reason_code="regular_price_observed")
        elif as_of < target_iso:
            # History exists but no trade on target date
            if halt_evidence_by_symbol and symbol in halt_evidence_by_symbol:
                halt_doc = halt_evidence_by_symbol[symbol]
                payload = {
                    "schema_version": 2,
                    "market": "US",
                    "symbol": symbol,
                    "as_of": as_of,
                    "target_market_date": target_iso,
                    "observation_as_of": target_iso,
                    "latest_regular_price_date": as_of,
                    "observation_kind": halt_doc.get("status", "officially_suspended"),
                    "trading_status_evidence": halt_doc,
                    "lineage": {
                        "source_schema_version": "us-official-status-v1",
                        "observation_as_of": target_iso,
                        "latest_regular_price_date": as_of,
                        "observation_kind": halt_doc.get("status", "officially_suspended"),
                        "trading_status_evidence_sha256": halt_doc.get("evidence_sha256"),
                    },
                    "rows": len(daily),
                    "latest": latest,
                    "backtest": {},
                    "daily": daily,
                }
                write_stock_artifact(root, "US", symbol, payload)
                return ObservationResult(symbol=symbol, kind="N", detail=halt_doc, reason_code="verified_halt")
            return ObservationResult(
                symbol=symbol,
                kind="M",
                detail=f"no_trade_on_target_date_last_trade_{as_of}",
                reason_code="provider_success_no_target_observation",
            )
        else:
            return ObservationResult(
                symbol=symbol,
                kind="OP_FAIL",
                detail=f"future date bar in history: {as_of} > {target_iso}",
                error_type="FutureDateError",
                reason_code="future_date_violation",
            )
    except USObservationUnavailable as exc:
        # Explicit legitimate observation absence class
        return ObservationResult(symbol=symbol, kind="M", detail=str(exc), reason_code="other_legitimate_unavailable")
    except USObservationError as exc:
        # Typed operational errors (USProviderOperationalError, USSchemaError, USIntegrityError, USRateLimitError)
        return ObservationResult(
            symbol=symbol,
            kind="OP_FAIL",
            detail=str(exc),
            error_type=type(exc).__name__,
            reason_code=type(exc).__name__,
        )
    except Exception as exc:
        # Any unexpected error is strictly classified as OP_FAIL (Fail-Closed)
        return ObservationResult(
            symbol=symbol,
            kind="OP_FAIL",
            detail=str(exc),
            error_type=type(exc).__name__,
            reason_code="unexpected_exception",
        )


def run_us_post_close(
    root: Path | str,
    target_market_date: datetime.date,
    *,
    now: datetime.datetime | None = None,
    coverage_threshold: float = 0.95,
    max_workers: int = 24,
    scope: str = "EQUITY_OBSERVATION",
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

    # 1. Calendars & Date Semantics
    cal_docs = get_us_calendar_documents(target_market_date.year - 1, target_market_date.year + 1)
    calendars = TradingCalendarSet.from_documents(cal_docs)
    if not calendars.is_session(target_market_date):
        raise ValueError(f"Target market date {target_market_date} is not a valid US trading session")
    applicable_trading_date = calendars.next_session(target_market_date)

    cal_dir = root / "publish" / "calendars" / "v1"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_file = cal_dir / f"US-{target_market_date.year}.json"
    if not cal_file.is_file():
        doc = generate_us_calendar_document(target_market_date.year)
        cal_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Authoritative US Active Universe & Breakdown
    breakdown: USUniverseBreakdown = get_us_universe_breakdown(root, scope=scope)
    symbols = breakdown.symbols
    print("==================================================")
    print("US ACTIVE UNIVERSE AUDIT")
    print("==================================================")
    print(f"* Configured/Listed SEC rows: {breakdown.configured_listed_count}")
    print(f"* Eligible Listed Securities: {breakdown.eligible_listed_count}")
    print(f"* Excluded non-major exchange: {breakdown.excluded_exchange_count}")
    print(f"* Excluded crypto terms:       {breakdown.excluded_crypto_count}")
    print(f"* Excluded invalid tickers:    {breakdown.excluded_invalid_count}")
    print(f"* Excluded derivative types:   {breakdown.excluded_derivative_count} ({breakdown.derivative_breakdown})")
    print(f"* Terminated / delisted count: {breakdown.terminated_delisted_count}")
    print(f"* Active Universe A Count:     {breakdown.active_universe_count}")
    print(f"* Exchange Breakdown:          {breakdown.exchange_counts}")
    print("==================================================")

    # 3. Ingest Authoritative US Trading Status Snapshot (N Partition)
    halt_evidence = get_us_trading_status_snapshot(target_market_date, symbols)
    print(f"Authoritative US Trading Halt Evidence Ingested: {len(halt_evidence)} symbols")

    # 4. Collect observations concurrently with error classification
    r_symbols: list[str] = []
    n_symbols: list[str] = []
    m_symbols: list[str] = []
    m_reasons: dict[str, str] = {}
    op_failures: list[ObservationResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_and_classify_symbol, root, sym, target_market_date, halt_evidence): sym
            for sym in symbols
        }
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res.kind == "R":
                r_symbols.append(res.symbol)
            elif res.kind == "N":
                n_symbols.append(res.symbol)
            elif res.kind == "M":
                m_symbols.append(res.symbol)
                m_reasons[res.symbol] = res.reason_code
            else:
                op_failures.append(res)

    r_symbols.sort()
    n_symbols.sort()
    m_symbols.sort()

    obs_count = len(r_symbols) + len(n_symbols)
    active_count = len(symbols)
    coverage = obs_count / active_count if active_count > 0 else 0.0

    print("==================================================")
    print("US OBSERVATION BATCH RESULT")
    print("==================================================")
    print(f"* Target Market Date:       {target_market_date}")
    print(f"* Source Market Date:       {target_market_date}")
    print(f"* Applicable Trading Date:  {applicable_trading_date}")
    print(f"* Active Universe A:        {active_count}")
    print(f"* R (Regular Price):        {len(r_symbols)}")
    print(f"* N (Verified Non-Price):   {len(n_symbols)}")
    print(f"* M (Legitimate Unavail):   {len(m_symbols)}")
    print(f"* Operational Failures:     {len(op_failures)}")
    print(f"* Observation Coverage:     {coverage:.4%}")
    print("==================================================")

    # 5. Save Machine-Readable Data Quality Audit Evidence
    dq_dir = root / "release" / "data-quality"
    dq_dir.mkdir(parents=True, exist_ok=True)
    audit_file = dq_dir / f"us-universe-audit-{target_market_date.strftime('%Y%m%d')}.json"

    # Aggregate M reasons
    m_reason_counts: dict[str, int] = {}
    for r in m_reasons.values():
        m_reason_counts[r] = m_reason_counts.get(r, 0) + 1

    audit_payload = {
        "schema_version": 1,
        "market": "US",
        "scope": scope,
        "target_market_date": target_market_date.isoformat(),
        "source_market_date": target_market_date.isoformat(),
        "applicable_trading_date": applicable_trading_date.isoformat(),
        "configured_listed_count": breakdown.configured_listed_count,
        "eligible_listed_count": breakdown.eligible_listed_count,
        "active_universe_count": breakdown.active_universe_count,
        "excluded_exchange_count": breakdown.excluded_exchange_count,
        "excluded_crypto_count": breakdown.excluded_crypto_count,
        "excluded_invalid_count": breakdown.excluded_invalid_count,
        "excluded_derivative_count": breakdown.excluded_derivative_count,
        "derivative_breakdown": breakdown.derivative_breakdown,
        "terminated_delisted_count": breakdown.terminated_delisted_count,
        "exchange_counts": breakdown.exchange_counts,
        "r_count": len(r_symbols),
        "n_count": len(n_symbols),
        "m_count": len(m_symbols),
        "f_count": len(op_failures),
        "observation_coverage": coverage,
        "m_reason_counts": m_reason_counts,
        "m_symbols": m_reasons,
        "operational_failures": [
            {"symbol": f.symbol, "error_type": f.error_type, "detail": str(f.detail)}
            for f in op_failures
        ],
    }
    audit_file.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved machine-readable audit evidence: {audit_file}")

    # 4. Strict Contract Invariant & Gate Checks
    # Invariant: R, N, M are mutually disjoint
    r_set, n_set, m_set = set(r_symbols), set(n_symbols), set(m_symbols)
    if r_set & n_set or r_set & m_set or n_set & m_set:
        raise RuntimeError("US observation partitions R, N, M are not mutually disjoint!")
    if (len(r_set) + len(n_set) + len(m_set) + len(op_failures)) != active_count:
        raise RuntimeError("US observation partition sum does not equal active universe count!")

    # Blocker 2 Gate: Any operational failure -> FAIL CLOSED
    if op_failures:
        failure_samples = [(f.symbol, f.error_type, f.detail) for f in op_failures[:10]]
        raise RuntimeError(
            f"US PostClose batch failed: {len(op_failures)} operational symbol failures encountered. "
            f"Fail-closed contract prevents publication. Samples: {failure_samples}"
        )

    # Strict >95% Observation Coverage Gate
    if obs_count * 100 <= active_count * 95:
        raise RuntimeError(
            f"US observation coverage {obs_count}/{active_count} ({coverage:.2%}) "
            f"fails strict >95% publishable threshold (exactly 95% fails)."
        )

    # 5. Publish Manifest v4 with full active universe
    manifest_path = publish_market_snapshot(
        root,
        "US",
        symbols,
        generated_at=now,
        failed_symbols=[],
        target_market_date=target_market_date,
        unavailable_symbols=m_symbols,
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
        "applicable_trading_date": applicable_trading_date.isoformat(),
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "data_as_of": target_market_date.isoformat(),
        "forecast_start_date": applicable_trading_date.isoformat(),
        "forecast_end_date": applicable_trading_date.isoformat(),
        "observation_start_date": target_market_date.isoformat(),
        "observation_end_date": applicable_trading_date.isoformat(),
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
    parser.add_argument("--scope", default="EQUITY_OBSERVATION", choices=["EQUITY_OBSERVATION", "ALL_LISTED"], help="Universe product scope")
    parser.add_argument("--max-workers", type=int, default=24, help="Max concurrent workers")
    args = parser.parse_args()

    target_date = datetime.date.fromisoformat(args.target_market_date)
    run_us_post_close(Path(args.root), target_date, scope=args.scope, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
