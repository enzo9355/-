"""Run the TW observation batch with date-addressable official bulk snapshots."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.integrations.market_data.tw_official_bulk import OfficialSourceFailure
from stock_papi.integrations.market_data.tw_official_historical_guarded import (
    MAX_CATCHUP_SESSIONS,
    OfficialSnapshotSeries,
    build_official_snapshot_series,
)
from stock_papi.quant.tw_artifact_audit import audit_artifact_dates
from stock_papi.quant.tw_incremental import OfficialCompatFetcher


def _universe_sha256(symbols: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(symbols, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_calendar_set(paths: list[Path]) -> TradingCalendarSet:
    if not paths:
        raise ValueError("at least one calendar artifact is required")
    documents = []
    for path in paths:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("calendar artifact is unreadable") from exc
        documents.append(document)
    return TradingCalendarSet.from_documents(documents)


def _required_trading_dates(
    calendars: TradingCalendarSet,
    *,
    earliest_latest_date: datetime.date,
    target_market_date: datetime.date,
) -> tuple[datetime.date, ...]:
    if not calendars.is_session(target_market_date):
        raise ValueError("target market date is not a trading session")
    if earliest_latest_date > target_market_date:
        raise ValueError("historical artifacts are newer than target")
    if not calendars.is_session(earliest_latest_date):
        raise ValueError("historical artifact as_of is not a trading session")
    if earliest_latest_date == target_market_date:
        return (target_market_date,)
    dates = []
    value = calendars.next_session(earliest_latest_date)
    while value <= target_market_date:
        dates.append(value)
        if len(dates) > MAX_CATCHUP_SESSIONS:
            raise ValueError("official catch-up exceeds the bounded session limit")
        if value == target_market_date:
            break
        value = calendars.next_session(value)
    if not dates or dates[-1] != target_market_date:
        raise ValueError("calendar cannot reach the target market date")
    return tuple(dates)


@contextlib.contextmanager
def _patched_pipeline(
    local_quant: Any,
    pipeline: Any,
    fetcher: OfficialCompatFetcher,
    series: OfficialSnapshotSeries,
    audit: Any,
) -> Iterator[None]:
    original_fetch = pipeline.fetch_finmind_dataset
    original_loader = local_quant.load_stock_pipeline
    original_batch = local_quant.run_market_batch
    original_build = local_quant.build_stock_snapshot

    def run_market_batch_with_source(
        root,
        market,
        symbols,
        analyze_symbol,
        *args,
        batch_identity=None,
        **kwargs,
    ):
        if market == "TW":
            identity = dict(batch_identity or {})
            identity.update(
                {
                    "source_mode": series.source_mode,
                    "source_schema_version": series.source_schema_version,
                    "official_series_manifest_sha256": series.manifest_sha256,
                    "official_snapshot_dates": [
                        value.isoformat() for value in series.dates
                    ],
                    "universe_sha256": _universe_sha256(
                        [str(item) for item in symbols]
                    ),
                    "historical_latest_date_counts": dict(
                        audit.latest_date_counts
                    ),
                    "historical_unavailable_count": len(
                        audit.unavailable_symbols
                    ),
                    "official_request_budget": {
                        "planned_minimum_requests": (
                            series.request_budget.planned_minimum_requests
                        ),
                        "planned_worst_case_requests": (
                            series.request_budget.planned_worst_case_requests
                        ),
                        "actual_request_count": series.request_count,
                    },
                }
            )
            batch_identity = identity
        return original_batch(
            root,
            market,
            symbols,
            analyze_symbol,
            *args,
            batch_identity=batch_identity,
            **kwargs,
        )

    def build_stock_snapshot_with_lineage(
        pipeline_arg, market, symbol, *args, **kwargs
    ):
        result = original_build(
            pipeline_arg, market, symbol, *args, **kwargs
        )
        if market == "TW":
            result = dict(result)
            result["source_lineage"] = fetcher.lineage_for(str(symbol))
        return result

    pipeline.fetch_finmind_dataset = fetcher
    local_quant.load_stock_pipeline = lambda _root: pipeline
    local_quant.run_market_batch = run_market_batch_with_source
    local_quant.build_stock_snapshot = build_stock_snapshot_with_lineage
    try:
        yield
    finally:
        pipeline.fetch_finmind_dataset = original_fetch
        local_quant.load_stock_pipeline = original_loader
        local_quant.run_market_batch = original_batch
        local_quant.build_stock_snapshot = original_build


def run(
    *,
    root: Path,
    target_market_date: datetime.date,
    calendar_artifacts: list[Path],
    limit: int,
    delay: float,
    series_builder=build_official_snapshot_series,
) -> int:
    import local_quant

    pipeline = local_quant.load_stock_pipeline(root)
    symbols = local_quant.get_taiwan_symbols(pipeline)
    if not symbols:
        raise RuntimeError("TW universe is empty")

    calendars = _load_calendar_set(calendar_artifacts)
    audit = audit_artifact_dates(
        root,
        symbols,
        target_date=target_market_date,
    )
    unavailable_ratio = len(audit.unavailable_symbols) / len(symbols)
    if audit.earliest_latest_date is None or unavailable_ratio >= 0.05:
        raise RuntimeError("historical artifact coverage is not publishable")
    for value in set(audit.latest_by_symbol.values()):
        if not calendars.is_session(value):
            raise RuntimeError("historical artifact date is not a trading session")

    trading_dates = _required_trading_dates(
        calendars,
        earliest_latest_date=audit.earliest_latest_date,
        target_market_date=target_market_date,
    )
    series = series_builder(root, trading_dates)
    if series.target_date != target_market_date:
        raise RuntimeError("official snapshot series target date mismatch")
    if not series.request_budget.capacity_proven:
        raise RuntimeError("official request capacity is not proven")

    universe = set(symbols)
    for value, snapshot in series.snapshots.items():
        missing_price = universe - set(snapshot.price_by_symbol)
        if len(missing_price) / len(universe) >= 0.05:
            raise RuntimeError(
                f"official price coverage is not publishable for {value}"
            )

    fetcher = OfficialCompatFetcher(root, series, pd=pipeline.pd)
    argv = [
        "--root",
        str(root),
        "--post-close",
        "--observation-only",
        "--market",
        "TW",
        "--target-market-date",
        target_market_date.isoformat(),
        "--limit",
        str(limit),
        "--delay",
        str(delay),
    ]
    with _patched_pipeline(local_quant, pipeline, fetcher, series, audit):
        return int(local_quant.main(argv))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ABSORB TW official-source post-close runner"
    )
    parser.add_argument("--root", default=r"D:\AbsorbData")
    parser.add_argument(
        "--target-market-date",
        required=True,
        type=datetime.date.fromisoformat,
    )
    parser.add_argument(
        "--calendar-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.limit < 1 or args.delay < 0:
            raise ValueError("limit and delay are invalid")
        return run(
            root=Path(args.root),
            target_market_date=args.target_market_date,
            calendar_artifacts=args.calendar_artifact,
            limit=args.limit,
            delay=args.delay,
        )
    except (
        OfficialSourceFailure,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"TW official post-close refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
