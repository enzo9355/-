"""Run the existing TW observation batch with a prefetched official bulk source."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceFailure,
    build_official_daily_snapshot,
)
from stock_papi.quant.tw_incremental import OfficialCompatFetcher


def _universe_sha256(symbols: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(symbols, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@contextlib.contextmanager
def _patched_pipeline(local_quant: Any, pipeline: Any, fetcher: OfficialCompatFetcher, snapshot: Any) -> Iterator[None]:
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
            identity.update({
                "source_mode": snapshot.source_mode,
                "source_schema_version": snapshot.source_schema_version,
                "official_manifest_sha256": snapshot.manifest_sha256,
                "universe_sha256": _universe_sha256([str(item) for item in symbols]),
                "official_request_budget": {
                    "planned_minimum_requests": snapshot.request_budget.planned_minimum_requests,
                    "planned_worst_case_requests": snapshot.request_budget.planned_worst_case_requests,
                    "actual_request_count": snapshot.request_count,
                },
            })
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

    def build_stock_snapshot_with_lineage(pipeline_arg, market, symbol, *args, **kwargs):
        result = original_build(pipeline_arg, market, symbol, *args, **kwargs)
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
    limit: int,
    delay: float,
    snapshot_builder=build_official_daily_snapshot,
) -> int:
    import local_quant

    pipeline = local_quant.load_stock_pipeline(root)
    symbols = local_quant.get_taiwan_symbols(pipeline)
    if not symbols:
        raise RuntimeError("TW universe is empty")

    snapshot = snapshot_builder(root, target_market_date)
    if snapshot.target_date != target_market_date:
        raise RuntimeError("official snapshot target date mismatch")
    if not snapshot.request_budget.capacity_proven:
        raise RuntimeError("official request capacity is not proven")

    universe = set(symbols)
    price_symbols = set(snapshot.price_by_symbol)
    missing_price = universe - price_symbols
    if len(missing_price) / len(universe) >= 0.05:
        raise RuntimeError("official price coverage is not publishable")

    fetcher = OfficialCompatFetcher(root, snapshot, pd=pipeline.pd)
    argv = [
        "--root", str(root),
        "--post-close",
        "--observation-only",
        "--market", "TW",
        "--target-market-date", target_market_date.isoformat(),
        "--limit", str(limit),
        "--delay", str(delay),
    ]
    with _patched_pipeline(local_quant, pipeline, fetcher, snapshot):
        return int(local_quant.main(argv))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ABSORB TW official-source post-close runner")
    parser.add_argument("--root", default=r"D:\AbsorbData")
    parser.add_argument("--target-market-date", required=True, type=datetime.date.fromisoformat)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.limit < 1 or args.delay < 0:
            raise ValueError("limit and delay are invalid")
        return run(
            root=Path(args.root),
            target_market_date=args.target_market_date,
            limit=args.limit,
            delay=args.delay,
        )
    except (OfficialSourceFailure, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TW official post-close refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
