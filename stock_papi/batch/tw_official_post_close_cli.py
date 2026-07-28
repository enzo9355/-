"""Run the TW observation batch with date-addressable official bulk snapshots."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

from stock_papi.batch.calendar import TradingCalendarSet
from stock_papi.integrations.market_data.tw_official_bulk import OfficialSourceFailure
from stock_papi.integrations.market_data.tw_official_historical import (
    MAX_CATCHUP_SESSIONS,
    OfficialSnapshotSeries,
    build_official_snapshot_series,
)
from stock_papi.quant.tw_artifact_audit import audit_artifact_dates
from stock_papi.quant.tw_incremental import (
    IncrementalHistoryError,
    OfficialCompatFetcher,
    load_incremental_artifact,
)
from stock_papi.quant.tw_legacy_reconciliation import LegacyArtifactBackupStore


_INCOMPLETE = "TW official observation recovery is incomplete"
_SYMBOL_RE = re.compile(r"[0-9]{4,6}")
_EXCLUSION_FIELDS = [
    "Symbol",
    "Name",
    "ExclusionDate",
    "ConsecutiveFailures",
    "State",
    "Type",
    "Reason",
    "OperatorAction",
]


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


def _reconciliation_trading_dates(
    calendars: TradingCalendarSet,
    *,
    baseline_date: datetime.date,
    target_market_date: datetime.date,
) -> tuple[datetime.date, ...]:
    dates = _required_trading_dates(
        calendars,
        earliest_latest_date=baseline_date,
        target_market_date=target_market_date,
    )
    if dates[0] != baseline_date:
        dates = (baseline_date, *dates)
    if len(dates) > MAX_CATCHUP_SESSIONS:
        raise ValueError("official catch-up exceeds the bounded session limit")
    return dates


def _enrich_batch_identity(
    identity: dict[str, Any],
    *,
    series: OfficialSnapshotSeries,
    audit: Any,
    symbols: list[str],
    reconcile_legacy_overlaps: bool,
) -> dict[str, Any]:
    result = dict(identity)
    result.update(
        {
            "source_mode": series.source_mode,
            "source_schema_version": series.source_schema_version,
            "official_series_manifest_sha256": series.manifest_sha256,
            "official_snapshot_dates": [value.isoformat() for value in series.dates],
            "universe_sha256": _universe_sha256(symbols),
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
    if reconcile_legacy_overlaps:
        result["legacy_overlap_policy"] = "replace_verified_legacy"
    else:
        result.update(
            {
                "historical_latest_date_counts": dict(audit.latest_date_counts),
                "historical_unavailable_count": len(audit.unavailable_symbols),
            }
        )
    return result


def _load_exclusion_state(root: Path) -> tuple[set[str], set[str]]:
    path = Path(root) / "checkpoints" / "exclusion_list-TW.csv"
    if not path.exists():
        return set(), set()
    pending: set[str] = set()
    excluded: set[str] = set()
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != _EXCLUSION_FIELDS:
                raise ValueError("invalid exclusion headers")
            for row in reader:
                if set(row) != set(_EXCLUSION_FIELDS) or any(
                    not isinstance(row[field], str) for field in _EXCLUSION_FIELDS
                ):
                    raise ValueError("invalid exclusion row")
                symbol = row["Symbol"].strip()
                state = row["State"].strip()
                if (
                    _SYMBOL_RE.fullmatch(symbol) is None
                    or symbol in seen
                    or row["OperatorAction"].strip()
                    or state not in {"", "Pending", "Excluded"}
                ):
                    raise ValueError("invalid exclusion row")
                seen.add(symbol)
                (excluded if state == "Excluded" else pending).add(symbol)
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as exc:
        raise RuntimeError(_INCOMPLETE) from exc
    return pending, excluded


def _load_checkpoint(root: Path) -> dict[str, Any]:
    path = Path(root) / "checkpoints" / "progress.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(_INCOMPLETE) from exc
    if not isinstance(document, dict):
        raise RuntimeError(_INCOMPLETE)
    return document


def _assert_complete(
    root: Path,
    *,
    symbols: list[str],
    target_market_date: datetime.date,
    expected_identity: dict[str, Any],
    official_series: OfficialSnapshotSeries | None = None,
    applied_reconciliation_artifacts: Mapping[str, str] | None = None,
) -> None:
    applied_reconciliation_artifacts = applied_reconciliation_artifacts or {}
    pending, excluded = _load_exclusion_state(root)
    universe = set(symbols)
    checkpoint = _load_checkpoint(root)
    failures = checkpoint.get("failed")
    next_index = checkpoint.get("next_index")
    if (
        checkpoint.get("stage") != "market_batch"
        or checkpoint.get("market") != "TW"
        or not isinstance(next_index, int)
        or isinstance(next_index, bool)
        or next_index < len(symbols)
        or checkpoint.get("batch_identity") != expected_identity
        or not isinstance(failures, list)
    ):
        raise RuntimeError(_INCOMPLETE)
    failed_symbols = set()
    for item in failures:
        if (
            not isinstance(item, dict)
            or set(item) != {"symbol", "error"}
            or not isinstance(item.get("symbol"), str)
            or item["symbol"] not in universe
            or not isinstance(item.get("error"), str)
        ):
            raise RuntimeError(_INCOMPLETE)
        failed_symbols.add(item["symbol"])
    active = universe - pending - excluded
    if failed_symbols & active:
        raise RuntimeError(_INCOMPLETE)
    if not active:
        return
    try:
        audit = audit_artifact_dates(
            root,
            sorted(active),
            target_date=target_market_date,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(_INCOMPLETE) from exc
    if (
        audit.unavailable_symbols
        or set(audit.latest_by_symbol) != active
        or any(value != target_market_date for value in audit.latest_by_symbol.values())
    ):
        raise RuntimeError(_INCOMPLETE)
    if official_series is None:
        return
    if not set(applied_reconciliation_artifacts).issubset(active):
        raise RuntimeError(_INCOMPLETE)
    expected_dates = [value.isoformat() for value in official_series.dates]
    expected_manifests = [
        {
            "date": value.isoformat(),
            "manifest_sha256": snapshot.manifest_sha256,
        }
        for value, snapshot in official_series.snapshots.items()
    ]
    for symbol in sorted(active):
        try:
            artifact = load_incremental_artifact(root, symbol)
        except IncrementalHistoryError as exc:
            raise RuntimeError(_INCOMPLETE) from exc
        lineage = artifact.document.get("source_lineage")
        reconciliation = (
            lineage.get("legacy_reconciliation")
            if isinstance(lineage, dict)
            else None
        )
        if (
            not OfficialCompatFetcher._valid_official_lineage(lineage, artifact)
            or lineage.get("source_mode") != official_series.source_mode
            or lineage.get("source_schema_version")
            != official_series.source_schema_version
            or lineage.get("target_market_date")
            != target_market_date.isoformat()
            or lineage.get("official_series_manifest_sha256")
            != official_series.manifest_sha256
            or lineage.get("official_snapshot_dates") != expected_dates
            or lineage.get("official_snapshot_manifests") != expected_manifests
            or lineage.get("official_target_price_available")
            != (
                symbol
                in official_series.snapshots[target_market_date].price_by_symbol
            )
            or (reconciliation is not None)
            != (symbol in applied_reconciliation_artifacts)
            or (
                reconciliation is not None
                and artifact.compressed_sha256
                != applied_reconciliation_artifacts[symbol]
            )
        ):
            raise RuntimeError(_INCOMPLETE)


@contextlib.contextmanager
def _patched_pipeline(
    local_quant: Any,
    pipeline: Any,
    fetcher: OfficialCompatFetcher,
    series: OfficialSnapshotSeries,
    audit: Any,
    *,
    backup_store: LegacyArtifactBackupStore | None = None,
) -> Iterator[None]:
    original_fetch = pipeline.fetch_finmind_dataset
    original_loader = local_quant.load_stock_pipeline
    original_batch = local_quant.run_market_batch
    original_build = local_quant.build_stock_snapshot
    original_writer = getattr(local_quant, "write_stock_artifact", None)

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
            batch_identity = _enrich_batch_identity(
                dict(batch_identity or {}),
                series=series,
                audit=audit,
                symbols=[str(item) for item in symbols],
                reconcile_legacy_overlaps=backup_store is not None,
            )
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

    def write_stock_artifact_with_backup(
        root, market, symbol, payload, *args, **kwargs
    ):
        if market != "TW":
            return original_writer(root, market, symbol, payload, *args, **kwargs)
        lineage = payload.get("source_lineage") if isinstance(payload, dict) else None
        evidence = (
            lineage.get("legacy_reconciliation")
            if isinstance(lineage, dict)
            else None
        )
        artifact_path = (
            Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
        )
        action = backup_store.backup_before_write(
            symbol=str(symbol),
            artifact_path=artifact_path,
            evidence=evidence,
        )
        if action == "noop":
            return artifact_path
        result = original_writer(root, market, symbol, payload, *args, **kwargs)
        if action == "write":
            backup_store.mark_applied(
                symbol=str(symbol), artifact_path=Path(result)
            )
        return result

    try:
        pipeline.fetch_finmind_dataset = fetcher
        local_quant.load_stock_pipeline = lambda _root: pipeline
        local_quant.run_market_batch = run_market_batch_with_source
        local_quant.build_stock_snapshot = build_stock_snapshot_with_lineage
        if backup_store is not None:
            if original_writer is None:
                raise RuntimeError("TW artifact writer is unavailable")
            local_quant.write_stock_artifact = write_stock_artifact_with_backup
        yield
    finally:
        if backup_store is not None and original_writer is not None:
            local_quant.write_stock_artifact = original_writer
        local_quant.build_stock_snapshot = original_build
        local_quant.run_market_batch = original_batch
        local_quant.load_stock_pipeline = original_loader
        pipeline.fetch_finmind_dataset = original_fetch


def run(
    *,
    root: Path,
    target_market_date: datetime.date,
    calendar_artifacts: list[Path],
    limit: int,
    delay: float,
    series_builder=build_official_snapshot_series,
    reconcile_legacy_overlaps: bool = False,
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

    resume = None
    if reconcile_legacy_overlaps:
        resume = LegacyArtifactBackupStore.discover_resume(
            root, target_date=target_market_date
        )
        baseline_date = (
            min(audit.earliest_latest_date, resume[1])
            if resume is not None
            else audit.earliest_latest_date
        )
        trading_dates = _reconciliation_trading_dates(
            calendars,
            baseline_date=baseline_date,
            target_market_date=target_market_date,
        )
    else:
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
    if resume is not None and series.manifest_sha256 != resume[0]:
        raise RuntimeError("official snapshot series does not match resume state")

    universe = set(symbols)
    for value, snapshot in series.snapshots.items():
        missing_price = universe - set(snapshot.price_by_symbol)
        if len(missing_price) / len(universe) >= 0.05:
            raise RuntimeError(
                f"official price coverage is not publishable for {value}"
            )

    policy = (
        "replace_verified_legacy" if reconcile_legacy_overlaps else "strict"
    )
    fetcher = OfficialCompatFetcher(
        root,
        series,
        pd=pipeline.pd,
        legacy_overlap_policy=policy,
    )
    backup_store = (
        LegacyArtifactBackupStore(
            root,
            target_date=target_market_date,
            series_manifest_sha256=series.manifest_sha256,
        )
        if reconcile_legacy_overlaps
        else None
    )
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
    with _patched_pipeline(
        local_quant,
        pipeline,
        fetcher,
        series,
        audit,
        backup_store=backup_store,
    ):
        result = int(local_quant.main(argv))
    if result != 0:
        return result
    applied_reconciliation_artifacts = {}
    if backup_store is not None:
        applied_reconciliation_artifacts = (
            backup_store.assert_current_state_complete() or {}
        )
    expected_identity = _enrich_batch_identity(
        {
            "target_market_date": target_market_date.isoformat(),
            "product_mode": "observation",
            "source_version": local_quant.OBSERVATION_SOURCE_VERSION,
        },
        series=series,
        audit=audit,
        symbols=[str(value) for value in symbols],
        reconcile_legacy_overlaps=reconcile_legacy_overlaps,
    )
    _assert_complete(
        root,
        symbols=[str(value) for value in symbols],
        target_market_date=target_market_date,
        expected_identity=expected_identity,
        official_series=series,
        applied_reconciliation_artifacts=applied_reconciliation_artifacts,
    )
    return 0


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
    parser.add_argument("--reconcile-legacy-overlaps", action="store_true")
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
            reconcile_legacy_overlaps=args.reconcile_legacy_overlaps,
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
