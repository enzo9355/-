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
from stock_papi.quant.tw_legacy_reconciliation import (
    LegacyArtifactBackupStore,
    resolve_truncated_daily_history,
)


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


def _load_recovery_symbol_allowlist(
    path: Path,
    *,
    expected_sha256: str | None,
) -> set[str]:
    if not isinstance(path, Path):
        raise TypeError("recovery symbol allowlist path is invalid")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("recovery symbol allowlist is unreadable") from exc
    symbols: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _SYMBOL_RE.fullmatch(stripped) is None:
            raise ValueError("recovery symbol allowlist is invalid")
        symbols.append(stripped)
    canonical = sorted(symbols)
    if len(canonical) != len(set(canonical)):
        raise ValueError("recovery symbol allowlist is invalid")
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("recovery symbol allowlist identity is invalid")
        if _universe_sha256(canonical) != expected_sha256:
            raise ValueError("recovery symbol allowlist identity does not match")
    return set(canonical)


def _required_symbols_by_exchange(
    symbols: list[str], *, registry: Mapping[str, Any] | None = None
) -> dict[str, set[str]]:
    if registry is None:
        import twstock

        registry = twstock.codes
    result = {"TWSE": set(), "TPEx": set()}
    source_to_exchange = {"twse": "TWSE", "tpex": "TPEx"}
    for symbol in symbols:
        info = registry.get(str(symbol))
        exchange = source_to_exchange.get(
            str(getattr(info, "data_source", "")).lower()
        )
        if exchange is None:
            raise RuntimeError(f"TW exchange metadata is unavailable for {symbol}")
        result[exchange].add(str(symbol))
    return result


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


def _assert_audit_publishable(
    audit: Any,
    *,
    symbols: list[str],
    target_market_date: datetime.date,
    calendars: TradingCalendarSet,
    active_universe_count: int | None = None,
) -> None:
    available = set(audit.latest_by_symbol)
    denominator = (
        active_universe_count
        if active_universe_count is not None and active_universe_count > 0
        else len(symbols)
    )
    if (
        not available
        or available != set(audit.observation_by_symbol)
        or len(audit.unavailable_symbols) / denominator >= 0.05
    ):
        raise RuntimeError("historical artifact coverage is not publishable")
    for symbol in available:
        latest = audit.latest_by_symbol[symbol]
        observation = audit.observation_by_symbol[symbol]
        if (
            latest > observation
            or observation > target_market_date
            or not calendars.is_session(latest)
            or not calendars.is_session(observation)
        ):
            raise RuntimeError("historical artifact date is not a trading session")


def _plan_recovery_stage(
    calendars: TradingCalendarSet,
    audit: Any,
    *,
    symbols: list[str],
    target_market_date: datetime.date,
    reconcile_legacy_overlaps: bool,
    ignored_symbols: set[str] | frozenset[str] = frozenset(),
    excluded_symbols: set[str] | frozenset[str] = frozenset(),
) -> tuple[datetime.date, list[str], datetime.date]:
    observations = audit.observation_by_symbol
    skip_symbols = set(ignored_symbols) | set(excluded_symbols)
    baseline = min(
        (
            observations[symbol]
            for symbol in symbols
            if symbol not in skip_symbols and symbol in observations
        ),
        default=target_market_date,
    )
    if baseline > target_market_date:
        raise ValueError("historical artifacts are newer than target")
    stage_target = baseline
    capacity = MAX_CATCHUP_SESSIONS - int(reconcile_legacy_overlaps)
    for _ in range(capacity):
        if stage_target == target_market_date:
            break
        stage_target = calendars.next_session(stage_target)
    if stage_target >= target_market_date:
        return target_market_date, list(symbols), baseline
    stage_symbols = [
        symbol
        for symbol in symbols
        if symbol not in skip_symbols
        and symbol in observations
        and observations[symbol] < stage_target
    ]
    if not stage_symbols:
        raise RuntimeError(_INCOMPLETE)
    return stage_target, stage_symbols, baseline


def _enrich_batch_identity(
    identity: dict[str, Any],
    *,
    series: OfficialSnapshotSeries,
    audit: Any,
    symbols: list[str],
    reconcile_legacy_overlaps: bool,
    recover_truncated_history: bool = False,
) -> dict[str, Any]:
    result = dict(identity)
    result["recover_truncated_history"] = bool(recover_truncated_history)
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
    target_snapshot = (
        official_series.snapshots[target_market_date]
        if official_series is not None
        else None
    )
    regular_symbols = (
        set(target_snapshot.price_by_symbol) & universe
        if target_snapshot is not None
        else set()
    )
    status_symbols = (
        set(target_snapshot.trading_status_by_symbol) & universe
        if target_snapshot is not None
        else set()
    )
    terminated_symbols = (
        set(target_snapshot.terminated_by_symbol) & universe
        if target_snapshot is not None
        else set()
    )
    if (
        not (regular_symbols | status_symbols | terminated_symbols).issubset(universe)
        or regular_symbols & status_symbols
        or regular_symbols & terminated_symbols
        or status_symbols & terminated_symbols
    ):
        raise RuntimeError(_INCOMPLETE)
    active = (
        universe
        - (pending - status_symbols)
        - (excluded - status_symbols)
        - terminated_symbols
    )
    if not active:
        return
    R = regular_symbols & active
    N = status_symbols & active
    M = active - (R | N)
    if R & N or R & M or N & M or (R | N | M) != active:
        raise RuntimeError(_INCOMPLETE)
    observation_coverage = len(R | N) / len(active)
    if observation_coverage <= 0.95:
        raise RuntimeError(_INCOMPLETE)
    observed = sorted(R | N)
    if failed_symbols & set(observed):
        raise RuntimeError(_INCOMPLETE)
    try:
        audit = audit_artifact_dates(
            root,
            observed,
            target_date=target_market_date,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(_INCOMPLETE) from exc
    if (
        audit.unavailable_symbols
        or set(audit.latest_by_symbol) != set(observed)
        or set(audit.observation_by_symbol) != set(observed)
        or any(
            value != target_market_date
            for value in audit.observation_by_symbol.values()
        )
        or any(
            audit.latest_by_symbol[symbol] != target_market_date
            for symbol in R
        )
        or any(
            audit.latest_by_symbol[symbol] >= target_market_date
            for symbol in N
        )
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
    for symbol in observed:
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
            or (
                symbol in status_symbols
                and (
                    artifact.trading_status_evidence
                    != dict(target_snapshot.trading_status_by_symbol[symbol])
                    or lineage.get("trading_status_evidence_sha256")
                    != target_snapshot.trading_status_by_symbol[symbol].get(
                        "evidence_sha256"
                    )
                )
            )
            or (
                symbol in regular_symbols
                and artifact.trading_status_evidence is not None
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
    symbols: list[str] | None = None,
    recover_truncated_history: bool = False,
    reconcile_legacy_overlaps: bool = False,
    recovery_rotated_symbols: set[str] | None = None,
    applied_reconciliation_artifacts: dict[str, str] | None = None,
) -> Iterator[None]:
    if type(recover_truncated_history) is not bool:
        raise TypeError("recover_truncated_history must be bool")
    if type(reconcile_legacy_overlaps) is not bool:
        raise TypeError("reconcile_legacy_overlaps must be bool")
    recovery_rotated_symbols = (
        recovery_rotated_symbols if recovery_rotated_symbols is not None else set()
    )
    applied_reconciliation_artifacts = (
        applied_reconciliation_artifacts
        if applied_reconciliation_artifacts is not None
        else {}
    )
    original_fetch = pipeline.fetch_finmind_dataset
    original_loader = local_quant.load_stock_pipeline
    original_batch = local_quant.run_market_batch
    original_build = local_quant.build_stock_snapshot
    original_writer = getattr(local_quant, "write_stock_artifact", None)
    original_exclusion_loader = getattr(
        local_quant, "load_exclusion_list", None
    )
    original_symbols_loader = getattr(local_quant, "get_taiwan_symbols", None)
    target_status_symbols = set(
        series.snapshots[series.target_date].trading_status_by_symbol
    )

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
                reconcile_legacy_overlaps=reconcile_legacy_overlaps,
                recover_truncated_history=recover_truncated_history,
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
        if market == "TW":
            status = fetcher.status_for(str(symbol))
            if status is not None:
                kwargs["trading_status"] = status
        result = original_build(
            pipeline_arg, market, symbol, *args, **kwargs
        )
        if market == "TW":
            result = dict(result)
            result["source_lineage"] = fetcher.lineage_for(
                str(symbol), persisted_daily=result.get("daily")
            )
        return result

    def load_exclusion_list_with_official_status(root, market):
        result = original_exclusion_loader(root, market)
        if market != "TW":
            return result
        pending, excluded, rows, invalid_actions = result
        return (
            set(pending) - target_status_symbols,
            set(excluded) - target_status_symbols,
            rows,
            invalid_actions,
        )

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
        if (
            recover_truncated_history
            and
            isinstance(lineage, dict)
            and lineage.get("legacy_reconciliation_history")
        ):
            recovery_rotated_symbols.add(str(symbol))
            return original_writer(root, market, symbol, payload, *args, **kwargs)
        artifact_path = (
            Path(root) / "artifacts" / "stocks" / "TW" / f"{symbol}.json.gz"
        )
        action = backup_store.backup_before_write(
            symbol=str(symbol),
            artifact_path=artifact_path,
            evidence=evidence,
        )
        if action == "noop":
            if evidence is not None and artifact_path.is_file():
                applied_reconciliation_artifacts[str(symbol)] = hashlib.sha256(
                    artifact_path.read_bytes()
                ).hexdigest()
            return artifact_path
        result = original_writer(root, market, symbol, payload, *args, **kwargs)
        if action == "write":
            backup_store.mark_applied(
                symbol=str(symbol), artifact_path=Path(result)
            )
            result_path = Path(result)
            if result_path.is_file():
                applied_reconciliation_artifacts[str(symbol)] = hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest()
        return result

    try:
        if symbols is not None:
            if original_symbols_loader is None:
                raise RuntimeError("TW universe loader is unavailable")
            local_quant.get_taiwan_symbols = lambda _pipeline: list(symbols)
        pipeline.fetch_finmind_dataset = fetcher
        local_quant.load_stock_pipeline = lambda _root: pipeline
        local_quant.run_market_batch = run_market_batch_with_source
        local_quant.build_stock_snapshot = build_stock_snapshot_with_lineage
        if original_exclusion_loader is not None:
            local_quant.load_exclusion_list = (
                load_exclusion_list_with_official_status
            )
        if backup_store is not None:
            if original_writer is None:
                raise RuntimeError("TW artifact writer is unavailable")
            local_quant.write_stock_artifact = write_stock_artifact_with_backup
        yield
    finally:
        if symbols is not None and original_symbols_loader is not None:
            local_quant.get_taiwan_symbols = original_symbols_loader
        if backup_store is not None and original_writer is not None:
            local_quant.write_stock_artifact = original_writer
        local_quant.build_stock_snapshot = original_build
        local_quant.run_market_batch = original_batch
        if original_exclusion_loader is not None:
            local_quant.load_exclusion_list = original_exclusion_loader
        local_quant.load_stock_pipeline = original_loader
        pipeline.fetch_finmind_dataset = original_fetch


def _run_stage(
    *,
    local_quant: Any,
    pipeline: Any,
    root: Path,
    target_market_date: datetime.date,
    calendars: TradingCalendarSet,
    symbols: list[str],
    baseline_date: datetime.date,
    limit: int,
    delay: float,
    series_builder: Any,
    reconcile_legacy_overlaps: bool,
    recover_truncated_history: bool,
    publish: bool,
    recovery_symbol_allowlist: set[str] | None = None,
    full_market_symbols: list[str] | None = None,
) -> tuple[int, set[str]]:
    if type(recover_truncated_history) is not bool:
        raise TypeError("recover_truncated_history must be bool")
    audit = audit_artifact_dates(
        root,
        symbols,
        target_date=target_market_date,
    )
    _assert_audit_publishable(
        audit,
        symbols=symbols,
        target_market_date=target_market_date,
        calendars=calendars,
        active_universe_count=len(full_market_symbols) if full_market_symbols else len(symbols),
    )

    resume = None
    if reconcile_legacy_overlaps:
        if recover_truncated_history:
            historical = []
            for symbol in symbols:
                artifact = load_incremental_artifact(root, str(symbol))
                lineage = artifact.document.get("source_lineage")
                if not OfficialCompatFetcher._valid_official_lineage(lineage, artifact):
                    continue
                for item in lineage.get("legacy_reconciliation_history", []):
                    reconciliation = item["reconciliation"]
                    historical.append(
                        (
                            reconciliation["official_series_manifest_sha256"],
                            min(
                                datetime.date.fromisoformat(value)
                                for value in reconciliation["overlap_dates"]
                            ),
                        )
                    )
            manifests = {manifest for manifest, _baseline in historical}
            if len(manifests) > 1:
                raise RuntimeError("recovery reconciliation series is ambiguous")
            if manifests:
                resume = (manifests.pop(), min(baseline for _manifest, baseline in historical))
        if resume is None:
            resume = LegacyArtifactBackupStore.discover_resume(
                root, target_date=target_market_date
            )
        baseline_date = (
            min(baseline_date, resume[1])
            if resume is not None
            else baseline_date
        )
        trading_dates = _reconciliation_trading_dates(
            calendars,
            baseline_date=baseline_date,
            target_market_date=target_market_date,
        )
    else:
        trading_dates = _required_trading_dates(
            calendars,
            earliest_latest_date=baseline_date,
            target_market_date=target_market_date,
        )
    required_symbols = _required_symbols_by_exchange(
        [str(symbol) for symbol in symbols]
    )
    series = series_builder(
        root,
        trading_dates,
        required_symbols_by_exchange=required_symbols,
    )
    if series.target_date != target_market_date:
        raise RuntimeError("official snapshot series target date mismatch")
    if not series.request_budget.capacity_proven:
        raise RuntimeError("official request capacity is not proven")
    if resume is not None and series.manifest_sha256 != resume[0]:
        raise RuntimeError("official snapshot series does not match resume state")

    market_symbols = full_market_symbols if full_market_symbols is not None else symbols
    market_universe = {str(value) for value in market_symbols}
    for value, snapshot in series.snapshots.items():
        covered = set(snapshot.price_by_symbol)
        if value == target_market_date:
            covered |= set(snapshot.trading_status_by_symbol)
            covered |= set(snapshot.terminated_by_symbol)
        missing_price = market_universe - covered
        if value == target_market_date:
            observation_coverage = len(covered & market_universe) / len(market_universe)
            if observation_coverage <= 0.95:
                raise RuntimeError(
                    f"official price coverage is not publishable for {value}"
                )
        elif len(missing_price) / len(market_universe) >= 0.05:
            raise RuntimeError(
                f"official price coverage is not publishable for {value}"
            )

    policy = (
        "replace_verified_legacy" if reconcile_legacy_overlaps else "strict"
    )
    recovery_resolver = None
    if recover_truncated_history:
        allowlist = (
            None
            if recovery_symbol_allowlist is None
            else frozenset(recovery_symbol_allowlist)
        )

        def _recovery_resolver(symbol, artifact, _root=root, _allowlist=allowlist):
            if _allowlist is not None and symbol not in _allowlist:
                return None
            return resolve_truncated_daily_history(_root, symbol, artifact)

        recovery_resolver = _recovery_resolver
    fetcher = OfficialCompatFetcher(
        root,
        series,
        pd=pipeline.pd,
        legacy_overlap_policy=policy,
        recovery_resolver=recovery_resolver,
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
    applied_reconciliation_artifacts = {}
    recovery_rotated_symbols: set[str] = set()
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
    try:
        with _patched_pipeline(
            local_quant,
            pipeline,
            fetcher,
            series,
            audit,
            backup_store=backup_store,
            symbols=symbols,
            recover_truncated_history=recover_truncated_history,
            reconcile_legacy_overlaps=reconcile_legacy_overlaps,
            recovery_rotated_symbols=recovery_rotated_symbols,
            applied_reconciliation_artifacts=applied_reconciliation_artifacts,
        ):
            result = int(local_quant.main(argv))
        if result != 0:
            return result, set()
    except (OfficialSourceFailure, RuntimeError):
        if backup_store is not None:
            backup_store.restore_all()
        raise
    if reconcile_legacy_overlaps and backup_store is not None:
        if recovery_rotated_symbols:
            applied_reconciliation_artifacts = dict(
                applied_reconciliation_artifacts
            )
        else:
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
        recover_truncated_history=recover_truncated_history,
    )
    _assert_complete(
        root,
        symbols=[str(value) for value in symbols],
        target_market_date=target_market_date,
        expected_identity=expected_identity,
        official_series=series,
        applied_reconciliation_artifacts=applied_reconciliation_artifacts,
    )
    pending, excluded = _load_exclusion_state(root)
    universe = {str(value) for value in symbols}
    target_snapshot = series.snapshots[target_market_date]
    status_symbols = set(target_snapshot.trading_status_by_symbol) & universe
    terminated_symbols = set(target_snapshot.terminated_by_symbol) & universe
    active_universe = (
        universe
        - (set(pending) - status_symbols)
        - (set(excluded) - status_symbols)
        - terminated_symbols
    )
    regular_symbols = set(target_snapshot.price_by_symbol) & active_universe
    observed_symbols = regular_symbols | (status_symbols & active_universe)
    unavailable_symbols = active_universe - observed_symbols
    operational_failures = (
        ((set(pending) | set(excluded) | terminated_symbols) & universe) - status_symbols
    ) | unavailable_symbols
    if publish:
        local_quant.publish_market_snapshot(
            root,
            "TW",
            sorted(active_universe),
            failed_symbols=sorted(unavailable_symbols),
            target_market_date=target_market_date,
        )
    return 0, operational_failures


def run(
    *,
    root: Path,
    target_market_date: datetime.date,
    calendar_artifacts: list[Path],
    limit: int,
    delay: float,
    series_builder=build_official_snapshot_series,
    reconcile_legacy_overlaps: bool = False,
    recover_truncated_history: bool = False,
    recovery_symbol_allowlist: set[str] | None = None,
) -> int:
    if type(recover_truncated_history) is not bool:
        raise TypeError("recover_truncated_history must be bool")
    import local_quant

    pipeline = local_quant.load_stock_pipeline(root)
    symbols = [str(value) for value in local_quant.get_taiwan_symbols(pipeline)]
    if not symbols:
        raise RuntimeError("TW universe is empty")
    calendars = _load_calendar_set(calendar_artifacts)
    if not calendars.is_session(target_market_date):
        raise ValueError("target market date is not a trading session")

    pending_exclusions, excluded_symbols = _load_exclusion_state(root)
    ignored_symbols: set[str] = set(excluded_symbols)
    audit = audit_artifact_dates(root, symbols, target_date=target_market_date)
    while True:
        _assert_audit_publishable(
            audit,
            symbols=symbols,
            target_market_date=target_market_date,
            calendars=calendars,
            active_universe_count=len(set(symbols) - set(excluded_symbols)),
        )
        stage_target, stage_symbols, baseline = _plan_recovery_stage(
            calendars,
            audit,
            symbols=symbols,
            target_market_date=target_market_date,
            reconcile_legacy_overlaps=reconcile_legacy_overlaps,
            ignored_symbols=ignored_symbols,
            excluded_symbols=excluded_symbols,
        )
        result, inactive = _run_stage(
            local_quant=local_quant,
            pipeline=pipeline,
            root=root,
            target_market_date=stage_target,
            calendars=calendars,
            symbols=stage_symbols,
            baseline_date=baseline,
            limit=limit,
            delay=delay,
            series_builder=series_builder,
            reconcile_legacy_overlaps=reconcile_legacy_overlaps,
            recover_truncated_history=recover_truncated_history,
            publish=stage_target == target_market_date,
            recovery_symbol_allowlist=recovery_symbol_allowlist,
            full_market_symbols=symbols,
        )
        if result != 0 or stage_target == target_market_date:
            return result
        ignored_symbols.update(inactive & set(stage_symbols))
        audit = audit_artifact_dates(root, symbols, target_date=target_market_date)
        _, _, next_baseline = _plan_recovery_stage(
            calendars,
            audit,
            symbols=symbols,
            target_market_date=target_market_date,
            reconcile_legacy_overlaps=reconcile_legacy_overlaps,
            ignored_symbols=ignored_symbols,
            excluded_symbols=excluded_symbols,
        )
        if next_baseline <= baseline:
            raise RuntimeError(_INCOMPLETE)


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
    parser.add_argument("--recover-truncated-history", action="store_true")
    parser.add_argument(
        "--recovery-symbol-allowlist",
        type=Path,
        default=None,
        help=(
            "Path to a text file (one symbol per line) listing the exact "
            "symbols eligible for truncated-history recovery; gates the "
            "fallback resolver so unrelated symbols are never modified."
        ),
    )
    parser.add_argument(
        "--recovery-allowlist-sha256",
        default=None,
        help=(
            "Expected SHA-256 of the canonical sorted recovery symbol "
            "allowlist; required when --recovery-symbol-allowlist is set."
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.limit < 1 or args.delay < 0:
            raise ValueError("limit and delay are invalid")
        recovery_symbol_allowlist: set[str] | None = None
        if args.recovery_symbol_allowlist is not None:
            if args.recovery_allowlist_sha256 is None:
                raise ValueError(
                    "recovery symbol allowlist identity is required"
                )
            recovery_symbol_allowlist = _load_recovery_symbol_allowlist(
                args.recovery_symbol_allowlist,
                expected_sha256=args.recovery_allowlist_sha256,
            )
        return run(
            root=Path(args.root),
            target_market_date=args.target_market_date,
            calendar_artifacts=args.calendar_artifact,
            limit=args.limit,
            delay=args.delay,
            reconcile_legacy_overlaps=args.reconcile_legacy_overlaps,
            recover_truncated_history=args.recover_truncated_history,
            recovery_symbol_allowlist=recovery_symbol_allowlist,
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
