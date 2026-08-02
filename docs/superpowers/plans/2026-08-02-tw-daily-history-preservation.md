# TW Daily History Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the complete canonical TW `daily` history across repeated official post-close updates while keeping calculation separate and allowing only explicit, manifest-bound recovery of already-truncated reconciled artifacts.

**Architecture:** `build_stock_snapshot()` keeps a canonical frame and a calculated copy, lets model/backtest inference finish mutating the calculated copy, and only then joins approved derived fields by market date for persistence. An optional `OfficialCompatFetcher` resolver validates and caches one lineage-authorized merged history per symbol; receipt fields are finalized later from the canonical rows that survive the normal persistence window. The CLI constructs this resolver only for `--recover-truncated-history`, records that mode in checkpoint identity, and leaves the normal runtime independent of quarantine.

**Tech Stack:** Existing Python runtime, pandas, NumPy, stdlib `unittest`, `ast`, JSON, gzip, SHA-256, `pathlib`, PowerShell parser, Node syntax check, Git, GitHub CLI, and the existing read-only `agy` reviewer; no new dependency.

## Global Constraints

- Approved design commit: `0d2293d6fa8fb61a740a949f8ad084c24a266a2c`, based on `main` commit `336c25acc1903b0d76503f3cc4589e8a8de950b7`, on `codex/tw-daily-history-preservation` in the existing isolated worktree.
- `daily` remains the single canonical persisted history. Do not add `canonical_daily`, `source_daily`, or another daily-history field.
- Preserve every canonical row returned inside the existing `get_data(symbol, 730)` request window, including indicator warm-up rows. Do not introduce unlimited retention.
- The canonical frame is the target-date-filtered result before calculation. Normalize dates to market dates, reject duplicate dates, and sort strictly ascending.
- Same-date canonical `Open`, `High`, `Low`, `Close`, and `Volume` are immutable across reruns. Conflicting duplicate-date OHLCV fails closed.
- `rows == len(daily)`, `latest == daily[-1]`, and `as_of` remains the latest regular-price date.
- For a regular-price session, `as_of == latest_regular_price_date == target_market_date == observation_as_of`.
- For `official_no_regular_trade` and `officially_suspended`, `target_market_date == observation_as_of` and `as_of == latest_regular_price_date < target_market_date`. A status session never creates or relabels a target-date price row.
- Preserve status evidence, evidence SHA-256, official lifecycle precedence, direct reconciliation, reconciliation history, and reused-symbol semantics.
- `calc_all()` receives a copy. Analysis, inference, and backtesting consume the calculated frame; persistence never uses `dropna()` as a row-retention policy.
- Snapshot construction order is exact: canonical frame; `calc_all(canonical.copy())`; model/backtest/latest inference mutation of the calculated frame, including `AI_P`; observation-only model-column exclusion; date join into canonical; serialization of `daily`, `latest`, and `rows`.
- Clear calculated and model fields on the canonical frame, then join available calculated values only on matching dates. Warm-up derived fields serialize as JSON `null`; canonical OHLCV and source fields remain present.
- Calculated data may not overwrite canonical `Date`, OHLCV, institutional, margin, short, market, option, or data-quality source values.
- The latest canonical regular-price date must exist in the fully calculated frame; otherwise retain `ValueError("calculated history is unavailable")`.
- Observation-only mode removes `OBSERVATION_MODEL_COLUMNS` from the calculated frame before the date join without removing canonical rows.
- Same-target reruns must preserve one row per date, stable historical OHLCV, and one deterministic non-duplicated receipt.
- Do not synthesize, forward-fill, interpolate, or relabel prices.
- Do not change rolling formulas, model features, LightGBM, backtests, prediction targets, or recommendation policy.
- Retaining the current artifact schema version is allowed only after a refreshed repository-wide persisted-`daily` reader audit classifies every production reader and compatibility tests prove warm-up indicator `null` values are safe. An unclassified reader or failed compatibility test stops implementation for a revised schema decision.
- Every feature-ready persisted-history consumer filters its exact required derived fields at its own boundary. Persisted warm-up rows are never deleted to satisfy a consumer.
- Missing or `null` `source_lineage` is a legitimate legacy artifact. Recovery returns `None`, and existing strict or `--reconcile-legacy-overlaps` behavior continues. Present invalid lineage fails closed. Valid official lineage without direct reconciliation or validated reconciliation history returns `None`.
- `--recover-truncated-history` and `run(recover_truncated_history=False)` are independent explicit opt-ins and default to false.
- With recovery disabled, do not construct a resolver or resolve, open, list, probe, glob, or scan any quarantine path. `OfficialCompatFetcher._daily_rows()` reads only the active artifact.
- Backup resolution and full merged canonical history are cached once per symbol. Price, institutional, and margin calls filter their own ranges from that cache. Receipt retention fields are finalized only from canonical rows actually retained in the artifact that will be persisted.
- Price, institutional, and margin request order must produce byte-identical `daily` and byte-identical `daily_history_recovery` receipts.
- Never use `rglob()`, recursive search, filename search, or first-match selection for recovery.
- Recovery authority comes only from valid direct `legacy_reconciliation` or validated `legacy_reconciliation_history` under valid official lineage.
- Backup target date is exactly `reconciliation.official_snapshot_dates[-1]`. The only path is `<root>/quarantine/tw-recovery/legacy-reconciliation/v2/<target-date>/<official_series_manifest_sha256>/manifest.json`.
- Require a complete schema-v2 manifest and `applied` symbol entry; exact symbol; `entry.original_sha256 == reconciliation.legacy_artifact_sha256`; `entry.backup_path == objects/<original_sha256>.json.gz`; compressed size and SHA-256; bounded uncompressed size; valid gzip; safe child paths; and no symlink or Windows reparse-point component.
- Historical recovery additionally requires `entry.new_sha256 == history_item.reconciled_artifact_sha256`. Direct recovery requires `entry.new_sha256 == active_artifact.compressed_sha256`.
- The exact raw bytes that pass compressed size and SHA-256 validation are the bytes decompressed. The exact decoded bytes that pass the uncompressed-size check are the bytes parsed as JSON. Never verify and reopen an object.
- The decoded backup object must match `TW:<symbol>` and contain valid, strictly ordered, unique daily dates whose declared dates agree with its rows.
- Zero or multiple distinct qualifying backup objects fail closed. Repeated references with the same validated original-object SHA deduplicate only when their complete authorization binding tuples are identical; conflicting target, series, result, path, size, uncompressed-size, or manifest-entry bindings fail as ambiguous. Missing manifest/object, changed bytes, hash/size/gzip/path/symbol/result mismatch, duplicate input dates, or overlap OHLCV conflict has no fallback source.
- On matching overlap OHLCV, the current row wins as a whole; backup-only earlier rows restore the prefix; current-only rows remain unchanged.
- Apply the normal target-date and 730-day range after merge. Receipt start/end/count and `restored_daily_sha256` cover only ordered backup-only canonical rows actually retained and restored after that filter.
- A zero-added-row opt-in rerun with an existing valid matching receipt retains it unchanged only after rebinding the current validated manifest entry and object. A zero-added-row run without a prior valid matching receipt writes no receipt.
- Opt-in receipt revalidation binds the complete manifest-entry SHA, original artifact SHA, expected reconciliation result SHA, backup target date, series manifest SHA, compressed size, uncompressed size, symbol, restored-row identity, and canonical receipt SHA. A changed but parseable entry fails closed.
- A successful direct recovery promotes that direct reconciliation to one validated schema-v2 history envelope with `reconciled_artifact_sha256 == input_artifact_sha256` before persisting the recovered artifact. The recovered artifact no longer carries the direct record, so its next opt-in rerun uses the approved historical result-SHA binding.
- A later normal flag-off run validates and carries the receipt from artifact metadata without quarantine access.
- Recovery errors use the existing generic `run_market_batch` failure path: record the symbol, do not overwrite its artifact, allow `next_index` to advance for a new-symbol failure, retry failed symbols before new symbols on resume, and let `_assert_complete` block publication while an active failure remains.
- Do not change provider-specific fail-fast checkpoint semantics and do not add `local_quant.run_market_batch` to the production implementation boundary.
- Do not weaken official source, lifecycle, reconciliation, artifact-size, gzip-expansion, SHA-256, path, symlink, reparse-point, schema, or publication validation.
- Implementation and fixtures use only temporary directories inside the worktree or OS temp area. Do not access `D:\AbsorbData` and do not interact with PID 17820.
- Do not perform live recovery, GCS, Cloud Run, Scheduled Tasks, publication, production pointer mutation, LINE delivery, merge, or any production operation.
- The future implementation handoff ends after pushing the reviewed branch and opening one Draft PR against `main`. It performs no merge or production operation.

## Command Environment

Before any Task 1-8 Python command, initialize and validate the repository-owned runtime in the current PowerShell session:

```powershell
. .\scripts\python_runtime.ps1
$RepoRoot = (Resolve-Path '.').Path
$PythonExe = Resolve-AbsorbPythonExecutable -RepoRoot $RepoRoot
Assert-AbsorbPythonRuntime -PythonExe $PythonExe -RepoRoot $RepoRoot
$env:PYTHONPATH = [string]::Join(
    [IO.Path]::PathSeparator,
    @($RepoRoot, (Join-Path $RepoRoot '.deps'))
)
```

Every Python command below assumes this block has succeeded and uses `& $PythonExe`. If resolution or the import smoke test fails, stop before RED; provisioning or substituting a runtime is outside this plan.

---

## File Responsibility Map

### Production files modified during later execution

| File | Responsibility |
| --- | --- |
| `stock_papi/quant/features.py` | Export the exact ordered `CALCULATED_COLUMNS` tuple assigned by `calc_all()` without changing formulas or `dropna()`. |
| `local_quant.py` | Normalize canonical dates; keep canonical/calculated frames separate; run inference before persistence; clear and join calculated/model fields by date; serialize complete canonical history. |
| `stock_papi/services/stock_analysis.py` | Filter complete feature rows before the 200-row and latest-row analysis checks. |
| `stock_papi/quant/tw_incremental.py` | Define recovery result/resolver types; cache one full merged history per symbol; filter each dataset independently; finalize or carry receipts from final persisted rows; validate receipt metadata without quarantine. |
| `stock_papi/quant/tw_legacy_reconciliation.py` | Perform one single-read verified backup load; derive exact lineage-bound candidates; validate direct/historical result bindings; merge canonical rows; return immutable recovery facts without dataset-range or receipt-finalization logic. |
| `stock_papi/batch/tw_official_post_close_cli.py` | Add the opt-in flag and programmatic argument; construct the resolver only when enabled; pass final `daily` to `lineage_for`; bind mode into checkpoint identity; retain publication gates. |

### Test files created during later execution

| File | Responsibility |
| --- | --- |
| `tests/test_tw_daily_history_preservation.py` | Calculated-column contract, canonical/calculated/model ordering, warm-up preservation, retention, ETF/short-history cases, and the required multi-stage simulation. |
| `tests/test_persisted_daily_reader_audit.py` | AST inventory of arbitrary `daily` receivers, loader aliases, and `StockSnapshot` consumers; exact classification/schema gate. |
| `tests/test_stock_analysis.py` | Complete-feature filtering before analysis history and latest-row checks. |
| `tests/test_oos_diagnostics.py` | Historical market-factor filtering while canonical liquidity remains usable. |

### Test files modified during later execution

| File | Responsibility |
| --- | --- |
| `tests/report_fixtures.py` | Produce valid OHLCV warm-up rows with every `CALCULATED_COLUMNS` field `None`, finite values for every field on ready rows, and a fully finite latest row. |
| `tests/test_local_quant_batch.py` | Snapshot serialization, latest/OOS `AI_P`, status dates, checkpoint failure/cursor/retry ordering, unchanged failed artifact, ETF, and short-history gates. |
| `tests/test_tw_legacy_reconciliation.py` | Single-read backup trust boundary, eligibility, direct/historical result binding, exact target path, merge conflict, ambiguity, existing receipt rebinding, and unsafe path cases. |
| `tests/test_tw_incremental.py` | One resolver call per symbol, dataset-order independence, range filtering, retention-finalized receipt, zero-row behavior, receipt validation, and flag-off carry-forward. |
| `tests/test_tw_official_post_close_cli.py` | Flag/default isolation, both-opt-in legacy behavior, resolver wiring, checkpoint identity, status/lifecycle, and publication blocking. |
| `tests/test_daily_report_source.py` | Reporting source loading accepts and preserves warm-up rows. |
| `tests/test_quant_snapshot_repository.py` | Hash-bound quant loading accepts warm-up rows and requires a fully ready latest row. |
| `tests/test_observation_views.py` | Canonical candles survive while historical MA20 points filter unavailable values. |
| `tests/test_observation_products.py` | Dashboard/market aggregation uses canonical history and latest-only indicators correctly. |
| `tests/test_industry_report_analytics.py` | Report analytics ignore unavailable historical derived values without deleting canonical rows. |
| `tests/test_industry_report_backtest.py` | Signal selection filters unavailable features while preserving the canonical price calendar. |
| `tests/test_pit_dataset.py` | PIT price/volume rows remain eligible when unrelated indicators are unavailable. |

No other production or test file is in scope. `local_quant.run_market_batch` is verified but not modified.

## Frozen Interfaces

```python
# stock_papi/quant/features.py
CALCULATED_COLUMNS: tuple[str, ...] = (
    "MA_5", "MA20", "RET_1", "RET_5", "RET_20", "RANGE_PCT",
    "VOL_RATIO", "VOL_CHG", "INST_NET_RATIO", "MARGIN_CHG",
    "SHORT_CHG", "RSI", "Volat", "MACD_DIF", "MACD",
    "MACD_OSC", "K", "D", "BB_UP", "BB_DN",
)
```

```python
# local_quant.py
def _canonical_history_frame(frame):
    """Return a copied, market-date-normalized, ascending, unique frame."""

def _persisted_history_frame(canonical_frame, calculated_frame):
    """Clear CALCULATED_COLUMNS and OBSERVATION_MODEL_COLUMNS, then join
    columns present in calculated_frame on exact dates without replacing sources."""
```

`build_stock_snapshot()` calls `_persisted_history_frame()` only after `run_ai_engine(calculated_frame)` or `run_latest_inference(calculated_frame)` returns. In observation-only mode, the model/backtest slot is a deliberate no-op, `OBSERVATION_MODEL_COLUMNS` are removed from `calculated_frame`, and only then is the helper called. The helper always clears stale `CALCULATED_COLUMNS + OBSERVATION_MODEL_COLUMNS` from canonical rows; it joins only those columns present in the calculated frame.

```python
# stock_papi/quant/tw_incremental.py
from stock_papi.quant.features import CALCULATED_COLUMNS


@dataclass(frozen=True)
class HistoryRecoveryResult:
    merged_daily: tuple[Mapping[str, Any], ...]
    restored_candidates: tuple[Mapping[str, Any], ...]
    backup_daily: tuple[Mapping[str, Any], ...]
    input_artifact_sha256: str
    original_artifact_sha256: str
    expected_result_sha256: str
    backup_target_market_date: _datetime.date
    backup_series_manifest_sha256: str
    backup_manifest_entry: Mapping[str, Any]
    existing_receipt: Mapping[str, Any] | None

HistoryRecoveryResolver = Callable[
    [str, IncrementalArtifact],
    HistoryRecoveryResult | None,
]

RECOVERY_DERIVED_FIELDS: frozenset[str] = frozenset((
    *CALCULATED_COLUMNS,
    "AI_P", "FUTURE_RET_5", "T",
))

def _canonical_recovery_source_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize Date to ISO and return persisted source fields for receipt hashes."""

def _finalize_daily_history_recovery(
    result: HistoryRecoveryResult,
    *,
    symbol: str,
    recovery_target_market_date: _datetime.date,
    persisted_daily: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Revalidate an existing receipt or hash retained persisted source rows."""

```

```text
OfficialCompatFetcher.__init__(
    self, root: Path, source: Any, *, pd: Any,
    legacy_overlap_policy: str = "strict",
    recovery_resolver: HistoryRecoveryResolver | None = None,
) -> None

OfficialCompatFetcher._ensure_history_recovery(
    self, symbol: str,
) -> HistoryRecoveryResult | None

OfficialCompatFetcher.lineage_for(
    self, symbol: str, *, persisted_daily: list[dict[str, Any]],
) -> dict[str, Any]
```

`merged_daily` is the complete validated canonical merge before any dataset range. `restored_candidates` is the ordered backup-only subset relative to the active input. `backup_daily` is retained only to identify and revalidate an existing receipt's authorized date range on opt-in rerun. Individual `_daily_rows(symbol, start, end)` calls filter `merged_daily` after `_ensure_history_recovery(symbol)` returns. `_finalize_daily_history_recovery()` derives receipt rows from the normalized final `persisted_daily`, restricted to authorized restored dates; it never hashes backup candidate objects or uses a dataset request range.

The receipt exact field set is:

```python
DAILY_HISTORY_RECOVERY_FIELDS = frozenset({
    "schema_version", "mode", "symbol", "recovery_target_market_date",
    "input_artifact_sha256", "original_artifact_sha256",
    "backup_target_market_date", "backup_series_manifest_sha256",
    "backup_manifest_entry_sha256", "backup_object_size",
    "backup_object_uncompressed_size", "restored_start_date",
    "restored_end_date", "restored_row_count", "restored_daily_sha256",
    "receipt_sha256",
})
```

`backup_manifest_entry_sha256` hashes the complete currently validated entry. `restored_daily_sha256` hashes the ordered canonical source-row projection copied from final normalized `persisted_daily` for authorized backup-only dates that survived retention, using sorted compact JSON with `allow_nan=False`. The projection contains every persisted source field and excludes all `CALCULATED_COLUMNS`, `OBSERVATION_MODEL_COLUMNS`, and private transport keys; its full canonical identity must equal the corresponding source projection of the verified backup row. `receipt_sha256` hashes every receipt field except itself with the same canonical JSON encoding.

If a non-`None` finalized receipt came from the cached direct reconciliation, `lineage_for()` replaces that direct record with one schema-v2 `legacy_reconciliation_history` item whose `reconciled_artifact_sha256` is `HistoryRecoveryResult.input_artifact_sha256`, recomputes `history_sha256`, and then attaches the receipt. It never persists both forms or invents a receipt-based exception to the direct `entry.new_sha256 == active_artifact.compressed_sha256` rule.

```python
# stock_papi/quant/tw_legacy_reconciliation.py
def _read_verified_object(
    root: Path,
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_uncompressed_size: int,
    expected_bytes: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Read once; validate raw; decode that raw; validate and return decoded."""

class LegacyArtifactBackupStore:
    def read_original_document(
        self,
        *,
        symbol: str,
        original_sha256: str,
        expected_result_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the parsed verified original and a copied applied entry."""

def resolve_truncated_daily_history(
    root: Path,
    symbol: str,
    artifact: IncrementalArtifact,
) -> HistoryRecoveryResult | None:
    """Resolve and merge one exact lineage-bound backup without range filtering."""
```

Dependency direction remains one-way: `tw_legacy_reconciliation` imports `HistoryRecoveryResult`, `IncrementalArtifact`, and validation primitives from `tw_incremental`; `tw_incremental` never imports `tw_legacy_reconciliation`. The CLI imports both modules and injects the resolver.

```text
# stock_papi/batch/tw_official_post_close_cli.py
_enrich_batch_identity(
    identity: dict[str, Any],
    *,
    series: OfficialSnapshotSeries,
    audit: Any,
    symbols: list[str],
    reconcile_legacy_overlaps: bool,
    recover_truncated_history: bool,
) -> dict[str, Any]

_patched_pipeline(
    local_quant: Any,
    pipeline: Any,
    fetcher: OfficialCompatFetcher,
    series: OfficialSnapshotSeries,
    audit: Any,
    *,
    backup_store: LegacyArtifactBackupStore | None = None,
    symbols: list[str] | None = None,
    recover_truncated_history: bool = False,
) -> Iterator[None]

_run_stage(
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
) -> tuple[int, set[str]]

run(
    *,
    root: Path,
    target_market_date: datetime.date,
    calendar_artifacts: list[Path],
    limit: int,
    delay: float,
    series_builder=build_official_snapshot_series,
    reconcile_legacy_overlaps: bool = False,
    recover_truncated_history: bool = False,
) -> int
```

The checkpoint identity always contains `"recover_truncated_history": false|true`. `build_stock_snapshot_with_lineage()` calls `fetcher.lineage_for(str(symbol), persisted_daily=result["daily"])` only after the original snapshot builder returns its final retained canonical `daily`.

## Approved-Design Traceability Matrix

| Approved design section | Exact task and test evidence | Command and required result |
| --- | --- | --- |
| Status/authorization and non-goals | Task 8 scope/diff gates; `PersistedDailyReaderAuditTests.test_only_mapped_production_readers_exist` | `git diff --name-only 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD`: only the plan plus mapped implementation files; no production operation. |
| Root cause and canonical history goals | Task 3; `TWHistoryPersistenceTests.test_multistage_history_does_not_erode_and_rerun_is_byte_stable` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests.test_multistage_history_does_not_erode_and_rerun_is_byte_stable -v`: GREEN with stable row counts and bytes. |
| Canonical history invariants | Task 3; `test_canonical_frame_rejects_duplicate_dates_and_sorts_strictly`, `test_warmup_rows_preserve_ohlcv_and_null_derived_fields` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests -v`: duplicate input rejected; ordered unique rows retained. |
| Date/status semantics | Tasks 3 and 7; `LocalQuantBatchTests.test_taiwan_status_snapshot_preserves_last_regular_price_date`, `TWOfficialIncrementalTests.test_status_fetcher_preserves_history_and_exposes_target_evidence` | `& $PythonExe -m unittest tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_status_snapshot_preserves_last_regular_price_date tests.test_tw_incremental.TWOfficialIncrementalTests.test_status_fetcher_preserves_history_and_exposes_target_evidence -v`: regular and non-price identities unchanged. |
| Calculated-frame separation and exact inference order | Tasks 1 and 3; `test_calculated_columns_match_calc_all_assignments_in_order`, `test_latest_inference_ai_p_is_joined_after_mutation`, `test_oos_ai_p_is_joined_on_matching_dates` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation -v`: latest and OOS `AI_P` persist only after inference. |
| Retention/idempotency | Tasks 3 and 6; `test_retention_keeps_only_normal_730_day_request_result`, `test_receipt_hashes_only_restored_rows_in_final_persisted_daily`, `test_dataset_call_orders_are_byte_identical` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation tests.test_tw_incremental -v`: GREEN and identical canonical/receipt bytes. |
| Persisted-reader compatibility/schema decision | Task 2; `PersistedDailyReaderAuditTests.test_only_mapped_production_readers_exist` plus all named warm-up compatibility methods | `& $PythonExe -m unittest tests.test_persisted_daily_reader_audit tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v`: every reader classified and compatible before Task 3. |
| Explicit recovery/default isolation | Tasks 5-7; `test_missing_or_null_lineage_is_legacy_and_not_recovery_eligible`, `test_cli_default_path_never_constructs_resolver_or_touches_quarantine` | `& $PythonExe -m unittest tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli -v`: legacy returns `None`; disabled mode performs zero quarantine calls. |
| Direct and historical reconciliation binding | Tasks 5 and 6; `test_resolver_binds_direct_result_sha_and_exact_snapshot_date`, `test_resolver_binds_historical_result_sha_and_exact_snapshot_date`, `test_direct_recovery_promotes_reconciliation_to_history_and_artifact_rerun_is_byte_identical` | `& $PythonExe -m unittest tests.test_tw_legacy_reconciliation tests.test_tw_incremental -v`: wrong date/series/result SHA fails; direct A0 rotates to history in A1; second opt-in uses exact historical binding. |
| Exact backup trust boundary and TOCTOU | Task 4; `test_verified_reader_reads_object_once_and_parses_same_bytes`, `test_verified_reader_binds_all_sizes_hash_gzip_and_path_checks` | `& $PythonExe -m unittest tests.test_tw_legacy_reconciliation -v`: one read; every mutation fails before parse or at its bound check. |
| Merge rules | Task 5; `test_merge_rejects_duplicate_dates_and_overlap_ohlcv_conflict`, `test_merge_keeps_current_whole_row_when_ohlcv_matches` | `& $PythonExe -m unittest tests.test_tw_legacy_reconciliation -v`: conflicts and non-prefix restoration fail; current overlap row wins. |
| Deterministic receipt and existing-receipt revalidation | Task 6; `test_receipt_hashes_only_restored_rows_in_final_persisted_daily`, `test_opt_in_rerun_rebinds_existing_receipt_to_current_verified_backup`, `test_existing_receipt_rebind_allows_only_retention_aged_rows_to_be_absent`, `test_zero_retained_rows_without_receipt_returns_none` | `& $PythonExe -m unittest tests.test_tw_incremental tests.test_tw_legacy_reconciliation -v`: receipt bytes stable; changed parseable entry or in-window missing row fails; retention-aged rows and no-zero-row behavior pass. |
| Checkpoint/resume/publication | Task 7; `test_recovery_failure_advances_cursor_without_overwriting_artifact`, `test_recovery_failure_resume_retries_failed_before_new`, `test_recovery_failure_blocks_assert_complete_and_publication` | `& $PythonExe -m unittest tests.test_tw_official_post_close_cli tests.test_local_quant_batch -v`: existing cursor semantics preserved and publication blocked. |
| Normal runtime quarantine isolation | Tasks 6 and 7; `test_flag_off_carries_valid_receipt_without_quarantine`, `test_cli_default_path_never_constructs_resolver_or_touches_quarantine` | `& $PythonExe -m unittest tests.test_tw_incremental tests.test_tw_official_post_close_cli -v`: patched quarantine access is never called. |
| ETF and short history | Task 3; `test_etf_history_preserves_warmup_rows`, `test_short_history_still_fails_closed_when_latest_is_not_calculated` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests -v`: ETF retains canonical rows; insufficient latest calculation raises the existing error. |
| Reused-symbol lifecycle | Task 7; existing `TWLegacyOverlapReconciliationTests.test_official_lineage_allows_symbol_history_after_series_start` and `tests.test_tw_trading_status` | `& $PythonExe -m unittest tests.test_tw_incremental.TWLegacyOverlapReconciliationTests.test_official_lineage_allows_symbol_history_after_series_start tests.test_tw_trading_status -v`: lifecycle/status regressions GREEN. |
| Required sequential simulation | Task 3; `test_multistage_history_does_not_erode_and_rerun_is_byte_stable` | `& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests.test_multistage_history_does_not_erode_and_rerun_is_byte_stable -v`: the five-stage temporary simulation is GREEN. |
| Final validation and Draft PR handoff | Task 8; focused/full suites, compile, Node/PowerShell parsing, diff, independent review, push, Draft PR | `& $PythonExe -m unittest discover -s tests -v`, `& $PythonExe -m compileall -q local_quant.py stock_papi reporting tests`, `node --check static/app.js`, parser/diff/review/push commands: every command exits `0`, reviewer has no Critical/Important finding, local/remote SHAs match before Draft PR. |

### Task 1: Freeze the calculated-column contract

**Files:**

- Create: `tests/test_tw_daily_history_preservation.py`
- Modify: `stock_papi/quant/features.py` in `calc_all`'s module-level contract only

**Interfaces:**

- Produces `CALCULATED_COLUMNS: tuple[str, ...]` exactly as frozen above.
- Leaves `calc_all(frame, *, pd, np)` formulas and `return frame.dropna()` unchanged.

- [ ] **Step 1 — RED: write the exact contract test**

```python
import ast
import inspect
import unittest


class TWCalculatedColumnContractTests(unittest.TestCase):
    def test_calculated_columns_match_calc_all_assignments_in_order(self):
        from stock_papi.quant import features

        tree = ast.parse(inspect.getsource(features.calc_all))
        assigned = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "frame"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    assigned.append((node.lineno, target.slice.value))
        names = tuple(name for _, name in sorted(assigned))
        self.assertEqual(features.CALCULATED_COLUMNS, names)
        self.assertEqual(len(names), 20)
```

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWCalculatedColumnContractTests.test_calculated_columns_match_calc_all_assignments_in_order -v
```

Expected RED: `AttributeError` states that `stock_papi.quant.features` has no `CALCULATED_COLUMNS`; test discovery and imports succeed.

- [ ] **Step 3 — Implementation: export the exact tuple**

```python
CALCULATED_COLUMNS: tuple[str, ...] = (
    "MA_5", "MA20", "RET_1", "RET_5", "RET_20", "RANGE_PCT",
    "VOL_RATIO", "VOL_CHG", "INST_NET_RATIO", "MARGIN_CHG",
    "SHORT_CHG", "RSI", "Volat", "MACD_DIF", "MACD",
    "MACD_OSC", "K", "D", "BB_UP", "BB_DN",
)
```

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWCalculatedColumnContractTests.test_calculated_columns_match_calc_all_assignments_in_order -v
```

Expected GREEN: one test passes; `git diff -- stock_papi/quant/features.py` shows no formula change.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/quant/features.py tests/test_tw_daily_history_preservation.py
git commit -m "test: freeze TW calculated column contract"
```

### Task 2: Audit every persisted-daily consumer and gate schema compatibility

**Files:**

- Create: `tests/test_persisted_daily_reader_audit.py`
- Create: `tests/test_stock_analysis.py`
- Create: `tests/test_oos_diagnostics.py`
- Modify: `tests/report_fixtures.py`
- Modify: `stock_papi/services/stock_analysis.py` in `snapshot_dataframe`
- Modify: `tests/test_daily_report_source.py`
- Modify: `tests/test_quant_snapshot_repository.py`
- Modify: `tests/test_observation_views.py`
- Modify: `tests/test_observation_products.py`
- Modify: `tests/test_industry_report_analytics.py`
- Modify: `tests/test_industry_report_backtest.py`
- Modify: `tests/test_pit_dataset.py`

**Interfaces:**

- Produces `warmup_stock_document(symbol, *, rows=70, warmup_rows=20, as_of="2026-07-03") -> dict` in `tests/report_fixtures.py`.
- Produces an exact AST inventory mapping every detected production boundary to `canonical-OHLCV`, `latest-only`, or `feature-ready-history`.
- Changes only `snapshot_dataframe()` in production; existing `finite_number`, `_number`, and explicit non-`null` guards remain the filtering boundaries elsewhere.

- [ ] **Step 1 — RED: write the exhaustive AST inventory, warm-up fixture, and compatibility tests**

The audit scans `local_quant.py`, `stock_papi/**/*.py`, `reporting/**/*.py`, and `scripts/**/*.py`. Its visitor tracks class/function scope, imported aliases of `load_incremental_artifact` and `StockSnapshot`, arbitrary receivers for `value["daily"]`, `value['daily']`, `value.get("daily")`, `value.get('daily')`, and `value.daily`, calls through loader aliases, `StockSnapshot.from_document`, and `StockSnapshot` type consumers. It records `(relative_file, qualified_scope)` once with all evidence kinds.

```python
import ast
from collections import defaultdict


class PersistedDailyVisitor(ast.NodeVisitor):
    def __init__(self, relative_file):
        self.relative_file = relative_file
        self.scope = []
        self.aliases = {
            "load_incremental_artifact": "load_incremental_artifact",
            "StockSnapshot": "StockSnapshot",
        }
        self.detected = defaultdict(set)

    def _record(self, kind):
        if self.scope:
            self.detected[(self.relative_file, ".".join(self.scope))].add(kind)

    def visit_ImportFrom(self, node):
        for item in node.names:
            if item.name in {"load_incremental_artifact", "StockSnapshot"}:
                self.aliases[item.asname or item.name] = item.name
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and node.slice.value == "daily":
            self._record("subscript-daily")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == "daily":
            self._record("attribute-daily")
        if node.attr == "StockSnapshot":
            self._record("StockSnapshot-consumer")
        if (
            node.attr == "from_document"
            and (
                (
                    isinstance(node.value, ast.Name)
                    and self.aliases.get(node.value.id) == "StockSnapshot"
                )
                or (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr == "StockSnapshot"
                )
            )
        ):
            self._record("StockSnapshot.from_document")
        self.generic_visit(node)

    def visit_Call(self, node):
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "daily"
        ):
            self._record("get-daily")
        if (
            isinstance(function, ast.Name)
            and self.aliases.get(function.id) == "load_incremental_artifact"
        ):
            self._record("load_incremental_artifact")
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "load_incremental_artifact"
        ):
            self._record("load_incremental_artifact")
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "StockSnapshot"
        ):
            self._record("StockSnapshot-construction")
        self.generic_visit(node)

    def visit_Name(self, node):
        if self.aliases.get(node.id) == "StockSnapshot":
            self._record("StockSnapshot-consumer")


def discover_persisted_daily_readers(repository_root):
    targets = [repository_root / "local_quant.py"]
    for directory in ("stock_papi", "reporting", "scripts"):
        targets.extend(sorted((repository_root / directory).rglob("*.py")))
    detected = defaultdict(set)
    for path in targets:
        relative = path.relative_to(repository_root).as_posix()
        visitor = PersistedDailyVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        for boundary, kinds in visitor.detected.items():
            detected[boundary].update(kinds)
    return detected
```

```python
READER_CONTRACTS = {
    ("local_quant.py", "_validated_artifact"): "latest-only",
    ("reporting/industry_analytics.py", "_stock_return"): "canonical-OHLCV",
    ("reporting/industry_analytics.py", "_foreign_net_5"): "canonical-OHLCV",
    ("reporting/industry_analytics.py", "_market_snapshot"): "feature-ready-history",
    ("reporting/industry_analytics.py", "_industry_snapshot"): "feature-ready-history",
    ("reporting/industry_analytics.py", "_risk_hints"): "latest-only",
    ("reporting/industry_analytics.py", "_model_quality"): "feature-ready-history",
    ("reporting/industry_analytics.py", "build_daily_report"): "feature-ready-history",
    ("reporting/industry_analytics.py", "build_daily_report.industry_snapshots"): "feature-ready-history",
    ("reporting/industry_backtest.py", "backtest_industry"): "feature-ready-history",
    ("reporting/migrate_quant_manifest.py", "_validate_stock"): "latest-only",
    ("reporting/schemas.py", "StockSnapshot.from_document"): "latest-only",
    ("reporting/schemas.py", "StockSnapshot.latest"): "latest-only",
    ("reporting/schemas.py", "LoadedReportSource"): "latest-only",
    ("reporting/source_loader.py", "_load_manifest_source"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_return_pct"): "canonical-OHLCV",
    ("stock_papi/batch/observation_products.py", "_validate_source"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_trading_status_observations"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_market_daily_returns"): "canonical-OHLCV",
    ("stock_papi/batch/observation_products.py", "_market_observation"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_industry_observations"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_stock_events"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_etf_observations"): "latest-only",
    ("stock_papi/batch/oos_diagnostics.py", "_enrich_point_in_time"): "feature-ready-history",
    ("stock_papi/batch/tw_official_post_close_cli.py", "_assert_complete"): "latest-only",
    ("stock_papi/quant/tw_artifact_audit.py", "audit_artifact_dates"): "latest-only",
    ("stock_papi/quant/tw_incremental.py", "load_incremental_artifact"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "audit_artifact_dates"): "latest-only",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._load_artifact"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._daily_rows"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._reconciliation_plan"): "canonical-OHLCV",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "LegacyArtifactBackupStore._validate_original"): "latest-only",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "LegacyArtifactBackupStore._validate_expected_result"): "latest-only",
    ("stock_papi/repositories/quant_snapshots.py", "fetch_quant_snapshot"): "latest-only",
    ("stock_papi/research/pit_dataset.py", "_history_rows"): "canonical-OHLCV",
    ("stock_papi/services/observation_view.py", "build_stock_observation"): "feature-ready-history",
    ("stock_papi/services/stock_analysis.py", "snapshot_dataframe"): "feature-ready-history",
}

NON_PERSISTED_BUILDERS = {
    ("scripts/generate_sample_daily_report.py", "build_documents"),
}
```

`test_only_mapped_production_readers_exist` asserts `detected - NON_PERSISTED_BUILDERS == set(READER_CONTRACTS)`, asserts the three allowed classification strings exactly, and asserts the one excluded script constructs test input rather than loading persisted production data.

The fixture implementation is exact:

```python
from stock_papi.quant.features import CALCULATED_COLUMNS


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
```

`WarmupFixtureContractTests.test_warmup_fixture_has_valid_ohlcv_null_warmup_and_finite_ready_rows` asserts `sample_data is False`, checks finite positive OHLCV on every row, every calculated field `None` only for the first 20 rows, every calculated field finite afterward, and finite `CALCULATED_COLUMNS + ("AI_P",)` on `daily[-1]`. This makes the fixture production-shaped without weakening either production sample-data rejection gate.

Write these exact compatibility methods before running the RED command:

```text
DailyReportSourceTests.test_loader_accepts_canonical_warmup_rows_with_null_indicators
QuantSnapshotRepositoryTests.test_repository_accepts_warmup_rows_and_finite_latest
ObservationViewTests.test_warmup_rows_remain_candles_and_null_ma20_is_filtered
ObservationProductsTests.test_market_aggregation_uses_canonical_rows_and_ready_latest
IndustryReportAnalyticsTests.test_report_filters_null_historical_features_without_dropping_prices
IndustryReportBacktestTests.test_backtest_filters_null_signals_but_keeps_price_calendar
PitDatasetTests.test_pit_history_uses_ohlcv_when_indicators_are_null
OOSDiagnosticsCompatibilityTests.test_oos_filters_null_market_factor_but_keeps_liquidity
StockAnalysisCompatibilityTests.test_snapshot_dataframe_filters_complete_features_before_history_gate
StockAnalysisCompatibilityTests.test_snapshot_dataframe_rejects_bool_and_non_finite_required_values
StockAnalysisCompatibilityTests.test_snapshot_dataframe_rejects_stale_complete_prefix_when_latest_is_not_ready
```

The stock-analysis history test calls `warmup_stock_document("2330", rows=220, warmup_rows=20)`: the current implementation incorrectly accepts 220 rows containing unavailable required fields, while the required implementation returns exactly 200 complete rows. A second 219-row case returns `None` after filtering because only 199 feature-ready rows remain. The hostile-value test injects `True`, `float("inf")`, and `float("-inf")` into required fields and proves each row is excluded. The stale-latest test supplies 200 complete older rows plus a latest row whose required field is `None`, boolean, or non-finite and requires `None`, never a frame ending on the prior date.

Every `feature-ready-history` mapping has this explicit boundary contract:

| Boundary | Exact required-value filter |
| --- | --- |
| `stock_papi.services.stock_analysis.snapshot_dataframe` | All twelve fields in the implementation snippet below must pass `finite_numeric`; the retained latest date must equal the persisted latest date before the 200-row gate. |
| `stock_papi.services.observation_view.build_stock_observation` | Historical `MA20` points require existing `_number(row.get("MA20")) is not None`; OHLCV candles remain independent. |
| `reporting.industry_analytics._market_snapshot` | Historical `MARKET_RET_1` requires `finite_number(...) is not None`; latest market returns and `AI_P` use the same finite guard. |
| `reporting.industry_analytics._industry_snapshot` | `_stock_return` requires finite canonical `Close` pairs; latest `AI_P`, `MA20`, `VOL_RATIO`, and `INST_NET_RATIO` each require `finite_number(...) is not None`. |
| `reporting.industry_analytics._model_quality` | A historical observation is appended only when `AI_P`, current `Close`, and horizon `Close` are finite and current `Close > 0`. |
| `reporting.industry_analytics.build_daily_report` and nested `build_daily_report.industry_snapshots` | They do not read a derived row unchecked; they delegate only to `_market_snapshot`, `_industry_snapshot`, and `_model_quality` above. |
| `reporting.industry_backtest.backtest_industry` | The calendar retains valid dates/prices; `MARKET_RET_1`, current/future `Close`, and signal `AI_P` pass `finite_number`, with positive current close required before signal selection. |
| `stock_papi.batch.oos_diagnostics._enrich_point_in_time` | Historical `MARKET_RET_20`, `Close`, and `Volume` require exact numeric non-bool values plus `math.isfinite`; liquidity requires positive close and volume. |

The named compatibility tests exercise each row of this table; a missing guard or changed required field fails Task 2 before the schema-version decision.

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_persisted_daily_reader_audit tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected RED: inventory and fixture-contract tests pass; the three `StockAnalysisCompatibilityTests` fail because `snapshot_dataframe()` counts warm-up rows, accepts bool/infinity, and can return a stale complete prefix when the persisted latest row is not feature-ready. No import or fixture-construction error is accepted as RED.

- [ ] **Step 3 — Implementation: enforce each consumer contract**

The only production change is:

```python
import math


required = [
    "Open", "High", "Low", "Close", "MA20", "RSI", "Volat",
    "MACD_OSC", "K", "D", "AI_P", "ForeignNet",
]
if frame.empty or not set(required).issubset(frame.columns):
    return None
persisted_latest_date = frame.index[-1]

def finite_numeric(value):
    if value is None or isinstance(value, (bool, str, bytes)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False

complete = pd.Series(True, index=frame.index)
for name in required:
    complete &= frame[name].map(finite_numeric)
frame = frame.loc[complete]
if len(frame) < 200 or frame.index[-1] != persisted_latest_date:
    return None
return frame
```

The other feature-ready boundaries already use `finite_number`, `_number`, or an explicit `None` check before consuming each derived value; their new tests freeze that behavior. Canonical-OHLCV tests assert warm-up rows remain present. Latest-only tests assert the loader accepts early `null` values and the latest row is finite.

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_persisted_daily_reader_audit tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected GREEN: every detected production boundary is classified, every named compatibility test passes, and the schema gate records that schema version remains unchanged. Any unmatched reader or non-finite latest field blocks Task 3.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/services/stock_analysis.py tests/report_fixtures.py tests/test_persisted_daily_reader_audit.py tests/test_stock_analysis.py tests/test_oos_diagnostics.py tests/test_daily_report_source.py tests/test_quant_snapshot_repository.py tests/test_observation_views.py tests/test_observation_products.py tests/test_industry_report_analytics.py tests/test_industry_report_backtest.py tests/test_pit_dataset.py
git commit -m "test: gate persisted daily reader compatibility"
```

### Task 3: Separate canonical persistence from calculated analysis

**Files:**

- Modify: `local_quant.py` in `build_stock_snapshot` and two private frame helpers
- Modify: `tests/test_tw_daily_history_preservation.py`
- Modify: `tests/test_local_quant_batch.py`

**Interfaces:**

- Consumes `CALCULATED_COLUMNS` and existing `OBSERVATION_MODEL_COLUMNS`.
- Produces `_canonical_history_frame(frame)` and `_persisted_history_frame(canonical_frame, calculated_frame)` exactly as frozen.
- Preserves the existing `build_stock_snapshot` public signature and result schema.

- [ ] **Step 1 — RED: write ordering, retention, and simulation tests**

```text
TWHistoryPersistenceTests.test_canonical_frame_rejects_duplicate_dates_and_sorts_strictly
TWHistoryPersistenceTests.test_warmup_rows_preserve_ohlcv_and_null_derived_fields
TWHistoryPersistenceTests.test_latest_inference_ai_p_is_joined_after_mutation
TWHistoryPersistenceTests.test_oos_ai_p_is_joined_on_matching_dates
TWHistoryPersistenceTests.test_retention_keeps_only_normal_730_day_request_result
TWHistoryPersistenceTests.test_etf_history_preserves_warmup_rows
TWHistoryPersistenceTests.test_short_history_still_fails_closed_when_latest_is_not_calculated
TWHistoryPersistenceTests.test_multistage_history_does_not_erode_and_rerun_is_byte_stable
LocalQuantBatchTests.test_taiwan_snapshot_persists_latest_and_oos_ai_p_after_model_mutation
```

The simulation uses the real `calc_all()` and temporary artifact writes/reloads:

```text
baseline -> 2026-07-29 -> 2026-07-30 -> 2026-07-31 -> rerun 2026-07-31
```

It records row count, first/last date, uniqueness, latest calculated availability, and canonical OHLCV bytes after every stage. The rerun must match the first 2026-07-31 artifact `daily` bytes exactly.

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_snapshot_persists_latest_and_oos_ai_p_after_model_mutation -v
```

Expected RED: the first real update persists only the `calc_all().dropna()` suffix; warm-up rows disappear, and the multi-stage case reaches `ValueError("calculated history is unavailable")`. The inference spies also show persistence is sourced from the shortened calculated frame.

- [ ] **Step 3 — Implementation: enforce the exact construction order**

```python
from stock_papi.quant.features import CALCULATED_COLUMNS


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
        result.loc[calculated_frame.index, name] = calculated_frame[name]
    return result


canonical_frame = _canonical_history_frame(frame)
calculated_frame = pipeline.calc_all(canonical_frame.copy())
if calculated_frame is None or calculated_frame.empty:
    raise ValueError("calculated history is unavailable")
if canonical_frame.index[-1] not in calculated_frame.index:
    raise ValueError("calculated history is unavailable")

compatibility = None
if observation_only:
    backtest = {}
    model_version = OBSERVATION_SOURCE_VERSION
    calculated_frame = calculated_frame.drop(
        columns=list(OBSERVATION_MODEL_COLUMNS), errors="ignore"
    )
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
        orient="records", date_format="iso", date_unit="ms"
    )
)
```

The promoted/degraded compatibility branch is the existing code with `frame` renamed to `calculated_frame`; no model formula or policy changes. `_persisted_history_frame()` runs after that entire branch, removes stale calculated/model fields from the canonical copy, rejects a calculated index outside canonical dates or duplicate indexes, and joins only named fields present in `calculated_frame`.

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_daily_history_preservation tests.test_local_quant_batch -v
```

Expected GREEN: canonical dates remain ordered/unique; warm-up OHLCV stays; unavailable derived fields are `null`; latest inference `AI_P` and OOS `AI_P` appear on matching dates; observation output excludes model fields; status, ETF, short-history, 730-day retention, and same-date rerun tests pass.

- [ ] **Step 5 — Commit**

```powershell
git add local_quant.py tests/test_tw_daily_history_preservation.py tests/test_local_quant_batch.py
git commit -m "fix: preserve canonical TW daily history"
```

### Task 4: Add a single-read verified backup reader

**Files:**

- Modify: `stock_papi/quant/tw_legacy_reconciliation.py` in object verification and `LegacyArtifactBackupStore`
- Modify: `tests/test_tw_legacy_reconciliation.py`

**Interfaces:**

- Produces `_read_verified_object(root, path, *, expected_sha256, expected_size, expected_uncompressed_size, expected_bytes=None) -> tuple[bytes, bytes]` and the frozen `LegacyArtifactBackupStore.read_original_document` signature.
- Reuses `_read_bytes`, `_decode_gzip`, `_assert_safe_child`, `_load_manifest`, `_ENTRY_FIELDS`, and existing manifest validation.
- Performs no write and no second object read.

- [ ] **Step 1 — RED: write exact trust-boundary tests**

```text
LegacyArtifactBackupStoreTests.test_verified_reader_reads_object_once_and_parses_same_bytes
LegacyArtifactBackupStoreTests.test_verified_reader_rejects_changed_bytes_before_decode
LegacyArtifactBackupStoreTests.test_verified_reader_binds_all_sizes_hash_gzip_and_path_checks
LegacyArtifactBackupStoreTests.test_verified_reader_rejects_symbol_market_or_daily_identity_mismatch
LegacyArtifactBackupStoreTests.test_verified_reader_rejects_symlink_and_windows_reparse_components
```

The first test patches `_read_bytes` with a valid first return and different second return, asserts call count is one, and asserts the parsed document came from the first validated decoded bytes. The second returns changed bytes on the single read and patches `_decode_gzip` to fail if called; SHA mismatch must stop before decode. The parameterized contract test mutates compressed size, compressed SHA, gzip bytes, expansion beyond `MAX_UNCOMPRESSED_BYTES`, exact uncompressed size, manifest path, and entry fields one at a time.

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_verified_reader_reads_object_once_and_parses_same_bytes tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_verified_reader_rejects_changed_bytes_before_decode tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_verified_reader_binds_all_sizes_hash_gzip_and_path_checks tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_verified_reader_rejects_symbol_market_or_daily_identity_mismatch tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_verified_reader_rejects_symlink_and_windows_reparse_components -v
```

Expected RED: `read_original_document` is absent. Test setup creates a valid schema-v2 manifest and object before invoking the missing method.

- [ ] **Step 3 — Implementation: validate and parse one byte sequence**

```python
def _read_verified_object(
    root,
    path,
    *,
    expected_sha256,
    expected_size,
    expected_uncompressed_size,
    expected_bytes=None,
):
    _assert_safe_child(root, path)
    raw = _read_bytes(path)
    if (
        len(raw) != expected_size
        or _sha256(raw) != expected_sha256
        or (expected_bytes is not None and raw != expected_bytes)
    ):
        raise LegacyReconciliationError(
            "legacy reconciliation backup object conflicts"
        )
    decoded = _decode_gzip(raw)
    if len(decoded) != expected_uncompressed_size:
        raise LegacyReconciliationError(
            "legacy reconciliation backup object conflicts"
        )
    return raw, decoded
```

Every call passes `self.root`; the trust root is never derived from the object path.

Refactor `_verify_object()` to call this helper and discard its return. `read_original_document()` loads the exact required manifest, selects `manifest["entries"][symbol]`, requires status `applied`, exact `original_sha256`, `new_sha256 == expected_result_sha256`, and exact `backup_path`, then calls `_read_verified_object()` once. It parses `json.loads(decoded.decode("utf-8"))` from the returned `decoded` and validates market, symbol, declared dates, ordered unique `daily`, finite OHLCV, and exact identity.

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_legacy_reconciliation -v
```

Expected GREEN: all new single-read/path/object tests and all existing backup write/resume/idempotency tests pass; object parsing is reachable only from returned verified decoded bytes.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/quant/tw_legacy_reconciliation.py tests/test_tw_legacy_reconciliation.py
git commit -m "feat: read verified TW reconciliation backups"
```

### Task 5: Resolve and merge exact lineage-authorized history

**Files:**

- Modify: `stock_papi/quant/tw_incremental.py` for `HistoryRecoveryResult` and `HistoryRecoveryResolver` definitions only
- Modify: `stock_papi/quant/tw_legacy_reconciliation.py` for resolver and merge helpers
- Modify: `tests/test_tw_legacy_reconciliation.py`
- Modify: `tests/test_persisted_daily_reader_audit.py` to classify new recovery readers as `canonical-OHLCV`

**Interfaces:**

- Produces `HistoryRecoveryResult`, `HistoryRecoveryResolver`, and `resolve_truncated_daily_history(root, symbol, artifact)` exactly as frozen.
- Resolver has no dataset start/end and does not build or finalize a receipt.
- Preserves the one-way import direction from `tw_legacy_reconciliation` to `tw_incremental`.

- [ ] **Step 1 — RED: write eligibility, binding, ambiguity, and merge tests**

```text
LegacyArtifactBackupStoreTests.test_missing_or_null_lineage_is_legacy_and_not_recovery_eligible
LegacyArtifactBackupStoreTests.test_present_invalid_lineage_fails_closed_for_recovery
LegacyArtifactBackupStoreTests.test_valid_official_lineage_without_reconciliation_returns_none
LegacyArtifactBackupStoreTests.test_resolver_binds_direct_result_sha_and_exact_snapshot_date
LegacyArtifactBackupStoreTests.test_resolver_binds_historical_result_sha_and_exact_snapshot_date
LegacyArtifactBackupStoreTests.test_resolver_rejects_missing_or_multiple_distinct_backups
LegacyArtifactBackupStoreTests.test_resolver_deduplicates_identical_repeated_history_bindings
LegacyArtifactBackupStoreTests.test_resolver_rejects_same_object_sha_with_conflicting_authorization_bindings
LegacyArtifactBackupStoreTests.test_merge_rejects_duplicate_dates_and_overlap_ohlcv_conflict
LegacyArtifactBackupStoreTests.test_merge_rejects_bool_nan_and_infinite_ohlcv
LegacyArtifactBackupStoreTests.test_merge_rejects_non_prefix_backup_only_rows
LegacyArtifactBackupStoreTests.test_merge_keeps_current_whole_row_when_ohlcv_matches
LegacyArtifactBackupStoreTests.test_resolver_returns_full_merge_and_candidates_without_range_filter
```

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_missing_or_null_lineage_is_legacy_and_not_recovery_eligible tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_present_invalid_lineage_fails_closed_for_recovery tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_valid_official_lineage_without_reconciliation_returns_none tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_binds_direct_result_sha_and_exact_snapshot_date tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_binds_historical_result_sha_and_exact_snapshot_date tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_rejects_missing_or_multiple_distinct_backups tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_deduplicates_identical_repeated_history_bindings tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_rejects_same_object_sha_with_conflicting_authorization_bindings tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_merge_rejects_duplicate_dates_and_overlap_ohlcv_conflict tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_merge_rejects_bool_nan_and_infinite_ohlcv tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_merge_rejects_non_prefix_backup_only_rows tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_merge_keeps_current_whole_row_when_ohlcv_matches tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_returns_full_merge_and_candidates_without_range_filter -v
```

Expected RED: import of `resolve_truncated_daily_history` or `HistoryRecoveryResult` fails; fixtures and the Task 4 verified reader succeed independently.

- [ ] **Step 3 — Implementation: separate eligibility, candidate binding, and merge**

Eligibility is exact:

```python
_RECOVERY_MISSING = object()

lineage = artifact.document.get("source_lineage", _RECOVERY_MISSING)
if lineage is _RECOVERY_MISSING or lineage is None:
    return None
if not OfficialCompatFetcher._valid_official_lineage(lineage, artifact):
    raise LegacyReconciliationError(
        f"daily history recovery lineage is invalid for TW:{symbol}"
    )
if (
    "legacy_reconciliation" not in lineage
    and "legacy_reconciliation_history" not in lineage
):
    return None
```

Add `Callable` to the existing `tw_incremental` typing import. Add `copy` and `MappingProxyType` imports to `tw_legacy_reconciliation`; do not import `tw_legacy_reconciliation` from `tw_incremental`.

Candidate tuples are `(reconciliation, expected_result_sha256)`. Direct always uses `artifact.compressed_sha256`; historical always uses each `history_item["reconciled_artifact_sha256"]`. For each candidate, parse `reconciliation["official_snapshot_dates"][-1]` as the exact target, instantiate `LegacyArtifactBackupStore(root, target_date=target, series_manifest_sha256=reconciliation["official_series_manifest_sha256"])`, and call `read_original_document(symbol=symbol, original_sha256=reconciliation["legacy_artifact_sha256"], expected_result_sha256=expected_result_sha256)`. Any malformed or missing candidate fails. Task 6 converts a successfully recovered direct reconciliation into history before A1 is persisted, so every later opt-in rerun reaches this same historical rule without a direct-binding exception.

After every candidate is fully verified, construct this binding tuple:

```python
binding = (
    entry["original_sha256"],
    target.isoformat(),
    reconciliation["official_series_manifest_sha256"],
    expected_result_sha256,
    entry["backup_path"],
    entry["original_size"],
    entry["original_uncompressed_size"],
    OfficialCompatFetcher._canonical_json_sha256(entry),
)
```

Repeated references deduplicate only when the complete tuple is identical. Two tuples with the same original SHA but any other unequal element fail as conflicting authorization; two distinct original SHAs also fail. Exactly one binding remains, so selection never depends on history order or first-match behavior.

```python
def _merge_recovery_daily(active_daily, backup_daily):
    active = _validated_daily_by_date(active_daily)
    backup = _validated_daily_by_date(backup_daily)
    for day in active.keys() & backup.keys():
        for name in ("Open", "High", "Low", "Close", "Volume"):
            if _finite_number(active[day].get(name)) != _finite_number(
                backup[day].get(name)
            ):
                raise LegacyReconciliationError(
                    "daily history OHLCV conflict"
                )
    merged = {**backup, **active}
    restored_dates = sorted(backup.keys() - active.keys())
    if restored_dates and restored_dates[-1] >= min(active):
        raise LegacyReconciliationError(
            "daily history recovery is not a missing prefix"
        )
    restored = [backup[day] for day in restored_dates]
    return (
        tuple(dict(merged[day]) for day in sorted(merged)),
        tuple(dict(row) for row in restored),
    )
```

`_validated_daily_by_date()` rejects non-list/non-dict rows, booleans, NaN, infinity, invalid dates, duplicate dates, and non-increasing input dates. Build `HistoryRecoveryResult` with full `merged_daily`, `restored_candidates`, and verified `backup_daily` rows each wrapped as `MappingProxyType(copy.deepcopy(row))`; artifact input SHA; entry original SHA; candidate result SHA; exact target/series; `MappingProxyType(copy.deepcopy(entry))`; and `MappingProxyType(copy.deepcopy(existing_receipt))` or `None`. Do not apply a request range and do not compute retention fields.

Extend the inventory with the direct backup-document readers introduced here:

```python
READER_CONTRACTS.update({
    (
        "stock_papi/quant/tw_legacy_reconciliation.py",
        "LegacyArtifactBackupStore.read_original_document",
    ): "canonical-OHLCV",
    (
        "stock_papi/quant/tw_legacy_reconciliation.py",
        "resolve_truncated_daily_history",
    ): "canonical-OHLCV",
})
```

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_legacy_reconciliation tests.test_persisted_daily_reader_audit -v
```

Expected GREEN: missing/null legacy lineage and valid no-reconciliation lineage return `None`; present invalid lineage and every hostile binding fail closed; direct/historical exact bindings, prefix-only restoration, and current-row precedence pass; the AST inventory contains each new recovery reader.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/quant/tw_incremental.py stock_papi/quant/tw_legacy_reconciliation.py tests/test_tw_legacy_reconciliation.py tests/test_persisted_daily_reader_audit.py
git commit -m "feat: resolve exact TW recovery history"
```

### Task 6: Cache merged history and finalize deterministic receipts

**Files:**

- Modify: `stock_papi/quant/tw_incremental.py` in `OfficialCompatFetcher`, receipt validation, and receipt finalization
- Modify: `tests/test_tw_incremental.py`
- Modify: `tests/test_tw_legacy_reconciliation.py`
- Modify: `tests/test_persisted_daily_reader_audit.py` to classify finalization/cache readers

**Interfaces:**

- Consumes `HistoryRecoveryResolver` and `HistoryRecoveryResult` from Task 5.
- Produces `_ensure_history_recovery(symbol)`, range-filtered `_daily_rows`, `_finalize_daily_history_recovery`, and `lineage_for(symbol, *, persisted_daily)` exactly as frozen.
- Stores one cache entry per symbol, including `None`, so all dataset orders resolve once.
- Promotes a selected direct reconciliation to a validated history envelope only after receipt finalization succeeds; it adds no direct-result binding exception.

- [ ] **Step 1 — RED: write cache, order, retention, and receipt tests**

```text
TWOfficialIncrementalTests.test_recovery_resolver_is_optional_and_not_called_by_default
TWOfficialIncrementalTests.test_recovery_resolver_is_called_once_for_all_dataset_orders
TWOfficialIncrementalTests.test_recovered_daily_rows_filter_each_requested_range_after_cache
TWOfficialIncrementalTests.test_dataset_call_orders_are_byte_identical
TWOfficialIncrementalTests.test_receipt_hashes_only_restored_rows_in_final_persisted_daily
TWOfficialIncrementalTests.test_receipt_hashes_final_persisted_source_projection_not_backup_candidate
TWOfficialIncrementalTests.test_zero_retained_rows_without_receipt_returns_none
TWOfficialIncrementalTests.test_opt_in_rerun_rebinds_existing_receipt_to_current_verified_backup
TWOfficialIncrementalTests.test_existing_receipt_revalidation_uses_current_persisted_source_projection
TWOfficialIncrementalTests.test_direct_recovery_promotes_reconciliation_to_history_and_artifact_rerun_is_byte_identical
TWOfficialIncrementalTests.test_existing_receipt_rebind_allows_only_retention_aged_rows_to_be_absent
TWOfficialIncrementalTests.test_changed_parseable_manifest_entry_fails_existing_receipt_revalidation
TWOfficialIncrementalTests.test_flag_off_carries_valid_receipt_without_quarantine
TWOfficialIncrementalTests.test_lineage_rejects_malformed_tampered_cross_symbol_or_zero_row_receipt
```

The order test executes all six permutations through a pipeline adapter whose `get_data()` requests price, institutional, and margin in the selected order with fixed dataset-specific subranges, merges the same source rows, and then calls the real `build_stock_snapshot()`. For each permutation it passes that run's final `result["daily"]` to `lineage_for`, canonical-JSON encodes that `daily` and `daily_history_recovery`, and asserts one unique byte string for each across all six permutations. It also asserts one resolver call per symbol in every permutation.

The direct-rerun integration test writes an A0 artifact carrying a valid direct reconciliation whose manifest `entry.new_sha256 == A0.compressed_sha256`, runs real opt-in resolution/finalization, and asserts the A1 lineage has no `legacy_reconciliation` and exactly one valid history envelope with `reconciled_artifact_sha256 == A0.compressed_sha256`. It atomically writes and reloads A1, constructs a fresh opt-in resolver, reruns through the historical binding, and requires byte-identical `daily_history_recovery` plus no duplicate daily rows. A finalizer-only synthetic result is not sufficient evidence.

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovery_resolver_is_optional_and_not_called_by_default tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovery_resolver_is_called_once_for_all_dataset_orders tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovered_daily_rows_filter_each_requested_range_after_cache tests.test_tw_incremental.TWOfficialIncrementalTests.test_dataset_call_orders_are_byte_identical tests.test_tw_incremental.TWOfficialIncrementalTests.test_receipt_hashes_only_restored_rows_in_final_persisted_daily tests.test_tw_incremental.TWOfficialIncrementalTests.test_receipt_hashes_final_persisted_source_projection_not_backup_candidate tests.test_tw_incremental.TWOfficialIncrementalTests.test_zero_retained_rows_without_receipt_returns_none tests.test_tw_incremental.TWOfficialIncrementalTests.test_opt_in_rerun_rebinds_existing_receipt_to_current_verified_backup tests.test_tw_incremental.TWOfficialIncrementalTests.test_existing_receipt_revalidation_uses_current_persisted_source_projection tests.test_tw_incremental.TWOfficialIncrementalTests.test_direct_recovery_promotes_reconciliation_to_history_and_artifact_rerun_is_byte_identical tests.test_tw_incremental.TWOfficialIncrementalTests.test_existing_receipt_rebind_allows_only_retention_aged_rows_to_be_absent tests.test_tw_incremental.TWOfficialIncrementalTests.test_changed_parseable_manifest_entry_fails_existing_receipt_revalidation tests.test_tw_incremental.TWOfficialIncrementalTests.test_flag_off_carries_valid_receipt_without_quarantine tests.test_tw_incremental.TWOfficialIncrementalTests.test_lineage_rejects_malformed_tampered_cross_symbol_or_zero_row_receipt -v
```

Expected RED: `OfficialCompatFetcher.__init__` rejects `recovery_resolver`, `lineage_for` rejects `persisted_daily`, and receipt finalization is absent. Test fixtures use the Task 5 result type exactly.

- [ ] **Step 3 — Implementation: cache before filtering and finalize after persistence**

```python
# __init__
self.recovery_resolver = recovery_resolver
self._history_recovery: dict[str, HistoryRecoveryResult | None] = {}

def _ensure_history_recovery(self, symbol):
    if symbol not in self._history_recovery:
        artifact = self._load_artifact(symbol)
        self._history_recovery[symbol] = (
            None
            if self.recovery_resolver is None
            else self.recovery_resolver(symbol, artifact)
        )
    return self._history_recovery[symbol]

# _daily_rows
artifact = self._load_artifact(symbol)
recovery = self._ensure_history_recovery(symbol)
daily = (
    recovery.merged_daily
    if recovery is not None
    else artifact.document["daily"]
)
rows = [
    dict(item, _date=_parse_date(item.get("Date")))
    for item in daily
    if start <= _parse_date(item.get("Date")) <= end
]
```

Import `CALCULATED_COLUMNS` from `stock_papi.quant.features`, define `RECOVERY_DERIVED_FIELDS` exactly as frozen, and implement `_canonical_recovery_source_row()` by copying the row, parsing `Date` with the existing strict date helper, replacing it with `date.isoformat()`, and excluding derived fields plus private keys beginning with `_`. The helper preserves every remaining source key/value; callers compare the complete resulting mappings, not OHLCV alone.

`_finalize_daily_history_recovery()` first validates final `persisted_daily` as strictly ordered, unique canonical rows and normalizes each `Date` to ISO text. With no existing receipt, it takes the authorized dates from `restored_candidates`, selects those dates from final `persisted_daily`, projects each selected persisted row through `_canonical_recovery_source_row()`, and requires exact equality with the same projection of the verified candidate row. It hashes only those ordered final persisted projections. Zero retained authorized dates returns `None`; otherwise it emits the exact 16-field receipt and canonical hashes. It uses the existing `OfficialCompatFetcher._canonical_json_sha256()` encoding for manifest entry, projected persisted rows, and receipt hashes.

With an existing receipt, it recomputes and requires:

```text
sha256(canonical_json(backup_manifest_entry)) == backup_manifest_entry_sha256
backup_manifest_entry.original_sha256 == original_artifact_sha256
backup_manifest_entry.new_sha256 == expected_result_sha256
backup_target_market_date == receipt.backup_target_market_date
backup_series_manifest_sha256 == receipt.backup_series_manifest_sha256
backup_manifest_entry.original_size == receipt.backup_object_size
backup_manifest_entry.original_uncompressed_size == receipt.backup_object_uncompressed_size
receipt.symbol == requested symbol
receipt.input_artifact_sha256 is a lowercase 64-hex SHA retained as historical provenance, not rebound to the current active artifact
ordered authorized dates come from verified backup_daily rows inside the receipt range
canonical hash/count/start/end of all verified backup_daily source projections in the original receipt range == receipt restored fields
each receipt-range row still present in final persisted_daily has a complete source projection equal to its verified backup-row projection
any receipt-range date absent from final persisted_daily is strictly earlier than min(final persisted_daily dates); absence inside the current retained window fails
sha256(canonical_json(receipt without receipt_sha256)) == receipt.receipt_sha256
```

This permits only rows aged out by the later normal retention floor; it does not rewrite the historical receipt for a later target. The finalizer returns a new plain `dict` with unchanged fields only when every comparison passes. `_valid_official_lineage()` validates the exact optional receipt fields and reconciliation cross-binding without filesystem access. `_lineage_kind()` caches a valid existing receipt. `lineage_for(symbol, persisted_daily=final_daily)` always calls `_ensure_history_recovery(symbol)` before choosing the finalizer or carry-forward path, so receipt creation is independent of whether any dataset method ran first; when recovery ran it calls `_finalize_daily_history_recovery(result, symbol=symbol, recovery_target_market_date=self.target_date, persisted_daily=final_daily)`, otherwise it carries the cached valid receipt unchanged.

When that call returns a non-`None` receipt and `self._existing_reconciliations.get(symbol)` is a direct record, `lineage_for()` performs this exact inline rotation before attaching the receipt:

```python
direct = self._existing_reconciliations[symbol]
if (
    self._existing_reconciliation_history.get(symbol)
    or result.expected_result_sha256 != result.input_artifact_sha256
    or direct["legacy_artifact_sha256"] != result.original_artifact_sha256
    or _datetime.date.fromisoformat(
        direct["official_snapshot_dates"][-1]
    ) != result.backup_target_market_date
    or direct["official_series_manifest_sha256"]
    != result.backup_series_manifest_sha256
):
    raise IncrementalHistoryError(
        "daily history recovery direct binding is invalid"
    )
history_item = {
    "schema_version": 2,
    "symbol": symbol,
    "reconciled_artifact_sha256": result.input_artifact_sha256,
    "reconciliation": copy.deepcopy(direct),
}
history_item["history_sha256"] = self._canonical_json_sha256(history_item)
lineage.pop("legacy_reconciliation", None)
lineage["legacy_reconciliation_history"] = [history_item]
lineage["daily_history_recovery"] = dict(receipt)
```

The direct record has already passed `_valid_official_lineage`; the extra comparisons bind the selected verified result to that same direct record before rotation. A historical recovery leaves the validated history array unchanged and only attaches the receipt. A later flag-off run validates and carries both the history and receipt without quarantine access.

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_persisted_daily_reader_audit -v
```

Expected GREEN: one resolution per symbol, six call orders with identical bytes, per-dataset range filtering, post-retention receipt fields, no zero-row receipt, changed parseable entry failure, first-write direct-to-history rotation, second opt-in historical rebinding with byte-identical receipt, and flag-off carry-forward with a patched quarantine accessor that raises if called.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/quant/tw_incremental.py tests/test_tw_incremental.py tests/test_tw_legacy_reconciliation.py tests/test_persisted_daily_reader_audit.py
git commit -m "feat: finalize verified TW recovery receipts"
```

### Task 7: Wire explicit recovery and preserve checkpoint semantics

**Files:**

- Modify: `stock_papi/batch/tw_official_post_close_cli.py` in `_enrich_batch_identity`, `_patched_pipeline`, `_run_stage`, `run`, and `main`
- Modify: `tests/test_tw_official_post_close_cli.py`
- Modify: `tests/test_local_quant_batch.py`
- Modify: `tests/test_persisted_daily_reader_audit.py` to classify the CLI lineage wrapper

**Interfaces:**

- Consumes `resolve_truncated_daily_history(root, symbol, artifact)` and the frozen `OfficialCompatFetcher` constructor with `recovery_resolver`.
- Produces `--recover-truncated-history`, `run(recover_truncated_history=False)`, and checkpoint identity field `recover_truncated_history`.
- Preserves `run_market_batch()` production code and its existing generic/provider exception paths.

- [ ] **Step 1 — RED: write CLI, combined-flags, checkpoint, and publication tests**

```text
TWOfficialPostCloseCLITests.test_cli_recovery_flag_is_explicit_opt_in
TWOfficialPostCloseCLITests.test_cli_default_path_never_constructs_resolver_or_touches_quarantine
TWOfficialPostCloseCLITests.test_cli_wires_recovery_resolver_only_when_enabled
TWOfficialPostCloseCLITests.test_cli_checkpoint_identity_rejects_changed_recovery_mode
TWOfficialPostCloseCLITests.test_both_recovery_and_reconcile_flags_keep_legacy_artifact_in_existing_reconciliation_flow
TWOfficialPostCloseCLITests.test_recovery_failure_blocks_assert_complete_and_publication
LocalQuantBatchTests.test_recovery_failure_advances_cursor_without_overwriting_artifact
LocalQuantBatchTests.test_recovery_failure_resume_retries_failed_before_new
```

The combined-flags test writes both missing-lineage and explicit-`null`-lineage legacy artifacts, enables both flags, asserts recovery returns `None`, and proves existing `replace_verified_legacy` reconciliation writes the expected official overlap. Failure tests snapshot artifact bytes, checkpoint failures, `next_index`, and call order.

- [ ] **Step 2 — RED command and expected evidence**

```powershell
& $PythonExe -m unittest tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_recovery_flag_is_explicit_opt_in tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_default_path_never_constructs_resolver_or_touches_quarantine tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_wires_recovery_resolver_only_when_enabled tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_checkpoint_identity_rejects_changed_recovery_mode tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_both_recovery_and_reconcile_flags_keep_legacy_artifact_in_existing_reconciliation_flow tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_recovery_failure_blocks_assert_complete_and_publication tests.test_local_quant_batch.LocalQuantBatchTests.test_recovery_failure_advances_cursor_without_overwriting_artifact tests.test_local_quant_batch.LocalQuantBatchTests.test_recovery_failure_resume_retries_failed_before_new -v
```

Expected RED: CLI/signature/identity/combined-flag tests fail because the recovery option is absent. The two `LocalQuantBatchTests` characterize the current generic failure behavior and may already pass; they must not motivate a `run_market_batch` edit.

- [ ] **Step 3 — Implementation: pass the opt-in through existing boundaries**

```python
result["recover_truncated_history"] = recover_truncated_history
parser.add_argument("--recover-truncated-history", action="store_true")

recovery_resolver = None
if recover_truncated_history:
    recovery_resolver = lambda symbol, artifact: (
        resolve_truncated_daily_history(root, symbol, artifact)
    )
fetcher = OfficialCompatFetcher(
    root,
    series,
    pd=pipeline.pd,
    legacy_overlap_policy=policy,
    recovery_resolver=recovery_resolver,
)
```

Require `type(recover_truncated_history) is bool`; pass the value through every `_run_stage`, `_patched_pipeline`, and `_enrich_batch_identity` call, including false. Do not construct a recovery `LegacyArtifactBackupStore` in the CLI; `backup_store` remains only for reconciliation writes. After `original_build()` returns:

```python
result["source_lineage"] = fetcher.lineage_for(
    str(symbol), persisted_daily=result["daily"]
)
```

Any resolver/finalizer error propagates to the existing generic per-symbol path before `write_stock_artifact`; the failed artifact remains byte-identical. `_assert_complete` remains unchanged.

Extend the inventory for the wrapper's new direct `result["daily"]` access:

```python
READER_CONTRACTS.update({
    (
        "stock_papi/batch/tw_official_post_close_cli.py",
        "_patched_pipeline.build_stock_snapshot_with_lineage",
    ): "canonical-OHLCV",
})
```

- [ ] **Step 4 — GREEN command and expected result**

```powershell
& $PythonExe -m unittest tests.test_tw_official_post_close_cli tests.test_local_quant_batch tests.test_tw_incremental.TWOfficialIncrementalTests.test_status_fetcher_preserves_history_and_exposes_target_evidence tests.test_tw_incremental.TWLegacyOverlapReconciliationTests.test_official_lineage_allows_symbol_history_after_series_start tests.test_tw_trading_status -v
```

Expected GREEN: opt-in/default isolation, both flags on legacy artifacts, identity mismatch, unchanged failed artifact, new-symbol cursor advance, failed-first resume, active-failure publication block, regular/status dates, reconciliation, ETF, short history, and reused-symbol lifecycle all pass.

- [ ] **Step 5 — Commit**

```powershell
git add stock_papi/batch/tw_official_post_close_cli.py tests/test_tw_official_post_close_cli.py tests/test_local_quant_batch.py tests/test_persisted_daily_reader_audit.py
git commit -m "feat: add opt-in TW history recovery"
```

### Task 8: Verify, independently review, push, and prepare the Draft PR handoff

**Files:**

- Verify only: every file in the responsibility map
- Modify during findings repair only: files already mapped to Tasks 1-7
- Never modify production data, deployment configuration, tasks, pointers, or cloud resources

**Interfaces:**

- Consumes the seven reviewed implementation commits.
- Produces fresh local verification, independent review evidence, a pushed branch whose remote SHA matches local HEAD, and one Draft PR URL. It performs no merge or production operation.

- [ ] **Step 1 — RED gate: rerun inventory and focused contracts before handoff**

```powershell
& $PythonExe -m unittest tests.test_persisted_daily_reader_audit tests.test_tw_daily_history_preservation tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_local_quant_batch tests.test_tw_trading_status -v
```

Expected pre-handoff gate: zero failures/errors. Any unmatched reader, skipped contract test, or failing test is RED for Task 8 and blocks every later step.

- [ ] **Step 2 — Implementation gate: run full and language-level validation**

```powershell
& $PythonExe -m unittest discover -s tests -v
& $PythonExe -m compileall -q local_quant.py stock_papi reporting tests
node --check static/app.js
powershell -NoProfile -Command '& { $parseErrors = @(); foreach ($path in @("scripts/python_runtime.ps1", "scripts/run_local_quant_task.ps1", "scripts/run_tw_post_close_pipeline.ps1", "scripts/invoke_pipeline_task.ps1")) { $tokens = $null; $errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $path), [ref]$tokens, [ref]$errors) | Out-Null; $parseErrors += $errors }; if ($parseErrors.Count) { $parseErrors | Format-List; exit 1 } }'
```

Expected: every command exits `0`. Report each environment-only skip by exact test and reason; a skip does not satisfy a required contract.

- [ ] **Step 3 — GREEN gate: prove scope, simulation, and Markdown/code hygiene**

```powershell
& $PythonExe -m unittest tests.test_tw_daily_history_preservation.TWHistoryPersistenceTests.test_multistage_history_does_not_erode_and_rerun_is_byte_stable -v
git diff --check 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git diff --name-only 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git status --short
rg -n 'rglob\(|canonical_daily|source_daily' local_quant.py stock_papi tests
```

Expected GREEN: the five-stage simulation passes in a temporary directory; diff check is empty; changed names equal the responsibility map plus this committed plan document; status is clean; no production `rglob(` or extra daily-history field exists. A test-only `Path.rglob("*.py")` in the AST inventory is permitted because it scans repository source, not recovery storage.

- [ ] **Step 4 — Independent review and findings repair**

```powershell
$reviewLog = Join-Path $env:TEMP "agy-tw-daily-history-preservation-review.log"
agy --sandbox --print "Read-only review of this branch against design commit 0d2293d6fa8fb61a740a949f8ad084c24a266a2c. Inspect repository APIs, canonical/calculated ordering, all persisted daily readers, recovery eligibility, exact manifest/object single-read binding, direct and historical SHA binding, receipt retention and rerun revalidation, checkpoint/resume, runtime quarantine isolation, status/lifecycle, ETF, short history, tests, and scope. Report Critical or Important findings; do not modify files." 2>&1 | Tee-Object -FilePath $reviewLog
Get-Content -Raw -Encoding utf8 $reviewLog
```

Expected: non-empty review output and no Critical or Important finding. Empty output or reviewer/tool failure blocks push. If a finding appears, continue to Step 5 before rerunning any clean-worktree gate; if none appears, skip Step 5 and continue to Step 6.

- [ ] **Step 5 — Commit final review repairs when and only when needed**

```powershell
& $PythonExe -m unittest tests.test_persisted_daily_reader_audit tests.test_tw_daily_history_preservation tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_local_quant_batch tests.test_tw_trading_status -v
git diff --check
git add -- local_quant.py stock_papi/quant/features.py stock_papi/quant/tw_incremental.py stock_papi/quant/tw_legacy_reconciliation.py stock_papi/batch/tw_official_post_close_cli.py stock_papi/services/stock_analysis.py tests/report_fixtures.py tests/test_tw_daily_history_preservation.py tests/test_persisted_daily_reader_audit.py tests/test_stock_analysis.py tests/test_oos_diagnostics.py tests/test_local_quant_batch.py tests/test_tw_legacy_reconciliation.py tests/test_tw_incremental.py tests/test_tw_official_post_close_cli.py tests/test_daily_report_source.py tests/test_quant_snapshot_repository.py tests/test_observation_views.py tests/test_observation_products.py tests/test_industry_report_analytics.py tests/test_industry_report_backtest.py tests/test_pit_dataset.py
git diff --cached --name-only
git commit -m "fix: address final TW history review"
```

Expected: this step is omitted when Step 4 has no finding. When used, repair only mapped files, require the complete focused contract command and diff check to pass, verify the staged-name list is a subset of the responsibility map, and create the focused repair commit. Then rerun Steps 1-4 from the beginning with a clean worktree. Any new finding returns to Step 5; push remains blocked until a post-commit Steps 1-4 rerun is fully GREEN with no Critical or Important finding.

- [ ] **Step 6 — Push and verify exact remote identity**

```powershell
git push origin codex/tw-daily-history-preservation
git fetch origin codex/tw-daily-history-preservation
git rev-parse HEAD
git rev-parse refs/remotes/origin/codex/tw-daily-history-preservation
git status --short
```

Expected: push/fetch succeed, both full SHAs are identical, and worktree status is empty.

- [ ] **Step 7 — Create one Draft PR and stop**

```powershell
gh pr create --draft --base main --head codex/tw-daily-history-preservation --title "fix: preserve TW daily history" --body "Implements approved design 0d2293d6fa8fb61a740a949f8ad084c24a266a2c with canonical daily preservation, complete persisted-reader compatibility gates, and explicit manifest-bound recovery. Verification used only temporary data. No production recovery, publication, deployment, merge, or production operation was performed."
```

Expected: one Draft PR URL. Do not merge, deploy, publish, access production storage, or perform recovery.

- [ ] **Step 8 — Commit accounting and terminal handoff**

```powershell
git log --oneline 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git diff --check 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git status --short
```

Expected: each task has its own reviewable commit, the optional review-repair commit is present only if needed, diff check is empty, worktree is clean, and execution stops for review.
