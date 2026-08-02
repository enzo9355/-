# TW Daily History Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the complete canonical TW `daily` history across repeated official post-close updates while keeping calculated rows separate and permitting only explicit, manifest-bound recovery of already-truncated reconciled artifacts.

**Architecture:** `build_stock_snapshot()` keeps one canonical frame and one calculated copy, then joins only approved derived columns back by market date. `OfficialCompatFetcher` accepts an optional cached recovery resolver; the resolver reuses `LegacyArtifactBackupStore` for exact read-only backup validation and returns merged rows plus one deterministic lineage receipt. The CLI constructs that resolver only for `--recover-truncated-history`, binds the mode into checkpoint identity, and otherwise never touches quarantine.

**Tech Stack:** Repository Python runtime, pandas, NumPy, stdlib `unittest`, JSON, gzip, SHA-256, `pathlib`, PowerShell parser checks, Node syntax checks, Git, GitHub CLI, and the existing `agy` read-only review tool; no new dependency.

## Global Constraints

- Approved design commit: `0d2293d6fa8fb61a740a949f8ad084c24a266a2c` on `codex/tw-daily-history-preservation` in the existing isolated worktree.
- `daily` remains the single canonical persisted history; do not add `canonical_daily`, `source_daily`, or another daily-history field.
- Preserve every canonical row returned inside the existing `get_data(symbol, 730)` request window, including indicator warm-up rows; do not introduce unlimited retention.
- The canonical frame is the target-date-filtered input before calculation. Normalize dates to market dates, reject duplicate dates, and sort strictly ascending.
- Same-date canonical `Open`, `High`, `Low`, `Close`, and `Volume` are immutable across reruns. Conflicting duplicate-date OHLCV fails closed.
- `rows == len(daily)`, `latest == daily[-1]`, and `as_of` remains the latest regular-price date.
- For regular price, `as_of == latest_regular_price_date == target_market_date == observation_as_of`.
- For `official_no_regular_trade` and `officially_suspended`, `target_market_date == observation_as_of` and `as_of == latest_regular_price_date < target_market_date`; status sessions never create or relabel a target-date price row.
- Preserve status evidence, evidence SHA-256, official lifecycle precedence, direct reconciliation, reconciliation history, and reused-symbol semantics.
- `calc_all()` receives a copy. Analysis, inference, and backtesting use the calculated frame; persistence never uses `dropna()` as a row-retention policy.
- Clear calculated fields on the canonical frame, then join calculated values only by matching date. Warm-up indicators serialize as JSON `null`; canonical OHLCV and source fields remain present.
- Calculated data may not overwrite canonical `Date`, OHLCV, institutional, margin, short, market, option, or data-quality source values.
- The latest canonical regular-price date must exist in the fully calculated frame; otherwise retain `ValueError("calculated history is unavailable")`.
- Observation-only mode removes its existing model columns before the date join without removing canonical rows.
- Re-running the same target date must preserve one row per date, stable historical OHLCV, and one non-duplicated deterministic receipt.
- No synthetic prices, forward-fill, interpolation, or relabeling of an old close as a target-session close.
- No change to rolling formulas, model features, LightGBM, backtests, prediction targets, or recommendation policy.
- Retaining the current artifact schema version is allowed only after a refreshed repository-wide persisted-`daily` reader audit classifies every reader and compatibility tests prove warm-up indicator `null` values are safe. If the proof fails, stop and revise the schema decision before continuing.
- Every feature-ready persisted-history consumer filters its exact required derived fields at its own boundary; persisted warm-up rows are never deleted to satisfy a consumer.
- `--recover-truncated-history` and the matching `run(..., recover_truncated_history=False)` argument are independent explicit opt-ins and default to false.
- With recovery disabled, do not construct a resolver or resolve, open, list, probe, glob, or scan any quarantine path. `OfficialCompatFetcher._daily_rows()` reads only the active artifact.
- Never use `rglob()`, recursive search, filename search, or first-match selection for recovery.
- Recovery authority comes only from a valid direct `legacy_reconciliation` or validated `legacy_reconciliation_history` item under valid official lineage.
- Backup target date is exactly `reconciliation.official_snapshot_dates[-1]`; the exact directory is `<root>/quarantine/tw-recovery/legacy-reconciliation/v2/<target-date>/<official_series_manifest_sha256>/manifest.json`.
- Require a complete schema-v2 manifest, an `applied` symbol entry, exact symbol, `entry.original_sha256 == reconciliation.legacy_artifact_sha256`, `entry.backup_path == objects/<original_sha256>.json.gz`, compressed size, SHA-256, bounded uncompressed size, gzip validity, safe child paths, and no symlink or Windows reparse-point component.
- Historical recovery additionally requires `manifest_entry.new_sha256 == history_item.reconciled_artifact_sha256`; direct recovery requires `manifest_entry.new_sha256 == active_artifact.compressed_sha256`.
- The decoded backup object must match `TW:<symbol>` and contain valid ordered unique daily dates whose declared date fields agree with the rows.
- Zero or multiple distinct qualifying backup objects fail closed. Missing manifest/object, changed bytes, hash/size/gzip/path/symbol/result binding mismatch, duplicate input dates, or overlap OHLCV conflict fails closed with no fallback source.
- On matching overlap OHLCV, the current row wins as a whole; backup-only earlier rows may restore the prefix; current-only rows remain unchanged.
- Apply the normal target-date and 730-day range after merge. Receipt start/end/count and `restored_daily_sha256` cover only backup-only rows actually retained and restored after that filter.
- A zero-added-row rerun with an existing valid matching receipt retains it unchanged. A zero-added-row run without a prior valid matching receipt writes no receipt; a zero-row receipt is invalid.
- A normal later run carries a valid receipt without reading quarantine. An opt-in rerun revalidates the exact manifest and backup before accepting a no-op.
- Recovery errors use the existing generic `run_market_batch` failure path: record the symbol, do not overwrite its artifact, allow `next_index` to advance for a new-symbol failure, retry failed symbols before new symbols on resume, and let `_assert_complete` block publication while an active failure remains.
- Do not change provider-specific fail-fast checkpoint semantics and do not add `local_quant.run_market_batch` to the production implementation boundary.
- Do not weaken official source, lifecycle, reconciliation, artifact-size, gzip-expansion, SHA-256, path, symlink, reparse-point, schema, or publication validation.
- Implementation and every test fixture use only temporary directories inside the worktree or OS temp area. Do not access `D:\AbsorbData` and do not interact with PID 17820.
- Do not perform live recovery, GCS, Cloud Run, Scheduled Tasks, publication, production pointer mutation, LINE delivery, merge, or any production operation.
- The future implementation pull request must contain no production recovery or publication operation. Draft PR creation is the terminal planned handoff; merge remains prohibited.

---

## File Responsibility Map

### Production files to modify during later execution

| File | Responsibility |
| --- | --- |
| `stock_papi/quant/features.py` | Export the exact ordered `CALCULATED_COLUMNS` tuple assigned by `calc_all()`; formulas remain unchanged. |
| `local_quant.py` | Keep canonical and calculated frames separate, validate canonical dates, clear/join derived columns by date, and serialize the full canonical frame. Existing checkpoint control flow remains unchanged. |
| `stock_papi/services/stock_analysis.py` | Filter the exact required feature columns before its minimum-history and latest-row analysis boundary. |
| `stock_papi/quant/tw_legacy_reconciliation.py` | Read one exact manifest-bound original backup without mutation; resolve direct/historical recovery; merge rows; apply the caller's normal date range; build deterministic receipts. |
| `stock_papi/quant/tw_incremental.py` | Define the resolver callable contract, invoke it at most once per symbol, cache recovered rows/receipt, validate and carry receipts, and keep the normal path independent of quarantine. |
| `stock_papi/batch/tw_official_post_close_cli.py` | Add the opt-in flag and programmatic argument, construct/wire the resolver only when enabled, bind recovery mode into batch identity, and retain `_assert_complete` publication gating. |

### Test files to create

| File | Responsibility |
| --- | --- |
| `tests/test_tw_daily_history_preservation.py` | Calculated-column contract, canonical/calculated join, warm-up preservation, date uniqueness, ETF/short-history cases, and the required multi-stage rerun simulation. |
| `tests/test_stock_analysis.py` | Feature-ready persisted-history filtering at `snapshot_dataframe()`. |
| `tests/test_oos_diagnostics.py` | Research enrichment behavior when historical derived market fields are `null` but canonical liquidity fields remain valid. |

### Test files to modify

| File | Responsibility |
| --- | --- |
| `tests/report_fixtures.py` | Provide a deterministic persisted snapshot fixture with valid OHLCV and configurable warm-up indicator `null` rows. |
| `tests/test_local_quant_batch.py` | Snapshot serialization, status semantics, checkpoint advance/retry ordering, failed artifact preservation, ETF, and short-history gates. |
| `tests/test_tw_legacy_reconciliation.py` | Exact backup reader, direct/historical result SHA binding, exact target path, merge conflict, retention, zero-row receipt, missing/ambiguous backup, and idempotency. |
| `tests/test_tw_incremental.py` | Optional resolver call/cache contract plus receipt validation and carry-forward without quarantine. |
| `tests/test_tw_official_post_close_cli.py` | CLI flag/default isolation, resolver wiring, checkpoint identity mismatch, `_assert_complete`, status/lifecycle, and publication blocking. |
| `tests/test_daily_report_source.py` | Reporting source loader accepts and preserves warm-up rows with `null` indicators. |
| `tests/test_quant_snapshot_repository.py` | Hash-bound quant loader accepts warm-up `null` indicators and keeps the latest row ready. |
| `tests/test_observation_views.py` | Canonical candles survive while the historical MA20 series filters unavailable values. |
| `tests/test_observation_products.py` | Dashboard and market aggregation use canonical price history and latest-only indicators correctly. |
| `tests/test_industry_report_analytics.py` | Report generation explicitly ignores unavailable historical derived fields. |
| `tests/test_industry_report_backtest.py` | Backtest signal selection filters unavailable features without deleting the canonical price calendar. |
| `tests/test_pit_dataset.py` | PIT price/volume research rows remain eligible when unrelated indicators are `null`. |

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

def _persisted_history_frame(
    canonical_frame,
    calculated_frame,
    *,
    excluded_columns=(),
):
    """Clear/join derived values by index without replacing canonical source fields."""
```

```python
# stock_papi/quant/tw_incremental.py
HistoryRecoveryResult = tuple[list[dict[str, Any]], dict[str, Any] | None]
HistoryRecoveryResolver = Callable[
    [
        str,
        IncrementalArtifact,
        _datetime.date,
        _datetime.date,
        _datetime.date,
    ],
    HistoryRecoveryResult | None,
]

OfficialCompatFetcher.__init__(
    self,
    root: Path,
    source: Any,
    *,
    pd: Any,
    legacy_overlap_policy: str = "strict",
    recovery_resolver: HistoryRecoveryResolver | None = None,
) -> None
```

`HistoryRecoveryResult[0]` is the complete validated merged daily list; the fetcher applies the requested `start`/`end` range when serving each dataset. `HistoryRecoveryResult[1]` is one valid positive-row receipt, an unchanged prior valid receipt, or `None` for a verified zero-added-row no-op without a prior receipt.

```python
DAILY_HISTORY_RECOVERY_FIELDS = frozenset({
    "schema_version",
    "mode",
    "symbol",
    "recovery_target_market_date",
    "input_artifact_sha256",
    "original_artifact_sha256",
    "backup_target_market_date",
    "backup_series_manifest_sha256",
    "backup_manifest_entry_sha256",
    "backup_object_size",
    "backup_object_uncompressed_size",
    "restored_start_date",
    "restored_end_date",
    "restored_row_count",
    "restored_daily_sha256",
    "receipt_sha256",
})
```

```python
# stock_papi/quant/tw_legacy_reconciliation.py
class LegacyArtifactBackupStore:
    def read_original_document(
        self,
        *,
        symbol: str,
        original_sha256: str,
        expected_result_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a validated decoded original and a copied applied manifest entry."""

def resolve_truncated_daily_history(
    root: Path,
    symbol: str,
    artifact: IncrementalArtifact,
    start: datetime.date,
    end: datetime.date,
    recovery_target_market_date: datetime.date,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None] | None:
    """Resolve exactly one lineage-bound backup, merge it, and build a receipt."""
```

```python
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

The checkpoint identity always contains `"recover_truncated_history": false|true`. A checkpoint created under the other value is a different batch and cannot resume under changed recovery semantics.

### Task 1: Freeze the calculated-column contract

**Files:**

- Modify: `stock_papi/quant/features.py:11-62`
- Modify: `tests/test_tw_daily_history_preservation.py`

**Interfaces:**

- Produces: `CALCULATED_COLUMNS: tuple[str, ...]` exactly as frozen above.
- Consumed by: Task 2 fixtures and Task 3 persistence join.

- [ ] **Step 1: Write the failing contract test**

```python
class TWCalculatedColumnContractTests(unittest.TestCase):
    def test_calculated_columns_match_calc_all_assignments_in_order(self):
        from stock_papi.quant import features
        assigned = []
        tree = ast.parse(inspect.getsource(features.calc_all))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
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

- [ ] **Step 2: Run the test and record RED**

```powershell
python -m unittest tests.test_tw_daily_history_preservation.TWCalculatedColumnContractTests.test_calculated_columns_match_calc_all_assignments_in_order -v
```

Expected RED: `AttributeError` because `CALCULATED_COLUMNS` does not exist.

- [ ] **Step 3: Add the exact tuple from Frozen Interfaces above**

```python
CALCULATED_COLUMNS = (
    "MA_5", "MA20", "RET_1", "RET_5", "RET_20", "RANGE_PCT",
    "VOL_RATIO", "VOL_CHG", "INST_NET_RATIO", "MARGIN_CHG",
    "SHORT_CHG", "RSI", "Volat", "MACD_DIF", "MACD",
    "MACD_OSC", "K", "D", "BB_UP", "BB_DN",
)
```

Do not change any formula or the final `frame.dropna()`.

- [ ] **Step 4: Run focused GREEN**

```powershell
python -m unittest tests.test_tw_daily_history_preservation.TWCalculatedColumnContractTests.test_calculated_columns_match_calc_all_assignments_in_order -v
```

Expected GREEN: one passing test with the exact 20-column order.

- [ ] **Step 5: Commit**

```powershell
git add stock_papi/quant/features.py tests/test_tw_daily_history_preservation.py
git commit -m "test: freeze TW calculated column contract"
```

### Task 2: Audit persisted-daily consumers and gate the schema decision

**Files:**

- Modify: `stock_papi/services/stock_analysis.py:7-18`
- Modify: `tests/report_fixtures.py`
- Create: `tests/test_stock_analysis.py`
- Create: `tests/test_oos_diagnostics.py`
- Modify: `tests/test_daily_report_source.py`
- Modify: `tests/test_quant_snapshot_repository.py`
- Modify: `tests/test_observation_views.py`
- Modify: `tests/test_observation_products.py`
- Modify: `tests/test_industry_report_analytics.py`
- Modify: `tests/test_industry_report_backtest.py`
- Modify: `tests/test_pit_dataset.py`

**Interfaces:**

- Consumes: `CALCULATED_COLUMNS`.
- Produces: current-schema compatibility evidence and explicit required-field filtering in `snapshot_dataframe(snapshot, *, pd)`.

- [ ] **Step 1: Run the mandatory repository-wide inventory**

```powershell
rg -n --glob "*.py" 'document\["daily"\]|document\.get\("daily"\)|snapshot\["daily"\]|snapshot\.get\("daily"\)|daily\[-1\]|\.daily\b|load_incremental_artifact|StockSnapshot' local_quant.py stock_papi reporting scripts
```

Every match must map to this closed classification; an unmatched reader stops execution before any production edit:

| Classification | Exact reader boundaries |
| --- | --- |
| canonical-OHLCV | `tw_incremental`, `pit_dataset`, observation-view candles, observation-products price/returns, industry return/flow paths |
| latest-only | `_validated_artifact`, quant snapshot repository, artifact audit, legacy reconciliation state, `_assert_complete`, reporting loader/schemas/migration, observation-view headlines, observation-products indicators, daily-products/application callers |
| feature-ready-history | `stock_analysis.snapshot_dataframe`, observation-view MA20 series, industry historical market/model analytics, industry backtest signals, OOS market factors |

- [ ] **Step 2: Add one exact warm-up fixture and named tests**

```python
def stock_document_with_indicator_warmup(
    symbol: str,
    *,
    rows: int = 220,
    warmup_rows: int = 20,
    as_of: str = "2026-07-31",
) -> dict:
    document = stock_document(symbol, rows=rows, as_of=as_of)
    for index, row in enumerate(document["daily"]):
        close = float(row["Close"])
        row.update(Open=close - 0.5, High=close + 1.0,
                   Low=close - 1.0, Volume=1000 + index)
        if index < warmup_rows:
            for name in CALCULATED_COLUMNS:
                row[name] = None
            row["AI_P"] = None
    document["latest"] = dict(document["daily"][-1])
    return document
```

Add exactly:

```text
DailyReportSourceTests.test_loader_preserves_ohlcv_rows_with_null_warmup_indicators
QuantSnapshotRepositoryTests.test_repository_accepts_null_warmup_indicators_and_latest_remains_ready
ObservationViewTests.test_warmup_null_indicators_keep_candles_and_filter_ma20_line
ObservationProductsTests.test_market_aggregation_uses_ohlcv_with_null_warmup_indicators
IndustryReportAnalyticsTests.test_report_generation_filters_null_historical_features
IndustryReportBacktestTests.test_backtest_filters_null_signal_rows_without_dropping_price_calendar
PitDatasetTests.test_price_dataset_keeps_valid_ohlcv_when_indicators_are_null
StockAnalysisSnapshotTests.test_snapshot_dataframe_filters_required_feature_rows_before_minimum_history
OOSDiagnosticsTests.test_enrichment_filters_null_market_features_but_keeps_liquidity
```

Each test asserts early OHLCV is retained, latest required features remain finite, and historical derived calculations exclude nulls. The stock-analysis fixture has 220 rows/20 warm-up rows and must return exactly 200 complete rows.

- [ ] **Step 3: Run and record RED**

```powershell
python -m unittest tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected RED: the stock-analysis test fails because incomplete warm-up rows survive its boundary. Loaders must accept JSON null; any other failure is fixed only at its named consumer boundary.

- [ ] **Step 4: Apply the minimal filter**

```python
required = {
    "Open", "High", "Low", "Close", "MA20", "RSI", "Volat",
    "MACD_OSC", "K", "D", "AI_P", "ForeignNet",
}
if not required.issubset(frame.columns):
    return None
frame = frame.dropna(subset=sorted(required))
return frame if len(frame) >= 200 else None
```

Retain existing `finite_number`, `_number`, `dropna`, and non-null filters elsewhere. Do not delete persisted rows.

- [ ] **Step 5: Run focused GREEN and decide schema**

```powershell
python -m unittest tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected GREEN: all modules pass. Only then retain the current artifact schema; otherwise stop for a revised design.

- [ ] **Step 6: Commit**

```powershell
git add stock_papi/services/stock_analysis.py tests/report_fixtures.py tests/test_stock_analysis.py tests/test_oos_diagnostics.py tests/test_daily_report_source.py tests/test_quant_snapshot_repository.py tests/test_observation_views.py tests/test_observation_products.py tests/test_industry_report_analytics.py tests/test_industry_report_backtest.py tests/test_pit_dataset.py
git commit -m "test: prove persisted daily consumer compatibility"
```

### Task 3: Separate canonical persistence from calculated analysis

**Files:**

- Modify: `local_quant.py:1566-1722`
- Modify: `tests/test_local_quant_batch.py`
- Modify: `tests/test_tw_daily_history_preservation.py`

**Interfaces:**

- Consumes: `CALCULATED_COLUMNS`.
- Produces: `_canonical_history_frame(frame)` and `_persisted_history_frame(canonical_frame, calculated_frame, *, excluded_columns=())`.
- Preserves: the public `build_stock_snapshot(...)` signature and artifact schema.

- [ ] **Step 1: Write exact failing tests**

```text
LocalQuantBatchTests.test_taiwan_snapshot_persists_canonical_warmup_rows_and_joins_indicators_by_date
LocalQuantBatchTests.test_taiwan_snapshot_rejects_duplicate_canonical_market_dates
LocalQuantBatchTests.test_taiwan_snapshot_rejects_calculated_frame_missing_latest_canonical_date
LocalQuantBatchTests.test_taiwan_status_snapshot_preserves_full_history_and_last_regular_price_date
LocalQuantBatchTests.test_taiwan_etf_snapshot_preserves_canonical_history
LocalQuantBatchTests.test_taiwan_short_history_keeps_ohlcv_but_fails_when_latest_is_not_calculated
TWMultiStageDailyHistoryPreservationTests.test_baseline_through_20260731_rerun_never_erodes_history
```

The integration test uses a temporary root, real `calc_all`, `build_stock_snapshot`, `write_stock_artifact`, and `OfficialCompatFetcher` reloads:

Define the test-local method `build_and_reload_stage(self, root: Path, target: datetime.date) -> dict` before the test. For the baseline it builds the fixed 40-row OHLCV frame ending `2026-07-28`; for each later target it constructs the existing `OfficialSnapshotSeries` fixture for that target, installs `OfficialCompatFetcher` as the pipeline fetcher, calls `build_stock_snapshot(..., target_market_date=target, observation_only=True)`, writes with `write_stock_artifact()`, and returns the decoded artifact. On the repeated `2026-07-31` call it supplies the same official row. Capture `ohlcv_before_rerun` after the first `2026-07-31` stage and `ohlcv_after_rerun` after the repeated stage as ordered `(Date, Open, High, Low, Close, Volume)` tuples.

```python
stages = (
    ("baseline", date(2026, 7, 28), 40),
    ("2026-07-29", date(2026, 7, 29), 41),
    ("2026-07-30", date(2026, 7, 30), 42),
    ("2026-07-31", date(2026, 7, 31), 43),
    ("rerun-2026-07-31", date(2026, 7, 31), 43),
)
for label, target, expected_rows in stages:
    payload = build_and_reload_stage(root, target)
    self.assertEqual(payload["rows"], expected_rows, label)
    dates = [row["Date"][:10] for row in payload["daily"]]
    self.assertEqual(dates, sorted(set(dates)), label)
    self.assertTrue(all(payload["daily"][0][c] is None for c in CALCULATED_COLUMNS))
    self.assertTrue(all(payload["latest"][c] is not None for c in CALCULATED_COLUMNS))
self.assertEqual(ohlcv_before_rerun, ohlcv_after_rerun)
```

- [ ] **Step 2: Run and record RED**

```powershell
python -m unittest tests.test_tw_daily_history_preservation.TWMultiStageDailyHistoryPreservationTests tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_snapshot_persists_canonical_warmup_rows_and_joins_indicators_by_date tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_snapshot_rejects_duplicate_canonical_market_dates tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_snapshot_rejects_calculated_frame_missing_latest_canonical_date tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_status_snapshot_preserves_full_history_and_last_regular_price_date tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_etf_snapshot_preserves_canonical_history tests.test_local_quant_batch.LocalQuantBatchTests.test_taiwan_short_history_keeps_ohlcv_but_fails_when_latest_is_not_calculated -v
```

Expected RED: 40 rows persist as 20, later stages erode again, and duplicate/latest-calculated gates are absent.

- [ ] **Step 3: Implement the minimal frame helpers**

```python
def _canonical_history_frame(frame):
    result = frame.copy()
    normalized = result.index.normalize()
    if normalized.hasnans:
        raise ValueError("canonical history contains an invalid date")
    result.index = normalized
    if result.index.has_duplicates:
        raise ValueError("canonical history contains duplicate dates")
    return result.sort_index()


def _persisted_history_frame(canonical_frame, calculated_frame, *, excluded_columns=()):
    from stock_papi.quant.features import CALCULATED_COLUMNS
    excluded = frozenset(excluded_columns)
    result = canonical_frame.drop(columns=list(excluded), errors="ignore").copy()
    model = tuple(
        c for c in OBSERVATION_MODEL_COLUMNS
        if c in result.columns or c in calculated_frame.columns
    )
    derived = tuple(c for c in (*CALCULATED_COLUMNS, *model) if c not in excluded)
    if calculated_frame.index.has_duplicates:
        raise ValueError("calculated history contains duplicate dates")
    if len(calculated_frame.index.difference(result.index)):
        raise ValueError("calculated history contains an unknown date")
    for column in derived:
        result[column] = math.nan
    available = [c for c in derived if c in calculated_frame.columns]
    result.loc[calculated_frame.index, available] = calculated_frame.loc[:, available]
    return result
```

- [ ] **Step 4: Rewire snapshot construction**

```python
canonical_frame = _canonical_history_frame(frame)
calculated_frame = pipeline.calc_all(canonical_frame.copy())
if calculated_frame is None or calculated_frame.empty:
    raise ValueError("calculated history is unavailable")
if canonical_frame.index[-1] not in calculated_frame.index:
    raise ValueError("calculated history is unavailable")
analysis_frame = calculated_frame.drop(
    columns=list(OBSERVATION_MODEL_COLUMNS), errors="ignore"
) if observation_only else calculated_frame
persisted_frame = _persisted_history_frame(
    canonical_frame,
    analysis_frame,
    excluded_columns=(OBSERVATION_MODEL_COLUMNS if observation_only else ()),
)
```

Existing model/backtest calls receive `analysis_frame`; JSON serialization receives `persisted_frame`. Source columns, formulas, status, and backtest logic remain unchanged.

- [ ] **Step 5: Run focused GREEN**

```powershell
python -m unittest tests.test_tw_daily_history_preservation tests.test_local_quant_batch -v
```

Expected GREEN: stage counts are `40/41/42/43/43`; dates/OHLCV are stable; warm-up indicators are null; latest is complete; status, ETF, short-history, inference, and observation-only tests pass.

- [ ] **Step 6: Commit**

```powershell
git add local_quant.py tests/test_local_quant_batch.py tests/test_tw_daily_history_preservation.py
git commit -m "fix: preserve canonical TW daily history"


### Task 4: Add the exact read-only verified backup reader

**Files:**

- Modify: `stock_papi/quant/tw_legacy_reconciliation.py:516-522`
- Modify: `tests/test_tw_legacy_reconciliation.py`

**Interfaces:**

- Produces: `LegacyArtifactBackupStore.read_original_document(...)`.
- Consumed by: Task 5's resolver.

- [ ] **Step 1: Write exact failing tests**

```text
LegacyArtifactBackupStoreTests.test_read_original_document_requires_applied_exact_manifest_identity
LegacyArtifactBackupStoreTests.test_read_original_document_requires_original_and_result_sha_bindings
LegacyArtifactBackupStoreTests.test_read_original_document_rejects_hash_size_gzip_symbol_and_path_tampering
LegacyArtifactBackupStoreTests.test_read_original_document_is_read_only_and_does_not_scan
```

The success test asserts the returned document equals `{**legacy_document(), "schema_version": 1, "market": "TW", "symbol": "2330"}`, the entry is copied, manifest/object mtimes do not change, and patched `Path.rglob` is never called. Tamper subtests mutate `symbol`, `original_sha256`, `new_sha256`, both size fields, `backup_path`, gzip bytes, decoded market, and decoded symbol.

- [ ] **Step 2: Run and record RED**

```powershell
python -m unittest tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_read_original_document_requires_applied_exact_manifest_identity tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_read_original_document_requires_original_and_result_sha_bindings tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_read_original_document_rejects_hash_size_gzip_symbol_and_path_tampering tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_read_original_document_is_read_only_and_does_not_scan -v
```

Expected RED: four `AttributeError` errors because the reader is absent.

- [ ] **Step 3: Implement the read-only method**

```python
def read_original_document(
    self,
    *,
    symbol: str,
    original_sha256: str,
    expected_result_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = self._load_manifest(required=True)
    entry = self._validate_entry(symbol, manifest["entries"].get(symbol))
    if (
        entry["status"] != "applied"
        or entry["original_sha256"] != original_sha256
        or entry["new_sha256"] != expected_result_sha256
    ):
        raise LegacyReconciliationError(
            f"legacy reconciliation backup identity mismatch for TW:{symbol}"
        )
    self._verify_entry_object(entry)
    raw = _read_bytes(self.backup_root / entry["backup_path"])
    try:
        document = json.loads(_decode_gzip(raw).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise LegacyReconciliationError(
            f"legacy reconciliation backup document is invalid for TW:{symbol}"
        ) from exc
    if not _valid_original_document(document, symbol=symbol):
        raise LegacyReconciliationError(
            f"legacy reconciliation backup document is invalid for TW:{symbol}"
        )
    return document, dict(entry)
```

Define `_valid_original_document(document: Any, *, symbol: str) -> bool` in the same file. It accepts only market `TW`, exact symbol, non-empty dict rows, strictly ascending unique ISO dates, last row date equal to `as_of`, optional `rows == len(daily)`, and optional `latest.Date == daily[-1].Date`. It performs no write.

- [ ] **Step 4: Run focused GREEN**

```powershell
python -m unittest tests.test_tw_legacy_reconciliation -v
```

Expected GREEN: all new reader and existing backup/write/resume tests pass.

- [ ] **Step 5: Commit**

```powershell
git add stock_papi/quant/tw_legacy_reconciliation.py tests/test_tw_legacy_reconciliation.py
git commit -m "feat: read exact reconciliation backup"
```

### Task 5: Resolve, merge, retain, and receipt recovered history

**Files:**

- Modify: `stock_papi/quant/tw_legacy_reconciliation.py`
- Modify: `tests/test_tw_legacy_reconciliation.py`

**Interfaces:**

- Consumes: `IncrementalArtifact` and Task 4's reader.
- Produces: `resolve_truncated_daily_history(root, symbol, artifact, start, end, recovery_target_market_date)`.

- [ ] **Step 1: Write exact failing resolver tests**

```text
LegacyArtifactBackupStoreTests.test_resolver_binds_direct_result_sha_and_exact_snapshot_date
LegacyArtifactBackupStoreTests.test_resolver_binds_historical_result_sha_without_searching
LegacyArtifactBackupStoreTests.test_resolver_rejects_missing_or_multiple_distinct_backups
LegacyArtifactBackupStoreTests.test_resolver_rejects_duplicate_dates_and_overlap_ohlcv_conflict
LegacyArtifactBackupStoreTests.test_resolver_hashes_only_post_retention_restored_rows
LegacyArtifactBackupStoreTests.test_resolver_zero_added_rows_without_prior_receipt_returns_none_receipt
LegacyArtifactBackupStoreTests.test_resolver_revalidates_and_reuses_matching_receipt_idempotently
```

Direct fixtures bind `entry.new_sha256` to `artifact.compressed_sha256`; historical fixtures bind it to `history_item["reconciled_artifact_sha256"]`. Both assert the directory uses `official_snapshot_dates[-1]`. Retention removes two old backup rows and asserts the receipt hashes only remaining backup-only rows. Zero-row/no-prior-receipt returns `None` receipt.

- [ ] **Step 2: Run and record RED**

```powershell
python -m unittest tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_binds_direct_result_sha_and_exact_snapshot_date tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_binds_historical_result_sha_without_searching tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_rejects_missing_or_multiple_distinct_backups tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_rejects_duplicate_dates_and_overlap_ohlcv_conflict tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_hashes_only_post_retention_restored_rows tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_zero_added_rows_without_prior_receipt_returns_none_receipt tests.test_tw_legacy_reconciliation.LegacyArtifactBackupStoreTests.test_resolver_revalidates_and_reuses_matching_receipt_idempotently -v
```

Expected RED: import failure for `resolve_truncated_daily_history`.

- [ ] **Step 3: Implement candidate and merge helpers**

```python
def _recovery_candidates(lineage, artifact_sha256):
    direct = lineage.get("legacy_reconciliation")
    if isinstance(direct, dict):
        return [(direct, artifact_sha256)]
    return [
        (item["reconciliation"], item["reconciled_artifact_sha256"])
        for item in lineage.get("legacy_reconciliation_history", [])
    ]


def _merge_recovery_daily(active_daily, backup_daily):
    active = {_daily_date(row): dict(row) for row in active_daily}
    backup = {_daily_date(row): dict(row) for row in backup_daily}
    if len(active) != len(active_daily) or len(backup) != len(backup_daily):
        raise LegacyReconciliationError("daily history contains duplicate dates")
    for value in set(active) & set(backup):
        for name in ("Open", "High", "Low", "Close", "Volume"):
            if _finite_number(active[value].get(name)) != _finite_number(
                backup[value].get(name)
            ):
                raise LegacyReconciliationError("daily history OHLCV conflict")
    merged = {**backup, **active}
    return [merged[value] for value in sorted(merged)]
```

`_daily_date()` parses `row["Date"]` to `datetime.date` and rejects invalid values. `_finite_number()` rejects booleans, NaN, and infinity and returns `float`; neither helper synthesizes a value.

- [ ] **Step 4: Implement exact resolution and receipt**

```python
def resolve_truncated_daily_history(
    root,
    symbol,
    artifact,
    start,
    end,
    recovery_target_market_date,
):
    lineage = artifact.document.get("source_lineage")
    if not OfficialCompatFetcher._valid_official_lineage(lineage, artifact):
        raise LegacyReconciliationError(
            f"daily history recovery lineage is invalid for TW:{symbol}"
        )
    candidates = _recovery_candidates(lineage, artifact.compressed_sha256)
    if not candidates:
        return None
    resolved = []
    for reconciliation, result_sha in candidates:
        target = datetime.date.fromisoformat(
            reconciliation["official_snapshot_dates"][-1]
        )
        store = LegacyArtifactBackupStore(
            root,
            target_date=target,
            series_manifest_sha256=reconciliation[
                "official_series_manifest_sha256"
            ],
        )
        document, entry = store.read_original_document(
            symbol=symbol,
            original_sha256=reconciliation["legacy_artifact_sha256"],
            expected_result_sha256=result_sha,
        )
        resolved.append((reconciliation, document, entry))
    if len({entry["original_sha256"] for _, _, entry in resolved}) != 1:
        raise LegacyReconciliationError(
            f"daily history recovery is ambiguous for TW:{symbol}"
        )
    reconciliation, original, entry = resolved[0]
    merged = _merge_recovery_daily(
        artifact.document["daily"], original["daily"]
    )
    active_dates = {_daily_date(row) for row in artifact.document["daily"]}
    restored = [
        row for row in merged
        if start <= _daily_date(row) <= end
        and _daily_date(row) not in active_dates
    ]
    existing = lineage.get("daily_history_recovery")
    if not restored:
        return merged, (dict(existing) if isinstance(existing, dict) else None)
    receipt = _daily_history_receipt(
        symbol=symbol,
        artifact=artifact,
        reconciliation=reconciliation,
        entry=entry,
        restored=restored,
        recovery_target_market_date=recovery_target_market_date,
    )
    if isinstance(existing, dict) and existing != receipt:
        raise LegacyReconciliationError(
            f"daily history recovery receipt conflict for TW:{symbol}"
        )
    return merged, receipt
```

`_daily_history_receipt()` emits exactly the approved 17 fields, maps `original_size` and `original_uncompressed_size`, hashes the complete entry and ordered restored rows using sorted compact JSON with `allow_nan=False`, and hashes every receipt field except `receipt_sha256`. It rejects zero rows and non-increasing restored dates.

- [ ] **Step 5: Run focused GREEN**

```powershell
python -m unittest tests.test_tw_legacy_reconciliation -v
```

Expected GREEN: direct/historical binding, exact path, no scan, missing/ambiguous/conflict failures, post-retention hash, zero-row behavior, and idempotency all pass.

- [ ] **Step 6: Commit**

```powershell
git add stock_papi/quant/tw_legacy_reconciliation.py tests/test_tw_legacy_reconciliation.py
git commit -m "feat: resolve truncated TW daily history"


### Task 6: Cache recovery in the fetcher and validate lineage receipts

**Files:**

- Modify: `stock_papi/quant/tw_incremental.py:14-15,207-298,455-623,665-677,1102-1185`
- Modify: `tests/test_tw_incremental.py`

**Interfaces:**

- Consumes: the frozen `HistoryRecoveryResolver` result.
- Produces: one resolver call per symbol, cached merged rows, and one valid `daily_history_recovery` carried by `lineage_for()`.

- [ ] **Step 1: Write exact failing tests**

```text
TWOfficialIncrementalTests.test_recovery_resolver_is_optional_and_not_called_by_default
TWOfficialIncrementalTests.test_recovery_resolver_is_called_once_and_cached_for_all_datasets
TWOfficialIncrementalTests.test_recovered_daily_rows_are_filtered_by_requested_range
TWOfficialIncrementalTests.test_lineage_carries_valid_recovery_receipt_without_resolver
TWOfficialIncrementalTests.test_lineage_rejects_malformed_tampered_cross_symbol_or_zero_row_receipt
TWOfficialIncrementalTests.test_lineage_rejects_receipt_not_bound_to_direct_or_historical_reconciliation
```

The cache test calls price, institutional, and margin datasets and asserts one resolver call. The default test patches the backup reader to raise if touched. Receipt subtests alter every field, receipt SHA, restored dates/count, symbol, original SHA, target date, and series SHA.

- [ ] **Step 2: Run and record RED**

```powershell
python -m unittest tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovery_resolver_is_optional_and_not_called_by_default tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovery_resolver_is_called_once_and_cached_for_all_datasets tests.test_tw_incremental.TWOfficialIncrementalTests.test_recovered_daily_rows_are_filtered_by_requested_range tests.test_tw_incremental.TWOfficialIncrementalTests.test_lineage_carries_valid_recovery_receipt_without_resolver tests.test_tw_incremental.TWOfficialIncrementalTests.test_lineage_rejects_malformed_tampered_cross_symbol_or_zero_row_receipt tests.test_tw_incremental.TWOfficialIncrementalTests.test_lineage_rejects_receipt_not_bound_to_direct_or_historical_reconciliation -v
```

Expected RED: constructor rejects `recovery_resolver`; receipt validation/carry-forward is absent.

- [ ] **Step 3: Add the callable alias, cache, and one-call path**

```python
HistoryRecoveryResult = tuple[list[dict[str, Any]], dict[str, Any] | None]
HistoryRecoveryResolver = Callable[
    [
        str,
        IncrementalArtifact,
        _datetime.date,
        _datetime.date,
        _datetime.date,
    ],
    HistoryRecoveryResult | None,
]

# __init__
self.recovery_resolver = recovery_resolver
self._recovered_daily: dict[str, list[dict[str, Any]]] = {}
self._recovery_receipts: dict[str, dict[str, Any]] = {}
self._recovery_checked: set[str] = set()

# _daily_rows before range filtering
artifact = self._load_artifact(symbol)
if symbol not in self._recovery_checked:
    self._recovery_checked.add(symbol)
    if self.recovery_resolver is not None:
        recovered = self.recovery_resolver(
            symbol, artifact, start, end, self.target_date
        )
        if recovered is not None:
            merged, receipt = recovered
            self._recovered_daily[symbol] = [dict(row) for row in merged]
            if receipt is not None:
                self._recovery_receipts[symbol] = dict(receipt)
daily = self._recovered_daily.get(symbol, artifact.document["daily"])
```

- [ ] **Step 4: Add exact receipt validation and carry-forward**

Define `_DAILY_HISTORY_RECOVERY_FIELDS` as the 16 approved names listed in Frozen Interfaces. Add `_valid_daily_history_recovery(cls, value, *, artifact, reconciliations) -> bool` requiring exact fields, schema/mode/symbol, ISO ordered start/end, positive integer sizes/count, all six SHA fields as 64 lowercase hex, canonical `receipt_sha256`, and one reconciliation whose legacy SHA, final official snapshot date, and series SHA match the receipt.

```python
receipt = lineage.get("daily_history_recovery", _MISSING)
reconciliations = (
    [reconciliation]
    if isinstance(reconciliation, dict)
    else [item["reconciliation"] for item in history or []]
)
if receipt is not _MISSING and not cls._valid_daily_history_recovery(
    receipt, artifact=artifact, reconciliations=reconciliations
):
    return False
```

Cache a deep copy of a valid existing receipt in `_lineage_kind()`. In `lineage_for()`, emit the resolver receipt when present, otherwise the cached receipt. Carry-forward reads no quarantine.

- [ ] **Step 5: Run focused GREEN**

```powershell
python -m unittest tests.test_tw_incremental -v
```

Expected GREEN: resolver/cache/receipt tests and all strict overlap, reconciliation history, status, lifecycle, reused-symbol, and uniqueness tests pass.

- [ ] **Step 6: Commit**

```powershell
git add stock_papi/quant/tw_incremental.py tests/test_tw_incremental.py
git commit -m "feat: carry verified daily recovery lineage"
```

### Task 7: Wire explicit CLI recovery and preserve checkpoint semantics

**Files:**

- Modify: `stock_papi/batch/tw_official_post_close_cli.py:197-233,443-577,580-end`
- Modify: `tests/test_tw_official_post_close_cli.py`
- Modify: `tests/test_local_quant_batch.py`

**Interfaces:**

- Consumes: `resolve_truncated_daily_history()` and the fetcher's resolver parameter.
- Produces: `--recover-truncated-history`, `run(..., recover_truncated_history=False)`, and checkpoint identity key `recover_truncated_history`.
- Preserves: existing `run_market_batch()` generic/provider failure control flow without modifying that function.

- [ ] **Step 1: Write exact failing tests**

```text
TWOfficialPostCloseCLITests.test_cli_recovery_flag_is_explicit_opt_in
TWOfficialPostCloseCLITests.test_cli_default_path_never_constructs_resolver_or_touches_quarantine
TWOfficialPostCloseCLITests.test_cli_wires_recovery_resolver_only_when_enabled
TWOfficialPostCloseCLITests.test_cli_checkpoint_identity_rejects_changed_recovery_mode
TWOfficialPostCloseCLITests.test_recovery_failure_checkpoint_blocks_assert_complete_and_publication
LocalQuantBatchTests.test_recovery_failure_advances_cursor_without_overwriting_artifact
LocalQuantBatchTests.test_recovery_failure_resume_retries_failed_symbol_before_new_symbols
```

Temporary fixtures use a resolver raising `LegacyReconciliationError`. Assert artifact bytes unchanged, failure recorded, new-symbol `next_index` advanced, resume call order `[failed_symbol, next_symbol]`, repeated retry did not advance further, and `_assert_complete` raised before the publisher mock.

- [ ] **Step 2: Run and record RED**

```powershell
python -m unittest tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_recovery_flag_is_explicit_opt_in tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_default_path_never_constructs_resolver_or_touches_quarantine tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_wires_recovery_resolver_only_when_enabled tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_cli_checkpoint_identity_rejects_changed_recovery_mode tests.test_tw_official_post_close_cli.TWOfficialPostCloseCLITests.test_recovery_failure_checkpoint_blocks_assert_complete_and_publication tests.test_local_quant_batch.LocalQuantBatchTests.test_recovery_failure_advances_cursor_without_overwriting_artifact tests.test_local_quant_batch.LocalQuantBatchTests.test_recovery_failure_resume_retries_failed_symbol_before_new_symbols -v
```

Expected RED: flag/signature/identity tests fail because the recovery mode is absent. The two batch tests document existing behavior and do not authorize a `run_market_batch` production edit.

- [ ] **Step 3: Add the opt-in and identity key**

```python
# _enrich_batch_identity
result["recover_truncated_history"] = recover_truncated_history

# main parser
parser.add_argument("--recover-truncated-history", action="store_true")

# run and _run_stage
recover_truncated_history: bool = False
```

Require `type(recover_truncated_history) is bool`. Pass it through every stage and identity call, including false.

- [ ] **Step 4: Construct the resolver only when enabled**

```python
recovery_resolver = None
if recover_truncated_history:
    recovery_resolver = lambda symbol, artifact, start, end, target: (
        resolve_truncated_daily_history(
            root, symbol, artifact, start, end, target
        )
    )
fetcher = OfficialCompatFetcher(
    root,
    series,
    pd=pipeline.pd,
    legacy_overlap_policy=policy,
    recovery_resolver=recovery_resolver,
)
```

Do not create a recovery `LegacyArtifactBackupStore` in the CLI. Existing `backup_store` remains exclusively for `--reconcile-legacy-overlaps` writes.

- [ ] **Step 5: Run focused GREEN with status/lifecycle coverage**

```powershell
python -m unittest tests.test_tw_official_post_close_cli tests.test_local_quant_batch tests.test_tw_incremental.TWOfficialIncrementalTests.test_status_fetcher_preserves_history_and_exposes_target_evidence tests.test_tw_incremental.TWLegacyOverlapReconciliationTests.test_official_lineage_allows_symbol_history_after_series_start -v
```

Expected GREEN: opt-in/default isolation, identity, failure/resume, publication gate, regular/status, reconciliation, lifecycle, ETF, and short-history coverage passes. No command opens `D:\AbsorbData`.

- [ ] **Step 6: Commit**

```powershell
git add stock_papi/batch/tw_official_post_close_cli.py tests/test_tw_official_post_close_cli.py tests/test_local_quant_batch.py
git commit -m "feat: add opt-in TW history recovery"
```

### Task 8: Final verification, independent review, push, and Draft PR handoff

**Files:**

- Verify only: every file in the responsibility map.
- Do not modify production data, deployment files, tasks, pointers, or cloud resources.

**Interfaces:**

- Consumes: Tasks 1-7 completed commits.
- Produces: fresh local evidence, independent review evidence, pushed branch, and one Draft PR; no merge or deployment.

- [ ] **Step 1: Re-run the reader inventory and schema gate**

```powershell
rg -n --glob "*.py" 'document\["daily"\]|document\.get\("daily"\)|snapshot\["daily"\]|snapshot\.get\("daily"\)|daily\[-1\]|\.daily\b|load_incremental_artifact|StockSnapshot' local_quant.py stock_papi reporting scripts
python -m unittest tests.test_stock_analysis tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected: every match remains classified and every compatibility test passes. An unclassified reader or failure blocks later steps.

- [ ] **Step 2: Run all focused modules**

```powershell
python -m unittest tests.test_tw_daily_history_preservation tests.test_local_quant_batch tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_tw_trading_status tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_observation_views tests.test_observation_products tests.test_industry_report_analytics tests.test_industry_report_backtest tests.test_pit_dataset tests.test_oos_diagnostics -v
```

Expected: zero failures/errors, including the full multi-stage simulation, checkpoint/resume, direct/historical SHA binding, retention, zero-row receipt, quarantine isolation, status/lifecycle, ETF, short-history, reporting, dashboard, research, and quant-loader cases.

- [ ] **Step 3: Run the full Python suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: zero failures/errors. Record every environment-only skip by exact name/reason; a skip is not a pass.

- [ ] **Step 4: Run compile and syntax checks**

```powershell
python -m compileall -q local_quant.py stock_papi reporting tests
node --check static/app.js
powershell -NoProfile -Command '& { $parseErrors = @(); foreach ($path in @("scripts/python_runtime.ps1", "scripts/run_local_quant_task.ps1", "scripts/run_tw_post_close_pipeline.ps1", "scripts/invoke_pipeline_task.ps1")) { $tokens = $null; $errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $path), [ref]$tokens, [ref]$errors) | Out-Null; $parseErrors += $errors }; if ($parseErrors.Count) { $parseErrors | Format-List; exit 1 } }'
```

Expected: exit `0`. PowerShell files are parsed, not executed.

- [ ] **Step 5: Verify diff and prohibited scope**

```powershell
git diff --check 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git diff --name-only 0d2293d6fa8fb61a740a949f8ad084c24a266a2c..HEAD
git status --short
rg -n "rglob\(|canonical_daily|source_daily" local_quant.py stock_papi tests
```

Expected: clean diff, only mapped files, clean status, and no production `rglob(`, `canonical_daily`, or `source_daily`. The plan document commit is expected relative to the approved design base.

- [ ] **Step 6: Run independent read-only review and inspect output**

```powershell
$reviewLog = Join-Path $env:TEMP "agy-tw-daily-history-preservation-review.log"
agy --sandbox --print "唯讀審查目前 branch 相對 0d2293d6fa8fb61a740a949f8ad084c24a266a2c 的 diff。重點檢查資料契約、SHA/size/gzip/path binding、checkpoint/resume、retention、receipt、normal runtime quarantine isolation、consumer null compatibility；禁止修改檔案。" 2>&1 | Tee-Object -FilePath $reviewLog
Get-Content -Raw -Encoding utf8 $reviewLog
```

Expected: non-empty output and no unresolved Critical/Important finding. Empty output, tool/auth failure, or unresolved high-impact finding blocks push and Draft PR.

- [ ] **Step 7: Push the reviewed branch**

```powershell
git push origin codex/tw-daily-history-preservation
git rev-parse HEAD
git rev-parse refs/remotes/origin/codex/tw-daily-history-preservation
```

Expected: push succeeds and both full SHAs match.

- [ ] **Step 8: Create one Draft PR and stop**

```powershell
gh pr create --draft --base main --head codex/tw-daily-history-preservation --title "fix: preserve TW daily history" --body "Implements approved design 0d2293d6fa8fb61a740a949f8ad084c24a266a2c with canonical daily preservation, explicit manifest-bound recovery, persisted-reader compatibility gates, and temporary-directory verification only. No production recovery, publication, merge, or deployment has been performed."
```

Expected: one Draft PR URL. Do not merge and do not perform production operations.

```

```
