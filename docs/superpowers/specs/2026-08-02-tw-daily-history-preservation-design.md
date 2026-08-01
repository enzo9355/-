# TW Multi-Stage Daily History Preservation Design

## Status and authorization

Approved design direction for a later TDD implementation in the isolated worktree created from `main` commit `336c25acc1903b0d76503f3cc4589e8a8de950b7`.

This document authorizes only the design commit. It does not authorize RED tests, implementation, access to or mutation of `D:\AbsorbData`, live recovery, GCS, Cloud Run, Scheduled Tasks, production pointers, LINE delivery, merge, or interaction with PID 17820. Implementation remains blocked until this document is reviewed and approved.

## Problem and reproduced root cause

`local_quant.build_stock_snapshot()` currently performs this sequence:

1. Load up to the configured 730-day history with `pipeline.get_data(symbol, 730)`.
2. Replace that frame with `pipeline.calc_all(frame)`.
3. Serialize the calculated frame as the artifact `daily` array.

`stock_papi.quant.features.calc_all()` calculates rolling features and returns `frame.dropna()`. The first rows required only as MA, return, volatility, and volume-ratio warm-up are therefore removed before persistence.

`OfficialCompatFetcher._daily_rows()` later reloads only the persisted `daily` array. A subsequent recovery stage appends a new official session to an already shortened frame, calculates again, removes another warm-up prefix, and overwrites `daily` again. A retry after the artifact write but before checkpoint advancement can repeat the same erosion on the same target date.

The exact base was reproduced with the real `build_stock_snapshot()` and `calc_all()` flow:

```text
2026-07-29: input 40, persisted 20
2026-07-30: input 21, persisted 1
2026-07-31: input 2, ValueError("calculated history is unavailable")
```

The defect is at the snapshot construction and persistence boundary. Quarantine lookup in every normal `_daily_rows()` call would hide the symptom while making recovery storage a permanent runtime dependency, so that approach is rejected.

## Goals

1. Keep `daily` as the single canonical persisted history; do not add `canonical_daily` or `source_daily`.
2. Preserve every canonical row returned inside the existing 730-day request window, including indicator warm-up rows.
3. Use calculated rows for analysis and inference without using row deletion as a persistence policy.
4. Keep dates strictly increasing and unique, and make same-target reruns stable.
5. Preserve existing regular-price, official non-price, lineage, evidence-hash, reconciliation, and reused-symbol lifecycle contracts.
6. Recover already-truncated reconciled artifacts only through an explicit opt-in path backed by exact manifest identities.
7. Keep normal runtime independent of quarantine.
8. Fail closed for genuinely insufficient calculated history, unverifiable backups, or conflicting historical prices.

## Non-goals

- No synthetic prices, forward-fill, interpolation, or relabeling of an old close as a target-session close.
- No change to rolling formulas, model features, LightGBM, backtests, prediction targets, or recommendation policy.
- No second daily-history field and no artifact schema-version bump solely for history preservation.
- No unlimited retention beyond the existing `get_data(symbol, 730)` contract.
- No automatic recovery, quarantine scan, filename search, `rglob()`, or arbitrary backup selection.
- No weakening of official source, lifecycle, reconciliation, artifact-size, gzip-expansion, SHA-256, path, symlink, or reparse-point validation.
- No production recovery or publication in the implementation pull request.

## Frozen invariants

### Canonical daily history

- `daily` remains the canonical history read by incremental updates and existing consumers.
- The canonical frame is the target-date-filtered result of `get_data(symbol, 730)` before indicator calculation.
- Canonical dates are normalized to market dates, duplicate dates are rejected, and output is sorted ascending.
- Persistence never removes a row merely because an indicator is unavailable.
- Same-date canonical OHLCV is immutable across a rerun. A duplicate date with conflicting OHLCV is an error, not a last-write-wins update.
- `rows == len(daily)`, `latest == daily[-1]`, and `as_of` continues to describe the latest regular price row.

### Date and status semantics

- For a regular-price session, `as_of == latest_regular_price_date == target_market_date == observation_as_of`.
- For `official_no_regular_trade` and `officially_suspended`, `target_market_date == observation_as_of`, while `as_of == latest_regular_price_date < target_market_date`.
- Status evidence, its SHA-256, and official lifecycle precedence remain unchanged.
- A status session never creates or renames a price row for the target date.

### Derived data

- `calc_all()` receives a copy of the canonical frame.
- Analysis, inference, and backtesting consume the calculated frame and may require fully populated indicator rows.
- Persistence starts from the full canonical frame, clears every calculated indicator column to missing, and joins calculated values back only on matching dates.
- Indicator warm-up rows serialize missing values as JSON `null`; OHLCV and other canonical source fields remain present.
- The calculated frame may not overwrite canonical `Date`, OHLCV, institutional, margin, short, market, option, or data-quality source values.
- The latest canonical regular-price date must also exist in the fully calculated frame. Otherwise the existing fail-closed calculated-history error remains appropriate.

### Retention and idempotency

- No additional unlimited archive is introduced. The persisted frame is bounded by the existing 730-day retrieval behavior.
- When the moving window drops an old date, that is configured retention, not rolling-indicator erosion.
- Before each join, old calculated columns are cleared across the canonical frame. This prevents a row that has moved into the new warm-up prefix from retaining a stale value calculated with data now outside the retention window.
- Re-running the same target date produces one row per date and the same canonical OHLCV. Recovery receipts are deterministic and are not duplicated.

## Selected normal construction flow

`build_stock_snapshot()` will separate the two in-memory concerns without changing the artifact shape:

```text
get_data(symbol, 730)
  -> target-date filter
  -> validate unique dates and sort
  -> canonical_frame

canonical_frame.copy()
  -> calc_all()
  -> validate non-empty and contains latest canonical date
  -> calculated_frame
  -> analysis / inference / backtest

canonical_frame
  -> clear calculated columns
  -> join calculated columns by date
  -> full persisted_frame
  -> daily / latest / rows / date semantics
```

`stock_papi.quant.features` will expose one exact ordered calculated-column contract covering the columns assigned by `calc_all()`:

```text
MA_5, MA20, RET_1, RET_5, RET_20, RANGE_PCT, VOL_RATIO, VOL_CHG,
INST_NET_RATIO, MARGIN_CHG, SHORT_CHG, RSI, Volat, MACD_DIF, MACD,
MACD_OSC, K, D, BB_UP, BB_DN
```

This list is not a new feature abstraction. It is the minimum shared contract needed to clear and join existing indicators without treating source columns such as `MARKET_RET_1` as derived output. A focused contract test will keep the list aligned with `calc_all()` assignments.

Observation-only mode applies its existing model-column removal to the calculated frame before the date join. It does not remove canonical rows.

Consumers that require complete indicators continue to receive the calculated frame directly inside snapshot construction. Any reader that analyzes persisted `daily` must explicitly filter the required indicator columns with `dropna(subset=required_columns)` or an equivalent check at that consumer boundary; it must not assume every persisted row is feature-ready.

## Explicit recovery boundary

The official post-close CLI gains a separate boolean opt-in named `--recover-truncated-history`, defaulting to false. Its programmatic `run()` argument also defaults to false, and the batch/checkpoint identity records the selected recovery mode so a checkpoint cannot resume under different semantics.

Without the flag:

- no recovery resolver is constructed;
- no quarantine path is resolved, opened, listed, or probed;
- `OfficialCompatFetcher._daily_rows()` uses only the active artifact;
- existing strict and `--reconcile-legacy-overlaps` behavior remains unchanged.

With the flag, the CLI injects a read-only resolver into `OfficialCompatFetcher`. The fetcher does not import the backup store and does not know a quarantine directory. The resolver is called at most once per symbol and its verified merged rows and deterministic receipt are cached for all price, institutional, and margin dataset requests during that run.

`--recover-truncated-history` is independent of `--reconcile-legacy-overlaps`. The former repairs canonical history already lost from an official artifact; the latter replaces eligible legacy overlap values from official snapshots. Either flag remains explicit, and neither changes the default path.

## Authorized backup resolution

Recovery authority comes only from an already valid `source_lineage`:

1. Load the current artifact through `load_incremental_artifact()` and require `_valid_official_lineage()` to pass.
2. If the valid lineage has neither direct `legacy_reconciliation` nor validated `legacy_reconciliation_history`, the symbol is not recovery-eligible and continues unchanged even when the flag is enabled.
3. For an eligible symbol, read the direct `legacy_reconciliation`, or the reconciliation records inside validated `legacy_reconciliation_history` envelopes.
4. Derive each possible backup location from that record's exact reconciliation target and `official_series_manifest_sha256`:

   ```text
   <root>/quarantine/tw-recovery/legacy-reconciliation/v2/
     <reconciliation-target-date>/<series-manifest-sha256>/manifest.json
   ```

5. Do not recursively scan, glob, search by filename, or select the first filesystem match.
6. A candidate qualifies only when its validated manifest entry binds the same symbol and `legacy_artifact_sha256` recorded by the lineage.
7. Every recovery-eligible symbol must resolve exactly one distinct qualifying backup object. Repeated lineage references to the same object are deduplicated by the same SHA. Zero or multiple different qualifying objects fail closed.

The existing `LegacyArtifactBackupStore` is extended with a read-only original-document method so recovery reuses its path, manifest, object, gzip, and size trust boundary instead of implementing a weaker reader.

The method requires:

- the exact schema-v2 target-date and series-manifest directory;
- a complete valid manifest with an `applied` entry for the requested symbol;
- `entry.original_sha256 == reconciliation.legacy_artifact_sha256`;
- `entry.backup_path == objects/<original_sha256>.json.gz`;
- exact compressed size and SHA-256;
- exact bounded uncompressed size;
- safe child paths with no symlink or Windows reparse-point component;
- a decoded object whose market and symbol match `TW:<symbol>`, whose `daily` dates are ordered and unique, and whose declared date fields agree with the rows.

Missing manifests, missing objects, unsupported schemas, malformed entries, hash or size mismatch, decompression overflow, changed bytes, unsafe paths, symbol mismatch, or identity mismatch raise a recovery error. There is no fallback to another backup file or network source.

## Merge rules for a truncated artifact

The resolver merges the verified original backup `daily` with the active artifact `daily` by market date:

- Each input independently requires valid, unique dates.
- Backup-only earlier rows restore the missing canonical prefix.
- Current-only rows remain unchanged.
- On an overlapping date, `Open`, `High`, `Low`, `Close`, and `Volume` must be numerically identical after the same finite-number validation used by the incremental reader.
- Conflicting overlap OHLCV fails closed.
- When overlap OHLCV agrees, the current artifact row wins as a whole so later verified official institutional, margin, status-related, and reconciliation results are not replaced by older legacy values.
- The merged result is sorted ascending and contains exactly one row per date.
- The existing 730-day request range is applied after the merge by the normal fetch path; recovery does not create unlimited persisted retention.

The resolver never writes the backup or active artifact. The normal atomic artifact writer persists the rebuilt full `daily` only after calculation, date checks, lineage generation, and all existing artifact validation succeed.

## Deterministic migration receipt

A successful recovery adds one optional `daily_history_recovery` object under `source_lineage`. It is preserved unchanged by later normal runs without reopening quarantine.

The receipt has exact fields:

```json
{
  "schema_version": 1,
  "mode": "restore_verified_reconciliation_backup",
  "symbol": "2330",
  "recovery_target_market_date": "2026-07-31",
  "input_artifact_sha256": "<64 hex>",
  "original_artifact_sha256": "<64 hex>",
  "backup_target_market_date": "2026-07-24",
  "backup_series_manifest_sha256": "<64 hex>",
  "backup_manifest_entry_sha256": "<64 hex>",
  "backup_object_size": 1234,
  "backup_object_uncompressed_size": 5678,
  "restored_start_date": "2024-08-01",
  "restored_end_date": "2026-07-16",
  "restored_row_count": 500,
  "restored_daily_sha256": "<64 hex>",
  "receipt_sha256": "<64 hex>"
}
```

Hash inputs use the repository's canonical JSON encoding. `backup_manifest_entry_sha256` hashes the complete validated manifest entry. `restored_daily_sha256` hashes the ordered backup-only canonical rows that were added. `receipt_sha256` hashes every receipt field except itself.

No wall-clock timestamp is included, so a same-input recovery is deterministic. `input_artifact_sha256` records the truncated artifact that triggered recovery; `original_artifact_sha256` records the immutable backup object. If no row is added because the artifact already carries the same valid receipt and merged history, recovery is a verified no-op and the existing receipt is retained. A different receipt for the same artifact fails closed rather than creating an unbounded receipt history.

An opt-in rerun still revalidates the exact authorized manifest and backup object before accepting that no-op. Only a later normal run with the flag off carries the receipt without reading quarantine.

`_valid_official_lineage()` validates the optional receipt's exact field set, hashes, symbol, date ordering, positive sizes/count, and cross-binding to a validated direct or historical reconciliation record. `lineage_for()` carries a valid existing receipt forward even when the recovery flag is off. Carry-forward validates only artifact metadata and the receipt itself; it never reads quarantine.

## Failure behavior

The existing `ValueError("calculated history is unavailable")` remains for an empty calculated frame. A calculated frame that does not contain the latest canonical regular-price date is also rejected rather than publishing a latest row with unavailable required indicators.

Recovery failures use one stable domain error at the CLI boundary while retaining specific exception causes in tests. No failure overwrites the active artifact, advances the checkpoint, publishes a candidate, changes an exclusion, or marks a backup as repaired.

Examples that must fail closed:

- recovery requested for a reconciled artifact but the authorized manifest or object is missing;
- hash, compressed size, uncompressed size, gzip expansion, symbol, target, series, or path binding differs;
- more than one different authorized original object qualifies;
- either input contains duplicate dates;
- overlapping OHLCV conflicts;
- lineage or recovery receipt is malformed, tampered, cross-symbol, or not bound to reconciliation evidence;
- canonical history is too short to calculate the latest regular-price row after verified recovery.

## Test design for the later TDD phase

No tests are added or run by this design-only commit. After approval, RED tests will be written before production code and will cover:

### Sequential preservation and warm-up

- Use the real `calc_all()` with a canonical baseline sufficient for MA20.
- Run baseline, `2026-07-29`, `2026-07-30`, and `2026-07-31` through artifact write and `OfficialCompatFetcher` reload.
- Assert counts do not lose a rolling prefix, dates remain ordered/unique, and no calculated-history error occurs.
- Assert early OHLCV rows remain and all calculated indicator columns are JSON `null` until available.
- Assert the latest row is fully calculated.

### Repeated erosion and idempotency

- Run at least N, N+1, and N+2 and prove no repeated 20-row loss.
- Rerun the same target date and prove identical date membership and historical OHLCV with no duplicate rows.
- Cover the write-before-checkpoint retry shape.

### Date, status, lineage, and lifecycle

- Regular-price session date equality.
- `official_no_regular_trade` and `officially_suspended` target/observation dates with an older latest regular price date.
- Existing lineage, evidence SHA, direct reconciliation, reconciliation history, and recovery receipt validation/carry-forward.
- ETF, short-history fail-closed behavior, legacy-overlap reconciled artifacts, and PR #22 reused-symbol lifecycle precedence.

### Runtime isolation and recovery trust boundary

- A normal update succeeds when no quarantine directory exists.
- A normal update is instrumented so any quarantine filesystem access fails the test.
- Recovery uses only the exact lineage-derived manifest path.
- Missing manifest/object, altered bytes, wrong hash/size/uncompressed size, unsafe path, wrong symbol, multiple distinct candidates, and overlap OHLCV conflict all fail closed.
- A valid recovery restores the prefix, preserves current overlap rows, records the deterministic receipt, and is idempotent.
- A later normal run preserves and validates the receipt without touching quarantine.

### Required sequential simulation evidence

The final implementation validation will run a temporary-directory simulation:

```text
baseline -> 2026-07-29 -> 2026-07-30 -> 2026-07-31 -> rerun 2026-07-31
```

It will report before/after row counts, first/last dates, uniqueness, latest calculated availability, and stable historical OHLCV. All paths will be temporary fixtures; `D:\AbsorbData` and production systems remain out of scope.

## Expected implementation boundaries after approval

The later implementation should remain limited to the smallest existing boundaries that own the behavior:

- `local_quant.py`: separate canonical persistence from calculated analysis and join indicators by date.
- `stock_papi/quant/features.py`: expose the exact calculated-column list.
- `stock_papi/quant/tw_incremental.py`: optional injected resolver, merged-row cache, receipt validation/carry-forward.
- `stock_papi/quant/tw_legacy_reconciliation.py`: exact read-only manifest-bound backup load.
- `stock_papi/batch/tw_official_post_close_cli.py`: opt-in flag, resolver wiring, and checkpoint identity.
- Existing focused test modules, plus at most one dedicated sequential-history test module if that keeps the real multi-stage reproduction readable.

No new dependency, artifact format, general migration framework, rollback tool, or production script is required.

## Acceptance gates for implementation

Implementation is not complete until all of the following have fresh evidence:

1. Mandatory RED failures observed before production changes.
2. Focused preservation, incremental, reconciliation, status, ETF, short-history, and lifecycle tests GREEN.
3. Full Python suite GREEN, with any environment-only skip reported exactly.
4. `python -m compileall` succeeds.
5. Applicable Node syntax and PowerShell parser checks succeed.
6. Temporary sequential simulation shows no erosion, no duplicate dates, no ValueError, and a stable same-date rerun.
7. Normal runtime test proves no quarantine dependency.
8. `git diff --check` succeeds and worktree status contains only intended files.
9. Independent code and security/data-contract review has no unresolved Critical or Important finding.
10. One implementation commit is pushed and a draft pull request is opened against `main`; merge and production operations remain prohibited.
