# TW Legacy Artifact Reconciliation Design

## Status

Approved implementation scope from the Phase 1L restart brief. This design authorizes code and fixture-only tests in the isolated worktree only. It does not authorize Production execution, `D:` writes, live market-data calls, publication, GCS, Cloud Run, LINE, Scheduled Task mutation, exclusion mutation, model changes, merge, or rollback execution.

## Problem

The official TW bulk compatibility path verifies every overlapping historical row exactly. That is correct for artifacts already produced from `tw_official_bulk_v2`, but some otherwise valid pre-lineage artifacts contain historical OHLCV or chip values that differ from the verified official snapshots. Strict verification stops those `ACTIVE_STALE` symbols before the normal pipeline can reach the target date.

The recovery path must be explicit and narrower than a tolerance rule. It may replace only verified overlap rows in artifacts whose `source_lineage` is absent or `None`, must preserve an immutable copy of the original compressed artifact before overwrite, and must remain safe across process interruption and retry.

Missing-baseline symbols, including `00947B` and `00948B`, are not part of this phase. They remain fail-closed for a separate bootstrap design.

## Goals

1. Preserve the current strict path as the default.
2. Allow an explicit CLI opt-in to replace verified legacy overlap inputs with official rows.
3. Re-run the existing pipeline over the reconciled input so derived fields are recalculated normally.
4. Preserve original compressed bytes in a deterministic content-addressed backup before artifact overwrite.
5. Make the backup/write/apply sequence idempotent and recoverable after interruption.
6. Refuse success when any active symbol remains failed, missing, stale, or future-dated.

## Non-goals

- Global numeric tolerance or changes to `_verify_existing()`.
- Synthesis of missing institutional or margin data.
- Bootstrap of missing historical artifacts.
- Automated rollback tooling.
- Changes to official endpoints, parsers, raw cache, checkpoint schema, exclusion policy, publication, reporting, models, or scheduling.
- Production recovery or historical publication backfill.

## Selected approach

Extend `OfficialCompatFetcher` with one policy switch and keep the existing strict verifier intact. In reconciliation mode, classify the loaded artifact lineage once, select official rows only for eligible legacy overlap dates, and record deterministic per-symbol evidence. The existing `calc_all()` path consumes those replacement inputs and recalculates the complete artifact.

Add a small standard-library backup store. The official CLI patches `write_stock_artifact()` only inside the existing context manager and only when the opt-in is enabled. The wrapper checks any existing manifest state before deciding whether to back up, write, repair an interrupted state, no-op, or fail closed.

After `local_quant.main()` returns zero, the official CLI independently checks the final checkpoint, raw exclusions, and all active artifacts. General `local_quant.main()` exit semantics remain unchanged.

## Artifact lineage classification

The fetcher accepts only these classes:

- `LEGACY_NO_LINEAGE`: `source_lineage` is absent or exactly `None`.
- `OFFICIAL_V2_LINEAGE`: `source_lineage` is a dictionary with `source_mode == "tw_official_bulk_v2"` and a valid complete identity.

The official identity validation accepts only the known `tw-official-historical-v1` and `tw-official-historical-v2` schema versions. Both remain strict-only. It also requires a valid target date equal to the artifact `as_of`, a 64-hex series manifest, a non-empty ordered unique snapshot-date list ending at the target date, matching per-snapshot manifest identities, the same symbol, and valid historical identity fields. Early or partial v1 lineage that lacks this identity still fails closed; it is never treated as legacy. If present, `legacy_reconciliation` must also satisfy its schema and hash/date contracts.

Any other present lineage is unknown or malformed and raises:

```python
IncrementalHistoryError(
    f"historical artifact lineage is not eligible for reconciliation: TW:{symbol}"
)
```

Official v2 lineage always uses `_verify_existing()`, even when the opt-in policy is selected. This prevents a previously official artifact from being silently reinterpreted as legacy.

## Fetcher policy and replacement rules

`OfficialCompatFetcher.__init__()` gains:

```python
legacy_overlap_policy: str = "strict"
```

Only `strict` and `replace_verified_legacy` are accepted. Unknown values raise `ValueError("unknown legacy overlap policy")`.

For `strict`, and for every official v2 artifact, behavior is unchanged.

For an eligible legacy artifact in `replace_verified_legacy` mode:

- Every artifact/snapshot overlap date must have an official price row. Missing price fails closed.
- Price output uses official OHLCV instead of legacy OHLCV for each overlap.
- Institutional output uses official rows when present; otherwise it preserves the legacy synthetic rows.
- Margin output uses the official row when present; otherwise it preserves the legacy margin values.
- Official dates later than the artifact `as_of` continue to append through the existing incremental path.
- Output remains sorted and deduplicated using the current dataset identities.

The current strict date planner starts at the trading session after the earliest artifact date, so it cannot provide a snapshot for the baseline overlap itself. Only in reconciliation mode, the CLI prepends `earliest_latest_date` to the planned official dates. The baseline date counts toward `MAX_CATCHUP_SESSIONS`; if baseline plus later sessions exceeds the existing bound, the run refuses before source loading. Strict mode retains the current date plan unchanged. Because the series then spans every trading session from the earliest artifact through the target, artifacts with later `as_of` values also receive their own exact overlap snapshot.

An interrupted retry must retain that same baseline even if the last stale artifact was already written and the fresh audit now reports only the target date. Before planning opt-in dates, the backup store performs a read-only discovery below the fixed target-date directory. No manifest means a first run. Exactly one fully validated series directory may supply its earliest `replaced_dates` value as the prior baseline; multiple series identities, malformed entries, or path/link violations fail closed. The planner uses the earlier of current audit and discovered baseline, rebuilds the bounded dates, and then requires the resulting official series manifest to equal the discovered directory identity. This makes the backup root and checkpoint identity stable across the post-write/pre-apply crash window without trusting mutable checkpoint state.

The fetcher accumulates evidence per symbol and exposes `reconciliation_for(symbol)`. Evidence contains schema version, mode, original compressed SHA-256, all overlap dates, price/institutional/margin replaced-date lists, and per-date booleans so absent optional official data is explicitly recorded as not replaced.

`lineage_for(symbol)` adds this evidence as `legacy_reconciliation` only after reconciliation. A later official run preserves a valid existing reconciliation record so the audit trail is not discarded.

## Backup store

`stock_papi.quant.tw_legacy_reconciliation` defines `LegacyReconciliationError` and `LegacyArtifactBackupStore`.

The store uses only the Python standard library and writes under:

```text
<root>/quarantine/tw-recovery/legacy-reconciliation/v1/
  <target-date>/<series-manifest-sha256>/
```

It contains:

```text
objects/<original-sha256>.json.gz
manifest.json
```

The constructor validates the date and 64-hex series identity. Symbol values must match the existing TW symbol contract. Artifact paths must equal the expected TW artifact path under the configured root. Existing path components are checked without following links; symlinks and Windows reparse points are rejected. Resolved paths must remain below their expected roots.

Backup objects contain the original compressed bytes. A new object is written to a unique same-directory `O_EXCL` temporary file, flushed/fsynced, then atomically published without overwrite using `os.link(temp, final)` and the temporary link is removed. If the final name wins a race, the store verifies the existing object instead. An unsupported no-clobber operation fails closed. The final object is always re-read to verify bytes, size, and SHA. Manifest writes use a unique same-directory `O_EXCL` temporary file, flush/fsync, and atomic `os.replace()`.

Every manifest read validates the complete top-level and entry contract: schema, target date, series manifest, unique valid symbols, status, hashes, sizes, replacement dates, and status-specific `new_sha256`. `backup_path` is not trusted input; it must exactly equal `objects/<original-sha256>.json.gz`. All existing artifact, backup-root, object, manifest, and temporary-path components are checked without following links and reject symlinks or Windows reparse points.

Read-only resume discovery scans only immediate 64-hex series directories under the fixed target-date parent. It accepts at most one directory containing a valid manifest and verifies that directory name against the manifest series identity. It never creates, deletes, or repairs state; repair remains a writer-wrapper operation after the exact series has been rebuilt.

The manifest has one immutable original identity per symbol. It records both compressed and bounded uncompressed sizes so a later controlled rollback can verify the same gzip safety contract as the artifact loader:

```json
{
  "schema_version": 1,
  "target_market_date": "2026-07-24",
  "official_series_manifest_sha256": "<64 hex>",
  "entries": {
    "2330": {
      "symbol": "2330",
      "status": "backup_complete",
      "original_sha256": "<64 hex>",
      "original_size": 1234,
      "original_uncompressed_size": 5678,
      "backup_path": "objects/<sha>.json.gz",
      "replaced_dates": ["2026-07-16"],
      "new_sha256": null
    }
  }
}
```

Only `backup_complete` to `applied` is a valid transition. `applied` requires the verified current artifact SHA in `new_sha256`. A symbol cannot acquire a second original SHA for the same target/series manifest.

## Writer ordering and state machine

In opt-in mode, `_patched_pipeline()` saves and patches `local_quant.write_stock_artifact`, then restores it in `finally`. Non-TW writes always call the original writer.

For every TW write, the wrapper first checks the manifest entry, including cases where this run's fetcher has no new evidence. This is necessary because a post-write crash leaves an official-lineage artifact, so a retry no longer produces legacy evidence.

| Manifest state | Current artifact | Action |
|---|---|---|
| no entry, no new evidence | any normal artifact | call original writer unchanged |
| no entry, new evidence | SHA equals evidence original | write/verify immutable backup, atomically record `backup_complete`, then call original writer |
| `backup_complete` | SHA equals original and new evidence exactly matches original SHA, replacement dates, target, and series | retry original writer without creating another backup |
| `backup_complete` | verified expected official v2 result | atomically repair entry to `applied`, return current path without rewriting |
| `backup_complete` | anything else | fail state conflict |
| `applied` | SHA equals `new_sha256` | return current path without rewriting or changing mtime |
| `applied` | SHA differs | fail state conflict |

An expected interrupted result must pass the normal artifact loader and prove: official v2 mode, current target date, current series manifest, and a legacy reconciliation original SHA equal to the manifest original SHA.

After an actual write, the wrapper re-reads the artifact, validates the same expected lineage, computes its compressed SHA, and atomically marks the entry `applied`. A writer failure leaves the backup and `backup_complete` entry intact.

The backup store never deletes an object or performs rollback.

## CLI integration

`run()` gains `reconcile_legacy_overlaps: bool = False`. The parser gains `--reconcile-legacy-overlaps` as `store_true`.

Without the flag:

- the fetcher policy is `strict`;
- no backup store is created;
- the writer is not patched;
- all existing callers preserve behavior.

With the flag, the fetcher uses `replace_verified_legacy` and `_patched_pipeline()` receives a backup store tied to the exact target date and official series manifest. The checkpoint identity includes the fixed policy name. In this mode it omits `historical_latest_date_counts` and `historical_unavailable_count`, because those values change after a partial successful write and would incorrectly turn an interrupted retry into a different batch. The default strict identity remains unchanged.

All monkey-patch assignments are inside the context manager's `try` block. `finally` restores writer, builder, batch, loader, and fetcher in reverse order, including when an assignment or the wrapped pipeline raises.

Missing artifacts still fail in `load_incremental_artifact()` and never reach backup creation. There is no symbol-specific bootstrap branch.

## Post-run completeness gate

The gate runs only after the official CLI invokes `local_quant.main()`:

1. A nonzero result is returned unchanged.
2. For zero, load the final TW checkpoint and require `stage == "market_batch"`, `market == "TW"`, and `next_index >= len(universe)`. Require a valid `failed` list with no malformed/unknown item and no failure whose symbol remains active. Failures belonging only to current pending/excluded symbols are not active recovery failures; this resolves the existing retry-checkpoint case where an excluded symbol can remain in a checkpoint that was not rewritten.
3. Read the raw exclusion CSV directly rather than using `load_exclusion_list()`, which intentionally swallows read errors. Require the canonical headers, unique valid TW symbols, blank `OperatorAction`, and only `Pending`, blank-as-pending, or `Excluded` state. Encoding, CSV, schema, duplicate, action, symbol, or state errors are incomplete.
4. Compute `active = universe - pending - excluded`.
5. Require checkpoint batch identity to match the requested target, observation product, official source mode/schema, series manifest, ordered-universe SHA, and reconciliation policy.
6. Run the existing artifact audit only for `active`.
7. Require no unavailable active symbols and require every active latest date to equal the requested target. The audit already rejects future artifacts.

Any failure raises exactly:

```python
RuntimeError("TW official observation recovery is incomplete")
```

An excluded symbol may remain stale. A pending symbol is also outside the active recovery set, but the opt-in does not mutate either state.

## Security and failure boundaries

- No network call is added.
- No production path, cache, checkpoint, exclusion, quarantine, or artifact is touched by tests; all use temporary roots and mocked pipeline calls.
- No hash, size, JSON schema, date, path containment, link/reparse, or lineage validation is weakened.
- Backup bytes are captured and verified before the existing atomic writer runs.
- Strict official overlaps retain exact numeric equality; no `math.isclose()` is introduced.
- Context-managed monkey patches are restored on success and exception.
- The two TW Scheduled Tasks remain disabled throughout this code phase.

## Test strategy

TDD is split into four bounded groups:

1. Fetcher lineage classification, strict preservation, replacement semantics, evidence, deduplication, and missing-baseline refusal.
2. Backup ordering, content addressing, immutable objects, manifest transitions, retry/repair/no-op, conflict, and path/reparse rejection.
3. CLI explicit opt-in, writer patch/restore, before-write backup ordering, and unchanged default path.
4. Completeness-gate checkpoint failures, missing/stale/future active artifacts, excluded stale artifacts, and complete active success.

The final focused suite covers incremental, backup, official CLI, local quant, and local quant batch tests. The full suite, Python/Node/PowerShell static gates, diff check, scope audit, secret scan, task state, network/data counters, and independent whole-branch review remain mandatory before a Draft PR can be reported ready.

## Rollback evidence contract

This phase records enough evidence for a later separately authorized rollback: immutable original bytes, original SHA/size, current applied SHA, deterministic source/target identity, replacement dates, and unambiguous status. It deliberately does not implement or execute rollback.

## Success criteria

- Default strict behavior is unchanged.
- Only explicit opt-in legacy artifacts replace verified overlaps.
- Official v2 and malformed lineage remain strict/fail-closed.
- Original compressed bytes are durably verified before overwrite.
- All specified interruption states resume or fail deterministically.
- Already applied state performs no artifact rewrite.
- Official CLI returns zero only when every active artifact is complete at the target date.
- Missing-baseline symbols remain outside this phase.
- No Production, network, publication, scheduling, or data mutation occurs.
