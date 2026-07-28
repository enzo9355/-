# TW Legacy Artifact Reconciliation Implementation Plan

> **Execution:** Use `superpowers:subagent-driven-development` for independent read-only review checkpoints and `superpowers:test-driven-development` for every production change. Terra owns all high-risk writes and final verification.

**Goal:** Add an explicit, backup-first, resumable path that replaces verified overlaps only in lineage-free legacy TW artifacts, then refuse official-CLI success until every active artifact reaches the target date.

**Architecture:** `OfficialCompatFetcher` classifies artifact provenance and selects strict verification or verified legacy replacement. A standard-library `LegacyArtifactBackupStore` owns content-addressed original bytes and an atomic manifest state machine. The official post-close CLI opts into both, preserves a stable series across retries, temporarily wraps the existing artifact writer, and performs a final checkpoint/exclusion/artifact completeness audit.

**Tech stack:** Python 3.10-compatible stdlib, pandas test dependency, `unittest`, gzip/JSON/SHA-256, Windows PowerShell 5.1 parser gates.

## Global constraints

- Exact baseline: `993fe68634cac228865e6c8e958455ed86bb9e07`.
- Worktree: `C:\Users\enzo\Documents\absorb-phase1l-legacy-reconciliation`.
- Branch: `codex/tw-legacy-overlap-reconciliation`.
- Default path remains strict; reconciliation requires `--reconcile-legacy-overlaps`.
- `_verify_existing()` remains byte-for-byte behaviorally strict; no tolerance or `math.isclose()`.
- Missing artifacts, including `00947B` and `00948B`, remain failures. No bootstrap branch.
- Do not modify `local_quant.py` or `tests/test_local_quant.py`; patch the writer only inside the official CLI context.
- Tests use `TemporaryDirectory`, constructed snapshots, and local fakes only.
- Zero TWSE/TPEx/FinMind/yfinance network calls and zero `D:\AbsorbData` writes.
- No endpoint, parser, cache, exclusion schema, publish, task, deploy, GCS, Cloud Run, LINE, model, or backtest change.
- Keep `ABSORB-TW-PostClose` and `ABSORB-TW-PreMarket` disabled.
- Before every commit run `git diff --cached --check`.
- One implementation task at a time. After each GREEN commit, run independent spec review followed by code-quality/security review; resolve findings before advancing.

## Planned files

**Create**

- `stock_papi/quant/tw_legacy_reconciliation.py`
- `tests/test_tw_legacy_reconciliation.py`
- `docs/superpowers/specs/2026-07-27-tw-legacy-artifact-reconciliation-design.md`
- `docs/superpowers/plans/2026-07-27-tw-legacy-artifact-reconciliation.md`

**Modify**

- `stock_papi/quant/tw_incremental.py`
- `stock_papi/batch/tw_official_post_close_cli.py`
- `tests/test_tw_incremental.py`
- `tests/test_tw_official_post_close_cli.py`

## Task 1: Freeze the approved design

**Files**

- Create: `docs/superpowers/specs/2026-07-27-tw-legacy-artifact-reconciliation-design.md`

**Failing review case**

- Trace current date planning, writer call order, checkpoint persistence, and lineage generation against the incident requirements.
- Demonstrate that evidence-only writer wrapping cannot repair a post-write/pre-apply crash.
- Demonstrate that the current date planner excludes the legacy baseline overlap.

**RED review evidence**

- Existing implementation has no policy, backup store, writer state recovery, baseline overlap snapshot, or completeness gate.

**Minimal implementation**

- Document strict/legacy classification, exact replacement semantics, stable bounded date planning, immutable no-clobber backup objects, atomic manifest transitions, read-only resume discovery, writer ordering, post-run gate, and rollback evidence boundary.

**GREEN review**

- Independent `sol_deep` review reports no Critical/Important design findings.
- `git diff --check` exits zero.

**Commit**

```text
docs: design legacy artifact reconciliation
```

### Task 1B: Approve and commit this executable plan

**Files**

- Create: `docs/superpowers/plans/2026-07-27-tw-legacy-artifact-reconciliation.md`

**Failing review case**

- Map every approved design contract to an exact RED test, production file, GREEN command, and commit.
- Reject the plan if any trust boundary, crash state, completeness rule, final gate, or planned file is missing.

**Minimal implementation**

- Record Tasks 2–10, the execution ledger, required review packages, and the exact focused/full/static/safety commands in this file.

**GREEN review**

- Independent `sol_deep` plan review reports no Critical/Important findings.
- `git diff --check` exits zero and only this plan is staged.

**Commit**

```text
docs: add legacy reconciliation implementation plan
```

## Task 2: Prove the fetcher replacement contract — RED

**Files**

- Modify: `tests/test_tw_incremental.py`

**Failing tests**

Add literal fixtures and tests that each name one break:

- `test_replace_verified_legacy_overrides_overlapping_price`
- `test_replace_verified_legacy_overrides_institutional_and_margin`
- `test_replace_verified_legacy_preserves_missing_optional_official_rows`
- `test_replace_verified_legacy_recalculates_without_duplicate_dates`
- `test_reconciliation_requires_official_price_for_every_overlap`
- `test_reconciliation_rejects_official_row_identity_mismatch`
- `test_strict_policy_still_rejects_overlap_mismatch`
- `test_official_lineage_v1_still_rejects_overlap_mismatch`
- `test_official_lineage_v2_still_rejects_overlap_mismatch`
- `test_unknown_lineage_is_not_treated_as_legacy`
- `test_malformed_official_lineage_is_not_treated_as_legacy`
- `test_reconciliation_evidence_records_original_hash_and_dates`
- `test_later_official_run_preserves_legacy_reconciliation_evidence`
- `test_reconciliation_does_not_bootstrap_missing_artifact`
- `test_unknown_legacy_overlap_policy_is_rejected`

Fixtures must construct a legacy artifact with an overlap row whose price, institutional, and margin values visibly differ from the official snapshot. Expected official values and metadata are hand-written literals, not derived through production helpers.

**RED command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental -v
```

Expected RED causes: constructor rejects `legacy_overlap_policy`, `reconciliation_for()` is absent, legacy mismatches still enter `_verify_existing()`, and malformed lineage is not classified.

**Commit**

```text
test: prove legacy overlap replacement contract
```

## Task 3: Implement the explicit fetcher policy — GREEN

**Files**

- Modify: `stock_papi/quant/tw_incremental.py`

**Consumes**

- Existing validated `IncrementalArtifact` and official snapshot series.
- Existing `_verify_existing()` unchanged.

**Produces**

```python
OfficialCompatFetcher(
    root,
    source,
    *,
    pd,
    legacy_overlap_policy="strict",
)

reconciliation_for(symbol) -> dict[str, Any] | None
```

**Minimal implementation**

1. Validate the policy against `strict` and `replace_verified_legacy`.
2. Classify lineage once per cached artifact:
   - absent/`None` is legacy;
   - complete official v2 lineage with known historical schema v1 or v2 is strict;
   - everything else raises the required eligibility error.
3. Keep `_verify_existing()` unchanged and call it for strict policy and all official lineage.
4. In eligible legacy replacement mode, require official price for every requested overlap and select official price/institutional/margin rows according to the design.
5. Accumulate deterministic date evidence, including explicit false optional replacements.
6. Add `legacy_reconciliation` to `lineage_for()` only when produced, and preserve an already-valid prior record on later official runs.
7. Reuse current DataFrame sorting/deduplication and append behavior; add no new abstraction outside small private validation/selection helpers.

**GREEN command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental -v
```

Then run strict regressions:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental tests.test_local_quant tests.test_local_quant_batch -v
```

Expected: zero failures/errors and all pre-existing mismatch tests still reject.

**Commit**

```text
feat: add explicit legacy overlap policy
```

**Review package**

- Baseline and GREEN commit SHAs.
- Exact test command/output.
- Diff of `tw_incremental.py` and its tests.
- Explicit checks: `_verify_existing()` unchanged, no tolerance, missing artifact still fails, official v1/v2 remain strict.

## Task 4: Prove immutable backup and resume behavior — RED

**Files**

- Create: `tests/test_tw_legacy_reconciliation.py`

**Failing tests**

- `test_backup_is_written_before_artifact_replacement`
- `test_backup_object_is_content_addressed_and_immutable`
- `test_backup_manifest_records_original_and_new_hashes`
- `test_backup_manifest_records_compressed_and_uncompressed_sizes`
- `test_backup_retry_is_idempotent`
- `test_backup_complete_state_can_resume`
- `test_applied_state_is_noop_without_mtime_change`
- `test_backup_state_conflict_fails_closed`
- `test_existing_object_conflict_fails_closed`
- `test_manifest_original_identity_cannot_change`
- `test_malformed_manifest_or_backup_path_fails_closed`
- `test_path_escape_or_symlink_is_rejected`
- `test_resume_discovery_restores_original_baseline`
- `test_resume_discovery_rejects_multiple_series`

Use a real temporary gzip artifact and real filesystem operations. Mock only the writer callback/order boundary; do not mock the backup store itself. The immutable-object test attempts a conflicting existing object and proves it is not overwritten. The no-op test captures `st_mtime_ns` and SHA before retry.

**RED command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_legacy_reconciliation -v
```

Expected RED: import failure because `stock_papi.quant.tw_legacy_reconciliation` does not exist.

**Commit**

```text
test: prove immutable reconciliation backups
```

## Task 5: Implement the backup store — GREEN

**Files**

- Create: `stock_papi/quant/tw_legacy_reconciliation.py`

**Produces**

```python
class LegacyReconciliationError(RuntimeError): ...

class LegacyArtifactBackupStore:
    def __init__(self, root, *, target_date, series_manifest_sha256): ...
    def backup_before_write(self, *, symbol, artifact_path, evidence): ...
    def mark_applied(self, *, symbol, artifact_path): ...
    @classmethod
    def discover_resume(cls, root, *, target_date): ...
```

`backup_before_write()` returns a small action value used by the caller: write, passthrough, or no-op after verified repair. It does not call the production writer.

**Minimal implementation**

1. Validate constructor identities, symbol regex, exact artifact path, containment, and every existing symlink/reparse component.
2. Read original bytes once; verify compressed bound, bounded gzip expansion, SHA, and evidence identity.
3. Create a unique same-directory temp with `O_EXCL`, flush/fsync, verify it, then publish with atomic no-clobber `os.link`; never replace an existing object.
4. Parse and fully validate every existing manifest field and derive object paths from hashes rather than trusting JSON paths.
5. Write manifest JSON atomically through a unique same-directory temp and `os.replace`.
6. Enforce only no-entry → `backup_complete` → `applied`, with exact retry, repair, no-op, and conflict branches from the design table.
7. Verify expected written artifacts through the existing incremental loader plus exact official/reconciliation lineage identities before `applied`.
8. Implement read-only target-date resume discovery with at most one validated series identity and no mutation.

No new dependency, rollback command, delete path, or production-root special case.

**GREEN command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_legacy_reconciliation -v
```

Then run the artifact loader regressions:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_legacy_reconciliation tests.test_tw_incremental tests.test_local_quant_batch -v
```

Expected: zero failures/errors; all backup tests use temporary roots.

**Commit**

```text
feat: add legacy artifact backup store
```

**Review package**

- RED and GREEN commit SHAs/output.
- State-transition table mapped to test names.
- Filesystem/path validation diff.
- Explicit checks: original bytes before write, no overwrite primitive, complete manifest validation, no delete/rollback, no absolute path trusted from JSON.

## Task 6: Prove CLI opt-in, stable retry, writer restoration, and completion — RED

**Files**

- Modify: `tests/test_tw_official_post_close_cli.py`

**Failing tests**

- `test_cli_reconciliation_flag_is_explicit_opt_in`
- `test_cli_default_path_remains_strict`
- `test_cli_reconciliation_prepends_baseline_overlap_date`
- `test_cli_reconciliation_counts_baseline_in_session_limit`
- `test_cli_resume_reuses_discovered_baseline_and_series_identity`
- `test_cli_reconciliation_patches_and_restores_writer`
- `test_cli_reconciliation_recalculates_through_existing_calc_all_before_write`
- `test_cli_restores_all_patches_when_assignment_or_pipeline_fails`
- `test_cli_backup_happens_before_existing_writer`
- `test_cli_reconciliation_does_not_bootstrap_missing_artifact`
- `test_cli_refuses_success_when_checkpoint_has_active_symbol_failures`
- `test_cli_refuses_success_when_checkpoint_is_partial_or_wrong_identity`
- `test_cli_refuses_success_when_active_artifact_is_stale`
- `test_cli_refuses_success_when_active_artifact_is_missing`
- `test_cli_refuses_success_when_active_artifact_is_future_dated`
- `test_cli_allows_excluded_symbol_to_remain_stale`
- `test_cli_allows_checkpoint_failure_only_for_excluded_symbol`
- `test_cli_refuses_malformed_raw_exclusion_state`
- `test_cli_returns_success_when_all_active_artifacts_match_target`
- `test_cli_repairs_last_artifact_post_write_pre_apply_without_rewrite`
- `test_cli_returns_nonzero_local_quant_status_unchanged`

Reuse full-shape fake `OfficialSnapshotSeries`, checkpoint, and canonical exclusion CSV fixtures. Keep real temp artifacts and the real final artifact audit. Mock official series construction and `local_quant.main()` only to avoid network/production execution. The `calc_all` case uses a pipeline spy and asserts the written artifact reflects its derived output, rather than merely asserting the spy was called. The last-artifact crash case starts with every artifact already at target plus one `backup_complete` entry, then proves baseline discovery rebuilds the same series, the writer repairs `applied`, artifact SHA/mtime stay unchanged, and the final gate returns success.

**RED command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_official_post_close_cli -v
```

Expected RED causes: parser/run lack the opt-in argument, date planning omits baseline, writer is never wrapped, and zero is returned without a completeness audit.

**Commit**

```text
test: prove official CLI rejects incomplete recovery
```

## Task 7: Integrate reconciliation and completeness gate — GREEN

**Files**

- Modify: `stock_papi/batch/tw_official_post_close_cli.py`

**Minimal implementation**

1. Add `run(..., reconcile_legacy_overlaps=False)` and parser `store_true` flag.
2. In opt-in mode, read-only discover a prior baseline, prepend the earliest baseline to the bounded date plan, build the series, and require any discovered series SHA to match.
3. Construct the fetcher with explicit strict or replacement policy.
4. In opt-in batch identity add `legacy_overlap_policy`; omit only the two mutable historical audit fields. Preserve strict default identity.
5. Extend `_patched_pipeline()` with an optional backup store and save the original writer.
6. Put every monkey-patch assignment inside `try`; restore all five globals in reverse order in `finally`.
7. For TW writes in opt-in mode, ask the store for passthrough/write/no-op state before calling the existing writer. Back up before write and mark applied only after re-read validation.
8. Parse raw exclusion CSV directly and fail closed on schema, symbol, duplicate, state, action, encoding, or CSV errors.
9. When local main returns zero, validate final checkpoint stage/market/index/failures/fixed identity and audit only `universe - pending - excluded` at the exact target date. Raise exactly `TW official observation recovery is incomplete` on any gate failure.
10. Return nonzero local status unchanged. Make no `local_quant.py` change.

**GREEN command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_official_post_close_cli -v
```

Then run the required focused suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_local_quant tests.test_local_quant_batch -v
```

Expected: zero failures/errors; default strict tests and observation-only no-publish/no-exclusion-mutation tests remain green.

**Commit**

```text
feat: integrate reconciliation and completeness gate
```

**Review package**

- RED/GREEN commit SHAs and exact outputs.
- Patch/restore ordering, batch identity, date planning, writer ordering, and completeness-gate diffs.
- Explicit checks: no `local_quant.py`, publish, exclusion writer, official endpoint/parser/cache, or bootstrap changes.

## Task 8: Focused regression, whole-branch review, and documentation status

**Files**

- Modify only if verification evidence requires status update:
  - `docs/superpowers/specs/2026-07-27-tw-legacy-artifact-reconciliation-design.md`
  - `docs/superpowers/plans/2026-07-27-tw-legacy-artifact-reconciliation.md`

**Verification command**

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_local_quant tests.test_local_quant_batch -v
```

Expected: zero failures/errors.

Run independent whole-branch `sol_deep` review against exact baseline. Resolve all Critical/Important findings and rerun focused tests. A documentation-only correction uses:

```text
docs: finalize legacy reconciliation verification record
```

Do not mark Production recovery complete.

## Task 9: Full/static/safety verification

**Full test**

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

Required: at least 930 tests, zero failures, zero errors, exit zero.

**Python static gates**

Set an ignored temporary pycache prefix outside tracked files, then run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q reporting stock_papi tests
.\.venv\Scripts\python.exe -m py_compile local_quant.py
```

**Other static gates**

```powershell
node --check static/app.js
git diff --check 993fe68634cac228865e6c8e958455ed86bb9e07..HEAD
git status --short
```

Run the existing PowerShell 5.1 parser gates for:

- `scripts/python_runtime.ps1`
- `scripts/run_tw_post_close_pipeline.ps1`
- `scripts/run_tw_pre_market_pipeline.ps1`
- `scripts/invoke_pipeline_task.ps1`
- `scripts/native_process.ps1`
- `scripts/upload_local_quant.ps1`

**Diff inventory**

```powershell
git diff --name-status 993fe68634cac228865e6c8e958455ed86bb9e07..HEAD
```

Require exactly the eight planned files. Confirm `local_quant.py`, endpoints, parsers, official cache, exclusion schema, publish, Scheduled Task scripts, deployment, and bootstrap are absent.

**Safety checks**

- Changed-file secret scan for authorization, bearer, cookie, password, token, service-role, Supabase, LINE, and report-admin values.
- Four Production regression-readiness flags remain false.
- `backtests/v1/latest-TW.json` remains absent.
- No tracked fixture/cache/artifact/checkpoint/quarantine/credential files.
- Re-check both TW Scheduled Tasks are `Disabled`.
- Re-check original checkout branch, HEAD, and seven user-owned untracked paths are unchanged.
- Record counters: official HTTP 0, FinMind 0, yfinance remote 0, `D:\AbsorbData` mutations 0, Scheduled Task mutations 0.

## Task 10: Draft PR and stop

Push only `codex/tw-legacy-overlap-reconciliation` and create a Draft PR against `main` titled:

```text
feat: reconcile verified legacy TW artifact overlaps
```

PR body sections:

- Problem
- Design
- Out of Scope
- TDD
- Verification
- Safety
- Rollback

Require PR `OPEN`, `DRAFT`, `UNMERGED`. Inspect available workflow/check evidence without claiming a missing CI gate passed. Do not mark ready, merge, delete branch, run Production recovery, write `D:\AbsorbData`, enable tasks, publish, promote, deploy, upload, or notify LINE.

## Execution ledger

| Task | RED commit | GREEN commit | Spec review | Code/security review | Status |
|---|---|---|---|---|---|
| Design | n/a | `cdca6e37` | approved | approved | done |
| Fetcher | `1cef5979` | `ff7f2a05` | approved | approved | done |
| Backup store | `d3eb4743` | `93f2f56e` | approved | approved | done |
| CLI/gate | `f32ab0c1` | `2f4b1605` | approved | approved | done |
| Whole branch | n/a | `2f4b1605` | approved | approved | done |

## Phase 1L verification record (superseded by Phase 1N revision)

- Required focused suite: 143 tests passed, one explicit Windows symlink-capability skip.
- Full suite: 990 tests passed, one explicit Windows symlink-capability skip.
- Python compile, Node syntax, PowerShell 5.1 AST, diff, scope, and changed-file secret gates passed.
- Baseline-to-implementation diff contains exactly the eight planned files.
- Both TW Scheduled Tasks remained `Disabled`; no Production recovery, live market request, `D:\AbsorbData` mutation, publish, promotion, deployment, upload, or LINE notification was performed.

## Plan self-review

- Every production file is preceded by observable RED tests and a dedicated RED commit.
- Every task names exact files, RED command, minimal implementation, GREEN command, commit, and review evidence.
- Fetcher, backup store, CLI integration, completeness, retry discovery, and rollback evidence map to explicit tests.
- The plan does not require `local_quant.py`, Production data, network, publish, deployment, or task mutation.
- There are no placeholders, deferred implementation decisions, or automatic rollback steps.

## Phase 1N revision execution plan

The Phase 1M independent validation makes the prior verification record insufficient. The existing Draft PR is revised in place from `826617fd65d82147b6d52dce029b87ee797faab2`; the base remains `993fe68634cac228865e6c8e958455ed86bb9e07`. The branch, eight-file scope, Production prohibitions, and fail-closed missing-artifact boundary are unchanged.

### N1: Reproduce independent findings — RED

**Files**

- `tests/test_tw_incremental.py`
- `tests/test_tw_legacy_reconciliation.py`
- `tests/test_tw_official_post_close_cli.py`

Add the required current-artifact resume, post-run manifest gate, outer/inner cross-binding, no-price preservation, optional independence, later-session, deduplication, provenance, backup-before-write, idempotent resume, strict-mode, and missing-baseline tests. Fixtures use literal expected OHLCV and temporary roots only. Run the three focused test modules against the unchanged Production implementation and record the failing test names, count, and exit code.

**Commit**

```text
test: reproduce PR 19 independent validation findings
```

### N2: Validate current artifacts and successful terminal state — GREEN

**Files**

- `stock_papi/quant/tw_legacy_reconciliation.py`
- `stock_papi/batch/tw_official_post_close_cli.py`

Bump the backup manifest and deterministic namespace to v2 and rename entry `replaced_dates` to `overlap_dates`. In read-only discovery, validate every exact safe current artifact path and backup object. `applied` requires exact `new_sha256` plus full expected-result validation. `backup_complete` permits only the exact original SHA or a full post-write/pre-apply result. Add `assert_current_state_complete()` and invoke it after `local_quant.main()` returns zero and before `_assert_complete()`; it accepts only fully validated `applied` entries. The final active-artifact gate also requires complete official lineage for the requested series and a matching applied entry whenever reconciliation evidence is present.

Run `tests.test_tw_legacy_reconciliation` and `tests.test_tw_official_post_close_cli` after the fix.

**Commit**

```text
fix: validate current artifacts during reconciliation resume
```

### N3: Cross-bind lineage identities — GREEN

**Files**

- `stock_papi/quant/tw_incremental.py`
- `stock_papi/quant/tw_legacy_reconciliation.py`

Extend the one official-lineage validator so outer historical SHA/as-of and source/schema/series/snapshot identity equal the inner reconciliation values exactly. When preserving prior reconciliation evidence, `lineage_for()` keeps those outer duplicated fields bound to the original legacy/source identity instead of replacing them with an intermediate run identity. The backup result validator reuses that validator and adds only manifest-entry-specific checks. Run the cross-binding tests in `tests.test_tw_incremental` and `tests.test_tw_legacy_reconciliation`.

**Commit**

```text
fix: cross-bind legacy reconciliation lineage identities
```

### N4: Preserve legacy values when official overlap rows are absent — GREEN

**Files**

- `stock_papi/quant/tw_incremental.py`

Represent every overlap with independent price, institutional, and margin actions. Missing official price preserves literal legacy OHLCV; available optional rows still replace independently; unavailable optional rows preserve legacy values. Later sessions continue through the existing append path. Emit reconciliation schema v2 with `overlap_dates`, disjoint per-dataset replaced/preserved partitions, and exact action evidence. Do not add a symbol allowlist, synthesize data, or change `_verify_existing()`.

Run `tests.test_tw_incremental`, then the five-module focused suite.

**Commit**

```text
feat: preserve legacy overlap price when official row is unavailable
```

### N5: Read-only nine-symbol validation and final gates

Build the seven-date official series from the existing `D:\AbsorbData` warm cache using a `NoNetworkSession` whose `get()` raises. For each of `1589`, `3064`, `3067`, `4183`, `4305`, `4804`, `6236`, `6242`, and `8905`, load the existing artifact read-only, run all three fetcher datasets, build the snapshot in memory, and record SHA, original/final dates, actions, row counts, duplicates, finite-value status, and exception. No batch or writer is called.

Then run:

```powershell
python -m unittest tests.test_tw_incremental tests.test_tw_legacy_reconciliation tests.test_tw_official_post_close_cli tests.test_local_quant tests.test_local_quant_batch -v
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q reporting stock_papi tests
python -m py_compile local_quant.py
node --check static/app.js
```

Also run PowerShell 5.1 AST parsing, `git diff --check`, exact eight-file inventory, secret/scope/tracked-output checks, Scheduled Task recheck, and protected original-checkout comparison. The full count must exceed 990 with zero failures/errors and at most the existing single symlink-capability skip.

### N6: Independent reviews and existing Draft PR update

Run two independent final-head reviews: security/state-machine and data/provenance. Resolve every Critical or Important finding and rerun affected gates. Update and push only the existing Draft PR #19. It remains `OPEN`, `DRAFT`, and `UNMERGED`; do not claim missing CI evidence passed, complete the active universe, or perform Production recovery.

## Phase 1N plan self-review

- Each Phase 1M finding has an observable RED test before Production code changes.
- Resume discovery and the first-run success path both validate current artifacts.
- One official-lineage validator owns all outer/inner identity bindings.
- Schema v2 distinguishes overlap from per-dataset replacement or preservation without symbol exceptions.
- The nine-symbol check is read-only, network-denying, writer-free, and batch-free.
- `00947B` and `00948B` remain fail closed; no bootstrap or exclusion mutation exists.

## Phase 1N execution result (2026-07-28)

The three Phase 1M implementation findings now have RED/GREEN coverage:

| Revision | RED commit | GREEN commit | Focused result |
|---|---|---|---|
| resume current artifact | `3883fc8`, `ad7177d` | `51a9739b`, `07b1b30` | included below |
| outer/inner lineage binding | `3883fc8` | `e531334` | included below |
| no-price overlap/schema v2 | `3883fc8`, `a786157` | `d1c16c9` | included below |
| cross-date lineage state | `65be2fd1` | `2e79bc3` | 171 tests, one skip |
| terminal artifact SHA binding | `65be2fd1` | `2e79bc3` | 171 tests, one skip |
| manifest concurrent update | `4f5aab9` | `2e79bc3` | 171 tests, one skip |

The first pair of final reviews correctly rejected the prior head because a later target could retain an internally impossible old direct lineage, the terminal gate compared only reconciliation symbol membership rather than the same artifact SHA, and manifest read-modify-write transitions were not serialized across processes. The remediation keeps current outer lineage current, moves older valid evidence to independently validated history, binds each applied manifest SHA to the loaded terminal artifact, and uses a persistent standard-library file lock around manifest transactions.

Fresh code verification on the remediated head passed: the five required focused modules ran 171 tests with one existing Windows symlink-capability skip, and the full suite ran 1,018 tests with the same single skip. Python syntax compilation also passed with bytecode output redirected to the system temporary directory. The remaining static, safety, and final independent-review gates are recorded separately before the Draft PR is updated.

The mandatory Production-cache acceptance gate did not pass. The seven-date warm series `bc195e0df4cd6e63ee93f90c31c746007fbfa320db44374ef1617c2d725f7544` was read with zero requests and zero selected-data mutations. All nine symbols produced all three datasets, schema-v2 baseline evidence, zero duplicate identities, and zero non-finite numeric values. However, the validated 2026-07-24 snapshot contains no official price row for any of the nine, so every target-dated in-memory stock snapshot correctly failed with `ValueError: target market date mismatch`. Their latest available price dates were:

| Symbol | Latest price date |
|---|---|
| 1589 | 2026-07-16 |
| 3064 | 2026-07-22 |
| 3067 | 2026-07-23 |
| 4183 | 2026-07-22 |
| 4305 | 2026-07-23 |
| 4804 | 2026-07-16 |
| 6236 | 2026-07-16 |
| 6242 | 2026-07-23 |
| 8905 | 2026-07-23 |

No price was synthesized or carried forward. Therefore this revision remains fail closed and cannot be reported ready for independent revalidation until a separately authorized source/cache correction provides trustworthy target-date price rows or the acceptance contract is revised.
