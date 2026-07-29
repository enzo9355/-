# TW Non-Price Trading Status Contract Implementation Plan

## Execution status

The B1-B4 code plan is implemented on `codex/tw-non-price-status-contract` through `223f3cc`. The remaining work is verification, review, PR merge, compatibility-first deployment, one controlled recovery, content-addressed publication, public acceptance, and Scheduled Task restoration.

This plan is executable end to end under the current authority. Production mutation is gated by the safety and rollback checks below.

## Goal and architecture

Preserve official no-price evidence without weakening the target-date price gate:

```text
official sources
  -> content-addressed raw cache
  -> daily price/status/termination partition
  -> schema-v2 stock artifacts with separate observation and price dates
  -> terminal completeness gate
  -> manifest v3
  -> dashboard/report/status-first views
  -> immutable-first upload and generation-guarded pointers
```

The implementation reuses the existing artifact and publication paths. It adds no status-only store, database, dependency, symbol allowlist, or fallback source.

## Global constraints

- Never infer suspension from a missing price row.
- Never forward-fill, synthesize, copy, or relabel OHLCV.
- Keep regular target-date price validation strict.
- Derive exchange and universe membership from verified metadata.
- Fail closed on missing, stale, conflicting, malformed, or hash-mismatched evidence.
- Keep exact cache v1, manifest v2, and manifest v3 branches.
- Tests use temporary roots, sanitized fixtures, mocked transport, and no production credentials.
- Production rollout uploads immutable dependencies before mutable pointers and uses generation preconditions.
- Scheduled Tasks remain disabled until every public acceptance check passes.

## Pre-stage safety hardening

Two narrow fixes were completed before B1:

- `7139eb9c`: accept nanosecond calendar timestamps without changing session identity.
- `3ff43001`: capture the complete observation rollback pointer set, including missing pointers.

Files: `stock_papi/batch/calendar.py`, `scripts/capture_observation_lkg.ps1`, `tests/test_batch_calendar.py`, and `tests/test_observation_release_scripts.py`.

## B1: Official raw-row and lifecycle evidence

### Exact files

Production:

- `stock_papi/integrations/market_data/tw_official_bulk.py`
- `stock_papi/integrations/market_data/tw_official_cache.py`
- `stock_papi/integrations/market_data/tw_official_historical.py`
- `stock_papi/integrations/market_data/tw_trading_status.py`

Tests:

- `tests/test_tw_official_bulk.py`
- `tests/test_tw_official_hardening.py`
- `tests/test_tw_official_historical.py`
- `tests/test_tw_trading_status.py`

### RED tests

- Content-addressed raw cache rejects metadata, parser, size, path, compressed hash, and payload hash drift.
- TWSE and TPEx blank OHLC rows remain hash-bound evidence and never become price rows.
- Partial blank/prose OHLC fails closed; valid OHLC with zero volume remains a price row.
- Non-negative official volume is preserved as evidence but never converted to current-session volume.
- Resume closes suspension on its effective session; termination is a disposition.
- Lifecycle cold and warm reads produce the same verified identity.
- Non-stock lifecycle rows are ignored by typed metadata, not by hardcoded symbols.
- Missing price without a covering status or termination fails closed.

Focused RED commits:

- `420857f4 test: freeze TW non-price row semantics`
- `bb5a9bd4 test: preserve TW raw non-price evidence`
- `0cb289e4 test: require verified TW lifecycle cache`
- `8fb1c373 test: require status-aware TW daily snapshots`
- `4d942108 test: freeze status-aware source schema v3`
- `98cd9336 test: distinguish TPEx warrants from stock lifecycle`

### Minimal GREEN

- Classify the official raw price row once into either a canonical price row or exact no-regular-trade evidence.
- Store raw payloads below `source-cache/tw-official/v2` with content addressing and two hashes.
- Normalize official lifecycle rows into hash-bound suspend/resume/terminate events.
- Load lifecycle sources only when the configured price partition has a gap.
- Resolve intervals and precedence, keep termination separate, and reject every unknown gap.
- Bind source schema `tw-official-historical-v3` to raw price, lifecycle, status, and termination hashes.

Focused command:

```powershell
python -B -m unittest tests.test_tw_official_bulk tests.test_tw_official_hardening tests.test_tw_official_historical tests.test_tw_trading_status -v
```

GREEN boundary: `e72a32c4 feat: preserve official TW non-price evidence`.

## B2: Artifact dates and terminal completeness

### Exact files

Production:

- `local_quant.py`
- `stock_papi/batch/tw_official_post_close_cli.py`
- `stock_papi/integrations/market_data/tw_official_historical.py`
- `stock_papi/quant/tw_artifact_audit.py`
- `stock_papi/quant/tw_incremental.py`

Tests:

- `tests/test_local_quant_batch.py`
- `tests/test_tw_incremental.py`
- `tests/test_tw_official_post_close_cli.py`

### RED tests

- Schema-v2 artifact writer preserves explicit observation fields.
- Status artifact keeps existing daily history and prior `as_of`.
- Audit reports observation date separately from latest price date.
- Status fetcher exposes only the exact target snapshot evidence.
- CLI injects status only into the matching symbol build.
- Terminal gate accepts the exact regular/status partition and rejects unknown gaps, hash drift, price/status overlap, target-dated status history, and stale observation identity.
- Exchange partition comes from catalog metadata rather than symbol syntax.

Focused RED commits:

- `e7893cf9 test: freeze TW artifact observation dates`
- `cfae57d4 test: require TW artifact schema v2 auditing`
- `e0be9c44 test: require status injection into TW artifacts`
- `91e49785 test: freeze TW status-aware terminal gate`
- `4e9bf41c test: require metadata-driven TW exchange partition`

### Minimal GREEN

- Keep `as_of == latest_regular_price_date == daily[-1].Date`.
- Add `target_market_date`, `observation_as_of`, `observation_kind`, and evidence to artifact schema v2.
- Do not append a row for a status session.
- Keep checkpoint, official lineage, reconciliation, artifact SHA, and exclusion validation unchanged.
- Treat only verified status as an alternative to target-date price; unknown missing price remains terminally incomplete.

Focused command:

```powershell
python -B -m unittest tests.test_local_quant_batch tests.test_tw_incremental tests.test_tw_official_post_close_cli -v
```

GREEN boundary: `351bb201 feat: separate TW observation and price dates`.

## B3: Manifest v3, loaders, and upload trust boundary

### Exact files

Production:

- `local_quant.py`
- `reporting/schemas.py`
- `reporting/source_loader.py`
- `scripts/upload_local_quant.ps1`
- `stock_papi/batch/tw_official_post_close_cli.py`
- `stock_papi/integrations/market_data/tw_trading_status.py`
- `stock_papi/quant/tw_incremental.py`
- `stock_papi/repositories/quant_snapshots.py`

Tests and fixtures:

- `tests/report_fixtures.py`
- `tests/test_daily_report_source.py`
- `tests/test_local_quant_batch.py`
- `tests/test_local_quant_publish.py`
- `tests/test_local_quant_task.py`
- `tests/test_quant_snapshot_repository.py`
- `tests/test_tw_incremental.py`
- `tests/test_tw_official_post_close_cli.py`
- `tests/test_tw_trading_status.py`

### RED tests

- Manifest v3 partitions regular, status, and operational symbols with exact arithmetic.
- Status entries cross-bind evidence SHA, artifact SHA, symbol, target date, and price date.
- Unknown missing artifact preserves the previous latest pointer.
- Report source loader rejects mixed versions, counter drift, overlap, bad dates, bad size/hash/gzip/JSON, and evidence mismatch.
- v2 remains exact and cannot carry status metadata.
- Cloud repository cache identity separates schema and manifest SHA and returns no data on tampering.
- PowerShell preflight rejects invalid local v3 input before any `gcloud` copy.
- Valid preflight proves immutable object and manifest ordering before the pointer.

RED boundary: `d7a31cf6 test: freeze TW status manifest v3 trust boundary`.

### Minimal GREEN

- Publish v3 only when an explicit TW target date and valid schema-v2 artifacts are present.
- Derive status membership from artifact evidence, never from caller input.
- Emit content-addressed objects, immutable manifest, then atomic latest pointer.
- Dispatch v2 and v3 validation through separate exact paths.
- Validate pointer, manifest, objects, arithmetic, dates, hashes, and evidence before report/dashboard consumers or upload.
- Add `-PreflightDataRoot` for non-mutating local upload validation.

Focused commands:

```powershell
python -B -m unittest tests.test_local_quant_publish tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_local_quant_task tests.test_tw_official_post_close_cli tests.test_tw_trading_status -v
powershell.exe -NoProfile -Command "$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'scripts/upload_local_quant.ps1'),[ref]$tokens,[ref]$errors) > $null; if($errors.Count){$errors | ForEach-Object {$_.Message}; exit 1}"
```

Alignment fix: `daac7708 fix: align TW status evidence field`.

GREEN boundary: `1084a96e feat: publish and load TW status manifest v3`.

## B4: Report and public-surface stale-price suppression

### Exact files

Production:

- `reporting/observation_v2.py`
- `reporting/professional_builder.py`
- `stock_papi/batch/observation_products.py`
- `stock_papi/integrations/line/flex.py`
- `stock_papi/integrations/market_data/tw_official_historical.py`
- `stock_papi/services/observation_view.py`
- `stock_papi/services/report_view.py`
- `stock_papi/web/routes/market.py`
- `templates/report_observation.html`
- `templates/reports/post_close_professional.html`
- `templates/stock_detail.html`

Tests:

- `tests/test_observation_products.py`
- `tests/test_observation_public_surfaces.py`
- `tests/test_professional_report_builder.py`
- `tests/test_professional_report_html.py`
- `tests/test_tw_official_historical.py`

### RED tests

- Extreme stale status values cannot change market, industry, event, or ETF output.
- Dashboard status summaries have exact labels, dates, SHA, and optional explicitly dated last close only.
- Evidence tampering rejects dashboard creation.
- Professional status observations remain outside event classification.
- Report view rejects extra price fields and malformed status identity.
- Status stock page and LINE card contain no current Close, move, volume, indicators, chart, or technical events.
- Lifecycle suspension remains lifecycle-only even when a blank price row exists.

RED boundaries:

- `ab0b050b test: freeze TW status report presentation`
- `2294f92 test: freeze TW suspension evidence precedence`

### Minimal GREEN

- Partition loaded stocks once; pass only regular artifacts into price calculations.
- Build a compact hash-bound status list and separate data-quality counters.
- Carry status through observation metadata and the professional report without adding event policy entries.
- Validate exact status summary shape at report boundaries.
- Branch status-first in stock and LINE views before reading `latest` as current.
- Render fixed labels and only an explicitly dated last regular close.
- Keep suspension evidence byte-equivalent to lifecycle resolution; retain any corroborating blank row only in raw cache.

Focused commands:

```powershell
python -B -m unittest tests.test_observation_products tests.test_professional_report_builder tests.test_professional_report_html tests.test_observation_public_surfaces -v
python -B -m unittest tests.test_tw_official_historical tests.test_tw_trading_status -v
```

GREEN boundaries:

- `ab2339cf feat: render verified TW non-price observations`
- `223f3cc fix: keep suspension identity lifecycle-only`

## Cross-stage verification

Run after all code and documentation changes:

```powershell
python -B -m unittest tests.test_tw_trading_status tests.test_tw_official_bulk tests.test_tw_official_historical tests.test_tw_incremental tests.test_tw_official_post_close_cli tests.test_local_quant_batch tests.test_local_quant_publish tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_local_quant_task tests.test_observation_products tests.test_observation_products_cli tests.test_observation_report_v2 tests.test_observation_views tests.test_observation_public_surfaces tests.test_professional_report_builder tests.test_professional_report_html tests.test_report_web -v
python -m compileall -q .
node --check static/app.js
git diff --check 9c10b6af385306d35582eec30df1b16b6034db7f..HEAD
```

Then run the repository's complete Python suite, PowerShell parser/release/upload tests, route inventory, cold-start heavy-import gate, secret scan, hardcoded-symbol scan, date-relabel scan, schema-permissiveness scan, cache v1/v2 tests, manifest v2/v3 tests, freshness tests, rollback tests, and Scheduled Task configuration tests. Record exact counts, skips, exit codes, and HEAD; retries cannot hide a failure.

## Compatibility-first production rollout

### 1. Immutable baseline

- Verify branch, origin/main, PR identity, clean worktrees, and commit graph.
- Keep the three repair-related Scheduled Tasks disabled.
- Capture DataRoot disk, ACL, lock, checkpoint, source cache, artifacts, publish roots, receipts, and logs.
- Capture every required GCS pointer as existing or missing with generation, hash, schema, date, and target object.
- Capture Cloud Run service, traffic, revision, image, service account, environment names, recent errors, and rollback revision.
- Capture canonical public HTTP status/data dates and application request IDs.
- Run `scripts/capture_observation_lkg.ps1` and validate its receipt before mutation.

### 2. Git and review

- Push without force and update PR #20 with root cause, contracts, commits, tests, rollout, and rollback.
- Resolve every Critical or Important independent-review finding and rerun affected tests.
- Mark Ready only with a clean branch, fresh full suite, mergeable PR, and no unresolved review.
- Merge with a merge commit bound to the expected head SHA; never squash.
- Sync a clean main checkout and verify main equals the merge commit.

### 3. Cloud Run compatibility gate

- Deploy only the exact merged main SHA.
- Keep GCS latest pointers on their previous schema while the new revision is verified.
- Confirm `/health` and old v2/missing-pointer behavior on the new revision.
- Record the old revision for traffic rollback.
- Do not select a v3 pointer until the dual v2/v3 loader revision is healthy.

### 4. One controlled TW recovery

- Resolve the latest completed TW trading session dynamically from the verified calendar and official publication readiness.
- Validate all six canonical sources: TWSE/TPEx price, institutional, and margin.
- Require exact top-level and nested dates, schema fingerprints, production row thresholds, non-zero TPEx margin rows, reproducible SHA, and bounded request count.
- Recheck DataRoot allowlist, LKG, disk, ACL, lock, checkpoint identity, source manifest, universe identity, and rollback receipt immediately before mutation.
- Run one controlled post-close recovery while Scheduled Tasks remain disabled.
- Require unknown missing count zero and validate representative regular, ETF, no-regular-trade, suspended, and terminated/disposition paths from real artifacts.

### 5. Controlled publication

Upload and verify in dependency order:

1. quant artifact objects;
2. quant manifest;
3. dashboard object;
4. report PDF when generated;
5. report metadata and professional artifact;
6. report index;
7. generation-guarded latest pointers.

For each mutable pointer, record before generation, use the precondition, record after generation, then read back and revalidate schema, target date, SHA, size, and referenced object existence. Stop on the first mismatch and use the LKG generation-matched rollback before continuing.

### 6. Public acceptance

From the canonical production URL, require successful and date-consistent responses for homepage, health, dashboard API, market-insights API, market page, stocks page, a regular stock page, one status stock page, reports, Ask GET, and one side-effect-free Ask POST.

Verify no stale-price language on status outputs, no missing global data banner, new report history plus retained old history, matching API/dashboard/report identities, and no new unhandled exception, schema error, hash mismatch, bucket 404, Ask traceback, template undefined, or credential exposure in logs.

### 7. Scheduled Tasks

Only after public acceptance:

- Verify action, arguments, clean-main WorkingDirectory, interpreter, DataRoot, principal, schedule, policy, and absence of plaintext secrets.
- Enable PostClose, PreMarket, and the recovery/upload task only when each configuration is valid.
- Read back Enabled state, next run, principal, action, and WorkingDirectory.
- Do not force a second production pipeline run.

## Explicit non-modification list

The implementation does not modify:

- model, feature, inference, promotion, backtest, or recommendation logic;
- historical price rows or numeric price semantics;
- legacy reconciliation algorithm or immutable backup content;
- exclusion CSV schema, thresholds, or operator actions;
- existing immutable GCS objects, manifests, reports, or history;
- unrelated Scheduled Tasks, IAM, credentials, or secret values;
- official price/institutional/margin source definitions except the status-aware parsing and evidence preservation required here.

No production symbol list is embedded in code or documentation as configuration.

## Rollback boundary

- B1: raw cache v2 and lifecycle objects are immutable evidence and remain after code rollback; canonical cache v1 is untouched.
- B2: schema-v2 status artifacts are not selected until a v3 manifest is published.
- B3: restore the last verified pointer with its generation precondition before reverting dual-reader code. Old code must never see a v3 pointer.
- B4: presentation can be rolled back only after the selected data pointer is compatible with the destination code.
- Cloud Run: move traffic to the recorded previous revision if the deployed merge SHA fails service checks.
- Data/GCS: never delete evidence or immutable objects; rollback changes only generation-guarded pointers and traffic.
- Scheduled Tasks remain disabled throughout rollback and are enabled only after the restored public state passes acceptance.

## Plan self-review

- Each B stage names its exact production and test files, RED behavior, minimal GREEN, focused commands, and commit boundary.
- The implemented field names match across source, artifact, manifest, loader, dashboard, report, stock, and LINE paths.
- Suspension identity is lifecycle-only and round-trip verifiable.
- Cache and manifest compatibility paths are exact rather than permissive coercions.
- Rollout and rollback order preserve a working reader before selecting new data.
- Every decision is final; no symbol-specific production rule or contract conflict remains.
