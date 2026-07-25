# TW Official Bulk Market-Data Source Design

## Status

Proposed design for review. This document defines a code-only migration path; it does not authorize Production execution, GCS mutation, Scheduled Task changes, LINE delivery, historical backfill, model promotion, or changes to the four Production regression-readiness flags.

## Problem

The Taiwan post-close observation pipeline currently obtains three FinMind datasets separately for every symbol:

- `TaiwanStockPrice`
- `TaiwanStockInstitutionalInvestorsBuySell`
- `TaiwanStockMarginPurchaseShortSale`

`stock_papi.quant.data.get_data()` requests all three datasets for each Taiwan symbol. A full universe therefore requires thousands of provider requests. The July 2026 incident demonstrated that the anonymous FinMind quota is exhausted after roughly one hundred symbols, preventing the daily market snapshot from advancing beyond 2026-07-16.

The repository already has a minimal official exchange adapter in `stock_papi/integrations/market_data/tw_exchange.py`, but it only returns trade value and volume and silently ignores source failures. It is not sufficient for the observation pipeline.

## Goals

1. Make the normal Taiwan post-close network phase bulk-first and deterministic.
2. Fetch each required official market dataset at most once per target trading date.
3. Preserve existing per-symbol historical artifacts and append only the missing official daily row.
4. Keep downstream quant feature contracts stable.
5. Use FinMind only as a bounded, explicitly budgeted fallback; the normal Production path must not require FinMind credentials.
6. Fail before the symbol loop when an official core dataset is unavailable, stale, malformed, inconsistent, or incomplete.
7. Keep the existing observation-only publication, hash, coverage, candidate, GCS, disclosure, and model-governance gates unchanged.

## Non-goals

- Rebuilding historical Taiwan data from official sources.
- Backfilling the missing 2026-07-17 through 2026-07-23 publication dates.
- Replacing GCS with Supabase.
- Using the colleague-provided Supabase project.
- Modifying model training, prediction, backtesting, recommendation policy, or report design.
- Enabling Production tasks during implementation.
- Publishing any fixture, sample, probe, or partial result.

## Considered approaches

### A. FinMind authentication only

Register or obtain credentials and increase the request allowance.

This requires the smallest code change, but the current per-symbol call graph still requires thousands of requests per run. A normal authenticated allowance is not a reliable capacity contract for the full market, so this does not solve the architecture problem.

### B. Supabase as a replacement data provider

Read Taiwan market data from the colleague-provided Supabase project.

This is rejected because the project schema, source lineage, update process, RLS policies, ownership, completeness, and authorization are unknown. Supabase is a storage service, not evidence of a trustworthy market-data source.

### C. Official bulk snapshots plus incremental local history

Fetch current-day market-wide snapshots from TWSE and TPEx, normalize them to the existing FinMind-shaped interfaces, and append the verified day to existing local historical artifacts.

This is the selected approach. It reduces the normal daily exchange request count to a small fixed number, preserves the historical rows already stored in ABSORB per-symbol artifacts, and keeps source lineage under ABSORB control.

## Authoritative sources

### TWSE

- Price snapshot: `https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL`
- Margin snapshot: `https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN`
- Institutional snapshot: `https://www.twse.com.tw/fund/T86?response=json&date={YYYYMMDD}&selectType=ALL`

### TPEx

- Price snapshot: `https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes`
- Margin snapshot: `https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_balance`
- Institutional snapshot: `https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading`

The TWSE and TPEx OpenAPI resources are latest-snapshot sources, not a historical backfill service. T86 is requested for the explicit target date. Every source still must prove the target trading date through its response contract; the runner may not assign the requested date to undated or stale rows.

Endpoint paths must be isolated in one source-definition table. Parsers must not depend on column positions when the response supplies field names. The implementation must accept documented name aliases only; unknown schema changes fail closed.

## Architecture

### 1. Official source adapter

Add a focused integration module, provisionally:

`stock_papi/integrations/market_data/tw_official_bulk.py`

It owns:

- HTTP request execution with timeout, bounded retry for transient transport and 5xx failures, and explicit response-size limits.
- TWSE and TPEx schema parsing.
- ROC/Gregorian date normalization where required.
- Numeric normalization for commas, signs, blanks, and non-trading placeholders.
- Required-column, duplicate-key, source-date, and row-count validation.
- Redacted, structured source failures.

The adapter returns an immutable daily snapshot object containing:

- target market date
- source metadata and content hashes
- price rows keyed by symbol
- institutional rows keyed by symbol
- margin rows keyed by symbol
- source request count
- per-source coverage metrics

No Production pointer, per-symbol artifact, checkpoint, report, or notification is written by this module.

### 2. Canonical compatibility layer

Official rows are normalized to the shapes already consumed by `stock_papi.quant.data`.

Price for one symbol:

```python
{
    "date": "YYYY-MM-DD",
    "stock_id": "2330",
    "open": 0.0,
    "max": 0.0,
    "min": 0.0,
    "close": 0.0,
    "Trading_Volume": 0,
}
```

Institutional data is emitted as up to three logical rows per symbol and date so the existing `merge_chip_data()` grouping remains valid:

- `Foreign`
- `InvestmentTrust`
- `Dealer`

Each row contains `date`, `stock_id`, `name`, `buy`, and `sell`. Dealer proprietary and hedge components are summed into the dealer row. Foreign proprietary activity, where separately reported, is included in the foreign row and documented in source metadata.

Margin data:

```python
{
    "date": "YYYY-MM-DD",
    "stock_id": "2330",
    "MarginPurchaseTodayBalance": 0,
    "ShortSaleTodayBalance": 0,
}
```

The compatibility layer prevents downstream model and report code from branching on TWSE versus TPEx field names.

### 3. Raw snapshot cache

Store verified official responses under the Data Root, never in Git:

`D:\AbsorbData\source-cache\tw-official\v1\<date>\`

Each source has:

- compressed canonical JSON payload
- metadata JSON
- content SHA-256
- compressed SHA-256
- source URL identifier without query secrets
- fetched timestamp
- source date
- row and symbol counts
- parser/schema version

Writes are atomic. An existing valid cache prevents a network request. Hash mismatch, source-date mismatch, parser-version incompatibility, or malformed metadata fails closed.

The source cache is not a published quant artifact and cannot update GCS.

### 4. Prefetch before symbol execution

For a TW post-close observation run, `local_quant.main()` must build or load the complete official daily snapshot before calling `run_market_batch()`.

Required behavior:

1. Resolve the explicit `--target-market-date`.
2. Load/fetch all six source datasets.
3. Verify every source represents that exact date.
4. Validate market coverage and cross-source identities.
5. Build the immutable lookup context.
6. Only then enter the symbol loop.

A provider-wide official-source failure therefore occurs before `progress.json` is reset or the first symbol artifact is written.

### 5. Incremental per-symbol history

The latest valid per-symbol artifact supplies historical rows before the target date. For each symbol, the official snapshot contributes the target-day price, institutional, and margin data. This is a new read path over existing artifacts; the current FinMind-backed `get_data()` path does not yet consume those artifacts as history.

The data path must:

- load and validate the existing local per-symbol artifact when available;
- reject an existing latest date later than the target date;
- append the official row only when that date is missing;
- replace an existing same-date row only if an explicit integrity comparison proves the prior row is stale and the operation is recorded;
- sort and deduplicate by date;
- preserve at least the current 730-day analysis window;
- write only through existing atomic per-symbol artifact handling;
- retain source lineage showing local historical and official-daily components.

When no valid local history exists, the normal Production observation run fails that symbol unless a separately authorized, budgeted bootstrap source is available. The implementation must not issue a FinMind history request merely because one symbol enters the batch.

### 6. TAIEX and market context

The official stock snapshot does not provide the complete historical TAIEX series used by current market-context features. Existing validated local history and current non-FinMind market-context paths may remain in place.

The official-source migration must isolate the three FinMind stock datasets only. It must not silently change S&P 500, TAIEX, ETF50, VIX, yfinance cross-check, or option-context behavior.

### 7. Bounded fallback

FinMind fallback is optional and disabled by default for Production observation runs.

If enabled explicitly in a future operator flow:

- a request-budget object must be computed before any fallback call;
- the default hard limit is 20 requests per run;
- retries count toward the worst-case budget;
- fallback is allowed only for a small set of symbols genuinely absent from a validated official source;
- source lineage must identify every fallback field;
- exceeding the budget fails before the fallback network phase;
- fallback cannot turn a missing core market-wide source into a publishable run.

### 8. Checkpoint identity

The TW market-batch identity must include:

- target market date
- product mode
- source mode: `tw_official_bulk_v1`
- official snapshot manifest SHA-256
- universe identity
- code/source schema version

A checkpoint from the FinMind per-symbol mode is incompatible and must not be resumed or overwritten as if it were the same batch. Existing checkpoint archives and incident backups remain untouched.

### 9. Publication and governance

The existing publication path remains unchanged:

- all symbol artifacts must complete within existing failure thresholds;
- `publish_market_snapshot()` remains the quant publication gate;
- candidate building and promotion remain separate operations;
- immutable objects and hashes remain mandatory;
- GCS upload remains fail-closed;
- exact report disclosure remains unchanged;
- all four Production regression-readiness flags remain `False`;
- `backtests/v1/latest-TW.json` remains forbidden.

## Validation rules

A source snapshot is rejected when any of the following holds:

- HTTP or JSON failure after bounded retry
- response exceeds configured size
- target date is absent or inconsistent
- required fields are missing
- symbol identifiers are malformed
- duplicate primary keys exist after documented aggregation
- numeric fields cannot be normalized
- price OHLC relationships are impossible
- close is non-positive for a normally traded row
- source returns zero meaningful securities
- TWSE or TPEx price coverage falls below a conservative fixture-backed threshold
- institutional or margin coverage is unexpectedly empty
- cross-source symbol identity is inconsistent beyond a documented tolerance

Suspended, newly listed, non-marginable, and zero-volume securities need explicit parser semantics rather than blanket rejection.

## Testing strategy

All repository tests use fixtures; no normal unit test contacts TWSE, TPEx, FinMind, GCS, Cloud Run, Supabase, or LINE.

Required focused tests include:

1. TWSE price mapping.
2. TWSE T86 institutional mapping.
3. TWSE margin mapping.
4. TPEx price mapping.
5. TPEx institutional mapping.
6. TPEx margin mapping.
7. ROC date conversion and exact target-date enforcement.
8. Numeric normalization and placeholder handling.
9. Duplicate and malformed row rejection.
10. Dealer component aggregation.
11. Foreign-row compatibility with `foreign_flow_mask()`.
12. Official snapshot cache hit produces zero HTTP calls.
13. Hash mismatch fails closed.
14. Atomic cache writes.
15. Six source definitions are called at most once each.
16. Source failure prevents the symbol loop.
17. A universe of two thousand symbols does not create per-symbol official HTTP calls.
18. Incremental append preserves historical rows.
19. Same-date duplicate handling is deterministic.
20. Future local history fails closed.
21. Missing-symbol fallback is disabled by default.
22. Fallback budget is enforced before requests.
23. Checkpoint identity rejects the old FinMind mode.
24. Existing observation publication tests remain green.
25. Existing FinMind structured-error tests remain green.
26. Full test suite, Python compilation, JavaScript syntax, PowerShell 5.1 parser, diff check, secret scan, readiness flags, and forbidden-artifact checks pass.

## Live verification boundary

Code implementation and fixture tests can be completed in GitHub without Production access.

Antigravity must later perform a controlled local probe using no more than the six official source requests. The probe records only safe metadata:

- HTTP status
- response size
- row count
- unique symbol count
- source date
- required-field result
- duplicate count
- elapsed time

It must not run the full market pipeline, write Production pointers, enable tasks, upload GCS, deploy Cloud Run, or send LINE until the source contracts and local incremental build are independently validated.

## Rollout

1. Merge the code only after exact-head Windows Python 3.10 verification.
2. Antigravity updates Runner V2 to the merged main.
3. Perform the six-request official-source probe.
4. Build one local target-date observation without publication.
5. Validate per-symbol artifacts, manifest, dashboard, reports, hashes, and lineage.
6. Promote and upload only after all existing gates pass.
7. Verify the website.
8. Point only the two TW Scheduled Tasks at Runner V2 and enable them with missed-run prevention.

## Success criteria

- Normal post-close official-source requests are fixed and independent of symbol count.
- No FinMind credential is required for the normal TW observation run.
- The target trading date appears in validated local symbol histories.
- Existing historical rows are preserved.
- Core source failure occurs before the symbol loop and before checkpoint mutation.
- All existing publication and governance gates remain intact.
- No Production operation occurs during the code PR.
