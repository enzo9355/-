# TW Official Bulk Market-Data Source Design

## Status

Approved and implemented in Draft PR #15. Exact-head verification and Production rollout remain separate gates. This document does not authorize Production execution, GCS mutation, Scheduled Task changes, LINE delivery, historical publication backfill, model promotion, Task D, or changes to the four Production regression-readiness flags.

## Problem

The Taiwan post-close observation pipeline previously requested three FinMind datasets separately for every Taiwan symbol:

- `TaiwanStockPrice`
- `TaiwanStockInstitutionalInvestorsBuySell`
- `TaiwanStockMarginPurchaseShortSale`

That call graph required thousands of provider requests for a full universe. During the July 2026 incident, the anonymous FinMind quota was exhausted after roughly one hundred symbols and Production remained at 2026-07-16.

The repository already had a small official-exchange activity adapter, but it returned only trade value and volume and silently ignored failures. It was not sufficient for an observation-grade post-close pipeline.

## Goals

1. Make the normal Taiwan post-close network phase bulk-first and deterministic.
2. Fetch each required official dataset at most once per required trading session.
3. Preserve validated per-symbol historical artifacts and append only missing official daily rows.
4. Keep downstream quant feature contracts stable.
5. Remove FinMind credentials from the normal Production observation path.
6. Fail before the symbol loop and checkpoint mutation when a core official source is unavailable, stale, malformed, inconsistent, or materially incomplete.
7. Keep observation publication, coverage, hash, candidate, GCS, disclosure, and model-governance gates unchanged.

## Non-goals

- Rebuilding all historical Taiwan market data from official sources.
- Publishing reports for the missing 2026-07-17 through 2026-07-23 dates.
- Replacing GCS with Supabase.
- Using the colleague-provided Supabase project.
- Changing model training, prediction, backtesting, recommendation policy, or report design.
- Enabling Production tasks during implementation.
- Publishing fixtures, probes, samples, or partial results.
- Creating `backtests/v1/latest-TW.json`.

## Selected approach

Use date-addressable official TWSE and TPEx reports as the daily bulk source. Normalize the six market-wide responses into the FinMind-shaped contracts already consumed by quant code. Read existing per-symbol artifacts as local history and enrich only the missing trading sessions required to reach the explicit target date.

This makes the normal request count independent of the number of symbols. FinMind remains an optional future fallback, disabled by default and bounded by an explicit request budget.

## Authoritative sources

### TWSE

- Price report: `https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX`
- Institutional report: `https://www.twse.com.tw/rwd/zh/fund/T86`
- Margin report: `https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN`

### TPEx

- Price report: `https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php`
- Institutional report: `https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php`
- Margin report: `https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php`

All six requests include the explicit target date. Every response and nested target table must prove that date. The runner must not assign the requested date to undated or stale rows.

## Architecture

### 1. Official contracts and institutional parsers

`stock_papi/integrations/market_data/tw_official_bulk.py` owns:

- immutable source/result/request-budget contracts;
- strict symbol, date, and numeric normalization;
- TWSE and TPEx institutional normalization;
- duplicate-key rejection;
- structured, redacted source failures;
- FinMind fallback budget metadata, with fallback disabled by default.

Canonical price rows contain:

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

Institutional data is emitted as up to three logical rows per symbol and date:

- `Foreign`
- `InvestmentTrust`
- `Dealer`

Each row contains `date`, `stock_id`, `name`, `buy`, and `sell`. Dealer proprietary and hedge activity are aggregated into the dealer total. The TPEx parser uses the official group sequence and reads foreign total, investment trust, and dealer total rather than an unlabelled repeated-column guess.

Canonical margin rows contain:

```python
{
    "date": "YYYY-MM-DD",
    "stock_id": "2330",
    "MarginPurchaseTodayBalance": 0,
    "ShortSaleTodayBalance": 0,
}
```

### 2. Date-addressable historical report adapter

`stock_papi/integrations/market_data/tw_official_historical.py` owns:

- the six date-addressable source definitions;
- request parameters for Gregorian and ROC dates;
- response-size limits;
- bounded retry for transport errors and retryable 5xx failures;
- exact nested-table selection and field-index mappings;
- source-specific coverage gates;
- daily snapshot assembly;
- bounded multi-session snapshot-series assembly.

The implemented source mode is:

```text
tw_official_bulk_v2
```

The source schema version is separate from the source mode and participates in checkpoint identity.

### 3. TPEx institutional schema fingerprint

The TPEx institutional parser must select exactly one nested table with:

- title `三大法人買賣明細資訊`;
- `columnNum=25`;
- the exact verified sequence of 24 field cells;
- an exact table date matching the response and requested date;
- a non-empty data list;
- exactly 24 cells for every accepted list row.

The verified group sequence after code and name is:

1. foreign and Mainland investors excluding foreign dealers;
2. foreign dealers;
3. foreign and Mainland investors total;
4. investment trust;
5. dealer proprietary trading;
6. dealer hedging;
7. dealer total;
8. three-institution net total.

The compatibility rows use foreign-total buy/sell, investment-trust buy/sell, and dealer-total buy/sell.

### 4. Raw source cache

`stock_papi/integrations/market_data/tw_official_cache.py` stores verified canonical source rows under:

```text
D:\AbsorbData\source-cache\tw-official\v1\<date>\
```

Each cached source has:

- deterministic compressed canonical JSON;
- metadata JSON;
- canonical content SHA-256;
- compressed SHA-256;
- source identifier and URL without secret query material;
- fetch timestamp and source date;
- row and unique-symbol counts;
- parser version.

Writes are atomic. A valid cache hit causes zero network requests. Hash mismatch, date mismatch, parser-version mismatch, malformed metadata, or coverage below the current gate fails closed.

The source cache is not a published quant artifact and cannot update GCS.

### 5. Source-specific coverage gates

The read-only 2026-07-24 contract probe observed:

- TWSE institutional: 1,231 numeric symbols;
- TWSE margin: 1,059 numeric symbols;
- TPEx institutional: 794 numeric symbols;
- TPEx margin: 812 numeric symbols.

Production defaults deliberately use lower but non-trivial fail-closed minimums:

- TWSE institutional: 500;
- TWSE margin: 400;
- TPEx institutional: 300;
- TPEx margin: 300.

Price reports retain their market-specific minimum coverage gates. Tests may inject smaller explicit thresholds for compact fixtures, but Production defaults cannot silently fall back to one row.

### 6. Prefetch before symbol execution

For a TW post-close observation run, `stock_papi.batch.tw_official_post_close_cli` must complete these steps before entering `local_quant.run_market_batch()`:

1. Load and audit the Taiwan symbol universe and existing artifacts.
2. Resolve every missing trading session required to reach the explicit target date.
3. Refuse more than ten missing trading sessions.
4. Load or fetch all six official sources for every required session.
5. Validate dates, schemas, hashes, and source-specific coverage.
6. Build an immutable official snapshot series.
7. Add source mode, source schema, snapshot dates, manifest hash, universe identity, and request budget to batch identity.
8. Only then invoke the existing symbol loop.

A core source failure therefore occurs before the new batch can reset `progress.json` or write the first symbol artifact.

### 7. Incremental per-symbol history

`stock_papi/quant/tw_artifact_audit.py` and `stock_papi/quant/tw_incremental.py` provide the new local-history path.

The path must:

- load and validate an existing per-symbol artifact;
- ensure declared `as_of` matches the latest daily row;
- reject history later than the target date;
- append each missing official trading session in order;
- verify an existing same-date row rather than duplicating it;
- fail on a same-date integrity mismatch;
- sort and deduplicate by date;
- preserve the existing analysis window and feature contracts;
- attach official source lineage to the generated immutable symbol artifact;
- avoid any normal per-symbol FinMind request.

A missing or invalid local history remains a symbol failure unless a separately authorized bootstrap path exists. Official bulk data supplies daily increments; it is not an implicit two-year history rebuild.

### 8. Catch-up is not publication backfill

A stale local artifact may require several official trading sessions before the target date can be analyzed. The implementation may enrich at most ten missing trading sessions in local per-symbol history.

This does not create historical report publications. The post-close run still targets one explicit newest date, and only that target may advance the latest quant, dashboard, and report pointers after every existing publication gate passes.

### 9. Request budget and fallback

For one missing trading session:

- cold minimum: six official requests;
- bounded worst case: twelve attempts under the two-attempt retry contract;
- warm valid cache: zero requests.

For ten missing sessions, the hard upper planning bounds are sixty cold requests and 120 attempts. The count remains independent of symbol count.

FinMind fallback is disabled by default. A future explicitly enabled fallback must:

- compute capacity before any fallback request;
- include retries in the worst-case budget;
- remain at or below 20 requests per run;
- apply only to a small set of symbols missing from otherwise valid official coverage;
- preserve field-level lineage;
- fail before the fallback network phase when the budget is exceeded;
- never make an incomplete core market-wide source publishable.

### 10. Checkpoint identity

The Taiwan observation batch identity includes:

- target market date;
- product mode;
- source mode `tw_official_bulk_v2`;
- official snapshot-series manifest SHA-256;
- included official snapshot dates;
- universe identity;
- source schema/code version;
- request-budget metadata.

FinMind-per-symbol checkpoints and earlier official-source modes are incompatible. They must not be resumed or overwritten as if they were the same batch. Incident backups remain untouched.

### 11. Publication and governance

The existing publication path remains authoritative:

- all symbol artifacts must satisfy existing failure thresholds;
- `publish_market_snapshot()` remains the quant publication gate;
- candidate build and promotion remain separate;
- immutable objects and hashes remain mandatory;
- GCS upload remains fail-closed;
- exact report disclosure remains unchanged;
- all four Production regression-readiness flags remain `False`;
- `backtests/v1/latest-TW.json` remains forbidden.

## Validation rules

A source snapshot is rejected for:

- HTTP or JSON failure after bounded retry;
- response larger than the configured limit;
- missing or inconsistent response/table date;
- missing or reordered required fields;
- malformed symbol identifiers;
- conflicting duplicate primary keys;
- invalid numeric normalization;
- impossible price relationships;
- non-positive close for a normally traded row;
- zero meaningful securities;
- price coverage below its market threshold;
- institutional or margin coverage below its source-specific threshold;
- incompatible cache metadata or hashes;
- inconsistent cross-source identity.

Suspended, newly listed, non-marginable, and zero-volume securities require explicit parser semantics rather than blanket rejection.

## Testing strategy

Normal repository tests use fixtures and must not contact TWSE, TPEx, FinMind, GCS, Cloud Run, Supabase, or LINE.

Focused coverage includes:

1. TWSE and TPEx institutional mapping.
2. Exact TPEx 24-field schema fingerprint.
3. TWSE and TPEx price/margin nested-table mapping.
4. Gregorian and ROC date normalization.
5. Numeric placeholder handling and duplicate rejection.
6. Deterministic cache round trip and hash failure.
7. Six calls per cold session and zero calls per warm session.
8. Source-specific coverage rejection.
9. Incremental history, same-date verification, and future-date rejection.
10. Source failure before local batch execution.
11. Checkpoint identity and source lineage.
12. FinMind-free compatibility fetches.
13. Existing observation, scheduler, FinMind-error, publication, report, and governance tests.
14. Full test suite, Python compilation, JavaScript syntax, PowerShell 5.1 parsing, diff check, credential scan, readiness flags, and forbidden-artifact checks.

## Live verification boundary

GitHub code verification may use read-only public schema probes but must not run the Production pipeline.

Antigravity must later:

1. update Runner V2 to the merged main;
2. keep both TW Scheduled Tasks disabled;
3. perform a controlled single-date six-source probe;
4. determine the actual missing trading sessions from validated local artifacts and the trading calendar;
5. prove the catch-up count is at most ten;
6. build one target-date local observation without `-PublishObservation`;
7. validate source cache, per-symbol history, manifests, dashboard, reports, PDF, hashes, and lineage;
8. promote and upload only after every existing gate passes;
9. verify the website before changing tasks;
10. point only the two TW tasks to Runner V2 and enable them with missed-run prevention.

It must not use FinMind, Supabase, LINE, sample data, model promotion, Task D, or historical report publication during this recovery.

## Success criteria

- Official network requests are fixed per missing trading session and independent of symbol count.
- The normal Taiwan observation run requires no FinMind credential.
- Valid local histories reach the explicit target trading date using verified official increments.
- Existing historical rows are preserved.
- Core source failure occurs before symbol-loop and checkpoint mutation.
- Latest publication advances only after all existing gates pass.
- All governance constraints remain intact.
- No Production operation occurs during the code PR.
