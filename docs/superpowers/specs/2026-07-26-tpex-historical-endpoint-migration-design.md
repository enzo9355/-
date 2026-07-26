# TPEx Historical Endpoint Migration Design

## Status

Approved for implementation by the Phase 1E prompt. This design authorizes only the TPEx historical price and margin request-contract migration described below. It does not authorize Production recovery, historical backfill, source-cache promotion, GCS or Cloud Run mutation, LINE delivery, Scheduled Task changes, publication, or model promotion.

## 1. Problem statement

The TPEx historical price definition currently uses the legacy `stk_quote_result.php` endpoint. That endpoint accepts a requested historical date but can return the newest trading day instead. The existing parser correctly rejects that response because its top-level or table date does not match the requested date. The date validation must remain fail-closed.

The TPEx historical margin definition also uses a legacy PHP endpoint that is no longer the verified date-addressable contract. Phase 1E migrates only these two request contracts to the verified modern TPEx endpoints. TPEx institutional data stays on the existing individual-security endpoint because it is already date-addressable and produces the required 2,439 canonical rows for the verified diagnostic date.

## 2. Verified official contracts

### TPEx price

- URL: `https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes`
- Method: `GET`
- Parameters: `date=YYYY/MM/DD`, `response=json`
- Headers: `User-Agent: ABSORB/1.0`, `X-Requested-With: XMLHttpRequest`
- No Cookie, session, token, Referer, Authorization, or browser automation is required.
- The response proves the target date through top-level `date=YYYYMMDD` and table `date=ROC/MM/DD`, has `stat=ok`, and exposes the `上櫃股票行情` table.

### TPEx margin

- URL: `https://www.tpex.org.tw/www/zh-tw/margin/balance`
- Method: `GET`
- Parameters: `date=YYYY/MM/DD`, `response=json`
- Header: `User-Agent: ABSORB/1.0`
- No Cookie, session, token, Referer, Authorization, or browser automation is required.
- The response proves the target date through top-level `date=YYYYMMDD` and table `date=ROC/MM/DD`, has `stat=ok`, and exposes the `上櫃股票融資融券餘額` table.

### TPEx institutional

- URL remains `https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php`.
- Parameters remain `l=zh-tw`, `o=json`, `se=EW`, `t=D`, `d=ROC/MM/DD`, `s=0,asc`.
- `parse_tpex_institutional()` remains unchanged. The market-summary endpoint is not an acceptable substitute because it does not contain individual-security rows.

## 3. Source definition migration

Only `HISTORICAL_SOURCE_DEFINITIONS["tpex_price"].url` and `HISTORICAL_SOURCE_DEFINITIONS["tpex_margin"].url` change. All source identifiers, exchange and dataset labels, response kinds, response-size limits, ordering, and the four other source definitions remain unchanged.

The migration is deliberately an endpoint substitution rather than a new adapter, dataclass field, provider abstraction, or fallback path. The existing historical adapter already owns transport retry, response-size enforcement, JSON decoding, parser dispatch, coverage checks, snapshot assembly, and cache integration.

## 4. Source-specific parameter strategy

`_params()` continues to branch explicitly by source identifier:

- TWSE sources retain `date=YYYYMMDD`, `response=json` and their existing source-specific parameters.
- `tpex_price` returns exactly `{"date": "YYYY/MM/DD", "response": "json"}`.
- `tpex_margin` returns exactly `{"date": "YYYY/MM/DD", "response": "json"}`.
- `tpex_institutional` retains its existing ROC `d` parameter and legacy flags.

No generic TPEx date branch is introduced because the three TPEx sources now have two distinct official date contracts.

## 5. Source-specific header strategy

A private `_request_headers(source_id: str) -> dict[str, str]` helper returns the shared `User-Agent: ABSORB/1.0` header for every source and adds `X-Requested-With: XMLHttpRequest` only for `tpex_price`. `_request_payload()` calls this helper with `definition.source_id`.

`OfficialSourceDefinition` remains unchanged. Headers are request behavior, not immutable source metadata, and the helper satisfies the verified contract with the smallest local change.

## 6. Parser compatibility decision

`parse_tpex_price_report`, `parse_tpex_margin_report`, and `parse_tpex_institutional` remain unchanged. Sanitized synthetic fixtures model the verified modern response shape and prove that the current nested-table selection, schema fingerprints, field indexes, OHLC validation, margin normalization, duplicate rejection, coverage behavior, and exact target-date checks remain compatible.

Any fixture that proves a real incompatibility would require a new failing parser test and a separate parser-semantics decision. This implementation must stop rather than silently broaden table selection, weaken dates, accept ambiguous tables, or coerce invalid values to zero.

## 7. Parser-version decision

`PARSER_VERSION` remains `tw-official-historical-parser-v2`. The canonical parser semantics and row schema do not change; only the upstream URL, request parameters, and one required request header change.

A version bump would invalidate all six sources' existing content-addressed cache entries and is outside this phase. If implementation discovers that canonical parsing semantics must change, work stops before changing the version and reports the cache, migration, and rollback impact.

## 8. Cache compatibility

Policy A per-source cache behavior remains unchanged. Warm cache reads validate stored parser version, source date, hashes, metadata, row counts, and coverage before reuse. Because `PARSER_VERSION`, canonical rows, manifest schema, and cache directory schema do not change, verified existing cache entries remain reusable and warm reads continue to make zero requests.

Cold reads use the migrated endpoint contracts and store the same canonical representation. Two dates therefore still require twelve cold source requests and zero warm requests, with an identical warm manifest SHA-256.

## 9. Error handling

The existing bounded retry count, timeout, status handling, content-length and response-size checks, JSON validation, parser errors, coverage thresholds, cross-source overlap checks, and duplicate rejection remain unchanged.

Top-level and nested table dates must both equal the requested target date. A response for 2026-07-24 when 2026-07-16 was requested raises `ValueError`; it is never relabelled, cached, or used to assemble a snapshot. No fallback provider or live recovery path is added.

## 10. Test strategy

All tests are offline and use minimal sanitized synthetic payloads.

1. Assert the exact modern price and margin URLs and the unchanged institutional URL.
2. Assert exact parameters for all three TPEx sources on `2026-07-16`.
3. Record and assert request `source_id`, URL, date, params, and headers.
4. Assert the shared User-Agent for every source and `X-Requested-With` only where required.
5. Assert institutional requests contain no Cookie, Authorization, or Token.
6. Parse modern price and margin fixtures into exact canonical values.
7. Reject top-level and table-date mismatches independently for price and margin.
8. Preserve institutional Foreign, InvestmentTrust, and Dealer canonicalization.
9. Preserve the twelve-cold-request, zero-warm-request, and identical-manifest cache regression.
10. Preserve the bounded catch-up limit of ten sessions.
11. Run focused historical, cache, post-close CLI, and bulk tests; then the complete unittest suite and static gates.

No unit test makes a live TWSE, TPEx, FinMind, Supabase, GCS, Cloud Run, or LINE request.

## 11. Rollback

Before merge, close the Draft PR and delete the feature branch if the migration is rejected. After merge, revert the implementation commits or merge commit. Reversion restores the legacy price and margin URLs; the existing exact-date validation still causes the legacy price endpoint to fail closed for incorrect historical responses.

No Production cache or publication data is modified by this PR, so no data rollback is expected. Any accidental write under `D:\AbsorbData` requires an immediate stop, preservation of evidence, and a separate rollback proposal.

## 12. Non-goals

- Lifecycle, active-universe, exclusion-list, or all-market instrument classification redesign.
- Investigation of unrelated artifacts or symbols.
- Date-level staging transactions, cache transaction refactors, parser-version migration, backfill, or source-cache promotion.
- Local Observation execution, GCS, Cloud Run, LINE, FinMind, Supabase, publication, model promotion, or Scheduled Task changes.
- Changes to `MAX_CATCHUP_SESSIONS`, retries, timeouts, size limits, coverage thresholds, source ordering, manifest schema, cache paths, canonical row schema, or parser dispatch.
- New dependencies, browser automation, fallback providers, or unrelated refactoring.

## 13. Production safety boundaries

All implementation and testing occurs in the isolated worktree created from `0a3d78916b9a751b6c7d9b6a7aa8e00491d95edf`. The original checkout and its user-owned untracked files remain untouched.

Both `ABSORB-TW-PostClose` and `ABSORB-TW-PreMarket` Scheduled Tasks must remain Disabled before implementation and at completion. This work must not run `stock_papi.batch.tw_official_post_close_cli`, fetch the seven recovery dates, create Production source cache, run Local Observation, upload or publish artifacts, deploy Cloud Run, notify LINE, update `latest-TW.json`, or enable, start, trigger, or modify either task.

The implementation is ready for Antigravity validation only after focused and full local verification, static checks, independent code review, a Draft PR, required CI success, and a final safety check.
