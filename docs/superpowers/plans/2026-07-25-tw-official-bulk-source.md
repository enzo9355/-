# TW Official Bulk Market-Data Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the normal Taiwan post-close FinMind-per-symbol data path with a fixed-request TWSE/TPEx bulk snapshot, incremental local history, and fail-closed prefetch before the symbol loop.

**Architecture:** A new official-source integration boundary fetches and validates six market-wide datasets, persists a content-addressed local source cache, and exposes a canonical immutable snapshot keyed by symbol. `local_quant.py` builds this snapshot before checkpoint mutation, reconstructs each symbol's 730-day input frame from its existing local artifact plus the official target-day row, then passes that frame through the existing feature and publication pipeline. FinMind remains available to unrelated interactive code, but Production observation mode does not call it and bounded fallback remains disabled by default.

**Tech Stack:** Python 3.10, `requests`, `pandas`, standard-library `dataclasses`, `gzip`, `hashlib`, `json`, `pathlib`, `unittest`, Windows PowerShell 5.1 validation.

## Global Constraints

- Normal TW observation runs must use source mode `tw_official_bulk_v1` and require no FinMind credentials.
- Fetch exactly six official source definitions at most once per target date: TWSE price, TWSE institutional, TWSE margin, TPEx price, TPEx institutional, TPEx margin.
- Official-source failure must occur before `run_market_batch()`, before `progress.json` mutation, and before symbol artifact writes.
- Existing per-symbol history remains authoritative for dates before the target date; append only the missing target-day row.
- Existing GCS, candidate, promotion, report, disclosure, model-governance, and publication gates remain unchanged.
- FinMind fallback is disabled by default and may never repair a missing market-wide core source.
- Maximum future FinMind fallback budget is 20 requests per run, with retries counted before network execution.
- No live HTTP call in unit tests.
- No Supabase dependency.
- No historical publication backfill.
- All four Production regression-readiness flags remain `False`.
- `backtests/v1/latest-TW.json` remains absent.
- No Production task, GCS, Cloud Run, LINE, model promotion, or Task D operation is part of this plan.

---

## File Structure

**Create**

- `stock_papi/integrations/market_data/tw_official_bulk.py` — source definitions, structured failures, six parsers, canonical snapshot types, HTTP orchestration, coverage validation.
- `stock_papi/integrations/market_data/tw_official_cache.py` — atomic compressed source-cache writes, metadata/hash validation, cache reads.
- `stock_papi/quant/tw_incremental.py` — validate existing TW stock artifacts, reconstruct historical input frames, append/verify the official target-day row, preserve lineage.
- `tests/fixtures/tw_official/` — minimal redacted JSON fixtures for all six source schemas and malformed variants.
- `tests/test_tw_official_bulk.py` — source parser, orchestration, request-count, validation, and cache tests.
- `tests/test_tw_incremental.py` — artifact-history reconstruction and target-row append tests.

**Modify**

- `local_quant.py` — prefetch official snapshot before `run_market_batch()`, pass reconstructed frames to `build_stock_snapshot()`, add source identity to checkpoint and artifact lineage.
- `tests/test_local_quant_batch.py` — prefetch-before-loop, checkpoint incompatibility, and zero-per-symbol-official-request tests.
- `tests/test_local_quant.py` — CLI observation-mode wiring and failure-before-mutation tests.
- `tests/test_local_quant_publish.py` — lineage remains non-published metadata and existing publish gates remain unchanged.
- `docs/superpowers/specs/2026-07-25-tw-official-bulk-source-design.md` — update status to implemented only after final verification.

---

### Task 1: Define Official Source Contracts and Structured Failure Model

**Files:**
- Create: `stock_papi/integrations/market_data/tw_official_bulk.py`
- Test: `tests/test_tw_official_bulk.py`

**Interfaces:**
- Produces: `OfficialSourceFailure`, `OfficialSourceDefinition`, `OfficialSourceResult`, `OfficialDailySnapshot`, `SOURCE_DEFINITIONS`, `parse_number()`, `normalize_symbol()`, `normalize_market_date()`.
- Consumes: `requests.Session`-compatible object and `datetime.date`.

- [ ] **Step 1: Write failing contract tests**

Add tests that import the new names and assert:

```python
from datetime import date

from stock_papi.integrations.market_data.tw_official_bulk import (
    OfficialSourceDefinition,
    OfficialSourceFailure,
    SOURCE_DEFINITIONS,
    normalize_market_date,
    normalize_symbol,
    parse_number,
)


def test_source_definitions_cover_exact_six_core_sources():
    assert tuple(SOURCE_DEFINITIONS) == (
        "twse_price",
        "twse_institutional",
        "twse_margin",
        "tpex_price",
        "tpex_institutional",
        "tpex_margin",
    )
    assert all(isinstance(item, OfficialSourceDefinition) for item in SOURCE_DEFINITIONS.values())


def test_normalizers_are_fail_closed():
    assert normalize_symbol(" 2330 ") == "2330"
    assert normalize_market_date("115/07/24") == date(2026, 7, 24)
    assert normalize_market_date("2026-07-24") == date(2026, 7, 24)
    assert parse_number("1,234") == 1234.0
    with self.assertRaises(ValueError):
        normalize_symbol("TOTAL")
    with self.assertRaises(ValueError):
        parse_number("--")
```

Add a failure test asserting `OfficialSourceFailure` exposes only `source_id`, `category`, `http_status`, `safe_message`, and `retryable`; it must not retain response bodies or authorization values.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_tw_official_bulk -v
```

Expected: import failure because `tw_official_bulk.py` does not exist.

- [ ] **Step 3: Implement the minimal contracts**

Create immutable dataclasses:

```python
@dataclass(frozen=True)
class OfficialSourceDefinition:
    source_id: str
    market: str
    dataset: str
    url: str
    response_kind: str
    max_bytes: int


@dataclass(frozen=True)
class OfficialSourceResult:
    source_id: str
    market: str
    dataset: str
    target_date: date
    rows: tuple[dict, ...]
    symbol_count: int
    content_sha256: str
    response_size_bytes: int


@dataclass(frozen=True)
class OfficialDailySnapshot:
    target_date: date
    price_by_symbol: Mapping[str, Mapping[str, object]]
    institutional_by_symbol: Mapping[str, tuple[Mapping[str, object], ...]]
    margin_by_symbol: Mapping[str, Mapping[str, object]]
    source_results: Mapping[str, OfficialSourceResult]
    manifest_sha256: str
    request_count: int
```

Use exact official source URLs from the approved spec. Use response limits of 20 MiB per source initially. Implement numeric normalization for commas, Unicode signs, parentheses negatives, blanks, and documented non-trading placeholders. Only placeholders explicitly passed as `allow_empty=True` may become `None`; required numeric fields fail closed.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_official_bulk -v
```

Expected: contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add stock_papi/integrations/market_data/tw_official_bulk.py tests/test_tw_official_bulk.py
git commit -m "test(data): define TW official source contracts"
```

---

### Task 2: Implement Six Canonical Fixture Parsers

**Files:**
- Modify: `stock_papi/integrations/market_data/tw_official_bulk.py`
- Create: `tests/fixtures/tw_official/twse_price.json`
- Create: `tests/fixtures/tw_official/twse_institutional.json`
- Create: `tests/fixtures/tw_official/twse_margin.json`
- Create: `tests/fixtures/tw_official/tpex_price.json`
- Create: `tests/fixtures/tw_official/tpex_institutional.json`
- Create: `tests/fixtures/tw_official/tpex_margin.json`
- Modify: `tests/test_tw_official_bulk.py`

**Interfaces:**
- Produces: `parse_twse_price()`, `parse_twse_institutional()`, `parse_twse_margin()`, `parse_tpex_price()`, `parse_tpex_institutional()`, `parse_tpex_margin()`.
- Produces canonical rows compatible with `stock_papi.quant.data.merge_chip_data()`.

- [ ] **Step 1: Add redacted fixtures with documented field aliases**

Each fixture must contain at least two valid securities, one suspended/blank-price row where appropriate, and no real user/account data. Preserve official response structure, including TWSE T86 `fields` + `data` shape when applicable.

Price output must be:

```python
{
    "date": "2026-07-24",
    "stock_id": "2330",
    "open": 1130.0,
    "max": 1140.0,
    "min": 1120.0,
    "close": 1135.0,
    "Trading_Volume": 12345678,
}
```

Institutional output must contain exactly the logical names `Foreign`, `InvestmentTrust`, and `Dealer` when the source exposes those categories:

```python
{
    "date": "2026-07-24",
    "stock_id": "2330",
    "name": "Foreign",
    "buy": 1000,
    "sell": 700,
}
```

Margin output must be:

```python
{
    "date": "2026-07-24",
    "stock_id": "2330",
    "MarginPurchaseTodayBalance": 5000,
    "ShortSaleTodayBalance": 200,
}
```

- [ ] **Step 2: Write failing parser tests**

Tests must cover:

- exact target-date enforcement;
- field aliases documented by TWSE/TPEx responses;
- dealer proprietary + hedge aggregation;
- foreign proprietary inclusion when separately exposed;
- ROC and Gregorian date normalization;
- malformed symbol rejection;
- duplicate `(date, stock_id, category)` rejection after documented aggregation;
- impossible OHLC rejection (`high < max(open, close)` or `low > min(open, close)`);
- suspended rows excluded from tradable price output without rejecting the whole source;
- institutional and margin rows may omit non-eligible symbols without fabricating zeros.

- [ ] **Step 3: Run parser tests and verify RED**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialParserTests -v
```

Expected: parser functions missing.

- [ ] **Step 4: Implement parser dispatch and alias maps**

Implement each parser with explicit alias tuples, for example:

```python
TWSE_PRICE_FIELDS = {
    "symbol": ("Code", "證券代號"),
    "open": ("OpeningPrice", "開盤價"),
    "high": ("HighestPrice", "最高價"),
    "low": ("LowestPrice", "最低價"),
    "close": ("ClosingPrice", "收盤價"),
    "volume": ("TradeVolume", "成交股數"),
}
```

Do not accept arbitrary fuzzy matches. Implement a helper that requires exactly one known alias or raises a schema failure. Aggregate documented dealer components before duplicate validation. Return tuples sorted by symbol and category for deterministic hashing.

- [ ] **Step 5: Run parser tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialParserTests -v
```

Expected: all six parser groups pass.

- [ ] **Step 6: Commit**

```bash
git add stock_papi/integrations/market_data/tw_official_bulk.py tests/test_tw_official_bulk.py tests/fixtures/tw_official
git commit -m "feat(data): normalize TWSE and TPEx bulk snapshots"
```

---

### Task 3: Add Atomic Content-Addressed Official Source Cache

**Files:**
- Create: `stock_papi/integrations/market_data/tw_official_cache.py`
- Modify: `tests/test_tw_official_bulk.py`

**Interfaces:**
- Consumes: `OfficialSourceDefinition`, raw response bytes, parsed canonical rows, target date, parser version.
- Produces: `OfficialCacheEntry`, `load_cached_source()`, `store_cached_source()`.

- [ ] **Step 1: Write failing cache tests**

Test root layout:

```text
<root>/source-cache/tw-official/v1/2026-07-24/
    twse_price-<content_sha256>.json.gz
    twse_price.metadata.json
```

Tests must assert:

- deterministic gzip (`mtime=0`);
- atomic temporary-file replacement;
- metadata includes source id, target date, row count, symbol count, canonical content hash, compressed hash, parser version, fetched timestamp, and source URL identifier;
- metadata contains no authorization, cookie, query token, username, or password fields;
- a valid cache returns rows without invoking HTTP;
- compressed-hash mismatch fails closed;
- canonical-content mismatch fails closed;
- target-date mismatch fails closed;
- parser-version mismatch is a cache miss, not silent reuse;
- a temporary file never becomes a valid cache entry.

- [ ] **Step 2: Run cache tests and verify RED**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialCacheTests -v
```

Expected: cache module missing.

- [ ] **Step 3: Implement cache module**

Use canonical UTF-8 JSON with sorted keys and compact separators for content hashing. Use `gzip.GzipFile(filename="", mtime=0)`. Write payload first, fsync, atomically replace; then write metadata atomically. Cache reads validate file size, both hashes, schema version, parser version, target date, source id, row count, and symbol count before returning.

- [ ] **Step 4: Run cache tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialCacheTests -v
```

Expected: all cache tests pass.

- [ ] **Step 5: Commit**

```bash
git add stock_papi/integrations/market_data/tw_official_cache.py tests/test_tw_official_bulk.py
git commit -m "feat(data): cache verified official market snapshots"
```

---

### Task 4: Build the Six-Request Snapshot Orchestrator

**Files:**
- Modify: `stock_papi/integrations/market_data/tw_official_bulk.py`
- Modify: `tests/test_tw_official_bulk.py`

**Interfaces:**
- Consumes: `build_official_daily_snapshot(root, target_date, session=None, now=None)`.
- Produces: one `OfficialDailySnapshot` with immutable mappings and a manifest hash.
- Uses: `load_cached_source()` and `store_cached_source()`.

- [ ] **Step 1: Write failing orchestration tests**

Tests must use a fake session and assert:

- all six sources requested at most once on a cold cache;
- zero requests on a warm valid cache;
- only cache-missing sources are requested;
- timeout and 500/502/503/504 receive at most two bounded attempts with injected sleep;
- 400/401/403/404/422/429 are not retried;
- response content length and actual bytes cannot exceed source `max_bytes`;
- invalid JSON fails before any snapshot is returned;
- one source failure prevents partial snapshot return;
- target-date mismatch in any source rejects the whole snapshot;
- TWSE and TPEx price symbol coverage meets conservative fixture-backed minima;
- institutional or margin source with zero meaningful symbols rejects the whole snapshot;
- manifest hash is deterministic regardless of source completion order;
- `request_count` counts every network attempt, including retries.

- [ ] **Step 2: Run orchestration tests and verify RED**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialOrchestratorTests -v
```

Expected: orchestrator missing.

- [ ] **Step 3: Implement request and validation flow**

For every source:

1. Attempt validated cache load.
2. If absent, perform bounded HTTP GET using the source definition.
3. For T86 only, construct explicit `date=YYYYMMDD`, `selectType=ALL`, `response=json` parameters without logging the complete URL.
4. Parse the official response.
5. Validate target date and coverage.
6. Store canonical rows in source cache.
7. Combine source results only after all six pass.

Combine TWSE and TPEx maps with duplicate symbol rejection across markets. Freeze nested mappings using `MappingProxyType` and tuples. Compute snapshot manifest hash from source ids, source content hashes, target date, parser version, and coverage metrics.

- [ ] **Step 4: Run focused orchestration tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_official_bulk -v
```

Expected: all contract, parser, cache, and orchestration tests pass.

- [ ] **Step 5: Commit**

```bash
git add stock_papi/integrations/market_data/tw_official_bulk.py tests/test_tw_official_bulk.py
git commit -m "feat(data): prefetch fixed-request TW official snapshot"
```

---

### Task 5: Reconstruct Incremental Symbol Frames from Existing Artifacts

**Files:**
- Create: `stock_papi/quant/tw_incremental.py`
- Create: `tests/test_tw_incremental.py`

**Interfaces:**
- Consumes: `build_incremental_tw_frame(root, pipeline, symbol, target_date, snapshot, days=730)`.
- Produces: `IncrementalTWFrame(frame, lineage)` where `frame` is ready for `pipeline.calc_all()` and `lineage` is JSON-safe.

- [ ] **Step 1: Write failing artifact-history tests**

Create temporary gzip artifacts using the existing schema:

```python
{
    "schema_version": 1,
    "market": "TW",
    "symbol": "2330",
    "as_of": "2026-07-23",
    "daily": [
        {
            "Date": "2026-07-23T00:00:00.000",
            "Open": 1100.0,
            "High": 1120.0,
            "Low": 1090.0,
            "Close": 1110.0,
            "Volume": 1000,
            "InstitutionalNet": 100,
            "ForeignNet": 80,
            "MarginBalance": 5000,
            "ShortBalance": 200,
        }
    ],
}
```

Tests must assert:

- missing artifact fails closed without calling FinMind;
- malformed gzip, oversized expansion, schema mismatch, symbol mismatch, and invalid JSON fail closed;
- artifact latest date after target date fails closed;
- artifact history is clipped to the requested 730-calendar-day window;
- target row is appended once when absent;
- same-date row is accepted only if official OHLCV and chip values are exactly equivalent after numeric normalization;
- same-date mismatch fails closed and does not overwrite history;
- rows remain sorted and unique by `Date`;
- historical `InstitutionalNet`, `ForeignNet`, `MarginBalance`, and `ShortBalance` are preserved;
- target-day institutional values are computed as total net and foreign net from canonical institutional rows;
- target-day margin balances use canonical margin values;
- lineage contains historical artifact SHA-256, official manifest SHA-256, target date, source mode, and symbol, but no secret fields;
- the helper invokes existing `pipeline.fetch_yfinance_price_history`, market/ETF context, option context, price-quality, market-context, option-context, and clean functions, but never invokes `pipeline.fetch_finmind_dataset` or `pipeline.get_data`.

- [ ] **Step 2: Run incremental tests and verify RED**

Run:

```bash
python -m unittest tests.test_tw_incremental -v
```

Expected: module missing.

- [ ] **Step 3: Implement validated artifact loading**

Reuse the same compressed and uncompressed size limits as `local_quant.py`. Parse only JSON-safe values. Build the historical base frame from:

```python
(
    "Date", "Open", "High", "Low", "Close", "Volume",
    "InstitutionalNet", "ForeignNet", "MarginBalance", "ShortBalance",
)
```

Historical missing chip columns may be filled with `0.0`; missing OHLCV or `Date` fails closed. Compute the target row from the official snapshot. Do not mutate the artifact in this module.

- [ ] **Step 4: Reapply existing non-FinMind context**

Use the pipeline's existing functions over the reconstructed price frame:

```python
yf_price = pipeline.fetch_yfinance_price_history([ticker], start_date, end_date)
market = pipeline.fetch_yfinance_price_history("^TWII", start_date, end_date)
etf50 = pipeline.fetch_yfinance_price_history("0050.TW", start_date, end_date)
frame = pipeline.add_price_quality_features(frame, yf_price)
frame = pipeline.add_market_context_features(frame, market, etf50)
frame = pipeline.add_option_context_features(
    frame,
    *pipeline.fetch_option_context_history(start_date, end_date),
)
frame = pipeline._clean_df(frame)
```

Use the existing TWSE/TPEx suffix classification from `twstock.codes`. Do not alter US behavior or option/market-context contracts.

- [ ] **Step 5: Run incremental tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_incremental -v
```

Expected: all incremental tests pass.

- [ ] **Step 6: Commit**

```bash
git add stock_papi/quant/tw_incremental.py tests/test_tw_incremental.py
git commit -m "feat(quant): append official daily rows to local TW history"
```

---

### Task 6: Inject Prefetched Frames into the Existing Snapshot Builder

**Files:**
- Modify: `local_quant.py`
- Modify: `tests/test_local_quant_batch.py`
- Modify: `tests/test_local_quant.py`

**Interfaces:**
- Modify: `build_stock_snapshot(..., source_frame=None, source_lineage=None)`.
- Modify: `run_market_batch(..., batch_identity=...)` without changing its public return shape.
- Consume: `OfficialDailySnapshot` and `build_incremental_tw_frame()`.

- [ ] **Step 1: Write failing snapshot-injection tests**

Tests must assert:

```python
source_frame = pandas.DataFrame(...).set_index("Date")
result = build_stock_snapshot(
    pipeline,
    "TW",
    "2330",
    target_market_date=date(2026, 7, 24),
    observation_only=True,
    source_frame=source_frame,
    source_lineage={"source_mode": "tw_official_bulk_v1"},
)
```

- `pipeline.get_data` is not called when `source_frame` is supplied;
- `pipeline.calc_all` is still called;
- target-date mismatch remains fail closed;
- artifact payload includes `source_lineage` only when supplied;
- default callers without `source_frame` preserve existing behavior.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest tests.test_local_quant_batch.LocalQuantSnapshotTests -v
```

Expected: unexpected keyword argument.

- [ ] **Step 3: Implement minimal source-frame injection**

In `build_stock_snapshot()`, use:

```python
frame = source_frame.copy() if source_frame is not None else pipeline.get_data(symbol, 730)
```

Validate that `source_frame` is a non-empty DataFrame-like object. Copy `source_lineage` through JSON validation before adding it to the result. Keep model, observation-only, target-date, and publication behavior unchanged.

- [ ] **Step 4: Write prefetch-before-loop CLI tests**

Patch `build_official_daily_snapshot`, `build_incremental_tw_frame`, and `run_market_batch` to assert:

- in `--post-close --observation-only --market TW`, official snapshot is built before `run_market_batch`;
- if official snapshot raises, `run_market_batch`, `save_checkpoint`, and `write_stock_artifact` are not called;
- official snapshot is built once, not once per symbol;
- 2,000 symbols still produce exactly one snapshot build;
- `batch_identity` contains target date, `product_mode=observation`, `source_mode=tw_official_bulk_v1`, official manifest SHA-256, universe SHA-256, and source schema version;
- a legacy FinMind-mode checkpoint is not resumed because identity differs;
- TW non-post-close and all US paths preserve existing behavior.

- [ ] **Step 5: Implement CLI wiring**

After the TW universe is known but before `run_market_batch()`:

1. Hash the ordered universe.
2. Build/load the official snapshot.
3. Extend `batch_identity` with official source identity.
4. Define an `analyze_symbol` closure that calls `build_incremental_tw_frame()` and passes its frame/lineage to `build_stock_snapshot()`.
5. Pass that closure to `run_market_batch()`.

Do not modify the checkpoint before the official snapshot succeeds. Do not allow `--limit` to change source request count.

- [ ] **Step 6: Run focused local-quant tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_local_quant tests.test_local_quant_batch -v
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add local_quant.py tests/test_local_quant.py tests/test_local_quant_batch.py
git commit -m "feat(batch): prefetch official TW data before symbol execution"
```

---

### Task 7: Add Explicit Request Budget and Disabled-by-Default Fallback Contract

**Files:**
- Modify: `stock_papi/integrations/market_data/tw_official_bulk.py`
- Modify: `stock_papi/quant/tw_incremental.py`
- Modify: `tests/test_tw_official_bulk.py`
- Modify: `tests/test_tw_incremental.py`

**Interfaces:**
- Produces: `OfficialRequestBudget`, `plan_official_request_budget()`, `assert_fallback_capacity()`.
- Default: `fallback_enabled=False`, `max_finmind_fallback_requests=20`.

- [ ] **Step 1: Write failing request-budget tests**

Tests must cover:

```python
budget = plan_official_request_budget(
    cold_source_count=6,
    retry_attempts=2,
    fallback_symbols=0,
    fallback_requests_per_symbol=3,
    fallback_enabled=False,
)
assert budget.planned_minimum_requests == 6
assert budget.planned_worst_case_requests == 12
assert budget.finmind_requests == 0
assert budget.capacity_proven is True
```

Also assert:

- warm cache produces zero official network requests;
- retries are included in worst case;
- fallback disabled with missing symbols fails before fallback network calls;
- fallback enabled with 7 symbols × 3 requests exceeds the 20-request hard limit and fails before network calls;
- a missing market-wide source can never be repaired by fallback;
- budget metadata is included in snapshot lineage/checkpoint identity;
- no credential or token values enter budget serialization.

- [ ] **Step 2: Run budget tests and verify RED**

Run:

```bash
python -m unittest tests.test_tw_official_bulk.TWOfficialBudgetTests tests.test_tw_incremental.TWOfficialFallbackTests -v
```

Expected: budget interfaces missing.

- [ ] **Step 3: Implement the budget model**

Use an immutable dataclass with integer fields and explicit validation. `capacity_proven` for the official path is true when every planned source is either a valid cache hit or within the bounded official request plan. Do not model TWSE/TPEx as FinMind quota. Fallback capacity is a separate hard-limit decision and remains disabled in Production observation wiring.

- [ ] **Step 4: Run budget tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tw_official_bulk tests.test_tw_incremental -v
```

Expected: all official-source and incremental tests pass.

- [ ] **Step 5: Commit**

```bash
git add stock_papi/integrations/market_data/tw_official_bulk.py stock_papi/quant/tw_incremental.py tests/test_tw_official_bulk.py tests/test_tw_incremental.py
git commit -m "feat(data): enforce bounded official and fallback request plans"
```

---

### Task 8: Preserve Publication/Governance Contracts and Add Lineage Tests

**Files:**
- Modify: `tests/test_local_quant_publish.py`
- Modify: `tests/test_prediction_pipeline.py`
- Modify: `docs/superpowers/specs/2026-07-25-tw-official-bulk-source-design.md`

**Interfaces:**
- Consumes: stock artifact `source_lineage`.
- Produces: unchanged quant manifest and report contracts plus traceable per-symbol source metadata.

- [ ] **Step 1: Add failing publication-regression tests**

Tests must assert:

- source lineage remains inside immutable stock artifacts and is hashed by existing publication;
- `publish_market_snapshot()` still rejects TW failure rate `>= 5%`;
- mixed market dates are still excluded/rejected;
- observation artifacts contain no `AI_P`, `FUTURE_RET_5`, or `T` columns;
- no official-source code changes the exact report disclosure;
- four Production readiness flags remain false;
- `backtests/v1/latest-TW.json` is not created or referenced;
- no Supabase, LINE, GCS upload, Cloud Run, or task mutation code is introduced.

- [ ] **Step 2: Run regression tests and verify RED only where new lineage expectations are absent**

Run:

```bash
python -m unittest tests.test_local_quant_publish tests.test_prediction_pipeline -v
```

Expected: new lineage assertion fails until payload handling is complete; all pre-existing assertions remain green.

- [ ] **Step 3: Apply minimal lineage fixes**

Only adjust stock artifact payload creation/validation where required. Do not add lineage to mutable latest pointers unless an existing schema explicitly permits it. Do not change report schema, model schema, recommendation policy, or disclosure text.

- [ ] **Step 4: Mark design status implemented after all focused tests pass**

Change the spec status from `Proposed design for review` to `Implemented in Draft PR; Production rollout pending controlled local verification`. Add the implementation branch and test command inventory, but no Production-success claim.

- [ ] **Step 5: Run focused regression tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.test_tw_official_bulk \
  tests.test_tw_incremental \
  tests.test_local_quant \
  tests.test_local_quant_batch \
  tests.test_local_quant_publish \
  tests.test_prediction_pipeline \
  -v
```

Expected: zero failures and errors.

- [ ] **Step 6: Commit**

```bash
git add tests/test_local_quant_publish.py tests/test_prediction_pipeline.py docs/superpowers/specs/2026-07-25-tw-official-bulk-source-design.md local_quant.py
git commit -m "test(data): preserve TW observation publication governance"
```

---

### Task 9: Full Verification, Draft PR, Windows Python 3.10 Gate, and Merge Decision

**Files:**
- Review all files changed from `origin/main`.
- Do not add a permanent GitHub Actions workflow solely for this incident.

**Interfaces:**
- Produces: Draft PR `feat: use official bulk data for TW post-close`.

- [ ] **Step 1: Run complete local verification**

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q reporting stock_papi tests
python -m py_compile local_quant.py
node --check static/app.js
git diff --check origin/main...HEAD
git status --short
```

Expected: zero failures/errors, clean working tree after commits.

- [ ] **Step 2: Run PowerShell 5.1 parser validation**

Validate without execution:

```text
scripts/python_runtime.ps1
scripts/run_tw_post_close_pipeline.ps1
scripts/run_tw_pre_market_pipeline.ps1
scripts/invoke_pipeline_task.ps1
scripts/native_process.ps1
```

Expected: zero parser errors.

- [ ] **Step 3: Run changed-file secret and forbidden-surface scan**

Inspect changed files for:

```text
Authorization
Bearer
Cookie
password
token
service_role
sb_secret_
SUPABASE_KEY
LINE_CHANNEL_ACCESS_TOKEN
REPORT_ADMIN_USER_ID
```

Legitimate symbol names/tests are allowed only when no value is present. Confirm no `.env`, source-cache object, official live response, GCS pointer, font, credential, or Production artifact is tracked.

- [ ] **Step 4: Verify governance invariants**

Confirm all four flags remain false and `backtests/v1/latest-TW.json` is absent. Confirm the normal observation path contains no call to `fetch_finmind_dataset` after official prefetch begins.

- [ ] **Step 5: Create Draft PR**

Title:

```text
feat: use official bulk data for TW post-close
```

PR body must include:

- incident root cause;
- old FinMind request shape;
- new cold-cache and warm-cache request counts;
- exact six official source ids;
- incremental artifact contract;
- failure-before-loop contract;
- fixture-only test statement;
- focused and full test results;
- safety declarations: no live official probe, no FinMind, no GCS, no tasks, no LINE, no Cloud Run, no backfill, no model promotion.

Keep Draft until Windows Python 3.10 exact-head verification passes.

- [ ] **Step 6: Verify exact PR head on Windows Python 3.10**

Use an ephemeral workflow or Antigravity local Runner only for:

- dependency installation and `pip check`;
- focused official-source tests;
- full test suite;
- compileall/py_compile;
- Node syntax;
- PowerShell 5.1 parser;
- diff check;
- readiness flags and forbidden artifact.

No live official API, FinMind, GCS, Cloud Run, Scheduled Task, LINE, or Production credential may be used in this gate.

- [ ] **Step 7: Review exact diff and decide merge**

Merge with a merge commit only when:

- exact clean head passed Windows Python 3.10;
- no unresolved review finding remains;
- cold-cache normal request plan is fixed at six source requests plus bounded retries;
- warm-cache plan is zero network requests;
- official failure occurs before symbol loop/checkpoint mutation;
- fallback remains disabled by default;
- all governance invariants pass.

Otherwise keep Draft and report the precise blocker.

- [ ] **Step 8: Produce Antigravity Production handoff**

The handoff must require:

1. update Runner V2 to merged main;
2. keep both TW tasks disabled;
3. perform at most six live official-source probes for one completed trading date;
4. compare actual field names with committed aliases;
5. run one local observation build without `-PublishObservation`;
6. validate source cache, symbol histories, manifest, dashboard, reports, hashes, lineage, coverage, and dates;
7. only then promote/upload, verify website, repoint the two TW tasks, and enable with missed-run prevention;
8. never send LINE, backfill, promote a model, change regression flags, or create the forbidden backtest pointer.

---

## Plan Self-Review

- **Spec coverage:** All architecture, source, cache, incremental-history, checkpoint, fallback, validation, testing, rollout, and governance requirements map to Tasks 1–9.
- **Placeholder scan:** No `TBD`, `TODO`, deferred implementation instruction, or undefined interface remains.
- **Type consistency:** `OfficialDailySnapshot`, `OfficialSourceResult`, `IncrementalTWFrame`, `OfficialRequestBudget`, `build_official_daily_snapshot()`, and `build_incremental_tw_frame()` names are used consistently across producing and consuming tasks.
- **Scope check:** This plan changes only the TW post-close observation data source and its local history input. It does not include Production execution or unrelated data/model refactors.
