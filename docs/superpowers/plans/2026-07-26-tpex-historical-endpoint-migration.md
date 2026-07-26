# TPEx Historical Endpoint Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the TPEx historical price and margin request contracts to the verified date-addressable endpoints while preserving institutional parsing, fail-closed date validation, canonical parser semantics, cache reuse, and every Production safety boundary.

**Architecture:** Keep the existing six-source historical adapter. Change only two source URLs, split `_params()` by the three TPEx source contracts, and route request headers through one private source-specific helper. Reuse the existing parsers and cache format; prove compatibility with offline synthetic fixtures and focused regressions.

**Tech Stack:** Python 3.14.6, standard-library `unittest`, existing requests-compatible session boundary, Git worktree, PowerShell 5.1 parser, Node.js syntax checker, GitHub CLI.

## Global Constraints

- Work only in `C:\Users\enzo\Documents\absorb-phase1e-tpex` on `codex/tw-tpex-historical-endpoint-migration`, created from `0a3d78916b9a751b6c7d9b6a7aa8e00491d95edf`.
- Do not modify, stage, move, delete, or copy any user-owned untracked file from `C:\Users\enzo\Documents\line bot`.
- Production code changes are limited to `stock_papi/integrations/market_data/tw_official_historical.py`; tests are limited to `tests/test_tw_official_historical.py`.
- Change only `tpex_price` and `tpex_margin` URLs, parameters, and the required price request header. `tpex_institutional` URL, parameters, and parser remain unchanged.
- Keep `PARSER_VERSION = "tw-official-historical-parser-v2"`, `MAX_CATCHUP_SESSIONS = 10`, source ordering, retry attempts, timeouts, size limits, coverage thresholds, parser dispatch, canonical rows, manifest schema, and cache directory schema unchanged.
- Do not add dependencies, browser automation, Cookie, Referer, Authorization, Token, fallback providers, or live-network unit tests.
- Do not run `stock_papi.batch.tw_official_post_close_cli`, Local Observation, recovery backfill, source-cache promotion, GCS, Cloud Run, LINE, FinMind, Supabase, publish, promotion, or `latest-TW.json` writes.
- Keep `ABSORB-TW-PostClose` and `ABSORB-TW-PreMarket` Disabled; never start, enable, trigger, or modify them.
- Use `.\.venv\Scripts\python.exe` for local Python 3.14.6 verification. Python 3.10 parity may be reported only if actually executed in an isolated compatible environment.
- Use strict RED/GREEN evidence for every production behavior. Preservation-only gates explicitly identify that no RED implementation change is expected.

## File structure

- `stock_papi/integrations/market_data/tw_official_historical.py`: owns the six source definitions, source-specific parameters, request headers, transport boundary, parser dispatch, coverage, and snapshot-series assembly.
- `tests/test_tw_official_historical.py`: owns sanitized six-source payloads, the recording session, request-contract assertions, parser compatibility and fail-closed regressions, institutional preservation, cache reuse, and bounded catch-up tests.
- `docs/superpowers/specs/2026-07-26-tpex-historical-endpoint-migration-design.md`: approved architecture and safety boundary.
- `docs/superpowers/plans/2026-07-26-tpex-historical-endpoint-migration.md`: this execution and evidence plan.

## Reviewed plan commit boundary

Before Task 1, commit this reviewed plan by itself:

```powershell
git add -- docs/superpowers/plans/2026-07-26-tpex-historical-endpoint-migration.md
git diff --cached --check
git commit -m "docs: plan TPEx historical endpoint migration"
```

Expected: one plan file committed, `git status --short` empty, and no production or test file staged.

---

### Task 1: Contract tests

**Files:**

- Modify: `tests/test_tw_official_historical.py:11-20,22,158-185`
- Reference: `stock_papi/integrations/market_data/tw_official_historical.py:73-124`

**Interfaces:**

- Consumes: `HISTORICAL_SOURCE_DEFINITIONS`, private `_params(source_id: str, target_date: datetime.date) -> dict[str, str]`.
- Produces: `HistoricalRequestContractTests`, which Tasks 2 and 3 make GREEN without changing institutional behavior.

- [ ] **Step 1: Write exact failing request-contract tests**

Import `_params`, define `CONTRACT_TARGET = datetime.date(2026, 7, 16)`, and add:

```python
class HistoricalRequestContractTests(unittest.TestCase):
    def test_tpex_price_contract_is_modern(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_price"].url,
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
        )
        self.assertEqual(
            _params("tpex_price", CONTRACT_TARGET),
            {"date": "2026/07/16", "response": "json"},
        )

    def test_tpex_margin_contract_is_modern(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_margin"].url,
            "https://www.tpex.org.tw/www/zh-tw/margin/balance",
        )
        self.assertEqual(
            _params("tpex_margin", CONTRACT_TARGET),
            {"date": "2026/07/16", "response": "json"},
        )

    def test_tpex_institutional_contract_is_unchanged(self):
        self.assertEqual(
            HISTORICAL_SOURCE_DEFINITIONS["tpex_institutional"].url,
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php",
        )
        self.assertEqual(_params("tpex_institutional", CONTRACT_TARGET), {
            "l": "zh-tw", "o": "json", "se": "EW", "t": "D",
            "d": "115/07/16", "s": "0,asc",
        })
```

- [ ] **Step 2: Run RED and preserve evidence**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests -v
```

Expected RED reason: `test_tpex_price_contract_is_modern` and `test_tpex_margin_contract_is_modern` still see legacy PHP URLs and ROC `d` parameter dictionaries. `test_tpex_institutional_contract_is_unchanged` must pass. Record command, exit code 1, failing test names, assertion differences, and why those differences prove the modern contracts are absent.

- [ ] **Step 3: Commit the RED contract specification**

```powershell
git add -- tests/test_tw_official_historical.py
git diff --cached --check
git commit -m "test: specify modern TPEx historical request contracts"
```

Commit boundary: tests only; the branch is intentionally RED for the new contract tests until Tasks 2 and 3.

- [ ] **Step 4: Define the later GREEN command**

Run after Tasks 2 and 3:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests -v
```

Expected GREEN result: 3 tests pass; the unchanged institutional URL and parameters pass in the same run.

---

### Task 2: Price endpoint migration

**Files:**

- Modify: `stock_papi/integrations/market_data/tw_official_historical.py:87-91,118-119`
- Test: `tests/test_tw_official_historical.py:HistoricalRequestContractTests`

**Interfaces:**

- Consumes: the exact price URL and parameter assertions from Task 1.
- Produces: `tpex_price` definition using the modern endpoint and `_params("tpex_price", date)` using Gregorian slash-separated dates.

- [ ] **Step 1: Re-run the price RED assertions**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests.test_tpex_price_contract_is_modern -v
```

Expected RED reason: the price URL and price parameters are still legacy.

- [ ] **Step 2: Implement the minimum price contract**

Change only:

```python
"tpex_price": OfficialSourceDefinition(
    "tpex_price", "TPEx", "price",
    "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes", "tpex_tables",
    30 * 1024 * 1024,
),
```

and:

```python
if source_id == "tpex_price":
    return {"date": target_date.strftime("%Y/%m/%d"), "response": "json"}
```

Do not change `OfficialSourceDefinition`, parser dispatch, retries, timeout, size, or coverage.

- [ ] **Step 3: Run the price-focused GREEN proof**

Run the exact price contract test and existing parsers:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests.test_tpex_price_contract_is_modern -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalParserTests -v
git diff --check
```

Expected GREEN result: the price contract and all existing parser tests pass; `git diff --check` exits 0. Do not describe the whole contract class as GREEN until Task 3.

- [ ] **Step 4: Commit the price migration**

```powershell
git add -- stock_papi/integrations/market_data/tw_official_historical.py
git diff --cached --check
git commit -m "fix: migrate TPEx price historical endpoint"
```

Commit boundary: price URL and price parameters only.

---

### Task 3: Margin endpoint migration

**Files:**

- Modify: `stock_papi/integrations/market_data/tw_official_historical.py:97-101,122-123`
- Test: `tests/test_tw_official_historical.py:HistoricalRequestContractTests`

**Interfaces:**

- Consumes: Task 1 exact margin assertions and Task 2 unchanged source mapping.
- Produces: `tpex_margin` definition using the modern endpoint and `_params("tpex_margin", date)` using Gregorian slash-separated dates; completes the request-contract GREEN state.

- [ ] **Step 1: Confirm the remaining RED behavior**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests.test_tpex_margin_contract_is_modern -v
```

Expected RED reason: the margin URL and margin parameters remain legacy after Task 2.

- [ ] **Step 2: Implement the minimum margin contract**

Change only:

```python
"tpex_margin": OfficialSourceDefinition(
    "tpex_margin", "TPEx", "margin",
    "https://www.tpex.org.tw/www/zh-tw/margin/balance", "tpex_tables",
    15 * 1024 * 1024,
),
```

and:

```python
if source_id == "tpex_margin":
    return {"date": target_date.strftime("%Y/%m/%d"), "response": "json"}
```

- [ ] **Step 3: Run full contract GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalParserTests -v
git diff --check
```

Expected GREEN result: contract 3/3 and parser tests all pass, with institutional unchanged.

- [ ] **Step 4: Commit the margin migration**

```powershell
git add -- stock_papi/integrations/market_data/tw_official_historical.py
git diff --cached --check
git commit -m "fix: migrate TPEx margin historical endpoint"
```

Commit boundary: margin URL and margin parameters only.

---

### Task 4: Request header behavior

**Files:**

- Modify: `tests/test_tw_official_historical.py:133-155,HistoricalRequestContractTests`
- Modify: `stock_papi/integrations/market_data/tw_official_historical.py:303-313`

**Interfaces:**

- Consumes: `_request_payload()` and a requests-compatible `session.get(url, params=..., headers=..., timeout=...)` boundary.
- Produces: private `_request_headers(source_id: str) -> dict[str, str]`; source-aware `Session` calls shaped as `{"source_id", "url", "date", "params", "headers"}`.

- [ ] **Step 1: Make the test session understand all three date contracts**

Update `Session.get()` before adding the new header assertion. Accept `headers` explicitly, derive `value` from the actual parameters, and retain the exact transport call:

```python
def get(self, url, *, params, headers, **_kwargs):
    source_id = next(
        key for key, definition in HISTORICAL_SOURCE_DEFINITIONS.items()
        if definition.url == url
    )
    if source_id.startswith("twse"):
        value = datetime.datetime.strptime(params["date"], "%Y%m%d").date()
    elif source_id in {"tpex_price", "tpex_margin"}:
        value = datetime.datetime.strptime(params["date"], "%Y/%m/%d").date()
    else:
        year, month, day = map(int, params["d"].split("/"))
        value = datetime.date(year + 1911, month, day)
    self.calls.append({
        "source_id": source_id,
        "url": url,
        "date": value,
        "params": dict(params),
        "headers": dict(headers),
    })
    return Response(payloads(value)[source_id])
```

- [ ] **Step 2: Prove the helper change preserves the series test**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalSeriesTests.test_two_dates_use_twelve_cold_requests_then_zero_warm_requests -v
```

Expected GREEN result: 12 cold calls, zero warm calls, and identical manifest SHA-256. This helper-only change does not implement the missing price header.

- [ ] **Step 3: Write the failing exact transport-contract test**

Call `build_historical_daily_snapshot()` in a temporary directory with `session=Session()`, target `CONTRACT_TARGET`, `minimum_price_symbols={"TWSE": 2, "TPEx": 2}`, and `minimum_chip_symbols=1`. Assert every request field from the actual parsed call:

```python
session = Session()
with tempfile.TemporaryDirectory() as temporary:
    build_historical_daily_snapshot(
        Path(temporary),
        CONTRACT_TARGET,
        session=session,
        minimum_price_symbols={"TWSE": 2, "TPEx": 2},
        minimum_chip_symbols=1,
    )

self.assertEqual(
    {call["source_id"] for call in session.calls},
    set(HISTORICAL_SOURCE_DEFINITIONS),
)
for call in session.calls:
    source_id = call["source_id"]
    self.assertEqual(call["url"], HISTORICAL_SOURCE_DEFINITIONS[source_id].url)
    self.assertEqual(call["date"], CONTRACT_TARGET)
    self.assertEqual(call["params"], _params(source_id, CONTRACT_TARGET))
    self.assertEqual(call["headers"]["User-Agent"], "ABSORB/1.0")
    if source_id == "tpex_price":
        self.assertEqual(
            call["headers"]["X-Requested-With"],
            "XMLHttpRequest",
        )
    else:
        self.assertNotIn("X-Requested-With", call["headers"])

institutional = next(
    call for call in session.calls if call["source_id"] == "tpex_institutional"
)
for forbidden in ("Cookie", "Authorization", "Token"):
    self.assertNotIn(forbidden, institutional["headers"])
```

- [ ] **Step 4: Run RED and record the missing header**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests.test_request_headers_are_source_specific -v
```

Expected RED reason: `_request_payload()` sends only `User-Agent`, so the price request lacks `X-Requested-With`. Exit code must be 1 for that assertion, not a fixture error.

- [ ] **Step 5: Implement the minimum private helper**

```python
def _request_headers(source_id: str) -> dict[str, str]:
    headers = {"User-Agent": "ABSORB/1.0"}
    if source_id == "tpex_price":
        headers["X-Requested-With"] = "XMLHttpRequest"
    return headers
```

Change `_request_payload()` only at the call site:

```python
headers=_request_headers(definition.source_id),
```

Do not add headers to the dataclass or add browser headers.

- [ ] **Step 6: Run header and request-contract GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalRequestContractTests -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalParserTests -v
git diff --check
```

Expected GREEN result: all request-contract and parser tests pass; institutional contains no forbidden header.

- [ ] **Step 7: Commit header behavior**

```powershell
git add -- stock_papi/integrations/market_data/tw_official_historical.py tests/test_tw_official_historical.py
git diff --cached --check
git commit -m "fix: send required TPEx price request header"
```

Commit boundary: test-session date/transport recording, `_request_headers`, its call site, and exact transport assertions only.

---

### Task 5: Parser compatibility regression

**Files:**

- Modify: `tests/test_tw_official_historical.py:44-155,HistoricalParserTests,HistoricalSeriesTests`
- Verify unchanged: `stock_papi/integrations/market_data/tw_official_historical.py:127-300`

**Interfaces:**

- Consumes: modern TPEx payload shape, existing `parse_tpex_price_report`, `parse_tpex_margin_report`, `parse_tpex_institutional`, and migrated source-specific parameters.
- Produces: source-aware test `Session`, exact modern canonical assertions, independent top-level/table-date mismatch tests, and institutional canonical regression coverage.

- [ ] **Step 1: Run the parser compatibility preservation gate**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalParserTests tests.test_tw_official_historical.HistoricalSeriesTests -v
```

Expected RED reason: not applicable by design. Task 4 already made the test session source-aware; this task adds characterization coverage without changing production parser semantics. A failure here blocks parser work and requires evidence from the sanitized fixture before any parser modification.

- [ ] **Step 2: Strengthen sanitized modern parser fixtures**

Keep top-level date `ymd(value)`, `stat="ok"`, table date `roc(value)` (which is `115/07/16` for `CONTRACT_TARGET`), title `上櫃股票行情` with two valid price rows, and title `上櫃股票融資融券餘額` with one valid margin row. Assert exact price OHLC and volume values and exact margin balances.

Add separate tests that mutate only one date proof at a time:

```python
for source_id, parser in (
    ("tpex_price", parse_tpex_price_report),
    ("tpex_margin", parse_tpex_margin_report),
):
    wrong_top = payloads(CONTRACT_TARGET)[source_id]
    wrong_top["date"] = "20260724"
    with self.assertRaises(ValueError):
        parser(wrong_top, CONTRACT_TARGET)

    wrong_table = payloads(CONTRACT_TARGET)[source_id]
    wrong_table["tables"][0]["date"] = "115/07/24"
    with self.assertRaises(ValueError):
        parser(wrong_table, CONTRACT_TARGET)
```

Import `parse_tpex_institutional` from `tw_official_bulk` and assert the fixture still canonicalizes the exact `Foreign`, `InvestmentTrust`, and `Dealer` rows without changing expected output to accommodate price or margin work.

```python
self.assertEqual(parse_tpex_institutional(data["tpex_institutional"], CONTRACT_TARGET), (
    {"date": "2026-07-16", "stock_id": "6488", "name": "Dealer", "buy": 20.0, "sell": 21.0},
    {"date": "2026-07-16", "stock_id": "6488", "name": "Foreign", "buy": 8.0, "sell": 9.0},
    {"date": "2026-07-16", "stock_id": "6488", "name": "InvestmentTrust", "buy": 11.0, "sell": 12.0},
))
```

- [ ] **Step 3: Run parser, helper, and series GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalParserTests -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalSeriesTests -v
git diff --check
```

Expected GREEN result: modern price/margin parsing, four independent date-mismatch cases, institutional mapping, 12 cold requests, zero warm requests, and bounded catch-up all pass. Production parser code and `PARSER_VERSION` remain unchanged.

- [ ] **Step 4: Commit parser and institutional regressions**

```powershell
git add -- tests/test_tw_official_historical.py
git diff --cached --check
git commit -m "test: preserve TPEx parser compatibility regressions"
```

Commit boundary: sanitized fixtures and regression tests only; no production parser changes.

---

### Task 6: Cache reuse regression

**Files:**

- Verify: `tests/test_tw_official_historical.py:HistoricalSeriesTests.test_two_dates_use_twelve_cold_requests_then_zero_warm_requests`
- Verify: `tests/test_tw_official_cache.py`
- Verify unchanged: `stock_papi/integrations/market_data/tw_official_historical.py:35,388-424`

**Interfaces:**

- Consumes: unchanged parser version, canonical cache loader/store, two-date snapshot-series assembly.
- Produces: evidence that endpoint migration does not invalidate verified canonical warm cache entries.

- [ ] **Step 1: Run the preservation gate before further changes**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalSeriesTests.test_two_dates_use_twelve_cold_requests_then_zero_warm_requests -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_bulk.TWOfficialCacheTests -v
```

Expected RED reason: not applicable by design. This task introduces no production behavior; both commands must already be GREEN after Task 5. A failure means the migration changed canonical/cache semantics and blocks implementation rather than authorizing a parser-version bump.

- [ ] **Step 2: Inspect immutable compatibility facts**

```powershell
Select-String -Path stock_papi/integrations/market_data/tw_official_historical.py -Pattern 'PARSER_VERSION|MAX_CATCHUP_SESSIONS'
git diff 0a3d78916b9a751b6c7d9b6a7aa8e00491d95edf -- stock_papi/integrations/market_data/tw_official_cache.py
```

Expected: parser version is exactly `tw-official-historical-parser-v2`, catch-up remains 10, and cache module diff is empty.

- [ ] **Step 3: Re-run GREEN and record exact evidence**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical.HistoricalSeriesTests.test_two_dates_use_twelve_cold_requests_then_zero_warm_requests tests.test_tw_official_historical.HistoricalSeriesTests.test_series_rejects_more_than_bounded_catchup -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_bulk.TWOfficialCacheTests -v
git diff --check
```

Expected GREEN result: two-date cold count 12, warm count 0, manifest SHA-256 identical, catch-up over 10 rejected, cache suite passes.

- [ ] **Step 4: Commit boundary**

No commit is expected because this is a preservation-only gate. If a fixture-only correction is required, add it to the Task 5 test commit after repeating Task 5 review; do not change parser version or cache code.

---

### Task 7: Full verification

**Files:**

- Verify only: entire repository
- Changed-file allowlist: the two docs, `stock_papi/integrations/market_data/tw_official_historical.py`, and `tests/test_tw_official_historical.py`

**Interfaces:**

- Consumes: completed commits from Tasks 1-5 and clean cache gate from Task 6.
- Produces: fresh focused/full/static/safety evidence for independent review and PR creation.

- [ ] **Step 1: Run focused related suites**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_historical -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_bulk.TWOfficialCacheTests -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_post_close_cli -v
.\.venv\Scripts\python.exe -m unittest tests.test_tw_official_bulk -v
```

Expected RED reason: not applicable; this is the post-implementation gate. Any failure blocks completion and must be debugged from its root cause before review.

- [ ] **Step 2: Run the complete Python suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected GREEN result: at least 898 tests, 0 failures, 0 errors; report skipped count and exit code 0.

- [ ] **Step 3: Run static and PowerShell 5.1 gates**

```powershell
.\.venv\Scripts\python.exe -m compileall -q reporting stock_papi tests
.\.venv\Scripts\python.exe -m py_compile local_quant.py
node --check static/app.js
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command '$null = [System.Management.Automation.Language.Parser]::ParseFile("scripts\upload_local_quant.ps1", [ref]$null, [ref]$null)'
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command '[scriptblock]::Create((Get-Content "scripts\upload_local_quant.ps1" -Raw)) | Out-Null'
git diff --check
```

Expected GREEN result: every command exits 0 with no syntax or whitespace failure.

- [ ] **Step 4: Verify scope and safety facts**

```powershell
git diff --name-only 0a3d78916b9a751b6c7d9b6a7aa8e00491d95edf...HEAD
git status --short
Get-ScheduledTask -TaskName ABSORB-TW-PostClose | Select-Object TaskName,State
Get-ScheduledTask -TaskName ABSORB-TW-PreMarket | Select-Object TaskName,State
```

Expected: only the four allowlisted files; worktree clean; both tasks Disabled. Confirm no `D:\AbsorbData`, cache, quarantine, publication, GCS, Cloud Run, LINE, FinMind, Supabase, or `latest-TW.json` mutation occurred.

- [ ] **Step 5: Commit boundary**

No verification-only commit. If verification reveals a real defect, use systematic debugging, write a focused failing test, make the smallest fix, commit it separately, and repeat all Task 7 commands before review.

---

### Task 8: Draft PR

**Files:**

- Create externally: one GitHub Draft PR targeting `main`
- No repository file changes expected

**Interfaces:**

- Consumes: verified clean branch, final independent code-review approval, and exact Phase 1E evidence.
- Produces: Draft PR titled `fix: migrate TPEx historical price and margin endpoints`, required CI status, and an Antigravity validation handoff. Never merges.

- [ ] **Step 1: Confirm the pre-PR RED state**

```powershell
gh pr view codex/tw-tpex-historical-endpoint-migration --json number,url,isDraft,baseRefName,headRefName
```

Expected RED reason: before creation, GitHub reports that no pull request exists for the branch. An existing PR must be inspected and reused only if it is Draft, targets `main`, and has the exact branch head.

- [ ] **Step 2: Push the verified branch**

```powershell
git push -u origin codex/tw-tpex-historical-endpoint-migration
```

Do not force push.

- [ ] **Step 3: Create the Draft PR**

Use title `fix: migrate TPEx historical price and margin endpoints`. The body must contain `Problem`, `Verified Contracts`, `Scope`, `Non-Goals`, `TDD Evidence`, `Verification`, `Safety`, and `Rollback`; include exact RED/GREEN commands and results, institutional preservation, unchanged parser version/cache policy, both tasks Disabled, and no Production mutation.

```powershell
gh pr create --draft --base main --head codex/tw-tpex-historical-endpoint-migration --title "fix: migrate TPEx historical price and margin endpoints" --body-file "$env:TEMP\absorb-phase1e-tpex-pr-body.md"
```

The reviewed PR body file lives outside the repository in `%TEMP%` and is not committed. Run `git status --short` again after PR creation to prove the worktree remains clean.

- [ ] **Step 4: Run PR GREEN checks**

```powershell
gh pr view codex/tw-tpex-historical-endpoint-migration --json number,url,isDraft,baseRefName,headRefName,headRefOid,files
gh pr checks codex/tw-tpex-historical-endpoint-migration --watch
```

Expected GREEN result: `isDraft=true`, `baseRefName=main`, correct head branch/OID, allowlisted file inventory, and every required check passes. Record PR number, URL, workflows, jobs, conclusions, and any failing step evidence.

- [ ] **Step 5: CI failure loop and commit boundary**

If CI fails, inspect the failing job and step, prove whether it is branch-related, and fix only an in-scope branch defect using TDD. Re-run Task 7 locally, commit the focused fix, push normally, and wait for required checks again. Baseline or unrelated CI failures must be evidenced and leave the final gate FAIL.

No PR-only commit is expected. Do not merge, enable auto-merge, rebase unknown `main`, delete the branch, deploy, backfill, publish, promote, or enable Scheduled Tasks.
