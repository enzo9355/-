# ABSORB Dual-Market Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver market-aware TW/US research surfaces, restore authoritative TW freshness, and eliminate local schedule/runtime conflicts.

**Architecture:** Keep one Flask product shell and one stock-analysis engine, but bind every navigation surface to an explicit market context. Build US summary views from hash-verified canonical professional reports, extend TW lifecycle evidence generically, and make Windows scheduling contention-safe without weakening publication gates.

**Tech Stack:** Python 3, Flask/Jinja, vanilla CSS/JS, PowerShell Scheduled Tasks, unittest, Cloud Run, GCS.

**Spec:** `docs/specs/2026-08-24-dual-market-production.md`

## Global Constraints

- Never synthesize, forward-fill, or relabel prices or statuses.
- Preserve fail-closed official-data and publication gates.
- UI labels are exactly `ABSORB` and `ASK ABSORB`.
- TW and US expose the same five navigation capabilities with verified market-specific data.
- Preserve hidden task execution, identity, working directory, logs, exit codes, retries, and idempotency.
- Use Avenir for English and the existing matching Traditional Chinese font stack.
- All new behavior follows RED, GREEN, REFACTOR and includes mobile/accessibility coverage.

---

### Task 1: Market-aware product shell and US research summary

**Files:**
- Create: `stock_papi/services/market_summary.py`
- Create: `templates/us_dashboard.html`
- Create: `templates/us_market.html`
- Create: `templates/us_industries.html`
- Modify: `stock_papi/web/routes/reports.py`
- Modify: `stock_papi/web/routes/market.py`
- Modify: `templates/base.html`
- Modify: `templates/stocks.html`
- Modify: `static/app.css`
- Modify: `static/app.js`
- Modify: `DESIGN.md`
- Test: `tests/test_web_product.py`
- Test: `tests/test_report_web.py`
- Test: `tests/test_route_inventory.py`
- Test: `tests/test_absorb_brand.py`

**Interfaces:**
- Produces: `build_market_summary_view(report: ProfessionalPostCloseReport) -> dict`
- Produces routes: `us_dashboard_page`, `us_market_page`, `us_industries_page`, `us_stocks_page`
- Consumes existing verified canonical object loader and stock analyzer.

- [ ] Write route and rendered-behavior tests proving `/us` is a summary, all five US links retain US context, `/stock/AAPL` renders US context, and uppercase labels are accessible.
- [ ] Run focused tests and confirm they fail because the routes/context do not exist and labels remain mixed case.
- [ ] Extract the existing canonical-object validation into a private reusable loader in `reports.py`; never bypass SHA, path, schema, or metadata binding checks.
- [ ] Implement the US summary view from literal professional-report fields: market, industries, key events, securities, validation, source date, and applicable date.
- [ ] Implement context-aware desktop/mobile navigation and market-specific stock search without duplicating analysis logic.
- [ ] Apply the incumbent design system, resilient empty/error states, reduced-motion behavior, and exact uppercase brand copy.
- [ ] Run focused web/brand/route tests, then refactor duplication while keeping them green.
- [ ] Commit the task.

### Task 2: Authoritative TW suspension lifecycle coverage

**Files:**
- Modify: `stock_papi/integrations/market_data/tw_trading_status.py`
- Modify: `stock_papi/integrations/market_data/tw_official_historical.py`
- Test: `tests/test_tw_trading_status.py`
- Test: `tests/test_tw_official_historical.py`
- Test: `tests/test_tpex_lifecycle_cache.py`

**Interfaces:**
- Produces: generic TWSE listing-change suspension events with the existing `_lifecycle_event(...)` evidence contract.
- Preserves: `LifecycleSnapshot.status_by_symbol`, `terminated_by_symbol`, source hashes, and request budget.
- Official fixture: `https://investoredu.twse.com.tw/FileSystem/FileUpload/88ff18ef-5726-4b33-b207-f92310023328.pdf`, 139878 bytes, SHA-256 `3ff4455c1435b5d0dc62803953241d184c13775662eb46f2feaf25d3d300c768`.

- [ ] Add a fixture matching the official TWSE listing-change payload for 2867 with stop date `2026-08-20` and delisting date `2026-09-01`.
- [ ] Add tests proving 8/19 active, 8/20-8/31 suspended, 9/1 terminated, hash/date/schema tampering rejected, and same-session regular price conflict remains fail-closed.
- [ ] Run focused lifecycle tests and confirm the 8/20 suspension case fails as unrecognized.
- [ ] Extend the existing TWSE lifecycle definition/parser generically, preserving official source identity and hash binding; do not special-case symbol 2867.
- [ ] Thread the event source through historical lifecycle loading and manifest source hashes.
- [ ] Run lifecycle, historical, incremental, post-close, and observation-product tests; refactor only after green.
- [ ] Commit the task.

### Task 3: Contention-safe scheduler and runtime completion guard

**Files:**
- Modify: `scripts/invoke_pipeline_task.ps1`
- Modify: `scripts/install_pipeline_tasks.ps1`
- Modify: `stock_papi/batch/full_backtest_cli.py`
- Modify: `scripts/python_runtime.ps1`
- Modify: `scripts/run_us_daily.ps1`
- Modify: `scripts/run_full_backtest.ps1`
- Test: `tests/test_pipeline_scheduler.py`
- Test: `tests/test_local_quant_task.py`

**Interfaces:**
- Produces: market-specific observation mutex selection and bounded wait helper.
- Produces: a pre-import completed-checkpoint exit contract for full backtest.
- Changes: `ABSORB-FullBacktest` trigger to daily `22:30` with a 3h45m execution limit.

- [ ] Add executable scheduler tests using controlled mutex contention and a completed checkpoint with yfinance deliberately unavailable.
- [ ] Add installer tests proving daily 22:30, bounded execution, hidden launcher, and no one-minute repetition.
- [ ] Run focused tests and confirm current immediate mutex failure and pre-guard yfinance import fail.
- [ ] Implement TW/US mutex separation plus bounded wait and status receipt fields without exposing secrets.
- [ ] Move completion detection before `load_stock_pipeline` and make the wrapper disable the completed task idempotently.
- [ ] Replace the two wrappers' hard-coded bundled-Python preference with `Resolve-AbsorbPythonExecutable` and `Assert-AbsorbPythonRuntime`; validate required imports including yfinance and fail with a safe actionable error when incomplete.
- [ ] Run focused scheduler/runtime tests and PowerShell AST parsing, then refactor while green.
- [ ] Commit the task.

### Task 4: Data freshness surface and full release verification

**Files:**
- Modify: `stock_papi/web/routes/system.py`
- Modify: `stock_papi/web/route_registration.py`
- Modify: `templates/dashboard.html`
- Modify: `templates/us_dashboard.html`
- Modify: `static/app.css`
- Modify: `scripts/deploy_observation_production.ps1` if provenance is not current-commit bound
- Test: `tests/test_web_product.py`
- Test: `tests/test_observation_deploy_scripts.py`

**Interfaces:**
- Produces: `/health/data` JSON that distinguishes service health from TW/US data freshness.
- Consumes: latest verified TW dashboard and TW/US report index dates.

- [ ] Add tests for current, updating, stale, and unavailable freshness states without treating a stale pointer as healthy data.
- [ ] Run tests and confirm the data-health route and visible stale state are absent.
- [ ] Implement a pure freshness classifier and server-rendered status presentation with exact source/applicable dates.
- [ ] Bind deployment provenance to the current commit and verify rollback revision remains available.
- [ ] Run focused tests, the complete suite, compile/static/PowerShell/JS checks, and the Impeccable detector once.
- [ ] Apply the authoritative installer to active tasks and read back XML, actions, triggers, states, and next-run times.
- [ ] Catch up only completed TW sessions through strict gates, publish with independent GCS readback, deploy Cloud Run, and validate desktop/mobile live routes, searches, console, network, dates, health, traffic, and rollback.
- [ ] Commit any final source-only corrections before deployment and record final evidence.
