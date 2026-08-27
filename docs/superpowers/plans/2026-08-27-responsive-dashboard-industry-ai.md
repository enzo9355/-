# Responsive Dashboard and Industry AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved navigation, 4K readability, merged industry disclosure cards, deterministic company attention list, and taller Ask ABSORB sheet.

**Architecture:** Preserve the observation product's no-model-output contract. Add a deterministic actual-momentum company list to each published industry observation, render optional verified forecast fields only when present, and use native HTML disclosure controls plus existing CSS/Flask patterns.

**Tech Stack:** Python 3, Flask/Jinja, vanilla HTML/CSS/JavaScript, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-27-responsive-dashboard-industry-ai-design.md`

## Global Constraints

- Do not use「新手」or「初學者」in interface copy.
- Preserve the script ABSORB wordmark, its animation, and the Greek Villa editorial palette.
- Do not emit buy/sell instructions, fabricated forecasts, synthetic probabilities, or client-side rankings.
- Keep the observation artifact independent of raw AI probability and model-version fields.
- Add no dependency or frontend framework.

---

### Task 1: Deterministic industry attention companies

**Files:**
- Modify: `tests/test_observation_products.py`
- Modify: `stock_papi/batch/observation_products.py`

**Interfaces:**
- Consumes: existing `StockSnapshot.daily`, `symbol`, `name`, and industry membership.
- Produces: `industry_observations[*].ranking_basis == "actual_momentum"` and `attention_companies: list[dict]` with at most five rows.

- [ ] **Step 1: Write the failing test**

Add a test that builds one industry with six stocks and asserts a hand-derived symbol order based on five-day return, MA20 state, volume ratio, then symbol; assert each row contains only actual fields and no forbidden model keys.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_observation_products.ObservationProductsTests.test_industry_attention_companies_are_deterministic_actual_observations -v`

Expected: FAIL because `attention_companies` is absent.

- [ ] **Step 3: Write minimal implementation**

Add one private helper in `observation_products.py` that derives actual momentum rows, excludes rows without a five-day return, sorts deterministically, and returns the first five. Attach its result and `ranking_basis` inside `_industry_observations`.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_observation_products -v`

Expected: PASS.

### Task 2: Consolidated industry page and navigation

**Files:**
- Modify: `tests/test_web_product.py`
- Modify: `templates/base.html`
- Modify: `templates/dashboard.html`
- Modify: `templates/industries.html`

**Interfaces:**
- Consumes: `attention_companies`, `ranking_basis`, and existing industry metrics.
- Produces: top navigation links for `/ask` and `/learn`, no dashboard destination grid, and native industry disclosure cards linking to `/stock/<symbol>`.

- [ ] **Step 1: Write failing route/template tests**

Add tests that assert ASK ABSORB and 學習 are inside `.nav-list`, `.dashboard-destinations` is absent, the industries response contains one consolidated list, colored disclosure cards, actual-momentum labeling, and a stock link.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m unittest tests.test_web_product.WebProductTests.test_dashboard_destinations_live_in_top_navigation tests.test_web_product.WebProductTests.test_industries_merge_strength_and_attention_companies -v`

Expected: FAIL on the old duplicated dashboard grid and separate industry blocks.

- [ ] **Step 3: Implement minimal templates**

Add the two top links, delete the dashboard destination section, and replace `industries.html` with one native `<details>` list. Render verified forecast price/probability only when those fields exist; otherwise label the list「實際動能排序」.

- [ ] **Step 4: Run focused web tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_web_product -v`

Expected: PASS.

### Task 3: 4K density and Ask ABSORB workspace

**Files:**
- Modify: `tests/test_web_product.py`
- Modify: `static/app.css`

**Interfaces:**
- Consumes: existing research design tokens and responsive breakpoints.
- Produces: wider large-screen content, larger type, industry disclosure styling, and a desktop Ask sheet at least `60vh` high.

- [ ] **Step 1: Write a failing stylesheet contract test**

Assert the stylesheet exposes a wider content token, a large-screen typography rule, industry disclosure classes, and `min-height:60vh` for `.quick-ask-sheet` while retaining the mobile bottom-sheet breakpoint.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_web_product.WebProductTests.test_research_layout_supports_4k_and_tall_ask_workspace -v`

Expected: FAIL because the Ask sheet has only a maximum height and the large-screen type remains small.

- [ ] **Step 3: Implement minimal CSS**

Increase the content maximum to a fluid large-screen width, enlarge navigation/body/card metadata at `min-width: 1800px`, make the Ask sheet a flex column with `min-height:60vh`, and style the consolidated industry disclosures with the existing hot/steady/cold colors.

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_web_product -v`

Expected: PASS.

### Task 4: Regression and visual acceptance

**Files:**
- Modify: `tests/visual_qa_server.py` only if its snapshot lacks company attention rows.

**Interfaces:**
- Produces: browser-verifiable desktop, 4K, and mobile pages using production-shaped fixtures.

- [ ] **Step 1: Run complete focused regression**

Run: `.venv\Scripts\python.exe -m unittest tests.test_observation_products tests.test_web_product tests.test_five_session_forecast -v`

Expected: PASS with zero failures.

- [ ] **Step 2: Run static checks**

Run: `.venv\Scripts\python.exe -m compileall -q stock_papi tests`

Run: `git diff --check`

Expected: both exit 0.

- [ ] **Step 3: Start the visual QA server**

Run: `.venv\Scripts\python.exe tests\visual_qa_server.py`

Expected: local server starts without traceback.

- [ ] **Step 4: Browser acceptance**

Verify `/`, `/industries`, and Ask ABSORB at 1440×1000, 2560×1440, 3840×2160, and 390×844. Confirm top navigation, larger typography, reduced outer whitespace, native industry expansion, stock links, Ask height, focus/close behavior, and no console/network errors.

- [ ] **Step 5: Commit implementation**

Run: `git add stock_papi/batch/observation_products.py templates/base.html templates/dashboard.html templates/industries.html static/app.css tests/test_observation_products.py tests/test_web_product.py tests/visual_qa_server.py`

Run: `git commit -m "feat: consolidate industry dashboard experience"`
