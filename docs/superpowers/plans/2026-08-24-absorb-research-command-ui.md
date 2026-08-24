# ABSORB Research Command UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Research Command shell, verified-data dashboard, responsive mobile layout, and floating Ask ABSORB dialog without changing production data or deployment state.

**Architecture:** Keep Flask routes and snapshot readers unchanged. Recompose the shared Jinja shell and dashboard around existing observation fields, add progressive CSS visualization using native elements and SVG attributes, and extend the existing browser bundle so every conversation form uses the same JSON-only endpoint.

**Tech Stack:** Flask, Jinja2, semantic HTML, CSS, vanilla JavaScript, Python unittest, Playwright CLI.

**Spec:** `docs/superpowers/specs/2026-08-24-absorb-research-command-ui-design.md`

## Global Constraints

- Use only verified observation fields; never synthesize market values or relabel TW data as US data.
- Preserve all public routes and the `/api/conversation` `{question: string}` JSON contract.
- Keep canonical brand image assets unchanged for favicon, social, LINE, and metadata uses.
- Navigation wordmark is lowercase cursive text `absorb` without a circular icon and links to `/`.
- Font stack begins with locally installed `Avenir Next`, `Avenir`, then `Noto Sans TC`; add no font files or external font requests.
- Motion is 120 to 180 milliseconds, honors `prefers-reduced-motion`, and avoids layout-property animation.
- Validate 1440px, 736px, and 390px with no horizontal overflow.
- Do not deploy or mutate Cloud Run, GCS, schedulers, reports, or market data.

---

### Task 1: Brand and shell contract

**Files:**
- Modify: `tests/test_web_product.py`
- Modify: `tests/test_absorb_brand.py`
- Modify: `DESIGN.md`
- Modify: `templates/base.html`
- Modify: `static/app.css`

**Interfaces:**
- Consumes: existing Flask endpoint names and `STATIC_ASSET_VERSION`.
- Produces: `[data-brand-wordmark]`, `[data-market-switch]`, `[data-quick-ask-open]`, and the existing `.nav-link` active-state contract.

- [ ] **Step 1: Write failing tests** asserting a text wordmark linked to `/`, no navigation `brand-mark` image, the Avenir/Noto stack, honest `/reports/us` market switch, and preserved route links.
- [ ] **Step 2: Run** `python -m unittest tests.test_absorb_brand tests.test_web_product.WebProductTests.test_base_shell_uses_absorb_brand_and_light_theme tests.test_web_product.WebProductTests.test_navigation_has_route_active_state_and_no_primary_hash_links -v` and confirm the new assertions fail because the current sidebar shell remains.
- [ ] **Step 3: Implement** the top navigation shell and update `DESIGN.md` so canonical image assets remain unchanged while navigation uses the approved text wordmark.
- [ ] **Step 4: Run the same tests** and confirm they pass.

### Task 2: Verified-data Research Command dashboard

**Files:**
- Modify: `tests/test_web_product.py`
- Modify: `templates/dashboard.html`
- Modify: `static/app.css`

**Interfaces:**
- Consumes: `observation.market_observation`, `observation.industry_observations`, `observation.daily_focus`, `observation.data_quality`, `daily_cards`, `observation.observation_as_of`, and `observation.generated_at`.
- Produces: semantic sections `[data-research-summary]`, `[data-market-command]`, `[data-breadth-visual]`, and `[data-sector-flow]`.

- [ ] **Step 1: Write failing tests** asserting the Research Command landmarks, verified source/date labels, risk state, data quality, native progress/SVG visualization, existing report links, and absence of fabricated index values or inline styles.
- [ ] **Step 2: Run** the two dashboard tests in `tests.test_web_product.WebProductTests` and confirm failure because the old card grid is still rendered.
- [ ] **Step 3: Implement** the dashboard using only verified snapshot fields and fail-closed Jinja branches for missing values.
- [ ] **Step 4: Run** the dashboard tests plus `tests.test_observation_public_surfaces` and confirm all pass.

### Task 3: Floating Ask ABSORB dialog

**Files:**
- Modify: `tests/test_web_product.py`
- Modify: `templates/base.html`
- Modify: `templates/ask.html`
- Modify: `static/app.css`
- Modify: `static/app.js`

**Interfaces:**
- Consumes: existing elements marked `data-conversation-endpoint`, `data-conversation-form`, `data-conversation-log`, and `/api/conversation`.
- Produces: reusable conversation setup for every panel, `[data-quick-ask-dialog]`, Escape/backdrop close, and Cmd/Ctrl+K open.

- [ ] **Step 1: Write failing tests** asserting the dialog semantics, visible quick-question label, 1200-character limit, endpoint reuse, no `.innerHTML`, and explicit open/close keyboard handlers.
- [ ] **Step 2: Run** the new UI test and `tests.test_absorb_conversation_web` and confirm only the new UI assertions fail.
- [ ] **Step 3: Implement** the shared dialog and refactor the conversation initializer to bind each panel independently while preserving safe `textContent` rendering.
- [ ] **Step 4: Run** the same tests and confirm all pass.

### Task 4: Responsive polish and browser acceptance

**Files:**
- Modify: `static/app.css`
- Create: `tests/visual_qa_research_command.py` only if the existing visual QA server cannot expose the required fixture state.

**Interfaces:**
- Consumes: completed shell, dashboard, and Ask dialog.
- Produces: 1440px, 736px, and 390px verified layouts with screenshots saved outside tracked source.

- [ ] **Step 1: Start** `tests/visual_qa_server.py` using the repository virtual environment without production credentials or external writes.
- [ ] **Step 2: Use Playwright** to visit `/`, open and close Ask ABSORB, activate the US switch and confirm `/reports/us`, return through the wordmark, and collect console/page errors.
- [ ] **Step 3: At 1440px, 736px, and 390px assert** `scrollWidth === clientWidth`, visible focusable controls, preserved mobile bottom navigation, and a dialog contained within the viewport.
- [ ] **Step 4: Run** `python -m unittest tests.test_absorb_brand tests.test_absorb_security tests.test_absorb_conversation_web tests.test_web_product tests.test_route_inventory tests.test_dashboard_service tests.test_reports_template` and confirm zero failures.
- [ ] **Step 5: Run** `git diff --check` and inspect the final diff for unrelated files, fake market data, inline styles, external fonts, unsafe CSP changes, and accidental `.playwright-cli` artifacts.

