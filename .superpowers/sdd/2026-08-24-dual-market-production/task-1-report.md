# Task 1 report — Market-aware product shell and US research summary

## Scope completed

- Added `/us`, `/us/market`, `/us/industries`, and `/us/stocks`; US navigation retains the five market-specific destinations on desktop and mobile.
- Added a US market-summary adapter that emits literal fields from `ProfessionalPostCloseReport`: market, source/applicable dates, industries, key events, securities, and validation.
- Extracted the existing Canonical Object verification into `_load_verified_professional_report`. Both existing post-close reports and new US summaries now use the same size, SHA-256, immutable path, schema, and metadata-binding checks.
- `/stock/<code>` now passes a market context selected from the accepted symbol; `/stock/AAPL` renders the US product shell. The stock search preserves its originating market for an unsuccessful query.
- Updated the product shell to use `ABSORB` and `ASK ABSORB` consistently, including visible labels and close control accessibility text.
- Added US templates that retain the incumbent evidence-first research presentation, safe unavailable states, long-text wrapping, tabular numeric dates, mobile layout, and existing reduced-motion behavior.

## TDD record

- RED: `tests.test_web_product`, `tests.test_report_web`, `tests.test_route_inventory`, and `tests.test_absorb_brand` failed as expected for missing `/us*` routes, missing market-summary module, old mixed-case labels, and absent route inventory.
- RED: the dedicated US-summary success case returned `404` before route implementation.
- GREEN: focused product/report/route/brand suite passed 65 tests.
- Refactor check: an insertion-boundary issue temporarily made the optional regression overlay unreachable; root cause was verified by source inspection, then `tests.test_regression_route_integration` passed 6/6 after the helper was moved outside that function.

## Verification

- Focused product/report/route/brand: 65 tests passed.
- US Canonical Object tamper test: valid-but-hash-mismatched canonical content returns safe `503` without exposing the altered value.
- Regression route integration: 6 tests passed.
- Full `unittest discover -s tests -p 'test_*.py'`: passed (exit code 0).
- `python -m compileall -q stock_papi reporting tests`: passed.
- `git diff --check`: passed.
- Impeccable detector ran once over changed UI files. It found no issues, but operated in degraded regex mode because its optional HTML/CSS parser modules are absent; it is not a replacement for a browser visual pass.

## Concerns

- Browser desktop/390px visual inspection was not completed in this local run. The rendered routes, responsive CSS rules, and accessibility assertions were verified, but the detector's degraded mode leaves visual inspection as remaining acceptance work.
