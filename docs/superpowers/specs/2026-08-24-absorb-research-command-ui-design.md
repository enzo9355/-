# ABSORB Research Command UI Design

## Goal

Replace the sidebar-led observation website shell and card-grid dashboard with the approved Research Command visual direction while preserving every existing public route, observation-only contract, fail-closed state, and Ask ABSORB API boundary.

## Approved visual direction

- Information architecture follows the Command Room preview: research summary first, then a dense evidence workspace, then reports and source limitations.
- Palette uses cool canvas, paper surfaces, deep navy, desaturated blue, restrained sage, and coral only for risk or negative values.
- Professional credibility comes from real observation dates, source coverage, risk state, limitations, and report provenance.
- English typography prefers locally installed `Avenir Next` or `Avenir`; Traditional Chinese pairs with `Noto Sans TC`, then `PingFang TC` and `Microsoft JhengHei`.
- The navigation wordmark is lowercase cursive text `absorb`, smaller than the original mark and without a circular icon. Existing canonical image assets remain unchanged for favicons, social metadata, LINE, and other non-navigation uses.
- Interaction motion is 120 to 180 milliseconds and limited to transform, opacity, color, and shadow. Reduced-motion preferences disable nonessential motion.

## Shell

- Desktop uses a top navigation bar with the cursive wordmark, existing route links, and a market switch.
- The wordmark links to `/` so it always returns to the main dashboard.
- The market switch is honest about the current data topology: `台股` links to the TW observation dashboard and `美股` links to the verified US report index at `/reports/us`. It must never relabel TW data as US data.
- Mobile keeps the compact header plus the existing five-entry bottom navigation contract.

## Dashboard

- Use only fields already present in the verified observation dashboard snapshot.
- Never display preview-only values, synthetic index prices, invented timestamps, implied live quotes, or fabricated invalidation thresholds.
- Research summary displays observation date, generated time when available, coverage, available universe count, and the first verified daily focus item.
- The evidence workspace visualizes market returns, advancing and declining counts, MA20 and MA60 breadth, new highs and lows, volume ratio, realized volatility, industry relative strength, and data quality.
- Native `progress`, SVG attributes, and server-rendered text provide visualization without inline styles or unsafe CSP changes.
- Empty and unavailable values remain explicit and are never converted to zero.
- Existing daily report cards and all existing navigation destinations remain reachable.

## Ask ABSORB

- Add one floating Ask ABSORB trigger to the shared shell.
- Desktop opens a restrained side/bottom dialog; mobile opens a bottom-sheet-style dialog.
- Reuse the existing JSON-only `/api/conversation` endpoint and the exact `{question: string}` request contract.
- Keep visible labels, 1200-character limit, `aria-live`, keyboard Escape close, backdrop close, and Cmd/Ctrl+K open.
- The standalone `/ask` route remains available and uses the same conversation behavior.

## Responsive and accessibility

- Validate at 1440px, 736px, and 390px.
- No horizontal page overflow.
- Mobile preserves research conclusion, risk state, limitations, observation date, and data quality.
- Touch targets are at least 44px.
- Preserve skip link, semantic headings, native controls, visible focus, `aria-current`, dialog labeling, and reduced motion.

## Non-goals

- No deployment, Cloud Run change, GCS write, scheduler change, or data pipeline change.
- No new US dashboard snapshot reader or fake same-page US dashboard.
- No proprietary font files or external font requests.
- No changes to the conversation API security contract.

