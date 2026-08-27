# ABSORB Responsive Dashboard and Industry AI Design

## Goal

Make the primary navigation, 4K reading experience, industry exploration, and Ask ABSORB workspace easier to use without weakening the published-data and model-quality contracts.

## Product language

- Do not use「新手」or「初學者」in interface copy.
- Use「AI 關注公司」instead of buy, sell, or recommendation language.
- Backtests validate model reliability. The published run's current inference ranks companies.
- The same market date, model version, and input snapshot must produce the same ranking. A new market date or model version may change the list.

## Navigation and responsive layout

- Remove the duplicated destination grid from the bottom of the dashboard.
- Keep the existing primary destinations in the top Dashboard navigation and add ASK ABSORB and 學習 there.
- On narrow screens, retain five primary mobile destinations. ASK ABSORB remains available through its persistent trigger; 學習 is reached from the top navigation overflow rather than adding another bottom row.
- Increase the readable content width on large displays and raise the base, navigation, metadata, and card text sizes. The layout must remain fluid at desktop, 4K, tablet, and mobile widths.
- Preserve the script ABSORB wordmark, its current animation, the Greek Villa neutral palette, and the restrained editorial research-desk visual language.

## Industry observation

- Replace the separate「產業實際強弱」and「產業觀察清單」blocks with one interactive industry list.
- Each industry card uses the existing positive, neutral, and negative heat tones derived from five-day relative market return.
- The collapsed card shows industry name, five-day relative return, coverage, and number of available companies.
- Activating a card expands an inline detail region with breadth, MA20 participation, volume position, institutional flow, and up to five AI attention companies.
- The interaction uses native buttons and `aria-expanded`; only one card needs to be open by default, and the page remains useful without JavaScript.

## Dynamic company ranking

- Build the ranking once in the published observation artifact so all viewers of that release see the same result.
- Prefer current five-session model outputs when they are present and eligible. Rank by predicted five-session return, then upward probability, then deterministic symbol order.
- A company is eligible only when the source snapshot, model output, and quality gate are valid for the same market date. Ineligible or incomplete rows are excluded rather than imputed.
- If eligible company-level model outputs are unavailable in the current artifact, fall back to an explicitly labeled「實際動能排序」using five-day return, MA20 status, volume position, and deterministic symbol order. This fallback is descriptive, not an AI forecast.
- Each company row shows name, symbol, ranking basis, five-day forecast price and upward probability when verified, plus a link to the existing stock page.
- If neither verified model output nor sufficient actual observations exist, show「本次發布沒有足夠資料形成關注名單」and no company names.

## Ask ABSORB

- Keep the existing rectangular click-open sheet and conversation behavior.
- On desktop, the sheet occupies at least half the viewport height, targeting about 60vh, and the conversation log flexes to fill the available space.
- On mobile, retain the existing bottom-sheet behavior while ensuring the usable height remains at least half the viewport and does not collide with safe-area navigation.
- Preserve focus handling, Escape-to-close behavior, and the statement that the assistant does not provide trading instructions.

## Data flow

1. The observation builder receives the authoritative stock and industry snapshots plus any eligible five-session company forecasts.
2. It computes each industry's deterministic `attention_companies` list and records `ranking_basis` as `verified_ai_forecast` or `actual_momentum`.
3. The published dashboard artifact carries that list with the existing industry observations.
4. The industries template renders only the published artifact. It does not calculate rankings in the browser or fetch unverified live prices.

## Failure handling

- Missing forecast fields never become zero, neutral probability, or synthetic prices.
- A failed model gate prevents the `verified_ai_forecast` label and values from appearing.
- Missing or malformed company lists degrade to the empty published-state message while the rest of the industry metrics remain visible.
- Existing market and publication fail-closed behavior remains unchanged.

## Verification

- Unit tests prove deterministic ranking, model-gate exclusion, actual-momentum fallback, and empty-state behavior.
- Route/template tests prove the bottom dashboard destinations are removed, the top navigation exposes ASK ABSORB and 學習, merged industry cards render accessible expansion controls, and company links point to existing stock pages.
- CSS acceptance checks cover larger typography, wider 4K content, and an Ask ABSORB sheet at least half a viewport high.
- Browser acceptance covers desktop, 4K, and mobile layouts; card expansion; company navigation; Ask ABSORB open/close/focus; and console/network errors.

## Scope limits

- No buy/sell instruction engine.
- No new frontend framework or dependency.
- No client-side ranking or randomization.
- No fabricated forecast, probability, price, company, or coverage value.
