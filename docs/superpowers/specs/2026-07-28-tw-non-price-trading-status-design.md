# TW Non-Price Trading Status Contract Design Freeze

## Status

This is the authoritative contract for the implemented TW non-price observation path. It supersedes the earlier planning-time docs-only boundary. The implementation is based on `9c10b6af385306d35582eec30df1b16b6034db7f` and keeps the existing price history immutable.

The contract has three publishable observation kinds:

- `regular_price`: verified target-session OHLCV exists.
- `official_no_regular_trade`: the official target-date price payload contains the symbol and all four OHLC fields are official empty markers. The official volume field is preserved verbatim; an empty marker, zero, or another non-negative numeric value is not converted into regular-session volume.
- `officially_suspended`: independent official lifecycle evidence proves a suspension interval covers the target session.

An effective termination is a universe disposition, not a publishable trading status. Any other missing-price state fails closed.

## Root cause and safety invariant

Before this change, `_price_row()` returned `None` for an official row with blank OHLC, so the parser discarded both the raw row and proof that no regular price existed. The terminal gate then saw only a stale stock artifact and correctly rejected the batch. Relaxing that gate would have allowed the stale `Close` to enter current-day market breadth, events, reports, and stock pages.

The invariant is therefore:

> A target session is publishable only when every configured symbol is accounted for by a target-date regular price, a same-symbol and same-date verified non-price status, or an explicit operational/universe disposition. A prior `Close` is never relabelled as the target session.

No code path may forward-fill, synthesize, copy, or date-shift OHLCV.

## Universe partition

For a configured universe `U`, manifest v3 freezes the disjoint partition:

```text
U = R union N union F
R intersect N = R intersect F = N intersect F = empty
```

- `R`: artifacts with `observation_kind == regular_price` and target-date OHLCV.
- `N`: artifacts with one of the two verified non-price statuses.
- `F`: operational failures and universe dispositions, including effective termination and existing exclusion state.

Unknown missing prices are not members of `F`; they abort the run before manifest creation.

## A. Official lifecycle evidence

### Sources and request identity

Lifecycle data is loaded only when a configured symbol is absent from the regular-price partition. Exchange membership comes from catalog metadata, never from symbol patterns or symbol allowlists.

| Source ID | Exchange | Official endpoint | Parameters |
|---|---|---|---|
| `twse_current_stop` | TWSE | `https://www.twse.com.tw/rwd/zh/violation/stop` | `response=json` |
| `twse_intraday_halt` | TWSE | `https://openapi.twse.com.tw/v1/exchangeReport/TWTAWU` | none |
| `twse_reduction_resume` | TWSE | `https://www.twse.com.tw/rwd/zh/reducation/TWTAUU` | target date and `response=json` |
| `twse_reduction_detail` | TWSE | `https://www.twse.com.tw/rwd/zh/reducation/TWTAVUDetail` | symbol and source file date derived from the verified resume row |
| `twse_termination` | TWSE | `https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml` | none |
| `tpex_current_mode` | TPEx | `https://www.tpex.org.tw/openapi/v1/tpex_cmode` | none |
| `tpex_suspend_history` | TPEx | `https://www.tpex.org.tw/openapi/v1/tpex_spendi_history` | none |
| `tpex_termination` | TPEx | `https://www.tpex.org.tw/www/zh-tw/company/deListed` | empty code, target year, all reasons |

Every response must pass HTTP, size, JSON, schema, date, source identity, row-shape, and duplicate/conflict validation.

### Content-addressed source cache

Raw official evidence is stored below DataRoot only:

```text
source-cache/tw-official/v2/<target-date>/<source-id>/
  metadata.json
  objects/<payload-sha256>.json.gz
```

`metadata.json` schema v2 binds:

- source ID and target market date;
- object path and uncompressed payload SHA-256;
- compressed SHA-256, compressed size, and uncompressed size;
- parser version;
- official URL identity;
- fetch timestamp and date-verification mode.

The existing canonical cache v1 remains readable by the legacy exact path. Status-aware source schema `tw-official-historical-v3` requires raw evidence where the old canonical cache cannot prove a dropped row. Missing metadata, wrong parser version, path escape, size mismatch, decompression failure, or either hash mismatch fails closed; it is never treated as a cache miss.

### Normalized lifecycle event

Each `suspend`, `resume`, or `terminate` event is schema v1 and contains:

```json
{
  "schema_version": 1,
  "exchange": "TWSE or TPEx",
  "symbol": "normalized symbol",
  "event_type": "suspend or resume or terminate",
  "effective_date": "YYYY-MM-DD",
  "source_id": "official source ID",
  "payload_sha256": "64-hex",
  "raw_row_sha256": "64-hex",
  "raw_fields": "the exact source row",
  "parser_version": "tw-lifecycle-parser-v2",
  "evidence_sha256": "64-hex"
}
```

`evidence_sha256` is SHA-256 over canonical UTF-8 JSON with that field omitted, sorted keys, compact separators, and no NaN.

### Interval and precedence rules

Events are ordered by effective date, then `suspend`, `resume`, `terminate` precedence for events on the same date.

- A suspension is valid on `valid_from <= target < valid_through_exclusive`.
- A resume closes the interval on the resume session.
- A termination effective on or before target produces `officially_terminated` and removes the symbol from the active publishable partition.
- A later official suspension starts a new lifecycle era after an older termination of a reused symbol.
- A target-session official raw price row (regular or all-blank OHLC) on the same exchange supersedes only a strictly earlier termination for a reused symbol. The snapshot manifest binds the superseded termination evidence hash. A same-session termination, any active suspension/price overlap, or an unbound identity conflict still aborts.
- Multiple open suspensions, resume without an open suspension, mixed symbol/exchange events, duplicate conflicting events, or overlapping price/status identity fail closed.
- When a covering lifecycle suspension and a blank official price row coexist, the lifecycle status wins. The blank row remains in the raw cache but is not mixed into the suspension evidence identity.

## B. Daily status evidence

### `official_no_regular_trade`

This status is derived only from an official target-date raw price row for the same symbol. All four OHLC values must be official empty markers. The raw volume field must be an official empty marker or parse to a non-negative number; it is evidence only and is never exposed as current volume.

Required evidence fields are:

```json
{
  "schema_version": 1,
  "status": "official_no_regular_trade",
  "market": "TW",
  "exchange": "TWSE or TPEx",
  "symbol": "normalized symbol",
  "target_market_date": "YYYY-MM-DD",
  "source_id": "twse_price or tpex_price",
  "payload_sha256": "64-hex",
  "raw_row_sha256": "64-hex",
  "raw_fields": {
    "symbol": "exact normalized symbol",
    "name": "official display name",
    "open": "official raw value",
    "high": "official raw value",
    "low": "official raw value",
    "close": "official raw value",
    "volume": "official raw value"
  },
  "parser_version": "tw-official-historical-parser-v3",
  "evidence_sha256": "64-hex"
}
```

The completed daily snapshot may add `lifecycle_source_hashes` as precedence lineage. If present it is covered by the final evidence hash. A partial OHLC row, prose value, negative/unparseable volume, wrong source/exchange pair, or hash mismatch is invalid.

### `officially_suspended`

This status is produced only by resolving valid lifecycle events:

```json
{
  "schema_version": 1,
  "status": "officially_suspended",
  "market": "TW",
  "exchange": "TWSE or TPEx",
  "symbol": "normalized symbol",
  "target_market_date": "YYYY-MM-DD",
  "valid_from": "YYYY-MM-DD",
  "valid_through_exclusive": "YYYY-MM-DD or null",
  "evaluated_through": "YYYY-MM-DD",
  "lifecycle_events": ["exact normalized event chain"],
  "parser_version": "tw-lifecycle-parser-v2",
  "evidence_sha256": "64-hex"
}
```

Validation re-runs the lifecycle resolver and requires byte-equivalent normalized output. Price-row absence is never proof of suspension, and production code contains no symbol-specific status table.

## C. Snapshot and cache contract

An `OfficialDailySnapshot` with source schema `tw-official-historical-v3` carries three disjoint mappings:

- `price_by_symbol`: canonical target-date price rows;
- `trading_status_by_symbol`: verified `official_no_regular_trade` and `officially_suspended` evidence;
- `terminated_by_symbol`: effective lifecycle dispositions.

The daily snapshot manifest binds the six canonical official source hashes, raw price source hashes, lifecycle source hashes, per-symbol status evidence hashes, termination evidence hashes, parser version, target date, and production validation thresholds.

No blank OHLC raw row is converted into a price row. Warm-cache reuse must reproduce the same hashes and partitions as a cold fetch.

## D. Artifact and terminal gate

Stock artifact schema v2 keeps `as_of` as the price date:

```json
{
  "schema_version": 2,
  "target_market_date": "YYYY-MM-DD",
  "observation_as_of": "YYYY-MM-DD",
  "latest_regular_price_date": "YYYY-MM-DD",
  "as_of": "YYYY-MM-DD",
  "observation_kind": "regular_price or verified status",
  "trading_status_evidence": "object or null",
  "daily": ["unchanged historical price rows"]
}
```

Frozen equalities:

```text
observation_as_of == target_market_date
as_of == latest_regular_price_date == date(daily[-1])
regular_price       => as_of == target_market_date and status evidence is null
non-price status    => as_of < target_market_date and evidence symbol/date/status match
```

The terminal gate retains checkpoint, batch identity, official-series manifest, reconciliation lineage, artifact SHA, and exclusion-file checks. It accepts a non-target price date only for an exact status member of the same target snapshot. Unknown gaps, evidence drift, price/status overlap, target-dated synthetic rows, or missing history abort the run.

Catch-up progress is measured by `observation_as_of`, never by `as_of` or `latest_regular_price_date`; a verified non-price session advances observation progress without inventing a price. When the gap exceeds the ten-snapshot request bound, the same controlled invocation advances only lagging artifacts through intermediate verified sessions. Every segment retains the bound (and counts the reconciliation baseline), intermediate segments cannot publish, and only the final full-universe terminal gate may update the quant pointer.

## E. Manifest and publish contract

Manifest schema v3 uses `target_market_date` and `observation_as_of`; it does not overload `market_as_of`.

Required counters and formulas are:

```text
observation_count = regular_price_symbol_count + expected_non_price_symbol_count
universe_count = observation_count + operational_failure_count
regular_price_denominator = universe_count - expected_non_price_symbol_count
regular_price_coverage = regular_price_symbol_count / regular_price_denominator
observation_coverage = observation_count / universe_count
operational_failure_rate = operational_failure_count / universe_count
```

`expected_non_price_symbols` maps each status symbol to status, evidence SHA, artifact SHA, and latest regular price date. `operational_failed_symbols` is a separate sorted list. Every symbol entry binds the content-addressed gzip object, compressed and uncompressed sizes, artifact SHA, observation kind, observation date, and price date.

Publication derives all partitions from validated artifacts; callers cannot supply expected status membership. Any unknown missing artifact preserves the previous latest pointer.

### v2/v3 compatibility

- Pointer, manifest, and object schemas dispatch on exact versions; mixed versions are rejected.
- Existing schema-v2 price-only manifests remain readable and rollbackable and cannot carry status metadata.
- Repository cache identity includes market, manifest schema, and manifest SHA.
- Upload preflight validates all local pointer, manifest, object, arithmetic, date, hash, gzip, JSON, and evidence bindings before the first copy.
- Upload order is immutable objects, immutable manifest, dependent products, then generation-guarded mutable pointers.

## F. Dashboard, reports, stock pages, and LINE

Only `regular_price` artifacts enter market, industry, event, ETF, return, breadth, volume, or technical calculations.

Dashboard schema v2 adds `trading_status_observations`, each with exact symbol, name, status, fixed Chinese label, observation date, latest regular price date, evidence SHA, and an optional `last_regular_close`. The optional value is valid only when explicitly paired with its actual latest regular price date.

Data quality reports separate:

- `regular_price_count`;
- `verified_status_count`;
- `operational_failure_count`;
- regular-price and observation coverage.

Observation metadata and the professional report carry the status list separately from `stock_events`. Statuses never enter `EVENT_POLICY_TABLE`, positive/risk/high-anomaly lists, or daily price events.

Status-first stock and LINE views expose no current price, price move, return, volume ratio, indicators, chart, technical event, risk event, or recommendation. They may show only the fixed status label, evidence-bound observation date, evidence SHA, and an explicitly dated last regular close.

## Fail-closed matrix

| Condition | Result |
|---|---|
| Target OHLCV is valid | `regular_price` |
| All OHLC fields are official empty markers and the raw row is hash-bound | `official_no_regular_trade` |
| Independent lifecycle interval covers target | `officially_suspended` |
| Effective termination covers target | universe disposition; no status artifact required |
| No price row and no lifecycle evidence | abort |
| Partial blank/prose OHLC or invalid volume | abort |
| Cache metadata, parser version, size, path, or hash mismatch | abort |
| Price and lifecycle status overlap for one exchange identity | abort |
| Artifact/evidence/manifest symbol, date, or SHA mismatch | abort |
| v2/v3 pointer or manifest mismatch | reject load/upload |
| Status attempts to enter price calculations | reject dashboard/report build |

## Explicit non-modifications

This contract does not change:

- historical OHLCV values or price numeric semantics;
- model features, inference, promotion, backtests, or recommendation policy;
- exclusion CSV schema or operator actions;
- immutable objects already published;
- IAM, credentials, or secret handling;
- the requirement to use official TWSE/TPEx sources for the production path.

No dependency, database, generalized event framework, status-only artifact family, or symbol-specific production configuration is introduced.

## Rollback boundary

1. Deploy code that reads both manifest v2 and v3 before selecting a v3 pointer.
2. Capture pointer generations, object identities, Cloud Run revision, and Scheduled Task state before mutation.
3. If publication validation fails, stop later pointers and restore the last verified pointer with its generation precondition.
4. Move Cloud Run traffic back to the recorded compatible revision if service validation fails.
5. Revert code only after no mutable pointer references schema v3.
6. Never delete or rewrite raw evidence, status artifacts, immutable manifests, or report history during rollback.

## Design self-review

- All date fields have one meaning; artifact `as_of` remains the latest regular price date.
- Suspension proof is lifecycle-only and round-trip verifiable.
- No missing row is interpreted as a status.
- Blank OHLC evidence never becomes a price row or current-session volume.
- Manifest denominators and all three universe partitions are explicit.
- v1 cache, v2 manifest, and v3 status-aware paths remain separate and exact.
- Consumers cannot expose stale values as target-session prices.
- Every decision and field is final, and no production symbol is encoded as configuration.
