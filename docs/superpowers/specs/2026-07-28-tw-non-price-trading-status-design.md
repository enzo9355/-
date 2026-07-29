# TW Non-Price Trading Status Contract Design Freeze

## Status

Phase 1Q-A2 Design Freeze. This document freezes the contract only. It does not authorize Production code changes, test changes, network access, pipeline execution, recovery, publication, upload, GCS mutation, website changes, Scheduled Task changes, LINE delivery, or writes to `D:\AbsorbData`.

Base revision: `9c10b6af385306d35582eec30df1b16b6034db7f`.

## Problem and confirmed root cause

`stock_papi/integrations/market_data/tw_official_historical.py::_price_row()` currently returns `None` when any OHLC or volume value is an official empty marker. The caller therefore retains neither the official raw row nor proof that the row described a symbol without a normal trade. The terminal gate in `stock_papi/batch/tw_official_post_close_cli.py::_assert_complete()` then requires every active artifact's latest daily date to equal the target session.

Relaxing that date check is unsafe. A stale `Close` would become indistinguishable from a target-session price and would contaminate the quant manifest, observation dashboard, reports, technical events, and stock pages. This design never forward-fills, synthesizes, copies, or relabels OHLCV.

## Frozen approach

Three approaches were considered:

1. Relax the terminal date gate. Rejected because stale prices would pass as current.
2. Publish a separate status-only artifact family. Rejected for this phase because it duplicates identity, hash, upload, and loader paths.
3. Preserve the existing stock artifact while separating observation date from price date and binding a verified status record. Selected because it reuses the current content-addressed artifact and manifest flow without changing any historical OHLC row.

A symbol observed for the target session has exactly one `observation_kind`:

- `regular_price`: verified target-session OHLCV exists.
- `official_no_regular_trade`: the target-date official price payload contains the same symbol, and all required OHLCV cells use official empty markers.
- `officially_suspended`: an independent official lifecycle record proves that suspension covers the target session.

Any other missing-price state is unrecognized and fails closed. A symbol with no prior valid regular-price artifact also fails closed in this phase; status-only bootstrap is out of scope.

## Date semantics

The following meanings are immutable across artifact, manifest, loader, dashboard, and report code:

| Field | Meaning | Rule |
|---|---|---|
| `target_market_date` | Session requested by the run | Fixed in batch identity before source loading |
| `observation_as_of` | Market session fully verified by the terminal gate | Must equal `target_market_date` before any candidate or publish step |
| `latest_regular_price_date` | Date of the last verified OHLCV row | Must equal the last `daily[].Date` date |
| artifact `as_of` | Existing price-date meaning | Retained; must equal `latest_regular_price_date`, never `observation_as_of` merely because a status exists |

Manifest v3 does not use `market_as_of`; it uses `observation_as_of`. Manifest v2 retains its original `market_as_of` meaning and is never rewritten into v3 in place.

## A. Official lifecycle evidence

### Authoritative sources

Lifecycle evidence is accepted only from these four official exchange datasets:

| `source_id` | Authority | Official dataset |
|---|---|---|
| `twse_suspend_resume` | TWSE | 上市公司停止與恢復交易資訊 |
| `tpex_suspend_resume` | TPEx | 上櫃公司停止與恢復交易資訊 |
| `twse_termination` | TWSE | 終止上市公司資訊 |
| `tpex_termination` | TPEx | 終止上櫃公司資訊 |

`stock_papi/integrations/market_data/tw_trading_status.py` will own an allowlisted source definition for each dataset. Acquisition is HTTPS GET of the official dataset's JSON download with an explicit target-date range and bounded pagination. Redirects to a different origin, non-JSON responses, incomplete pagination, duplicate page identity, response-size overflow, or a field fingerprint different from the pinned fixture fail before the symbol loop. Search results, news, broker pages, price-row absence, and code-maintained symbol lists are never lifecycle evidence.

### Content-addressed cache

Lifecycle payloads live independently from daily price caches:

```text
<DataRoot>/source-cache/tw-lifecycle/v1/objects/<payload_sha256>.json.gz
<DataRoot>/source-cache/tw-lifecycle/v1/index/<target_market_date>.json
```

The immutable object contains the exact response bytes, deterministically gzipped. The atomic index contains only validated references. Existing objects are never overwritten. The index schema is:

```json
{
  "schema_version": 1,
  "target_market_date": "YYYY-MM-DD",
  "valid_from": "YYYY-MM-DD",
  "valid_through": "YYYY-MM-DD",
  "parser_version": "tw-lifecycle-parser-v1",
  "sources": {
    "source_id": {
      "authority": "TWSE",
      "dataset_title": "official dataset title",
      "source_url_identifier": "query-free official URL",
      "payload": "objects/<64-hex>.json.gz",
      "payload_sha256": "64-hex",
      "compressed_sha256": "64-hex",
      "compressed_size": 1,
      "row_count": 1,
      "fetched_at": "RFC3339 UTC"
    }
  },
  "events_sha256": "64-hex"
}
```

`valid_from <= target_market_date <= valid_through` is mandatory. A missing source, expired interval, hash/size mismatch, parser mismatch, malformed raw row, contradictory event, or incomplete pagination makes the entire lifecycle index unusable. A valid cache hit performs zero requests.

### Normalized lifecycle event

Each accepted event is canonical JSON with these exact fields:

```json
{
  "schema_version": 1,
  "market": "TW",
  "exchange": "TWSE",
  "symbol": "numeric symbol",
  "event_type": "suspend",
  "announcement_id": "exchange-issued identifier",
  "announcement_date": "YYYY-MM-DD",
  "effective_date": "YYYY-MM-DD",
  "source_id": "twse_suspend_resume",
  "payload_sha256": "64-hex",
  "raw_row_sha256": "64-hex",
  "parser_version": "tw-lifecycle-parser-v1"
}
```

`raw_row_sha256` is SHA-256 of the canonical UTF-8 JSON representation of the complete raw row using sorted keys, compact separators, and no NaN. `payload_sha256` hashes the exact uncompressed response bytes. The cache must retain those bytes so both hashes can be recomputed.

### Suspension interval and override rules

- `suspend` is effective from its `effective_date`, inclusive.
- `resume` closes the matching suspension at its `effective_date`, exclusive; the resume session is not suspended.
- `terminate` closes any open suspension at its `effective_date`, exclusive. A terminated symbol remaining in the active universe is an operational contract failure, not `officially_suspended`.
- Same-symbol events are ordered by `(effective_date, event precedence, announcement_id)`, where `terminate` overrides `resume`, and `resume` overrides `suspend` on the same date.
- A duplicate `announcement_id` with different content, a second open suspension without an intervening close, a resume without an open suspension, or any event chain that cannot produce one deterministic interval is contradictory and fails closed.
- An announcement dated after `target_market_date` cannot prove the target session.
- `officially_suspended` requires exactly one open, non-terminated suspension interval containing the target session.

No Production module contains a symbol-specific branch, allowlist, or constant.

## B. Daily non-price evidence

### `official_no_regular_trade`

The price parser must classify a raw row before attempting numeric conversion. The classification is valid only when:

- top-level and table dates equal `target_market_date`;
- the symbol is valid and belongs to the source exchange;
- all five required cells, open/high/low/close/volume, are official empty markers;
- the full row length and field fingerprint match the pinned source schema;
- the payload and raw row remain available for hash verification.

A partial placeholder row, valid OHLC with missing volume, missing OHLC with numeric volume, or any impossible numeric relationship is malformed source data and fails the source. A valid zero-volume row with valid OHLC remains `regular_price`.

The normalized evidence is:

```json
{
  "schema_version": 1,
  "status": "official_no_regular_trade",
  "market": "TW",
  "exchange": "TPEx",
  "symbol": "numeric symbol",
  "target_market_date": "YYYY-MM-DD",
  "source_id": "tpex_price",
  "payload_sha256": "64-hex",
  "raw_row_sha256": "64-hex",
  "raw_fields": {
    "symbol": "raw text",
    "name": "raw text",
    "open": "raw text",
    "high": "raw text",
    "low": "raw text",
    "close": "raw text",
    "volume": "raw text"
  },
  "parser_version": "tw-official-historical-parser-v3"
}
```

### `officially_suspended`

The evidence uses the same envelope fields but `status` is `officially_suspended` and `lifecycle_events` contains the exact normalized event records that open and, when applicable, bound the interval. `evidence_sha256` is computed over the complete normalized status evidence excluding the `evidence_sha256` field itself.

Price-row absence never creates this status. If lifecycle evidence says suspended while the same session has valid OHLCV, or termination covers the session, the snapshot is contradictory and fails closed. When a blank official price row corroborates suspension, the lifecycle evidence remains primary and both hashes are retained.

## C. Snapshot and cache contract

`OfficialDailySnapshot` gains one immutable mapping:

```python
trading_status_by_symbol: Mapping[str, Mapping[str, Any]]
```

For a date, `price_by_symbol` and `trading_status_by_symbol` are disjoint. Every status record must match the snapshot date and symbol. The daily snapshot manifest hash includes sorted price source hashes, lifecycle source hashes, and each status `evidence_sha256`.

Daily official cache v2 stores exact raw response bytes as well as canonical parsed rows:

```text
<DataRoot>/source-cache/tw-official/v2/<date>/<source-id>/objects/<payload_sha256>.json.gz
<DataRoot>/source-cache/tw-official/v2/<date>/<source-id>/metadata.json
```

Cache v1 remains readable for canonical price rows. It cannot prove a non-price status because it lacks raw payload and raw-row identity. Therefore:

- a v1 cache hit is accepted only when every required active symbol has canonical price data for that source date;
- if a missing active price needs classification, v2 evidence is mandatory;
- v1 is never rewritten or deleted;
- a parser-version mismatch is a cache miss only when no incompatible metadata claims the same v2 identity; malformed or hash-mismatched existing v2 data fails closed.

No blank raw row is converted into a canonical price row. Snapshot-series schema advances to `tw-official-historical-v3`; v2 checkpoints are incompatible and are archived through the existing checkpoint identity rule.

## D. Artifact and terminal gate

### Stock artifact schema v2

The existing content-addressed stock artifact becomes schema v2 for status-aware TW runs:

```json
{
  "schema_version": 2,
  "market": "TW",
  "symbol": "numeric symbol",
  "target_market_date": "YYYY-MM-DD",
  "observation_as_of": "YYYY-MM-DD",
  "latest_regular_price_date": "YYYY-MM-DD",
  "as_of": "YYYY-MM-DD",
  "observation_kind": "regular_price|official_no_regular_trade|officially_suspended",
  "trading_status_evidence": null,
  "daily": [],
  "latest": {},
  "backtest": {}
}
```

For all kinds, `as_of == latest_regular_price_date == date(daily[-1].Date)`. For `regular_price`, those dates also equal `observation_as_of == target_market_date` and `trading_status_evidence` is null. For either non-price kind, `observation_as_of == target_market_date`, `as_of < observation_as_of`, the historical `daily` array is unchanged, and `trading_status_evidence` is mandatory and hash-bound. A status artifact never receives a synthetic target-date `daily` row.

### Terminal partition

For the exact active universe, the terminal gate creates three disjoint sets:

- `regular_price_symbols`: target-date canonical price and target-date artifact.
- `expected_non_price_symbols`: target-date, same-symbol, same-snapshot verified status and a valid prior regular-price artifact.
- `operational_failed_symbols`: recognized runtime, artifact, pending, or excluded failures; never a substitute for missing status proof.

The union must equal the universe and pairwise intersections must be empty. An unclassified symbol, unknown missing price, status hash mismatch, stale `observation_as_of`, future price date, price/status conflict, or artifact/status mismatch fails the run before manifest creation. Existing strict source lineage and reconciliation checks remain required.

## E. Manifest and publish contract

### Manifest schema v3

New status-aware TW publication emits immutable manifest schema v3 and pointer schema v3. The manifest contains:

```json
{
  "schema_version": 3,
  "market": "TW",
  "generated_at": "RFC3339 UTC",
  "target_market_date": "YYYY-MM-DD",
  "observation_as_of": "YYYY-MM-DD",
  "universe_count": 1,
  "observation_count": 1,
  "regular_price_symbol_count": 1,
  "expected_non_price_symbol_count": 0,
  "operational_failure_count": 0,
  "regular_price_denominator": 1,
  "regular_price_coverage": 1.0,
  "observation_coverage": 1.0,
  "operational_failure_rate": 0.0,
  "expected_non_price_symbols": {},
  "operational_failed_symbols": [],
  "symbols": {}
}
```

The exact arithmetic is:

```text
observation_count = regular_price_symbol_count + expected_non_price_symbol_count
universe_count = observation_count + operational_failure_count
regular_price_denominator = universe_count - expected_non_price_symbol_count
regular_price_coverage = regular_price_symbol_count / regular_price_denominator
observation_coverage = observation_count / universe_count
operational_failure_rate = operational_failure_count / universe_count
```

`regular_price_denominator` must be positive. `expected_non_price_symbols` is a sorted object keyed by symbol; each value contains `status`, `evidence_sha256`, `artifact_sha256`, and `latest_regular_price_date`. Every key also exists in `symbols` with `observation_kind` and the same hashes. `operational_failed_symbols` is a sorted unique list. The three symbol partitions are validated from content, never trusted from counters.

The existing TW failure threshold still applies to `operational_failure_rate`. The official terminal gate remains stricter for active symbols: unknown missing-price failures cannot be published merely because the rate is below five percent.

### v2/v3 compatibility

- Publisher emits v3 only for the new status-aware source schema.
- Existing immutable v2 manifests and schema-v2 pointers remain readable and rollbackable as price-only snapshots.
- A v2 manifest may not contain or imply non-price status.
- A v3 pointer may reference only a v3 manifest; a v2 pointer may reference only v2.
- Report and Cloud Run source loaders accept both versions through separate exact validators, not a permissive shared branch.
- `reporting/migrate_quant_manifest.py` remains v2-only and cannot convert price-only history into status evidence.
- Upload validates pointer, manifest, object size, compressed/uncompressed hashes, partition arithmetic, and evidence cross-bindings before copying any v3 object. It uploads immutable objects and manifest before the pointer.
- Any unknown schema, mixed pointer/manifest version, missing v3 field, extra partition symbol, evidence mismatch, invalid denominator, or inconsistent date is rejected.

## F. Reports, dashboard, and stock surfaces

Report loading separates verified price observations from verified status observations. Market breadth, returns, ranking, price movement, volume, indicators, and technical events consume only `regular_price` artifacts whose price date equals `observation_as_of`.

For non-price symbols:

- no target-session `Close`, change, `price_move`, volume, volume ratio, moving average state, RSI, MACD, KD, breakout, technical event, or recommendation is generated;
- the allowed labels are `停止買賣` for `officially_suspended` and `當日無正常交易` for `official_no_regular_trade`;
- the label is emitted only from a status record already validated against the manifest evidence hash;
- an optional last regular close may be displayed only as `最後正常交易收盤（latest_regular_price_date）`; it is never headed as today's close or included in target-session calculations;
- report/dashboard quality shows regular-price, expected-non-price, and operational-failure counts separately.

Observation products add `trading_status_observations` as a separate list. They do not place status records in `stock_events`. `source_market_date`, `data_as_of`, and dashboard `observation_as_of` use the verified observation session, while any stale price uses its explicit `latest_regular_price_date`.

## Fail-closed matrix

| Condition | Result |
|---|---|
| Valid target-date OHLCV | `regular_price` |
| All required raw OHLCV fields are official empty markers, hashes valid | `official_no_regular_trade` |
| Independent lifecycle interval covers target and hashes valid | `officially_suspended` |
| Price row missing without either proof | Reject run |
| Partial empty markers | Reject source |
| Valid price and suspension evidence on same session | Reject snapshot |
| Resume effective on target | Not suspended; require regular price or valid no-trade evidence |
| Termination effective on/before target while symbol remains active | Operational contract failure |
| Cache missing, expired, contradictory, or hash mismatched | Reject before symbol loop |
| v1 cache needed to prove status | Reject; v2 evidence required |

## Implementation boundaries

The implementation may change only the official source contracts/cache, artifact/terminal gate, manifest/upload/loaders, and report/status presentation named in the implementation plan.

It must not change:

- LightGBM, feature formulas, backtest formulas, prediction, recommendation policy, or model promotion;
- price endpoints or canonical OHLC numeric semantics;
- legacy reconciliation replacement rules or immutable backups;
- exclusion-list state transitions or hardcode any symbol;
- GCS bucket layout, IAM, Cloud Run deployment, Scheduled Tasks, LINE delivery, or report publication policy;
- historical OHLC rows, `D:\AbsorbData`, or existing cache/object bytes during implementation tests;
- `reporting/migrate_quant_manifest.py` or any existing immutable v2 manifest.

## Rollback boundary

Rollback is pointer-based and non-destructive:

1. Before first v3 publication, old code and all v2 pointers remain unchanged.
2. After v3 publication, restore the last verified schema-v2 pointer using the existing generation-matched rollback procedure.
3. v3 manifests, objects, cache v2 objects, and lifecycle objects remain immutable and are not deleted.
4. Code rollback is safe only after the pointer no longer references v3.
5. Rollback never rewrites an artifact, forward-fills a price, or converts status evidence into a failure exclusion.

## Acceptance criteria

- A raw official blank row survives parsing as evidence but never as price.
- Suspension is never inferred from a price gap and never encoded as a symbol constant.
- Every accepted status is same-date, same-symbol, payload-bound, row-bound, parser-bound, and snapshot-bound.
- Artifact `as_of` always means latest regular price date.
- Manifest arithmetic partitions the complete universe without overlap.
- All consumers suppress target-session price and technical output for status symbols.
- v2 remains read-only compatible; v3 fails closed in old or mismatched loaders.
- No placeholder, unresolved contract choice, or automatic data repair remains in this design.

## Design self-review

- Placeholder scan: no deferred decision or placeholder remains.
- Consistency: artifact price date, observation date, snapshot date, and manifest date have one meaning each.
- Security/data trust: hashes, paths, sizes, schemas, date windows, pagination, and source authority fail closed.
- Scope: no Production operation or unrelated model/refactor work is authorized.
- Symbol policy: all status membership comes from verified data; Production code contains no symbol-specific rule.
