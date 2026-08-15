# TEJ private research integration

TEJ is an optional, private research source. It does not replace TWSE, TPEx,
the official calendar, official prices, official lifecycle status, or TW
terminal completeness. The normal `ABSORB-TW-PostClose` path never imports,
fetches, or depends on TEJ.

## Source and entitlement contract

The adapter follows the official TEJ Python API documentation:

- Python API and `tejapi.ApiConfig.info()` entitlement discovery:
  <https://api.tej.com.tw/document_python.html>
- REST/API authentication and usage limits:
  <https://api.tej.com.tw/documents.html>
- REST table access semantics:
  <https://api.tej.com.tw/document_rest.html>

No table code is assumed by ABSORB. Operators must discover the tables returned
by the authenticated account and then pass an entitled table to `fetch`.
`TWN/APRCD` is an example in TEJ's documentation, not a repository default or
an entitlement claim.

Required environment contract:

```text
TEJ_ENABLED=false
TEJ_API_KEY=<environment-only secret>
```

The key is never accepted as a command-line argument, written to cache
metadata, put in a scheduled-task argument, or printed in an exception.
Absent or disabled credentials produce a machine-readable status and do not
fail TW production.

## Private workflow

The separate wrapper is `scripts/run_tej_research.ps1`. It supports:

```text
status       entitlement discovery
fetch        bounded, paginated fetch of an explicitly entitled table
normalize    explicit TEJ field/entity mapping into PIT rows
factor       PIT-safe cross-sectional research factors
challenger   baseline-vs-TEJ walk-forward challenger evaluation
shadow       advisory comparison against official TW values
```

Raw responses are content-addressed under:

```text
D:\AbsorbData\raw\tej\v1\
```

Normalized rows, factor snapshots, challenger results, and shadow reports are
private under `D:\AbsorbData\research\tej\v1\`. They are not public GCS,
Cloud Run, LINE, or website artifacts. Identical payloads reuse the same SHA;
conflicting content at an existing identity fails closed.

The normalize command requires an explicit mapping for the TEJ entity field,
effective date, announcement/availability timestamp, and every normalized
field. Missing announcement timestamps are a schema error. A row is visible
to a prediction only when `available_at <= prediction_time`; later restatements
therefore cannot leak into an earlier backtest.

## Factors and challenger

The factor layer only emits families supported by mapped, numeric source fields:
VALUE, GROWTH, QUALITY, MOMENTUM, LIQUIDITY, RISK, and SENTIMENT. It records
the source payload identity and the as-of time. It does not invent field
mappings or use future cross-sectional samples.

The challenger schema is `tej-challenger-lgbm-v1`. It is evaluated on the same
eligible symbols, dates, walk-forward folds, and transaction assumptions as the
official-feature baseline. The existing `MODEL_FEATURES` and live model version
are unchanged. Automatic promotion and production traffic are permanently
disabled for this integration until an explicit, evidence-backed promotion
contract is satisfied.

Shadow mismatches report both values, symbol/date/field, and both source
identities. They are advisory only and never override official market truth or
block the TW writer.

## Scheduler boundary

`scripts/install_tej_research_task.ps1` is separate from the TW task and is
disabled unless explicitly invoked with `-Enable` while both the enabled flag
and environment-provided key are present. It is not required for daily TW
operation and is not installed by repository tests or deployment scripts.
