# TW Non-Price Trading Status Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve official TW non-price trading evidence without converting stale or blank rows into target-session OHLCV, then carry the verified status through artifacts, terminal completeness, manifest v3, loaders, dashboard, and reports.

**Architecture:** Extend the existing official daily snapshot with one hash-bound status mapping, retain stock artifact `as_of` as the latest regular price date, and add `observation_as_of` for the verified session. Publish a schema-v3 manifest that partitions the complete universe into regular-price, expected-non-price, and operational-failure symbols; keep schema v2 on a separate exact read path for rollback compatibility.

**Tech Stack:** Python 3.10 standard library and existing `requests`/`pandas`, `unittest`, deterministic gzip/JSON/SHA-256, Windows PowerShell 5.1.

## Global Constraints

- Implement from base `9c10b6af385306d35582eec30df1b16b6034db7f` on `codex/tw-non-price-status-contract`.
- Never forward-fill, synthesize, copy, or relabel OHLCV.
- Never infer suspension from price-row absence and never hardcode a symbol.
- Artifact `as_of` remains `latest_regular_price_date`; `observation_as_of` is the verified target session.
- Missing, expired, contradictory, incomplete, or hash-mismatched status evidence fails closed.
- Official-source failure occurs before the symbol loop and before checkpoint or artifact mutation.
- Tests use temporary roots and sanitized fixtures only; no HTTP, FinMind, GCS, Cloud Run, LINE, Scheduled Task, publish, upload, recovery, or `D:\AbsorbData` access.
- Do not modify LightGBM, feature formulas, backtest formulas, recommendation policy, legacy reconciliation semantics, exclusion transitions, or `reporting/migrate_quant_manifest.py`.
- Every RED commit contains tests only. Every GREEN commit contains the minimum Production changes for that stage.
- Do not run the full suite until all four stages and independent review are complete; each stage uses only its named focused commands.

---

## File map

### B1: Source evidence and cache

- Create `stock_papi/integrations/market_data/tw_trading_status.py` — lifecycle sources, status schemas, hashes, interval evaluation, price-row status classification.
- Modify `stock_papi/integrations/market_data/tw_official_bulk.py` — add immutable `trading_status_by_symbol` to `OfficialDailySnapshot`.
- Modify `stock_papi/integrations/market_data/tw_official_historical.py` — retain raw payload/row identity and assemble price plus status without overlap.
- Modify `stock_papi/integrations/market_data/tw_official_cache.py` — add raw payload cache v2 while preserving v1 price-only reads.
- Create `tests/test_tw_trading_status.py`.
- Modify `tests/test_tw_official_historical.py`.
- Modify `tests/test_tw_official_bulk.py`.

### B2: Artifact dates and terminal gate

- Modify `local_quant.py` — emit artifact schema v2 and pass verified status into snapshot building.
- Modify `stock_papi/quant/tw_incremental.py` — expose immutable per-symbol target status and preserve historical daily rows.
- Modify `stock_papi/quant/tw_artifact_audit.py` — audit observation date separately from price date.
- Modify `stock_papi/batch/tw_official_post_close_cli.py` — partition the universe and enforce the hash-bound terminal gate.
- Modify `tests/test_tw_incremental.py`.
- Modify `tests/test_tw_official_post_close_cli.py`.
- Modify `tests/test_local_quant.py`.

### B3: Manifest v3, upload, and loaders

- Modify `local_quant.py` — publish immutable manifest/pointer v3 from the terminal partition.
- Modify `reporting/schemas.py` — represent v2 price-only and v3 observation/status source metadata.
- Modify `reporting/source_loader.py` — exact v2/v3 validators and object/status cross-binding.
- Modify `stock_papi/repositories/quant_snapshots.py` — fail-closed Cloud Run v3 manifest and artifact loading.
- Modify `scripts/upload_local_quant.ps1` — v3 preflight and immutable-before-pointer upload ordering.
- Modify `tests/test_local_quant_publish.py`.
- Modify `tests/test_daily_report_source.py`.
- Modify `tests/test_quant_snapshot_repository.py`.
- Modify `tests/test_local_quant_task.py`.

### B4: Reports and public presentation

- Modify `stock_papi/batch/observation_products.py` — price-only calculations and separate `trading_status_observations`.
- Modify `reporting/observation_v2.py` — bind the status list into observation metadata.
- Modify `reporting/professional_builder.py` — carry evidence-bound statuses without technical classification.
- Modify `stock_papi/services/observation_view.py` — suppress current price/volume/technical values for a status artifact.
- Modify `stock_papi/services/report_view.py` — validate and expose the status list.
- Modify `templates/report_observation.html` — render verified status labels separately.
- Modify `templates/stock_detail.html` — render status-first stock view and dated last regular close only.
- Modify `tests/test_observation_products.py`.
- Modify `tests/test_professional_report_builder.py`.
- Modify `tests/test_observation_public_surfaces.py`.

---

## B1: Preserve official raw-row and lifecycle evidence

### Interfaces

`tw_trading_status.py` produces these exact public contracts:

```python
LIFECYCLE_PARSER_VERSION = "tw-lifecycle-parser-v1"
STATUS_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class PriceRowClassification:
    price: dict[str, Any] | None
    status: dict[str, Any] | None

def classify_price_row(
    target_date: datetime.date,
    source_id: str,
    exchange: str,
    fields: Sequence[str],
    raw_row: Sequence[Any],
    indices: Mapping[str, int],
    payload_sha256: str,
) -> PriceRowClassification: ...

def load_lifecycle_statuses(
    root: Path,
    target_date: datetime.date,
    *,
    session: Any,
    source_definitions: Mapping[str, OfficialSourceDefinition],
) -> Mapping[str, Mapping[str, Any]]: ...

def load_lifecycle_index(root: Path, target_date: datetime.date) -> Mapping[str, Any]: ...

def resolve_lifecycle_status(
    events: Sequence[Mapping[str, Any]],
    target_date: datetime.date,
    *,
    active: bool,
) -> Mapping[str, Any] | None: ...

def evidence_sha256(document: Mapping[str, Any]) -> str: ...
```

`build_historical_daily_snapshot()` and `build_official_snapshot_series()` add `required_symbols: Collection[str] = ()`. The CLI passes the audited universe. A cache-v1 price source is reusable only when it contains every required symbol for that snapshot; otherwise the new path requires raw cache-v2 evidence.

`OfficialDailySnapshot` adds:

```python
trading_status_by_symbol: Mapping[str, Mapping[str, Any]]
```

The snapshot manifest hash covers source result hashes, lifecycle index `events_sha256`, and sorted `(symbol, evidence_sha256)` pairs.

### RED tests

- [ ] Add the following exact tests to `tests/test_tw_trading_status.py`:

```python
def test_all_official_empty_ohlcv_becomes_no_regular_trade_not_price():
    result = classify_price_row(TARGET, "tpex_price", "TPEx", FIELDS, BLANK_ROW, INDICES, "a" * 64)
    assert result.price is None
    assert result.status["status"] == "official_no_regular_trade"
    assert result.status["target_market_date"] == TARGET.isoformat()
    assert result.status["payload_sha256"] == "a" * 64
    assert len(result.status["raw_row_sha256"]) == 64

def test_partial_empty_ohlcv_is_rejected():
    with pytest_raises(ValueError, "partial official price row"):
        classify_price_row(TARGET, "tpex_price", "TPEx", FIELDS, PARTIAL_ROW, INDICES, "a" * 64)

def test_suspend_interval_is_closed_by_resume_or_termination():
    self.assertEqual(
        resolve_lifecycle_status(SUSPEND_EVENTS, TARGET, active=True)["status"],
        "officially_suspended",
    )
    self.assertIsNone(resolve_lifecycle_status(RESUMED_EVENTS, TARGET, active=True))
    with self.assertRaisesRegex(ValueError, "terminated symbol remains active"):
        resolve_lifecycle_status(TERMINATED_EVENTS, TARGET, active=True)

def test_lifecycle_cache_rejects_missing_expired_conflicting_and_hash_mismatched_sources():
    for mutation in (missing_source, expired_index, conflicting_events, changed_payload):
        with self.subTest(mutation=mutation.__name__), self.assertRaises(OfficialCacheError):
            load_lifecycle_index(mutation(valid_index), TARGET)
```

The fixture helpers write complete index and payload bytes below a temporary root; `load_lifecycle_index()` receives that root and the target date. No new test dependency is introduced.

- [ ] Add to `tests/test_tw_official_historical.py`:

  - `test_tpex_blank_row_is_preserved_as_hash_bound_status`
  - `test_twse_blank_row_is_preserved_as_hash_bound_status`
  - `test_valid_zero_volume_ohlc_remains_regular_price`
  - `test_price_and_status_maps_are_disjoint`
  - `test_valid_price_conflicting_with_suspension_fails_snapshot`
  - `test_daily_snapshot_manifest_changes_when_status_evidence_changes`
  - `test_v1_warm_cache_with_complete_required_universe_remains_readable`
  - `test_v1_warm_cache_cannot_classify_missing_active_price`
  - `test_v2_warm_cache_recomputes_payload_and_row_hashes_without_network`

- [ ] Extend `tests/test_tw_official_bulk.py::test_cache_round_trip_is_hash_verified_and_secret_free` to assert the v2 payload object is content-addressed, the exact raw payload hash is recomputed, query parameters are absent from metadata, and a one-byte payload mutation raises `OfficialCacheError`.

### RED command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_trading_status tests.test_tw_official_historical tests.test_tw_official_bulk -v
```

Expected: failures because `tw_trading_status`, cache v2, status classification, lifecycle interval validation, and `trading_status_by_symbol` do not exist.

### RED commit boundary

```powershell
git add -- tests/test_tw_trading_status.py tests/test_tw_official_historical.py tests/test_tw_official_bulk.py
git commit -m "test: freeze TW non-price source evidence"
```

### Minimal GREEN

- [ ] Implement canonical JSON and SHA-256 with `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)` and `hashlib.sha256`; add no dependency.
- [ ] Classify a price row once in `tw_trading_status.classify_price_row()`. Return one regular price or one status. Reject partial placeholders before numeric normalization.
- [ ] Fetch the four allowlisted official lifecycle datasets with existing timeout/retry limits, verify complete pagination and exact schema fingerprints, write immutable payload objects, then atomically replace only the target-date index.
- [ ] Parse `suspend`, `resume`, and `terminate`; reject future announcements and same-date contradictions; compute the target state from the frozen precedence rules.
- [ ] Extend `store_cached_source()` and `load_cached_source()` with a separate v2 raw-payload path. Keep the current v1 functions readable for canonical price-only hits.
- [ ] Pass the audited universe as `required_symbols` when building the series. Reuse cache v1 only for a complete required price set; require cache v2 raw evidence for any missing required price.
- [ ] Build `price_by_symbol` and `trading_status_by_symbol` as disjoint `MappingProxyType` mappings. Lifecycle status wins only when independently proven; a valid price/lifecycle conflict raises `OfficialSourceFailure`.
- [ ] Advance official snapshot source schema to `tw-official-historical-v3` and bind all status hashes into daily and series manifest hashes.

The central branch is:

```python
classification = classify_price_row(...)
if classification.price is not None:
    price_rows.append(classification.price)
elif classification.status is not None:
    status_rows.append(classification.status)
else:
    raise ValueError("official price row is unclassified")
```

### GREEN command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_trading_status tests.test_tw_official_historical tests.test_tw_official_bulk -v
```

Required: zero failures/errors and all existing request-count/cache tests remain green.

### GREEN commit boundary

```powershell
git add -- stock_papi/integrations/market_data/tw_trading_status.py stock_papi/integrations/market_data/tw_official_bulk.py stock_papi/integrations/market_data/tw_official_historical.py stock_papi/integrations/market_data/tw_official_cache.py
git commit -m "feat: preserve official TW non-price evidence"
```

Stop for focused diff and data-contract review before B2. B1 is independently acceptable only if no symbol constant, raw-to-price conversion, networked test, or v1 cache rewrite exists.

---

## B2: Freeze artifact date semantics and terminal completeness

### Interfaces

`OfficialCompatFetcher` adds:

```python
def status_for(self, symbol: str) -> Mapping[str, Any] | None: ...
```

`build_stock_snapshot()` adds one keyword-only input:

```python
def build_stock_snapshot(
    pipeline,
    market,
    symbol,
    target_market_date=None,
    promoted_backtest=None,
    degraded_bootstrap=False,
    observation_only=False,
    *,
    trading_status_evidence=None,
): ...
```

`ArtifactDateAudit` adds immutable `observation_by_symbol` and `observation_kind_by_symbol` mappings while retaining `latest_by_symbol` as the latest regular-price date.

### RED tests

- [ ] Add to `tests/test_tw_incremental.py`:

  - `test_status_query_preserves_last_regular_daily_row_without_target_fill`
  - `test_status_query_returns_immutable_same_symbol_same_date_evidence`
  - `test_status_query_rejects_evidence_from_other_symbol_date_or_series`
  - `test_regular_price_and_status_cannot_both_exist_for_symbol_date`

The first test must assert literal equality of the full pre-run `daily` array, not only its length.

- [ ] Add to `tests/test_local_quant.py`:

```python
def test_status_artifact_keeps_price_as_of_and_sets_observation_as_of(self):
    artifact = build_stock_snapshot(..., target_market_date=TARGET, trading_status_evidence=STATUS)
    self.assertEqual(artifact["schema_version"], 2)
    self.assertEqual(artifact["as_of"], PRIOR.isoformat())
    self.assertEqual(artifact["latest_regular_price_date"], PRIOR.isoformat())
    self.assertEqual(artifact["observation_as_of"], TARGET.isoformat())
    self.assertEqual(artifact["observation_kind"], "official_no_regular_trade")
    self.assertEqual(date_of(artifact["daily"][-1]), PRIOR.isoformat())

def test_regular_artifact_still_requires_target_date_price(self):
    with self.assertRaisesRegex(ValueError, "target market date mismatch"):
        build_stock_snapshot(..., target_market_date=TARGET, trading_status_evidence=None)
```

Also add `test_status_artifact_rejects_missing_history_future_price_and_hash_mismatch`.

- [ ] Add to `tests/test_tw_official_post_close_cli.py`:

  - `test_terminal_gate_accepts_target_price_and_verified_status_partition`
  - `test_terminal_gate_rejects_unknown_missing_price_below_failure_threshold`
  - `test_terminal_gate_rejects_status_without_same_snapshot_hash_binding`
  - `test_terminal_gate_rejects_stale_observation_with_valid_stale_price_date`
  - `test_terminal_gate_rejects_status_artifact_with_target_date_daily_row`
  - `test_terminal_gate_rejects_price_status_overlap_and_partition_gap`
  - `test_terminal_gate_keeps_reconciliation_lineage_checks_for_status_artifact`

Each case uses a temporary root, real gzip artifact bytes, real `audit_artifact_dates()`, and no mocked terminal audit.

### RED command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental tests.test_tw_official_post_close_cli tests.test_local_quant -v
```

Expected: failures because artifact schema v2, separate observation dates, status pass-through, and partition-aware terminal validation do not exist.

### RED commit boundary

```powershell
git add -- tests/test_tw_incremental.py tests/test_tw_official_post_close_cli.py tests/test_local_quant.py
git commit -m "test: freeze TW status artifact and terminal semantics"
```

### Minimal GREEN

- [ ] Add `OfficialCompatFetcher.status_for()` as a read-only lookup of the target snapshot mapping. Do not place a blank price row into any `daily` frame.
- [ ] In the CLI patch wrapper, pass only the fetched symbol's verified status into `build_stock_snapshot()`.
- [ ] Pass the audited symbol universe into `build_official_snapshot_series(..., required_symbols=symbols)` so cache-v1 compatibility is decided before the symbol loop.
- [ ] In `build_stock_snapshot()`, retain the existing strict target-date check when status is absent. When status is present, require `as_of < target_market_date`, exact symbol/date/source lineage, and an existing non-empty history; set artifact schema/date/status fields without appending a row.
- [ ] Extend `write_stock_artifact()` and `_validated_artifact()` to require artifact schema v2 for status-aware TW observation runs and validate:

```python
document["as_of"] == document["latest_regular_price_date"] == latest_daily_date
document["observation_as_of"] == document["target_market_date"]
(document["observation_kind"] == "regular_price") == (document["as_of"] == document["observation_as_of"])
```

- [ ] Extend `audit_artifact_dates()` without changing the meaning of `latest_by_symbol`.
- [ ] Replace the single target-date predicate in `_assert_complete()` with a three-set partition. Verify each status evidence SHA against the exact daily snapshot and official series manifest. Keep the existing checkpoint identity, exclusion parsing, official lineage, reconciliation history, and applied-artifact SHA checks.
- [ ] Treat missing history, unknown missing price, termination-in-active-universe, or evidence mismatch as `_INCOMPLETE`; do not add them to an allowlist or exclusion file.

### GREEN command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_incremental tests.test_tw_official_post_close_cli tests.test_local_quant -v
```

Required: zero failures/errors; strict regular-price and legacy reconciliation tests remain green.

### GREEN commit boundary

```powershell
git add -- local_quant.py stock_papi/quant/tw_incremental.py stock_papi/quant/tw_artifact_audit.py stock_papi/batch/tw_official_post_close_cli.py
git commit -m "feat: enforce TW status artifact completeness"
```

Stop for focused diff and terminal-state review before B3. B2 must not publish a manifest or change upload/report code.

---

## B3: Publish and load manifest v3 without weakening v2

### Interfaces

`publish_market_snapshot()` adds explicit observation identity:

```python
def publish_market_snapshot(
    root,
    market,
    symbols,
    generated_at=None,
    failed_symbols=(),
    *,
    target_market_date=None,
): ...
```

`ReportSourceManifest` adds optional v3 fields with exact version dispatch, and `StockSnapshot` adds `observation_as_of`, `latest_regular_price_date`, `observation_kind`, and `trading_status_evidence`. Version-specific validators remain separate:

```python
def _validate_manifest_v2(document: dict[str, Any], market: str) -> None: ...
def _validate_manifest_v3(document: dict[str, Any], market: str) -> None: ...
```

### RED tests

- [ ] Add to `tests/test_local_quant_publish.py`:

  - `test_v3_manifest_partitions_regular_status_and_operational_symbols`
  - `test_v3_regular_price_coverage_uses_frozen_denominator`
  - `test_v3_status_entry_cross_binds_evidence_and_artifact_hashes`
  - `test_v3_publish_rejects_counter_partition_and_date_mismatch`
  - `test_v3_unknown_missing_price_preserves_previous_latest`
  - `test_v2_identical_rerun_and_rollback_pointer_remain_unchanged`

Assert the exact arithmetic from the spec and the absence of `market_as_of` in v3.

- [ ] Add to `tests/test_daily_report_source.py`:

  - `test_v3_loader_separates_regular_and_verified_status_artifacts`
  - `test_v3_loader_rejects_mixed_pointer_manifest_versions`
  - `test_v3_loader_rejects_status_hash_symbol_date_and_artifact_mismatch`
  - `test_v3_loader_rejects_invalid_denominator_or_partition_overlap`
  - `test_v2_price_only_loader_remains_exact_and_cannot_carry_status`

- [ ] Expand `tests/test_quant_snapshot_repository.py` with real in-memory pointer, manifest, and gzip object bytes:

  - `test_repository_accepts_hash_bound_v3_status_snapshot`
  - `test_repository_returns_none_for_v3_status_tampering`
  - `test_repository_keeps_v2_cache_key_separate_from_v3_identity`

- [ ] Add to `tests/test_local_quant_task.py`:

  - `test_uploader_validates_v3_partition_and_status_hashes_before_copy`
  - `test_uploader_uploads_v3_objects_and_manifest_before_pointer`
  - `test_uploader_rejects_unknown_or_mixed_schema_without_gcloud_copy`

The PowerShell tests inspect the script and use mocked command capture; they must not invoke `gcloud`.

### RED command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_local_quant_publish tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_local_quant_task -v
```

Expected: failures because schema v3 publishing, exact v3 validation, repository loading, and uploader preflight do not exist.

### RED commit boundary

```powershell
git add -- tests/test_local_quant_publish.py tests/test_daily_report_source.py tests/test_quant_snapshot_repository.py tests/test_local_quant_task.py
git commit -m "test: freeze TW status manifest v3 trust boundary"
```

### Minimal GREEN

- [ ] Make `publish_market_snapshot()` dispatch: existing schema v2 behavior for legacy/non-status inputs; schema v3 only when artifacts carry the new official source schema and `target_market_date`.
- [ ] Derive all counts and partitions from validated artifact content. Never accept caller-provided expected status membership.
- [ ] Store regular and status artifacts under the existing content-addressed `objects/<sha>.json.gz` path. Add `observation_kind` to every v3 symbol entry and bind status entries to evidence and artifact SHA.
- [ ] Write immutable v3 manifest, then atomically write schema-v3 latest pointer. Reuse the current identical-rerun comparison and immutable-object checks.
- [ ] In `reporting/source_loader.py`, dispatch on exact schema version before reading entries. v2 keeps every existing equality. v3 validates arithmetic, disjoint partitions, dates, entry/object sizes, compressed/uncompressed hashes, artifact schema v2, and evidence cross-bindings.
- [ ] In `quant_snapshots.py`, include manifest schema plus manifest SHA in the cache identity so a v2 cached object cannot mask a v3 pointer. Return `None` on every mismatch.
- [ ] In `upload_local_quant.ps1`, add a schema-v3 validation branch. Reject before `Invoke-GcloudCopyBatch` unless local pointer, manifest, all referenced objects, counters, dates, and hashes pass. Keep immutable objects/manifest before pointer and the existing generation-matched pointer update.
- [ ] Leave `reporting/migrate_quant_manifest.py` unchanged and cover that decision with the v2 compatibility tests.

### GREEN commands

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_local_quant_publish tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_local_quant_task -v

powershell.exe -NoProfile -Command "$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'scripts/upload_local_quant.ps1'),[ref]$tokens,[ref]$errors) > $null; if($errors.Count){$errors | ForEach-Object {$_.Message}; exit 1}"
```

Required: zero unittest failures/errors and zero PowerShell parser errors.

### GREEN commit boundary

```powershell
git add -- local_quant.py reporting/schemas.py reporting/source_loader.py stock_papi/repositories/quant_snapshots.py scripts/upload_local_quant.ps1
git commit -m "feat: publish and load TW status manifest v3"
```

Stop for focused security and data-contract review before B4. B3 must not modify report calculations or templates.

---

## B4: Suppress false price output and render verified status

### Interfaces

Observation dashboard schema v2 retains its version but adds one required list when its source manifest is v3:

```json
"trading_status_observations": [
  {
    "symbol": "numeric symbol",
    "name": "display name",
    "status": "official_no_regular_trade|officially_suspended",
    "label": "當日無正常交易|停止買賣",
    "observation_as_of": "YYYY-MM-DD",
    "latest_regular_price_date": "YYYY-MM-DD",
    "evidence_sha256": "64-hex"
  }
]
```

`build_stock_observation()` returns the same status identity and sets current-session price, change, volume, indicator, technical, and recommendation fields to `None` for non-price observations. It may return `last_regular_close` only with `latest_regular_price_date`.

### RED tests

- [ ] Add to `tests/test_observation_products.py`:

  - `test_status_symbols_are_excluded_from_market_industry_return_and_volume_math`
  - `test_status_symbols_create_evidence_bound_status_observations_only`
  - `test_status_symbols_never_create_price_move_or_technical_events`
  - `test_status_count_is_separate_from_operational_failure_count`
  - `test_status_evidence_hash_mismatch_rejects_dashboard_build`

Use one regular artifact and two status artifacts with deliberately extreme stale values; assert market and industry metrics equal the regular-only result byte-for-byte.

- [ ] Add to `tests/test_professional_report_builder.py`:

  - `test_professional_report_carries_status_without_event_classification`
  - `test_professional_report_rejects_unbound_status_observation`
  - `test_status_does_not_enter_positive_risk_or_high_anomaly_lists`

- [ ] Add to `tests/test_observation_public_surfaces.py`:

```python
def test_status_stock_page_never_labels_stale_close_as_current():
    html = render_status_stock_page(status="officially_suspended", prior_close=VALUE, prior_date=PRIOR)
    self.assertIn("停止買賣", html)
    self.assertIn(f"最後正常交易收盤（{PRIOR}）", html)
    for forbidden in ("最新收盤", "今日漲跌", "量比", "RSI", "MACD", "KD"):
        self.assertNotIn(forbidden, html)

def test_no_regular_trade_page_uses_exact_verified_label():
    html = render_status_stock_page(status="official_no_regular_trade", prior_close=VALUE, prior_date=PRIOR)
    self.assertIn("當日無正常交易", html)
```

Also add `test_report_view_rejects_status_without_evidence_hash` and `test_report_template_lists_status_separately_from_technical_events`.

### RED command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_observation_products tests.test_professional_report_builder tests.test_observation_public_surfaces -v
```

Expected: failures because status symbols still flow through latest-price calculations and the report/public contracts have no separate status list.

### RED commit boundary

```powershell
git add -- tests/test_observation_products.py tests/test_professional_report_builder.py tests/test_observation_public_surfaces.py
git commit -m "test: freeze TW status report presentation"
```

### Minimal GREEN

- [ ] Partition `LoadedReportSource.stocks` once by `observation_kind` at the start of `build_observation_dashboard()`. Pass only regular-price stocks to existing market, industry, event, and ETF functions.
- [ ] Build `trading_status_observations` from already validated artifact evidence. Map only the two frozen labels; reject unknown status instead of showing generic text.
- [ ] Include regular-price, expected-non-price, and operational-failure counts in `data_quality`; keep `observation_as_of` as the target session.
- [ ] Carry the status list through `observation_v2.py` and `professional_builder.py` as factual observations. Do not add the two statuses to `EVENT_POLICY_TABLE`.
- [ ] In `observation_view.py`, branch on `observation_kind` before reading `latest` as current. For status, expose no `price`, returns, volume, indicator, technical, risk-event, or recommendation values.
- [ ] In `report_view.py`, validate exact keys, labels, dates, symbol syntax, and 64-hex evidence SHA before returning status data.
- [ ] In `report_observation.html`, add a status section separate from `stock_events`.
- [ ] In `stock_detail.html`, use a status-first branch. The regular branch remains byte-for-byte behaviorally equivalent; the status branch shows the verified label and optionally the explicitly dated last regular close.

The required consumer guard is:

```python
if stock.observation_kind != "regular_price":
    return build_status_observation(stock)
latest = stock.latest
return build_regular_price_observation(stock, latest)
```

### GREEN command

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_observation_products tests.test_professional_report_builder tests.test_observation_public_surfaces -v
```

Required: zero failures/errors; regular-price dashboard/report snapshots remain unchanged except for the empty status list required by v3 sources.

### GREEN commit boundary

```powershell
git add -- stock_papi/batch/observation_products.py reporting/observation_v2.py reporting/professional_builder.py stock_papi/services/observation_view.py stock_papi/services/report_view.py templates/report_observation.html templates/stock_detail.html
git commit -m "feat: render verified TW non-price status"
```

Stop after B4 for whole-branch review and separately authorized full verification. Do not publish, upload, deploy, recover, or modify Production state.

---

## Focused cross-stage verification before full-suite authorization

Run only after B1-B4 are all green:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_tw_trading_status tests.test_tw_official_bulk tests.test_tw_official_historical tests.test_tw_incremental tests.test_tw_official_post_close_cli tests.test_local_quant tests.test_local_quant_publish tests.test_daily_report_source tests.test_quant_snapshot_repository tests.test_local_quant_task tests.test_observation_products tests.test_professional_report_builder tests.test_observation_public_surfaces -v

.\.venv\Scripts\python.exe -m py_compile local_quant.py stock_papi/integrations/market_data/tw_trading_status.py stock_papi/integrations/market_data/tw_official_bulk.py stock_papi/integrations/market_data/tw_official_historical.py stock_papi/integrations/market_data/tw_official_cache.py stock_papi/quant/tw_incremental.py stock_papi/quant/tw_artifact_audit.py stock_papi/batch/tw_official_post_close_cli.py reporting/schemas.py reporting/source_loader.py stock_papi/repositories/quant_snapshots.py stock_papi/batch/observation_products.py reporting/observation_v2.py reporting/professional_builder.py stock_papi/services/observation_view.py stock_papi/services/report_view.py

git diff --check 9c10b6af385306d35582eec30df1b16b6034db7f..HEAD
```

Record test count, failures, errors, skips, exit codes, and exact HEAD. A full suite, live official-source probe, warm-cache validation, Production pipeline, upload, or recovery requires separate authorization.

## Explicit non-modification list

The four implementation stages do not modify:

- `stock_papi/quant/model.py`, LightGBM, features, inference, prediction ledger, backtests, or recommendation policy;
- price numeric semantics, historical rows, or the existing verified price endpoint definitions;
- `stock_papi/quant/tw_legacy_reconciliation.py` or legacy backup objects/manifests;
- exclusion CSV schema, thresholds, operator actions, or symbol universe code;
- `reporting/migrate_quant_manifest.py` or any immutable manifest/object already written;
- GCS configuration, IAM, Cloud Run, Docker, website deployment, Scheduled Tasks, LINE messages, secrets, or credentials;
- `D:\AbsorbData` during development and tests.

No new dependency, database, service, generalized event framework, status-only artifact family, or symbol-specific configuration is introduced.

## Rollback boundary

- B1 rollback removes only new code/cache readers; v1 cache remains untouched.
- B2 rollback is safe before any schema-v2 status artifact is selected by a manifest; created local test artifacts are temporary only.
- B3 rollback first restores the last verified schema-v2 pointer through the existing generation-matched pointer workflow, then reverts code. Never point old code at v3.
- B4 rollback changes presentation only after the pointer/code compatibility gate is satisfied.
- Immutable v3 objects, lifecycle payloads, and official cache v2 payloads are retained; rollback never deletes or rewrites evidence.

## Plan self-review

- Spec coverage: lifecycle evidence, dates, snapshot/cache migration, terminal gate, manifest counts/denominators, upload/load rejection, report suppression, non-modifications, and rollback each map to one stage.
- RED/GREEN: every Production file is preceded by named failing tests and a tests-only commit.
- Type consistency: `observation_kind`, `observation_as_of`, `latest_regular_price_date`, `trading_status_evidence`, and `evidence_sha256` use the same names in B1-B4.
- Compatibility: v1 cache, v2 manifest, and v3 status paths are explicit and do not silently coerce one another.
- Scope: no implementation, Production operation, full suite, or subagent execution is authorized by this planning commit.
- Placeholder scan: the plan contains no deferred decision, symbol list, or unspecified acceptance condition.
