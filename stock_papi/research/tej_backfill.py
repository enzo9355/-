"""Quota-aware historical acquisition and incremental research dataset builder for TEJ."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from .tej import (
    TEJ_PROVIDER,
    TEJ_SCHEMA_VERSION,
    TejClient,
    TejError,
    _canonical,
    _canonical_sha,
    _write_immutable,
    build_factor_snapshot,
    normalize_pit_records,
    write_tej_raw_cache,
)
from .tej_quota import TejQuotaManager
from .tej_schema_discovery import TEJ_TABLE_SCHEMAS, ABSORB_FEATURE_MAPPINGS


# Default representative liquid securities for research dataset
DEFAULT_RESEARCH_UNIVERSE = [
    "2330",  # TSMC
    "2317",  # Hon Hai
    "2454",  # MediaTek
    "2308",  # Delta
    "2382",  # Quanta
    "2881",  # Fubon
    "2882",  # Cathay
    "2412",  # Chunghwa Telecom
    "2603",  # Evergreen
    "2303",  # UMC
    "1101",  # Taiwan Cement
    "1301",  # Formosa Plastics
    "2002",  # China Steel
    "2886",  # Mega
    "2891",  # CTBC
]

# Prioritized table list for research value
RESEARCH_TABLE_PRIORITY = [
    "TRAIL/TASALE",
    "TRAIL/TAIM1AA",
    "TRAIL/TAIM1AQ",
    "TRAIL/TAIM1A",
    "TRAIL/TATINST1",
    "TRAIL/TAQFII",
    "TRAIL/TAMT",
    "TRAIL/TAPRCD",
]


class TejBackfillEngine:
    """Acquires and accumulates immutable research datasets within daily trial quota."""

    def __init__(
        self,
        root: Path | str,
        *,
        quota_manager: TejQuotaManager | None = None,
        client: TejClient | None = None,
        universe: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.root = Path(root).resolve()
        self.client = client or TejClient.from_env()
        self.quota_mgr = quota_manager or TejQuotaManager(self.root)
        self.universe = list(universe or DEFAULT_RESEARCH_UNIVERSE)
        self.logger = logger or logging.getLogger(__name__)

        self.research_dir = self.root / "research" / "tej" / "v1"
        self.state_file = self.research_dir / "backfill_state.json"
        self.coverage_file = self.research_dir / "coverage_dashboard.json"
        self.state: dict[str, Any] = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning("Failed to load backfill state: %s", exc)
        return {
            "schema_version": TEJ_SCHEMA_VERSION,
            "kind": "absorb-tej-backfill-state",
            "watermarks": {},
            "accumulated_rows": 0,
            "accumulated_requests": 0,
            "tables": {},
            "last_run": None,
        }

    def _save_state(self) -> None:
        self.research_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.state, ensure_ascii=False, indent=2)
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(self.state_file)

    def run_slice(
        self,
        *,
        max_symbols_per_table: int = 5,
        max_rows_per_fetch: int = 5000,
    ) -> dict[str, Any]:
        """Execute one bounded, quota-safe backfill slice across prioritized tables."""
        if not self.client.enabled:
            return {"status": "disabled", "message": "TEJ is disabled"}

        # Sync quota state
        quota_state = self.quota_mgr.load_or_sync_state(self.client)
        if quota_state.get("is_exhausted"):
            self.logger.warning("Quota budget is exhausted for today (%s)", quota_state.get("date"))
            return {
                "status": "quota_budget_exhausted",
                "quota": self.quota_mgr.summary(),
                "accumulated_rows": self.state.get("accumulated_rows", 0),
            }

        # Check discovery
        discovery = self.client.discover()
        if discovery.get("status") != "authentication_valid":
            return {
                "status": discovery.get("status", "error"),
                "reason": discovery.get("reason"),
            }

        entitled_tables = set(discovery.get("entitled_datasets", []))
        total_rows_fetched = 0
        total_requests_made = 0
        slice_reports = []

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

        for table in RESEARCH_TABLE_PRIORITY:
            if table not in entitled_tables:
                continue

            table_meta = TEJ_TABLE_SCHEMAS.get(table, {})
            table_state = self.state.setdefault("tables", {}).setdefault(table, {
                "symbols_completed": [],
                "total_rows": 0,
                "earliest_mdate": None,
                "latest_mdate": None,
            })

            completed_symbols = set(table_state.get("symbols_completed", []))
            remaining_symbols = [s for s in self.universe if s not in completed_symbols]

            if not remaining_symbols:
                continue

            symbols_to_fetch = remaining_symbols[:max_symbols_per_table]

            for symbol in symbols_to_fetch:
                # Check quota budget before each fetch
                if not self.quota_mgr.can_consume(estimated_requests=1, estimated_rows=500):
                    self.logger.info("Stopping backfill slice: daily quota safety budget reached")
                    break

                try:
                    # Fetch from TEJ API
                    fetch_res = self.client.fetch_dataset(table, filters={"coid": symbol})
                    total_requests_made += 1

                    if fetch_res.get("status") != "dataset_entitled":
                        self.logger.warning("Fetch failed for %s (%s): %s", table, symbol, fetch_res)
                        continue

                    records = fetch_res.get("records", [])
                    row_count = len(records)
                    total_rows_fetched += row_count

                    self.quota_mgr.record_consumption(1, row_count, table=table)

                    if row_count > 0:
                        # Write immutable raw cache
                        raw_meta = {
                            "provider": TEJ_PROVIDER,
                            "dataset": table,
                            "query": {"coid": symbol},
                            "requested_at": now_str,
                            "retrieved_at": now_str,
                            "effective_date": now_str[:10],
                            "available_at": now_str,
                            "entity_field": "coid",
                            "row_count": row_count,
                            "schema_version": TEJ_SCHEMA_VERSION,
                            "client_version": "absorb-tej-backfill-v1",
                        }
                        cached = write_tej_raw_cache(self.root, records, raw_meta)

                        # Update table state
                        table_state["total_rows"] = table_state.get("total_rows", 0) + row_count
                        dates = [r.get("mdate") for r in records if r.get("mdate")]
                        if dates:
                            earliest = min(dates)
                            latest = max(dates)
                            if not table_state.get("earliest_mdate") or earliest < table_state["earliest_mdate"]:
                                table_state["earliest_mdate"] = str(earliest)
                            if not table_state.get("latest_mdate") or latest > table_state["latest_mdate"]:
                                table_state["latest_mdate"] = str(latest)

                    table_state.setdefault("symbols_completed", []).append(symbol)
                    slice_reports.append({
                        "table": table,
                        "symbol": symbol,
                        "row_count": row_count,
                    })

                except Exception as exc:
                    self.logger.warning("Error fetching %s for %s: %s", table, symbol, exc)
                    continue

                if not self.quota_mgr.can_consume(estimated_requests=1, estimated_rows=200):
                    break

            if not self.quota_mgr.can_consume(estimated_requests=1, estimated_rows=200):
                break

        self.state["accumulated_rows"] = self.state.get("accumulated_rows", 0) + total_rows_fetched
        self.state["accumulated_requests"] = self.state.get("accumulated_requests", 0) + total_requests_made
        self.state["last_run"] = now_str
        self._save_state()

        # Update coverage dashboard
        coverage = self.build_coverage_dashboard()

        return {
            "status": "slice_completed",
            "rows_fetched_this_slice": total_rows_fetched,
            "requests_made_this_slice": total_requests_made,
            "total_accumulated_rows": self.state["accumulated_rows"],
            "quota_summary": self.quota_mgr.summary(),
            "slice_details": slice_reports,
            "coverage": coverage,
        }

    def build_coverage_dashboard(self) -> dict[str, Any]:
        """Generate and persist machine-readable research coverage dashboard."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        quota = self.quota_mgr.summary()

        table_summaries = {}
        for table, meta in self.state.get("tables", {}).items():
            schema = TEJ_TABLE_SCHEMAS.get(table, {})
            table_summaries[table] = {
                "cname": schema.get("cname", ""),
                "category": schema.get("category", ""),
                "pit_safe": schema.get("pit_safe", False),
                "symbols_covered": len(meta.get("symbols_completed", [])),
                "symbols": meta.get("symbols_completed", []),
                "total_rows": meta.get("total_rows", 0),
                "earliest_period": meta.get("earliest_mdate"),
                "latest_period": meta.get("latest_mdate"),
            }

        # Check challenger eligibility
        # Requirements: At least 3 liquid symbols, at least 12 months history in TASALE, PIT safe
        tasale_cov = table_summaries.get("TRAIL/TASALE", {})
        tasale_symbols = tasale_cov.get("symbols_covered", 0)
        has_min_history = (
            tasale_cov.get("earliest_period") is not None
            and tasale_cov.get("latest_period") is not None
        )

        # Full multi-year cross-sectional OOS walk-forward folds requires multi-year data
        challenger_eligible = False
        eligibility_reason = "Trial dataset provides sample history for research enrichment; full multi-year cross-sectional walk-forward folds require multi-year commercial history."

        dashboard = {
            "schema_version": TEJ_SCHEMA_VERSION,
            "kind": "absorb-tej-coverage-dashboard",
            "generated_at": now_str,
            "trial_validity": {
                "startDate": quota.get("startDate"),
                "endDate": quota.get("endDate"),
            },
            "quota_status": quota,
            "accumulated_research_dataset": {
                "total_rows": self.state.get("accumulated_rows", 0),
                "total_requests": self.state.get("accumulated_requests", 0),
                "universe_target_size": len(self.universe),
                "tables": table_summaries,
            },
            "challenger_readiness": {
                "model_name": "tej-challenger-lgbm-v1",
                "eligible": challenger_eligible,
                "reason_not_eligible": eligibility_reason,
                "production_features_changed": False,
            },
        }

        self.research_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(dashboard, ensure_ascii=False, indent=2)
        temp_file = self.coverage_file.with_suffix(".tmp")
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(self.coverage_file)
        return dashboard
