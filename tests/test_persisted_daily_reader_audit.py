import ast
import math
import unittest
from collections import defaultdict
from pathlib import Path

from stock_papi.quant.features import CALCULATED_COLUMNS
from tests.report_fixtures import warmup_stock_document


class PersistedDailyVisitor(ast.NodeVisitor):
    def __init__(self, relative_file):
        self.relative_file = relative_file
        self.scope = []
        self.aliases = {
            "load_incremental_artifact": "load_incremental_artifact",
            "StockSnapshot": "StockSnapshot",
        }
        self.detected = defaultdict(set)

    def _record(self, kind):
        if self.scope:
            self.detected[(self.relative_file, ".".join(self.scope))].add(kind)

    def visit_ImportFrom(self, node):
        for item in node.names:
            if item.name in {"load_incremental_artifact", "StockSnapshot"}:
                self.aliases[item.asname or item.name] = item.name
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and node.slice.value == "daily":
            self._record("subscript-daily")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == "daily":
            self._record("attribute-daily")
        if node.attr == "StockSnapshot":
            self._record("StockSnapshot-consumer")
        if (
            node.attr == "from_document"
            and (
                isinstance(node.value, ast.Name)
                and self.aliases.get(node.value.id) == "StockSnapshot"
                or isinstance(node.value, ast.Attribute)
                and node.value.attr == "StockSnapshot"
            )
        ):
            self._record("StockSnapshot.from_document")
        self.generic_visit(node)

    def visit_Call(self, node):
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "daily"
        ):
            self._record("get-daily")
        if isinstance(function, ast.Name) and self.aliases.get(function.id) == "load_incremental_artifact":
            self._record("load_incremental_artifact")
        if isinstance(function, ast.Attribute) and function.attr == "load_incremental_artifact":
            self._record("load_incremental_artifact")
        if isinstance(function, ast.Attribute) and function.attr == "StockSnapshot":
            self._record("StockSnapshot-construction")
        self.generic_visit(node)

    def visit_Name(self, node):
        if self.aliases.get(node.id) == "StockSnapshot":
            self._record("StockSnapshot-consumer")


def discover_persisted_daily_readers(repository_root):
    targets = [repository_root / "local_quant.py"]
    for directory in ("stock_papi", "reporting", "scripts"):
        targets.extend(sorted((repository_root / directory).rglob("*.py")))
    detected = defaultdict(set)
    for path in targets:
        relative = path.relative_to(repository_root).as_posix()
        visitor = PersistedDailyVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        for boundary, kinds in visitor.detected.items():
            detected[boundary].update(kinds)
    return detected


READER_CONTRACTS = {
    ("local_quant.py", "_validated_artifact"): "latest-only",
    ("reporting/industry_analytics.py", "_stock_return"): "canonical-OHLCV",
    ("reporting/industry_analytics.py", "_foreign_net_5"): "canonical-OHLCV",
    ("reporting/industry_analytics.py", "_market_snapshot"): "feature-ready-history",
    ("reporting/industry_analytics.py", "_industry_snapshot"): "feature-ready-history",
    ("reporting/industry_analytics.py", "_risk_hints"): "latest-only",
    ("reporting/industry_analytics.py", "_model_quality"): "feature-ready-history",
    ("reporting/industry_analytics.py", "build_daily_report"): "feature-ready-history",
    ("reporting/industry_analytics.py", "build_daily_report.industry_snapshots"): "feature-ready-history",
    ("reporting/industry_backtest.py", "backtest_industry"): "feature-ready-history",
    ("reporting/migrate_quant_manifest.py", "_validate_stock"): "latest-only",
    ("reporting/schemas.py", "StockSnapshot.from_document"): "latest-only",
    ("reporting/schemas.py", "StockSnapshot.latest"): "latest-only",
    ("reporting/schemas.py", "LoadedReportSource"): "latest-only",
    ("reporting/source_loader.py", "_load_manifest_source"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_return_pct"): "canonical-OHLCV",
    ("stock_papi/batch/observation_products.py", "_validate_source"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_trading_status_observations"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_market_daily_returns"): "canonical-OHLCV",
    ("stock_papi/batch/observation_products.py", "_market_observation"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_industry_observations"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_stock_events"): "latest-only",
    ("stock_papi/batch/observation_products.py", "_etf_observations"): "latest-only",
    ("stock_papi/batch/oos_diagnostics.py", "_enrich_point_in_time"): "feature-ready-history",
    ("stock_papi/batch/tw_official_post_close_cli.py", "_assert_complete"): "latest-only",
    ("stock_papi/batch/tw_official_post_close_cli.py", "_run_stage"): "canonical-OHLCV",
    ("stock_papi/batch/tw_official_post_close_cli.py", "_patched_pipeline.build_stock_snapshot_with_lineage"): "canonical-OHLCV",
    ("stock_papi/quant/tw_artifact_audit.py", "audit_artifact_dates"): "latest-only",
    ("stock_papi/quant/tw_incremental.py", "load_incremental_artifact"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "audit_artifact_dates"): "latest-only",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._load_artifact"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._daily_rows"): "canonical-OHLCV",
    ("stock_papi/quant/tw_incremental.py", "OfficialCompatFetcher._reconciliation_plan"): "canonical-OHLCV",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "LegacyArtifactBackupStore._validate_original"): "latest-only",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "LegacyArtifactBackupStore._validate_expected_result"): "latest-only",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "LegacyArtifactBackupStore.read_original_document"): "canonical-OHLCV",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "resolve_truncated_daily_history"): "canonical-OHLCV",
    ("stock_papi/quant/tw_legacy_reconciliation.py", "_recover_via_historical_artifact_sha256"): "canonical-OHLCV",
    ("stock_papi/repositories/quant_snapshots.py", "fetch_quant_snapshot"): "latest-only",
    ("stock_papi/research/pit_dataset.py", "_history_rows"): "canonical-OHLCV",
    ("stock_papi/services/observation_view.py", "build_stock_observation"): "feature-ready-history",
    ("stock_papi/services/stock_analysis.py", "snapshot_dataframe"): "feature-ready-history",
}
NON_PERSISTED_BUILDERS = {("scripts/generate_sample_daily_report.py", "build_documents")}


class PersistedDailyReaderAuditTests(unittest.TestCase):
    def test_only_mapped_production_readers_exist(self):
        root = Path(__file__).resolve().parents[1]
        detected = discover_persisted_daily_readers(root)
        self.assertEqual(set(detected) - NON_PERSISTED_BUILDERS, set(READER_CONTRACTS))
        self.assertEqual(
            set(READER_CONTRACTS.values()),
            {"canonical-OHLCV", "latest-only", "feature-ready-history"},
        )
        self.assertIn("document[\"daily\"]", (root / "scripts/generate_sample_daily_report.py").read_text(encoding="utf-8"))
        self.assertNotIn("load_incremental_artifact", (root / "scripts/generate_sample_daily_report.py").read_text(encoding="utf-8"))


class WarmupFixtureContractTests(unittest.TestCase):
    def test_warmup_fixture_has_valid_ohlcv_null_warmup_and_finite_ready_rows(self):
        document = warmup_stock_document("2330")
        self.assertFalse(document["sample_data"])
        for index, row in enumerate(document["daily"]):
            for field in ("Open", "High", "Low", "Close", "Volume"):
                self.assertTrue(math.isfinite(row[field]) and row[field] > 0)
            for field in CALCULATED_COLUMNS:
                self.assertIsNone(row[field]) if index < 20 else self.assertTrue(math.isfinite(row[field]))
        for field in (*CALCULATED_COLUMNS, "AI_P"):
            self.assertTrue(math.isfinite(document["daily"][-1][field]))


if __name__ == "__main__":
    unittest.main()
