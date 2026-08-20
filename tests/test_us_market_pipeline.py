"""Comprehensive end-to-end unit and integration tests for US market pipeline."""

import datetime
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zoneinfo

from stock_papi.batch.calendar import TradingCalendar, TradingCalendarSet
from stock_papi.integrations.market_data.us_calendar import (
    generate_us_calendar_document,
    get_us_exchange_holidays,
)
from stock_papi.integrations.market_data.us_universe import (
    get_us_symbols,
    parse_sec_us_universe,
    validate_us_ticker,
)
from stock_papi.integrations.market_data.us_market_data import (
    compute_us_technical_indicators,
    fetch_us_stock_history,
)
from local_quant import publish_market_snapshot, write_stock_artifact
from stock_papi.batch.observation_products import (
    build_observation_dashboard,
    promote_observation_candidate,
    validate_observation_dashboard,
    write_observation_candidate,
)
from stock_papi.config.capabilities import PredictionCapabilityState
from stock_papi.repositories.quant_snapshots import (
    published_quant_manifest,
    published_stock_artifact,
)
from stock_papi.repositories.report_store import load_report_index
from stock_papi.services.observation_view import build_stock_observation
from reporting.source_loader import load_report_source, _validate_manifest_v4

NEW_YORK = zoneinfo.ZoneInfo("America/New_York")
UTC = datetime.timezone.utc


class USMarketPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.target_date = datetime.date(2026, 8, 19)
        self.now = datetime.datetime(2026, 8, 20, 2, 0, tzinfo=UTC)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_mock_stock_artifact(self, symbol: str, target_date: datetime.date, as_of: datetime.date | None = None):
        as_of_date = as_of or target_date
        daily = []
        for i in range(30):
            d = (target_date - datetime.timedelta(days=35 - i)).isoformat()
            daily.append({
                "Date": d,
                "Open": 150.0 + i,
                "High": 155.0 + i,
                "Low": 145.0 + i,
                "Close": 152.0 + i,
                "Volume": 10000000,
                "MA5": 150.0 + i,
                "MA20": 148.0 + i,
                "MA60": 145.0 + i,
                "RSI": 58.5,
                "MACD": 1.2,
                "K": 60.0,
                "D": 58.0,
                "VOL_RATIO": 1.1,
            })
        latest = daily[-1]
        latest["Date"] = as_of_date.isoformat()
        payload = {
            "schema_version": 2,
            "market": "US",
            "symbol": symbol,
            "as_of": as_of_date.isoformat(),
            "target_market_date": target_date.isoformat(),
            "observation_as_of": target_date.isoformat(),
            "latest_regular_price_date": as_of_date.isoformat(),
            "observation_kind": "regular_price",
            "lineage": {
                "source_schema_version": "us-market-data-v1",
                "observation_as_of": target_date.isoformat(),
                "latest_regular_price_date": as_of_date.isoformat(),
                "observation_kind": "regular_price",
            },
            "rows": len(daily),
            "latest": latest,
            "backtest": {},
            "daily": daily,
        }
        return write_stock_artifact(self.root, "US", symbol, payload)

    def test_us_manifest_v4_publishing_and_validation(self):
        # 100 symbols universe, 98 available, 2 unavailable -> 98% coverage (>95%)
        universe = [f"SYM{i:03d}" for i in range(100)]
        available = universe[:98]
        unavailable = universe[98:]

        for sym in available:
            self._create_mock_stock_artifact(sym, self.target_date)

        pointer_path = publish_market_snapshot(
            self.root,
            "US",
            universe,
            generated_at=self.now,
            failed_symbols=unavailable,
            target_market_date=self.target_date,
            unavailable_symbols=unavailable,
        )
        self.assertTrue(pointer_path.exists())

        pointer_doc = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = self.root / "publish" / "quant" / "v1" / pointer_doc["manifest"]
        self.assertTrue(manifest_path.exists())

        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_doc["schema_version"], 4)
        self.assertEqual(manifest_doc["market"], "US")
        self.assertEqual(manifest_doc["active_universe_count"], 100)
        self.assertEqual(manifest_doc["observation_count"], 98)
        self.assertEqual(manifest_doc["regular_price_symbol_count"], 98)
        self.assertEqual(manifest_doc["operational_failure_count"], 0)
        self.assertEqual(manifest_doc["unavailable_count"], 2)
        self.assertAlmostEqual(manifest_doc["observation_coverage"], 0.98)

        # Validate with source_loader
        _validate_manifest_v4(manifest_doc, "US")

    def test_us_manifest_fail_closed_at_95_percent(self):
        # 100 symbols universe, 95 available -> 95.0% coverage -> strictly FAIL CLOSED (must be >95%)
        universe = [f"SYM{i:03d}" for i in range(100)]
        available = universe[:95]
        unavailable = universe[95:]

        for sym in available:
            self._create_mock_stock_artifact(sym, self.target_date)

        with self.assertRaises(RuntimeError) as ctx:
            publish_market_snapshot(
                self.root,
                "US",
                universe,
                generated_at=self.now,
                failed_symbols=unavailable,
                target_market_date=self.target_date,
                unavailable_symbols=unavailable,
            )
        self.assertIn("not publishable", str(ctx.exception))

    def test_us_observation_dashboard_and_candidate_promotion(self):
        universe = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
        for sym in universe:
            self._create_mock_stock_artifact(sym, self.target_date)

        manifest_path = publish_market_snapshot(
            self.root,
            "US",
            universe,
            generated_at=self.now,
            target_market_date=self.target_date,
        )

        source = load_report_source(self.root, market="US")
        industry_map = {"ETF專區": ["SPY", "QQQ"]}
        pred_cap = PredictionCapabilityState.from_environment()
        dashboard = build_observation_dashboard(
            source, industry_map, pred_cap, generated_at=self.now, today=self.target_date
        )
        self.assertEqual(dashboard["schema_version"], 2)
        self.assertEqual(dashboard["market"], "US")
        self.assertEqual(dashboard["product_mode"], "observation")

        report_meta = {
            "schema_version": 2,
            "kind": "absorb-report",
            "product_mode": "observation",
            "market": "US",
            "report_type": "post_close",
            "source_market_date": self.target_date.isoformat(),
            "applicable_trading_date": self.target_date.isoformat(),
            "published_at": self.now.isoformat().replace("+00:00", "Z"),
            "data_as_of": self.target_date.isoformat(),
            "forecast_start_date": self.target_date.isoformat(),
            "forecast_end_date": self.target_date.isoformat(),
            "observation_start_date": self.target_date.isoformat(),
            "observation_end_date": self.target_date.isoformat(),
            "source_manifest": f"quant/v1/{source.manifest.manifest_path}",
            "source_manifest_sha256": source.manifest.manifest_sha256,
            "model_versions": {},
            "title": f"ABSORB 美股盤後市場觀察報告 ({self.target_date})",
            "summary": [f"美股 {self.target_date} 交易日收盤觀察與市場結構概況。"],
            "warnings": [],
            "content": {
                "market_observation": dashboard["market_observation"],
                "industry_observations": dashboard["industry_observations"],
                "data_quality": dashboard["data_quality"],
                "stock_events": dashboard["stock_events"],
                "trading_status_observations": dashboard.get("trading_status_observations", []),
                "etf_observations": dashboard["etf_observations"],
                "daily_focus": dashboard["daily_focus"],
            },
            "content_sha256": source.manifest.manifest_sha256,
            "prediction_capability": pred_cap.to_document(),
        }

        cand_dir = write_observation_candidate(self.root, report_meta, dashboard)
        promoted = promote_observation_candidate(self.root, cand_dir)
        self.assertIn("dashboard_latest", promoted)
        self.assertIn("report_latest", promoted)

        # Verify local published artifacts
        dash_ptr = self.root / "publish" / "dashboard" / "v1" / "latest-US.json"
        rep_ptr = self.root / "publish" / "reports" / "v2" / "latest-US-post_close.json"
        idx_ptr = self.root / "publish" / "reports" / "v2" / "index-US.json"
        self.assertTrue(dash_ptr.exists())
        self.assertTrue(rep_ptr.exists())
        self.assertTrue(idx_ptr.exists())

        # Test index loading
        def mock_load_object(path, max_bytes):
            p = self.root / "publish" / path
            return p.read_bytes() if p.exists() else None

        index = load_report_index(load_object=mock_load_object, max_bytes=1000000, version="v2", market="US")
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0]["report_type"], "post_close")
        self.assertEqual(index[0]["source_market_date"], self.target_date.isoformat())

    def test_stock_observation_view_for_us_ticker(self):
        self._create_mock_stock_artifact("AAPL", self.target_date)
        snapshot_path = self.root / "artifacts" / "stocks" / "US" / "AAPL.json.gz"
        with gzip.GzipFile(snapshot_path, "rb") as f:
            snapshot = json.loads(f.read().decode("utf-8"))

        obs = build_stock_observation(snapshot)
        self.assertIsNotNone(obs)
        self.assertEqual(obs["price"], 181.0)  # Close of last row (152 + 29)
        self.assertEqual(obs["trend_observation"], "above_ma20_ma60")
        self.assertEqual(obs["prediction_status"], "AI 預測研究中")
        self.assertEqual(obs["quant_source"], "已驗證本地快照")

    def test_us_web_routes(self):
        import app as stock_app
        client = stock_app.app.test_client()

        # 1. /reports/us when empty
        resp = client.get("/reports/us")
        self.assertIn(resp.status_code, (200, 503))

        # 2. /reports/us/<date>/post-close invalid date -> 404
        resp = client.get("/reports/us/invalid-date/post-close")
        self.assertEqual(resp.status_code, 404)

        # 3. /stock/<us_code> for valid US ticker
        resp = client.get("/stock/AAPL")
        self.assertIn(resp.status_code, (200, 404))

        # 4. /stock/<invalid_code> -> 404
        resp = client.get("/stock/INVALID_999_TICKER")
        self.assertEqual(resp.status_code, 404)
