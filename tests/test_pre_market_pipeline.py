import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reporting.schemas import ReportMetadataV2
from stock_papi.batch.pre_market import PreMarketPipeline, PreMarketPipelineError
from stock_papi.integrations.market_data.overnight import (
    OvernightSourceError,
    OvernightSourceSpec,
    fetch_overnight_source,
)
from tests.test_batch_calendar import calendar_document


UTC = datetime.timezone.utc


def base_receipt():
    metadata = {
        "schema_version": 2,
        "kind": "absorb-report",
        "product_mode": "observation",
        "report_type": "post_close",
        "market": "TW",
        "source_market_date": "2026-07-14",
        "applicable_trading_date": "2026-07-15",
        "published_at": "2026-07-14T10:00:00Z",
        "forecast_start_date": "2026-07-15",
        "forecast_end_date": "2026-07-15",
        "observation_start_date": "2026-07-14",
        "observation_end_date": "2026-07-15",
        "backtest_as_of": None,
        "data_as_of": "2026-07-14",
        "source_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
        "source_manifest_sha256": "a" * 64,
        "model_versions": {},
        "prediction_capability": {
            "mode": "research",
            "observation_enabled": True,
            "probability_allowed": False,
            "ranking_allowed": False,
            "strong_action_allowed": False,
            "performance_endorsement_allowed": False,
        },
        "title": "盤後市場觀察",
        "summary": ["市場風險狀態為中性"],
        "warnings": [],
        "content": {
            "market_observation": {"risk_state": "normal"},
            "daily_focus": ["市場風險狀態為中性"],
        },
    }
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"metadata": metadata, "metadata_sha256": hashlib.sha256(encoded).hexdigest()}


def overnight(name, signal="risk_off", as_of="2026-07-14T23:30:00Z"):
    return {
        "source": name,
        "as_of": as_of,
        "signal": signal,
        "summary": f"{name} 隔夜變化",
        "attribution_url": "https://example.com/market-data",
    }


def _us_source():
    from reporting.schemas import StockSnapshot
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        market="US", market_as_of=datetime.date(2026, 7, 14),
        manifest_path="manifests/US-20260714T220000Z-aaaaaaaaaaaa.json",
        manifest_sha256="a" * 64,
    )
    stocks = []
    for index, symbol in enumerate(("SPY", "QQQ", "TSM", "UMC", "ASX")):
        stocks.append(StockSnapshot(
            symbol=symbol, name=symbol, market="US", as_of=manifest.market_as_of,
            model_version="test", daily=[
                {"Date": "2026-07-13T00:00:00", "Close": 100.0},
                {"Date": "2026-07-14T00:00:00", "Close": 101.0 if index < 4 else 99.0},
            ], backtest={}, sha256="b" * 64, size=1,
        ))
    return SimpleNamespace(manifest=manifest, stocks=stocks)


def _us_calendars():
    from stock_papi.batch.calendar import TradingCalendarSet
    from stock_papi.integrations.market_data.us_calendar import (
        get_us_calendar_documents,
    )

    return TradingCalendarSet.from_documents(get_us_calendar_documents(2026, 2026))


class PreMarketPipelineTests(unittest.TestCase):
    def test_overnight_fetch_enforces_timeout_size_schema_timestamp_and_freshness(self):
        spec = OvernightSourceSpec(
            name="US futures",
            url="https://example.com/futures",
            timeout_seconds=3,
            max_bytes=1024,
            max_age=datetime.timedelta(hours=12),
        )
        calls = []
        result = fetch_overnight_source(
            spec,
            fetch_bytes=lambda url, timeout, max_bytes: calls.append((url, timeout, max_bytes))
            or json.dumps(overnight("US futures")).encode("utf-8"),
            now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC),
        )
        self.assertEqual(calls, [(spec.url, 3, 1024)])
        self.assertEqual(result["signal"], "risk_off")
        with self.assertRaises(OvernightSourceError):
            fetch_overnight_source(
                spec,
                fetch_bytes=lambda *_: b"x" * 1025,
                now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC),
            )
        with self.assertRaises(OvernightSourceError):
            fetch_overnight_source(
                spec,
                fetch_bytes=lambda *_: json.dumps(overnight("US futures", as_of="2026-07-13T00:00:00Z")).encode(),
                now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC),
            )

    def test_missing_or_invalid_base_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            prediction_base = base_receipt()
            prediction_base["metadata"].pop("product_mode")
            prediction_base["metadata_sha256"] = hashlib.sha256(
                json.dumps(
                    prediction_base["metadata"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for value in (
                None,
                {
                    "metadata": {"report_type": "pre_market"},
                    "metadata_sha256": "a" * 64,
                },
                prediction_base,
            ):
                with self.subTest(value=value), self.assertRaises(PreMarketPipelineError):
                    PreMarketPipeline(
                        Path(temporary),
                        applicable_trading_date=datetime.date(2026, 7, 15),
                        load_base=lambda value=value: value,
                        source_loaders=[],
                        publish=lambda _metadata: {},
                        notify=lambda _receipt: {},
                    ).run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))

    def test_publishes_reduced_core_with_overnight_overlay_without_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_receipt()
            published = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=lambda: base,
                source_loaders=[],
                us_source_loader=_us_source,
                us_calendars=_us_calendars(),
                publish=lambda metadata: published.append(metadata) or {"content_sha256": "b" * 64},
                notify=lambda _receipt: {"sent": True},
            )

            result = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))

            document = published[0]
            parsed = ReportMetadataV2.from_document(document)
            core = document["content"]["core"]
            self.assertEqual(set(core), {"market_observation", "data_quality", "daily_focus"})
            self.assertEqual(core["market_observation"], base["metadata"]["content"]["market_observation"])
            self.assertEqual(parsed.product_mode, "observation")
            self.assertEqual(document["content"]["overnight_overlay"]["status"], "risk_on")
            self.assertEqual(
                [item["symbol"] for item in document["content"]["overnight_overlay"]["symbols"]],
                list(("SPY", "QQQ", "TSM", "UMC", "ASX")),
            )
            self.assertNotIn("unavailable", document["content"]["overnight_overlay"])
            self.assertEqual(document["product_mode"], "observation")
            self.assertEqual(document["model_versions"], {})
            self.assertIsNone(document["backtest_as_of"])
            self.assertEqual(
                document["prediction_capability"],
                base["metadata"]["prediction_capability"],
            )
            self.assertEqual(document["title"], "ABSORB 盤前風險更新")
            self.assertNotIn("pdf_path", document)
            self.assertEqual(result["status"], "completed")

    def test_missing_us_source_fails_closed_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt,
                source_loaders=[],
                publish=lambda metadata: calls.append("publish") or {"content_sha256": "b" * 64},
                notify=lambda receipt: calls.append("notify") or {"sent": True},
            )
            with self.assertRaises(PreMarketPipelineError):
                pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
            self.assertEqual(calls, [])

    def test_completed_rerun_does_not_duplicate_notification(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt,
                source_loaders=[],
                us_source_loader=_us_source,
                us_calendars=_us_calendars(),
                publish=lambda metadata: calls.append("publish") or {"content_sha256": "b" * 64},
                notify=lambda receipt: calls.append("notify") or {"sent": True},
            )
            first = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
            second = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, 5, tzinfo=UTC))

            overlay = first["outputs"]["metadata"]["content"]["overnight_overlay"]
            self.assertEqual(overlay["status"], "risk_on")
            self.assertEqual(calls, ["publish", "notify"])
            self.assertEqual(second["status"], "completed")

    def test_pre_market_core_only_carries_benchmark_summary_not_full_industry_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_receipt()
            published = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=lambda: base,
                source_loaders=[],
                us_source_loader=_us_source,
                us_calendars=_us_calendars(),
                publish=lambda metadata: published.append(metadata) or {"content_sha256": "b" * 64},
                notify=lambda _receipt: {"sent": True},
            )
            result = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
            self.assertEqual(result["status"], "completed")
            core_after = published[0]["content"]["core"]
            self.assertNotIn("industry_observations", core_after)
            self.assertNotIn("stock_events", core_after)
            self.assertNotIn("etf_observations", core_after)
            self.assertNotIn("heatmap", core_after)


ROOT = Path(__file__).resolve().parents[1]


class PreMarketCliFreshnessTests(unittest.TestCase):
    """PreMarket 必須拒絕以舊盤後 base 發布「今天的盤前」。"""

    @staticmethod
    def _publish_post_close_base(root, applicable="2026-07-16"):
        from reporting.observation_v2 import build_post_close_observation_metadata
        from reporting.publisher import publish_report_v2
        from tests.test_observation_public_surfaces import observation_dashboard

        class Calendar:
            def next_session(self, value):
                return datetime.date.fromisoformat(applicable)

        metadata = build_post_close_observation_metadata(
            observation_dashboard(), Calendar()
        )
        publish_report_v2(Path(root), metadata)
        return metadata

    @staticmethod
    def _calendar_artifact(root, closed=()):
        document = calendar_document(2026, closed=closed)
        path = Path(root) / "TW-2026.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def _us_completed_session():
        import zoneinfo

        from stock_papi.batch.calendar import TradingCalendarSet
        from stock_papi.integrations.market_data.us_calendar import (
            get_us_calendar_documents,
        )

        new_york = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.datetime.now(UTC)
        ny_now = now.astimezone(new_york)
        calendars = TradingCalendarSet.from_documents(
            get_us_calendar_documents(2025, 2027)
        )
        completed = calendars.latest_session_on_or_before(
            ny_now.date()
            if ny_now.time() >= datetime.time(16)
            else ny_now.date() - datetime.timedelta(days=1)
        )
        previous = calendars.session_offset(completed, -1)
        return completed, previous

    @staticmethod
    def _publish_us_quant(root, completed, previous):
        from local_quant import publish_market_snapshot, write_stock_artifact

        universe = list(("SPY", "QQQ", "TSM", "UMC", "ASX"))
        for symbol in universe:
            daily = [
                {"Date": previous.isoformat(), "Close": 100.0},
                {"Date": completed.isoformat(), "Close": 101.0},
            ]
            payload = {
                "schema_version": 2,
                "market": "US",
                "symbol": symbol,
                "name": symbol,
                "as_of": completed.isoformat(),
                "target_market_date": completed.isoformat(),
                "observation_as_of": completed.isoformat(),
                "latest_regular_price_date": completed.isoformat(),
                "observation_kind": "regular_price",
                "model_version": "test",
                "lineage": {
                    "source_schema_version": "us-market-data-v1",
                    "observation_as_of": completed.isoformat(),
                    "latest_regular_price_date": completed.isoformat(),
                    "observation_kind": "regular_price",
                },
                "rows": len(daily),
                "latest": dict(daily[-1]),
                "backtest": {},
                "daily": daily,
            }
            write_stock_artifact(root, "US", symbol, payload)
        publish_market_snapshot(
            Path(root),
            "US",
            universe,
            generated_at=datetime.datetime.now(UTC),
            target_market_date=completed,
        )

    def _run_cli(self, root, applicable, calendars, extra=None):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / ".deps")]
        )
        command = [
            sys.executable,
            "-m",
            "stock_papi.batch.cli",
            "pre-market",
            "--root",
            str(root),
            "--applicable-trading-date",
            applicable,
        ]
        for calendar in calendars:
            command += ["--calendar-artifact", str(calendar)]
        command += list(extra or [])
        return subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

    def test_stale_post_close_base_blocks_pre_market_with_machine_readable_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._publish_post_close_base(root, applicable="2026-07-16")
            calendar = self._calendar_artifact(
                temporary, closed=("2026-07-15",)
            )

            completed = self._run_cli(
                root, "2026-07-16", [calendar]
            )

        self.assertEqual(completed.returncode, 4)
        self.assertIn(
            "stale_post_close_base", completed.stderr
        )

    def test_fresh_post_close_base_allows_pre_market_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._publish_post_close_base(root, applicable="2026-07-16")
            completed_session, previous_session = self._us_completed_session()
            self._publish_us_quant(root, completed_session, previous_session)
            calendar = self._calendar_artifact(temporary)

            completed = self._run_cli(
                root, "2026-07-16", [calendar]
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["status"], "completed")

    def test_source_file_is_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            completed = self._run_cli(
                root, "2026-07-16", [], extra=["--source-file", "legacy.json"]
            )

        self.assertEqual(completed.returncode, 4)
        self.assertIn(
            "TW pre-market only accepts verified US quant source",
            completed.stderr,
        )

    def test_missing_calendar_artifacts_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._publish_post_close_base(root, applicable="2026-07-16")

            completed = self._run_cli(
                root, "2026-07-16", [], extra=["--calendar-artifact", "missing.json"]
            )

        self.assertNotEqual(completed.returncode, 0)

if __name__ == "__main__":
    unittest.main()
