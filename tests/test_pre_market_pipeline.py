import copy
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

    def test_partial_sources_keep_core_bytes_unchanged_and_publish_without_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_receipt()
            before = json.dumps(base["metadata"]["content"], sort_keys=True, separators=(",", ":")).encode()
            published = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=lambda: base,
                source_loaders=[
                    lambda: overnight("US futures", "risk_off"),
                    lambda: (_ for _ in ()).throw(TimeoutError("provider timeout")),
                ],
                publish=lambda metadata: published.append(metadata) or {"content_sha256": "b" * 64},
                notify=lambda _receipt: {"sent": True},
            )

            result = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))

            document = published[0]
            parsed = ReportMetadataV2.from_document(document)
            after = json.dumps(document["content"]["core"], sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(before, after)
            self.assertEqual(parsed.product_mode, "observation")
            self.assertEqual(document["content"]["overnight_overlay"]["status"], "risk_off")
            self.assertEqual(len(document["content"]["overnight_overlay"]["unavailable"]), 1)
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

    def test_all_unavailable_is_insufficient_and_rerun_does_not_duplicate_notification(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=base_receipt,
                source_loaders=[lambda: None],
                publish=lambda metadata: calls.append("publish") or {"content_sha256": "b" * 64},
                notify=lambda receipt: calls.append("notify") or {"sent": True},
            )
            first = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
            second = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, 5, tzinfo=UTC))

            overlay = first["outputs"]["metadata"]["content"]["overnight_overlay"]
            self.assertEqual(overlay["status"], "insufficient")
            self.assertEqual(overlay["message"], "資料不足，維持盤後觀察")
            self.assertEqual(calls, ["publish", "notify"])
            self.assertEqual(second["status"], "completed")



    def test_pre_market_core_lineage_preserves_raw_observation_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = base_receipt()
            raw_core_before = copy.deepcopy(base["metadata"]["content"])
            published = []
            pipeline = PreMarketPipeline(
                Path(temporary),
                applicable_trading_date=datetime.date(2026, 7, 15),
                load_base=lambda: base,
                source_loaders=[lambda: overnight("US futures", "risk_on")],
                publish=lambda metadata: published.append(metadata) or {"content_sha256": "b" * 64},
                notify=lambda _receipt: {"sent": True},
            )
            result = pipeline.run(now=datetime.datetime(2026, 7, 15, 0, tzinfo=UTC))
            self.assertEqual(result["status"], "completed")
            core_after = published[0]["content"]["core"]
            self.assertEqual(raw_core_before, core_after)


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
            calendar = self._calendar_artifact(temporary)

            completed = self._run_cli(
                root, "2026-07-16", [calendar]
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["status"], "completed")

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
