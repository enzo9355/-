import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.report_fixtures import stock_document, write_quant_publish
from tests.test_batch_calendar import calendar_document


class ObservationProductsCliTests(unittest.TestCase):
    def test_build_rejects_validation_date_that_does_not_match_source_date(self):
        from stock_papi.batch.observation_products_cli import main

        arguments = [
            "build",
            "--source-market-date",
            "2026-08-07",
            "--source-validation-date",
            "2026-08-06",
            "--source-manifest",
            "quant/v1/manifests/TW-20260807T000000Z-aaaaaaaaaaaa.json",
            "--source-manifest-sha256",
            "a" * 64,
            "--calendar-artifact",
            "calendar.json",
        ]
        with patch(
            "stock_papi.batch.observation_products_cli.build"
        ) as build, self.assertRaisesRegex(
            ValueError, "must equal source market date"
        ):
            main(arguments, today=datetime.date(2026, 8, 15))
        build.assert_not_called()

    def test_build_rejects_validation_date_from_the_future(self):
        from stock_papi.batch.observation_products_cli import main

        arguments = [
            "build",
            "--source-market-date",
            "2026-08-16",
            "--source-validation-date",
            "2026-08-16",
            "--source-manifest",
            "quant/v1/manifests/TW-20260816T000000Z-aaaaaaaaaaaa.json",
            "--source-manifest-sha256",
            "a" * 64,
            "--calendar-artifact",
            "calendar.json",
        ]
        with patch(
            "stock_papi.batch.observation_products_cli.build"
        ) as build, self.assertRaisesRegex(
            ValueError, "cannot be in the future"
        ):
            main(arguments, today=datetime.date(2026, 8, 15))
        build.assert_not_called()

    def test_build_without_validation_override_keeps_freshness_gate(self):
        from stock_papi.batch.observation_products_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = stock_document("2330", as_of="2026-08-07")
            document.update(name="台積電", sample_data=False)
            publish = write_quant_publish(root, [document])
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            calendar = root / "calendar.json"
            calendar.write_text(
                json.dumps(calendar_document(2026)), encoding="utf-8"
            )

            with patch(
                "reporting.cli._load_industry_map",
                return_value={"全市場": ["2330"], "半導體": ["2330"]},
            ), self.assertRaisesRegex(ValueError, "stale"):
                main(
                    [
                        "build",
                        "--root",
                        str(root),
                        "--source-market-date",
                        "2026-08-07",
                        "--source-manifest",
                        f"quant/v1/{latest['manifest']}",
                        "--source-manifest-sha256",
                        latest["manifest_sha256"],
                        "--calendar-artifact",
                        str(calendar),
                    ],
                    today=datetime.date(2026, 8, 15),
                )

    def test_build_accepts_explicit_source_validation_date_for_historical_runs(self):
        from stock_papi.batch.observation_products_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = stock_document("2330", as_of="2026-08-07")
            document.update(name="台積電", sample_data=False)
            publish = write_quant_publish(root, [document])
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            calendar = root / "calendar.json"
            calendar.write_text(
                json.dumps(calendar_document(2026)), encoding="utf-8"
            )
            output = []
            with patch(
                "reporting.cli._load_industry_map",
                return_value={"全市場": ["2330"], "半導體": ["2330"]},
            ), patch("builtins.print", side_effect=output.append):
                exit_code = main(
                    [
                        "build",
                        "--root",
                        str(root),
                        "--source-market-date",
                        "2026-08-07",
                        "--source-validation-date",
                        "2026-08-07",
                        "--source-manifest",
                        f"quant/v1/{latest['manifest']}",
                        "--source-manifest-sha256",
                        latest["manifest_sha256"],
                        "--calendar-artifact",
                        str(calendar),
                    ]
                )

            receipt = json.loads(output[-1])
            self.assertEqual(exit_code, 0)
            self.assertEqual(receipt["observation_as_of"], "2026-08-07")

    def test_build_and_promote_use_explicit_verified_source_without_model_gate(self):
        from stock_papi.batch.observation_products_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = stock_document("2330", as_of="2026-07-16")
            document.update(name="台積電", sample_data=False)
            publish = write_quant_publish(root, [document])
            latest = json.loads(
                (publish / "latest-TW.json").read_text(encoding="utf-8")
            )
            calendar = root / "calendar.json"
            calendar.write_text(
                json.dumps(calendar_document(2026)), encoding="utf-8"
            )
            output = []
            with patch(
                "reporting.cli._load_industry_map",
                return_value={"全市場": ["2330"], "半導體": ["2330"]},
            ), patch("builtins.print", side_effect=output.append):
                exit_code = main(
                    [
                        "build",
                        "--root",
                        str(root),
                        "--source-market-date",
                        "2026-07-16",
                        "--source-manifest",
                        f"quant/v1/{latest['manifest']}",
                        "--source-manifest-sha256",
                        latest["manifest_sha256"],
                        "--calendar-artifact",
                        str(calendar),
                    ],
                    today=datetime.date(2026, 7, 23),
                )
            receipt = json.loads(output[-1])

            self.assertEqual(exit_code, 0)
            self.assertEqual(receipt["mode"], "observation-candidate")
            candidate = Path(receipt["candidate_path"])
            self.assertTrue((candidate / "candidate.json").is_file())
            self.assertFalse(
                (root / "publish" / "dashboard" / "v1" / "latest-TW.json").exists()
            )

            output.clear()
            with patch("builtins.print", side_effect=output.append):
                exit_code = main(
                    [
                        "promote",
                        "--root",
                        str(root),
                        "--candidate",
                        str(candidate),
                    ]
                )
            promoted = json.loads(output[-1])
            self.assertEqual(exit_code, 0)
            self.assertEqual(promoted["mode"], "observation-local-promotion")
            self.assertTrue(Path(promoted["dashboard_latest"]).is_file())
            self.assertTrue(Path(promoted["report_latest"]).is_file())

    def test_cli_source_contains_no_model_or_bootstrap_dependency(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "stock_papi"
            / "batch"
            / "observation_products_cli.py"
        ).read_text(encoding="utf-8")

        for forbidden in (
            "BacktestStore",
            "AI_P",
            "MODEL_VERSION",
            "allow-degraded-bootstrap",
            "recommendation_engine",
            "--today",
            "--ignore-freshness",
            "--allow-stale",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
