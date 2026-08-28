import contextlib
import datetime
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_papi.integrations.market_data.provider import FinMindFetchError

from local_quant import (
    LAYOUT_DIRS,
    TAIPEI,
    acquire_lock,
    build_market_insights_document,
    check_free_space,
    cleanup_expired_data,
    ensure_layout,
    _read_insights_metric,
    get_taiwan_symbols,
    load_checkpoint,
    main,
    prepare_daily_checkpoint,
    save_checkpoint,
    validate_data_root,
    window_phase,
)


def at(hour, minute):
    return datetime.datetime(2026, 7, 4, hour, minute, tzinfo=TAIPEI)


class LocalQuantTests(unittest.TestCase):
    def test_taiwan_universe_prefers_authoritative_security_master(self):
        calls = []
        master = type(
            "Master",
            (),
            {
                "entries": {"2330": object(), "0054": object(), "00679B": object()},
                "active_symbols": frozenset({"2330", "00679B"}),
            },
        )()
        resolver = type(
            "Resolver",
            (),
            {"get_master": lambda self, *, required=False: calls.append(required) or master},
        )()
        pipeline = type(
            "Pipeline",
            (),
            {
                "industry_map": {"全市場": ["2330", "00838B"]},
                "taiwan_security_master": resolver,
            },
        )()

        self.assertEqual(get_taiwan_symbols(pipeline), ["00679B", "2330"])
        self.assertEqual(calls, [True])

    def test_daily_checkpoint_archives_incompatible_target_before_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            base = {
                "schema_version": 1,
                "job_type": "daily_prediction",
                "source_manifest": "quant/v1/manifests/TW-20260714T090000Z-aaaaaaaaaaaa.json",
                "source_manifest_sha256": "a" * 64,
                "model_version": "lgbm-5d-v1",
                "next_index": 0,
                "completed_symbols": [],
                "failed_symbols": [],
                "started_at": "2026-07-14T17:00:00+08:00",
                "updated_at": "2026-07-14T17:00:00+08:00",
                "status": "running",
            }
            first = dict(
                base,
                run_id="20260714T090000Z-aaaaaaaa",
                target_market_date="2026-07-14",
            )
            second = dict(
                base,
                run_id="20260715T090000Z-bbbbbbbb",
                target_market_date="2026-07-15",
                source_manifest="quant/v1/manifests/TW-20260715T090000Z-bbbbbbbbbbbb.json",
                source_manifest_sha256="b" * 64,
                started_at="2026-07-15T17:00:00+08:00",
                updated_at="2026-07-15T17:00:00+08:00",
            )

            current = prepare_daily_checkpoint(root, first)
            replacement = prepare_daily_checkpoint(root, second)

            self.assertEqual(current["run_id"], first["run_id"])
            self.assertEqual(replacement["run_id"], second["run_id"])
            archive = (
                root
                / "checkpoints"
                / "daily_prediction"
                / "archive"
                / f"{first['run_id']}.json"
            )
            self.assertEqual(json.loads(archive.read_text(encoding="utf-8")), current)
            persisted = json.loads(
                (root / "checkpoints" / "daily_prediction" / "current.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted, replacement)

    def test_insights_metric_keeps_core_values_when_optional_field_is_dirty(self):
        document = {
            "name": "台積電", "as_of": "2026-07-06",
            "latest": {
                "AI_P": 68, "Close": 1000, "MA20": 980,
                "RET_1": "N/A", "INST_NET_RATIO": 0.4,
            },
        }
        with patch("local_quant._validated_artifact", return_value=(None, None, document)):
            result = _read_insights_metric(Path("."), "2330")

        self.assertEqual(result["prob"], 68)
        self.assertEqual(result["close"], 1000)
        self.assertIsNone(result["return_1d"])
        self.assertEqual(result["inst_ratio"], 0.4)

    def test_market_insights_reads_theme_symbols_not_full_universe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type("Pipeline", (), {
                "industry_map": {
                    "全市場": ["9999"],
                    "半導體": ["2330", "2303", "2454", "3034", "2379"],
                },
                "get_stock_name": staticmethod(lambda symbol: f"公司{symbol}"),
            })()
            with patch("local_quant._read_insights_metric") as read_metric:
                read_metric.return_value = None
                document = build_market_insights_document(
                    root, pipeline, now=at(6, 0),
                    fetch_json=lambda _url: [], fetch_etf=lambda _etf: [],
                )

            symbols = {call.args[1] for call in read_metric.call_args_list}
            self.assertNotIn("9999", symbols)
            self.assertEqual(len(symbols & {"2330", "2303", "2454", "3034", "2379"}), 5)
            self.assertEqual(len(document["industries"][0]["leaders"]), 5)
            self.assertEqual(document["industries"][0]["coverage"], 0)
            self.assertIn("2330", symbols)

    def test_cli_insights_builds_and_publishes_without_market_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type("Pipeline", (), {"industry_map": {"全市場": ["2330"]}})()
            document = {
                "schema_version": 1, "as_of": "2026-07-04",
                "industries": [], "mops": [], "etfs": [], "supply_chains": [], "sources": [],
            }
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data", return_value={}),
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch("local_quant.build_market_insights_document", return_value=document) as build,
                patch("local_quant.publish_market_insights") as publish,
                patch("local_quant.run_market_batch") as batch,
            ):
                result = main(
                    ["--root", str(root), "--insights"],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            build.assert_called_once_with(root, pipeline, now=at(6, 0))
            publish.assert_called_once_with(root, document, generated_at=at(6, 0))
            batch.assert_not_called()

    def test_cleanup_expired_data_is_allowlisted_and_age_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "StockPapiData"
            ensure_layout(root)
            now = at(6, 0)

            files = {
                "old_tmp": root / "cache" / "tmp" / "nested" / "old.tmp",
                "new_tmp": root / "cache" / "tmp" / "new.tmp",
                "old_log": root / "logs" / "old.log",
                "artifact": root / "artifacts" / "stocks" / "TW" / "2330.json.gz",
                "secret": root / "secrets" / "token.txt",
                "progress": root / "checkpoints" / "progress.json",
                "active_lock": root / "checkpoints" / "runner.lock",
                "stale_lock": root / "checkpoints" / "runner.lock.stale.20260601T000000",
            }
            for path in files.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")

            old_2_days = (now - datetime.timedelta(days=2)).timestamp()
            old_31_days = (now - datetime.timedelta(days=31)).timestamp()
            old_8_days = (now - datetime.timedelta(days=8)).timestamp()
            os.utime(files["old_tmp"], (old_2_days, old_2_days))
            os.utime(files["old_log"], (old_31_days, old_31_days))
            os.utime(files["stale_lock"], (old_8_days, old_8_days))
            for name in ("artifact", "secret", "progress", "active_lock"):
                os.utime(files[name], (old_31_days, old_31_days))

            outside = base / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            link = root / "cache" / "tmp" / "outside-link"
            linked = False
            try:
                link.symlink_to(outside)
                linked = True
            except OSError:
                pass

            with patch("local_quant.validate_data_root", return_value=root):
                summary = cleanup_expired_data(root, now=now)

            self.assertFalse(files["old_tmp"].exists())
            self.assertFalse(files["old_log"].exists())
            self.assertFalse(files["stale_lock"].exists())
            self.assertTrue(files["new_tmp"].exists())
            for name in ("artifact", "secret", "progress", "active_lock"):
                self.assertTrue(files[name].exists())
            self.assertTrue(outside.exists())
            if linked:
                self.assertTrue(link.is_symlink())
            self.assertEqual(
                set(summary),
                {"deleted_files", "reclaimed_bytes", "failed", "skipped_reparse_points"},
            )
            self.assertEqual(summary["deleted_files"], 3)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["skipped_reparse_points"], int(linked))
            self.assertFalse((root / "cache" / "tmp" / "nested").exists())

    def test_data_root_must_be_stock_papi_directory_on_d_drive(self):
        self.assertEqual(
            validate_data_root(Path("D:/StockPapiData")),
            Path("D:/StockPapiData"),
        )
        for invalid in ("C:/StockPapiData", "D:/Other", "StockPapiData"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_data_root(Path(invalid))

    def test_window_phases_enforce_run_drain_checkpoint_and_closed(self):
        self.assertEqual(window_phase(at(2, 29)), "closed")
        self.assertEqual(window_phase(at(2, 30)), "run")
        self.assertEqual(window_phase(at(5, 29)), "run")
        self.assertEqual(window_phase(at(9, 19)), "run")
        self.assertEqual(window_phase(at(9, 20)), "drain")
        self.assertEqual(window_phase(at(9, 25)), "checkpoint")
        self.assertEqual(window_phase(at(9, 30)), "closed")

    def test_layout_creates_only_expected_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                set(LAYOUT_DIRS),
            )

    def test_free_space_guard_rejects_low_capacity(self):
        with self.assertRaises(RuntimeError):
            check_free_space(Path("D:/StockPapiData"), 100, free_bytes=99 * 1024**3)
        self.assertEqual(
            check_free_space(
                Path("D:/StockPapiData"), 100, free_bytes=101 * 1024**3
            ),
            101 * 1024**3,
        )

    def test_lock_is_single_instance_and_releases_its_own_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            lock = acquire_lock(root, now=at(5, 30))
            with self.assertRaises(RuntimeError):
                acquire_lock(root, now=at(5, 31))
            lock.release()
            replacement = acquire_lock(root, now=at(5, 32))
            replacement.release()

    def test_previous_day_lock_is_archived_before_reacquiring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            lock_path = root / "checkpoints" / "runner.lock"
            lock_path.write_text(
                json.dumps({"token": "old", "started_at": at(5, 30).isoformat()}),
                encoding="utf-8",
            )

            replacement = acquire_lock(
                root,
                now=at(5, 30) + datetime.timedelta(days=1),
            )

            self.assertEqual(len(list(lock_path.parent.glob("runner.lock.stale.*"))), 1)
            replacement.release()

    def test_checkpoint_round_trip_uses_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(root, {"stage": "prices", "symbol": "2330"})
            self.assertEqual(
                load_checkpoint(root),
                {"stage": "prices", "symbol": "2330"},
            )
            self.assertFalse(
                (root / "checkpoints" / "progress.json.tmp").exists()
            )

    def test_cli_initialization_does_not_overwrite_market_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            original = {"stage": "market_batch", "market": "TW", "next_index": 200}
            save_checkpoint(root, original)
            with patch("local_quant.validate_data_root", return_value=root):
                result = main(
                    ["--root", str(root), "--init", "--dry-run"],
                    now=at(5, 30),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            self.assertEqual(load_checkpoint(root), original)
            self.assertFalse((root / "checkpoints" / "runner.lock").exists())
            status = json.loads(
                (root / "logs" / "runner-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["phase"], "run")

    def test_cli_outside_window_reports_closed_without_work_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("local_quant.validate_data_root", return_value=root):
                result = main(
                    ["--root", str(root), "--init", "--dry-run"],
                    now=at(22, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            self.assertFalse((root / "checkpoints" / "runner.lock").exists())
            self.assertEqual(load_checkpoint(root), {})

    def test_cli_returns_nonzero_when_free_space_is_below_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("local_quant.validate_data_root", return_value=root):
                result = main(
                    ["--root", str(root), "--init", "--dry-run"],
                    now=at(5, 30),
                    free_bytes=50 * 1024**3,
                )

            self.assertEqual(result, 2)

    def test_cli_run_loads_pipeline_only_inside_work_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type(
                "Pipeline",
                (),
                {"industry_map": {"全市場": ["2330", "2317", "../bad", "2330"]}},
            )()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch(
                    "local_quant.cleanup_expired_data",
                    return_value={
                        "deleted_files": 1,
                        "reclaimed_bytes": 10,
                        "failed": 0,
                        "skipped_reparse_points": 0,
                    },
                ) as cleanup,
                patch("local_quant.load_stock_pipeline", return_value=pipeline) as loader,
                patch("local_quant.run_market_batch", return_value={"attempted": 2}) as batch,
            ):
                result = main(
                    [
                        "--root", str(root), "--run", "--market", "TW",
                        "--limit", "2", "--delay", "0",
                    ],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            cleanup.assert_called_once_with(root, now=at(6, 0))
            loader.assert_called_once_with(root)
            self.assertEqual(batch.call_args.args[2], ["2317", "2330"])
            status = json.loads(
                (root / "logs" / "runner-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["cleanup"]["deleted_files"], 1)

            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data") as closed_cleanup,
                patch("local_quant.load_stock_pipeline") as closed_loader,
            ):
                result = main(
                    ["--root", str(root), "--run", "--market", "TW"],
                    now=at(22, 0),
                    free_bytes=200 * 1024**3,
                )
            self.assertEqual(result, 0)
            closed_cleanup.assert_not_called()
            closed_loader.assert_not_called()

    def test_cli_returns_nonzero_without_publish_on_provider_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type(
                "Pipeline",
                (),
                {"industry_map": {"全市場": ["2330", "2317"]}},
            )()
            error = FinMindFetchError(
                "quota_or_rate_limit",
                "TaiwanStockPrice",
                "2330",
                "2026-07-01",
                "2026-07-23",
                http_status=402,
                exception_type="HTTPError",
                blocked_until=2000,
                retry_after_seconds=3600,
            )
            stderr = io.StringIO()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data", return_value={}),
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch("local_quant.run_market_batch", side_effect=error),
                patch("local_quant.publish_market_snapshot") as publish,
                contextlib.redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "--root",
                        str(root),
                        "--run",
                        "--market",
                        "TW",
                        "--delay",
                        "0",
                    ],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 2)
            publish.assert_not_called()
            self.assertIn("category=quota_or_rate_limit", stderr.getvalue())
            self.assertNotIn("token", stderr.getvalue().lower())

    def test_cli_all_runs_taiwan_then_us_with_independent_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type("Pipeline", (), {"industry_map": {"全市場": ["2330"]}})()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch(
                    "local_quant.cleanup_expired_data",
                    return_value={
                        "deleted_files": 0,
                        "reclaimed_bytes": 0,
                        "failed": 0,
                        "skipped_reparse_points": 0,
                    },
                ),
                patch("local_quant.load_stock_pipeline", return_value=pipeline) as loader,
                patch("local_quant.get_us_symbols", return_value=["AAPL"]) as us_symbols,
                patch(
                    "local_quant.run_market_batch",
                    side_effect=[{"attempted": 1}, {"attempted": 1}],
                ) as batch,
            ):
                result = main(
                    [
                        "--root", str(root), "--run", "--market", "ALL",
                        "--limit", "5000", "--delay", "0",
                    ],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            loader.assert_called_once_with(root)
            us_symbols.assert_called_once_with(root, now=at(6, 0))
            self.assertEqual(
                [(call.args[1], call.args[2]) for call in batch.call_args_list],
                [("TW", ["2330"]), ("US", ["AAPL"])],
            )

    def test_cli_publishes_only_after_market_batch_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_checkpoint(
                root,
                {
                    "stage": "market_batch",
                    "market": "TW",
                    "next_index": 1,
                    "failed": [],
                    "cycle_completed_on": "2026-07-04",
                },
            )
            pipeline = type("Pipeline", (), {"industry_map": {"全市場": ["2330"]}})()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch(
                    "local_quant.cleanup_expired_data",
                    return_value={
                        "deleted_files": 0,
                        "reclaimed_bytes": 0,
                        "failed": 0,
                        "skipped_reparse_points": 0,
                    },
                ),
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch(
                    "local_quant.run_market_batch",
                    return_value={
                        "attempted": 1,
                        "completed": 1,
                        "failed": [],
                        "next_index": 1,
                    },
                ),
                patch("local_quant.publish_market_snapshot") as publish,
            ):
                result = main(
                    ["--root", str(root), "--run", "--market", "TW", "--delay", "0"],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            publish.assert_called_once_with(
                root, "TW", ["2330"], generated_at=at(6, 0), failed_symbols=[]
            )
            checkpoint = load_checkpoint(root)
            self.assertEqual(checkpoint["published_cycle_on"], "2026-07-04")

    def test_cli_publishes_partial_market_below_failure_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            symbols = [f"{number:04d}" for number in range(100)]
            pipeline = type(
                "Pipeline", (), {"industry_map": {"全市場": symbols}}
            )()
            failed = [{"symbol": "0099", "error": "ValueError"}]
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data", return_value={}),
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch(
                    "local_quant.run_market_batch",
                    return_value={"failed": failed, "next_index": 100},
                ),
                patch("local_quant.publish_market_snapshot") as publish,
            ):
                publish.return_value = root / "publish" / "quant" / "v1" / "latest-TW.json"
                result = main(
                    ["--root", str(root), "--run", "--market", "TW", "--delay", "0"],
                    now=at(6, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            publish.assert_called_once_with(
                root,
                "TW",
                symbols,
                generated_at=at(6, 0),
                failed_symbols=["0099"],
            )

    def test_cli_refuses_us_market_before_0530(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.load_stock_pipeline") as loader,
                patch("local_quant.get_us_symbols") as us_symbols,
            ):
                result = main(
                    ["--root", str(root), "--run", "--market", "US"],
                    now=at(3, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            loader.assert_not_called()
            us_symbols.assert_not_called()
            status = json.loads(
                (root / "logs" / "runner-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["phase"], "closed")

    def test_observation_only_post_close_skips_market_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type("Pipeline", (), {"industry_map": {"全市場": ["2330"]}})()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data", return_value={}),
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch(
                    "local_quant.run_market_batch",
                    return_value={
                        "attempted": 1,
                        "completed": 1,
                        "failed": [],
                        "pending": [],
                        "excluded": [],
                        "next_index": 1,
                    },
                ),
                patch("local_quant.publish_market_snapshot") as publish,
            ):
                result = main(
                    [
                        "--root",
                        str(root),
                        "--post-close",
                        "--observation-only",
                        "--market",
                        "TW",
                        "--target-market-date",
                        "2026-07-24",
                        "--delay",
                        "0",
                    ],
                    now=at(17, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            publish.assert_not_called()
            latest_path = root / "publish" / "quant" / "v1" / "latest-TW.json"
            self.assertFalse(latest_path.exists())
            checkpoint = load_checkpoint(root, market="TW")
            self.assertNotIn("published_at", checkpoint)
            self.assertNotIn("published_cycle_on", checkpoint)
            self.assertNotIn("published_failure_count", checkpoint)

    def test_observation_only_post_close_skips_retention_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            pipeline = type("Pipeline", (), {"industry_map": {"全市場": ["2330"]}})()
            with (
                patch("local_quant.validate_data_root", return_value=root),
                patch("local_quant.cleanup_expired_data", return_value={}) as cleanup,
                patch("local_quant.load_stock_pipeline", return_value=pipeline),
                patch(
                    "local_quant.run_market_batch",
                    return_value={
                        "attempted": 1,
                        "completed": 1,
                        "failed": [],
                        "pending": [],
                        "excluded": [],
                        "next_index": 1,
                    },
                ),
                patch("local_quant.publish_market_snapshot"),
            ):
                result = main(
                    [
                        "--root",
                        str(root),
                        "--post-close",
                        "--observation-only",
                        "--market",
                        "TW",
                        "--target-market-date",
                        "2026-07-24",
                        "--delay",
                        "0",
                    ],
                    now=at(17, 0),
                    free_bytes=200 * 1024**3,
                )

            self.assertEqual(result, 0)
            cleanup.assert_not_called()
            status = json.loads(
                (root / "logs" / "runner-status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status.get("cleanup", {}).get("skipped"))
            self.assertEqual(
                status.get("cleanup", {}).get("reason"), "observation_only"
            )

    def test_observation_batch_does_not_rewrite_existing_exclusion_list(self):
        import hashlib
        from local_quant import run_market_batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            csv_dir = root / "checkpoints"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / "exclusion_list-TW.csv"
            initial_content = (
                "Symbol,Name,ExclusionDate,ConsecutiveFailures,State,Type,Reason,OperatorAction\n"
                "3426,3426,2026-07-26,5,Excluded,delisted,test,\n"
            )
            csv_path.write_text(initial_content, encoding="utf-8")
            stat_before = csv_path.stat()
            before_bytes = csv_path.read_bytes()
            before_sha256 = hashlib.sha256(before_bytes).hexdigest()

            result = run_market_batch(
                root,
                "TW",
                ["3426", "2330"],
                analyze_symbol=lambda s: {"mock": True},
                delay=0,
                enforce_window=False,
                mutate_exclusions=False,
            )

            after_bytes = csv_path.read_bytes()
            after_sha256 = hashlib.sha256(after_bytes).hexdigest()
            stat_after = csv_path.stat()

            self.assertEqual(after_bytes, before_bytes)
            self.assertEqual(after_sha256, before_sha256)
            self.assertEqual(stat_after.st_mtime_ns, stat_before.st_mtime_ns)
            self.assertIn("3426", result["excluded"])

    def test_observation_batch_does_not_add_pending_exclusion(self):
        from local_quant import save_consecutive_failures, run_market_batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            save_consecutive_failures(root, "TW", {"2330": 4})
            csv_path = root / "checkpoints" / "exclusion_list-TW.csv"

            def failing_analyzer(symbol):
                raise ValueError("delisted symbol 2330")

            result = run_market_batch(
                root,
                "TW",
                ["2330"],
                analyze_symbol=failing_analyzer,
                delay=0,
                enforce_window=False,
                mutate_exclusions=False,
            )

            self.assertFalse(csv_path.exists())
            self.assertEqual(len(result["failed"]), 1)
            self.assertEqual(result["failed"][0]["symbol"], "2330")

    def test_observation_batch_refuses_pending_operator_actions(self):
        import hashlib
        from unittest.mock import Mock
        from local_quant import run_market_batch
        for action in ("Approve", "Reinstate"):
            with self.subTest(action=action):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    ensure_layout(root)
                    csv_dir = root / "checkpoints"
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = csv_dir / "exclusion_list-TW.csv"
                    initial_content = (
                        "Symbol,Name,ExclusionDate,ConsecutiveFailures,State,Type,Reason,OperatorAction\n"
                        f"3426,3426,2026-07-26,5,Excluded,delisted,test,{action}\n"
                    )
                    csv_path.write_text(initial_content, encoding="utf-8")
                    stat_before = csv_path.stat()
                    before_bytes = csv_path.read_bytes()
                    before_sha256 = hashlib.sha256(before_bytes).hexdigest()

                    analyzer = Mock()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "observation-only refuses pending exclusion operator actions",
                    ):
                        run_market_batch(
                            root,
                            "TW",
                            ["3426"],
                            analyze_symbol=analyzer,
                            delay=0,
                            enforce_window=False,
                            mutate_exclusions=False,
                        )

                    analyzer.assert_not_called()
                    after_bytes = csv_path.read_bytes()
                    after_sha256 = hashlib.sha256(after_bytes).hexdigest()
                    stat_after = csv_path.stat()

                    self.assertEqual(after_bytes, before_bytes)
                    self.assertEqual(after_sha256, before_sha256)
                    self.assertEqual(stat_after.st_mtime_ns, stat_before.st_mtime_ns)
                    self.assertFalse((root / "artifacts" / "stocks" / "TW" / "3426.json.gz").exists())
                    self.assertFalse((root / "checkpoints" / "progress.json").exists())

    def test_observation_batch_refuses_exclusion_schema_without_operator_action(self):
        import hashlib
        from unittest.mock import Mock
        from local_quant import run_market_batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ensure_layout(root)
            csv_dir = root / "checkpoints"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / "exclusion_list-TW.csv"
            initial_content = (
                "Symbol,Name,ExclusionDate,ConsecutiveFailures,State,Type,Reason\n"
                "3426,3426,2026-07-26,5,Excluded,delisted,test\n"
            )
            csv_path.write_text(initial_content, encoding="utf-8")
            stat_before = csv_path.stat()
            before_bytes = csv_path.read_bytes()
            before_sha256 = hashlib.sha256(before_bytes).hexdigest()

            analyzer = Mock()
            with self.assertRaisesRegex(
                RuntimeError,
                "observation-only exclusion preflight failed",
            ):
                run_market_batch(
                    root,
                    "TW",
                    ["3426"],
                    analyze_symbol=analyzer,
                    delay=0,
                    enforce_window=False,
                    mutate_exclusions=False,
                )

            analyzer.assert_not_called()
            after_bytes = csv_path.read_bytes()
            after_sha256 = hashlib.sha256(after_bytes).hexdigest()
            stat_after = csv_path.stat()

            self.assertEqual(after_bytes, before_bytes)
            self.assertEqual(after_sha256, before_sha256)
            self.assertEqual(stat_after.st_mtime_ns, stat_before.st_mtime_ns)
            self.assertFalse((root / "artifacts" / "stocks" / "TW" / "3426.json.gz").exists())
            self.assertFalse((root / "checkpoints" / "progress.json").exists())


if __name__ == "__main__":
    unittest.main()
