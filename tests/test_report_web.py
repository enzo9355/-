import datetime
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")
os.environ.setdefault("RENDER_GIT_COMMIT", "b" * 40)

import app as stock_app

from reporting.observation_v2 import build_post_close_observation_metadata
from reporting.publisher import publish_report_v2
from reporting.web import validate_report_index
from tests.test_observation_public_surfaces import observation_dashboard


class Calendar:
    def next_session(self, value):
        self.requested = value
        return datetime.date(2026, 7, 16)


class ReportWebTests(unittest.TestCase):
    def _objects(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        metadata = build_post_close_observation_metadata(
            observation_dashboard(), Calendar()
        )
        from reporting.professional_builder import build_professional_post_close_artifact
        prof_report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        publish_report_v2(root, metadata, professional_report=prof_report)
        publish = root / "publish" / "reports" / "v2"
        objects = {
            f"reports/v2/{path.relative_to(publish).as_posix()}": path.read_bytes()
            for path in publish.rglob("*")
            if path.is_file()
        }
        return temporary, objects, metadata

    def _production_shaped_objects(self):
        """以脫敏的 Production schema 形狀建立完整 v2 artifacts。"""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "production_observation_report_shapes.json"
        )
        shapes = json.loads(fixture_path.read_text(encoding="utf-8"))
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        post_close = build_post_close_observation_metadata(
            observation_dashboard(), Calendar()
        )
        post_close["content"] = shapes["post_close_content"]
        post_close["summary"] = ["市場廣度維持中性"]
        from reporting.professional_builder import build_professional_post_close_artifact
        prof_report = build_professional_post_close_artifact(
            post_close, code_commit_sha="b" * 40
        )
        publish_report_v2(root, post_close, professional_report=prof_report)

        publish = root / "publish" / "reports" / "v2"
        post_close_item = next(
            item
            for item in validate_report_index((publish / "index-TW.json").read_bytes())
            if item["report_type"] == "post_close"
        )

        pre_market = copy.deepcopy(post_close)
        pre_market_content = copy.deepcopy(shapes["pre_market_content"])
        pre_market_content["base_metadata_sha256"] = post_close_item[
            "metadata_sha256"
        ]
        pre_market.update(
            report_type="pre_market",
            published_at="2026-07-16T23:30:00Z",
            title="2026-07-16 盤前風險更新",
            summary=["隔夜訊號分歧"],
            warnings=[],
            content=pre_market_content,
        )
        # Drop professional_report pointer from pre_market if any
        pre_market.pop("professional_report", None)
        publish_report_v2(root, pre_market)
        objects = {
            f"reports/v2/{path.relative_to(publish).as_posix()}": path.read_bytes()
            for path in publish.rglob("*")
            if path.is_file()
        }
        return temporary, objects

    def test_production_shaped_daily_reports_have_distinct_canonical_pages(self):
        temporary, objects = self._production_shaped_objects()
        self.addCleanup(temporary.cleanup)

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ):
            client = stock_app.app.test_client()
            post_close = client.get("/reports/2026-07-15/post-close")
            pre_market = client.get("/reports/2026-07-16/pre-market")
            legacy_index = client.get("/reports/trading-day/2026-07-16")

        self.assertEqual(post_close.status_code, 200)
        self.assertIn("台股市場、產業與量化研究日報", post_close.get_data(as_text=True))
        self.assertEqual(pre_market.status_code, 200)
        self.assertIn("隔夜訊號分歧", pre_market.get_data(as_text=True))
        self.assertIn("有效標的</dt><dd>1042</dd>", post_close.get_data(as_text=True))
        self.assertEqual(legacy_index.status_code, 200)
        index_html = legacy_index.get_data(as_text=True)
        self.assertIn("/reports/2026-07-15/post-close", index_html)
        self.assertIn("/reports/2026-07-16/pre-market", index_html)

    def test_v2_observation_report_is_the_only_formal_report_surface(self):
        temporary, objects, metadata = self._objects()
        self.addCleanup(temporary.cleanup)

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ):
            client = stock_app.app.test_client()
            listing = client.get("/reports")
            trading_day = client.get("/reports/2026-07-15/post-close")
            pre_market = client.get("/reports/2026-07-16/pre-market")
            weekly = client.get("/reports/weekly/2026-W29")

        self.assertEqual(listing.status_code, 200)
        listing_html = listing.get_data(as_text=True)
        self.assertIn(metadata["title"], listing_html)
        self.assertIn("盤後觀察", listing_html)
        self.assertIn("閱讀盤後觀察", listing_html)

        self.assertEqual(trading_day.status_code, 200)
        html = trading_day.get_data(as_text=True)
        for label in (
            "市場總體與風險",
            "產業輪動與排名",
            "個股異常事件",
            "ETF 觀察",
            "資料治理與方法論",
        ):
            self.assertIn(label, html)

        self.assertEqual(
            trading_day.headers["Cache-Control"], "public, max-age=300"
        )
        self.assertEqual(pre_market.status_code, 404)
        self.assertEqual(weekly.status_code, 404)

    def test_legacy_reports_are_hidden_and_not_loaded_in_research_mode(self):
        with patch.object(
            stock_app, "_gcs_get_report_object", return_value=b"must-not-load"
        ) as legacy:
            client = stock_app.app.test_client()
            listing = client.get("/reports")
            report = client.get("/reports/2026-07-03")
            preview = client.get("/reports/2026-07-03/preview")
            download = client.get("/reports/2026-07-03/download")

        self.assertEqual(listing.status_code, 503)
        self.assertEqual(listing.headers["Cache-Control"], "no-store")
        self.assertEqual(report.status_code, 404)
        self.assertEqual(preview.status_code, 302)
        self.assertEqual(download.status_code, 302)
        legacy.assert_not_called()

    def test_empty_reports_page_has_clear_state(self):
        with patch.object(
            stock_app, "_published_report_index_v2", return_value=[]
        ):
            response = stock_app.app.test_client().get("/reports")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "目前沒有可用的每日報告",
            response.get_data(as_text=True),
        )

    def test_missing_report_index_has_dedicated_503_state(self):
        with patch.object(
            stock_app, "_gcs_get_report_v2_object", return_value=None, create=True
        ):
            response = stock_app.app.test_client().get("/reports")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertIn("報告暫時無法", response.get_data(as_text=True))

    def test_pre_market_rejects_wrong_post_close_lineage(self):
        temporary, objects = self._production_shaped_objects()
        self.addCleanup(temporary.cleanup)
        index = json.loads(objects["reports/v2/index-TW.json"])
        pre_market = next(
            item for item in index["reports"] if item["report_type"] == "pre_market"
        )
        metadata_path = f"reports/v2/{pre_market['metadata']}"
        metadata = json.loads(objects[metadata_path])
        metadata["content"]["base_metadata_sha256"] = "f" * 64
        content_bytes = json.dumps(
            metadata["content"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metadata["content_sha256"] = hashlib.sha256(content_bytes).hexdigest()
        pre_market["content_sha256"] = metadata["content_sha256"]
        encoded = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metadata_sha256 = hashlib.sha256(encoded).hexdigest()
        pre_market["metadata"] = f"metadata/{metadata_sha256}.json"
        pre_market["metadata_sha256"] = metadata_sha256
        objects["reports/v2/index-TW.json"] = json.dumps(
            index,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        objects[f"reports/v2/{pre_market['metadata']}"] = encoded

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path),
            create=True,
        ):
            response = stock_app.app.test_client().get(
                "/reports/2026-07-16/pre-market"
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_unexpected_report_render_error_is_safe_and_correlated(self):
        temporary, objects, _metadata = self._objects()
        self.addCleanup(temporary.cleanup)

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ), patch(
            "stock_papi.web.routes.reports.build_professional_report_view",
            side_effect=RuntimeError("private object detail")
        ), self.assertLogs(stock_app.app.logger, level="ERROR") as logs:
            response = stock_app.app.test_client().get(
                "/reports/2026-07-15/post-close"
            )

        correlation_id = response.headers["X-Correlation-ID"]
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertNotIn("private object detail", body)
        self.assertIn(correlation_id, body)
        self.assertTrue(any(correlation_id in message for message in logs.output))

    def test_sample_download_redirects_to_public_html_list(self):
        response = stock_app.app.test_client().get("/reports/sample/download")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/reports"))
        self.assertNotEqual(response.mimetype, "application/pdf")

    def test_corrupt_observation_metadata_fails_closed(self):
        temporary, objects, _metadata = self._objects()
        self.addCleanup(temporary.cleanup)
        metadata_path = next(
            path for path in objects if "/metadata/" in path
        )
        objects[metadata_path] = b"corrupt"

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path),
            create=True,
        ):
            client = stock_app.app.test_client()
            with self.assertLogs(stock_app.app.logger, level="ERROR") as logs:
                bad_hash = client.get("/reports/2026-07-15/post-close")
            bad_date = client.get("/reports/trading-day/not-a-date")
            missing = client.get("/reports/trading-day/2026-07-17")
            traversal = client.get("/reports/../../secret")

        self.assertEqual(bad_hash.status_code, 503)
        self.assertNotIn("metadata/", bad_hash.get_data(as_text=True))
        self.assertEqual(bad_hash.headers["Cache-Control"], "no-store")
        self.assertEqual(len(bad_hash.headers["X-Correlation-ID"]), 16)
        self.assertTrue(
            any(
                bad_hash.headers["X-Correlation-ID"] in message
                for message in logs.output
            )
        )
        self.assertEqual(bad_date.status_code, 404)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(traversal.status_code, 404)

    def test_post_close_integrity_failure_returns_safe_503(self):
        temporary, objects, _metadata = self._objects()
        self.addCleanup(temporary.cleanup)
        canonical_key = next(k for k in objects if "objects/canonical/" in k)
        objects[canonical_key] = b'{"corrupted": true}'

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ):
            client = stock_app.app.test_client()
            response = client.get("/reports/2026-07-15/post-close")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertNotIn("corrupted", response.get_data(as_text=True))
        self.assertNotIn("objects/canonical", response.get_data(as_text=True))

    def test_post_close_redirect_and_routing_semantics(self):
        temporary, objects = self._production_shaped_objects()
        self.addCleanup(temporary.cleanup)

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ):
            client = stock_app.app.test_client()
            # 1. Canonical source date -> 200
            post_close_canonical = client.get("/reports/2026-07-15/post-close")
            self.assertEqual(post_close_canonical.status_code, 200)

            # 2. Applicable date -> 302 canonical source date
            post_close_applicable = client.get("/reports/2026-07-16/post-close")
            self.assertEqual(post_close_applicable.status_code, 302)
            self.assertEqual(
                post_close_applicable.headers["Location"],
                "/reports/2026-07-15/post-close",
            )

            # Follow redirect gives 200 (no redirect loop)
            followed = client.get(post_close_applicable.headers["Location"])
            self.assertEqual(followed.status_code, 200)

            # 3. Missing report properly returns 404
            missing_report = client.get("/reports/2026-07-25/post-close")
            self.assertEqual(missing_report.status_code, 404)

            # 4. Homepage CTA uses canonical source date for post-close
            home = client.get("/")
            if home.status_code == 200:
                home_html = home.get_data(as_text=True)
                self.assertIn("/reports/2026-07-15/post-close", home_html)
                self.assertNotIn("/reports/2026-07-16/post-close", home_html)

    def test_post_close_same_date_source_and_applicable(self):
        class SameDateCalendar:
            def next_session(self, value):
                return value

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        metadata = build_post_close_observation_metadata(
            observation_dashboard(), SameDateCalendar()
        )
        from reporting.professional_builder import build_professional_post_close_artifact
        prof_report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        publish_report_v2(root, metadata, professional_report=prof_report)
        publish = root / "publish" / "reports" / "v2"
        objects = {
            f"reports/v2/{path.relative_to(publish).as_posix()}": path.read_bytes()
            for path in publish.rglob("*")
            if path.is_file()
        }

        with patch.object(
            stock_app,
            "_gcs_get_report_v2_object",
            side_effect=lambda path, _size: objects.get(path) if path.startswith("reports/v2/") else objects.get(f"reports/v2/{path}"),
            create=True,
        ):
            client = stock_app.app.test_client()
            res = client.get(f"/reports/{metadata['source_market_date']}/post-close")
            self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
