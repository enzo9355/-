"""Real Flask route tests for the optional regression overlay."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from flask import Flask

from reporting.professional_schema import compute_content_sha256
from reporting.regression_schema import (
    MAX_REGRESSION_ARTIFACT_BYTES,
    serialize_regression_artifact,
)
from stock_papi.web.routes.reports import register_report_routes
from tests.regression_fixtures import rehash_artifact_document
from tests import test_regression_binding as binding_helpers


REGRESSION_ARTIFACT_UNAVAILABLE_REASON = "量化回歸研究尚未提供。"


def _json_bytes(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class TestRegressionRouteIntegration(unittest.TestCase):
    def setUp(self):
        fixture = binding_helpers.TestRegressionBinding()
        fixture.setUp()
        self.report = copy.deepcopy(fixture.report)
        self.metadata = copy.deepcopy(fixture.metadata)
        self.artifact = copy.deepcopy(fixture.artifact)
        self.pointer = copy.deepcopy(fixture.pointer)
        self.report["market"]["data"].update(
            ma20_breadth_pct=None,
            realized_volatility_20d_pct=None,
            advancing_count=None,
            declining_count=None,
        )
        self.report["securities"]["data"] = {
            "stock_events": [],
            "etf_observations": [],
        }
        self.report["validation"]["data"] = {
            "gates": {},
            "gate_detail_status": "not_present_in_observation_metadata",
        }
        self.report["governance"]["data"].update(
            symbol_count=None,
            failure_count=None,
        )
        self.report["identity"]["content_sha256"] = compute_content_sha256(
            self.report
        )

    def _client(self, *, regression_raw=None, pointer=None, include_pointer=True,
                canonical_raw=None, regression_loader_error=None):
        metadata = copy.deepcopy(self.metadata)
        report = copy.deepcopy(self.report)
        effective_pointer = copy.deepcopy(pointer or self.pointer)
        if regression_raw is None:
            regression_raw = serialize_regression_artifact(self.artifact)

        if include_pointer:
            metadata["regression_research"] = effective_pointer
        else:
            metadata.pop("regression_research", None)
            report["quantitative_research"] = {
                "status": "unavailable",
                "reason": "量化研究尚未提供",
                "data": {},
            }
            report["identity"]["content_sha256"] = compute_content_sha256(report)

        if canonical_raw is None:
            canonical_raw = _json_bytes(report)
        canonical_sha = hashlib.sha256(canonical_raw).hexdigest()
        metadata["professional_report"] = {
            "object": f"objects/canonical/{canonical_sha}.json",
            "sha256": canonical_sha,
            "content_sha256": report["identity"]["content_sha256"],
            "schema_version": 1,
            "generator_version": report["identity"]["generator_version"],
            "code_commit_sha": report["identity"]["code_commit_sha"],
        }
        item = {
            "report_type": "post_close",
            "source_market_date": metadata["source_market_date"],
            "applicable_trading_date": metadata["applicable_trading_date"],
            "product_mode": "observation",
        }
        calls = []

        app = Flask(
            __name__,
            template_folder=str(Path(__file__).resolve().parents[1] / "templates"),
        )
        def load_regression(path, max_bytes):
            calls.append((path, max_bytes))
            if regression_loader_error is not None:
                raise regression_loader_error
            return regression_raw

        register_report_routes(
            app,
            load_index=lambda: [],
            load_metadata=lambda _item: None,
            load_index_v2=lambda market="TW": [item],
            load_metadata_v2=lambda _item: metadata,
            load_canonical_object=lambda _path, max_bytes: canonical_raw,
            load_regression_artifact=load_regression,
            prediction_capability=SimpleNamespace(mode="research"),
        )
        for endpoint in (
            "account_page",
            "ask_page",
            "dashboard_page",
            "industries_page",
            "learn_page",
            "line_login",
            "market_page",
            "stocks_page",
        ):
            app.add_url_rule(f"/_test/{endpoint}", endpoint, lambda: "")
        return app.test_client(), calls

    def _raw_state(self, raw, *, artifact=None):
        artifact = artifact or self.artifact
        object_sha = hashlib.sha256(raw).hexdigest()
        pointer = {
            "object": f"objects/regression/{object_sha}.json",
            "sha256": object_sha,
            "content_sha256": artifact["identity"]["content_sha256"],
            "schema_version": 1,
            "generator_version": artifact["identity"]["generator_version"],
            "code_commit_sha": artifact["identity"]["code_commit_sha"],
        }
        self.report["quantitative_research"]["data"]["regression_reference"].update(
            object_sha256=object_sha,
            content_sha256=artifact["identity"]["content_sha256"],
        )
        self.report["identity"]["content_sha256"] = compute_content_sha256(
            self.report
        )
        return pointer

    def test_missing_pointer_is_http_200_and_does_not_call_loader(self):
        client, calls = self._client(include_pointer=False)
        response = client.get("/reports/2026-07-17/post-close")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [])
        self.assertIn(REGRESSION_ARTIFACT_UNAVAILABLE_REASON, response.get_data(as_text=True))

    def test_valid_artifact_is_loaded_once_and_rendered_as_structured_html(self):
        client, calls = self._client()
        response = client.get("/reports/2026-07-17/post-close")
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [(self.pointer["object"], MAX_REGRESSION_ARTIFACT_BYTES)])
        self.assertIn("Newey-West HAC", body)
        self.assertIn("HAC 標準誤", body)
        self.assertNotIn("<pre>{'", body)

    def test_optional_pointer_and_payload_failures_degrade_without_leaking_details(self):
        malformed = copy.deepcopy(self.pointer)
        malformed["object"] = "private/secret.json"
        cases = [("malformed-pointer", b"unused", malformed)]
        for name, raw in (
            ("oversized", b"x" * (MAX_REGRESSION_ARTIFACT_BYTES + 1)),
            ("invalid-utf8", b"\xff"),
            ("invalid-json", b"{"),
            ("invalid-schema", b"{}"),
        ):
            cases.append((name, raw, self._raw_state(raw)))

        tampered = copy.deepcopy(self.artifact)
        tampered["presentation"]["summary"] = "tampered without content rehash"
        tampered_raw = _json_bytes(tampered)
        cases.append(("content-hash", tampered_raw, self._raw_state(tampered_raw)))

        for name, raw, pointer in cases:
            with self.subTest(name=name):
                client, _calls = self._client(
                    regression_raw=raw,
                    pointer=pointer,
                )
                response = client.get("/reports/2026-07-17/post-close")
                body = response.get_data(as_text=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn(REGRESSION_ARTIFACT_UNAVAILABLE_REASON, body)
                self.assertNotIn("private/secret", body)
                self.assertNotIn("tampered", body)

    def test_missing_object_invalid_type_and_sha_mismatch_degrade(self):
        cases = (
            ("object-missing", None, FileNotFoundError("private object missing")),
            ("invalid-bytes-type", "not bytes", None),
            ("sha-mismatch", b"different bytes", None),
        )
        for name, raw, loader_error in cases:
            with self.subTest(name=name):
                client, calls = self._client(
                    regression_raw=raw,
                    regression_loader_error=loader_error,
                )
                response = client.get("/reports/2026-07-17/post-close")
                body = response.get_data(as_text=True)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(calls), 1)
                self.assertIn(REGRESSION_ARTIFACT_UNAVAILABLE_REASON, body)
                self.assertNotIn("private object", body)

    def test_schema_valid_but_cross_object_binding_mismatch_degrades(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["identity"]["source_manifest_sha256"] = "f" * 64
        rehash_artifact_document(artifact)
        raw = serialize_regression_artifact(artifact)
        pointer = self._raw_state(raw, artifact=artifact)
        client, calls = self._client(regression_raw=raw, pointer=pointer)
        response = client.get("/reports/2026-07-17/post-close")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertIn(
            REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
            response.get_data(as_text=True),
        )

    def test_canonical_corruption_remains_fatal(self):
        client, calls = self._client(canonical_raw=b"corrupted")
        response = client.get("/reports/2026-07-17/post-close")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(calls, [])
        self.assertNotIn("corrupted", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
