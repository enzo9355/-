"""Real overlay and Flask/Jinja rendering tests for regression research."""

from pathlib import Path
import unittest

from flask import Flask, render_template

from reporting.professional_html import (
    REGRESSION_UNAVAILABLE_REASON,
    build_professional_report_view,
)
from reporting.professional_schema import ProfessionalPostCloseReport
from reporting.regression_schema import RegressionResearchArtifact
from tests.regression_fixtures import DISCLOSURE, make_artifact_document
from tests.test_professional_report_schema import ProfessionalReportSchemaTests


REGRESSION_ARTIFACT_UNAVAILABLE_REASON = "量化回歸研究尚未提供。"


class TestRegressionPresentation(unittest.TestCase):
    def setUp(self):
        self.report = ProfessionalPostCloseReport.from_document(
            ProfessionalReportSchemaTests()._document()
        )
        self.artifact = RegressionResearchArtifact.from_document(make_artifact_document())

    def test_available_overlay_is_structured_and_does_not_mutate_canonical_report(self):
        before = self.report.to_document()
        view = build_professional_report_view(
            self.report,
            regression_artifact=self.artifact,
        )
        research = view["quantitative_research"]
        self.assertEqual(research["status"], "available")
        self.assertEqual(research["sample_count"], 60)
        self.assertEqual(research["ai_label"], "AI 模型參考建議")
        self.assertEqual(research["output_name"], "模型方向參考")
        self.assertEqual(research["confidence_level"], 0.95)
        self.assertEqual(research["r_squared"], 0.2)
        self.assertEqual(research["adjusted_r_squared"], 0.15)
        self.assertEqual(research["warnings"], [])
        self.assertEqual(research["disclosure"], DISCLOSURE)
        self.assertEqual(len(research["factors"]), 3)
        self.assertEqual(self.report.to_document(), before)
        serialized_view = repr(view)
        self.assertNotIn("objects/regression", serialized_view)
        self.assertNotIn(self.artifact.identity.content_sha256, serialized_view)

    def test_unavailable_overlay_uses_default_reason(self):
        view = build_professional_report_view(self.report)
        self.assertEqual(
            view["quantitative_research"]["reason"],
            REGRESSION_UNAVAILABLE_REASON,
        )

    def test_unavailable_overlay_accepts_only_whitelisted_reason(self):
        view = build_professional_report_view(
            self.report,
            regression_unavailable_reason=REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
        )
        self.assertEqual(
            view["quantitative_research"]["reason"],
            REGRESSION_ARTIFACT_UNAVAILABLE_REASON,
        )

        view = build_professional_report_view(
            self.report,
            regression_unavailable_reason="private bucket/path exception",
        )
        research = view["quantitative_research"]
        self.assertEqual(research["status"], "unavailable")
        self.assertNotIn("private", research["reason"])
        self.assertEqual(research["reason"], REGRESSION_UNAVAILABLE_REASON)
        self.assertEqual(research["data"], {})

        view = build_professional_report_view(
            self.report,
            regression_unavailable_reason=RuntimeError("raw database exception"),
        )
        self.assertEqual(
            view["quantitative_research"]["reason"],
            REGRESSION_UNAVAILABLE_REASON,
        )
        self.assertNotIn("database", repr(view))

    def test_real_flask_response_renders_structured_statistics_without_raw_dict(self):
        app = Flask(
            __name__,
            template_folder=str(Path(__file__).resolve().parents[1] / "templates"),
        )
        for endpoint in (
            "account_page",
            "ask_page",
            "dashboard_page",
            "industries_page",
            "learn_page",
            "line_login",
            "market_page",
            "reports_page",
            "stocks_page",
            "us_reports_page",
        ):
            app.add_url_rule(f"/_test/{endpoint}", endpoint, lambda: "")

        @app.get("/report")
        def report_page():
            view = build_professional_report_view(
                self.report,
                regression_artifact=self.artifact,
            )
            for section_name in (
                "market",
                "capital_flows",
                "industries",
                "securities",
                "validation",
                "next_session",
                "governance",
                "ai_reference",
            ):
                view[section_name] = {
                    "status": "unavailable",
                    "reason": "測試資料未提供。",
                    "data": {},
                }
            return render_template(
                "reports/post_close_professional.html",
                report=view,
            )

        response = app.test_client().get("/report")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        for expected in (
            "研究期間",
            "樣本數",
            "Newey-West HAC",
            "OLS /",
            "HAC 標準誤",
            "t-stat",
            "p-value",
            "95% CI",
            "Adjusted R²",
            "VIF",
            "Breusch-Pagan",
            "Durbin-Watson",
            "Jarque-Bera",
            "AI 模型參考建議",
            "模型方向參考",
            DISCLOSURE,
        ):
            self.assertIn(expected, html)
        self.assertNotIn("<pre>{'", html)
        for forbidden in ("Probability", "勝率", "上漲機率", "下跌機率", "買進訊號", "賣出訊號", "保證獲利"):
            self.assertNotIn(forbidden, html)


if __name__ == "__main__":
    unittest.main()
