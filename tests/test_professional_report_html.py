import pathlib
import unittest

from jinja2 import DictLoader, Environment

from reporting.professional_builder import build_professional_post_close_artifact
from reporting.professional_html import build_professional_report_view


class ProfessionalReportHtmlTests(unittest.TestCase):
    def _metadata(self):
        return {
            "schema_version": 2,
            "report_type": "post_close",
            "product_mode": "observation",
            "market": "TW",
            "source_market_date": "2026-07-17",
            "applicable_trading_date": "2026-07-20",
            "published_at": "2026-07-17T10:30:00Z",
            "source_manifest": "quant/v1/manifests/TW-20260717T091000Z-123456789abc.json",
            "source_manifest_sha256": "a" * 64,
            "prediction_capability": {
                "mode": "research",
                "observation_enabled": True,
                "probability_allowed": False,
                "ranking_allowed": False,
                "strong_action_allowed": False,
                "performance_endorsement_allowed": False,
            },
            "content": {
                "market_observation": {
                    "return_1d_pct": -0.72,
                    "advancing_count": 520,
                    "declining_count": 812,
                    "ma20_breadth_pct": 39.7,
                    "realized_volatility_20d_pct": 18.2,
                },
                "industry_observations": [
                    {"name": "半導體製造", "available_count": 6, "component_count": 6, "relative_return_5d_pct": 4.31},
                    {"name": "航運", "available_count": 8, "component_count": 9, "relative_return_5d_pct": -3.20},
                ],
                "heatmap": [],
                "stock_events": [],
                "trading_status_observations": [
                    {
                        "symbol": "2303",
                        "name": "測試股票 2303",
                        "status": "official_no_regular_trade",
                        "label": "當日無正常交易",
                        "observation_as_of": "2026-07-17",
                        "latest_regular_price_date": "2026-07-16",
                        "evidence_sha256": "c" * 64,
                        "last_regular_close": 100.0,
                    }
                ],
                "etf_observations": [],
                "daily_focus": ["市場廣度降至四成以下"],
                "data_quality": {"coverage": 0.982, "symbol_count": 1332, "failure_count": 24},
            },
        }

    def _view(self):
        report = build_professional_post_close_artifact(
            self._metadata(), code_commit_sha="b" * 40
        )
        return build_professional_report_view(
            report, pdf_download_url="/reports/2026-07-17/post-close/download"
        )

    def test_view_model_does_not_expose_internal_manifest_path(self):
        view = self._view()
        self.assertNotIn("source_manifest", view["identity"])
        self.assertEqual(view["identity"]["source_manifest_sha256_short"], "aaaaaaaaaaaa")
        self.assertEqual(view["pdf_download_url"], "/reports/2026-07-17/post-close/download")

    def test_template_renders_single_h1_and_professional_sections(self):
        template_text = pathlib.Path(
            "templates/reports/post_close_professional.html"
        ).read_text(encoding="utf-8")
        env = Environment(
            loader=DictLoader(
                {
                    "reports/post_close_professional.html": template_text,
                    "base.html": "{% block title %}{% endblock %}{% block nav_reports %}{% endblock %}{% block content %}{% endblock %}",
                }
            )
        )
        output = env.get_template("reports/post_close_professional.html").render(
            report=self._view()
        )
        self.assertEqual(output.count("<h1"), 1)
        for anchor in (
            "executive-summary",
            "market-analysis",
            "capital-flows",
            "industry-analysis",
            "security-analysis",
            "quantitative-research",
            "model-validation",
            "next-session",
            "data-governance",
            "ai-reference",
        ):
            self.assertIn(f'id="{anchor}"', output)
        self.assertIn("下載完整 PDF", output)
        self.assertIn("法人流向尚未納入", output)
        self.assertIn("當日無正常交易", output)
        self.assertIn("最後正常交易收盤 100.00（2026-07-16）", output)
        self.assertNotIn("quant/v1/manifests/", output)

    def test_etf_observations_render_verified_fields_and_safe_fallback(self):
        template_text = pathlib.Path(
            "templates/reports/post_close_professional.html"
        ).read_text(encoding="utf-8")
        env = Environment(
            loader=DictLoader(
                {
                    "reports/post_close_professional.html": template_text,
                    "base.html": "{% block title %}{% endblock %}{% block nav_reports %}{% endblock %}{% block content %}{% endblock %}",
                }
            )
        )
        metadata = self._metadata()
        metadata["content"]["etf_observations"] = [
            {
                "symbol": "0050",
                "name": "元大台灣50",
                "price": 106.4,
                "return_1d_pct": -0.28,
                "return_5d_pct": 3.45,
                "volume_ratio": 0.4,
                "trend_observation": "above_ma20",
                "as_of": "2026-07-17",
            },
            {
                "symbol": "0056",
                "name": "元大高股息",
                "price": None,
                "return_1d_pct": None,
                "return_5d_pct": None,
                "volume_ratio": None,
                "trend_observation": None,
                "as_of": "2026-07-17",
            },
        ]
        report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        view = build_professional_report_view(report)
        output = env.get_template("reports/post_close_professional.html").render(
            report=view
        )
        # Verify 0050 full fields
        self.assertIn("元大台灣50", output)
        self.assertIn("0050", output)
        self.assertIn("106.40", output)
        self.assertIn("-0.28%", output)
        self.assertIn("+3.45%", output)
        self.assertIn("站上 MA20", output)
        self.assertIn("0.40", output)

        # Verify 0056 fallback
        self.assertIn("元大高股息", output)
        self.assertIn("0056", output)

    def test_scenario_empty_and_non_empty_states(self):
        template_text = pathlib.Path(
            "templates/reports/post_close_professional.html"
        ).read_text(encoding="utf-8")
        env = Environment(
            loader=DictLoader(
                {
                    "reports/post_close_professional.html": template_text,
                    "base.html": "{% block title %}{% endblock %}{% block nav_reports %}{% endblock %}{% block content %}{% endblock %}",
                }
            )
        )
        view = self._view()
        # View generated with breadth=39.7 has negative scenario populated, positive scenario empty
        self.assertEqual(len(view["next_session"]["data"]["positive"]), 0)
        self.assertGreater(len(view["next_session"]["data"]["negative"]), 0)

        output = env.get_template("reports/post_close_professional.html").render(
            report=view
        )
        self.assertIn("目前沒有符合此情境的已驗證條件", output)
        self.assertIn("站上 MA20 比例 <= 40%", output)

        # Invert: positive populated, negative empty
        metadata = self._metadata()
        metadata["content"]["market_observation"]["ma20_breadth_pct"] = 72.0
        metadata["content"]["market_observation"]["realized_volatility_20d_pct"] = 12.0
        report = build_professional_post_close_artifact(
            metadata, code_commit_sha="b" * 40
        )
        view_pos = build_professional_report_view(report)
        self.assertGreater(len(view_pos["next_session"]["data"]["positive"]), 0)
        self.assertEqual(len(view_pos["next_session"]["data"]["negative"]), 0)

        output_pos = env.get_template("reports/post_close_professional.html").render(
            report=view_pos
        )
        self.assertIn("站上 MA20 比例 >= 60%", output_pos)
        self.assertIn("目前沒有符合此情境的已驗證條件", output_pos)


if __name__ == "__main__":
    unittest.main()
