import pathlib
import unittest

from jinja2 import DictLoader, Environment


class ReportsTemplateTests(unittest.TestCase):
    def _render(self, *, reports_v2, reports, unavailable=False):
        template_text = pathlib.Path("templates/reports.html").read_text(encoding="utf-8")
        env = Environment(
            loader=DictLoader(
                {
                    "reports.html": template_text,
                    "base.html": "{% block title %}{% endblock %}{% block nav_reports %}{% endblock %}{% block content %}{% endblock %}",
                }
            )
        )
        env.globals["url_for"] = lambda endpoint, **values: f"/{endpoint}"
        return env.get_template("reports.html").render(
            reports_v2=reports_v2,
            reports=reports,
            unavailable=unavailable,
        )

    def test_v2_reports_do_not_render_false_empty_state(self):
        output = self._render(
            reports_v2=[
                {
                    "report_type": "post_close",
                    "title": "盤後研究報告",
                    "summary": ["摘要"],
                    "source_market_date": "2026-07-17",
                    "applicable_trading_date": "2026-07-20",
                }
            ],
            reports=[],
        )
        self.assertIn("盤後研究報告", output)
        self.assertNotIn("目前沒有可用的每日報告", output)

    def test_v2_reports_are_grouped_into_filterable_premarket_and_postclose_lanes(self):
        output = self._render(
            reports_v2=[
                {
                    "report_type": "pre_market",
                    "title": "盤前研究報告",
                    "summary": ["盤前摘要"],
                    "source_market_date": "2026-07-17",
                    "applicable_trading_date": "2026-07-20",
                },
                {
                    "report_type": "post_close",
                    "title": "盤後研究報告",
                    "summary": ["盤後摘要"],
                    "source_market_date": "2026-07-17",
                    "applicable_trading_date": "2026-07-20",
                },
            ],
            reports=[],
        )

        self.assertIn('data-report-filters', output)
        self.assertIn('data-report-filter="all"', output)
        self.assertIn('data-report-filter="pre_market"', output)
        self.assertIn('data-report-filter="post_close"', output)
        self.assertEqual(output.count('class="report-lane"'), 2)
        self.assertIn('data-report-type="pre_market"', output)
        self.assertIn('data-report-type="post_close"', output)
        self.assertIn("盤前研究報告", output)
        self.assertIn("盤後研究報告", output)

    def test_empty_state_only_when_both_collections_are_empty(self):
        output = self._render(reports_v2=[], reports=[])
        self.assertIn("目前沒有可用的每日報告", output)

    def test_unavailable_empty_state_uses_service_error_copy(self):
        output = self._render(reports_v2=[], reports=[], unavailable=True)
        self.assertIn("報告服務暫時無法讀取", output)
        self.assertNotIn("本地量化流程完成發布後", output)


if __name__ == "__main__":
    unittest.main()
