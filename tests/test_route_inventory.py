import unittest

import app as stock_app


EXPECTED_ROUTES = {
    ("/", "dashboard_page", frozenset({"GET"})),
    ("/account", "account_page", frozenset({"GET"})),
    ("/account/watchlist", "account_watchlist_page", frozenset({"GET"})),
    ("/api/account/state", "account_state", frozenset({"GET"})),
    ("/api/account/watchlist", "account_watchlist_api", frozenset({"POST"})),
    ("/api/conversation", "conversation_api", frozenset({"POST"})),
    ("/api/dashboard", "dashboard_api", frozenset({"GET"})),
    ("/api/market-insights", "market_insights_api", frozenset({"GET"})),
    ("/broadcast_weekly", "broadcast_weekly", frozenset({"GET"})),
    ("/auth/line/callback", "line_callback", frozenset({"GET"})),
    ("/auth/line/login", "line_login", frozenset({"GET"})),
    ("/auth/logout", "auth_logout", frozenset({"POST"})),
    ("/callback", "callback", frozenset({"POST"})),
    ("/dashboard", "dashboard_page", frozenset({"GET"})),
    ("/health", "healthz", frozenset({"GET"})),
    ("/healthz", "healthz", frozenset({"GET"})),
    ("/market", "market_page", frozenset({"GET"})),
    ("/market-map", "market_map_page", frozenset({"GET"})),
    ("/industries", "industries_page", frozenset({"GET"})),
    ("/stocks", "stocks_page", frozenset({"GET"})),
    ("/ask", "ask_page", frozenset({"GET"})),
    ("/learn", "learn_page", frozenset({"GET"})),
    ("/preview/report", "preview_report_page", frozenset({"GET"})),
    ("/reports", "reports_page", frozenset({"GET"})),
    ("/reports/<report_date>", "report_page", frozenset({"GET"})),
    ("/reports/<report_date>/download", "report_download", frozenset({"GET"})),
    ("/reports/<report_date>/preview", "report_preview", frozenset({"GET"})),
    ("/reports/<trading_date>/pre-market", "pre_market_report_page", frozenset({"GET"})),
    ("/reports/<trading_date>/post-close", "post_close_report_page", frozenset({"GET"})),
    ("/reports/sample/download", "sample_report_download", frozenset({"GET"})),
    ("/reports/trading-day/<trading_date>", "trading_day_report_page", frozenset({"GET"})),
    ("/reports/us", "us_reports_page", frozenset({"GET"})),
    ("/reports/us/<trading_date>/post-close", "us_post_close_report_page", frozenset({"GET"})),
    ("/reports/us/<trading_date>/pre-market", "us_pre_market_report_page", frozenset({"GET"})),
    ("/reports/us/trading-day/<trading_date>", "us_trading_day_report_page", frozenset({"GET"})),
    ("/reports/weekly/<week_id>", "weekly_report_page", frozenset({"GET"})),
    ("/search", "search_page", frozenset({"GET"})),
    ("/stock/<code>", "stock_page", frozenset({"GET"})),
    ("/tasks/check-alerts", "check_alerts_task", frozenset({"POST"})),
    ("/tasks/refresh-sector-signals", "refresh_sector_signals_task", frozenset({"POST"})),
    ("/watchlist", "watchlist_page", frozenset({"GET"})),
}


def route_inventory(flask_app):
    return {
        (
            rule.rule,
            rule.endpoint,
            frozenset(rule.methods - {"HEAD", "OPTIONS"}),
        )
        for rule in flask_app.url_map.iter_rules()
        if rule.endpoint != "static"
    }


class RouteInventoryTests(unittest.TestCase):
    def test_public_routes_keep_rules_endpoints_and_methods(self):
        self.assertEqual(route_inventory(stock_app.app), EXPECTED_ROUTES)


if __name__ == "__main__":
    unittest.main()
