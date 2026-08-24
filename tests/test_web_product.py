import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")

import app as stock_app

from tests.test_observation_public_surfaces import (
    observation_dashboard,
    quant_snapshot,
)


class WebProductTests(unittest.TestCase):
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_information_architecture_has_distinct_server_rendered_pages(self, load):
        load.return_value = observation_dashboard()
        client = stock_app.app.test_client()
        expectations = {
            "/": "台股市場研究摘要",
            "/market": "市場實況",
            "/industries": "產業觀察",
            "/stocks": "個股與 ETF",
            "/ask": "ASK ABSORB",
            "/learn": "市場觀察小辭典",
        }

        for path, heading in expectations.items():
            with self.subTest(path=path):
                response = client.get(path)
                html = response.get_data(as_text=True)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(html.count("<h1"), 1)
                self.assertIn(heading, html)

        home = client.get("/").get_data(as_text=True)
        self.assertNotIn('id="industry-observations"', home)
        self.assertNotIn('id="stock-events"', home)
        self.assertIn('href="/industries"', home)
        self.assertIn('href="/stocks"', home)
        self.assertIn('href="/ask"', home)
        self.assertIn('href="/learn"', home)

    def test_legacy_market_map_redirects_to_industries(self):
        response = stock_app.app.test_client().get("/market-map")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/industries"))

    def test_dashboard_starts_with_today_market_preparation_cards(self):
        response = stock_app.app.test_client().get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("今日市場準備", html)
        self.assertIn("盤後觀察", html)
        self.assertIn("盤前風險更新", html)

    def test_every_papi_theme_has_at_least_five_companies(self):
        self.assertTrue(
            all(
                len(names) >= 5
                for names in stock_app.PAPI_THEME_SECTORS.values()
            )
        )

    def test_build_market_heatmap_orders_strongest_first_for_preview(self):
        cards = [
            {
                "name": "弱勢",
                "count": 1,
                "score": 42,
                "leader": {"code": "1101", "prob": 42},
            },
            {
                "name": "強勢",
                "count": 2,
                "score": 68,
                "leader": {"code": "2330", "prob": 68},
            },
        ]

        result = stock_app.build_market_heatmap(cards)

        self.assertEqual([item["name"] for item in result], ["強勢", "弱勢"])
        self.assertEqual(result[0]["tone"], "hot")
        self.assertEqual(result[1]["tone"], "cold")

    def test_find_industry_peers_excludes_current_stock(self):
        market_map = {
            "全市場": ["2330", "2454", "2303"],
            "半導體": ["2330", "2454", "2303"],
        }

        peers = stock_app.find_industry_peers("2330", market_map, limit=2)

        self.assertEqual(
            peers, {"category": "半導體", "codes": ["2454", "2303"]}
        )

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_industries_page_renders_verified_actual_observations(self, load):
        load.return_value = observation_dashboard()

        response = stock_app.app.test_client().get("/industries")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "產業實際強弱",
            "近 5 日相對大盤報酬",
            "產業觀察",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "推薦",
            "回測",
            "勝率",
        ):
            self.assertNotIn(forbidden, html)

    def test_root_renders_dashboard_and_search_redirects_known_stock(self):
        client = stock_app.app.test_client()

        root = client.get("/")
        with patch.object(
            stock_app,
            "search_stock_code",
            side_effect=[("2330", "台積電"), (None, None)],
        ):
            found = client.get("/search?q=台積電")
            missing = client.get(
                "/search?q=不存在股票", follow_redirects=True
            )

        self.assertEqual(root.status_code, 200)
        self.assertIn("ABSORB", root.get_data(as_text=True))
        self.assertEqual(found.status_code, 302)
        self.assertTrue(found.headers["Location"].endswith("/stock/2330"))
        self.assertIn("找不到", missing.get_data(as_text=True))

    def test_empty_search_stays_on_dashboard_with_clear_error(self):
        response = stock_app.app.test_client().get(
            "/search?q=", follow_redirects=True
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("找不到", response.get_data(as_text=True))

    def test_base_shell_uses_absorb_brand_and_light_theme(self):
        response = stock_app.app.test_client().get("/dashboard")
        html = response.get_data(as_text=True)
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("ABSORB", html)
        self.assertIn('class="brand-wordmark"', html)
        self.assertIn('data-brand-wordmark', html)
        self.assertIn('aria-label="回到 ABSORB 主畫面"', html)
        self.assertNotIn('class="brand-mark"', html)
        self.assertIn("今天市場", html)
        self.assertIn("使用 LINE 登入", html)
        self.assertIn("已驗證市場觀察", html)
        self.assertIn('data-market-switch', html)
        self.assertIn('href="/us"', html)
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertIn("--absorb-navy:#122643", css)
        self.assertIn("--absorb-canvas:#f7f9fc", css)
        self.assertIn('"Avenir Next",Avenir,"Noto Sans TC"', css)
        self.assertIn(".research-command", css)
        self.assertIn(".market-switch a{display:grid;min-height:44px", css)
        self.assertIn(".quick-ask-backdrop[hidden]{display:none}", css)
        self.assertIn(".quick-ask-header button{display:grid;width:44px;height:44px", css)
        self.assertIn("--command-content-max:1760px", css)
        self.assertIn("--command-muted:#536575", css)
        self.assertIn(".evidence-canvas{", css)
        self.assertIn("background:var(--command-accent-surface)", css)
        self.assertIn(".brand-wordmark:hover{", css)
        self.assertIn("rotate(-1.5deg)", css)
        self.assertIn("@view-transition{navigation:auto}", css)
        self.assertIn("button,input,select,textarea{font:inherit}", css)
        self.assertIn(".button{display:inline-flex", css)
        self.assertIn(".command-metrics{grid-column:1/-1;grid-template-columns:repeat(4,minmax(0,1fr))", css)
        self.assertNotIn("border-left:4px", css.replace(" ", ""))
        self.assertNotIn("border-top:3px", css.replace(" ", ""))
        version = re.search(r'/static/app\.css\?v=([0-9a-f]{12})', html)
        self.assertIsNotNone(version)
        self.assertIn(f'/static/app.js?v={version.group(1)}', html)
        asset = stock_app.app.test_client().get(
            f'/static/app.js?v={version.group(1)}'
        )
        self.assertIn("immutable", asset.headers["Cache-Control"])
        asset.close()

    def test_us_product_navigation_and_stock_detail_keep_us_context(self):
        client = stock_app.app.test_client()
        us_stocks = client.get("/us/stocks")

        self.assertEqual(us_stocks.status_code, 200)
        html = us_stocks.get_data(as_text=True)
        self.assertIn('data-market="US"', html)
        for href in (
            'href="/us"',
            'href="/us/market"',
            'href="/us/industries"',
            'href="/us/stocks"',
            'href="/reports/us"',
        ):
            with self.subTest(href=href):
                self.assertIn(href, html)

        with patch.object(stock_app, "build_stock_observation", return_value={
            "code": "AAPL", "name": "Apple Inc.", "market": "US",
            "observation_kind": "status", "status_label": "資料不足",
            "observation_as_of": "2026-07-15", "evidence_sha256": "a" * 64,
        }):
            detail = client.get("/stock/AAPL")

        self.assertEqual(detail.status_code, 200)
        self.assertIn('data-market="US"', detail.get_data(as_text=True))

    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_market_page_accepts_production_data_quality_fields(self, load):
        snapshot = observation_dashboard()
        snapshot["data_quality"] = {
            "coverage": 0.998,
            "available_count": 2072,
            "failure_count": 4,
            "universe_count": 2076,
        }
        load.return_value = snapshot

        response = stock_app.app.test_client().get("/market")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("有效標的</dt><dd>2072</dd>", html)
        self.assertNotIn("有效標的</dt><dd>資料不足</dd>", html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_web_security_headers_and_pinned_chart_supply_chain(
        self, fetch
    ):
        response = stock_app.app.test_client().get("/dashboard")
        csp = response.headers["Content-Security-Policy"]
        fetch.return_value = quant_snapshot()
        stock_html = stock_app.app.test_client().get(
            "/stock/2330"
        ).get_data(as_text=True)

        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("form-action 'self'", csp)
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("lightweight-charts@4.2.2", stock_html)
        self.assertIn('integrity="sha384-', stock_html)
        self.assertNotIn("style=", stock_html)

    def test_dashboard_page_is_the_observation_dashboard(self):
        with patch.object(stock_app, "analyze") as analyze, patch.object(
            stock_app,
            "_published_dashboard_snapshot",
            return_value=observation_dashboard(),
        ):
            response = stock_app.app.test_client().get("/dashboard")

        self.assertEqual(response.status_code, 200)
        analyze.assert_not_called()
        html = response.get_data(as_text=True)
        for label in (
            "台股市場研究摘要",
            "市場指揮台",
            "市場廣度",
            "期間報酬",
            "波動與風險",
            "產業相對強度",
            "資料覆蓋",
            "資料基準日 2026-07-15",
            "今日焦點",
            "產業觀察",
            "市場實況",
            "個股與 ETF",
            "Ask ABSORB",
            "AI 預測研究中",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "精選標的",
            "產業預測",
            "data-top-picks",
        ):
            self.assertNotIn(forbidden, html)

    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=observation_dashboard()
    )
    def test_dashboard_research_command_uses_verified_data_without_inline_styles(
        self, _load
    ):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)

        for marker in (
            'data-research-summary',
            'data-market-command',
            'data-breadth-visual',
            'data-sector-flow',
            '<progress',
            'value="61.2"',
            '1200',
            '700',
            '17.5%',
            '半導體',
            '+1.85%',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)
        self.assertNotIn("23,742.18", html)
        self.assertNotIn("style=", html)

    def test_quick_ask_reuses_conversation_contract_and_has_dialog_controls(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        for marker in (
            'data-quick-ask-open',
            'data-quick-ask-dialog',
            'role="dialog"',
            'aria-modal="true"',
            'data-conversation-endpoint="/api/conversation"',
            'maxlength="1200"',
            'aria-live="polite"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)
        for marker in (
            "dataQuickAskOpen",
            'event.key === "Escape"',
            'event.key === "Tab"',
            'event.key.toLowerCase() === "k"',
            "querySelectorAll(\"[data-conversation-form]\")",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, script)
        self.assertNotIn(".innerHTML", script)

    def test_public_pages_only_request_private_account_state_when_session_cookie_exists(self):
        html = stock_app.app.test_client().get("/dashboard").get_data(as_text=True)
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('data-account-session="anonymous"', html)
        self.assertIn('document.body.dataset.accountSession !== "present"', script)

    def test_report_filters_and_retry_control_have_progressive_enhancement(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )
        unavailable = Path("templates/report_unavailable.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("[data-report-filter]", script)
        self.assertIn("[data-report-type]", script)
        self.assertIn("data-report-retry", unavailable)
        self.assertIn("window.location.reload()", script)

    def test_dashboard_has_route_based_section_navigation(self):
        html = stock_app.app.test_client().get(
            "/dashboard"
        ).get_data(as_text=True)

        for marker in (
            'href="/market"',
            'href="/industries"',
            'href="/stocks"',
            'href="/ask"',
            'href="/learn"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)

    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=observation_dashboard()
    )
    def test_navigation_has_route_active_state_and_no_primary_hash_links(self, _load):
        client = stock_app.app.test_client()
        for path, label in (
            ("/dashboard", "今天市場"),
            ("/market", "市場實況"),
            ("/industries", "產業觀察"),
            ("/stocks", "個股與 ETF"),
            ("/reports", "每日報告"),
        ):
            with self.subTest(path=path):
                html = client.get(path).get_data(as_text=True)
                self.assertIn(
                    f'class="nav-link active" href="{path if path != "/dashboard" else "/"}" aria-current="page">{label}',
                    html,
                )
        home = client.get("/dashboard").get_data(as_text=True)
        primary_nav = home.split('<nav class="nav-list">', 1)[1].split("</nav>", 1)[0]
        mobile_nav = home.split('<nav class="mobile-nav"', 1)[1].split("</nav>", 1)[0]
        self.assertNotIn('href="#', primary_nav)
        self.assertNotIn('href="#', mobile_nav)
        self.assertIn('aria-current="page">今天</a>', mobile_nav)
        ask = client.get("/ask").get_data(as_text=True)
        self.assertIn('<h1>ASK ABSORB</h1>', ask)

    def test_legacy_hash_migrator_uses_only_fixed_canonical_routes(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        expected = {
            '"#market-pulse": "/market"',
            '"#market-heatmap": "/industries"',
            '"#industry-observations": "/industries"',
            '"#stock-search": "/stocks"',
            '"#stock-events": "/stocks"',
            '"#etf-observations": "/stocks?tab=etf"',
            '"#learn": "/learn"',
            '"#daily-focus": "/"',
        }
        for mapping in expected:
            self.assertIn(mapping, script)
        self.assertIn(
            "if(!['/','/dashboard'].includes",
            script.replace('"', "'").replace(" ", ""),
        )
        self.assertNotIn("window.location.hash.slice", script)

    @patch.object(stock_app, "analyze")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_dashboard_api_returns_verified_observation_without_analysis(
        self, load_snapshot, analyze
    ):
        load_snapshot.return_value = observation_dashboard()

        response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        analyze.assert_not_called()
        self.assertEqual(payload["product_mode"], "observation")
        self.assertEqual(payload["observation_as_of"], "2026-07-15")
        self.assertEqual(
            payload["market_observation"]["advancing_count"], 1200
        )
        self.assertEqual(
            payload["industry_observations"][0]["name"], "半導體"
        )
        self.assertEqual(payload["prediction_status"], "AI 預測研究中")
        self.assertNotIn("top_picks", payload)
        self.assertNotIn("opportunities", payload)

    @patch.object(stock_app, "analyze")
    @patch.object(
        stock_app, "_published_dashboard_snapshot", return_value=None
    )
    def test_dashboard_api_fails_closed_without_snapshot(
        self, _load_snapshot, analyze
    ):
        response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 503)
        analyze.assert_not_called()
        self.assertEqual(
            response.get_json()["status"], "observation_unavailable"
        )

    def test_preview_report_is_not_public_without_preview_prefix(self):
        response = stock_app.app.test_client().get("/preview/report")

        self.assertEqual(response.status_code, 404)

    @patch.object(stock_app, "analyze")
    @patch.object(stock_app, "_published_dashboard_snapshot")
    def test_preview_dashboard_keeps_isolated_candidate_api(
        self, load_snapshot, analyze
    ):
        analyze.return_value = {
            "price": 23150.0,
            "prob": 58,
            "trend": "多頭",
            "as_of": "2026-07-15",
            "recommendation": {},
        }
        load_snapshot.return_value = {
            "baseline_status": "initial_backtest_bootstrap",
            "inference_as_of": "2026-07-15",
            "backtest_as_of": None,
            "model_version": "lgbm-5d-v1",
            "backtest_version": None,
            "feature_schema_version": 1,
            "recommendation_policy_version": "recommendation-v1",
            "presentation": {
                "model_output_label": "模型方向分數",
                "strong_action_allowed": False,
                "performance_endorsement_allowed": False,
            },
            "sector_snapshot": {
                "sectors": {
                    "網通設備": [
                        {
                            "code": "4906",
                            "name": "正文",
                            "prob": 73.7,
                            "trend": "跌破 MA20",
                            "as_of": "2026-07-15",
                        }
                    ]
                }
            },
            "heatmap": [{"name": "網通設備", "tone": "steady"}],
            "daily_focus": ["candidate focus"],
            "top_picks": [{"code": "4906", "name": "正文"}],
        }
        with patch.object(
            stock_app, "PREVIEW_CANDIDATE_PREFIX", "previews/demo"
        ), patch.object(stock_app, "cached_opportunities", return_value=[]):
            response = stock_app.app.test_client().get("/api/dashboard")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["inference_as_of"], "2026-07-15")
        self.assertEqual(payload["sector_cards"][0]["leader"]["code"], "4906")
        self.assertEqual(payload["daily_focus"], ["candidate focus"])

    @patch.object(
        stock_app,
        "find_industry_peers",
        return_value={"category": "半導體", "codes": ["2454"]},
    )
    @patch.object(stock_app, "get_stock_name", return_value="聯發科")
    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_is_the_observation_workspace(
        self, fetch, _name, _peers
    ):
        fetch.return_value = quant_snapshot()

        response = stock_app.app.test_client().get("/stock/2330")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "個股觀察摘要",
            "價格與均線",
            "籌碼觀察",
            "技術指標",
            "風險事件",
            "欄位怎麼看",
            "產業同儕",
            "聯發科",
        ):
            self.assertIn(label, html)
        for forbidden in (
            "五日上漲機率",
            "投資金額試算",
            "支持這項建議",
            "回測",
            "勝率",
        ):
            self.assertNotIn(forbidden, html)
        self.assertIn("data-watchlist-toggle", html)
        self.assertIn("data-chart-range", html)
        self.assertIn('aria-label="個股觀察導覽"', html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_does_not_render_untrusted_snapshot_news(self, fetch):
        snapshot = quant_snapshot()
        snapshot["news"] = [
            {
                "title": "不安全來源",
                "link": "javascript:alert(1)",
            }
        ]
        fetch.return_value = snapshot

        html = stock_app.app.test_client().get(
            "/stock/2330"
        ).get_data(as_text=True)

        self.assertNotIn("不安全來源", html)
        self.assertNotIn('href="javascript:', html)

    @patch.object(stock_app, "fetch_published_quant_snapshot")
    def test_stock_page_accepts_standard_us_ticker(self, fetch):
        fetch.return_value = quant_snapshot("AAPL", market="US")

        response = stock_app.app.test_client().get("/stock/AAPL")

        self.assertEqual(response.status_code, 200)
        fetch.assert_called_once_with("AAPL")

    def test_dashboard_script_does_not_insert_api_text_with_inner_html(self):
        script = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(".innerHTML", script)
        self.assertIn("AbortController", script)
        self.assertNotIn('title: "五日預測"', script)

    def test_web_is_observation_only_and_old_watchlist_redirects(self):
        response = stock_app.app.test_client().get("/watchlist")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/dashboard"))

    @patch.object(stock_app, "analyze")
    def test_stock_summary_api_removed_with_browser_watchlist(self, analyze):
        response = stock_app.app.test_client().get(
            "/api/stock/2330/summary"
        )

        self.assertEqual(response.status_code, 404)
        analyze.assert_not_called()

    def test_line_navigation_maps_six_observation_entries(self):
        navigation = stock_app.build_line_navigation_flex(
            "https://example.com/"
        )

        self.assertEqual(navigation["type"], "carousel")
        self.assertEqual(len(navigation["contents"]), 6)
        actual_uri = {}
        actual_message = {}
        for card in navigation["contents"]:
            action = card["footer"]["contents"][0]["action"]
            title = card["body"]["contents"][0]["text"]
            if action["type"] == "uri":
                actual_uri[title] = action["uri"]
            else:
                actual_message[title] = action["text"]
        self.assertEqual(
            actual_uri,
            {
                "看大盤": "https://example.com/market",
                "看產業": "https://example.com/industries",
                "市場觀察": "https://example.com/dashboard",
            },
        )
        self.assertEqual(
            actual_message,
            {
                "查自選": "我的關注",
                "設提醒": "提醒管理",
                "查股票": "2330",
            },
        )

    def test_rich_menu_source_matches_observation_navigation(self):
        svg = Path("assets/rich-menu.svg").read_text(encoding="utf-8")

        for label in (
            "看大盤",
            "看產業",
            "查自選",
            "設提醒",
            "查股票",
            "市場觀察",
        ):
            self.assertIn(label, svg)
        for removed in (
            "找機會",
            "算報酬",
            "深度分析",
            "熱門題材與排行",
            "投入金額快速試算",
            "圖表、回測、新聞",
        ):
            self.assertNotIn(removed, svg)
        for marker in ("ABSORB", "#122643", "#ffffff", "#eaf0f7"):
            self.assertIn(marker, svg)

    def test_line_summary_card_has_one_clear_cta(self):
        card = stock_app.build_line_summary_card(
            "市場觀察",
            ["2330 台積電", "最新收盤 1000.00"],
            "查看完整觀察",
            "https://example.com/stock/2330",
        )

        self.assertEqual(len(card["footer"]["contents"]), 1)
        self.assertEqual(
            card["footer"]["contents"][0]["action"]["uri"],
            "https://example.com/stock/2330",
        )

    def test_web_shell_supports_keyboard_and_mobile_interactions(self):
        response = stock_app.app.test_client().get("/dashboard")
        html = response.get_data(as_text=True)
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )

        for marker in (
            'class="skip-link"',
            'id="main-content"',
            'aria-live="polite"',
        ):
            self.assertIn(marker, html)
        for rule in (
            ":focus-visible",
            "prefers-reduced-motion",
            "min-height:44px",
        ):
            self.assertIn(rule, css)
        self.assertIn("grid-template-columns:repeat(5,1fr)", css)
        self.assertIn('href="/reports">每日報告</a>', html)

    def test_browser_bundle_has_no_local_watchlist_storage(self):
        source = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        for removed in (
            "localStorage",
            "quant-watchlist",
            "data-alert-open",
            "data-alert-form",
        ):
            self.assertNotIn(removed, source)
        self.assertIn("if (!entries.length) return", source)

    def test_health_check_is_separate_from_dashboard(self):
        client = stock_app.app.test_client()

        for path in ("/health", "/healthz"):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_data(as_text=True), "ok")

    def test_stock_chart_is_clipped_and_resizes_with_its_panel(self):
        css = Path(stock_app.app.static_folder, "app.css").read_text(
            encoding="utf-8"
        )
        js = Path(stock_app.app.static_folder, "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn(".chart-shell{overflow:hidden", css)
        self.assertIn(".stock-chart{", css)
        self.assertIn("min-height:320px", css)
        self.assertIn("function measureChartHeight", js)
        self.assertIn("Math.min(460", js)
        self.assertIn("ResizeObserver", js)


if __name__ == "__main__":
    unittest.main()
