import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")

import app as stock_app

from absorb.conversation.schemas import ConversationAnswer


class AbsorbConversationWebTests(unittest.TestCase):
    @staticmethod
    def _payload(question="台積電如何？", market="TW", page="home"):
        return {"question": question, "market": market, "page": page}

    def test_web_conversation_is_json_only_private_and_cookie_is_httponly(self):
        client = stock_app.app.test_client()
        with patch.object(
            stock_app,
            "run_absorb_conversation",
            return_value=ConversationAnswer("結論：等待確認", data_quality="partial"),
        ) as converse:
            response = client.post("/api/conversation", json=self._payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "結論：等待確認")
        self.assertEqual(response.headers["Cache-Control"], "private, no-store, max-age=0")
        self.assertIn("HttpOnly", response.headers["Set-Cookie"])
        kwargs = converse.call_args.kwargs
        self.assertTrue(kwargs["principal"].startswith("web:"))
        self.assertEqual(kwargs["access"], "public")
        self.assertEqual(kwargs["market_context"], "TW")
        self.assertEqual(kwargs["page_context"], "home")

    def test_authenticated_web_conversation_gets_server_side_action_executor(self):
        client = stock_app.app.test_client()
        with (
            patch.object(
                stock_app,
                "_web_conversation_identity",
                return_value=("line:U0123456789abcdef0123456789abcdef", "authenticated"),
            ),
            patch.object(stock_app, "_line_conversation_action_executor", return_value="executor") as factory,
            patch.object(stock_app, "run_absorb_conversation", return_value=ConversationAnswer("ok")) as converse,
        ):
            response = client.post("/api/conversation", json=self._payload("確認操作"))

        self.assertEqual(response.status_code, 200)
        factory.assert_called_once_with("U0123456789abcdef0123456789abcdef")
        self.assertEqual(converse.call_args.kwargs["action_executor"], "executor")

    def test_web_conversation_rejects_extra_fields_and_non_json(self):
        client = stock_app.app.test_client()
        self.assertEqual(client.post("/api/conversation", data="x").status_code, 415)
        self.assertEqual(
            client.post("/api/conversation", json={"question": "x", "user_id": "victim"}).status_code,
            400,
        )

    def test_web_conversation_accepts_allowlisted_learn_context(self):
        client = stock_app.app.test_client()
        with patch.object(
            stock_app,
            "run_absorb_conversation",
            return_value=ConversationAnswer("ok"),
        ) as converse:
            response = client.post(
                "/api/conversation", json=self._payload(page="learn")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(converse.call_args.kwargs["page_context"], "learn")
        self.assertEqual(
            client.post("/api/conversation", json=self._payload(market="EU")).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/api/conversation", json=self._payload(page="admin")
            ).status_code,
            400,
        )

    def test_us_market_context_answers_generic_question_from_us_report(self):
        report = {
            "market": "US",
            "summary": ["S&P 500 收高，市場廣度改善"],
            "source_market_date": "2026-08-21",
            "data_quality": "available",
        }
        with patch.object(
            stock_app, "_conversation_search_stock", return_value=(None, None)
        ), patch.object(
            stock_app, "_conversation_report_lookup", return_value=report
        ) as lookup:
            answer = stock_app._observation_conversation(
                question="今天市場如何？",
                access="public",
                market_context="US",
                page_context="market",
            )

        lookup.assert_called_once_with("post_close", market="US")
        self.assertIn("美股市場實況", answer.text)
        self.assertEqual(answer.data_as_of, "2026-08-21")

    def test_explicit_tw_symbol_overrides_us_page_context(self):
        observation = {
            "code": "2330",
            "name": "台積電",
            "price": 1200.0,
            "trend_observation": "above_ma20_ma60",
            "rsi": 58.0,
            "volume_ratio": 1.2,
            "risk_events": [],
            "as_of": "2026-08-21",
        }
        with patch.object(
            stock_app, "_conversation_search_stock", return_value=("2330", "台積電")
        ), patch.object(
            stock_app, "fetch_published_quant_snapshot", return_value={"market": "TW"}
        ) as fetch, patch.object(
            stock_app, "build_stock_observation", return_value=observation
        ), patch.object(stock_app, "_conversation_report_lookup") as report_lookup:
            answer = stock_app._observation_conversation(
                question="2330 現在如何？",
                access="public",
                market_context="US",
                page_context="stock",
            )

        fetch.assert_called_once_with("2330")
        report_lookup.assert_not_called()
        self.assertIn("台積電（2330）", answer.text)

    def test_browser_clients_receive_isolated_principals(self):
        principals = []

        def converse(**kwargs):
            principals.append(kwargs["principal"])
            return ConversationAnswer("ok")

        with patch.object(stock_app, "run_absorb_conversation", side_effect=converse):
            stock_app.app.test_client().post("/api/conversation", json=self._payload("RSI 是什麼？"))
            stock_app.app.test_client().post("/api/conversation", json=self._payload("RSI 是什麼？"))
        self.assertNotEqual(principals[0], principals[1])


if __name__ == "__main__":
    unittest.main()
