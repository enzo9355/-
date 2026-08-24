from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import secrets
from typing import Any

from absorb.conversation.context import MemoryContextStore
from absorb.conversation.errors import InputRejected, ModelUnavailable, ToolRejected, ToolUnavailable
from absorb.conversation.metrics import CONVERSATION_METRICS
from absorb.conversation.policies import (
    MAX_TOOL_CALLS,
    contains_prompt_injection,
    is_chase_question,
    looks_like_prompt_injection,
    requires_tool_data,
    validate_question,
)
from absorb.conversation.prompts import answer_prompt, planning_prompt
from absorb.conversation.schemas import ConversationAnswer, ConversationContext, PendingConfirmation, ToolCall
from absorb.conversation.tools import resolve_entities


_ACTIONS = {
    "優先布局", "分批布局", "持有觀察", "等待確認", "降低部位", "暫時避開",
    "控制追價", "提高防守", "逢回布局", "積極選股", "分批觀察", "優先關注", "降低曝險",
}
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?")


def _normalized_number(value: str) -> str:
    try:
        return format(float(value), ".12g")
    except ValueError:
        return value


def _grounded_numeric_values(value, *, key=None) -> set[str]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float)):
        return {_normalized_number(str(value))}
    if isinstance(value, str) and key in {"symbol", "code"}:
        return {_normalized_number(item) for item in _NUMBER.findall(value)}
    if isinstance(value, dict):
        return set().union(*(
            _grounded_numeric_values(item, key=item_key)
            for item_key, item in value.items()
        ))
    if isinstance(value, (list, tuple)):
        return set().union(*(_grounded_numeric_values(item, key=key) for item in value))
    return set()


def numbers_are_grounded(text: str, question: str, tool_results: list[dict[str, Any]]) -> bool:
    allowed = _grounded_numeric_values(tool_results)
    allowed.update(_normalized_number(value) for value in _NUMBER.findall(question))
    for value in tuple(allowed):
        try:
            number = float(value)
        except ValueError:
            continue
        if 0 <= abs(number) <= 1:
            allowed.add(_normalized_number(str(number * 100)))
    for match in _NUMBER.finditer(text):
        value = _normalized_number(match.group())
        if value in allowed:
            continue
        line_prefix = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
        suffix = text[match.end():match.end() + 1]
        if value in {str(item) for item in range(1, 8)} and not line_prefix.strip() and suffix in ".、)）":
            continue
        return False
    return True


def _evidence_action_labels(value) -> set[str]:
    if isinstance(value, dict):
        labels = {value["action_label"]} if value.get("action_label") in _ACTIONS else set()
        return labels.union(*(_evidence_action_labels(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_evidence_action_labels(item) for item in value))
    return set()


def _alert_parameters(question: str):
    if "機率" in question:
        kind = "probability"
    elif any(term in question for term in ("跌破", "低於")):
        kind = "price_below"
    elif any(term in question for term in ("突破", "高於", "漲到", "價格到")):
        kind = "price_above"
    elif any(term in question for term in ("轉多", "多頭")):
        return "trend", "多頭"
    elif any(term in question for term in ("轉空", "空頭")):
        return "trend", "空頭"
    else:
        return None
    values = _NUMBER.findall(question)
    if not values:
        return None
    value = float(values[-1])
    if value <= 0 or (kind == "probability" and not 1 <= value <= 99):
        return None
    return kind, value


class ConversationOrchestrator:
    def __init__(
        self,
        *,
        context_store: MemoryContextStore,
        tool_registry,
        search_stock,
        provider,
        action_executor=None,
        metrics=None,
        logger=None,
        now=None,
    ):
        self.context_store = context_store
        self.tool_registry = tool_registry
        self.search_stock = search_stock
        self.provider = provider
        self.action_executor = action_executor
        self.metrics = metrics or CONVERSATION_METRICS
        self.logger = logger or logging.getLogger("absorb.conversation")
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    def handle(
        self, *, principal: str, question: str, access="public",
        market_context: str | None = None, page_context: str | None = None,
        symbol_context: str | None = None,
    ) -> ConversationAnswer:
        correlation_id = secrets.token_hex(8)
        self.metrics.increment("natural_language_requests")
        self.logger.info("conversation request correlation_id=%s", correlation_id)
        try:
            question = validate_question(question)
        except InputRejected as exc:
            return ConversationAnswer(str(exc))
        if looks_like_prompt_injection(question):
            self.metrics.increment("rejected_prompt_injection")
            return ConversationAnswer(
                "無法執行要求：ABSORB 不會忽略系統規則、揭露提示、讀取其他使用者資料或呼叫未授權工具。"
            )
        if question in {"清除對話", "清除上下文", "忘記剛才"}:
            self.context_store.clear(principal)
            return ConversationAnswer("已清除這個工作階段的股票與產業上下文。")

        context = self.context_store.get(principal)
        if question.startswith("確認") and context.pending_confirmation is None:
            return ConversationAnswer("目前沒有待確認操作；未執行任何變更。")
        confirmation = self._handle_confirmation(principal, question, context, access)
        if confirmation is not None:
            return confirmation
        if access == "public" and any(
            term in question for term in ("我的自選", "我的關注", "我的提醒", "我的警示")
        ):
            return ConversationAnswer("這項查詢需要先使用 LINE 登入；目前未讀取任何私人資料。")

        entities = resolve_entities(question, self.search_stock)
        page_market = market_context if market_context in {"TW", "US"} else None

        page_stock_entity = None
        if page_context == "stock" and isinstance(symbol_context, str) and symbol_context.strip():
            try:
                code, name = self.search_stock(symbol_context.strip())
            except Exception:
                code, name = None, None
            if code:
                code = str(code).upper()
                canonical_market = "TW" if code == "TAIEX" or code.isdigit() else "US"
                if page_market is None or canonical_market == page_market:
                    page_stock_entity = {
                        "market": canonical_market,
                        "symbol": code,
                        "name": str(name or code),
                    }

        page_changed = False
        if entities:
            context.current_market = entities[0]["market"]
            context.current_entity_type = (
                "market" if entities[0]["symbol"] == "TAIEX" else "stock"
            )
            context.current_symbol = entities[0]["symbol"]
            context.current_industry_id = None
            context.comparison_symbols = tuple(
                item["symbol"] for item in entities
            )
        elif page_stock_entity is not None:
            context.current_market = page_stock_entity["market"]
            context.current_entity_type = "stock"
            context.current_symbol = page_stock_entity["symbol"]
            context.current_industry_id = None
            context.comparison_symbols = (page_stock_entity["symbol"],)
            entities = [page_stock_entity]
        elif page_market is not None:
            page_changed = (
                context.last_page_market is not None
                and context.last_page_market != page_market
            )
            if page_changed:
                context.current_entity_type = "market"
                context.current_symbol = None
                context.current_industry_id = None
                context.comparison_symbols = ()
            if not (
                context.current_symbol
                or context.current_industry_id
                or context.comparison_symbols
            ):
                context.current_market = page_market
                context.current_entity_type = "market"
        if page_market is not None:
            context.last_page_market = page_market
        if page_changed or page_stock_entity is not None or entities:
            # An explicit entity, page entity, or real page transition is authoritative
            # even when a pronoun-only question returns before the normal end-of-request save.
            self.context_store.save(principal, context)
        if not entities and context.comparison_symbols and "第二檔" in question:
            symbol = context.comparison_symbols[1] if len(context.comparison_symbols) > 1 else None
            if symbol:
                entities = [{"market": "TW" if symbol.isdigit() else "US", "symbol": symbol, "name": symbol}]
        if not entities and context.comparison_symbols and "第一檔" in question:
            symbol = context.comparison_symbols[0]
            entities = [{"market": "TW" if symbol.isdigit() else "US", "symbol": symbol, "name": symbol}]
        if not entities and context.current_symbol and any(term in question for term in ("這檔", "那檔", "那能", "剛才", "它")):
            entities = [{
                "market": context.current_market,
                "symbol": context.current_symbol,
                "name": context.current_symbol,
            }]
        if not entities and any(term in question for term in ("這檔", "那檔", "那能追", "它現在")):
            self.metrics.increment("clarification_requests")
            return ConversationAnswer("你指的是哪一檔股票？請提供股票代碼或名稱。")

        proposal = self._propose_action(principal, question, context, entities, access)
        if proposal is not None:
            return proposal

        context_payload = {
            "market": context.current_market,
            "entity_type": context.current_entity_type,
            "symbol": context.current_symbol,
            "industry_id": context.current_industry_id,
            "report_type": context.current_report_type,
            "comparison_symbols": list(context.comparison_symbols),
            "page": page_context,
        }
        allowed_entities = {(item["market"], item["symbol"]) for item in entities}
        if context.current_symbol and context.current_market:
            allowed_entities.add((context.current_market, context.current_symbol))
        for symbol in context.comparison_symbols:
            allowed_entities.add(("TW" if symbol == "TAIEX" or symbol.isdigit() else "US", symbol))
        must_use_tools = bool(entities) or requires_tool_data(
            question,
            has_context=bool(
                context.current_symbol or context.current_industry_id or context.comparison_symbols
            ),
        )
        try:
            raw_plan = self.provider.plan(
                planning_prompt(question, context_payload, self.tool_registry.catalog_for(access), entities)
            )
            calls = self._parse_plan(raw_plan)
        except Exception:
            self.metrics.increment("llm_error")
            return ConversationAnswer("ABSORB 自然語言分析暫時無法使用；固定指令與股票代碼查詢仍可正常使用。")

        if must_use_tools and not calls:
            self.metrics.increment("insufficient_data_answers")
            return ConversationAnswer("目前無法取得足夠的最新資料，因此不提供進場或追價判斷。")
        results = []
        for call in calls:
            self.metrics.increment("tool_calls")
            try:
                arguments = dict(call.arguments)
                if call.name == "get_industry_analysis" and context.current_industry_id:
                    arguments.setdefault("industry_id", context.current_industry_id)
                result = self.tool_registry.execute(
                    call.name,
                    arguments,
                    principal_access=access,
                    allowed_entities=allowed_entities,
                )
                results.append(result.to_dict())
            except (ToolRejected, ToolUnavailable) as exc:
                self.metrics.increment("tool_errors")
                results.append({"name": call.name, "ok": False, "data": None, "error": str(exc)})

        if any(item.get("ok") and contains_prompt_injection(item.get("data")) for item in results):
            self.metrics.increment("rejected_prompt_injection")
            return ConversationAnswer("工具資料未通過安全驗證，因此未交給回答模型。")

        successful = [
            item for item in results
            if item.get("ok") and isinstance(item.get("data"), dict)
        ]
        core = [
            item for item in successful
            if item["data"].get("data_quality") != "unavailable"
        ]
        if must_use_tools and not core:
            self.metrics.increment("insufficient_data_answers")
            return ConversationAnswer("目前無法取得足夠的最新資料，因此不提供進場或追價判斷。")
        chase = is_chase_question(question)
        if chase and not any(
            item["data"].get("action_label") and item["data"].get("five_day_probability") is not None
            for item in core
        ):
            self.metrics.increment("insufficient_data_answers")
            return ConversationAnswer("目前無法取得足夠的最新資料，因此不提供進場或追價判斷。")

        try:
            text = str(self.provider.answer(answer_prompt(question, results, chase=chase)) or "").strip()
        except Exception:
            self.metrics.increment("llm_error")
            return ConversationAnswer("ABSORB 自然語言分析暫時無法使用；未執行任何寫入操作。")
        if not text:
            self.metrics.increment("llm_error")
            return ConversationAnswer("ABSORB 自然語言分析暫時無法使用；未執行任何寫入操作。")
        if not numbers_are_grounded(text, question, results):
            return ConversationAnswer("自然語言回覆包含未由資料工具提供的數字，因此已安全拒絕；請改用固定查詢查看原始量化分析。")

        evidence_labels = _evidence_action_labels(results)
        mentioned_labels = {label for label in _ACTIONS if label in text}
        if mentioned_labels - evidence_labels:
            return ConversationAnswer("自然語言回覆未通過行動標籤一致性驗證；請改用固定股票查詢查看原始量化分析。")
        action_label = next(iter(evidence_labels)) if len(evidence_labels) == 1 else None
        if action_label:
            if action_label not in text:
                text = f"結論：{action_label}\n\n{text}"

        first = core[0]["data"] if core else {}
        if entities:
            context.current_market = entities[0]["market"]
            context.current_entity_type = "market" if entities[0]["symbol"] == "TAIEX" else "stock"
            context.current_symbol = entities[0]["symbol"]
            context.comparison_symbols = tuple(item["symbol"] for item in entities)
            context.current_industry_id = None
        industry_id = next(
            (
                item["data"].get("industry_id")
                for item in core
                if item.get("name") == "get_industry_analysis" and item["data"].get("industry_id")
            ),
            None,
        )
        if industry_id:
            context.current_market = first.get("market") or context.current_market
            context.current_entity_type = "industry"
            context.current_symbol = None
            context.current_industry_id = str(industry_id)
            context.comparison_symbols = ()
        context.last_question_type = "chase_risk" if chase else "research"
        context.last_tool_entities = tuple(item["symbol"] for item in entities)
        self.context_store.save(principal, context)
        self.metrics.increment("llm_success")
        has_stale = any(
            item["data"].get("stale") is True
            or item["data"].get("data_quality") == "stale"
            for item in core
        )
        if has_stale:
            self.metrics.increment("stale_data_answers")
        if first:
            quality = "stale" if has_stale else first.get("data_quality", "unavailable")
            as_of = first.get("data_as_of") or "未提供"
            prefix = "資料已過期；" if has_stale else ""
            text = f"{text}\n\n資料狀態：{prefix}{quality}｜資料截至：{as_of}"
        return ConversationAnswer(
            text=text,
            data_as_of=first.get("data_as_of"),
            data_quality="stale" if has_stale else first.get("data_quality", "unavailable"),
            stale=has_stale,
            tools_used=tuple(item["name"] for item in results),
            action_label=action_label,
        )

    @staticmethod
    def _parse_plan(raw: Any) -> tuple[ToolCall, ...]:
        if isinstance(raw, str):
            if len(raw) > 8192:
                raise ModelUnavailable("plan too large")
            raw = json.loads(raw)
        if not isinstance(raw, dict) or set(raw) != {"tool_calls"} or not isinstance(raw["tool_calls"], list):
            raise ModelUnavailable("invalid plan")
        if len(raw["tool_calls"]) > MAX_TOOL_CALLS:
            raise ModelUnavailable("too many tools")
        calls = []
        for item in raw["tool_calls"]:
            if not isinstance(item, dict) or set(item) != {"name", "arguments"}:
                raise ModelUnavailable("invalid tool call")
            if not isinstance(item["name"], str) or not isinstance(item["arguments"], dict):
                raise ModelUnavailable("invalid tool call")
            calls.append(ToolCall(item["name"], item["arguments"]))
        return tuple(calls)

    def _propose_action(self, principal, question, context, entities, access):
        action = None
        alert_parameters = None
        if any(term in question for term in ("清空自選", "清空關注")):
            action = "watchlist_clear"
        elif any(term in question for term in ("關閉所有提醒", "清空提醒", "刪除所有提醒", "取消所有提醒")):
            action = "alerts_clear"
        elif "提醒" in question and any(
            term in question for term in ("刪除", "刪掉", "移除", "取消", "更新", "修改", "改成")
        ):
            return ConversationAnswer("單筆提醒更新或刪除請使用「提醒管理」；目前未執行任何變更。")
        elif any(term in question for term in ("加入自選", "加入關注")):
            action = "watchlist_add"
        elif any(term in question for term in ("移除自選", "刪掉", "取消關注")):
            action = "watchlist_remove"
        elif "提醒" in question and any(term in question for term in ("設定", "建立", "新增", "提醒我")):
            alert_parameters = _alert_parameters(question)
            if alert_parameters is None:
                return ConversationAnswer("請明確提供提醒條件，例如「台積電跌破 900 元時提醒我」。")
            action = "alert_create"
        if action is None:
            return None
        if access == "public":
            return ConversationAnswer("這項操作需要先使用 LINE 登入；目前未建立或執行任何變更。")
        if action not in {"watchlist_clear", "alerts_clear"} and not entities:
            return ConversationAnswer("請提供要變更的股票代碼或名稱。")
        if action in {"watchlist_clear", "alerts_clear"}:
            parameters = {}
        else:
            parameters = {
                "market": entities[0]["market"], "symbol": entities[0]["symbol"], "name": entities[0]["name"]
            }
            if action == "alert_create":
                parameters.update(kind=alert_parameters[0], value=alert_parameters[1])
        nonce = secrets.token_urlsafe(6)
        raw_key = f"{principal}|{action}|{json.dumps(parameters, sort_keys=True, ensure_ascii=False)}|{nonce}"
        pending = PendingConfirmation(
            action_type=action,
            parameters=parameters,
            nonce=nonce,
            expires_at=self.now() + dt.timedelta(minutes=5),
            idempotency_key=hashlib.sha256(raw_key.encode()).hexdigest(),
        )
        context.pending_confirmation = pending
        self.context_store.save(principal, context)
        self.metrics.increment("write_action_proposals")
        label = {
            "watchlist_add": "加入關注",
            "watchlist_remove": "移除關注",
            "watchlist_clear": "清空全部關注",
            "alert_create": "建立提醒",
            "alerts_clear": "關閉全部提醒",
        }[action]
        target = parameters.get("name") or "全部標的"
        return ConversationAnswer(
            f"待確認操作：{label}「{target}」。請在 5 分鐘內輸入「確認 {nonce}」；輸入「取消」可放棄。",
            requires_confirmation=True,
        )

    def _handle_confirmation(self, principal, question, context, access):
        pending = context.pending_confirmation
        if pending is None:
            return None
        if pending.expires_at <= self.now():
            pending.status = "expired"
            context.pending_confirmation = None
            self.context_store.save(principal, context)
            return ConversationAnswer("上一個待確認操作已逾時，請重新提出要求。") if question.startswith("確認") else None
        if question == "取消":
            pending.status = "cancelled"
            context.pending_confirmation = None
            self.context_store.save(principal, context)
            return ConversationAnswer("已取消待確認操作。")
        if not question.startswith("確認"):
            return None
        if question != f"確認 {pending.nonce}" or access == "public":
            return ConversationAnswer("確認碼不正確、已過期或與目前登入身分不符；未執行任何變更。")
        if not self.context_store.claim_action(pending.idempotency_key):
            return ConversationAnswer("這項操作已處理，未重複執行。")
        if self.action_executor is None:
            self.context_store.release_action(pending.idempotency_key)
            return ConversationAnswer("個人操作服務暫時無法使用；未執行任何變更。")
        try:
            self.action_executor(pending.action_type, dict(pending.parameters), pending.idempotency_key)
        except Exception:
            self.context_store.release_action(pending.idempotency_key)
            return ConversationAnswer("個人操作服務暫時無法使用；未執行任何變更。")
        pending.status = "confirmed"
        context.pending_confirmation = None
        self.context_store.save(principal, context)
        self.metrics.increment("write_confirmations")
        return ConversationAnswer("已完成確認的操作。")
