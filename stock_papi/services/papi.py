"""Legacy-prefixed compatibility adapter for ABSORB research summaries."""

import re
from stock_papi.shared.symbol import get_instrument_type
from stock_papi.integrations.market_data.tw_security_master import is_taiwan_symbol
from absorb.conversation.prompts import SYSTEM_PROMPT


def get_ai_insight_for_broadcast(name, data, bt, news, gemini_model):
    if not gemini_model: return "未設定 API Key，無法生成觀點。"
    n_txt = "\n".join([n['title'] for n in news])
    prompt = f"""{SYSTEM_PROMPT}
請以 ABSORB 的冷靜、證據導向語氣，針對{name}撰寫100字內洞見。只可引用下列資料；沒有 action label 時不得自創行動標籤。
最新價:{data['price']}
五日上漲機率:{data['prob']}%
夏普值:{bt['sharpe']:.2f}
新聞:\n{n_txt}"""
    try:
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety)
        return response.text.strip() if response.text else "AI 觀點生成為空。"
    except Exception as e:
        return "暫時無法生成 AI 觀點，請參考量化數據。"


class AbsorbResearchService:
    def __init__(
        self,
        *,
        requests_module,
        openalice_url,
        openalice_token,
        search_stock,
        get_stock_name,
        twstock_codes,
        industry_map,
        analyze,
        system_cache,
        cache_expiry_seconds,
        line_store,
        load_sector_snapshot,
        safe_float,
        gemini_model,
        now,
        sleep,
        logger,
        build_prompt_fn,
        extract_stock_fn,
        match_sector_fn,
        gather_sector_data_fn,
        build_single_context_fn,
        build_sector_examples_fn,
        taiwan_security_master=None,
    ):
        self.requests = requests_module
        self.openalice_url = openalice_url
        self.openalice_token = openalice_token
        self.search_stock = search_stock
        self.get_stock_name = get_stock_name
        self.twstock_codes = twstock_codes
        self.taiwan_security_master = taiwan_security_master
        self.industry_map = industry_map
        self.analyze = analyze
        self.system_cache = system_cache
        self.cache_expiry_seconds = cache_expiry_seconds
        self.line_store = line_store
        self.load_sector_snapshot = load_sector_snapshot
        self.safe_float = safe_float
        self.gemini_model = gemini_model
        self.now = now
        self.sleep = sleep
        self.logger = logger
        self.build_prompt_fn = build_prompt_fn
        self.extract_stock_fn = extract_stock_fn
        self.match_sector_fn = match_sector_fn
        self.gather_sector_data_fn = gather_sector_data_fn
        self.build_single_context_fn = build_single_context_fn
        self.build_sector_examples_fn = build_sector_examples_fn


    def call_openalice(self, prompt):
        response = self.requests.post(
            self.openalice_url,
            headers={"Authorization": f"Bearer {self.openalice_token}"},
            json={"prompt": self.build_prompt_fn(prompt)},
            timeout=4,
        )
        response.raise_for_status()
        payload = response.json()
        summary = str(
            payload.get("summary") or payload.get("text") or payload.get("message") or ""
        ).strip()
        detail_url = str(payload.get("detail_url") or payload.get("url") or "").strip()
        if not summary:
            summary = "ABSORB 沒有回傳可用摘要。"
        return summary + (f"\n\n詳細分析：{detail_url}" if detail_url else "")


    def extract_stock(self, prompt):
        """Extract a stock or market target from a natural-language question."""
        prompt = str(prompt or "").strip()
        code_match = re.search(r"(?<![0-9A-Z])([0-9]{4,5}[0-9A-Z]?)(?![0-9A-Z])", prompt.upper())
        if code_match and (
            code_match.group(1) in self.twstock_codes
            or (
                self.taiwan_security_master is not None
                and self.taiwan_security_master.contains(code_match.group(1))
            )
        ):
            code = code_match.group(1)
            return code, self.get_stock_name(code)

        name_matches = [
            (code, info.name) for code, info in self.twstock_codes.items()
            if info.name and info.name in prompt
        ]
        if name_matches:
            return max(name_matches, key=lambda item: len(item[1]))

        ticker_match = re.search(r"(?<![A-Z])([A-Z]{3,5})(?![A-Z])", prompt)
        if ticker_match and ticker_match.group(1) not in {"PAPI", "ABSORB", "RSI", "MACD", "ETF"}:
            code, name = self.search_stock(ticker_match.group(1))
            if code:
                return code, name

        m = re.search(r"分析\s+(.+)", prompt)
        if m:
            keyword = m.group(1).strip()
            code, name = self.search_stock(keyword)
            if code:
                return code, name
        # Also try if the entire prompt is just a stock code or name
        code, name = self.search_stock(prompt)
        if code:
            return code, name
        if any(term in prompt for term in ("台股", "台灣股市", "大盤", "加權指數", "盤勢")):
            return "TAIEX", "台股大盤"
        return None, None


    def match_sector(self, prompt):
        """Try to match a prompt to an industry-map category.

        Returns (category_name, stock_codes_list) or (None, None).
        """
        keywords = prompt.upper()
        best_cat = None
        best_len = 0
        for cat in self.industry_map:
            cat_upper = cat.upper()
            if cat_upper in keywords and len(cat_upper) > best_len:
                best_cat = cat
                best_len = len(cat_upper)
        if best_cat:
            return best_cat, self.industry_map[best_cat]
        return None, None


    def build_single_context(self, data):
        """Build a data context string for a single analyzed stock."""
        bt = data.get("bt", {})
        foreign = data.get("foreign_flow", {})
        foreign_str = ""
        if foreign.get("available"):
            foreign_str = f"外資買賣超：{foreign.get('status', '未知')}（近5日淨額 {foreign.get('net_5', 0):.0f}）"
        news_titles = "\n".join(
            [f"  - {n['title']}" for n in data.get("news", [])[:3]]
        )
        return (
            f"▸ {data.get('name', '?')} ({data.get('code', '?')})："
            f"收盤 {data['price']:.2f}，"
            f"五日上漲機率 {data['prob']}%，"
            f"趨勢 {data['trend']}，"
            f"RSI {data['rsi']:.1f}，"
            f"{'紅柱' if data['macd_osc'] > 0 else '綠柱'}，"
            f"KD {'黃金交叉' if data['k'] > data['d'] else '死亡交叉'}，"
            f"情緒 {data['s_status']}（{data['s_score']:.0f}），"
            f"情緒動能 {data.get('news_momentum', 0):+.0f}，"
            f"情緒分歧 {data.get('news_disagreement', 0):.0f}，"
            f"情緒波動 {data.get('news_weighted_volatility', 0):.0f}，"
            f"{foreign_str}，"
            f"回測策略報酬 {bt.get('strat_cum', 0):.1f}%，"
            f"策略交易勝率 {bt.get('win_rate', 0):.0f}%，"
            f"夏普 {bt.get('sharpe', 0):.2f}"
        )


    def gather_sector_data(self, codes, max_fresh=2, max_total=5):
        """Gather analysis data for a sector. Prioritize cache, analyze at most max_fresh new stocks.

        Returns a list of (code, data) tuples.
        """
        results = []
        fresh_count = 0
        now = self.now()

        # First pass: collect cached stocks
        for code in codes:
            if len(results) >= max_total:
                break
            if code in self.system_cache:
                cached_data, ts = self.system_cache[code]
                if now - ts < self.cache_expiry_seconds and cached_data:
                    results.append((code, cached_data))

        # Second pass: analyze a few uncached stocks if we need more
        if len(results) < max_total:
            for code in codes:
                if len(results) >= max_total or fresh_count >= max_fresh:
                    break
                if any(r[0] == code for r in results):
                    continue
                try:
                    data = self.analyze(code)
                    if data:
                        results.append((code, data))
                        fresh_count += 1
                except Exception:
                    continue

        return results


    def build_sector_examples(self, limit=3):
        if not self.line_store:
            return ""
        try:
            snapshot = self.load_sector_snapshot(self.line_store)
        except Exception:
            return ""
        items = []
        for category, signals in (snapshot or {}).get("sectors", {}).items():
            for item in signals or []:
                items.append((category, item))
        items.sort(key=lambda pair: self.safe_float(pair[1].get("score")), reverse=True)
        lines = []
        for category, item in items[:limit]:
            code = item.get('code')
            is_etf = get_instrument_type(code) == "ETF"
            foreign_str = f"，外資5日 {int(self.safe_float(item.get('foreign_net_5'))):,}" if not is_etf and item.get('foreign_net_5') is not None else ""
            lines.append(
                f"- {item.get('name')} ({code})：{category}，"
                f"五日上漲機率 {int(self.safe_float(item.get('prob')))}%，"
                f"{item.get('trend', '中性')}"
                + foreign_str
            )
        if not lines:
            return ""
        return "\n每日產業預測可舉例標的（只可從這裡挑，不要自己編）：\n" + "\n".join(lines)


    def build_prompt(self, prompt):
        data_context = ""

        # 1. Try individual stock first
        code, name = self.extract_stock_fn(prompt)
        if code:
            try:
                data = self.analyze(code)
            except Exception:
                data = None
            if data:
                data_context = f"""
    以下是 {name} ({code}) 的最新量化分析數據（來自我們的 LightGBM 模型與技術指標系統）：
    {self.build_single_context_fn(data)}
    - 回測結論：{data.get('bt', {}).get('conclusion', '無')}

    請根據以上「真實數據」來回答使用者的問題。數據是核心依據，你的角色是用白話文幫新手解讀這些數據。
    """
            else:
                data_context = f"""
    已辨識{name} ({code})，但本次未取得可用的量化分析數據。
    只能說目前資料暫時無法取得；不得改用其他股票或產業資料回答，也不得猜測失敗原因。
    """
        # 2. If no individual stock, try sector/industry match
        if not data_context:
            cat, cat_codes = self.match_sector_fn(prompt)
            if cat and cat_codes:
                sector_data = self.gather_sector_data_fn(cat_codes)
                if sector_data:
                    stock_lines = "\n".join(
                        self.build_single_context_fn(d) for _, d in sector_data
                    )
                    avg_prob = sum(d["prob"] for _, d in sector_data) / len(sector_data)
                    bullish = sum(1 for _, d in sector_data if d["trend"] == "多頭")
                    total = len(sector_data)
                    data_context = f"""
    以下是「{cat}」產業的量化分析數據（來自我們的 LightGBM 模型，共掃描 {total} 檔代表性個股）：

    產業概覽：
    - 平均 AI 五日上漲機率：{avg_prob:.0f}%
    - 多頭比例：{bullish}/{total} 檔呈多頭趨勢
    - {'產業整體偏多' if bullish > total / 2 else '產業整體偏空' if bullish < total / 2 else '產業多空分歧'}

    個股明細：
    {stock_lines}

    請根據以上「真實數據」綜合分析該產業的整體狀態與投資方向。引用具體個股數據來支撐你的論點，幫新手理解產業全貌。
    """
        if not data_context:
            data_context = self.build_sector_examples_fn()
        if not data_context:
            data_context = "\n目前沒有與問題直接對應的量化資料，請明確說明資料不足，不要推測原因。"

        return f"""{SYSTEM_PROMPT}

    {data_context}

    使用者問題：{prompt}

    請使用繁體中文。結論必須沿用資料中的既有 action label；沒有 action label 時不得自創。
    若使用者問追高，必須區分中期模型方向與短線追價風險，並列出支持與反對證據、失效條件、資料日期與限制。
    若問題要求推薦或列舉標的，最多提出 2 到 3 檔，且只能使用上方提供的資料。
    不得宣稱資料庫未收錄，也不得捏造系統、模型或資料取得失敗的原因。"""


    def call_gemini(self, prompt):
        if not self.gemini_model:
            return None
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        max_retries = 3
        backoff = 0.5
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(self.build_prompt_fn(prompt), safety_settings=safety)
                summary = (getattr(response, "text", "") or "").strip()
                if not summary:
                    return None
                return summary
            except Exception:
                self.logger.warning("ABSORB Gemini request failed (%s/%s)", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    self.sleep(backoff)
                    backoff *= 2
                    continue
                self.logger.error("ABSORB Gemini request failed after retries")
                raise


# Temporary import compatibility. New code uses AbsorbResearchService.
PapiService = AbsorbResearchService
