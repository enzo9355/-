# app.py
# v3.1 修正版：修復 API 參數錯誤、統一 V4 版本、優化大盤抓取效能
# --------------------------------------------------

import os
import io
import base64
import datetime
import requests
import pandas as pd
import twstock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, abort, render_template_string

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)

# ==================================================
# 基本設定
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FINMIND_USER = os.getenv("FINMIND_USER")
FINMIND_PASSWORD = os.getenv("FINMIND_PASSWORD")

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

finmind_token = ""
CATEGORY_PAGE_SIZE = 12

# ==================================================
# FinMind 登入
# ==================================================
def finmind_login():
    global finmind_token

    if finmind_token:
        return

    if not FINMIND_USER or not FINMIND_PASSWORD:
        return

    try:
        r = requests.post(
            "https://api.finmindtrade.com/api/v4/login",
            data={
                "user_id": FINMIND_USER,
                "password": FINMIND_PASSWORD
            },
            timeout=10
        ).json()

        if r.get("msg") == "success":
            finmind_token = r["token"]
    except Exception as e:
        print(f"FinMind 登入失敗: {e}")

# ==================================================
# 工具
# ==================================================
def get_stock_name(code):
    if code == "TAIEX":
        return "台股大盤（加權指數）"

    if code in twstock.codes:
        return twstock.codes[code].name

    return code

def search_stock_code(keyword):
    keyword = keyword.upper().strip()

    if keyword in ["TAIEX", "加權指數", "台股大盤", "大盤"]:
        return "TAIEX", "台股大盤（加權指數）"

    if keyword.isdigit():
        return keyword, get_stock_name(keyword)

    for code, info in twstock.codes.items():
        if keyword in info.name.upper():
            return code, info.name

    return None, None

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ==================================================
# 抓加權指數（真實大盤）
# ==================================================
def _get_taiex_data(days=180):
    finmind_login()

    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 方法 A：FinMind 加權指數 (改用日線資料，避免 5 秒資料過載)
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": "TAIEX",
            "start_date": start_date,
            "end_date": end_date
        }
        if finmind_token:
            params["token"] = finmind_token

        r = requests.get(url, params=params, timeout=15).json()

        if r.get("msg") == "success" and r.get("data"):
            df = pd.DataFrame(r["data"])
            df["Date"] = pd.to_datetime(df["date"], errors="coerce")
            df["Close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df[["Date", "Close"]].dropna()
            df.set_index("Date", inplace=True)
            if not df.empty:
                return df[["Close"]]
    except Exception as e:
        print(f"FinMind 大盤獲取失敗: {e}")

    # 方法 B：Yahoo Finance 備援 (十分穩定的備案)
    try:
        import yfinance as yf
        hist = yf.download("^TWII", start=start_date, progress=False, auto_adjust=False)
        if not hist.empty and "Close" in hist.columns:
            df = hist[["Close"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.dropna()
    except Exception as e:
        print(f"yfinance 大盤獲取失敗: {e}")

    return pd.DataFrame()

# ==================================================
# 抓個股股價
# ==================================================
def get_stock_data(stock_code, days=180):
    finmind_login()

    if stock_code == "TAIEX":
        return _get_taiex_data(days)

    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # 教練修正：統一使用 V4，並修正參數名稱為 data_id 與 start_date
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_code,
            "start_date": start_date,
            "end_date": end_date
        }
        if finmind_token:
            params["token"] = finmind_token

        data = requests.get(url, params=params, timeout=15).json()

        if not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[["Date", "Close"]].dropna()
        df.set_index("Date", inplace=True)

        return df[["Close"]].dropna()

    except Exception as e:
        print(f"個股資料獲取失敗 ({stock_code}): {e}")
        return pd.DataFrame()

# ==================================================
# 指標
# ==================================================
def calc_indicators(df):
    df = df.copy()
    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

# ==================================================
# 動態勝率
# ==================================================
def calc_win_probability(df):
    close = df["Close"]

    last = _safe_float(close.iloc[-1])
    ma20 = _safe_float(df["MA20"].iloc[-1])
    rsi = _safe_float(df["RSI"].iloc[-1])

    ret_5 = _safe_float(close.pct_change(5).iloc[-1] * 100)
    ret_10 = _safe_float(close.pct_change(10).iloc[-1] * 100)
    vol_10 = _safe_float(close.pct_change().rolling(10).std().iloc[-1] * 100)

    if ma20 == 0:
        return 50

    ma_gap = ((last - ma20) / ma20) * 100

    score = 50
    score += ma_gap * 3.0
    score += (rsi - 50) * 0.6
    score += ret_5 * 1.5
    score += ret_10 * 0.8
    score -= vol_10 * 1.2

    prob = round(score)
    prob = max(5, min(95, prob))

    return int(prob)

# ==================================================
# 圖表
# ==================================================
def create_chart(df, title):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="收盤價", color="black")
    plt.plot(df.index, df["MA20"], label="MA20", color="red", linestyle="--")

    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    plt.close()

    img.seek(0)
    return base64.b64encode(img.read()).decode()

# ==================================================
# 分析股票
# ==================================================
def analyze_stock(code):
    df = get_stock_data(code, 180)

    if df.empty or len(df) < 30:
        return None

    df = calc_indicators(df)

    if df.empty:
        return None

    last = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    trend = "多頭" if last > ma20 else "空頭"
    prob = calc_win_probability(df)

    chart = create_chart(df.tail(60), f"{get_stock_name(code)} ({code})")

    return {
        "code": code,
        "name": get_stock_name(code),
        "price": last,
        "ma20": ma20,
        "rsi": rsi,
        "trend": trend,
        "prob": prob,
        "chart": chart
    }

# ==================================================
# 大盤預測
# ==================================================
def market_forecast():
    df = get_stock_data("TAIEX", 180)

    if df.empty or len(df) < 30:
        return None

    df = calc_indicators(df)

    if df.empty:
        return None

    last = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    trend = "多頭" if last > ma20 else "空頭"
    prob = calc_win_probability(df)

    chart = create_chart(df.tail(60), "台灣加權指數 (TAIEX)")

    return {
        "code": "TAIEX",
        "name": "台股大盤（加權指數）",
        "price": last,
        "ma20": ma20,
        "rsi": rsi,
        "trend": trend,
        "prob": prob,
        "chart": chart
    }

# ==================================================
# 動態分類
# ==================================================
def build_market_map():
    market = {
        "全市場": [],
        "ETF專區": [],
        "AI伺服器": []
    }

    ai_names = {
        "鴻海", "廣達", "緯創", "英業達", "仁寶",
        "和碩", "華碩", "微星", "技嘉", "神達",
        "緯穎", "勤誠", "雙鴻", "奇鋐", "宏碁"
    }

    official_groups = set()

    for code, info in twstock.codes.items():
        if len(code) not in [4, 5]:
            continue

        group = getattr(info, "group", None) or getattr(info, "type", None)
        if group and isinstance(group, str) and group.strip():
            official_groups.add(group.strip())

    for g in sorted(official_groups):
        market[g] = []

    for code, info in twstock.codes.items():
        if len(code) not in [4, 5]:
            continue

        name = info.name
        group = getattr(info, "group", None) or getattr(info, "type", None)
        group = group.strip() if isinstance(group, str) else None

        market["全市場"].append(code)

        if code.startswith("00"):
            market["ETF專區"].append(code)

        if name in ai_names:
            market["AI伺服器"].append(code)

        if group and group in market:
            market[group].append(code)

    market = {k: v for k, v in market.items() if v}
    return market

industry_map = build_market_map()

# ==================================================
# 產業分類分頁
# ==================================================
def get_all_categories():
    return list(industry_map.keys())

def get_category_total_pages():
    cats = get_all_categories()
    if not cats:
        return 1
    return (len(cats) + CATEGORY_PAGE_SIZE - 1) // CATEGORY_PAGE_SIZE

def get_category_page(page=1):
    cats = get_all_categories()
    total = get_category_total_pages()

    if page < 1:
        page = 1
    if page > total:
        page = total

    start = (page - 1) * CATEGORY_PAGE_SIZE
    end = start + CATEGORY_PAGE_SIZE

    return cats[start:end], page, total

def build_category_quick_reply(page=1):
    categories, page, total = get_category_page(page)
    items = []

    for ind in categories:
        items.append(
            QuickReplyButton(
                action=MessageAction(
                    label=ind[:20],
                    text=f"選產業_{ind}"
                )
            )
        )

    if page < total and len(items) < 13:
        items.append(
            QuickReplyButton(
                action=MessageAction(
                    label="更多分類▶",
                    text=f"分類第_{page + 1}頁"
                )
            )
        )

    return QuickReply(items=items), f"請選擇市場類別（第 {page}/{total} 頁）👇"

def build_category_list_text():
    cats = get_all_categories()
    lines = ["📚 產業分類總表\n"]
    for i, c in enumerate(cats, 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines[:120])

# ==================================================
# 推薦
# ==================================================
def build_style_result(category):
    arr = industry_map.get(category, [])
    arr = arr[:10]

    if not arr:
        return "❌ 無資料"

    aggressive = arr[:5]
    conservative = arr[5:10]

    lines = [f"📈 {category} Top10\n"]

    lines.append("🔥 激進型")
    for i, c in enumerate(aggressive, 1):
        lines.append(f"{i}. {c} {get_stock_name(c)}")

    lines.append("")
    lines.append("🛡 保守型")
    for i, c in enumerate(conservative, 1):
        lines.append(f"{i}. {c} {get_stock_name(c)}")

    return "\n".join(lines)

# ==================================================
# UI
# ==================================================
def render_dashboard(data):
    html = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{data['name']}</title>
<style>
body {{ margin:0; background:#050505; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
.wrap {{ max-width:920px; margin:auto; padding:30px 20px 60px; }}
h1 {{ font-size:42px; margin-bottom:24px; }}
.card {{ background:#1a1a1c; border-radius:22px; padding:26px; margin-bottom:24px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; }}
img {{ width:100%; border-radius:18px; background:#fff; }}
.small {{ font-size:18px; line-height:1.8; }}
@media (max-width: 640px) {{ h1 {{ font-size:30px; }} .small {{ font-size:16px; }} .card {{ padding:20px; border-radius:18px; }} }}
</style>
</head>
<body>
<div class="wrap">
<h1>{data['name']} ({data['code']})</h1>
<div class="card small">
💰 最新收盤：{data['price']:.2f}<br>
🌊 20日均線：{data['ma20']:.2f}<br>
🌡 RSI：{data['rsi']:.1f}<br>
📈 趨勢：{data['trend']}<br>
📊 上漲機率分數：{data['prob']}%
</div>
<div class="card">
<img src="data:image/png;base64,{data['chart']}">
</div>
<div class="grid">
<div class="card small">
📑 指標摘要<br><br>
📈 趨勢判讀：{data['trend']}<br>
🌊 均線狀態：{'站上 MA20' if data['price'] > data['ma20'] else '跌破 MA20'}<br>
🌡 RSI 強弱：{'偏強' if data['rsi'] >= 55 else '中性' if data['rsi'] >= 45 else '偏弱'}<br>
🎯 分數評估：{data['prob']}%
</div>
<div class="card small">
💡 觀察建議<br><br>
🛒 若趨勢轉強：可觀察分批布局<br>
🛡 若跌破均線：留意風險控管<br>
💰 接近前高壓力：可評估分段調節
</div>
</div>
<div class="card small">
📰 提醒<br><br>
本頁勝率為規則式技術分數，依均線、RSI、短期動能與波動度估算，僅供研究參考，不構成投資建議。
</div>
</div>
</body>
</html>
"""
    return render_template_string(html)

# ==================================================
# 網頁路由
# ==================================================
@app.route("/")
def home():
    return "<h1>AI 台股系統 v3.1 正常運作中</h1>"

@app.route("/stock/<stock_code>")
def stock_page(stock_code):
    code, _ = search_stock_code(stock_code)

    if not code:
        code = stock_code

    data = analyze_stock(code)

    if not data:
        return "查無資料"

    return render_dashboard(data)

@app.route("/market")
def market_page():
    data = market_forecast()

    if not data:
        return "查無資料"

    return render_dashboard(data)

# ==================================================
# LINE webhook
# ==================================================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

# ==================================================
# LINE 訊息處理
# ==================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.strip()

    # 大盤預測
    if msg == "大盤預測":
        data = market_forecast()

        if not data:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="大盤資料暫時無法取得，請稍後再試。")
            )
            return

        url = f"{request.host_url}market".replace("http://", "https://")

        text = (
            f"📊 台股大盤（加權指數）\n\n"
            f"💰 指數點位：{data['price']:.2f}\n"
            f"🌊 MA20：{data['ma20']:.2f}\n"
            f"🌡 RSI：{data['rsi']:.1f}\n"
            f"📈 趨勢：{data['trend']}\n"
            f"📊 上漲機率分數：{data['prob']}%\n\n"
            f"完整分析：{url}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text)
        )
        return

    # 預測分類首頁
    if msg == "預測":
        quick_reply, text = build_category_quick_reply(1)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text, quick_reply=quick_reply)
        )
        return

    # 分類分頁
    if msg.startswith("分類第_") and msg.endswith("頁"):
        try:
            page = int(msg.replace("分類第_", "").replace("頁", ""))
        except:
            page = 1

        quick_reply, text = build_category_quick_reply(page)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text, quick_reply=quick_reply)
        )
        return

    # 產業總表
    if msg == "產業列表":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=build_category_list_text())
        )
        return

    # 分類結果
    if msg.startswith("選產業_"):
        cat = msg.replace("選產業_", "")

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=build_style_result(cat))
        )
        return

    # 免責聲明
    if msg == "免責聲明":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="本系統資訊僅供研究參考，不構成投資建議，投資盈虧請自負。"
            )
        )
        return

    # 股票查詢
    code, name = search_stock_code(msg)

    if code:
        if code == "TAIEX":
            data = market_forecast()

            if not data:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="大盤資料暫時無法取得，請稍後再試。")
                )
                return

            url = f"{request.host_url}market".replace("http://", "https://")

            text = (
                f"📊 台股大盤（加權指數）\n\n"
                f"💰 指數點位：{data['price']:.2f}\n"
                f"🌊 20日均線：{data['ma20']:.2f}\n"
                f"🌡 RSI(14)：{data['rsi']:.1f}\n"
                f"📈 趨勢：{data['trend']}\n\n"
                f"🎯【預測區間：未來5日】\n"
                f"📊 上漲機率分數：{data['prob']}%\n\n"
                f"📌 點擊查看完整分析：\n{url}"
            )

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=text)
            )
            return

        data = analyze_stock(code)

        if not data:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="查無資料")
            )
            return

        url = f"{request.host_url}stock/{code}".replace("http://", "https://")
        direction = "偏向看漲 📈" if data["prob"] >= 55 else "中性震盪 ➖" if data["prob"] >= 45 else "偏向看跌 📉"

        text = (
            f"📊 {name} ({code})\n\n"
            f"💰 最新收盤：{data['price']:.2f}\n"
            f"🌊 20日均線：{data['ma20']:.2f}\n"
            f"🌡 RSI(14)：{data['rsi']:.1f}\n"
            f"📈 趨勢：{data['trend']}\n\n"
            f"🎯【預測區間：未來5日】\n"
            f"📊 上漲機率分數：{direction} ({data['prob']}%)\n\n"
            f"📌 點擊查看完整分析：\n{url}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text)
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="請輸入股票代碼，或輸入：預測 / 大盤預測 / 產業列表"
            )
        )

# ==================================================
# 啟動
# ==================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
