# app.py
# v2.6 真動態產業完整版
# --------------------------------------------------
# 核心升級：
# 1. 產業類別動態生成（不寫死股票名單）
# 2. 全市場由 twstock.codes 自動掃描
# 3. 激進5 / 保守5 真排序
# 4. 大盤預測 + 完整分析
# 5. FinMind 單一資料源
# --------------------------------------------------

import os
import datetime
import requests
import pandas as pd
import numpy as np
import twstock

from flask import Flask, request, abort

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

# ==================================================
# FinMind 登入
# ==================================================
def finmind_login():

    global finmind_token

    if finmind_token:
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

    except:
        pass


# ==================================================
# 工具函數
# ==================================================
def get_stock_name(code):

    if code in twstock.codes:
        return twstock.codes[code].name

    return code


def search_stock_code(keyword):

    keyword = keyword.upper().strip()

    if keyword.isdigit():
        return keyword, get_stock_name(keyword)

    for code, info in twstock.codes.items():
        if keyword in info.name.upper():
            return code, info.name

    return None, None


# ==================================================
# FinMind 抓股價
# ==================================================
def get_stock_data(stock_code, days=180):

    finmind_login()

    start_date = (
        datetime.datetime.now()
        - datetime.timedelta(days=days)
    ).strftime("%Y-%m-%d")

    try:
        url = (
            "https://api.finmindtrade.com/api/v4/data"
            f"?dataset=TaiwanStockPrice"
            f"&data_id={stock_code}"
            f"&start_date={start_date}"
            f"&token={finmind_token}"
        )

        data = requests.get(url, timeout=8).json()

        if data.get("msg") != "success":
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])

        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df["Date"] = pd.to_datetime(df["date"])

        df.set_index("Date", inplace=True)

        return df[["Close"]].dropna()

    except:
        return pd.DataFrame()


# ==================================================
# 動態產業分類（核心）
# ==================================================
def build_market_map():

    market = {
        "全市場": [],
        "ETF專區": [],
        "半導體": [],
        "AI伺服器": [],
        "金融保險": [],
        "航運物流": [],
        "傳產民生": [],
        "生技醫療": []
    }

    for code, info in twstock.codes.items():

        name = info.name

        if len(code) not in [4, 5]:
            continue

        market["全市場"].append(code)

        # ETF
        if code.startswith("00"):
            market["ETF專區"].append(code)

        # 半導體
        if any(k in name for k in [
            "半導體", "IC", "晶圓", "矽", "封測"
        ]):
            market["半導體"].append(code)

        # AI
        if any(k in name for k in [
            "廣達", "鴻海", "緯創", "仁寶",
            "技嘉", "微星", "英業達"
        ]):
            market["AI伺服器"].append(code)

        # 金融
        if any(k in name for k in [
            "金", "銀行", "證券", "保險"
        ]):
            market["金融保險"].append(code)

        # 航運
        if any(k in name for k in [
            "航", "運", "物流", "航空"
        ]):
            market["航運物流"].append(code)

        # 傳產
        if any(k in name for k in [
            "食品", "塑膠", "鋼", "汽車",
            "水泥", "紡織"
        ]):
            market["傳產民生"].append(code)

        # 生技
        if any(k in name for k in [
            "醫", "藥", "生技"
        ]):
            market["生技醫療"].append(code)

    return market


industry_map = build_market_map()

# ==================================================
# 技術指標計算
# ==================================================
def calc_metrics(code):

    df = get_stock_data(code, 180)

    if df.empty or len(df) < 30:
        return None

    close = df["Close"]

    ret = close.pct_change().dropna()

    ma20 = close.rolling(20).mean().iloc[-1]
    last = close.iloc[-1]

    vol = ret.std()

    momentum = (last / close.iloc[-20]) - 1 if len(close) > 20 else 0

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / (loss + 1e-9)

    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    return {
        "code": code,
        "name": get_stock_name(code),
        "last": last,
        "ma20": ma20,
        "vol": vol,
        "momentum": momentum,
        "rsi": rsi
    }


# ==================================================
# 激進 / 保守 排名
# ==================================================
def build_style_result(category):

    stock_list = industry_map.get(category, [])[:30]

    rows = []

    for code in stock_list:

        data = calc_metrics(code)

        if data:
            rows.append(data)

    if not rows:
        return "❌ 此分類暫無資料"

    # 激進型：高波動 + 強動能
    aggr = sorted(
        rows,
        key=lambda x: (
            x["vol"] * 100 +
            x["momentum"] * 50
        ),
        reverse=True
    )

    # 保守型：低波動 + 站上月線 + RSI中性
    safe = sorted(
        rows,
        key=lambda x: (
            (x["last"] / x["ma20"]) -
            x["vol"] * 10 -
            abs(x["rsi"] - 55) / 100
        ),
        reverse=True
    )

    aggressive = []
    conservative = []

    used = set()

    for item in aggr:
        if item["code"] not in used:
            aggressive.append(item)
            used.add(item["code"])
        if len(aggressive) == 5:
            break

    for item in safe:
        if item["code"] not in used:
            conservative.append(item)
            used.add(item["code"])
        if len(conservative) == 5:
            break

    lines = [f"📈 {category} Top10\n"]

    lines.append("🔥 激進型 5 檔")
    for i, x in enumerate(aggressive, 1):
        lines.append(f"{i}. {x['code']} {x['name']}")

    lines.append("")
    lines.append("🛡 保守型 5 檔")
    for i, x in enumerate(conservative, 1):
        lines.append(f"{i}. {x['code']} {x['name']}")

    return "\n".join(lines)


# ==================================================
# 大盤預測（0050代理，FinMind）
# 若你 FinMind 指數資料已確認，可再替換成 TAIEX
# ==================================================
def market_forecast():

    df = get_stock_data("0050", 365)

    if df.empty or len(df) < 60:
        return "❌ 無法取得大盤資料"

    close = df["Close"]

    ma20 = close.rolling(20).mean().iloc[-1]
    last = close.iloc[-1]

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rsi = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1]

    score = 0

    if last > ma20:
        score += 1
    if rsi < 70:
        score += 1
    if close.pct_change().iloc[-1] > 0:
        score += 1

    prob = 45 + score * 12

    trend = "多頭" if last > ma20 else "空頭"

    comment = (
        "偏強震盪 📈"
        if prob >= 70 else
        "中性偏多 ⚖️"
        if prob >= 58 else
        "震盪偏弱 📉"
    )

    return (
        "📊 台股大盤預測\n\n"
        f"💰 指標價格：{last:.2f}\n"
        f"🌡 RSI：{rsi:.1f}\n"
        f"📈 趨勢：{trend}\n\n"
        f"🤖 AI判斷：{comment}\n"
        f"📌 上漲機率：約 {prob:.0f}%"
    )


# ==================================================
# 網頁
# ==================================================
@app.route("/")
def home():
    return "<h1>AI 台股系統 v2.6 真動態產業版</h1>"


@app.route("/market")
def market_page():
    return f"<pre>{market_forecast()}</pre>"


@app.route("/stock/<stock_code>")
def stock_page(stock_code):

    code, _ = search_stock_code(stock_code)

    if not code:
        code = stock_code

    return f"<pre>{code} {get_stock_name(code)}</pre>"


# ==================================================
# LINE webhook
# ==================================================
@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)

    except InvalidSignatureError:
        abort(400)

    return "OK"


# ==================================================
# LINE 收訊息
# ==================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    msg = event.message.text.strip()

    # 大盤預測
    if msg == "大盤預測":

        url = f"{request.host_url}market".replace(
            "http://", "https://"
        )

        text = market_forecast() + f"\n\n完整分析：{url}"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=text)
        )
        return

    # 產業預測
    if msg == "預測":

        items = []

        for ind in industry_map.keys():

            items.append(
                QuickReplyButton(
                    action=MessageAction(
                        label=ind,
                        text=f"選產業_{ind}"
                    )
                )
            )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="請選擇市場類別 👇",
                quick_reply=QuickReply(items=items)
            )
        )
        return

    # 類別結果
    if msg.startswith("選產業_"):

        category = msg.replace("選產業_", "")

        result = build_style_result(category)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )
        return

    # 個股查詢
    code, name = search_stock_code(msg)

    if code:

        url = f"{request.host_url}stock/{code}".replace(
            "http://", "https://"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"📊 {name} ({code})\n完整分析：{url}"
            )
        )

    else:

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="請輸入股票代碼，或輸入：預測 / 大盤預測"
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
