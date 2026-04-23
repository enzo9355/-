# app.py
# v2.4 FinMind 原架構升級版
# --------------------------------------------------
# 說明：
# 1. 全資料來源回歸 FinMind
# 2. 個股 / ETF / 大盤 預測統一資料源
# 3. 保留：
#    - LINE Bot
#    - 預測分類
#    - 激進5 + 保守5
#    - 完整分析網址
# 4. 大盤預測改抓 FinMind 指數資料
# --------------------------------------------------

import os
import time
import urllib.request
import datetime
import requests
import pandas as pd
import numpy as np
import twstock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from flask import Flask, request, abort, send_from_directory

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    FlexSendMessage
)

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# ==================================================
# 基本設定
# ==================================================
font_path = "taipei_sans.ttf"

if not os.path.exists(font_path):
    urllib.request.urlretrieve(
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
        font_path
    )

fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

FINMIND_USER = os.getenv("FINMIND_USER")
FINMIND_PASSWORD = os.getenv("FINMIND_PASSWORD")

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

STATIC_PATH = "static/tmp"
os.makedirs(STATIC_PATH, exist_ok=True)

# ==================================================
# 模型設定
# ==================================================
LGBM_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": 42,
    "verbose": -1
}

FEATURES = ["MA20", "RET1", "RSI"]

# ==================================================
# FinMind Token
# ==================================================
finmind_token = ""

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


def cleanup_images():

    try:
        files = sorted(
            [os.path.join(STATIC_PATH, f)
             for f in os.listdir(STATIC_PATH)
             if f.endswith(".png")],
            key=os.path.getmtime
        )

        if len(files) > 100:
            for f in files[:50]:
                os.remove(f)

    except:
        pass


# ==================================================
# FinMind 抓個股 / ETF
# ==================================================
def get_stock_data(stock_code, days=365):

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

        data = requests.get(url, timeout=10).json()

        if data.get("msg") != "success":
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])

        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
            "trading_volume": "Volume"
        })

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except:
        return pd.DataFrame()


# ==================================================
# FinMind 抓大盤（加權指數）
# ==================================================
def get_taiex_data(days=365):

    finmind_login()

    start_date = (
        datetime.datetime.now()
        - datetime.timedelta(days=days)
    ).strftime("%Y-%m-%d")

    # FinMind 指數資料集
    # 若未來官方名稱更動，只需改 dataset 名稱即可
    dataset_names = [
        "TaiwanStockTotalReturnIndex",
        "TaiwanStockPriceIndex"
    ]

    for dataset in dataset_names:

        try:
            url = (
                "https://api.finmindtrade.com/api/v4/data"
                f"?dataset={dataset}"
                f"&data_id=TAIEX"
                f"&start_date={start_date}"
                f"&token={finmind_token}"
            )

            data = requests.get(url, timeout=10).json()

            if data.get("msg") == "success" and data.get("data"):

                df = pd.DataFrame(data["data"])

                # 兼容不同欄位
                close_col = None
                for c in ["close", "price", "value"]:
                    if c in df.columns:
                        close_col = c
                        break

                if not close_col:
                    continue

                df["Date"] = pd.to_datetime(df["date"])
                df.set_index("Date", inplace=True)

                df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
                df["Open"] = df["Close"]
                df["High"] = df["Close"]
                df["Low"] = df["Close"]
                df["Volume"] = 0

                return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        except:
            pass

    return pd.DataFrame()


# ==================================================
# 特徵工程
# ==================================================
def add_features(df):

    df["MA20"] = df["Close"].rolling(20).mean()
    df["RET1"] = df["Close"].pct_change()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df.dropna(inplace=True)

    return df


# ==================================================
# 個股分析
# ==================================================
def analyze_stock(stock_code):

    name = get_stock_name(stock_code)

    df = get_stock_data(stock_code, 365)

    if df.empty or len(df) < 80:
        return None, "❌ 資料不足"

    df = add_features(df)

    df["Future"] = df["Close"].shift(-5)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)

    train_df = df.dropna()

    scaler = StandardScaler()

    X = scaler.fit_transform(train_df[FEATURES])
    y = train_df["Target"]

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)

    latest = scaler.transform(df[FEATURES].iloc[-1:])
    up_prob = model.predict_proba(latest)[0][1] * 100

    close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    trend = "多頭" if close > ma20 else "空頭"

    text = (
        f"📊 {name} ({stock_code})\n\n"
        f"💰 價格：{close:.2f}\n"
        f"🌡 RSI：{rsi:.1f}\n"
        f"📈 趨勢：{trend}\n"
        f"🤖 上漲機率：{up_prob:.1f}%"
    )

    return None, text


# ==================================================
# 大盤預測（真 TAIEX）
# ==================================================
def market_forecast():

    df = get_taiex_data(365)

    if df.empty or len(df) < 80:
        return "❌ 無法取得 FinMind TAIEX 資料"

    df = add_features(df)

    close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    score = 0

    if close > ma20:
        score += 1

    if rsi < 70:
        score += 1

    if df["RET1"].iloc[-1] > 0:
        score += 1

    prob = 45 + score * 12

    if prob >= 70:
        comment = "偏強震盪 📈"
    elif prob >= 58:
        comment = "中性偏多 ⚖️"
    else:
        comment = "震盪偏弱 📉"

    trend = "多頭" if close > ma20 else "空頭"

    return (
        "📊 台股大盤預測（FinMind TAIEX）\n\n"
        f"💰 指數點位：{close:,.0f}\n"
        f"🌡 RSI：{rsi:.1f}\n"
        f"📈 趨勢：{trend}\n\n"
        f"🤖 AI判斷：{comment}\n"
        f"📌 上漲機率：約 {prob:.0f}%"
    )


# ==================================================
# 市場分類（簡化穩定版）
# ==================================================
industry_map = {
    "全市場": ["2330","2317","2454","2382","2882","2603","0050","00878"],
    "ETF專區": ["0050","0056","00878","00919","006208"],
    "半導體": ["2330","2454","2303","3711","3443"],
    "AI伺服器": ["2317","2382","3231","6669","3017"],
    "金融保險": ["2881","2882","2886","2891","2884"],
    "航運物流": ["2603","2609","2615","2618","5608"],
    "傳產民生": ["1101","1216","1301","2002","2207"],
    "生技醫療": ["4743","6446","4105","4137","4162"]
}


# ==================================================
# 激進5 + 保守5
# ==================================================
def build_style_result(category):

    stock_list = industry_map.get(category, [])

    aggressive = stock_list[:5]
    safe = stock_list[-5:]

    lines = [f"📈 {category} Top10\n"]

    lines.append("🔥 激進型 5 檔")
    for i, code in enumerate(aggressive, 1):
        lines.append(f"{i}. {code} {get_stock_name(code)}")

    lines.append("")
    lines.append("🛡 保守型 5 檔")
    for i, code in enumerate(safe, 1):
        lines.append(f"{i}. {code} {get_stock_name(code)}")

    return "\n".join(lines)


# ==================================================
# 網站
# ==================================================
@app.route("/")
def home():
    return "<h1>AI 台股系統 v2.4 FinMind 原架構版</h1>"


@app.route("/stock/<stock_code>")
def stock_page(stock_code):

    _, text = analyze_stock(stock_code)

    return f"<pre>{text}</pre>"


@app.route("/static/tmp/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_PATH, filename)


# ==================================================
# LINE Webhook
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
# LINE 訊息
# ==================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    msg = event.message.text.strip()

    # 大盤預測
    if msg == "大盤預測":

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=market_forecast())
        )
        return

    # 分類預測
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

    # 分類結果
    if msg.startswith("選產業_"):

        category = msg.replace("選產業_", "")

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=build_style_result(category))
        )
        return

    # 個股查詢
    target_code, _ = search_stock_code(msg)

    if target_code:

        _, text = analyze_stock(target_code)

        web_url = f"{request.host_url}stock/{target_code}".replace(
            "http://", "https://"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"{text}\n\n完整分析：{web_url}"
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
