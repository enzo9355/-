# app.py
# v2.2 正式版（產業擴充 + 大盤預測）
# 可直接整份覆蓋

import os
import time
import urllib.request
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

import requests
import datetime
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
# AI 參數
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
# 工具函數
# ==================================================
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
# 動態分類（v2.2 擴充版）
# ==================================================
def build_market_map():

    market = {
        "全市場": [],
        "ETF專區": [],
        "半導體": [],
        "AI伺服器": [],
        "金融保險": [],
        "航運物流": [],
        "電子零組件": [],
        "通信網路": [],
        "光電面板": [],
        "電機機械": [],
        "鋼鐵塑化": [],
        "食品民生": [],
        "生技醫療": [],
        "營建資產": [],
        "觀光百貨": [],
        "綠能電力": []
    }

    for code, info in twstock.codes.items():

        name = info.name

        if len(code) not in [4, 5]:
            continue

        market["全市場"].append(code)

        # ETF
        if code.startswith("00"):
            market["ETF專區"].append(code)

        if any(k in name for k in ["半導體", "晶圓", "IC", "矽", "積體電"]):
            market["半導體"].append(code)

        if any(k in name for k in ["廣達", "鴻海", "緯創", "仁寶", "技嘉", "微星", "伺服器"]):
            market["AI伺服器"].append(code)

        if any(k in name for k in ["金", "銀行", "保險", "證券"]):
            market["金融保險"].append(code)

        if any(k in name for k in ["航", "運", "海運", "航空", "物流"]):
            market["航運物流"].append(code)

        if any(k in name for k in ["電子", "科技", "精密", "零組件"]):
            market["電子零組件"].append(code)

        if any(k in name for k in ["電信", "通訊", "網路", "寬頻"]):
            market["通信網路"].append(code)

        if any(k in name for k in ["光", "面板", "鏡頭", "LED"]):
            market["光電面板"].append(code)

        if any(k in name for k in ["電機", "機械", "工業"]):
            market["電機機械"].append(code)

        if any(k in name for k in ["鋼", "塑膠", "化學", "水泥"]):
            market["鋼鐵塑化"].append(code)

        if any(k in name for k in ["食品", "餐飲", "飲料", "超商"]):
            market["食品民生"].append(code)

        if any(k in name for k in ["醫", "藥", "生技"]):
            market["生技醫療"].append(code)

        if any(k in name for k in ["建設", "營造", "開發", "資產"]):
            market["營建資產"].append(code)

        if any(k in name for k in ["觀光", "百貨", "飯店", "旅行"]):
            market["觀光百貨"].append(code)

        if any(k in name for k in ["電力", "能源", "綠能", "太陽能", "風電"]):
            market["綠能電力"].append(code)

    return market


industry_map = build_market_map()

# ==================================================
# FinMind 登入
# ==================================================
finmind_token = ""


def auto_login_finmind():

    global finmind_token

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

    except:
        pass


# ==================================================
# 抓資料
# ==================================================
def get_taiwan_stock_data(stock_code, period_days=365):

    global finmind_token

    start_date = (
        datetime.datetime.now()
        - datetime.timedelta(days=period_days)
    ).strftime("%Y-%m-%d")

    def fetch(token):

        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={stock_code}&start_date={start_date}"

        if token:
            url += f"&token={token}"

        return requests.get(url, timeout=10).json()

    try:

        if FINMIND_USER and not finmind_token:
            auto_login_finmind()

        data = fetch(finmind_token)

        if data.get("msg") == "success" and data.get("data"):

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
# AI 分析（個股）
# ==================================================
def analyze_and_predict_stock(stock_code, stock_name=None):

    try:
        stock_name = stock_name or get_stock_name(stock_code)

        df = get_taiwan_stock_data(stock_code, 365)

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

        cleanup_images()

        file_name = f"{stock_code}_{int(time.time())}.png"
        file_path = os.path.join(STATIC_PATH, file_name)

        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-60:], df["Close"].iloc[-60:], label="收盤價")
        plt.plot(df.index[-60:], df["MA20"].iloc[-60:], label="MA20")
        plt.legend()
        plt.grid(True)
        plt.title(f"{stock_name} ({stock_code})")
        plt.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close()

        close = float(df["Close"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])
        ma20 = float(df["MA20"].iloc[-1])

        trend = "多頭" if close > ma20 else "空頭"

        text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 價格：{close:.2f}\n"
            f"🌡 RSI：{rsi:.1f}\n"
            f"📈 趨勢：{trend}\n"
            f"🤖 上漲機率：{up_prob:.1f}%"
        )

        return file_name, text

    except:
        return None, "❌ 分析失敗"


# ==================================================
# 大盤預測（用0050代理）
# ==================================================
def market_forecast():

    try:
        df = get_taiwan_stock_data("0050", 365)

        if df.empty or len(df) < 80:
            return "❌ 大盤資料不足"

        df = add_features(df)

        close = float(df["Close"].iloc[-1])
        ma20 = float(df["MA20"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])

        trend = "多頭" if close > ma20 else "空頭"

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

        return (
            "📊 台股大盤預測（0050代理）\n\n"
            f"💰 指標價格：{close:.2f}\n"
            f"🌡 RSI：{rsi:.1f}\n"
            f"📈 趨勢：{trend}\n\n"
            f"🤖 AI判斷：{comment}\n"
            f"📌 上漲機率：約 {prob:.0f}%"
        )

    except:
        return "❌ 大盤預測失敗"


# ==================================================
# 分類推薦（激進5 + 保守5）
# ==================================================
def build_style_result(category):

    stock_list = industry_map.get(category, [])[:30]

    aggr = []
    safe = []

    for code in stock_list:

        try:
            df = get_taiwan_stock_data(code, 120)

            if df.empty or len(df) < 40:
                continue

            df = add_features(df)

            close = float(df["Close"].iloc[-1])
            ma20 = float(df["MA20"].iloc[-1])
            rsi = float(df["RSI"].iloc[-1])
            vol = df["RET1"].std()

            score_aggr = vol * 100 + (close / ma20)
            score_safe = (close / ma20) - vol * 30 - abs(rsi - 55) / 100

            aggr.append((score_aggr, code))
            safe.append((score_safe, code))

        except:
            continue

    aggr = sorted(aggr, reverse=True)[:5]
    safe = sorted(safe, reverse=True)[:5]

    lines = [f"📈 {category} Top10\n"]

    lines.append("🔥 激進型 5 檔")
    rank = 1
    for _, code in aggr:
        lines.append(f"{rank}. {code} {get_stock_name(code)}")
        rank += 1

    lines.append("")
    lines.append("🛡 保守型 5 檔")
    rank = 1
    for _, code in safe:
        lines.append(f"{rank}. {code} {get_stock_name(code)}")
        rank += 1

    return "\n".join(lines)


# ==================================================
# 網站
# ==================================================
@app.route("/")
def home():
    return "<h1>AI 台股系統 v2.2 運行中</h1>"


@app.route("/stock/<stock_code>")
def stock_page(stock_code):

    stock_name = get_stock_name(stock_code)
    _, text = analyze_and_predict_stock(stock_code, stock_name)

    return f"<pre>{text}</pre>"


@app.route("/static/tmp/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_PATH, filename)


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

    # ------------------------------
    # 大盤預測
    # ------------------------------
    if msg == "大盤預測":

        result = market_forecast()

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )
        return

    # ------------------------------
    # 預測分類
    # ------------------------------
    if msg == "預測":

        items = []

        keys = list(industry_map.keys())[:13]

        for ind in keys:
            items.append(
                QuickReplyButton(
                    action=MessageAction(
                        label=ind[:20],
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

    # ------------------------------
    # 分類結果
    # ------------------------------
    if msg.startswith("選產業_"):

        category = msg.replace("選產業_", "")

        result = build_style_result(category)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )
        return

    # ------------------------------
    # 個股查詢
    # ------------------------------
    target_code, target_name = (
        (msg, None)
        if msg.isdigit()
        else search_stock_code(msg)
    )

    if target_code:

        img_name, analysis = analyze_and_predict_stock(
            target_code,
            target_name
        )

        web_url = f"{request.host_url}stock/{target_code}".replace("http://", "https://")

        if img_name:

            hero_url = f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://")

            flex = {
                "type": "bubble",

                "hero": {
                    "type": "image",
                    "url": hero_url,
                    "size": "full",
                    "aspectRatio": "10:6",
                    "aspectMode": "cover"
                },

                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": analysis,
                            "wrap": True,
                            "size": "sm"
                        }
                    ]
                },

                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "button",
                            "style": "primary",
                            "action": {
                                "type": "uri",
                                "label": "查看完整分析",
                                "uri": web_url
                            }
                        }
                    ]
                }
            }

            line_bot_api.reply_message(
                event.reply_token,
                FlexSendMessage(
                    alt_text="股票分析",
                    contents=flex
                )
            )

        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=analysis)
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
