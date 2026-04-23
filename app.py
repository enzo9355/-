import os
import time
import urllib.request
import pandas as pd
import numpy as np
import twstock
import matplotlib
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
import requests
import datetime

# ==================================================
# v2.1 正式版（雙風格10檔）
# 更新內容：
# 1. 預測可正常使用（Quick Reply 類別精簡）
# 2. 每類輸出 10 檔 = 激進5 + 保守5
# 3. 動態分類（不手寫股票池）
# 4. 保留個股分析 + 完整分析網站
# ==================================================

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
matplotlib.use("Agg")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FINMIND_USER = os.getenv("FINMIND_USER")
FINMIND_PASSWORD = os.getenv("FINMIND_PASSWORD")

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

static_tmp_path = "static/tmp"
os.makedirs(static_tmp_path, exist_ok=True)

# ==================================================
# 模型參數
# ==================================================
LGBM_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": 42,
    "verbose": -1
}

# ==================================================
# 工具函數
# ==================================================
def cleanup_images():
    try:
        files = sorted(
            [os.path.join(static_tmp_path, f)
             for f in os.listdir(static_tmp_path)
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

    keyword = keyword.upper()

    if keyword.isdigit():
        return keyword, get_stock_name(keyword)

    for code, info in twstock.codes.items():
        if keyword in info.name.upper():
            return code, info.name

    return None, None


# ==================================================
# 動態分類（精簡 8 類，避免 LINE 超限）
# ==================================================
def build_market_map():

    market = {
        "全市場": [],
        "ETF專區": [],
        "半導體": [],
        "AI伺服器": [],
        "金融保險": [],
        "航運": [],
        "傳產民生": [],
        "生技醫療": []
    }

    for code, info in twstock.codes.items():

        name = info.name

        if len(code) not in [4,5]:
            continue

        market["全市場"].append(code)

        if code.startswith("00"):
            market["ETF專區"].append(code)

        if any(k in name for k in ["半導體","晶圓","IC","矽","積體電"]):
            market["半導體"].append(code)

        if any(k in name for k in ["廣達","鴻海","緯創","仁寶","技嘉","微星"]):
            market["AI伺服器"].append(code)

        if any(k in name for k in ["金","銀行","保險","證券"]):
            market["金融保險"].append(code)

        if any(k in name for k in ["航","運","海運","航空"]):
            market["航運"].append(code)

        if any(k in name for k in ["食品","塑膠","水泥","鋼","汽車","紡織"]):
            market["傳產民生"].append(code)

        if any(k in name for k in ["醫","藥","生技"]):
            market["生技醫療"].append(code)

    return market


industry_map = build_market_map()

# ==================================================
# FinMind
# ==================================================
finmind_auto_token = ""


def auto_login_finmind():

    global finmind_auto_token

    if not FINMIND_USER or not FINMIND_PASSWORD:
        return

    try:
        res = requests.post(
            "https://api.finmindtrade.com/api/v4/login",
            data={
                "user_id": FINMIND_USER,
                "password": FINMIND_PASSWORD
            },
            timeout=10
        ).json()

        if res.get("msg") == "success":
            finmind_auto_token = res["token"]

    except:
        pass


# ==================================================
# 股價資料
# ==================================================
def get_taiwan_stock_data(stock_code, period_days=365):

    global finmind_auto_token

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

        if FINMIND_USER and not finmind_auto_token:
            auto_login_finmind()

        data = fetch(finmind_auto_token)

        if data.get("msg") == "success" and data.get("data"):

            df = pd.DataFrame(data["data"])

            df = df.rename(columns={
                "date":"Date",
                "open":"Open",
                "max":"High",
                "min":"Low",
                "close":"Close",
                "Trading_Volume":"Volume",
                "trading_volume":"Volume"
            })

            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            for col in ["Open","High","Low","Close","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[["Open","High","Low","Close","Volume"]].dropna()

    except:
        pass

    return pd.DataFrame()


# ==================================================
# 特徵
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


FEATURES = ["MA20","RET1","RSI"]

# ==================================================
# AI 個股分析
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

        filename = f"{stock_code}_{int(time.time())}.png"
        filepath = os.path.join(static_tmp_path, filename)

        plt.figure(figsize=(10,6))
        plt.plot(df.index[-60:], df["Close"].iloc[-60:], label="收盤價")
        plt.plot(df.index[-60:], df["MA20"].iloc[-60:], label="MA20")
        plt.legend()
        plt.grid(True)
        plt.title(f"{stock_name} ({stock_code})")
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        current_price = float(df["Close"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])

        trend = "多頭" if current_price > float(df["MA20"].iloc[-1]) else "空頭"

        text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 價格：{current_price:.2f}\n"
            f"🌡 RSI：{rsi:.1f}\n"
            f"📈 趨勢：{trend}\n"
            f"🤖 上漲機率：{up_prob:.1f}%"
        )

        return filename, text

    except:
        return None, "❌ 分析失敗"


# ==================================================
# 風格排名（激進5 + 保守5）
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

            volatility = df["RET1"].std()

            score_aggr = volatility * 100 + (close / ma20)
            score_safe = (close / ma20) - volatility * 30 - abs(rsi - 55) / 100

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
    return "<h1>AI 台股系統 v2.1 雙風格版運行中</h1>"


@app.route("/stock/<stock_code>")
def stock_page(stock_code):

    stock_name = get_stock_name(stock_code)
    img_name, analysis = analyze_and_predict_stock(stock_code, stock_name)

    return f"<pre>{analysis}</pre>"


@app.route("/static/tmp/<path:filename>")
def serve_static(filename):
    return send_from_directory(static_tmp_path, filename)


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

    # 預測
    if msg == "預測":

        items = []

        for ind in industry_map.keys():

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

    # 類別 Top10
    if msg.startswith("選產業_"):

        category = msg.replace("選產業_", "")

        result = build_style_result(category)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )
        return

    # 個股分析
    target_code, target_name = (
        (msg, None)
        if msg.isdigit()
        else search_stock_code(msg)
    )

    if target_code:

        img_name, analysis = analyze_and_predict_stock(target_code, target_name)

        web_url = f"{request.host_url}stock/{target_code}".replace("http://","https://")

        if img_name:

            hero_url = f"{request.host_url}static/tmp/{img_name}".replace("http://","https://")

            flex = {
                "type":"bubble",

                "hero":{
                    "type":"image",
                    "url":hero_url,
                    "size":"full",
                    "aspectRatio":"10:6",
                    "aspectMode":"cover"
                },

                "body":{
                    "type":"box",
                    "layout":"vertical",
                    "contents":[
                        {
                            "type":"text",
                            "text":analysis,
                            "wrap":True,
                            "size":"sm"
                        }
                    ]
                },

                "footer":{
                    "type":"box",
                    "layout":"vertical",
                    "contents":[
                        {
                            "type":"button",
                            "style":"primary",
                            "action":{
                                "type":"uri",
                                "label":"查看完整分析",
                                "uri":web_url
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
            TextSendMessage(text="輸入股票代碼，或輸入 預測")
        )


# ==================================================
# 啟動
# ==================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
