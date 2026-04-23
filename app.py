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
# v2.0 正式版（動態分類系統）
# 更新內容：
# 1. 不再手寫 industry_map
# 2. 自動分類 ETF / 半導體 / 金融 / 航運 / AI伺服器...
# 3. 全市場 = 全部股票 + ETF
# 4. LINE 類別顯示數量
# 5. 保留完整分析頁
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
# 動態分類系統
# ==================================================
def build_market_map():

    market = {
        "全市場": [],
        "ETF專區": [],
        "半導體": [],
        "AI伺服器": [],
        "金融保險": [],
        "航運": [],
        "電子零組件": [],
        "通信網路": [],
        "光電": [],
        "傳產": [],
        "食品民生": [],
        "生技醫療": [],
        "綠能電力": [],
        "營建資產": [],
        "觀光百貨": []
    }

    for code, info in twstock.codes.items():

        name = info.name

        # 只收常見台股代碼
        if len(code) not in [4,5]:
            continue

        market["全市場"].append(code)

        # ETF
        if code.startswith("00"):
            market["ETF專區"].append(code)

        # 半導體
        if any(k in name for k in [
            "積體電","半導體","晶圓","矽","封測","IC"
        ]):
            market["半導體"].append(code)

        # AI伺服器
        if any(k in name for k in [
            "廣達","鴻海","緯創","英業達","仁寶","技嘉","微星","伺服器"
        ]):
            market["AI伺服器"].append(code)

        # 金融
        if any(k in name for k in [
            "金","銀行","保險","證券"
        ]):
            market["金融保險"].append(code)

        # 航運
        if any(k in name for k in [
            "航","運","海運","航空"
        ]):
            market["航運"].append(code)

        # 電子零組件
        if any(k in name for k in [
            "電","科技","電子","精密"
        ]):
            market["電子零組件"].append(code)

        # 通信
        if any(k in name for k in [
            "電信","網路","通訊","寬頻"
        ]):
            market["通信網路"].append(code)

        # 光電
        if any(k in name for k in [
            "光","鏡頭","面板","LED"
        ]):
            market["光電"].append(code)

        # 傳產
        if any(k in name for k in [
            "鋼","塑膠","水泥","汽車","橡膠","紡織"
        ]):
            market["傳產"].append(code)

        # 食品
        if any(k in name for k in [
            "食品","超商","餐飲","飲料"
        ]):
            market["食品民生"].append(code)

        # 生技
        if any(k in name for k in [
            "生技","醫","藥"
        ]):
            market["生技醫療"].append(code)

        # 綠能
        if any(k in name for k in [
            "電力","能源","綠能","太陽能","風電"
        ]):
            market["綠能電力"].append(code)

        # 營建
        if any(k in name for k in [
            "建設","營造","開發","資產"
        ]):
            market["營建資產"].append(code)

        # 觀光
        if any(k in name for k in [
            "飯店","觀光","百貨","旅行"
        ]):
            market["觀光百貨"].append(code)

    return market


# 啟動時建構一次
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
# 抓資料
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
# 特徵工程
# ==================================================
def add_features(df):

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    df["RET1"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df.dropna(inplace=True)

    return df


FEATURES = ["MA5","MA10","MA20","RET1","RSI"]

# ==================================================
# AI 分析
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
        plt.title(f"{stock_name} ({stock_code})")
        plt.grid(True)
        plt.legend()
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        current_price = float(df["Close"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])

        trend = "多頭" if current_price > float(df["MA20"].iloc[-1]) else "空頭"

        pred = (
            f"強勢看漲 📈 ({up_prob:.1f}%)"
            if up_prob > 60 else
            f"偏向看跌 📉 ({up_prob:.1f}%)"
            if up_prob < 40 else
            f"中性震盪 ⚖️ ({up_prob:.1f}%)"
        )

        text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 最新價格：{current_price:.2f}\n"
            f"🌡 RSI：{rsi:.1f}\n"
            f"📈 趨勢：{trend}\n\n"
            f"🤖 AI預測：{pred}"
        )

        return filename, text

    except:
        return None, "❌ 分析失敗"


# ==================================================
# 網站首頁
# ==================================================
@app.route("/")
def home():
    return "<h1>AI 台股系統 v2.0 動態分類運作中</h1>"


# ==================================================
# 個股頁
# ==================================================
@app.route("/stock/<stock_code>")
def stock_page(stock_code):

    stock_name = get_stock_name(stock_code)
    img_name, analysis = analyze_and_predict_stock(stock_code, stock_name)

    return f"<pre>{analysis}</pre>"


# ==================================================
# 圖片
# ==================================================
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

    if msg == "預測":

        items = []

        for ind in industry_map.keys():

            count = len(industry_map[ind])

            items.append(
                QuickReplyButton(
                    action=MessageAction(
                        label=f"{ind}({count})"[:20],
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

    if msg.startswith("選產業_"):

        ind = msg.replace("選產業_", "")

        stock_list = industry_map.get(ind, [])[:10]

        result = [f"📈 {ind}（{len(industry_map.get(ind, []))}檔）"]

        for code in stock_list:
            result.append(f"{code} {get_stock_name(code)}")

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="\n".join(result))
        )
        return

    target_code, target_name = (
        (msg, None)
        if msg.isdigit()
        else search_stock_code(msg)
    )

    if target_code:

        img_name, analysis = analyze_and_predict_stock(target_code, target_name)

        web_url = f"{request.host_url}stock/{target_code}".replace("http://","https://")

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"{analysis}\n\n完整分析：{web_url}"
            )
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
