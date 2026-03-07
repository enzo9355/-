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
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage,
    QuickReply, QuickReplyButton, MessageAction, FlexSendMessage
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import requests
import datetime

# ==========================================
# 1. 核心設定與字體下載
# ==========================================
font_path = 'taipei_sans.ttf'
if not os.path.exists(font_path):
    print("⏳ 下載中文字體...")
    urllib.request.urlretrieve("https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf", font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
matplotlib.use('Agg')

LINE_CHANNEL_ACCESS_TOKEN = 'lQyeonM1HkZGZXABONH+Xpd9atZVkppAIt5qnCZkz8D131NdHiW06EmtXXSQyJ2rc8CCbylLOBZLb+zbqvynFtkzGpp/7X0+MDLbk2FD3oMTATtUw2Kpf+PzMtpx07ofZ0vC9Do2KVYQN1Tl328otAdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'e5370d4d8f54d87f04a5cced565c1d4b'

# 移除 ETF，保留其他產業
industry_map = {
    "半導體業": ['2330', '2454', '2303', '3034', '3711', '3443', '2408', '3035', '3006', '3532'],
    "電腦周邊": ['2317', '2382', '3231', '2357', '2356', '2324', '6669', '2353', '2377', '2352'],
    "通信網路": ['2412', '3045', '4904', '2345', '5388', '3062', '2455', '3702', '6285', '3596'],
    "光電產業": ['3008', '2409', '3481', '6706', '4956', '6176', '3406', '2448', '3504', '3019'],
    "電子零組件": ['2308', '2327', '3037', '2383', '2059', '3042', '3044', '2492', '2313', '2316'],
    "金融保險": ['2881', '2882', '2886', '2891', '5880', '2892', '2884', '2885', '2880', '2890'],
    "航運業": ['2603', '2609', '2615', '2618', '2610', '2606', '2637', '2633', '5608', '2605'],
    "鋼鐵工業": ['2002', '2014', '2006', '2027', '2031', '2023', '2015', '2009', '2034'],
    "塑膠化學": ['1301', '1303', '6505', '1326', '1304', '1308', '1312', '1310', '1313', '4739']
}

all_watch_list = []
for stocks in industry_map.values():
    all_watch_list.extend(stocks)

# ==========================================
# 2. 資料獲取與處理 (FinMind API)
# ==========================================
def get_stock_name(stock_code):
    try:
        if stock_code in twstock.codes: return twstock.codes[stock_code].name
        return stock_code
    except: return stock_code

def search_stock_code(keyword):
    keyword = keyword.upper()
    for code in all_watch_list:
        name = get_stock_name(code)
        if keyword == name or keyword in name: return code, name
    for code, info in twstock.codes.items():
        if keyword == info.name or keyword in info.name:
            if len(code) <= 6: return code, info.name
    return None, None

def get_taiwan_stock_data(stock_code, period_days):
    """透過 FinMind API 獲取台股歷史資料"""
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=period_days)).strftime('%Y-%m-%d')
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={stock_code}&start_date={start_date}"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df = pd.DataFrame(data["data"])
            df = df.rename(columns={'close': 'Close', 'date': 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            return df[['Close']].dropna()
    except Exception as e:
        print(f"FinMind API 抓取失敗 ({stock_code}): {e}")
    return pd.DataFrame()

def analyze_and_predict_stock(stock_code, stock_name=None):
    try:
        if not stock_name: stock_name = get_stock_name(stock_code)

        df = get_taiwan_stock_data(stock_code, 365)
        if df.empty or len(df) < 30: return None, None

        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Momentum'] = df['Close'].pct_change(periods=5)
        
        df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['Target'] = np.where(df['Future_Return'] > 0, 1, 0)

        train_df = df.dropna()
        features = ['MA_10', 'MA_20', 'Volatility', 'Momentum']
        X = train_df[features]
        y = train_df['Target']

        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(X, y)

        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        up_probability = rf_model.predict_proba(latest_features)[0][1] * 100

        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='收盤價', color='black', linewidth=2)
        plt.plot(df.index[-60:], df['MA_10'].iloc[-60:], label='10日線', color='blue', linestyle='--')
        plt.plot(df.index[-60:], df['MA_20'].iloc[-60:], label='20日線(月線)', color='red', linestyle='-.')
        plt.title(f'{stock_code} {stock_name} - AI 預測模型', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('價格', fontsize=12)
        plt.legend(prop={'size': 12})
        plt.grid(True, alpha=0.3)

        filename = f"{stock_code}_{int(time.time())}.png"
        filepath = os.path.join(static_tmp_path, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()

        current_price = float(df['Close'].iloc[-1])
        ma20 = float(df['MA_20'].iloc[-1])
        
        if up_probability > 60: pred_msg = f"強勢看漲 📈 ({up_probability:.1f}%)"
        elif up_probability < 40: pred_msg = f"弱勢看跌 📉 ({100-up_probability:.1f}%)"
        else: pred_msg = f"震盪整理 ⚖️ ({up_probability:.1f}%)"

        analysis_text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 收盤價：{current_price:.2f}\n"
            f"🌊 月線值：{ma20:.2f}\n"
            f"⚡ 狀態：{'站上月線 (多頭)' if current_price > ma20 else '跌破月線 (空頭)'}\n\n"
            f"🤖 5日預測：{pred_msg}\n\n"
            f"👆 點擊上方圖表，查看進階策略"
        )
        return filename, analysis_text
    except Exception as e: 
        print(f"分析失敗: {e}")
        return None, None

def calculate_backtest(stock_code, stock_name=""):
    try:
        df = get_taiwan_stock_data(stock_code, 730)
        if df.empty or len(df) < 100:
            return "❌ 資料不足，無法進行回測計算。"

        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Momentum'] = df['Close'].pct_change(periods=5)
        
        df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = np.where(df['Next_Return'] > 0, 1, 0)
        
        df = df.dropna()
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:].copy()
        
        features = ['MA_10', 'MA_20', 'Volatility', 'Momentum']
        X_train, y_train = train_df[features], train_df['Target']
        X_test = test_df[features]
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(X_train, y_train)
        
        test_df['Prob'] = rf_model.predict_proba(X_test)[:, 1]
        test_df['Signal'] = np.where(test_df['Prob'] > 0.55, 1, 0)
        
        test_df['Strategy_Return'] = test_df['Signal'] * test_df['Next_Return']
        test_df['B&H_Return'] = test_df['Next_Return']
        
        strategy_cum = (1 + test_df['Strategy_Return']).cumprod().iloc[-1] - 1
        bnh_cum = (1 + test_df['B&H_Return']).cumprod().iloc[-1] - 1
        
        trades = test_df[test_df['Signal'] == 1]
        win_rate = (trades['Strategy_Return'] > 0).mean() * 100 if len(trades) > 0 else 0
        
        roll_max = (1 + test_df['Strategy_Return']).cumprod().cummax()
        drawdown = (1 + test_df['Strategy_Return']).cumprod() / roll_max - 1
        mdd = drawdown.min() * 100
        
        std_dev = test_df['Strategy_Return'].std()
        sharpe = (test_df['Strategy_Return'].mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        test_days = len(test_df)
        
        if strategy_cum > bnh_cum:
            if sharpe > 1: conclusion = "✅ AI 表現優異：走勢規律，AI 抓得很準。\n🛒 想買入：優質標的！若最新預測看漲可進場。\n💰 想賣出：預測看漲就抱著，看跌請獲利了結。"
            else: conclusion = "✅ AI 表現贏過大盤：長期會賺，但過程震盪。\n🛒 想買入：可以買，建議分批往下買進。\n💰 想賣出：見好就收！不想承受震盪建議先停利一半。"
        else:
            if mdd > -15: conclusion = "⏸️ AI 表現防禦：賺得比死抱著少，大跌時能保命。\n🛒 想買入：適合保守存股，想賺快錢別碰。\n💰 想賣出：若不想資金卡住可考慮換股。"
            else: conclusion = "⚠️ AI 表現不佳：容易被雙巴。\n🛒 想買入：請尋找其他 AI 表現優異的標的。\n💰 想賣出：看線圖手動停損，破月線請賣出。"

        res_text = (
            f"📑 {stock_name} ({stock_code}) 策略回測\n"
            f"⏳ 測試期間：近 {test_days} 個交易日\n\n"
            f"📊 實際績效\n"
            f"👑 跟著 AI 買賣：{strategy_cum*100:.2f}%\n"
            f"🗿 死抱著不放：{bnh_cum*100:.2f}%\n\n"
            f"🛡️ 風險評估\n"
            f"🎯 AI 勝率：{win_rate:.1f}%\n"
            f"⚠️ 最慘曾跌掉：{mdd:.2f}%\n"
            f"⚖️ CP值：{sharpe:.2f}\n\n"
            f"💡 行動指南：\n{conclusion}"
        )
        return res_text
    except Exception as e:
        return "❌ 回測計算發生錯誤，請確認資料是否齊全。"

# ==========================================
# 3. Flask 伺服器與 LINE Webhook 路由
# ==========================================
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

static_tmp_path = 'static/tmp'
if not os.path.exists(static_tmp_path): os.makedirs(static_tmp_path)

@app.route('/static/tmp/<path:filename>')
def serve_static(filename): return send_from_directory(static_tmp_path, filename)

@app.route("/")
def home():
    return "LINE Bot 正常運作中！"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except InvalidSignatureError: abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.strip()

    if msg == "教學":
        reply_text = "🔍 個股查詢教學\n\n直接在聊天框輸入：\n👉 股票代碼（例：2330）\n👉 股票名稱（例：台積電）\n\n系統將自動產出 AI 技術線圖與漲跌預測！"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    elif msg == "免責聲明":
        reply_text = "⚠️ 免責聲明\n\n本系統數據僅供學術研究與程式開發交流。\n模型歷史勝率不代表未來績效，不構成買賣建議。投資一定有風險，下單前請自行評估並嚴設停損。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    elif msg == "預測":
        items = [QuickReplyButton(action=MessageAction(label="全市場", text="選產業_全市場"))]
        for industry in industry_map.keys():
            items.append(QuickReplyButton(action=MessageAction(label=industry[:20], text=f"選產業_{industry}")))
        
        # 新增大盤預測按鈕
        items.append(QuickReplyButton(action=MessageAction(label="📊 台股大盤預測", text="大盤預測")))

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請選擇想分析的產業類別：👇", quick_reply=QuickReply(items=items))
        )

    # 處理大盤專屬通道
    elif msg == "大盤預測":
        img_name, analysis_txt = analyze_and_predict_stock("TAIEX", "台股加權指數(大盤)")
        if img_name and analysis_txt:
            img_url = f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://")
            flex_content = {
                "type": "bubble",
                "hero": {
                    "type": "image", "url": img_url, "size": "full", "aspectRatio": "10:6", "aspectMode": "cover",
                    "action": {"type": "message", "label": "action", "text": "詳細策略_TAIEX"}
                },
                "body": {
                    "type": "box", "layout": "vertical",
                    "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]
                }
            }
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="台股大盤預測", contents=flex_content))
        else: 
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 大盤資料獲取失敗，請稍後再試。"))

    elif msg.startswith("選產業_"):
        target_industry = msg.split("_")[1]
        items = [
            QuickReplyButton(action=MessageAction(label="🛡️ 穩健 (重風險控管)", text=f"分析_{target_industry}_穩健")),
            QuickReplyButton(action=MessageAction(label="⚔️ 激進 (追高報酬)", text=f"分析_{target_industry}_激進"))
        ]
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"已鎖定【{target_industry}】\n\n請選擇您的「風險偏好」：", quick_reply=QuickReply(items=items))
        )

    elif msg.startswith("分析_"):
        parts = msg.split("_")
        target_industry = parts[1]
        risk_type = parts[2] if len(parts) > 2 else "穩健"
        
        if target_industry == "全市場": codes = ['2330', '2317', '2454', '2308', '2881', '2603', '2002', '1301', '0050', '0056']
        else: codes = industry_map.get(target_industry, [])

        stock_features = []
        for code in codes:
            try:
                df_hist = get_taiwan_stock_data(code, 90)
                if not df_hist.empty and len(df_hist) > 10:
                    series = df_hist['Close']
                    start_p, end_p = float(series.iloc[0]), float(series.iloc[-1])
                    ret = (end_p - start_p) / start_p
                    vol = series.pct_change().std()
                    stock_features.append({'Code': code, 'Name': get_stock_name(code), 'Return': ret, 'Volatility': vol})
            except Exception as e:
                print(f"動態抓取失敗 {code}: {e}")

        df_target = pd.DataFrame(stock_features)

        if len(df_target) >= 5:
            X = df_target[['Return', 'Volatility']]
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                df_target['Cluster'] = kmeans.fit_predict(X)
                
                if risk_type == "穩健":
                    chosen_cluster = df_target.groupby('Cluster')['Volatility'].mean().idxmin()
                    potential_stocks = df_target[df_target['Cluster'] == chosen_cluster].copy()
                    potential_stocks['Score'] = potential_stocks['Return'] / (potential_stocks['Volatility'] + 0.0001)
                    top_5 = potential_stocks.sort_values('Score', ascending=False).head(5)
                else:
                    chosen_cluster = df_target.groupby('Cluster')['Return'].mean().idxmax()
                    potential_stocks = df_target[df_target['Cluster'] == chosen_cluster].copy()
                    top_5 = potential_stocks.sort_values('Return', ascending=False).head(5)
            except: 
                top_5 = df_target.sort_values('Return', ascending=False).head(5)
        else: 
            if not df_target.empty: top_5 = df_target.sort_values('Return', ascending=False).head(5)
            else: top_5 = pd.DataFrame()

        if top_5.empty:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料不足或抓取過於頻繁，請稍後再試。"))
            return

        reply_text = f"🚀 AI 智選｜{target_industry}\n🎯 風格：{risk_type}\n\n"
        emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for i, (_, row) in enumerate(top_5.iterrows()):
            reply_text += f"{emoji_list[i]} {row['Name']} ({row['Code']})\n"
            reply_text += f"📈 報酬: {row['Return']*100:.1f}% ｜ ⚡ 波動: {row['Volatility']:.2f}\n\n"
            
        reply_text += "💡 提示：輸入股票代碼即可查看預測圖表！"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    elif msg.startswith("詳細策略_"):
        stock_code = msg.split("_")[1]
        stock_name = "台股加權指數(大盤)" if stock_code == "TAIEX" else get_stock_name(stock_code)
        backtest_report = calculate_backtest(stock_code, stock_name)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=backtest_report))

    else:
        target_code, target_name = None, None
        if msg.isdigit() and 4 <= len(msg) <= 6: target_code = msg
        else: target_code, target_name = search_stock_code(msg)

        if target_code:
            img_name, analysis_txt = analyze_and_predict_stock(target_code, target_name)
            if img_name and analysis_txt:
                img_url = f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://")
                flex_content = {
                    "type": "bubble",
                    "hero": {
                        "type": "image", "url": img_url, "size": "full", "aspectRatio": "10:6", "aspectMode": "cover",
                        "action": {"type": "message", "label": "action", "text": f"詳細策略_{target_code}"}
                    },
                    "body": {
                        "type": "box", "layout": "vertical",
                        "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]
                    }
                }
                line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text=f"{target_name} AI 分析", contents=flex_content))
            else: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料不足，無法分析此股票。"))
        else: line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"❌ 找不到「{msg}」，請輸入正確代碼或名稱。"))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

