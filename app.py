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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
import requests
import datetime

# ==========================================
# 1. 核心設定與字體
# ==========================================
font_path = 'taipei_sans.ttf'
if not os.path.exists(font_path):
    urllib.request.urlretrieve("https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf", font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
matplotlib.use('Agg')

LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
FINMIND_USER = os.getenv('FINMIND_USER')
FINMIND_PASSWORD = os.getenv('FINMIND_PASSWORD')

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

all_watch_list = [stock for stocks in industry_map.values() for stock in stocks]

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

static_tmp_path = 'static/tmp'
os.makedirs(static_tmp_path, exist_ok=True)

def cleanup_images():
    try:
        files = [os.path.join(static_tmp_path, f) for f in os.listdir(static_tmp_path) if f.endswith('.png')]
        if len(files) > 100:
            files.sort(key=os.path.getmtime)
            for f in files[:50]: os.remove(f)
    except Exception as e:
        print(f"清理圖片失敗: {e}")

# ==========================================
# 2. 資料獲取與特徵工程
# ==========================================
finmind_auto_token = ""

def auto_login_finmind():
    global finmind_auto_token
    if not FINMIND_USER or not FINMIND_PASSWORD: return
    try:
        url = "https://api.finmindtrade.com/api/v4/login"
        payload = {"user_id": FINMIND_USER, "password": FINMIND_PASSWORD}
        res = requests.post(url, data=payload, timeout=10)
        data = res.json()
        if data.get("msg") == "success":
            finmind_auto_token = data.get("token")
    except Exception as e: pass

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

def get_taiwan_stock_data(stock_code, period_days=730):
    global finmind_auto_token
    try:
        if FINMIND_USER and not finmind_auto_token:
            auto_login_finmind()
            
        start_date = (datetime.datetime.now() - datetime.timedelta(days=period_days)).strftime('%Y-%m-%d')
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={stock_code}&start_date={start_date}"
        if finmind_auto_token: url += f"&token={finmind_auto_token}"
            
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") != 200 or "expired" in str(data.get("msg", "")).lower():
            auto_login_finmind()
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={stock_code}&start_date={start_date}"
            if finmind_auto_token: url += f"&token={finmind_auto_token}"
            res = requests.get(url, timeout=10)
            data = res.json()

        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df = pd.DataFrame(data["data"])
            df = df.rename(columns={
                'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 
                'volume': 'Volume', 'Trading_Volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception as e: pass
    return pd.DataFrame()

def add_advanced_features(df):
    if len(df) < 60: return df
    
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_60'] = df['Close'].rolling(60).mean()
    df['RET_1'] = df['Close'].pct_change()
    df['RET_5'] = df['Close'].pct_change(5)
    df['Volatility'] = df['RET_1'].rolling(20).std()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['RS_60'] = df['Close'] / df['Close'].rolling(60).mean()
    
    tr = np.maximum(df['High'] - df['Low'], 
         np.maximum(abs(df['High'] - df['Close'].shift()), 
                    abs(df['Low'] - df['Close'].shift())))
    df['ATR_14'] = tr.rolling(14).mean()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()

FEATURES = ['MA_5', 'MA_10', 'MA_20', 'MA_60', 'RET_1', 'RET_5', 'Volatility', 
            'RSI_14', 'MACD', 'MACD_Signal', 'Volume_MA20', 'OBV', 'RS_60', 'ATR_14']

# ==========================================
# 3. 預測與回測函數
# ==========================================
def analyze_and_predict_stock(stock_code, stock_name=None):
    try:
        if not stock_name: stock_name = get_stock_name(stock_code)
        df = get_taiwan_stock_data(stock_code, 730)
        if df.empty or len(df) < 100: return None, None
        
        df = add_advanced_features(df)
        if len(df) < 60: return None, None
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        if len(df) < 50: return None, None
        
        train_df = df.iloc[:-1]
        latest_features = df[FEATURES].iloc[-1:]
        
        X = train_df[FEATURES]
        y = train_df['Target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        model.fit(X_scaled, y)
        
        latest_scaled = scaler.transform(latest_features)
        up_prob = model.predict_proba(latest_scaled)[0][1] * 100
        
        cleanup_images() 
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='收盤價', color='black', linewidth=2)
        plt.plot(df.index[-60:], df['MA_10'].iloc[-60:], label='10日均線', color='blue', linestyle='--')
        plt.plot(df.index[-60:], df['MA_20'].iloc[-60:], label='20日均線', color='red', linestyle='-.')
        plt.title(f'{stock_code} {stock_name} - LGBM 預測', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('價格', fontsize=12)
        plt.legend(prop={'size': 12})
        plt.grid(True, alpha=0.3)
        
        filename = f"{stock_code}_{int(time.time())}.png"
        filepath = os.path.join(static_tmp_path, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        current_price = float(df['Close'].iloc[-1])
        ma20 = float(df['MA_20'].iloc[-1])
        rsi = float(df['RSI_14'].iloc[-1])
        
        if up_prob > 60: pred_msg = f"強勢看漲 📈 ({up_prob:.1f}%)"
        elif up_prob < 40: pred_msg = f"偏向看跌 📉 ({up_prob:.1f}%)"
        else: pred_msg = f"中性震盪 ⚖️ ({up_prob:.1f}%)"
        
        analysis_text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 最新收盤：{current_price:.2f}\n"
            f"🌊 20日均線：{ma20:.2f}\n"
            f"🌡️ RSI(14)：{rsi:.1f}\n"
            f"趨勢：{'多頭' if current_price > ma20 else '空頭'}\n\n"
            f"🤖 AI 上漲機率：{pred_msg}\n\n"
            f"📌 點擊上方圖表查看詳細策略回測"
        )
        return filename, analysis_text
    except Exception as e: return None, None

def calculate_backtest(stock_code, stock_name=""):
    try:
        df = get_taiwan_stock_data(stock_code, 730)
        if len(df) < 200: return "❌ 資料不足，無法回測。"
        
        df = add_advanced_features(df)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        if len(df) < 100: return "❌ 有效資料太少。"
        
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:].copy()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[FEATURES])
        model = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_train_scaled, train_df['Target'])
        
        X_test_scaled = scaler.transform(test_df[FEATURES])
        test_df['Prob'] = model.predict_proba(X_test_scaled)[:, 1]
        test_df['Signal'] = np.where(test_df['Prob'] > 0.60, 1, 0)
        
        test_df['Next_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
        test_df = test_df.dropna()
        
        test_df['Strategy_Return'] = test_df['Signal'] * test_df['Next_Return']
        
        strategy_ret = test_df['Strategy_Return'].values
        bh_ret = test_df['Next_Return'].values
        signals = test_df['Signal'].values
        
        if len(signals) == 0 or len(strategy_ret) == 0: 
            return "❌ 回測期間無有效資料。"
        
        strat_cum = np.cumprod(1 + strategy_ret)[-1] - 1
        bh_cum = np.cumprod(1 + bh_ret)[-1] - 1
        trades = strategy_ret[signals == 1]
        win_rate = (trades > 0).mean() * 100 if len(trades) > 0 else 0
        
        cum_ret = np.cumprod(1 + strategy_ret)
        roll_max = np.maximum.accumulate(cum_ret)
        drawdown = cum_ret / roll_max - 1
        mdd = drawdown.min() * 100
        
        std_dev = strategy_ret.std()
        sharpe = (strategy_ret.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        
        if sum(signals) == 0: conclusion = "⏸️ 訊號空窗：選擇空手觀望。\n🛒 買入建議：缺乏多頭動能，建議資金先停泊。\n💰 賣出建議：若已持有，請嚴守個人停損。"
        elif strat_cum > bh_cum:
            if sharpe > 1: conclusion = "✅ 策略優勢：高報酬且風險控制優異。\n🛒 買入建議：模型擅長標的，若預測看漲可進場。\n💰 賣出建議：預測轉跌時果斷停利。"
            else: conclusion = "✅ 擊敗大盤：能創造超額報酬，但過程震盪。\n🛒 買入建議：可進場，務必分批佈局。\n💰 賣出建議：見好就收，適時減碼。"
        else:
            if mdd > -15: conclusion = "🛡️ 下檔保護：總獲利雖輸大盤，但有避險作用。\n🛒 買入建議：適合防禦型配置。\n💰 賣出建議：不想資金閒置可換股。"
            else: conclusion = "⚠️ 模型失真：模型容易追高殺低。\n🛒 買入建議：請避開，不符合模型邏輯。\n💰 賣出建議：跌破重要均線請停損。"
        
        return (
            f"📑 {stock_name} ({stock_code}) 回測報告\n\n"
            f"📊 歷史績效\n🤖 AI 策略報酬：{strat_cum*100:.2f}%\n📈 死抱著不放：{bh_cum*100:.2f}%\n\n"
            f"🛡️ 風險與穩定度\n🎯 進場勝率：{win_rate:.1f}%\n⚠️ 最大回檔：{mdd:.2f}%\n⚖️ 夏普值：{sharpe:.2f}\n\n💡 建議：\n{conclusion}"
        )
    except Exception as e: return f"❌ 回測錯誤：{str(e)}"

# ==========================================
# 5. LINE Bot 路由與選單邏輯
# ==========================================
@app.route("/static/tmp/<path:filename>")
def serve_static(filename):
    return send_from_directory(static_tmp_path, filename)

@app.route("/")
def home(): return "LINE Bot 正常運作中！"

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
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🔍 個股查詢教學\n\n直接輸入：\n👉 股票代碼（例：2330）\n👉 股票名稱（例：台積電）\n\n系統將使用 LightGBM 產出分析與預測！"))
        return
    elif msg == "免責聲明":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ 免責聲明\n\n數據僅供程式開發交流。歷史勝率不代表未來績效，不構成買賣建議。請自行評估風險。"))
        return
    
    # 🌟 補回產業選單與大盤按鈕
    elif msg == "預測":
        items = [QuickReplyButton(action=MessageAction(label="全市場", text="選產業_全市場"))]
        for industry in industry_map.keys():
            items.append(QuickReplyButton(action=MessageAction(label=industry[:20], text=f"選產業_{industry}")))
        items.append(QuickReplyButton(action=MessageAction(label="📊 台股大盤預測", text="大盤預測")))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請選擇想分析的產業類別：👇", quick_reply=QuickReply(items=items))
        )
        return
        
    elif msg == "大盤預測":
        img_name, analysis_txt = analyze_and_predict_stock("TAIEX", "台股加權指數(大盤)")
        if img_name and analysis_txt:
            img_url = f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://")
            flex_content = {
                "type": "bubble",
                "hero": {"type": "image", "url": img_url, "size": "full", "aspectRatio": "10:6", "aspectMode": "cover",
                         "action": {"type": "message", "label": "action", "text": "詳細策略_TAIEX"}},
                "body": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}
            }
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="大盤預測", contents=flex_content))
        else: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料獲取失敗，請稍後再試。"))
        return

    # 🌟 補回風險偏好選擇
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
        return

    # 🌟 輕量化快速選股邏輯 (KMeans + 基本統計)
    elif msg.startswith("分析_"):
        parts = msg.split("_")
        target_industry = parts[1]
        risk_type = parts[2] if len(parts) > 2 else "穩健"
        
        if target_industry == "全市場": codes = ['2330', '2317', '2454', '2308', '2881', '2603', '2002', '1301', '0050', '0056']
        else: codes = industry_map.get(target_industry, [])

        stock_features = []
        for code in codes:
            try:
                # 為了極速回應，初篩只抓 90 天
                df_hist = get_taiwan_stock_data(code, 90)
                if not df_hist.empty and len(df_hist) > 10:
                    start_p, end_p = float(df_hist['Close'].iloc[0]), float(df_hist['Close'].iloc[-1])
                    ret = (end_p - start_p) / start_p
                    vol = df_hist['Close'].pct_change().std()
                    stock_features.append({'Code': code, 'Name': get_stock_name(code), 'Return': ret, 'Volatility': vol})
            except: pass

        df_target = pd.DataFrame(stock_features)
        if df_target.empty:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 抓取資料發生錯誤，請稍後再試。"))
            return
            
        df_target.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_target.dropna(inplace=True)

        if len(df_target) >= 2: # KMeans 至少需要兩筆資料
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
            except: top_5 = df_target.sort_values('Return', ascending=False).head(5)
        else: top_5 = df_target.sort_values('Return', ascending=False).head(5)

        reply_text = f"🚀 輕量快篩｜{target_industry}\n🎯 風格：{risk_type}\n\n"
        emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for i, (_, row) in enumerate(top_5.iterrows()):
            if i >= 5: break
            reply_text += f"{emoji_list[i]} {row['Name']} ({row['Code']})\n"
            reply_text += f"📈 區間報酬: {row['Return']*100:.1f}% ｜ ⚡ 波動: {row['Volatility']:.3f}\n\n"
            
        reply_text += "💡 【進階分析】：請直接輸入上方感興趣的「股票代碼」，啟動 LightGBM 深度預測！"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return

    elif msg.startswith("詳細策略_"):
        stock_code = msg.split("_")[1]
        stock_name = "台股加權指數(大盤)" if stock_code == "TAIEX" else get_stock_name(stock_code)
        backtest_report = calculate_backtest(stock_code, stock_name)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=backtest_report))
        return
    
    target_code, target_name = None, None
    if msg.isdigit() and 4 <= len(msg) <= 6: target_code = msg
    else: target_code, target_name = search_stock_code(msg)
    
    if target_code:
        img_name, analysis_txt = analyze_and_predict_stock(target_code, target_name)
        if img_name and analysis_txt:
            img_url = f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://")
            flex_content = {
                "type": "bubble",
                "hero": {"type": "image", "url": img_url, "size": "full", "aspectRatio": "10:6", "aspectMode": "cover",
                         "action": {"type": "message", "label": "詳細回測", "text": f"詳細策略_{target_code}"}},
                "body": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}
            }
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text=f"{target_name} 分析", contents=flex_content))
        else: line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料不足，無法分析"))
    else: line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"找不到「{msg}」，請輸入正確代碼或名稱"))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
