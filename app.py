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
        pass

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
# 3. 預測函數 (競賽專用：預測未來5個交易日)
# ==========================================
def analyze_and_predict_stock(stock_code, stock_name=None):
    try:
        if not stock_name: stock_name = get_stock_name(stock_code)
        df = get_taiwan_stock_data(stock_code, 730)
        if df.empty or len(df) < 100: return None, None
        
        df = add_advanced_features(df)
        if len(df) < 60: return None, None
        
        # 🎯 核心修改：改為 shift(-5)，預測未來 5 天 (即一週) 的漲跌
        df['Future_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Future_5d_Close'] > df['Close']).astype(int)
        
        # 將「今天」的特徵抽出來，因為今天還沒有未來 5 天的答案，只能用來「被預測」
        latest_features = df[FEATURES].iloc[-1:]
        
        # 訓練資料必須把最後 5 天刪掉，避免沒有答案的資料干擾 AI 學習
        train_df = df.dropna(subset=['Future_5d_Close'])
        if len(train_df) < 50: return None, None
        
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
        
        # 🎯 競賽視覺化：將圖表標題直接壓上競賽日期
        plt.title(f'{stock_code} {stock_name} - 競賽預測 (3/27-4/2)', fontsize=16)
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
        
        # 🎯 競賽專屬：文字對話泡泡也加上競賽日期
        analysis_text = (
            f"📊 {stock_name} ({stock_code})\n\n"
            f"💰 最新收盤：{current_price:.2f}\n"
            f"🌊 20日均線：{ma20:.2f}\n"
            f"🌡️ RSI(14)：{rsi:.1f}\n"
            f"趨勢：{'多頭' if current_price > ma20 else '空頭'}\n\n"
            f"🎯 【競賽預測區間：3/27 - 4/2】\n"
            f"🤖 AI 看漲機率：{pred_msg}\n\n"
            f"📌 點擊上方圖表查看詳細策略回測"
        )
        return filename, analysis_text
        
    except Exception as e:
        return None, None

# ==========================================
# 4. 回測 (配合 5 天預測邏輯)
# ==========================================
def calculate_backtest(stock_code, stock_name=""):
    try:
        df = get_taiwan_stock_data(stock_code, 730)
        if len(df) < 200: return "❌ 資料不足，無法回測。"
        
        df = add_advanced_features(df)
        
        # 🎯 回測也同步改成 5 天預測邏輯
        df['Future_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Future_5d_Close'] > df['Close']).astype(int)
        
        valid_df = df.dropna(subset=['Future_5d_Close']).copy()
        if len(valid_df) < 100: return "❌ 有效資料太少。"
        
        split_idx = int(len(valid_df) * 0.8)
        train_df = valid_df.iloc[:split_idx]
        test_df = valid_df.iloc[split_idx:].copy()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[FEATURES])
        model = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_train_scaled, train_df['Target'])
        
        X_test_scaled = scaler.transform(test_df[FEATURES])
        test_df['Prob'] = model.predict_proba(X_test_scaled)[:, 1]
        test_df['Signal'] = np.where(test_df['Prob'] > 0.60, 1, 0)
        
        test_df['Next_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
        test_df = test_df.dropna(subset=['Next_Return'])
        
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
        
        if sum(signals) == 0:
            conclusion = "⏸️ 訊號空窗：模型未發現高勝率進場點，選擇空手觀望。\n🛒 買入建議：缺乏多頭動能，建議資金先停泊。\n💰 賣出建議：若已持有，請嚴守個人停損。"
        elif strat_cum > bh_cum:
            if sharpe > 1: conclusion = "✅ 策略優勢：LGBM 精準抓到波段，高報酬且風險控制優異。\n🛒 買入建議：屬於模型極度擅長的標的，若預測未來一週看漲可進場。\n💰 賣出建議：預測轉跌時果斷停利。"
            else: conclusion = "✅ 擊敗大盤：能創造超額報酬，但過程資金震盪幅度較大。\n🛒 買入建議：可進場，但務必分批佈局攤平波動。\n💰 賣出建議：見好就收，適時減碼。"
        else:
            if mdd > -15: conclusion = "🛡️ 下檔保護：總獲利雖輸給死抱不放，但大跌時有發揮避險作用。\n🛒 買入建議：適合防禦型配置。\n💰 賣出建議：不想資金閒置可轉換至強勢股。"
            else: conclusion = "⚠️ 模型失真：模型在此標的容易追高殺低。\n🛒 買入建議：請避開，不符合模型邏輯。\n💰 賣出建議：回歸均線判斷，跌破請停損。"
        
        res_text = (
            f"📑 {stock_name} ({stock_code}) LGBM 回測報告\n"
            f"⏳ 測試區間：近 {len(signals)} 個交易日\n\n"
            f"📊 歷史績效\n"
            f"🤖 AI 策略報酬：{strat_cum*100:.2f}%\n"
            f"📈 買進持有報酬：{bh_cum*100:.2f}%\n\n"
            f"🛡️ 風險與穩定度\n"
            f"🎯 進場勝率：{win_rate:.1f}%\n"
            f"⚠️ 最大回檔：{mdd:.2f}%\n"
            f"⚖️ 夏普值：{sharpe:.2f}\n\n"
            f"💡 資產管理評估：\n{conclusion}"
        )
        return res_text
        
    except Exception as e:
        return f"❌ 回測錯誤：{str(e)}"

# ==========================================
# 5. LINE Bot 路由
# ==========================================
@app.route("/static/tmp/<path:filename>")
def serve_static(filename):
    return send_from_directory(static_tmp_path, filename)

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
        reply_text = "🔍 個股查詢教學\n\n直接輸入：\n👉 股票代碼（例：2330）\n👉 股票名稱（例：台積電）\n\n系統將使用 LightGBM 產出未來一週預測！"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return
    elif msg == "免責聲明":
        reply_text = "⚠️ 免責聲明\n\n數據僅供程式開發交流。歷史勝率不代表未來績效，不構成買賣建議。請自行評估風險。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return
    elif msg == "預測":
        items = [QuickReplyButton(action=MessageAction(label="全市場", text="選產業_全市場"))]
        for industry in industry_map.keys():
            items.append(QuickReplyButton(action=MessageAction(label=industry[:20], text=f"選產業_{industry}")))
        items.append(QuickReplyButton(action=MessageAction(label="📊 台股大盤預測", text="大盤預測")))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請選擇產業類別：👇", quick_reply=QuickReply(items=items))
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
                "body": {"type": "box", "layout": "vertical",
                         "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}
            }
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="大盤預測", contents=flex_content))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料獲取失敗，請稍後再試。"))
        return
    elif msg.startswith("選產業_"):
        target_industry = msg.split("_")[1]
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"已鎖定【{target_industry}】，因演算法升級需消耗大量算力，全產業群體掃描功能暫時維護中，請直接輸入「個股代碼」查詢！")
        )
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
                "body": {"type": "box", "layout": "vertical",
                         "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}
            }
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text=f"{target_name} 分析", contents=flex_content))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 資料不足，無法分析"))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"找不到「{msg}」，請輸入正確代碼或名稱"))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
