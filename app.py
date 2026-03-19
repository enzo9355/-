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

# 💡【極簡優化】統一 AI 模型參數，以後調參數只要改這裡！
LGBM_PARAMS = {
    'n_estimators': 80, 
    'learning_rate': 0.05, 
    'max_depth': 4, 
    'random_state': 42, 
    'verbose': -1
}

industry_map = {
    "半導體業": ['2330', '2454', '2303', '3711', '2408', '3034', '3443', '3035', '3006', '3532'],
    "電腦周邊": ['2317', '2382', '3231', '2324', '2353', '2357', '2356', '6669', '2377', '2352'],
    "通信網路": ['2412', '3045', '4904', '2345', '3702', '5388', '3062', '2455', '6285', '3596'],
    "光電產業": ['3008', '2409', '3481', '2448', '3019', '6706', '4956', '6176', '3406', '3504'],
    "電子零組件": ['2308', '2327', '3037', '2383', '2313', '2059', '3042', '3044', '2492', '2316'],
    "金融保險": ['2881', '2882', '2886', '2891', '2884', '5880', '2892', '2885', '2880', '2890'],
    "航運業": ['2603', '2609', '2615', '2618', '2610', '2606', '2637', '2633', '5608', '2605'],
    "鋼鐵工業": ['2002', '2014', '2006', '2015', '2027', '2031', '2023', '2009', '2034', '2038'],
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
        files = sorted([os.path.join(static_tmp_path, f) for f in os.listdir(static_tmp_path) if f.endswith('.png')], key=os.path.getmtime)
        if len(files) > 100:
            for f in files[:50]: os.remove(f)
    except Exception: pass

# ==========================================
# 2. 資料獲取與特徵工程
# ==========================================
finmind_auto_token = ""

def auto_login_finmind():
    global finmind_auto_token
    if not FINMIND_USER or not FINMIND_PASSWORD: return
    try:
        res = requests.post("https://api.finmindtrade.com/api/v4/login", 
                            data={"user_id": FINMIND_USER, "password": FINMIND_PASSWORD}, timeout=10).json()
        if res.get("msg") == "success": finmind_auto_token = res.get("token")
    except Exception: pass

def get_stock_name(stock_code):
    return twstock.codes[stock_code].name if stock_code in twstock.codes else stock_code

def search_stock_code(keyword):
    keyword = keyword.upper()
    for code in all_watch_list:
        name = get_stock_name(code)
        if keyword in name or keyword == name: return code, name
    for code, info in twstock.codes.items():
        if (keyword in info.name or keyword == info.name) and len(code) <= 6: return code, info.name
    return None, None

def get_taiwan_stock_data(stock_code, period_days=730):
    global finmind_auto_token
    start_date = (datetime.datetime.now() - datetime.timedelta(days=period_days)).strftime('%Y-%m-%d')
    
    # 💡【極簡優化】把抓資料包成小動作，失敗就自動重試一次
    def fetch_data(token):
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={stock_code}&start_date={start_date}"
        if token: url += f"&token={token}"
        return requests.get(url, timeout=10).json()

    try:
        if FINMIND_USER and not finmind_auto_token: auto_login_finmind()
        data = fetch_data(finmind_auto_token)
        
        if data.get("status") != 200 or "expired" in str(data.get("msg", "")).lower():
            auto_login_finmind()
            data = fetch_data(finmind_auto_token)

        if data.get("msg") == "success" and data.get("data"):
            df = pd.DataFrame(data["data"])
            df.columns = [str(c).lower() for c in df.columns]
            df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'volume': 'Volume', 'trading_volume': 'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            return df[df['Close'] > 0] # 防呆機制
    except Exception: pass
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
    df['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['RS_60'] = df['Close'] / df['Close'].rolling(60).mean()
    
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    df['ATR_14'] = tr.rolling(14).mean()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()

FEATURES = ['MA_5', 'MA_10', 'MA_20', 'MA_60', 'RET_1', 'RET_5', 'Volatility', 
            'RSI_14', 'MACD', 'MACD_Signal', 'Volume_MA20', 'OBV', 'RS_60', 'ATR_14']

# ==========================================
# 3. 預測函數群
# ==========================================
def analyze_and_predict_stock(stock_code, stock_name=None):
    try:
        stock_name = stock_name or get_stock_name(stock_code)
        df = get_taiwan_stock_data(stock_code, 365)
        if df.empty or len(df) < 100: return None, "❌ 該檔股票近期缺乏有效交易數據，無法進行分析。"
        
        df = add_advanced_features(df)
        if len(df) < 60 or float(df['Close'].iloc[-1]) <= 0: return None, "❌ 該檔股票資料異常或不足。"
        
        df['Future_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Future_5d_Close'] > df['Close']).astype(int)
        
        train_df = df.dropna(subset=['Future_5d_Close'])
        if len(train_df) < 50: return None, "❌ 有效訓練資料不足。"
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_df[FEATURES])
        
        # 💡【極簡優化】直接呼叫上面的字典，程式碼瞬間變乾淨
        model = LGBMClassifier(**LGBM_PARAMS).fit(X_scaled, train_df['Target'])
        
        latest_scaled = scaler.transform(df[FEATURES].iloc[-1:])
        up_prob = model.predict_proba(latest_scaled)[0][1] * 100
        
        cleanup_images() 
        now = datetime.datetime.now()
        date_range_str = f"{(now + datetime.timedelta(days=1)).strftime('%Y/%m/%d')}-{(now + datetime.timedelta(days=7)).strftime('%m/%d')}"
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='收盤價', color='black', linewidth=2)
        plt.plot(df.index[-60:], df['MA_10'].iloc[-60:], label='10日均線', color='blue', linestyle='--')
        plt.plot(df.index[-60:], df['MA_20'].iloc[-60:], label='20日均線', color='red', linestyle='-.')
        plt.title(f'{stock_code} {stock_name} - 預測區間 ({date_range_str})', fontsize=15, fontweight='bold')
        plt.xlabel('日期', fontsize=12); plt.ylabel('價格', fontsize=12)
        plt.legend(prop={'size': 12}); plt.grid(True, alpha=0.3)
        
        filename = f"{stock_code}_{int(time.time())}.png"
        filepath = os.path.join(static_tmp_path, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        current_price, ma20, rsi = float(df['Close'].iloc[-1]), float(df['MA_20'].iloc[-1]), float(df['RSI_14'].iloc[-1])
        pred_msg = f"強勢看漲 📈 ({up_prob:.1f}%)" if up_prob > 60 else f"偏向看跌 📉 ({up_prob:.1f}%)" if up_prob < 40 else f"中性震盪 ⚖️ ({up_prob:.1f}%)"
        
        analysis_text = (
            f"📊 {stock_name} ({stock_code})\n\n💰 最新收盤：{current_price:.2f}\n🌊 20日均線：{ma20:.2f}\n"
            f"🌡️ RSI(14)：{rsi:.1f}\n趨勢：{'多頭' if current_price > ma20 else '空頭'}\n\n"
            f"🎯 【預測區間：{date_range_str}】\n🤖 AI 上漲機率：{pred_msg}\n\n📌 點擊上方圖表查看詳細策略回測"
        )
        return filename, analysis_text
    except Exception: return None, "❌ 分析過程中發生預期外的錯誤。"

def fast_predict(stock_code):
    try:
        df = get_taiwan_stock_data(stock_code, 365) 
        if df.empty or len(df) < 100: return None
        df = add_advanced_features(df)
        if len(df) < 60 or float(df['Close'].iloc[-1]) <= 0: return None
        
        df['Future_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Future_5d_Close'] > df['Close']).astype(int)
        
        train_df = df.dropna(subset=['Future_5d_Close'])
        if len(train_df) < 50: return None
        
        scaler = StandardScaler()
        # 💡【極簡優化】一行搞定訓練與預測
        model = LGBMClassifier(**LGBM_PARAMS).fit(scaler.fit_transform(train_df[FEATURES]), train_df['Target'])
        up_prob = model.predict_proba(scaler.transform(df[FEATURES].iloc[-1:]))[0][1] * 100
        
        return get_stock_name(stock_code), float(df['Close'].iloc[-1]), up_prob
    except: return None

# ==========================================
# 4. 回測
# ==========================================
def calculate_backtest(stock_code, stock_name=""):
    try:
        df = get_taiwan_stock_data(stock_code, 730)
        if len(df) < 200: return "❌ 資料不足，無法回測。"
        
        df = add_advanced_features(df)
        df['Future_5d_Close'] = df['Close'].shift(-5)
        df['Target'] = (df['Future_5d_Close'] > df['Close']).astype(int)
        
        valid_df = df.dropna(subset=['Future_5d_Close']).copy()
        if len(valid_df) < 100: return "❌ 有效資料太少。"
        
        split_idx = int(len(valid_df) * 0.8)
        train_df, test_df = valid_df.iloc[:split_idx], valid_df.iloc[split_idx:].copy()
        
        scaler = StandardScaler()
        # 💡【極簡優化】再次套用字典
        model = LGBMClassifier(**LGBM_PARAMS).fit(scaler.fit_transform(train_df[FEATURES]), train_df['Target'])
        
        test_df['Prob'] = model.predict_proba(scaler.transform(test_df[FEATURES]))[:, 1]
        test_df['Signal'] = np.where(test_df['Prob'] > 0.60, 1, 0)
        test_df['Next_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Next_Return'])
        
        strategy_ret = (test_df['Signal'] * test_df['Next_Return']).values
        bh_ret = test_df['Next_Return'].values
        signals = test_df['Signal'].values
        
        if len(signals) == 0 or len(strategy_ret) == 0: return "❌ 回測期間無有效資料。"
        
        strat_cum = np.cumprod(1 + strategy_ret)[-1] - 1
        bh_cum = np.cumprod(1 + bh_ret)[-1] - 1
        trades = strategy_ret[signals == 1]
        win_rate = (trades > 0).mean() * 100 if len(trades) > 0 else 0
        
        cum_ret = np.cumprod(1 + strategy_ret)
        mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min() * 100
        std_dev = strategy_ret.std()
        sharpe = (strategy_ret.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        
        strat_cum = 0 if np.isnan(strat_cum) or np.isinf(strat_cum) else strat_cum
        bh_cum = 0 if np.isnan(bh_cum) or np.isinf(bh_cum) else bh_cum
        mdd = 0 if np.isnan(mdd) or np.isinf(mdd) else mdd
        sharpe = 0 if np.isnan(sharpe) or np.isinf(sharpe) else sharpe
        
        if sum(signals) == 0:
            conclusion = "⏸️ 訊號空窗：模型未發現高勝率進場點，選擇空手觀望。\n🛒 買入建議：缺乏多頭動能，建議資金先停泊。\n💰 賣出建議：若已持有，請嚴守個人停損。"
        elif strat_cum > bh_cum:
            conclusion = "✅ 策略優勢：高報酬且風險控制優異。\n🛒 買入建議：預測看漲可進場。\n💰 賣出建議：預測轉跌時果斷停利。" if sharpe > 1 else "✅ 擊敗大盤：能創造超額報酬。\n🛒 買入建議：可進場分批佈局。\n💰 賣出建議：見好就收。"
        else:
            conclusion = "🛡️ 下檔保護：大跌時具備避險作用。\n🛒 買入建議：適合防禦型配置。\n💰 賣出建議：不想資金閒置可轉換至強勢股。" if mdd > -15 else "⚠️ 模型失真：容易追高殺低。\n🛒 買入建議：請避開。\n💰 賣出建議：回歸均線判斷停損。"
        
        return (
            f"📑 {stock_name} ({stock_code}) LGBM 回測報告\n⏳ 測試區間：近 {len(signals)} 個交易日\n\n"
            f"📊 歷史績效\n🤖 AI 策略報酬：{strat_cum*100:.2f}%\n📈 買進持有報酬：{bh_cum*100:.2f}%\n\n"
            f"🛡️ 風險與穩定度\n🎯 進場勝率：{win_rate:.1f}%\n⚠️ 最大回檔：{mdd:.2f}%\n⚖️ 夏普值：{sharpe:.2f}\n\n"
            f"💡 資產管理評估：\n{conclusion}"
        )
    except Exception as e: return f"❌ 回測錯誤：{str(e)}"

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
    try: handler.handle(request.get_data(as_text=True), request.headers['X-Line-Signature'])
    except InvalidSignatureError: abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.strip()
    
    if msg == "教學":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="🔍 個股查詢教學\n\n直接輸入：\n👉 股票代碼（例：2330）\n👉 股票名稱（例：台積電）"))
    elif msg == "免責聲明":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ 免責聲明\n\n數據僅供交流。歷史不代表未來績效，請自行評估風險。"))
    elif msg == "預測":
        items = [QuickReplyButton(action=MessageAction(label="🌐 全市場 (權值示範)", text="選產業_全市場"))] + \
                [QuickReplyButton(action=MessageAction(label=ind[:20], text=f"選產業_{ind}")) for ind in industry_map.keys()] + \
                [QuickReplyButton(action=MessageAction(label="📊 台股大盤預測", text="大盤預測"))]
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請選擇您想觀察的【產業類別】：👇", quick_reply=QuickReply(items=items)))
    elif msg == "大盤預測":
        img_name, analysis_txt = analyze_and_predict_stock("TAIEX", "台股加權指數(大盤)")
        if img_name:
            flex = {"type": "bubble", "hero": {"type": "image", "url": f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://"), "size": "full", "aspectRatio": "10:6", "aspectMode": "cover", "action": {"type": "message", "label": "action", "text": "詳細策略_TAIEX"}}, "body": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}}
            line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="大盤預測", contents=flex))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=analysis_txt or "❌ 資料獲取失敗。"))
    elif msg.startswith("選產業_"):
        ind = msg.split("_")[1]
        items = [QuickReplyButton(action=MessageAction(label="🛡️ 穩健策略", text=f"執行掃描_穩健_{ind}")), QuickReplyButton(action=MessageAction(label="🔥 激進策略", text=f"執行掃描_激進_{ind}"))]
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"已鎖定【{ind}】\n請選擇您的投資風格：👇", quick_reply=QuickReply(items=items)))
    elif msg.startswith("執行掃描_"):
        _, strategy, ind = msg.split("_")
        base_list = ['2330', '2317', '2454', '2308', '2881', '2382', '2882', '2412', '2886', '2891'] if ind == "全市場" else industry_map.get(ind, [])
        stock_list = base_list[:5] if strategy == "穩健" else (base_list[5:10] if len(base_list) >= 10 else base_list[-5:])
        
        now = datetime.datetime.now()
        date_range_str = f"{(now + datetime.timedelta(days=1)).strftime('%m/%d')}-{(now + datetime.timedelta(days=7)).strftime('%m/%d')}"
        results_msg = [f"🚀 【{ind if ind != '全市場' else '全市場 (權值示範)'} - {strategy}】 AI 掃描\n⏳ 預測區間: {date_range_str}\n" + "-"*20]
        
        for code in stock_list:
            res = fast_predict(code)
            if res:
                trend = "📈 強勢看漲" if res[2] > 60 else "📉 偏向看跌" if res[2] < 40 else "⚖️ 中性震盪"
                results_msg.append(f"🔹 {code} {res[0]}\n   💰 收盤：{res[1]:.2f} 元\n   🤖 機率：{res[2]:.1f}% ({trend})")
        
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="\n\n".join(results_msg) if len(results_msg) > 1 else "❌ 掃描失敗或缺乏交易資料。"))
    elif msg.startswith("詳細策略_"):
        code = msg.split("_")[1]
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=calculate_backtest(code, "台股加權指數(大盤)" if code == "TAIEX" else get_stock_name(code))))
    else:
        target_code, target_name = (msg, None) if msg.isdigit() and 4 <= len(msg) <= 6 else search_stock_code(msg)
        if target_code:
            img_name, analysis_txt = analyze_and_predict_stock(target_code, target_name)
            if img_name:
                flex = {"type": "bubble", "hero": {"type": "image", "url": f"{request.host_url}static/tmp/{img_name}".replace("http://", "https://"), "size": "full", "aspectRatio": "10:6", "aspectMode": "cover", "action": {"type": "message", "label": "詳細回測", "text": f"詳細策略_{target_code}"}}, "body": {"type": "box", "layout": "vertical", "contents": [{"type": "text", "text": analysis_txt, "wrap": True, "size": "sm"}]}}
                line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text=f"{target_name} 分析", contents=flex))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=analysis_txt or "❌ 資料不足，無法分析"))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"找不到「{msg}」，請輸入正確代碼或名稱"))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
