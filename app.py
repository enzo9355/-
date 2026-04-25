# app.py
# v5.1 終極自動化版：整合 Gemini AI 撰稿與 LINE 主動廣播發報系統
# --------------------------------------------------

import os
import io
import datetime
import requests
import pandas as pd
import twstock
import xml.etree.ElementTree as ET
import urllib.parse
import numpy as np
import json
import google.generativeai as genai

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from flask import Flask, request, abort, render_template_string

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)

# ==================================================
# 1. 基本設定與 Gemini 初始化
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FINMIND_USER = os.getenv("FINMIND_USER")
FINMIND_PASSWORD = os.getenv("FINMIND_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 💡 安全金鑰：防止外部非法觸發廣播
BROADCAST_TOKEN = os.getenv("BROADCAST_TOKEN", "default_secret")

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

finmind_token = ""
CATEGORY_PAGE_SIZE = 12

# ==================================================
# 2. 資料抓取與清洗模組
# ==================================================
def finmind_login():
    global finmind_token
    if finmind_token or not FINMIND_USER or not FINMIND_PASSWORD: return
    try:
        r = requests.post(
            "https://api.finmindtrade.com/api/v4/login",
            data={"user_id": FINMIND_USER, "password": FINMIND_PASSWORD},
            timeout=10
        ).json()
        if r.get("msg") == "success": finmind_token = r["token"]
    except: pass

def get_stock_name(code):
    if code == "TAIEX": return "台股大盤"
    if code in twstock.codes: return twstock.codes[code].name
    return code

def search_stock_code(keyword):
    keyword = keyword.upper().strip()
    if keyword in ["TAIEX", "加權指數", "台股大盤", "大盤"]: return "TAIEX", "台股大盤"
    if keyword.isdigit(): return keyword, get_stock_name(keyword)
    for code, info in twstock.codes.items():
        if keyword in info.name.upper(): return code, info.name
    return None, None

def _clean_df(df):
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan)
    df = df.dropna(subset=['Date', 'Close'])
    return df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').set_index("Date")

def get_data(code, days=730):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if code == "TAIEX":
        finmind_login()
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockPrice", "data_id": "TAIEX", "start_date": start_date, "end_date": end_date}
            if finmind_token: params["token"] = finmind_token
            r = requests.get(url, params=params, timeout=15).json()
            if r.get("msg") == "success" and r.get("data"):
                df = pd.DataFrame(r["data"])
                df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                df["Open"] = pd.to_numeric(df["open"], errors="coerce")
                df["High"] = pd.to_numeric(df["max"], errors="coerce")
                df["Low"] = pd.to_numeric(df["min"], errors="coerce")
                df["Close"] = pd.to_numeric(df["close"], errors="coerce")
                return _clean_df(df[["Date", "Open", "High", "Low", "Close"]])
        except: pass
    else:
        finmind_login()
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockPrice", "data_id": code, "start_date": start_date, "end_date": end_date}
            if finmind_token: params["token"] = finmind_token
            data = requests.get(url, params=params, timeout=15).json()
            if data.get("data"):
                df = pd.DataFrame(data["data"])
                df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                df["Open"] = pd.to_numeric(df["open"], errors="coerce")
                df["High"] = pd.to_numeric(df["max"], errors="coerce")
                df["Low"] = pd.to_numeric(df["min"], errors="coerce")
                df["Close"] = pd.to_numeric(df["close"], errors="coerce")
                return _clean_df(df[["Date", "Open", "High", "Low", "Close"]])
        except: pass
    return pd.DataFrame()

# ==================================================
# 3. 核心運算模組 (LGBM + Gemini)
# ==================================================
def get_news(name):
    try:
        q = urllib.parse.quote(f"{name} 股票")
        url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.text)
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in root.findall('.//item')[:5]]
    except: return []

def calc_all(df):
    df = df.copy()
    c = df["Close"]
    df['MA_5'], df['MA20'], df['RET_1'] = c.rolling(5).mean(), c.rolling(20).mean(), c.pct_change()
    d = c.diff()
    g, l = d.clip(lower=0).rolling(14).mean(), -d.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (g / (l + 1e-9))))
    df['Volat'] = df['RET_1'].rolling(20).std()
    return df.dropna()

def run_ai_engine(df):
    try:
        feats = ['MA_5', 'MA20', 'RET_1', 'RSI', 'Volat']
        w_df = df.copy()
        w_df['T'] = (w_df['Close'].shift(-5) > w_df['Close']).astype(int)
        v_df = w_df.dropna(subset=['T']).copy()
        split = int(len(v_df) * 0.8)
        sc = StandardScaler()
        X_tr = sc.fit_transform(v_df.iloc[:split][feats])
        model = LGBMClassifier(n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_tr, v_df.iloc[:split]['T'])
        
        # 產出全歷史 AI 勝率
        df['AI_P'] = model.predict_proba(sc.transform(df[feats].ffill().bfill()))[:, 1] * 100
        
        # 回測數據
        X_te = sc.transform(v_df.iloc[split:][feats])
        probs = model.predict_proba(X_te)[:, 1]
        rets = v_df.iloc[split:]['Close'].shift(-1) / v_df.iloc[split:]['Close'] - 1
        strat_ret = np.where(probs > 0.6, rets, 0)
        
        # 提取重要性
        imps = model.feature_importances_
        f_map = {'MA_5':'短線動能', 'MA20':'月線趨勢', 'RET_1':'價格反轉', 'RSI':'強弱指標', 'Volat':'波動收斂'}
        top = [f"{f_map[f]} ({ (i/sum(imps))*100 :.1f}%)" for f, i in sorted(zip(feats, imps), key=lambda x:x[1], reverse=True)[:3]]
        
        return {"mdd": (np.cumprod(1+strat_ret)/np.maximum.accumulate(np.cumprod(1+strat_ret))-1).min()*100,
                "sharpe": (strat_ret.mean()/strat_ret.std())*np.sqrt(252) if strat_ret.std()!=0 else 0,
                "top": top, "win": (strat_ret[strat_ret!=0]>0).mean()*100 if len(strat_ret[strat_ret!=0])>0 else 0}
    except: return None

def get_ai_insight(name, data, bt, news):
    if not gemini_model: return "未設定 API Key"
    n_txt = "\n".join([n['title'] for n in news])
    p = f"請以資深分析師語氣，針對{name}撰寫100字內洞見。最新價:{data['price']}, 勝率:{data['prob']}%, 夏普值:{bt['sharpe']:.2f}。新聞:\n{n_txt}"
    try: return gemini_model.generate_content(p).text.replace('\n', '<br>')
    except: return "生成失敗"

# ==================================================
# 4. 分析總控
# ==================================================
def analyze(code):
    df = get_data(code)
    if df.empty or len(df) < 200: return None
    df = calc_all(df)
    bt = run_ai_engine(df)
    if not bt: return None
    
    last = df.iloc[-1]
    name = get_stock_name(code)
    news = get_news(name)
    insight = get_ai_insight(name, {"price": last['Close'], "prob": int(last['AI_P'])}, bt, news)
    
    # 準備繪圖 JSON
    tv_df = df.copy().reset_index()
    tv_df['Date'] = tv_df['Date'].dt.strftime('%Y-%m-%d')
    
    # 未來預測線
    drift = ((int(last['AI_P']) - 50) / 50.0) * (last['Volat'] * last['Close'])
    pred = [{'time': tv_df['Date'].iloc[-1], 'value': last['Close']}]
    curr_d = df.index[-1]
    curr_p = last['Close']
    for _ in range(5):
        curr_d += datetime.timedelta(days=1)
        while curr_d.weekday() >= 5: curr_d += datetime.timedelta(days=1)
        curr_p += drift
        pred.append({'time': curr_d.strftime('%Y-%m-%d'), 'value': round(curr_p, 2)})

    return {
        "code": code, "name": name, "price": last['Close'], "prob": int(last['AI_P']), 
        "insight": insight, "bt": bt, "news": news, "trend": "多頭" if last['Close'] > last['MA20'] else "空頭",
        "candles": json.dumps(tv_df[['Date','Open','High','Low','Close']].rename(columns={'Date':'time','Open':'open','High':'high','Low':'low','Close':'close'}).to_dict('records')),
        "ma20": json.dumps(tv_df[['Date','MA20']].rename(columns={'Date':'time','MA20':'value'}).to_dict('records')),
        "prob_h": json.dumps(tv_df[['Date','AI_P']].rename(columns={'Date':'time','AI_P':'value'}).to_dict('records')),
        "pred": json.dumps(pred)
    }

# ==================================================
# 5. UI 渲染 (精簡極致版)
# ==================================================
def render_web(d):
    html = f"""
<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{d['name']} 分析</title><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"></script>
<style>
    body {{ margin:0; background: #0f2027; color: #f1f1f1; font-family: 'Noto Sans TC'; }}
    .wrap {{ max-width:900px; margin:auto; padding:20px; }}
    .card {{ background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }}
    .highlight {{ color: #00f2fe; font-weight: bold; }}
    #tvchart {{ width: 100%; height: 400px; }}
    .news-link {{ display:block; color:#aaa; text-decoration:none; margin:10px 0; font-size:14px; }}
</style></head>
<body><div class="wrap">
    <h1>{d['name']} ({d['code']})</h1>
    <div class="card" style="border-left: 5px solid #00f2fe;">
        <h2 style="color:#00f2fe; margin-top:0;">🌞 AI 投資觀點</h2>
        <p style="line-height:1.8;">{d['insight']}</p>
    </div>
    <div class="card">
        收盤：<span class="highlight">{d['price']:.2f}</span> | 趨勢：{d['trend']} | AI 勝率：<span class="highlight">{d['prob']}%</span>
        <div id="tvchart"></div>
    </div>
    <div class="card">
        <h2>🤖 決策因子</h2>
        {' / '.join(d['bt']['top'])}
    </div>
    <div class="card">
        <h2>📊 歷史回測</h2>
        勝率：{d['bt']['win']:.1f}% | 夏普值：{d['bt']['sharpe']:.2f} | 最大回檔：{d['bt']['mdd']:.2f}%
    </div>
    <div class="card">
        <h2>📰 相關新聞</h2>
        {''.join([f'<a class="news-link" href="{n["link"]}">{n["title"]}</a>' for n in d['news']])}
    </div>
</div>
<script>
    const chart = LightweightCharts.createChart(document.getElementById('tvchart'), {{
        layout:{{background:{{color:'transparent'}},textColor:'#d1d4dc'}},
        grid:{{vertLines:{{color:'#2b2b2b'}},horzLines:{{color:'#2b2b2b'}}}},
        timeScale:{{timeVisible:true}}
    }});
    const candleS = chart.addCandlestickSeries({{upColor:'#ef5350',downColor:'#26a69a'}});
    candleS.setData({d['candles']});
    const maS = chart.addLineSeries({{color:'#00f2fe',lineWidth:1}});
    maS.setData({d['ma20']});
    const predS = chart.addLineSeries({{color:'#ff9800',lineStyle:2}});
    predS.setData({d['pred']});
    const probS = chart.addHistogramSeries({{priceScaleId:'',scaleMargins:{{top:0.8,bottom:0}}}});
    probS.setData({d['prob_h']}.map(x=>({{time:x.time,value:x.value,color:x.value>=50?'rgba(38,166,154,0.3)':'rgba(239,83,80,0.3)'}})));
</script></body></html>
"""
    return html

# ==================================================
# 6. 自動化發報引擎 (💡 本次核心新增)
# ==================================================
@app.route("/broadcast_weekly", methods=["GET"])
def broadcast_weekly():
    # 1. 驗證 Token
    token = request.args.get("token")
    if token != BROADCAST_TOKEN:
        return "身份驗證失敗", 403
    
    # 2. 執行分析
    d = analyze("TAIEX")
    if not d: return "分析失敗", 500
    
    # 3. 準備發送內容
    url = f"{request.host_url}market".replace("http://", "https://")
    clean_insight = d['insight'].replace('<br>', '\n')
    msg = f"🌞 周一 AI 投資晨報\n\n📊 大盤分析：\n{clean_insight[:120]}...\n\n🔗 點擊查看 AI 預測軌跡：\n{url}"
    
    # 4. 廣播發送
    try:
        line_bot_api.broadcast(TextSendMessage(text=msg))
        return f"廣播成功：{datetime.datetime.now()}", 200
    except Exception as e:
        return f"發送失敗：{str(e)}", 500

# ==================================================
# 7. 路由與 LINE 基礎指令
# ==================================================
@app.route("/stock/<code>")
def stock_page(code):
    d = analyze(code)
    return render_web(d) if d else "查無資料"

@app.route("/market")
def market_page():
    d = analyze("TAIEX")
    return render_web(d) if d else "資料更新中"

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except: abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.strip()
    if msg == "大盤":
        url = f"{request.host_url}market".replace("http://", "https://")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"📊 大盤即時 AI 報告：\n{url}"))
    else:
        code, name = search_stock_code(msg)
        if code:
            url = f"{request.host_url}stock/{code}".replace("http://", "https://")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"📈 {name} ({code}) AI 分析連結：\n{url}"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
