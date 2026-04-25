# app.py
# v4.0 升級版：導入真・AI 機率 (LGBM predict_proba)、修復 LINE 選單遺失
# --------------------------------------------------

import os
import io
import base64
import datetime
import requests
import pandas as pd
import twstock
import xml.etree.ElementTree as ET
import urllib.parse
import numpy as np
import json

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
# 1. 基本設定
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
FINMIND_USER = os.getenv("FINMIND_USER")
FINMIND_PASSWORD = os.getenv("FINMIND_PASSWORD")

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

finmind_token = ""
CATEGORY_PAGE_SIZE = 12

# ==================================================
# 2. API 與資料抓取工具 (前置防呆排序)
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
    if code == "TAIEX": return "台股大盤（加權指數）"
    if code in twstock.codes: return twstock.codes[code].name
    return code

def search_stock_code(keyword):
    keyword = keyword.upper().strip()
    if keyword in ["TAIEX", "加權指數", "台股大盤", "大盤"]: return "TAIEX", "台股大盤（加權指數）"
    if keyword.isdigit(): return keyword, get_stock_name(keyword)
    for code, info in twstock.codes.items():
        if keyword in info.name.upper(): return code, info.name
    return None, None

def _clean_dataframe(df):
    """共用清洗函式：處理遺失值並強制排序"""
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan)
    df = df.dropna(subset=['Date', 'Close'])
    return df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').set_index("Date")

def _get_taiex_data(days=730):
    finmind_login()
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
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
            return _clean_dataframe(df[["Date", "Open", "High", "Low", "Close"]])
    except: pass
    
    try:
        import yfinance as yf
        hist = yf.download("^TWII", start=start_date, progress=False)
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.droplevel(1)
        if not hist.empty and "Close" in hist.columns:
            df = hist.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.reset_index().rename(columns={'index': 'Date', 'Datetime': 'Date'})
            return _clean_dataframe(df[["Date", "Open", "High", "Low", "Close"]])
    except: pass
    return pd.DataFrame()

def get_stock_data(stock_code, days=730):
    if stock_code == "TAIEX": return _get_taiex_data(days)
    finmind_login()
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockPrice", "data_id": stock_code, "start_date": start_date, "end_date": end_date}
        if finmind_token: params["token"] = finmind_token
        data = requests.get(url, params=params, timeout=15).json()
        if not data.get("data"): return pd.DataFrame()
        
        df = pd.DataFrame(data["data"])
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["Open"] = pd.to_numeric(df["open"], errors="coerce")
        df["High"] = pd.to_numeric(df["max"], errors="coerce")
        df["Low"] = pd.to_numeric(df["min"], errors="coerce")
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        return _clean_dataframe(df[["Date", "Open", "High", "Low", "Close"]])
    except:
        return pd.DataFrame()

# ==================================================
# 3. 新聞與特徵工程
# ==================================================
def get_stock_news(keyword, limit=5):
    try:
        query = urllib.parse.quote(f"{keyword} 股票")
        url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        res = requests.get(url, timeout=5)
        root = ET.fromstring(res.text)
        news_list = []
        for item in root.findall('.//item')[:limit]:
            news_list.append({"title": item.find('title').text, "link": item.find('link').text})
        return news_list
    except: return []

def calc_indicators(df):
    df = df.copy()
    close = df["Close"]
    df['MA_5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['RET_1'] = close.pct_change()
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['Volatility'] = df['RET_1'].rolling(20).std()
    return df.dropna()

# ==================================================
# 4. 回測引擎與 AI 機率生成 (核心升級)
# ==================================================
def run_backtest_for_web(df):
    try:
        if len(df) < 200: return None
        work_df = df.copy()
        work_df['Future_5d_Close'] = work_df['Close'].shift(-5)
        work_df['Target'] = (work_df['Future_5d_Close'] > work_df['Close']).astype(int)
        
        valid_df = work_df.dropna(subset=['Future_5d_Close']).copy()
        if len(valid_df) < 100: return None
        
        split_idx = int(len(valid_df) * 0.8)
        train_df = valid_df.iloc[:split_idx]
        test_df = valid_df.iloc[split_idx:].copy()
        
        features = ['MA_5', 'MA20', 'RET_1', 'RSI', 'Volatility']
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        
        model = LGBMClassifier(n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_train, train_df['Target'])
        
        # 💡 [真・AI 機制] 利用訓練好的模型，重新預測歷史每一天的真實勝率
        X_all = scaler.transform(df[features].ffill().bfill())
        df['AI_Prob'] = model.predict_proba(X_all)[:, 1] * 100
        
        importances = model.feature_importances_
        total_importance = sum(importances)
        feature_map = {
            'MA_5': '5日均線短線動能', 'MA20': '月線趨勢支撐狀態',
            'RET_1': '單日股價反轉動能', 'RSI': 'RSI 超買超賣冷熱度',
            'Volatility': '近20日價格波動收斂度'
        }
        top_features = []
        if total_importance > 0:
            feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:3]
            for f, imp in feat_imp:
                pct = (imp / total_importance) * 100
                top_features.append(f"{feature_map.get(f, f)} (貢獻度: {pct:.1f}%)")

        test_df['Prob'] = model.predict_proba(X_test)[:, 1]
        test_df['Signal'] = np.where(test_df['Prob'] > 0.60, 1, 0)
        test_df['Next_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Next_Return'])
        
        strategy_ret = (test_df['Signal'] * test_df['Next_Return']).values
        bh_ret = test_df['Next_Return'].values
        signals = test_df['Signal'].values
        
        if len(signals) == 0: return None
        
        strat_cum = np.cumprod(1 + strategy_ret)[-1] - 1
        bh_cum = np.cumprod(1 + bh_ret)[-1] - 1
        trades = strategy_ret[signals == 1]
        total_trades = len(trades)
        win_rate = (trades > 0).mean() * 100 if total_trades > 0 else 0
        profit_factor = (trades[trades > 0].sum() / abs(trades[trades < 0].sum())) if trades[trades < 0].sum() != 0 else 99.99
        
        days_in_test = len(test_df)
        annualized_ret = ((1 + strat_cum) ** (252 / days_in_test) - 1) * 100 if days_in_test > 0 else 0
        cum_ret_arr = np.cumprod(1 + strategy_ret)
        mdd = (cum_ret_arr / np.maximum.accumulate(cum_ret_arr) - 1).min() * 100
        std_dev = strategy_ret.std()
        sharpe = (strategy_ret.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        
        if total_trades == 0: conclusion = "⏸️ 訊號空窗：模型未發現高勝率進場點，選擇空手觀望。<br>🛒 買入建議：缺乏多頭動能，建議資金先停泊。<br>💰 賣出建議：若已持有，請嚴守個人停損。"
        elif strat_cum > bh_cum: conclusion = "✅ 策略優勢：高報酬且風險控制優異。<br>🛒 買入建議：預測看漲可進場。<br>💰 賣出建議：預測轉跌時果斷停利。" if sharpe > 1 else "✅ 擊敗大盤：能創造超額報酬。<br>🛒 買入建議：可進場分批佈局。<br>💰 賣出建議：見好就收。"
        else: conclusion = "🛡️ 下檔保護：大跌時具備避險作用。<br>🛒 買入建議：適合防禦型配置。<br>💰 賣出建議：不想資金閒置可轉換至強勢股。" if mdd > -15 else "⚠️ 模型失真：容易追高殺低。<br>🛒 買入建議：請避開。<br>💰 賣出建議：回歸均線判斷停損。"

        return {
            "days": days_in_test, "strat_cum": strat_cum * 100, "bh_cum": bh_cum * 100,
            "ann_ret": annualized_ret, "win_rate": win_rate, "trades": total_trades,
            "profit_factor": profit_factor, "mdd": mdd, "sharpe": sharpe, 
            "conclusion": conclusion, "top_features": top_features
        }
    except: return None

# ==================================================
# 5. 分析總控
# ==================================================
def analyze_stock(code):
    df = get_stock_data(code, 730)
    if df.empty or len(df) < 30: return None
    df = calc_indicators(df)
    if df.empty: return None

    backtest_data = run_backtest_for_web(df)
    if not backtest_data or 'AI_Prob' not in df: return None

    last = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    
    # 💡 直接提取模型預測的真實機率
    prob = int(df['AI_Prob'].iloc[-1]) 
    trend = "多頭" if last > ma20 else "空頭"
    name = get_stock_name(code)
    news = get_stock_news(name, limit=5)

    tv_df = df.copy()
    tv_df.reset_index(inplace=True)
    tv_df['Open'] = tv_df['Open'].fillna(tv_df['Close'])
    tv_df['High'] = tv_df['High'].fillna(tv_df['Close'])
    tv_df['Low'] = tv_df['Low'].fillna(tv_df['Close'])
    tv_df['High_corr'] = tv_df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    tv_df['Low_corr'] = tv_df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    last_date = tv_df['Date'].iloc[-1]
    last_vol = tv_df['Volatility'].iloc[-1] if pd.notna(tv_df['Volatility'].iloc[-1]) else 0.02
    drift = ((prob - 50) / 50.0) * (last_vol * last)
    
    future_points = [{'time': last_date.strftime('%Y-%m-%d'), 'value': last}]
    curr_date = last_date
    curr_price = last
    for _ in range(5):
        curr_date += datetime.timedelta(days=1)
        while curr_date.weekday() >= 5: curr_date += datetime.timedelta(days=1)
        curr_price += drift
        future_points.append({'time': curr_date.strftime('%Y-%m-%d'), 'value': round(curr_price, 2)})

    tv_df['Date'] = tv_df['Date'].dt.strftime('%Y-%m-%d')
    candle_data = tv_df[['Date', 'Open', 'High_corr', 'Low_corr', 'Close']].rename(
        columns={'Date': 'time', 'Open':'open', 'High_corr':'high', 'Low_corr':'low', 'Close':'close'}
    ).to_dict('records')
    ma_df = tv_df.dropna(subset=['MA20'])
    ma_data = ma_df[['Date', 'MA20']].rename(columns={'Date': 'time', 'MA20':'value'}).to_dict('records')
    
    # 💡 副圖現在顯示的是真正的 AI 評估走勢
    prob_data = tv_df[['Date', 'AI_Prob']].rename(columns={'Date': 'time', 'AI_Prob': 'value'}).to_dict('records')

    return {
        "code": code, "name": name, "price": last, "ma20": ma20, "rsi": rsi, 
        "trend": trend, "prob": prob, "news": news, "backtest": backtest_data,
        "tv_candles": json.dumps(candle_data), "tv_ma20": json.dumps(ma_data),
        "tv_prob": json.dumps(prob_data), "tv_prediction": json.dumps(future_points)
    }

def market_forecast(): return analyze_stock("TAIEX")

# ==================================================
# 6. 動態產業分類
# ==================================================
def build_market_map():
    market = {"全市場": [], "ETF專區": [], "AI伺服器": []}
    ai_names = {"鴻海", "廣達", "緯創", "英業達", "仁寶", "和碩", "華碩", "微星", "技嘉", "神達", "緯穎", "勤誠", "雙鴻", "奇鋐", "宏碁"}
    official_groups = set()
    for code, info in twstock.codes.items():
        if len(code) not in [4, 5]: continue
        group = getattr(info, "group", None) or getattr(info, "type", None)
        if group and isinstance(group, str) and group.strip(): official_groups.add(group.strip())
    for g in sorted(official_groups): market[g] = []
    for code, info in twstock.codes.items():
        if len(code) not in [4, 5]: continue
        name = info.name
        group = getattr(info, "group", None) or getattr(info, "type", None)
        group = group.strip() if isinstance(group, str) else None
        market["全市場"].append(code)
        if code.startswith("00"): market["ETF專區"].append(code)
        if name in ai_names: market["AI伺服器"].append(code)
        if group and group in market: market[group].append(code)
    return {k: v for k, v in market.items() if v}

industry_map = build_market_map()

def get_all_categories(): return list(industry_map.keys())
def get_category_total_pages():
    cats = get_all_categories()
    return 1 if not cats else (len(cats) + CATEGORY_PAGE_SIZE - 1) // CATEGORY_PAGE_SIZE

def get_category_page(page=1):
    cats = get_all_categories()
    total = get_category_total_pages()
    page = max(1, min(page, total))
    start = (page - 1) * CATEGORY_PAGE_SIZE
    return cats[start:start + CATEGORY_PAGE_SIZE], page, total

def build_category_quick_reply(page=1):
    categories, page, total = get_category_page(page)
    items = [QuickReplyButton(action=MessageAction(label=ind[:20], text=f"選產業_{ind}")) for ind in categories]
    if page < total and len(items) < 13:
        items.append(QuickReplyButton(action=MessageAction(label="更多分類▶", text=f"分類第_{page + 1}頁")))
    return QuickReply(items=items), f"請選擇市場類別（第 {page}/{total} 頁）👇"

def build_category_list_text():
    lines = ["📚 產業分類總表\n"]
    for i, c in enumerate(get_all_categories(), 1): lines.append(f"{i}. {c}")
    return "\n".join(lines[:120])

def build_style_result(category):
    arr = industry_map.get(category, [])[:10]
    if not arr: return "❌ 無資料"
    lines = [f"📈 {category} Top10\n", "🔥 激進型"]
    for i, c in enumerate(arr[:5], 1): lines.append(f"{i}. {c} {get_stock_name(c)}")
    lines.extend(["", "🛡 保守型"])
    for i, c in enumerate(arr[5:10], 1): lines.append(f"{i}. {c} {get_stock_name(c)}")
    return "\n".join(lines)

# ==================================================
# 7. UI 網頁渲染
# ==================================================
def render_dashboard(data):
    news_html = ""
    if data.get('news'):
        for n in data['news']: news_html += f'<a href="{n["link"]}" target="_blank" class="news-link">🔹 {n["title"]}</a>'
    else: news_html = "暫無相關新聞"

    backtest_html = ""
    xai_html = ""
    
    if data.get('backtest'):
        bt = data['backtest']
        if 'top_features' in bt and len(bt['top_features']) >= 3:
            xai_html = f"""
            <div class="card small" style="border-left: 4px solid #ff9800;">
                <h2 style="color: #ff9800; border-bottom: none; margin-bottom: 5px;">🤖 AI 決策核心邏輯</h2>
                <div style="font-size: 15px; color: #bbb; margin-bottom: 15px;">模型運算之關鍵特徵權重解析 (Feature Importance)</div>
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 12px; margin-bottom: 10px;">🥇 <span style="color:#fff;">{bt['top_features'][0]}</span></div>
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 12px; margin-bottom: 10px;">🥈 <span style="color:#fff;">{bt['top_features'][1]}</span></div>
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 12px;">🥉 <span style="color:#fff;">{bt['top_features'][2]}</span></div>
            </div>
            """

        backtest_html = f"""
        <div class="card small">
            <h2>📊 AI 歷史回測報告 (近 {bt['days']} 交易日)</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px;">
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">AI 策略報酬</div><div class="highlight" style="font-size: 1.3em;">{bt['strat_cum']:.2f}%</div></div>
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">買進持有報酬</div><div style="font-size: 1.3em; color: #ddd;">{bt['bh_cum']:.2f}%</div></div>
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">進場勝率</div><div style="font-size: 1.3em; color: #ddd;">{bt['win_rate']:.1f}%</div></div>
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">交易次數</div><div style="font-size: 1.3em; color: #ddd;">{bt['trades']} 次</div></div>
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">最大回檔</div><div style="font-size: 1.3em; color: #ff6b6b;">{bt['mdd']:.2f}%</div></div>
                <div style="background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; text-align: center;"><div style="font-size: 13px; color: #aaa; margin-bottom: 5px;">夏普值</div><div style="font-size: 1.3em; color: #ddd;">{bt['sharpe']:.2f}</div></div>
            </div>
            <div style="background: rgba(0,242,254,0.05); border-left: 4px solid #00f2fe; padding: 18px; border-radius: 0 12px 12px 0;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #00f2fe; font-size: 18px;">💡 資產管理評估</div>
                <div style="color: #e0e0e0; line-height: 1.6;">{bt['conclusion']}</div>
            </div>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{data['name']} 分析報告</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"></script>
<style>
    body {{ margin:0; background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); background-attachment: fixed; color: #f1f1f1; font-family: 'Noto Sans TC', sans-serif; }}
    .wrap {{ max-width:920px; margin:auto; padding:30px 20px 60px; }}
    h1 {{ font-size:42px; margin-bottom:24px; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }}
    .card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.15); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); border-radius: 20px; padding: 26px; margin-bottom: 24px; transition: transform 0.3s ease; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; }}
    .small {{ font-size:17px; line-height:1.8; }}
    .highlight {{ color: #00f2fe; font-weight: bold; font-size: 1.1em; }}
    h2 {{ font-size: 22px; margin-top: 0; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }}
    .news-link {{ display: block; color: #e0e0e0; text-decoration: none; margin-bottom: 14px; line-height: 1.5; }}
    #tvchart {{ width: 100%; height: 450px; border-radius: 12px; overflow: hidden; margin-top: 10px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>{data['name']} ({data['code']})</h1>
<div class="card small">
    💰 最新收盤：<span class="highlight">{data['price']:.2f}</span><br>
    📈 當前趨勢：{data['trend']}<br>
    🎯 真實 AI 預測勝率：<span class="highlight">{data['prob']}%</span>
</div>

<div class="card">
    <h2>📈 互動式技術線圖與 AI 預測軌跡</h2>
    <div id="tvchart"></div>
</div>

<div class="grid">
    {xai_html}
    <div class="card small">
        <h2>📑 指標摘要</h2>
        📈 趨勢判讀：{data['trend']}<br>
        🌊 均線狀態：{'站上 MA20 (支撐強)' if data['price'] > data['ma20'] else '跌破 MA20 (壓力大)'}<br>
        🌡 RSI 強弱：{'動能偏強' if data['rsi'] >= 55 else '中性' if data['rsi'] >= 45 else '動能偏弱'}<br>
    </div>
</div>

{backtest_html}

<div class="card small">
    <h2>📰 相關即時新聞</h2>
    {news_html}
</div>
<div class="card small" style="font-size: 14px; color: #aaa;">
    免責聲明：本系統資訊與回測績效均為程式自動運算，新聞取自外部來源，不構成任何真實投資與買賣建議，歷史績效亦不代表未來表現。
</div>
</div>

<script>
    try {{
        const domElement = document.getElementById('tvchart');
        const chartOptions = {{
            autoSize: true,
            layout: {{ background: {{ type: 'solid', color: 'transparent' }}, textColor: '#d1d4dc' }},
            grid: {{ vertLines: {{ color: 'rgba(42, 46, 57, 0.15)' }}, horzLines: {{ color: 'rgba(42, 46, 57, 0.15)' }} }},
            timeScale: {{ timeVisible: true }},
        }};

        const chart = LightweightCharts.createChart(domElement, chartOptions);

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#ef5350', downColor: '#26a69a', borderDownColor: '#26a69a', borderUpColor: '#ef5350', wickDownColor: '#26a69a', wickUpColor: '#ef5350'
        }});
        const candleData = {data['tv_candles']};
        candleSeries.setData(candleData);

        const ma20Series = chart.addLineSeries({{ color: '#00f2fe', lineWidth: 1, title: 'MA20' }});
        ma20Series.setData({data['tv_ma20']});
        
        const predSeries = chart.addLineSeries({{
            color: '#ff9800', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'AI 5日預測'
        }});
        predSeries.setData({data['tv_prediction']});

        const probSeries = chart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '', 
            scaleMargins: {{ top: 0.8, bottom: 0 }}
        }});
        const rawProb = {data['tv_prob']};
        probSeries.setData(rawProb.map(d => ({{
            time: d.time, value: d.value, color: d.value >= 50 ? 'rgba(38, 166, 154, 0.4)' : 'rgba(239, 83, 80, 0.4)'
        }})));
        
        if (candleData.length > 120) {{
            chart.timeScale().setVisibleLogicalRange({{ from: candleData.length - 120, to: candleData.length + 5 }});
        }}
        
    }} catch (error) {{
        document.getElementById('tvchart').innerHTML = "<div style='color:#ff6b6b; padding: 20px;'>圖表繪製失敗：" + error.message + "</div>";
    }}
</script>
</body>
</html>
"""
    return render_template_string(html)

# ==================================================
# 8. 網頁路由與 LINE 處理 (💡 修復：完美恢復所有選單指令)
# ==================================================
@app.route("/")
def home(): return "<h1>AI 台股系統 v4.0 真・AI 決策版 正常運作中</h1>"

@app.route("/stock/<stock_code>")
def stock_page(stock_code):
    code, _ = search_stock_code(stock_code)
    if not code: code = stock_code
    data = analyze_stock(code)
    return render_dashboard(data) if data else "查無資料"

@app.route("/market")
def market_page():
    data = market_forecast()
    return render_dashboard(data) if data else "查無資料"

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except InvalidSignatureError: abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text.strip()
    
    # 💡 教練自首：把之前誤刪的按鈕與導覽全部乾淨俐落地加回來了
    if msg == "大盤預測":
        data = market_forecast()
        if not data:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="大盤資料暫時無法取得，請稍後再試。"))
            return
        url = f"{request.host_url}market".replace("http://", "https://")
        text = (f"📊 台股大盤（加權指數）\n\n💰 指數點位：{data['price']:.2f}\n"
                f"📈 當前趨勢：{data['trend']}\n🎯 真實 AI 預測勝率：{data['prob']}%\n\n"
                f"📌 點擊查看完整圖表與回測：\n{url}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
        
    elif msg == "預測":
        quick_reply, text = build_category_quick_reply(1)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text, quick_reply=quick_reply))
        
    elif msg.startswith("分類第_") and msg.endswith("頁"):
        try: page = int(msg.replace("分類第_", "").replace("頁", ""))
        except: page = 1
        quick_reply, text = build_category_quick_reply(page)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text, quick_reply=quick_reply))
        
    elif msg == "產業列表":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=build_category_list_text()))
        
    elif msg.startswith("選產業_"):
        cat = msg.replace("選產業_", "")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=build_style_result(cat)))
        
    elif msg == "免責聲明":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="本系統資訊僅供研究參考，不構成投資建議，投資盈虧請自負。"))
        
    else:
        code, name = search_stock_code(msg)
        if code:
            data = analyze_stock(code)
            if not data:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="查無資料，請稍後再試。"))
                return
            url = f"{request.host_url}stock/{code}".replace("http://", "https://")
            text = (f"📊 {name} ({code})\n\n💰 最新收盤：{data['price']:.2f}\n"
                    f"📈 當前趨勢：{data['trend']}\n🎯 真實 AI 預測勝率：{data['prob']}%\n\n"
                    f"📌 點擊查看完整圖表與回測：\n{url}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入股票代碼，或輸入：預測 / 大盤預測 / 產業列表"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
