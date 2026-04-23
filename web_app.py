import streamlit as st
import psycopg2
import os
import pandas as pd

# 頁面基本設定
st.set_page_config(page_title="AI 投資戰報", layout="wide")
st.title("🏆 AI 全市場篩選：今日最強勢標的")

# 連線至資料庫讀取資料的函式
@st.cache_data(ttl=3600) # 設定快取，避免頻繁讀取資料庫
def load_data():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    # 讀取剛剛 worker.py 寫入的資料
    df = pd.read_sql_query("SELECT stock_id, win_rate, rank_label, updated_at FROM top_stocks;", conn)
    conn.close()
    return df

# 執行讀取並顯示在網頁上
try:
    df = load_data()
    if df.empty:
        st.warning("目前尚無運算資料，請等待系統排程更新。")
    else:
        st.success(f"資料最後更新時間：{df['updated_at'].iloc[0]}")
        
        # 使用 Streamlit 欄位排版
        cols = st.columns(len(df))
        for index, row in df.iterrows():
            with cols[index]:
                st.metric(label=f"股票代碼：{row['stock_id']}", value=f"{row['win_rate']*100:.1f}%")
                st.write(f"綜合評判：{row['rank_label']}")
except Exception as e:
    st.error(f"資料庫連線失敗：{e}")
