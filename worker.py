import os
import psycopg2
from datetime import datetime

# 連線至 Render 的 PostgreSQL 資料庫
def get_db_connection():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    return psycopg2.connect(DATABASE_URL)

def run_background_analysis():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 建立資料表（若不存在）：確保存放最佳五檔股票的空間
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS top_stocks (
            id SERIAL PRIMARY KEY,
            stock_id VARCHAR(10),
            win_rate FLOAT,
            rank_label VARCHAR(20),
            updated_at TIMESTAMP
        );
    """)
    
    # 清空舊資料，確保網頁永遠顯示最新數據
    cursor.execute("TRUNCATE TABLE top_stocks;")
    
    # 【此處替換為您的 LightGBM 運算邏輯】
    # 假設您已經算出最佳五檔並存入 top_5_results 字典中
    top_5_results = {
        "2330": {"win_rate": 0.771, "rank_label": "強勢看漲"},
        "2317": {"win_rate": 0.652, "rank_label": "偏向看漲"},
        "2454": {"win_rate": 0.583, "rank_label": "中性震盪"},
        "2308": {"win_rate": 0.551, "rank_label": "中性震盪"},
        "2412": {"win_rate": 0.510, "rank_label": "偏向看跌"}
    }
    
    # 將最新數據寫入資料庫
    current_time = datetime.now()
    for stock_id, info in top_5_results.items():
        cursor.execute(
            "INSERT INTO top_stocks (stock_id, win_rate, rank_label, updated_at) VALUES (%s, %s, %s, %s)",
            (stock_id, info["win_rate"], info["rank_label"], current_time)
        )
        
    conn.commit()
    cursor.close()
    conn.close()
    print("背景運算完成，資料已寫入 PostgreSQL。")

if __name__ == "__main__":
    run_background_analysis()
