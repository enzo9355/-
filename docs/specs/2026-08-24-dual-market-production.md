# ABSORB 雙市場正式產品規格

## 目標

將 ABSORB 交付為台股與美股對等、資料來源可追溯、排程不互相阻塞的正式金融產品，並修復台股停在 2026-08-19 的生命週期資料缺口。

## 使用者介面

- 左上草寫文字標誌固定顯示 `ABSORB`，不搭配圓形圖示，並連回目前市場的研究摘要。
- 浮動問答入口、對話標題與相關無障礙文字固定顯示 `ASK ABSORB`。
- 市場切換的台股入口為 `/`，美股入口為 `/us`；兩者皆為市場研究摘要，而非報告列表。
- 台股與美股各自提供相同五個入口：市場研究摘要、市場實況、產業觀察、個股與 ETF、每日報告。
- 路由對應如下：
  - TW: `/`, `/market`, `/industries`, `/stocks?market=TW`, `/reports`
  - US: `/us`, `/us/market`, `/us/industries`, `/us/stocks`, `/reports/us`
- 美股市場頁只能使用已驗證的 US professional report 與 report metadata，不得複製或推定台股資料。
- `/stock/<code>` 必須依標的自動選擇 TW/US 導覽狀態。
- 桌面與行動版導覽都必須保留市場脈絡；無資料、延遲、錯誤與超長文字必須有安全呈現。
- 動畫需短促、低振幅、可由 `prefers-reduced-motion` 關閉。

## 資料品質與生命週期

- 2867 自 2026-08-20 起為官方停止買賣；2026-09-01 才依官方日期成為終止上市。
- 權威公告來源為 TWSE 新聞稿 `https://investoredu.twse.com.tw/FileSystem/FileUpload/88ff18ef-5726-4b33-b207-f92310023328.pdf`，2026-08-24 下載原始檔大小 139,878 bytes，SHA-256 `3ff4455c1435b5d0dc62803953241d184c13775662eb46f2feaf25d3d300c768`。
- 修正必須是一般化的 TWSE 官方生命週期來源／欄位解析，不得硬編碼單一股票或合成狀態。
- 生命週期事件必須保留 source id、effective date、原始列 hash、payload hash、parser version 與 evidence hash。
- 未知缺價、日期不符、hash 不符、來源格式變更或同日價格衝突仍須 fail closed。
- 狀態標的不得進入當日價格、報酬、成交量、市場寬度或異常漲跌計算。

## 排程與執行環境

- TW、US 觀察工作使用市場別 mutex；真正共享的發布流程使用有上限的等待，不得碰鎖立即失敗。
- `ABSORB-FullBacktest` 改為每日 22:30，最長執行至 02:15；完成 checkpoint 時須在載入 yfinance 前成功退出並停用排程。
- 正式 Python runtime 必須滿足 `requirements.txt`，避免 `US-Daily` 因缺少 yfinance 失敗。
- installer 與 active Scheduled Task 必須一致；保留 hidden launcher、使用者身分、工作目錄、日誌、exit code、retry 與 idempotency。

## 發布與驗收

- 先補跑缺失的已完成 TW 交易日，再以 immutable object、readback、generation precondition、reader-first/pointer-second 更新正式指標。
- 不得執行尚未完成的市場交易日，不得為通過 gate 刪除或偽造資料。
- Cloud Run 新 revision 必須帶正確 commit provenance 並承接 100% traffic；保留可回復 revision。
- 正式驗收需涵蓋 TW/US 摘要、五個入口、TW 與 AAPL 搜尋、資料日期、桌面與 390px 行動版、console、network、404/503、health 與 Cloud Run traffic。
