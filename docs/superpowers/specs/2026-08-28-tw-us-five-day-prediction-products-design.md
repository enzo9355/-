# 台美五交易日預測產物設計

## 目標

ABSORB 在台股與美股使用相同且可驗證的預測語義：

- 上漲機率：第 5 個交易日收盤價高於目前收盤價的機率。
- 預測價格：第 5 個交易日的預測收盤價。
- 預估漲跌幅：預測價格相對目前收盤價的變化。
- 不提供買進、賣出或部位指令。

首頁涵蓋 TAIEX、S&P 500、Nasdaq Composite 與道瓊工業指數；個股頁涵蓋所有通過當日資料與身分驗證的台股及美股。

## 架構

沿用現有 LightGBM、五交易日標籤、walk-forward OOS、不可變 quant 快照與 promotion gate，不建立第二套訓練平台。

現有分類器保留為方向機率輸出；同一份特徵、切割與五日 gap 增加一個 LightGBM 回歸器，預測五日報酬。第 5 日預測收盤價由目前已驗證收盤價乘以 `1 + predicted_return_5d` 得出。訓練、回測和每日推論只在本機批次執行；Cloud Run 只驗證並讀取產物。

四個指數與個股共用相同輸出語義。指數使用各市場的官方或既有已驗證價格序列；特殊指數代碼由明確 allowlist 接受，不放寬一般個股代碼驗證。

## 模型驗證與 promotion

方向模型沿用 leakage、五日 gap、calibration、schema、security 與 quality gate。價格模型增加以下 OOS 證據：

- MAE 與 RMSE。
- 相對「第 5 日價格等於當前價格」naive baseline 的 MAE 比率。
- 預測報酬與實際五日報酬的樣本數及有限值檢查。

正式 prediction 產物只接受所有 gate 通過、模型版本與特徵 schema 相容的 promoted backtest。價格品質 gate 未通過時，機率與價格一起停止正式發布，避免同一張卡片混合不同可信度狀態。

## 正式產物

新增 `predictions/v1/` GCS namespace，並保留既有 namespace allowlist 的 fail-closed 行為。每個市場每日只有一個不可變批次物件與一個 latest pointer：

- `predictions/v1/objects/<sha256>.json`
- `predictions/v1/latest-TW.json`
- `predictions/v1/latest-US.json`

不可變文件包含：schema、market、as-of、generated-at、horizon、來源 quant manifest 路徑與 SHA-256、promoted backtest SHA-256、model version、feature schema version，以及 entity map。每個 entity 僅包含代碼、類型、目前收盤價、五日上漲機率、五日預測收盤價、預估漲跌幅、目標交易日與可選的 OOS 品質摘要。

發布順序固定為 immutable object first、generation-guarded latest pointer second。讀取順序固定為 pointer first、size/hash 驗證、immutable object second、完整 schema 與來源綁定驗證。

## 資料與發布流程

台股在台灣收盤資料完成並通過驗證後發布；美股在美東收盤資料完成並通過驗證後發布。目標日期由各自交易所行事曆計算，不以自然日加五天。

批次流程為：

1. 載入當日已驗證 quant manifest 與 promoted backtest。
2. 驗證市場、日期、來源 SHA、模型版本、特徵 schema 與全部 gate。
3. 收集具有有限機率、預測報酬及當日收盤價的 entity。
4. 計算第 5 交易日與預測價格，建立不可變 prediction 文件。
5. 驗證文件後發布 immutable object，再原子更新 latest pointer。

任一關鍵驗證失敗時不更新 latest。網站可以顯示前次結果的明確日期，但不得把它當成當日預測。

## 網站整合

Cloud Run 以獨立 verified reader 讀取 TW 與 US prediction 產物，在服務層按市場、代碼與 as-of 合併進現有 observation view，不把預測欄位塞回 observation artifact。

首頁指數區顯示 K 線、目前價格、五日上漲機率、預測價格、預估漲跌幅與目標交易日。美股頁可切換 S&P 500、Nasdaq Composite 與道瓊；台股預設 TAIEX。

個股頁在現有真實 K 線右側延伸一個第 5 日預測終點與虛線，不生成中間四天的假 K 線。主要資訊顯示機率、預測價格與目標日；模型版本、OOS 指標、來源日與風險說明放在展開區。

無有效產物時顯示「本交易日預測尚未發布」。過期產物顯示其資料日並標示「前次預測」，不顯示為目前狀態。

## 安全與錯誤處理

- 不在 request-time 訓練、抓取全市場資料或推論。
- 不允許 observation 內嵌 prediction 欄位繞過 verified reader。
- 不接受 sample、NaN、Infinity、未知市場、未知指數、日期錯置或雜湊不符。
- 不因預測缺失而隱藏已驗證的市場實況；實況與預測狀態分開呈現。
- 不降低既有 coverage、資料完整性、生命週期或 security master gate。

## 驗證

- 單元測試：回歸標籤、五日 gap、OOS 價格指標、target session、schema、hash/size、stale、model mismatch 與 namespace allowlist。
- 整合測試：TW/US builder、publisher/reader round trip、observation merge、四個指數與一般個股。
- UI 測試：首頁與個股的可用、過期、缺失狀態；桌面、4K、手機；鍵盤、overflow、圖表 resize 與 console。
- 發布前執行 focused tests、完整測試、Python compile、JavaScript syntax、template smoke 與 `git diff --check`。

## 不在範圍

- 逐日五根預測 K 線。
- 買賣訊號、自動下單或投資組合配置。
- request-time 模型服務。
- 為通過發布而產生合成價格、降低 gate 或手動覆寫 latest pointer。

