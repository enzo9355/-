# 台美五交易日預測產物 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為台股、美股、TAIEX、S&P 500、Nasdaq Composite 與道瓊建立可驗證的五日上漲機率及第 5 交易日預測收盤價，並整合到首頁與個股頁。

**Architecture:** 沿用現有 LightGBM 五日分類模型與 walk-forward OOS，加入同特徵同切割的回歸頭。每日本機批次把通過 promotion 的 quant 輸出封裝為不可變 `predictions/v1` 產物；Cloud Run 只以 verified reader 讀取並合併到 observation view。

**Tech Stack:** Python 3.12、LightGBM、pandas、NumPy、Flask、Jinja、Lightweight Charts、unittest、PowerShell

**Spec:** `docs/superpowers/specs/2026-08-28-tw-us-five-day-prediction-products-design.md`

## Global Constraints

- 不增加模型或前端依賴。
- 預測期間固定為 5 個交易日；不建立逐日假 K 線。
- 台股與美股使用各自交易所行事曆。
- Cloud Run 不執行訓練、全市場資料抓取或 request-time 推論。
- 任一來源、hash、schema、promotion 或日期驗證失敗時 fail closed。
- 不提供買賣指令，也不降低既有資料與 publication gate。

---

### Task 1: 五日價格回歸輸出

**Files:**
- Modify: `stock_papi/quant/model.py`
- Modify: `stock_papi/quant/tw_incremental.py`
- Modify: `tests/test_daily_inference.py`
- Modify: `tests/test_prediction_pipeline.py`

**Interfaces:**
- Consumes: `add_prediction_target(frame)`、`build_time_splits(n)`、既有 `MODEL_FEATURES`。
- Produces: `AI_PRED_RET_5` 與 `AI_PRED_PRICE_5` 最新列；`run_latest_inference()` 回傳 `predicted_return_5d`、`predicted_price_5d`；`run_ai_engine(include_oos=True)` 回傳 OOS regression evidence。

- [ ] 在 `tests/test_daily_inference.py` 增加 fake classifier/regressor，斷言最新列的機率、五日預測報酬及預測價格皆為有限值。
- [ ] 執行 `python -m unittest tests.test_daily_inference -v`，確認因回歸輸出不存在而失敗。
- [ ] 在 `stock_papi/quant/model.py` 使用既有 `MODEL_SETTINGS` 建立 `LGBMRegressor`；每折使用相同 train/test index，OOS 預測 `FUTURE_RET_5`；全資料模型只寫入最新列。
- [ ] 計算 `mae`、`rmse`、`naive_mae`、`mae_ratio` 與 `sample_count`，拒絕非有限值、非正價格與少於 30 筆 OOS。
- [ ] 將兩個欄位加入 `stock_papi/quant/tw_incremental.py` 的允許欄位，不改變 OHLC 驗證。
- [ ] 重跑兩個 focused test files 並提交 `feat: add five-day price regression output`。

### Task 2: Prediction 產物 schema、builder 與 verified reader

**Files:**
- Create: `stock_papi/batch/prediction_products.py`
- Create: `stock_papi/repositories/prediction_snapshots.py`
- Modify: `stock_papi/repositories/gcs.py`
- Create: `tests/test_prediction_products.py`
- Create: `tests/test_prediction_snapshot_repository.py`
- Modify: `tests/test_gcs_repository.py`

**Interfaces:**
- Produces: `build_prediction_product(market, quant_manifest, snapshots, promoted_backtest, next_session)`、`validate_prediction_product(document)`、`load_prediction_snapshot(market, today, load_object)`。
- Entity shape: `symbol`、`entity_type`、`as_of`、`target_session`、`current_price`、`up_probability`、`predicted_price`、`predicted_change_pct`。

- [ ] 寫入 TW/US 成功案例，以及未知指數、日期錯置、model mismatch、未通過 gate、NaN、錯誤 hash、超限大小與 stale pointer 的失敗測試。
- [ ] 執行三個 focused test files，確認 module/namespace 尚不存在而失敗。
- [ ] 實作單一 schema validator 與 builder；特殊指數只允許 `TAIEX`、`^GSPC`、`^IXIC`、`^DJI`。
- [ ] 實作 reader-first/pointer-second 讀取：驗證 pointer、size、SHA-256、immutable object、market/as-of/source/model identity；cache key 必須含 market。
- [ ] 將 `predictions/v1/` 加入 GCS allowlist 並保留其他路徑拒絕測試。
- [ ] 重跑 focused tests 並提交 `feat: add verified prediction products`。

### Task 3: 交易日與服務層合併

**Files:**
- Modify: `stock_papi/integrations/market_data/calendar.py`
- Modify: `stock_papi/integrations/market_data/us_calendar.py`
- Create: `stock_papi/services/prediction_view.py`
- Modify: `stock_papi/services/observation_view.py`
- Create: `tests/test_prediction_views.py`
- Modify: `tests/test_batch_calendar.py`

**Interfaces:**
- Produces: `fifth_session_after(market, as_of)` 與 `prediction_for(snapshot, market, symbol, observation_as_of)`。

- [ ] 增加跨週末與已知休市日的第 5 交易日測試，以及 observation 日期不符、舊 prediction、未知 symbol 的 fail-closed 測試。
- [ ] 執行 focused tests，確認 helper 尚不存在而失敗。
- [ ] 沿用既有 calendar session resolver 計算第 5 個後續 session；不以自然日推算。
- [ ] 實作純服務合併，只在 market、symbol 與 as-of 完全一致時回傳 prediction view；觀察資料本身永遠可獨立顯示。
- [ ] 重跑 focused tests並提交 `feat: merge verified forecasts into observation views`。

### Task 4: Flask wiring 與 UI

**Files:**
- Modify: `stock_papi/application.py`
- Modify: `stock_papi/web/routes/dashboard.py`
- Modify: `stock_papi/web/routes/market.py`
- Modify: `templates/dashboard.html`
- Modify: `templates/stock_detail.html`
- Modify: `static/app.js`
- Modify: `static/style.css`
- Modify: `tests/test_web_product.py`
- Modify: `tests/visual_qa_server.py`

**Interfaces:**
- Consumes: `load_prediction_snapshot("TW"|"US")` 與 `prediction_for(...)`。
- Produces: 首頁 `market_prediction`、美股三指數切換資料，以及個股 `prediction`。

- [ ] 新增首頁與個股的 available、unavailable、stale 渲染測試，斷言無買賣字樣且沒有五根未來 K 線。
- [ ] 執行 `python -m unittest tests.test_web_product -v`，確認新資訊尚未渲染而失敗。
- [ ] 在 application bootstrap 建立 prediction reader；route 只注入已驗證 view，不讀 observation 內嵌 prediction。
- [ ] 首頁在指數 K 線右側顯示上漲機率、預測價格、預估漲跌幅與目標日；US 以文字 tabs 切換三指數。
- [ ] 個股圖只新增目前收盤至第 5 日終點的虛線與終點標籤；缺失時顯示「本交易日預測尚未發布」。
- [ ] 使用既有 CSS token 完成 desktop、4K、mobile，保持 opaque panel、鍵盤 focus 與圖表 resize。
- [ ] 重跑 web tests、`node --check static/app.js` 並提交 `feat: show verified five-day forecasts`。

### Task 5: 本機產製與既有 publication 串接

**Files:**
- Create: `stock_papi/batch/prediction_products_cli.py`
- Modify: `scripts/upload_local_quant.ps1`
- Modify: `scripts/deploy_observation_production.ps1`
- Modify: `scripts/verify_cutover.ps1`
- Create: `tests/test_prediction_products_cli.py`
- Modify: `tests/test_observation_release_scripts.py`
- Modify: `tests/test_observation_deploy_scripts.py`

**Interfaces:**
- CLI: `python -m stock_papi.batch.prediction_products_cli --root <path> --market TW|US`，只輸出本機 immutable object 與 latest pointer staging files。

- [ ] 新增 CLI 拒絕缺少 promoted backtest、來源 manifest mismatch、price gate failure 與空 entity 的測試。
- [ ] 執行 focused tests，確認 CLI 尚不存在而失敗。
- [ ] 實作 CLI 載入既有 local quant manifest、promotion 與 snapshots，建立並驗證 prediction product。
- [ ] 在既有 upload transaction 中先上傳 immutable object，再以 generation precondition 更新 prediction latest；不得 raw overwrite。
- [ ] deploy/verify 只要求 reader 與 API schema 能 fail closed；沒有有效正式 prediction 時不得把模式切成 enabled。
- [ ] 重跑 scripts focused tests 並提交 `feat: publish gated prediction artifacts`。

### Task 6: 完整驗證與發布

**Files:**
- Modify only if validation exposes an in-scope defect.

- [ ] 執行 prediction、calendar、repository、web、release scripts focused suites。
- [ ] 執行完整 `python -m unittest discover -s tests -v` 並記錄精確測試數。
- [ ] 執行 `python -m compileall -q stock_papi tests`、`node --check static/app.js`、template smoke 與 `git diff --check`。
- [ ] 啟動 `tests/visual_qa_server.py`，檢查 390x844、1440x1000、3840x2160 的首頁、US 切換、個股與缺失狀態，確認無 overflow 與 console error。
- [ ] 提交修正、push、更新 PR；只有真實 TW/US prediction artifacts 與所有 gate 通過時才 merge/deploy。
- [ ] 使用既有 production release path 部署，驗證 Cloud Run 100% traffic revision、首頁、個股、API、資產與 production pointer readback；否則保留 Draft/未部署並以證據標示 BLOCKED。

