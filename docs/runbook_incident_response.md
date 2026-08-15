# ABSORB 故障應對手冊

## 範圍與原則

本手冊處理本地量化批次、私有 GCS 快照與 Cloud Run 讀取端的營運事件。不要在事件期間修改 `app.py`、`local_quant.py`、模型權重、策略閾值或 Firestore 資料。

正式量化發布目前使用下列指標：

- `gs://line-stock-bot-498908-quant-snapshots/quant/v1/latest-TW.json`
- `gs://line-stock-bot-498908-quant-snapshots/quant/v1/latest-US.json`
- `gs://line-stock-bot-498908-quant-snapshots/quant/v1/manifests/<market>-<run>-<sha>.json`

發布順序固定為 object、manifest、latest。事件處理禁止刪除 object、manifest、checkpoint 或 Secret Manager secret。

## 事件分級

| 分級 | 條件 | 初始處置 |
| --- | --- | --- |
| WARNING | 新鮮度超時、覆蓋率介於 95% 與 100%、資料來源暫時限流 | 保留現行 latest，建立追蹤事件 |
| CRITICAL | Publish Gate 拒收、覆蓋率低於 95%、批次中斷、回滾觸發 | 停止發布、保留證據、通知 on-call |

所有事件都要記錄時間、market、現行 manifest、候選 manifest、命令 exit code 與處置結果；不得記錄 token、密碼、完整 Webhook URL 或 secret 值。

## API 429 或資料來源限流

1. 確認來源、market、首次失敗時間與連續失敗次數。
2. 不重跑全市場，也不手動修改快照內容。
3. 保留現行 latest，讓 Cloud Run 使用已驗證快照或既有即時計算降級。
4. 等候來源配額窗口後，以既有排程的單一執行個體重試。
5. 若連續兩個排程窗口失敗，升級為 CRITICAL，檢查 FinMind、Yahoo 與網路狀態。

## Publish Gate 拒收

1. 確認 `VALIDATION_FAILED` 的 staging 路徑與錯誤類別。
2. 驗證正式 latest 未改變：下載 latest，核對其 manifest SHA-256。
3. 不手動修補 staging JSON，也不覆寫正式 latest。
4. 依錯誤類別處理：schema 差異回到資料來源契約；雜湊不符重建批次；覆蓋率不足保留上一版本。
5. 重新執行前必須通過 Quality Gate。

## TW 最新已完成交易日追趕發布

只有在正式 TW latest 落後、且需要補上「本地今日以前最新一個已完成交易日」時，才可使用專用入口：

```powershell
.\scripts\catch_up_latest_completed_session.ps1 `
  -DataRoot 'D:\AbsorbData' `
  -TargetDate 'YYYY-MM-DD'
```

入口會固定台北當地日期，驗證 TargetDate 是 TW 交易日且沒有更晚的已完成交易日；再以正常 post-close pipeline 的 child-process exit code 驗證 terminal checkpoint、以 exact manifest path/SHA 建立並 promote Observation candidate，確認 report index 只新增該 TargetDate 的 observation post-close entry，最後才 capture LKG 並交給既有 generation-precondition uploader。入口與會寫入 TW observation release 的 task wrappers 共用跨進程 writer mutex；若 live 已是相同 TargetDate 且 identity 完整一致，會在任何 local pipeline/promote 前以 no-op idempotent 路徑結束。不得把這個入口擴充成任意歷史回補，也不得在正常 `run_tw_post_close_pipeline.ps1` 上加 `-PublishObservation` 來繞過 historical guard。

若四個 Observation pointer 日期不一致、已有 pending pointer journal、來源 manifest identity 不同、capture 前 generation 改變，或任何驗證／read-back 失敗，入口會停止；保留 LKG receipt、before evidence 與 pending journal 供人工 reconciliation，不得 fresh capture 重試或直接手動覆寫 latest。

## 自動或讀取端回滾事件

1. 確認現行 manifest 是否可讀、SHA-256 是否相符、Cloud Run 是否已降級。
2. 選擇最近一份已有 Quality Gate `PASS`、相同 market、覆蓋率至少 95% 的歷史 manifest 作為 LKG。
3. 記下 LKG manifest 路徑與 SHA-256；不要以檔名時間推測資料正確性。
4. 執行手動回滾，並在 10 秒內完成 latest 指標驗證。

```powershell
.\scripts\manual_rollback.ps1 `
  -Market TW `
  -LkgManifest 'manifests/TW-YYYYMMDDTHHMMSSZ-aaaaaaaaaaaa.json' `
  -Confirm
```

腳本會驗證 schema、market、coverage、manifest SHA-256，並以 GCS generation precondition 更新 latest。若 generation 已變更，表示存在並行發布，必須停止並重新判讀最新狀態。

5. 回滾後讀取 latest 與 manifest，確認 market、manifest 路徑與 SHA-256。
6. 驗證 `/health` 與一筆 TW、US 查詢；不要為回滾重啟 Cloud Run。

## 本機磁碟空間不足

1. 讀取 `D:\AbsorbData\logs\runner-status.json` 與 `upload-status.json`；尚未 cutover 時才讀舊 root。
2. 若可用容量低於 100 GB，停止新的批次與上傳。
3. 僅使用既有 allowlist 清理機制處理 cache、raw、暫存與過期發布資料。
4. 禁止刪除 checkpoint、已發布 artifact、secrets，或 allowlist 以外路徑。
5. 恢復到門檻以上後，以單一排程實例續跑。

## 事件結案

事件結案前必須具備：發生與恢復時間、影響 market、最終 latest SHA-256、Quality Gate 報告、測試結果、是否回滾與後續修正責任人。若原因未知，事件維持 Open，不得標示為已恢復。
