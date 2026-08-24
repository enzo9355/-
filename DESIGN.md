# ABSORB Design System

ABSORB 是 AI 量化市場情報與決策輔助系統。介面先服務資料閱讀，再表達品牌；任何視覺處理都不得暗示保證獲利。

## 1. Brand concept

「吸收資料、辨識證據、形成可驗證判斷」。品牌名稱固定寫作 `ABSORB`，不得使用舊品牌、人格式暱稱或蝴蝶意象。

## 2. Design principles

- Evidence first：結論、依據、反對證據、限制依序呈現。
- Quiet precision：白色畫布、深藍識別、細邊界、少裝飾。
- Honest states：缺值保留缺值，過期資料標日期，不用顏色掩飾資料品質。
- Same meaning everywhere：Web、LINE 與報告使用相同 action label、日期與限制。

## 3. Visual atmosphere

研究平台而非促銷頁。大量留白、清楚層級、平面表面；不使用玻璃擬態、金屬感、黑金風、裝飾性 K 線或漸層堆疊。

## 4. Logo usage

- Canonical 圖形來源維持 `static/brand/absorb-mark.png`，SHA-256 `2e7b3950809748d5e02648dfc26b0b403f7cabd2d706ce3130b28bad86c9443d`，供 favicon、社群預覽、LINE 與其他非導覽用途使用。
- Canonical 圖形維持純白背景、原始比例與安全邊距；不得裁切、重畫、改色、加字、陰影或漸層。使用圖形時 HTML alt 固定為「ABSORB logo」。
- 導覽文字標誌固定為小寫草寫 `absorb`，不得搭配圓形圖示，並連回主畫面。

## 5. Color system

Logo 實測主要深藍為 `#122643`。CSS token：

```css
--absorb-navy: #122643;
--absorb-navy-hover: #1b365d;
--absorb-navy-active: #0b1b31;
--absorb-navy-soft: #eaf0f7;
--absorb-white: #ffffff;
--absorb-canvas: #f7f9fc;
--absorb-surface: #ffffff;
--absorb-surface-raised: #fbfcfe;
--absorb-ink: #152033;
--absorb-muted: #586579;
--absorb-subtle: #7a8798;
--absorb-hairline: #d9e0e8;
--absorb-focus: #2b6cb0;
--absorb-success: #18704a;
--absorb-warning: #8a5b00;
--absorb-danger: #a33a44;
--absorb-info: #245b91;
```

文字與背景對比至少符合 WCAG AA。漲跌顏色必須同時搭配文字或符號。

## 6. Typography

英文字優先使用系統已安裝的 `Avenir Next` 或 `Avenir`，繁體中文搭配 `Noto Sans TC`，再依序退回 `PingFang TC`、`Microsoft JhengHei` 與 `sans-serif`。不下載、內嵌或提交專有字型，也不建立外部 font request。草寫只用於導覽文字標誌，內容標題維持人文無襯線。

## 7. Numeric typography

價格、機率、報酬、日期與表格數字使用 `font-variant-numeric: tabular-nums`。小數位由資料契約決定，不以補零假裝精度。

## 8. Spacing

基準 4px；常用 token 為 4、8、12、16、24、32、48px。內容卡內距以 16 或 24px 為主。

## 9. Radius

小元件 6px、按鈕與輸入 8px、卡片 10px。膠囊只用於短狀態標籤，不把所有容器做成大圓角。

## 10. Borders

一般邊界 1px `--absorb-hairline`；focus 不以 border 替代，使用可見的 2px outline。

## 11. Shadows

預設無陰影。需要區分浮層時只用 `0 8px 24px rgb(18 38 67 / 8%)`，不得套用於 Logo。

## 12. Surfaces

Canvas 使用淡中性，主要內容使用純白；raised surface 僅用於選單、dialog 或必要層級。

## 13. Cards

卡片只包一個決策單位，順序為標籤、結論、數據、限制。禁止把每段文字各包一張卡。

## 14. Buttons

Primary 為深藍底白字；secondary 為白底深藍邊界；danger 僅用於確實具破壞性的確認。disabled 不可只靠降低透明度。

## 15. Inputs

永遠有可見 label。錯誤文字緊鄰欄位並可由 screen reader 關聯。placeholder 不替代 label。

## 16. Tables

表頭固定語意，數字靠右，文字靠左；小螢幕優先水平捲動或轉為成組定義列，不藏核心欄位。

## 17. Navigation

桌面使用頂部導覽列與小寫草寫 `absorb` 文字標誌，目前頁面同時用文字與底線標示。市場切換必須導向實際存在且通過驗證的市場內容，不得只換標籤。行動版保留五個核心入口。

## 18. Charts

圖表只呈現可驗證資料。座標、資料日、圖例與缺值狀態必須可見；不可用面積、3D 或動畫誇大變化。

## 19. Data states

資料狀態固定為 available、partial、stale、unavailable。`None` 不轉為 0；stale 必須顯示資料日期且不得使用「現在」。

## 20. Empty states

說明缺少什麼、是否可稍後重試，以及仍可採取的安全操作；不得補 SAMPLE 或假資料。

## 21. Loading states

使用固定尺寸 skeleton 避免版面位移；超過合理等待時間顯示可理解的降級訊息。

## 22. Error states

對外只顯示安全錯誤與下一步，不顯示 stack trace、路徑、object name、token 或 provider 細節。

## 23. Mobile behavior

以 390px 為主要驗收寬度。觸控目標至少 44px，表單不造成水平溢出，結論、風險、失效條件與資料日期不可因篇幅被刪除。

## 24. Accessibility

支援鍵盤、skip link、語意標題、可見 focus、reduced motion、非顏色語意與有意義的 alt。裝飾圖使用空 alt。

## 25. Motion

只用 120–180ms 的狀態回饋；遵守 `prefers-reduced-motion`。資料更新不得以持續閃爍吸引注意。

## 26. Dark mode

目前不自動提供。未完成逐元件對比驗證前，不以反相或純黑加金色快速產生深色版。

## 27. Report layout

報告採研究文件層級：標題與資料日、執行摘要、證據、反對證據、限制、附錄。新報告 producer 顯示 ABSORB；歷史 immutable 產物不改寫。

## 28. LINE Flex visual rules

以深藍標題、白色 surface、清楚分隔線為主。每張卡只保留一個主 CTA；文字 fallback 必須含結論、風險、失效條件、資料日與限制。

## 29. Anti-patterns

禁止舊品牌、蝴蝶、父系／長輩人格、Logo watermark、圖片內嵌文字、暖米色主題、玻璃擬態、大面積純黑、金色裝飾、過量圓角、無來源即時數字、只靠顏色表示漲跌，以及把 Dashboard 改成促銷 Landing Page。
