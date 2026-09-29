# 布林縮口突破策略 — 雙市場版

> **交接給 Claude Code：直接把整個資料夾丟進工作目錄即可，`CLAUDE.md` 會被自動讀取。**

同一套進場邏輯，**兩個市場用不同參數，檔案完全分離**。

## 結論

| 市場 | 狀態 | 週期 | 關鍵參數 | PF | 樣本 |
|---|---|---|---|---|---|
| 加密貨幣 BTC | **已驗證** | BTCUSDT 4H | `stop_atr=2.0`, `volume_filter=true` | 1.71 | 48 筆 |
| 加密貨幣 ETH | **不成立** | ETHUSDT 4H | 同上 | **0.93** | 52 筆 |
| 台股 全體 | **可用，有偏誤待修** | 日線 | `stop_atr=4.0`, `volume_filter=false` | 中位數 1.43 | 608 檔 |
| 台股 權值股 | **最佳** | 日線 | 同上 | **1.92** | 152 檔 |
| 台股 中小型 | 最差 | 日線 | 同上 | 1.15 | 152 檔 |
| 低位階籌碼清洗 | **不成立** | 日線 | 見 `taiwan_washout` | 1.33（但**超額 −0.37%**） | 476 筆 |

**2026-09-03 更新**：第 3 套策略（低位階籌碼清洗）回測完成，**rejected** ——
兩個定義性條件（低位階、未發酵）正好是拖累績效的那兩個（`taiwan/WASHOUT_RESULTS.md`）。
同時建立**通用判準：超額報酬（相對等權買進持有）**，取代單看 PF；這也解決了
CLAUDE.md P2 #6 留下的方法論缺口。

**2026-09-01 更新**：台股全市場回測完成，PF 隨規模單調遞增 —— 「中小型股更有效」
的假設被推翻（`taiwan/P0_RESULTS.md`）。加密貨幣的週期與參數掃描完成，
「已驗證」僅對 BTCUSDT 成立（`crypto/SWEEP_RESULTS.md`）。

## 目錄

```
.
├── markets.yaml            ★ 兩市場參數的單一事實來源，程式都讀這支
├── crypto/
│   ├── strategy.pine       加密貨幣策略（TradingView）
│   ├── PARAMS.md           參數依據與驗證數據
│   └── results.csv         加密貨幣回測數據
├── taiwan/
│   ├── strategy.pine       台股策略（TradingView）
│   ├── screener_pine.pine  Pine 選股器 — 上限 40 檔，快速看盤用
│   ├── PARAMS.md           參數依據與未驗證缺口
│   ├── TUNING_LOG.md       台股調參實驗全紀錄
│   ├── results.csv         台股回測數據
│   └── python/             ★ 全市場掃描（1000+ 檔，突破 Pine 的 40 檔上限）
│       ├── .venv/          Python 3.14 虛擬環境（已建好）
│       ├── config.py       讀 markets.yaml
│       ├── universe.py     抓上市完整清單 → universe.csv
│       ├── margin.py       融資融券餘額快取
│       ├── qfiis.py       外資及陸資持股比率 + 發行股數（存量籌碼）
│       ├── tdcc.py        集保持股分級（TDCC 官方）★ 只有最新一週，需每週累積
│       ├── pyramid.py     集保大戶週資料（神秘金字塔）★ 單檔一次請求給 170 週
│       ├── chip_rotation.py  大戶籌碼輪動選股（每週清單，未回測）
│       ├── fundamentals.py 月營收快取（MOPS 靜態檔，點時間處理）
│       ├── washout.py      低位階籌碼清洗評分引擎
│       ├── washout_screen.py    今日候選清單（--relaxed 跑消融版）
│       ├── washout_backtest.py  回測 + ★ 等權買進持有基準
│       ├── washout_sweep.py     門檻／分項／出場／門檻消融掃描
│       ├── twse.py         ★ 證交所官方 API：行情快取 + 除權息還原
│       ├── data.py         行情存取（twse / yfinance / FinMind 三後端）
│       ├── signals.py      訊號計算，與 Pine 邏輯逐行對應
│       ├── screen.py       今日選股 → signals_YYYY-MM-DD.csv
│       ├── backtest.py     全市場回測 → 依成交金額分組看中小型股是否更有效
│       └── cache/          行情快取（day/*.parquet + exright.csv），不進版控
└── archive/                歷史版本（含原始 bug 版）
```

## 為什麼台股要用 Python 而不是 Pine

Pine 的 `request.security()` **上限 40 檔**，掃不了 1000+ 檔的台股全市場。
`screener_pine.pine` 保留給「盯 20 檔口袋名單」的日常使用；要掃全市場、要做橫斷面統計，只能用 `taiwan/python/`。

## 快速開始

**加密貨幣**：`crypto/strategy.pine` 貼進 TradingView，套 BTCUSDT 4H。

**不想打指令**：桌面上的 `台股工具.command` 點兩下就有選單，跑完會自動用 Numbers 開結果。

選單分三組：**選股 1~6**（期貨動能／現股動能／啟動觸發／低位階／縮口突破 ×2）、
**回測 7~9**、**維護 10**（一次更新所有快取）。

★ 桌面那份是「轉呼叫」的兩行小殼，不是複本 —— 以前是複本，結果它停在只有 1~4 個
選項的舊版本沒人發現。現在改專案根目錄那份就會直接生效，不用再 `cp`。
桌面殼壞掉時重建：

```bash
printf '#!/bin/bash\nexec "/Users/august/Claude/Projects/投資/台股工具.command"\n' > ~/Desktop/台股工具.command && chmod +x ~/Desktop/台股工具.command
```
專案若搬家，把檔案裡最上面的 `PROJ` 路徑改掉即可。

**台股全市場選股**（環境已建好，直接用 venv 裡的 python）：
```bash
cd taiwan/python
./.venv/bin/python twse.py --status      # 先確認行情快取範圍
./.venv/bin/python screen.py --small-cap # 排除成交金額前 100 大，專掃中小型
```

**台股全市場回測**（已跑過，結果見 `taiwan/P0_RESULTS.md`）：
```bash
./.venv/bin/python backtest.py --buckets 4
```

**參數同步檢查**（改完 markets.yaml 或 Pine 就跑一次）：
```bash
./.venv/bin/python check_sync.py
```

**加密貨幣掃描**（Binance 資料，工具在 `crypto/python/`）：
```bash
cd crypto/python && ../../taiwan/python/.venv/bin/python sweep.py --mode timeframe
```

**更新行情快取**（每天收盤後跑一次，只補缺的日期）：
```bash
./.venv/bin/python twse.py --start 2019-01-01
```

## 資料源：證交所官方 API

不用 yfinance。行情來自 TWSE `MI_INDEX`（逐交易日抓全市場），
除權息還原因子來自 `TWT49U`（除權除息計算結果表）。免 token、免限流風險。

**除權息還原是必要的，不是選配**：實測 2019~2026 共 7,007 筆除權息事件，
factor 中位數 0.9632 —— 台股平均一次除息就吃掉 3.7% 價格。
不還原的話，除息當天的假跳空會在 `stop_atr=4.0` 的日線策略上觸發假停損。

**上櫃（TPEx）預設排除**，因為它沒有歷史除權息資料可還原，
混進來會讓「中小型 vs 權值股」的比較混入資料品質差異。理由詳見 CLAUDE.md。

## 改參數的規則

**只改 `markets.yaml`。** Python 端透過 `config.py` 讀取，改一處全部生效。
Pine 檔的 input 預設值需手動同步，每個參數的 tooltip 都寫了實測依據——改之前先看。

## 三個已排除的方向

有明確反面證據，不要重跑：

- **做空** — BTC/ETH 4H 空方大幅虧損（−3,400 vs 多方 +1,801），程式碼已移除
- **移動停利** — 砍掉贏家（PF 1.44→1.36）
- **日線 `bbw_lookback` > 60** — 結果飽和，60/70/80/90/100 數字完全相同

## 還沒做完的事（2026-09-01 更新）

已完成：中小型股驗證（結果相反）、除權息還原、漲停過濾、crypto 參數與週期掃描、
Python 端全部實跑除錯。

**仍未解決：**

1. **★ 倖存者偏差** — universe 只含目前在市的公司，下市者不在樣本內，會高估所有數字。
   要修需要 TWSE 的終止上市公司歷史清單。**在此之前台股的絕對數字都不能當定論。**
2. **回撤 35~40%** — 四個分組都是，即使表現最好的權值股組也有 35.2%。
3. **台股 `stop_atr=4.0` 仍未做敏感度掃描**（3/5/6 都沒試）。crypto 端已掃完。
4. **上櫃股完全未納入** — 無歷史除權息資料可還原，需 FinMind token。
5. **ETH 4H 虧損** — crypto 的「已驗證」只對 BTCUSDT 成立，需要更多標的才能判斷
   是 ETH 特例還是策略本身不普適。
