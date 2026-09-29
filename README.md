# tw-crypto-quant

台股與加密貨幣的量化策略研究。資料全部來自**證交所官方 API** 與 Binance 公開 API，
回測含除權息還原、交易成本與漲停買不到的處理。

> ⚠ **這不是投資建議。** 下面所有回測數字都是**樣本內**結果（只有動能選股做過樣本外驗證），
> 台股布林策略的樣本還有倖存者偏差。策略訊號的定位是「選股雷達」，實際買不買由人判斷。

## 策略一覽

| 策略 | 市場 | 狀態 | 代表性結果 | 詳細報告 |
|---|---|---|---|---|
| **布林通道縮口突破** | BTCUSDT 4H | ✅ 已驗證（僅 BTC） | PF 1.71（TradingView）／1.80（Python） | [`crypto/SHARPE_RESULTS.md`](crypto/SHARPE_RESULTS.md) |
| ↳ 同參數套 ETH | ETHUSDT 4H | ❌ 不成立 | PF 0.93，回撤 36% | [`crypto/SWEEP_RESULTS.md`](crypto/SWEEP_RESULTS.md) |
| **布林通道縮口突破** | 台股日線 | 🟡 可用，有偏誤待修 | 全市場 PF 約 1.4～1.5 | [`taiwan/P0_RESULTS.md`](taiwan/P0_RESULTS.md) |
| **動能選股** | 台股現股／個股期 | ✅ 樣本外通過 | PF 1.72 → 1.71（樣本內 → 樣本外） | [`taiwan/MOMENTUM_RESULTS.md`](taiwan/MOMENTUM_RESULTS.md) |
| **大戶籌碼輪動** | 台股週線 | ⚪ 未回測 | 每週清單，只當雷達 | [`taiwan/MOMENTUM_HOLDERS.md`](taiwan/MOMENTUM_HOLDERS.md) |
| 低位階籌碼清洗 | 台股 | ❌ rejected | PF 1.33 但**輸給大盤** 0.37%／筆 | [`taiwan/WASHOUT_RESULTS.md`](taiwan/WASHOUT_RESULTS.md) |
| 崩盤反彈 | 台股個股期 | ⏸ 擱置 | 只有 6 次事件，無法驗證 | [`taiwan/REBOUND_RESULTS.md`](taiwan/REBOUND_RESULTS.md) |
| BTC 跨所價差套利 | 9 家交易所 | ❌ rejected | 散戶費率下 2,112 組全數淨虧 | [`crypto/ARB_RESULTS.md`](crypto/ARB_RESULTS.md) |

失敗的策略也留著報告 —— 它們回答了「為什麼不做」，避免之後重跑同樣的東西。

## 布林通道縮口突破

同一套進場邏輯，兩個市場用不同參數（全部在 [`markets.yaml`](markets.yaml)）。

### 台股：兩段式進場

```
① 縮口：布林帶寬觸及近 20 根的最低點
② 收盤向上穿越中線 → 進入候選
③ 候選期內出現收紅 K，且量 > 5 日均量 × 1.2、也 > 穿越以來均量 × 1.2 → 進場
   候選期間收盤跌破中線就作廢
```

過濾：站上 MA20、MA20 > MA60 > MA120 多頭排列、20 日均成交金額 ≥ 5,000 萬、
排除金融／電信／生技醫療／文化創意類股。停損 4 ATR。每天最多輸出 20 檔。

**最重要的發現：PF 隨股票規模單調遞增。** 原本假設中小型股更有效，結果相反：

| 依成交金額分四組 | 中位數 PF | PF > 1 的比例 | 中位數報酬 |
|---|---|---|---|
| Q1 最小 | 1.15 | 61% | −1.2% |
| Q2 | 1.28 | 68% | +4.9% |
| Q3 | 1.31 | 67% | +7.8% |
| **Q4 權值股** | **1.92** | **86%** | **+62.1%** |

### BTC：4H

縮口後收盤突破上軌、站上 MA200 與 MA60、量 > 20 根均量 × 1.5 進場；
停損 2 ATR，**收盤跌破 MA20 出場**（2026-09 從固定 2R 停利改過來，
Sharpe 在 2018～23 由 0.03 升到 0.88、2024～26 由 0.89 升到 1.66）。

## 動能選股

橫斷面動能評分（相對強弱 60／120／240 日、量能放大、均線排列、距 52 週高點的位階，再加投信連買），
搭配**市場寬度**（全市場站上自身 MA60 的比例）動態調整槓桿，寬度 < 40% 時停手。

| 對象（2019～2026） | CAGR | 最大回撤 | Sharpe |
|---|---|---|---|
| 策略（個股期、動態槓桿） | 70.8% | 30.5% | 1.87 |
| 大盤買進持有 | 22.9% | 31.6% | 1.17 |

**失敗年要一起看**：2024 年大盤 +29%、策略 −7.5%。7～8 月急殺時寬度訊號反應太慢，
用 2.5 倍槓桿進場被套。寬度防得了慢性走弱，防不了兩週內的暴跌。

## 研究方法

這個專案最花力氣的不是找訊號，是避免騙自己：

- **看超額報酬，不只看 PF。** 每筆交易都對照同期「流動性池等權買進持有」。
  一個 PF 2.1 的配置，扣掉市場漂移後超額只剩 +2.7%，75% 的交易輸給大盤。
- **看整體分布的中位數，不做個股最佳化。** 在中位數 0.72 的分布裡挑出前三名是 data mining。
- **除權息還原是必要的。** 2019～2026 共 7,007 筆除權息事件，因子中位數 0.963 ——
  不還原的話，除息缺口會在日線策略上觸發假停損。
- **樣本外紀錄每天存檔、存了就不改。** `taiwan/python/cache/bb_picks/`（布林每日選股）
  與 `cache/picks/`（大戶週選股）是唯一不受事後調參污染的驗證資料。
- **有反面證據的方向明確列出、不重跑**：做空、移動停利、台股短線（持有 4～14 天在台股
  0.585% 來回成本下獨立驗證了四次都虧）…完整清單見 [`CLAUDE.md`](CLAUDE.md)。

### 已知的限制

1. **倖存者偏差**：布林策略的台股樣本只含目前仍上市的公司。動能選股的回測有納入下市股，沒有這個問題。
2. **回撤大**：布林台股 35～40%，動能 30～50%（依槓桿）。
3. **上櫃股缺除權息歷史**：TPEx 沒有公開的歷史除權息資料，回測只用上市股；選股時可納入上櫃。
4. **大部分結論是樣本內**：只有動能選股做過 2019～23 挑參數、2024～26 只驗不調。
   而那段樣本外剛好三年都是多頭。

## 快速開始

需要 Python 3.11 以上。

```bash
git clone https://github.com/xxblv81/tw-crypto-quant.git
cd tw-crypto-quant/taiwan/python
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

**建立行情快取**（第一次約 1.5～2 小時，之後每天只補新的日期，幾秒鐘）：

```bash
./.venv/bin/python twse.py --start 2019-01-01 --include-otc
./.venv/bin/python twse.py --exright --start 2019-01-01
```

對證交所的請求間隔固定 3 秒，不要調低，否則會被暫時封鎖 IP。

**每日選股**：

```bash
./.venv/bin/python screen.py --include-otc          # 布林縮口，附前幾天的追蹤
./.venv/bin/python chips.py --start 2019-01-01      # 動能計分要用的法人買賣超
./.venv/bin/python momentum_screen.py --futures      # 動能選股（個股期）
```

**macOS 選單**：專案根目錄的 `台股工具.command` 點兩下就有選單，打開時會自動更新行情。
第一次用要把檔案最上面的 `PROJ` 改成你的專案路徑。

**回測**：

```bash
./.venv/bin/python backtest.py --buckets 4           # 布林縮口，全市場依規模分四組
./.venv/bin/python momentum_backtest.py --futures    # 動能選股
```

**改參數**只改 `markets.yaml`，之後跑 `check_sync.py` 確認 TradingView 的 Pine 腳本有沒有跟上。

## 目錄

```
markets.yaml           所有策略參數的唯一來源
CLAUDE.md              完整研究紀錄：每個決策的數據依據、踩過的坑、排除的方向
台股工具.command        macOS 雙擊選單
taiwan/
  strategy.pine        布林策略（TradingView）
  *_RESULTS.md 等       各策略的研究報告
  python/              全市場掃描、回測、資料層
    twse.py            證交所行情＋除權息還原
    signals.py         布林訊號（與 Pine 逐行對應）
    screen.py          布林每日選股
    bb_review.py       布林選股的逐檔後續追蹤
    momentum_*.py      動能選股
    cache/             資料快取（不進版控，除了 bb_picks/ 與 picks/）
crypto/
  strategy.pine        BTC 布林策略（TradingView）
  python/              Binance 資料＋參數掃描
  arb/                 跨所套利研究（rejected）
```

## 資料來源

| 資料 | 來源 |
|---|---|
| 台股日行情 | 證交所 `MI_INDEX`（逐交易日全市場） |
| 除權息還原因子 | 證交所 `TWT49U` |
| 三大法人買賣超 | 證交所 `T86` |
| 融資融券 | 證交所 `MI_MARGN`、櫃買中心 |
| 集保大戶持股 | 集保結算所 opendata、神秘金字塔 |
| 個股期貨清單 | 期交所 OpenAPI |
| 加密貨幣 K 線 | Binance 公開 API |

全部免費、不需要 token。
