"""
動能評分引擎 —— 對標 CMoney「愛德恩動能選股」的動能分數。

★ 先說清楚這不是什麼：
  原始 App 的動能分數公式**沒有公開**（付費 App，官方文件只講操作步驟）。
  可查證的公開規則只有三條：
      1. 選股頁找「動能 > 7.5 分」的個股
      2. 看投信是否連買（建議 > 1 天）
      3. 沒動、或跌破五日線就停損換股（停損不超過 20%）
  下面是依這三條 + 「強者恆強」的順勢邏輯重建的透明版本。
  分數尺度刻意也做成 0~10、門檻也放 7.5，但**與原始分數不可互相對照**。

為什麼用「橫斷面百分位」而不是絕對門檻：
  絕對門檻（例如「近月漲幅 > 20%」）在多頭全市場通過、空頭全市場不通過，
  那是擇時不是選股。改成每天拿全市場互相排名後，7.5 分永遠等於
  「當天市場前 25% 的強度」，多空頭都能選出相對最強的一批。

分數組成（權重見 markets.yaml[taiwan_momentum].params.weights）：
  rs        相對強度 —— 20/60/120 日報酬加權後的全市場百分位
  volume    量能動能 —— 5 日均成交金額 / 60 日均成交金額 的全市場百分位
  align     均線多頭排列 —— 規則式 0~10，不做百分位（它本來就是絕對條件）
  position  位階 —— 距 52 週高點多遠，貼著高點 10 分、低於 30% 得 0 分
  chips     投信連買天數（選配，需 chips.py 的快取）

★ 流動性必須先濾再排名。不先濾，殭屍股會佔掉百分位分母，
  7.5 分就不再是「市場前 25%」。理由同 screen.py 的 --min-turnover。
★ 成交金額一律用官方 Turnover 欄，不用 Close×Volume ——
  還原後的 Close 會系統性低估歷史成交金額（見 CLAUDE.md）。
"""
from __future__ import annotations

import pandas as pd


# ─────────────────────────── 面板 ───────────────────────────
def build_panel(bars: dict[str, pd.DataFrame],
                fields=("Open", "High", "Low", "Close", "Turnover"),
                drop_etf: bool = True) -> dict[str, pd.DataFrame]:
    """
    {code: 個股 df} → {欄位: 日期 × 代號 的寬表}。
    橫斷面排名需要所有標的對齊到同一組日期，這步就是幹這個。
    drop_etf：00 開頭是 ETF／ETN，動能選股不看它們（回測端也一向排除）。
    """
    codes = sorted(c for c in bars if not (drop_etf and c.startswith("00")))
    return {f: pd.DataFrame({c: bars[c][f] for c in codes if f in bars[c]}).sort_index()
            for f in fields}


# ─────────────────────────── 各分項 ───────────────────────────
def _pct_score(raw: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """在 mask 為 True 的標的之間做橫斷面百分位 → 0~10 分。"""
    return raw.where(mask).rank(axis=1, pct=True) * 10


def liquidity_mask(turnover: pd.DataFrame, p: dict) -> pd.DataFrame:
    """N 日均成交金額 ≥ 門檻。這是所有排名的分母。"""
    avg = turnover.rolling(p["turnover_window"], min_periods=p["turnover_window"]).mean()
    return avg >= p["min_turnover"]


def rs_score(close: pd.DataFrame, mask: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    相對強度：多期報酬加權後排名。
    三段都要強才拿得到高分 —— 只靠一天暴衝的股票會在 60/120 日段落被拉下來。
    """
    raw = None
    for k, w in p["rs_lookbacks"].items():
        r = close / close.shift(int(k)) - 1
        raw = r * float(w) if raw is None else raw + r * float(w)
    return _pct_score(raw, mask)


def volume_score(turnover: pd.DataFrame, mask: pd.DataFrame, p: dict) -> pd.DataFrame:
    """量能動能：近期成交金額相對自身常態放大幾倍，再跨市場排名。"""
    fast = turnover.rolling(p["vol_fast"], min_periods=p["vol_fast"]).mean()
    slow = turnover.rolling(p["vol_slow"], min_periods=p["vol_slow"]).mean()
    return _pct_score(fast / slow.replace(0, pd.NA), mask)


def align_score(close: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    均線多頭排列。五個條件各 2 分：
        收盤>MA5、MA5>MA10、MA10>MA20、MA20>MA60、MA60 較 N 日前上揚
    這項刻意**不做百分位** —— 「站上 5 日線」本來就是絕對條件，
    排名化會讓空頭時全市場都不排列的日子裡，硬選出一批「相對比較排列」的股票。
    回傳 (分數, 各條均線)；均線後面出場規則還要用。
    """
    ma = {n: close.rolling(n, min_periods=n).mean() for n in p["ma_set"]}
    ns = sorted(p["ma_set"])
    slow = ma[ns[-1]]
    conds = [close > ma[ns[0]]]
    conds += [ma[a] > ma[b] for a, b in zip(ns, ns[1:])]
    conds.append(slow > slow.shift(p["ma_slope_lookback"]))

    score = sum(c.astype(float) for c in conds) * (10.0 / len(conds))
    # slow MA 還沒算出來的期間，分數是假的（比較全為 False）→ 標成 NaN
    return score.where(slow.notna()), ma


def position_score(close: pd.DataFrame, high: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    位階：收盤 / 52 週最高。貼著高點 10 分，跌到 position_floor 以下 0 分。
    動能策略買的是創新高附近的股票，不是抄底 —— 這項就是把抄底的排除掉。
    """
    hi = high.rolling(p["high_window"], min_periods=p["high_window"]).max()
    pos = close / hi
    floor = p["position_floor"]
    return ((pos - floor) / (1 - floor)).clip(0, 1) * 10


def chips_score(streak: pd.DataFrame, index, columns, p: dict) -> pd.DataFrame:
    """
    投信連買天數 → 0~10 分，連買 chips_max_days 天以上滿分。

    ★ 沒有籌碼資料的個股當天得 0 分（＝投信沒動作），這是正確語意。
      但如果**整天**都沒資料（快取落後於行情），全市場就會一起拿 0，
      總分上限被壓到 (1-chips_weight)×10 = 8.5，門檻 8.5 會篩出零檔。
      這個失效是靜默的，所以呼叫端必須先用 chips.last_date() 檢查覆蓋範圍 ——
      momentum_screen.py / momentum_backtest.py 都已經擋在前面。
    """
    s = streak.reindex(index=index, columns=columns)
    return (s.clip(0, p["chips_max_days"]) / p["chips_max_days"] * 10).fillna(0.0)


# ─────────────────────────── 總分 ───────────────────────────
def compute_scores(panel: dict[str, pd.DataFrame], p: dict,
                   chips_streak: pd.DataFrame | None = None
                   ) -> dict[str, pd.DataFrame]:
    """
    回傳 {"score": 總分, "rs": .., "volume": .., "align": .., "position": ..,
          "chips": ..(選配), "mask": 流動性遮罩, "ma": {n: 均線}}
    分數為 0~10，未過流動性門檻者為 NaN。
    """
    close, high, turnover = panel["Close"], panel["High"], panel["Turnover"]
    mask = liquidity_mask(turnover, p)

    parts = {
        "rs": rs_score(close, mask, p),
        "volume": volume_score(turnover, mask, p),
        "position": position_score(close, high, p),
    }
    parts["align"], ma = align_score(close, p)

    w = dict(p["weights"])
    if p.get("use_chips") and chips_streak is not None:
        # 開啟籌碼後，其餘四項等比例縮小，總權重維持 1.0
        cw = float(p["chips_weight"])
        w = {k: v * (1 - cw) for k, v in w.items()}
        parts["chips"] = chips_score(chips_streak, close.index, close.columns, p)
        w["chips"] = cw

    total = sum(parts[k] * w[k] for k in w)
    out = {"score": total.where(mask), "mask": mask, "ma": ma, "weights": w}
    out.update(parts)
    return out


def normalize_params(p: dict) -> dict:
    """YAML 讀進來的 key 可能是字串，統一轉型；順便檢查權重加總。"""
    p = dict(p)
    p["rs_lookbacks"] = {int(k): float(v) for k, v in p["rs_lookbacks"].items()}
    p["ma_set"] = [int(n) for n in p["ma_set"]]
    for name, d in (("weights", p["weights"]), ("rs_lookbacks", p["rs_lookbacks"])):
        s = sum(d.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"markets.yaml[taiwan_momentum].params.{name} 加總 {s}，應為 1.0")
    return p
