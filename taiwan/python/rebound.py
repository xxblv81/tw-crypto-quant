"""
崩盤反彈策略 —— 大盤負乖離觸發，買最超跌的個股期標的。

邏輯（參數見 markets.yaml[taiwan_rebound]）：
    觸發   加權指數 BIAS20 ≤ index_trigger（預設 −6%）
    選股   point-in-time 流動性前 150 大中，個股 BIAS20 ≤ stock_max_bias（−10%），
           取最超跌的 picks_per_day 檔
    進場   隔日開盤
    出場   time：滿 hold_days 那天收盤出　／　revert：收盤站回 MA20 → 隔日開盤出（或滿期）
           兩者都另有 −stop_loss_pct 盤中硬停損

★ 為什麼觸發條件看「大盤」而不是個股：研究顯示乖離本身沒有選股能力 ——
  大盤平靜時買最超跌的股票是虧的（−0.69%／10 天），只有大盤本身超跌時才有效。
  見 markets.yaml[taiwan_rebound] 的註解與 taiwan/REBOUND_RESULTS.md。

★ 已知限制：崩盤當天個股常鎖跌停，停損價在盤中未必賣得掉。本模擬假設停損價成交，
  真實結果會比模擬差一點。進場不受影響（跌停價買得到，賣方多）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signals(panel: dict[str, pd.DataFrame], index_close: pd.Series, p: dict,
            turnover_ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    回傳 (pick, score, index_bias, ma)
        pick   日期 × 代號 布林：當天是否被選為反彈標的
        score  最超跌者分數高（= −個股乖離），投組模擬用來排優先順序
    """
    C = panel["Close"]
    idx = C.index
    ma = C.rolling(p["stock_bias_ma"], min_periods=p["stock_bias_ma"]).mean()
    sbias = (C - ma) / ma * 100

    ic = index_close.reindex(idx).ffill()
    ibias = (ic / ic.rolling(p["index_bias_ma"], min_periods=p["index_bias_ma"]).mean() - 1) * 100

    to60 = panel["Turnover"].rolling(60, min_periods=60).mean()
    pit = to60.rank(axis=1, ascending=False) <= p["pit_universe_rank"]
    elig = pit & turnover_ok & (sbias <= p["stock_max_bias"])
    elig = elig & (ibias <= p["index_trigger"]).to_numpy()[:, None]

    # 每天取最超跌的 K 檔
    rank = sbias.where(elig).rank(axis=1, method="first")          # 1 = 最超跌
    pick = (rank <= p["picks_per_day"]).fillna(False)
    return pick, -sbias, ibias, ma


def extract_trades(panel: dict[str, pd.DataFrame], pick: pd.DataFrame, score: pd.DataFrame,
                   ma: pd.DataFrame, p: dict) -> pd.DataFrame:
    """逐檔掃出交易。同一檔持有中不重複進場（出場後才看下一個訊號）。"""
    idx = panel["Close"].index
    H = int(p["hold_days"])
    stop_frac = 1 - p["stop_loss_pct"] / 100.0
    revert = p.get("exit_mode", "time") == "revert"
    rows = []
    for code in pick.columns:
        pk = pick[code].to_numpy(dtype=bool)
        if not pk.any():
            continue
        o = panel["Open"][code].to_numpy(dtype=float)
        l = panel["Low"][code].to_numpy(dtype=float)
        c = panel["Close"][code].to_numpy(dtype=float)
        m = ma[code].to_numpy(dtype=float)
        sc = score[code].to_numpy(dtype=float)
        n = len(c)
        i = 0
        while i < n - 1:
            if not pk[i] or not (o[i + 1] > 0):
                i += 1
                continue
            entry = o[i + 1]
            stop = entry * stop_frac
            last = min(i + H, n - 1)
            exit_px, j = None, i + 1
            while j <= last:
                if np.isfinite(l[j]) and l[j] <= stop:
                    exit_px = stop
                    break
                if revert and j > i + 1 and np.isfinite(m[j]) and c[j] >= m[j] \
                        and j + 1 < n and o[j + 1] > 0:
                    exit_px, j = o[j + 1], j + 1
                    break
                if j == last:
                    exit_px = c[j]
                    break
                j += 1
            if exit_px is None or not (exit_px > 0):
                i += 1
                continue
            rows.append({"code": code, "entry_i": i + 1, "exit_i": j,
                         "entry_date": idx[i + 1], "exit_date": idx[j],
                         "score": sc[i], "ret_gross": exit_px / entry - 1,
                         "hold_days": j - (i + 1)})
            i = j + 1
    return pd.DataFrame(rows)
