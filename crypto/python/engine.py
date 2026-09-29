"""
Mark-to-market backtest engine for the crypto Bollinger squeeze-breakout strategy.

Signals come from `taiwan/python/signals.py` (single source of truth shared with the
Taiwan equity version); parameters come from `markets.yaml` via `config.py`.

Execution model (kept deliberately conservative):
  * Signal is evaluated on a closed bar i; the order fills at the OPEN of bar i+1.
  * Stop-loss = entry - stop_atr * ATR(i). If a bar gaps through the stop, the fill is
    the bar's open, not the stop price (the old engine filled at the stop — optimistic).
  * If stop and target are both touched inside one bar, the stop is assumed first.
  * MA exit: close below MA(exit_ma) -> exit at the NEXT bar's open
    (matches TradingView's strategy.close with default order processing).
  * Costs are charged per side: commission + slippage, as fractions of notional.
  * Returns are marked to market on every bar, then compounded to daily returns,
    so Sharpe reflects idle days (exposure is typically ~15%).
"""
from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "taiwan" / "python"), str(Path(__file__).resolve().parent)]
import binance_data as bd  # noqa: E402
import config              # noqa: E402
import signals as sig      # noqa: E402

BASE = config.params("crypto")
COMMISSION = config.costs("crypto")["commission_pct_per_side"] / 100
START, END = dt.date(2018, 1, 1), dt.date(2026, 9, 30)
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def load(symbol: str, interval: str = "4h") -> pd.DataFrame:
    return bd.load(symbol, interval, START, END, verbose=False)


@dataclass
class Result:
    bar_ret: pd.Series      # per-bar strategy return (already weighted and net of costs)
    trades: pd.DataFrame    # one row per trade

    def daily(self) -> pd.Series:
        r = self.bar_ret
        return (1 + r).groupby(r.index.date).prod() - 1

    def slice(self, a: str, b: str) -> "Result":
        return Result(self.bar_ret.loc[a:b], self.trades.loc[a:b])


def simulate(df: pd.DataFrame, p: dict, commission: float = COMMISSION,
             slippage: float = 0.0, vol_target: float | None = None,
             vol_lookback: int = 180, gap_fill: bool = True) -> Result:
    """
    vol_target: annualised volatility target per position (e.g. 0.4). Position weight
        = min(1, vol_target / realised_vol), realised vol from the `vol_lookback` bars
        ending at the signal bar (known when the order is placed). None = use
        markets.yaml (`vol_target`, `vol_lookback`); 0 = always 100 % equity.
    gap_fill: False reproduces the old optimistic stop fill (used only by audit.py).
    """
    if vol_target is None:                  # default: take sizing from markets.yaml
        vol_target = p.get("vol_target") or None
        vol_lookback = int(p.get("vol_lookback", vol_lookback))
    s = sig.compute(df, p)
    o, h, l, c = (s[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    signal, atr = s["signal"].to_numpy(), s["atr"].to_numpy()
    n = len(s)
    exit_ma = s["Close"].rolling(int(p["exit_ma"])).mean().to_numpy() if p.get("exit_ma") else None
    bars_per_year = 6 * 365
    rvol = (s["Close"].pct_change().rolling(vol_lookback).std() * np.sqrt(bars_per_year)).to_numpy()
    side_cost = commission + slippage

    r = np.zeros(n)
    rows = []
    i = 0
    while i < n - 1:
        if not signal[i] or not np.isfinite(atr[i]):
            i += 1
            continue
        w = 1.0
        if vol_target:
            if not np.isfinite(rvol[i]) or rvol[i] <= 0:
                i += 1
                continue
            w = min(1.0, vol_target / rvol[i])
        entry, a = o[i + 1], atr[i]
        stop = entry - p["stop_atr"] * a
        target = entry + p["stop_atr"] * a * p["rr"] if p["take_profit"] else np.inf
        j, px, reason = i + 1, None, None
        while j < n:
            if l[j] <= stop:
                px = min(stop, o[j]) if (gap_fill and j > i + 1) else stop
                reason = "stop"
                break
            if h[j] >= target:
                px = max(target, o[j]) if (gap_fill and j > i + 1) else target
                reason = "target"
                break
            if exit_ma is not None and c[j] < exit_ma[j]:
                if p.get("exit_next_open") and j + 1 < n:
                    j += 1
                    px = o[j]
                else:
                    px = c[j]
                reason = "ma"
                break
            j += 1
        if px is None:                      # still open at the end of the data
            j, px, reason = n - 1, c[n - 1], "open"
        for k in range(i + 1, j + 1):
            prev = entry if k == i + 1 else c[k - 1]
            cur = px if k == j else c[k]
            r[k] += w * (cur / prev - 1)
        r[i + 1] -= w * side_cost
        r[j] -= w * side_cost
        rows.append({"entry_time": s.index[i + 1], "exit_time": s.index[j], "entry": entry,
                     "exit": px, "weight": w, "bars": j - (i + 1),   # bars held, entry open -> exit
                     "ret": w * (px / entry - 1 - 2 * side_cost), "reason": reason})
        i = j + 1
    trades = pd.DataFrame(rows)
    if len(trades):
        trades = trades.set_index("entry_time")
    return Result(pd.Series(r, index=s.index), trades)


def stats(res: Result) -> dict:
    d = res.daily()
    if len(d) < 2 or d.std() == 0:
        return {}
    eq = (1 + d).cumprod()
    t = res.trades["ret"].to_numpy() if len(res.trades) else np.array([])
    gp, gl = t[t > 0].sum(), -t[t < 0].sum()
    downside = np.sqrt((np.minimum(d, 0) ** 2).mean())
    return {
        "sharpe": d.mean() / d.std() * np.sqrt(365),
        "sortino": d.mean() / downside * np.sqrt(365) if downside > 0 else np.nan,
        "cagr": eq.iloc[-1] ** (365 / len(d)) - 1,
        "mdd": (1 - eq / eq.cummax()).max(),
        "trades": len(t),
        "pf": gp / gl if gl > 0 else np.inf,
        "win": (t > 0).mean() if len(t) else np.nan,
        "exposure": (res.bar_ret != 0).mean(),
    }


def buy_hold(df: pd.DataFrame) -> Result:
    r = df["Close"].pct_change().fillna(0)
    return Result(r, pd.DataFrame())


PERIODS = {"2018-23": ("2018-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2026-09-30")}


def fmt(st: dict) -> str:
    if not st:
        return "no trades"
    return (f"Sharpe {st['sharpe']:5.2f}  CAGR {st['cagr']*100:6.1f}%  MDD {st['mdd']*100:5.1f}%  "
            f"PF {st['pf']:5.2f}  n {st['trades']:4d}  expo {st['exposure']*100:4.0f}%")
