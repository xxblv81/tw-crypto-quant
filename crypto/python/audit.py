"""
Backtest integrity audit for the BTC 4H squeeze-breakout strategy.

    python audit.py

1. Look-ahead test: signals computed on truncated history must equal signals computed on
   the full history for every overlapping bar. Any mismatch means a future-dependent calc.
2. Engine parity: engine.simulate vs taiwan/python/backtest.run_one (the original engine).
3. Gap-through-stop: how much the old "fill at stop price" assumption flattered results.
4. Cost sensitivity: commission + slippage per side.
5. Entry-delay sensitivity: entering one bar late should degrade, not collapse, the edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E
from backtest import run_one

P = {**E.BASE, "vol_target": 0}   # audit trade logic at 100 % equity (run_one has no sizing)


def lookahead(df: pd.DataFrame) -> int:
    full = E.sig.compute(df, P)
    bad = 0
    for cut in np.linspace(3000, len(df) - 1, 12).astype(int):
        part = E.sig.compute(df.iloc[:cut], P)
        cols = ["signal", "narrow", "upper", "bbw", "atr"]
        a, b = part[cols], full[cols].iloc[:cut]
        diff = ~((a == b) | (a.isna() & b.isna()))
        bad += int(diff.to_numpy().sum())
    return bad


def main() -> None:
    for sym in E.SYMBOLS:
        df = E.load(sym)
        print(f"\n===== {sym}  {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d %H:%M} UTC, {len(df):,} bars")

        n_bad = lookahead(df)
        print(f"[1] look-ahead: {n_bad} mismatching cells across 12 truncation points "
              f"-> {'PASS' if n_bad == 0 else 'FAIL'}")

        old = E.simulate(df, P, gap_fill=False)
        new = E.simulate(df, P)
        ro = run_one(df, P, E.COMMISSION * 100, limit_up_guard=False)
        print(f"[2] parity: run_one {ro['trades']} trades PF {ro['profit_factor']:.3f} | "
              f"engine(no gap fill) {len(old.trades)} trades PF {E.stats(old)['pf']:.3f}")

        t = new.trades
        gapped = t[(t["reason"] == "stop")]
        n_gap = int((old.trades["ret"] - new.trades["ret"]).abs().gt(1e-12).sum())
        print(f"[3] gap-through-stop: {n_gap} of {len(gapped)} stop exits filled below the stop; "
              f"Sharpe {E.stats(old)['sharpe']:.3f} -> {E.stats(new)['sharpe']:.3f}")

        print("[4] cost per side (commission+slippage) -> full-period stats")
        for c, s in [(0.0005, 0), (0.0005, 0.0005), (0.001, 0), (0.001, 0.0005), (0.002, 0)]:
            st = E.stats(E.simulate(df, P, commission=c, slippage=s))
            print(f"    commission {c*100:.2f}% + slippage {s*100:.2f}% per side  {E.fmt(st)}")

        s = E.sig.compute(df, P)
        late = df.copy()
        late_sig = s["signal"].shift(1, fill_value=False)
        # re-run the engine with the signal delayed one bar
        orig = E.sig.compute
        E.sig.compute = lambda d, p: s.assign(signal=late_sig)
        st_late = E.stats(E.simulate(late, P))
        E.sig.compute = orig
        print(f"[5] entry one bar late: {E.fmt(st_late)}")
        print(f"    baseline:           {E.fmt(E.stats(new))}")


if __name__ == "__main__":
    main()
