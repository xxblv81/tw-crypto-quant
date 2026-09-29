"""
Robust optimisation study for the BTC 4H squeeze-breakout strategy.

    python optimize.py            # prints every table used in crypto/README.md

Every candidate change must improve BOTH symbols (BTC, ETH) in BOTH periods
(2018-23, 2024-26) before it is adopted. Picking the single best cell of a grid is
not allowed (project rule: structural changes judged on the whole distribution).

Sections
  A. Parameter plateau around the adopted exit (MA exit length)
  B. Anchored walk-forward: refit a 108-config grid every January on past data only,
     trade the next calendar year untouched, stitch the out-of-sample years together
  C. Random-entry null: same exits, random entry bars, similar trade count (500 draws)
  D. Volatility-targeted position sizing
  E. BTC + ETH equal-weight portfolio
  F. Stationary block bootstrap 90 % confidence interval of the Sharpe ratio
  G. Alternatives tested and not adopted
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import engine as E

V33 = {**E.BASE}                               # adopted v3.3 config from markets.yaml
V32 = {**E.BASE, "vol_target": 0}              # v3.2 = same rules, 100 % equity per trade
DATA = {s: E.load(s) for s in E.SYMBOLS}
BH = {s: E.buy_hold(DATA[s]) for s in E.SYMBOLS}
RNG = np.random.default_rng(7)


def period_stats(res: E.Result) -> dict:
    return {k: E.stats(res.slice(a, b)) for k, (a, b) in E.PERIODS.items()}


def row(label: str, per: dict) -> str:
    cells = []
    for k in E.PERIODS:
        st = per[k]
        cells.append(f"{st['sharpe']:5.2f} {st['cagr']*100:6.1f}% {st['mdd']*100:5.1f}% {st['trades']:4d}"
                     if st else "   n/a")
    return f"  {label:<28}" + "   |   ".join(cells)


def header(title: str) -> None:
    print(f"\n### {title}")
    print(f"  {'':<28}" + "   |   ".join(f"{k}: Sharpe  CAGR   MDD    n" for k in E.PERIODS))


# ---------------------------------------------------------------- 0. headline
def headline() -> None:
    print("\n### 0. Full period 2018-26 (fixed parameters from markets.yaml)")
    rows = [("BTC buy & hold", E.buy_hold(DATA["BTCUSDT"])),
            ("BTC v3.2 (100 % equity)", E.simulate(DATA["BTCUSDT"], V32)),
            ("BTC v3.3 (vol target 40%)", E.simulate(DATA["BTCUSDT"], V33)),
            ("ETH v3.3 (vol target 40%)", E.simulate(DATA["ETHUSDT"], V33))]
    for label, res in rows:
        st = E.stats(res)
        eq = (1 + res.daily()).cumprod()
        extra = f"  multiple {eq.iloc[-1]:4.1f}x"
        if len(res.trades):
            extra += f"  avg weight {res.trades['weight'].mean():.2f}  median hold {res.trades['bars'].median()*4:.0f}h"
        print(f"  {label:<28}{E.fmt(st)}{extra}")
    d = portfolio()
    eq = (1 + d).cumprod()
    print(f"  {'BTC+ETH 50/50 v3.3':<28}Sharpe {d.mean()/d.std()*(365**0.5):5.2f}  "
          f"CAGR {(eq.iloc[-1]**(365/len(d))-1)*100:6.1f}%  MDD {(1-eq/eq.cummax()).max()*100:5.1f}%  "
          f"multiple {eq.iloc[-1]:4.1f}x")


# ---------------------------------------------------------------- A. plateau
def plateau() -> None:
    for sym in E.SYMBOLS:
        header(f"A. MA-exit plateau — {sym}")
        print(row("buy & hold", period_stats(BH[sym])))
        print(row("old: fixed 2R take-profit", period_stats(
            E.simulate(DATA[sym], {**V32, "take_profit": True, "exit_ma": 0}))))
        for m in (10, 15, 20, 30, 50, 100):
            tag = "  <- adopted" if m == V32["exit_ma"] else ""
            print(row(f"MA{m} exit{tag}", period_stats(E.simulate(DATA[sym], {**V32, "exit_ma": m}))))


# ---------------------------------------------------------------- B. walk-forward
GRID = {"exit_ma": (10, 20, 30, 50), "volume_mult": (1.0, 1.5, 2.0),
        "squeeze_window": (10, 20, 30), "stop_atr": (1.5, 2.0, 3.0)}


def walk_forward(vol_target: float | None = 0) -> dict:
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    # Signals are causal (audit.py test 1), so one full-history run per config is
    # equivalent to re-running on each training window; we only slice by date.
    daily = {s: [E.simulate(DATA[s], {**V32, **c}, vol_target=vol_target).daily() for c in configs]
             for s in E.SYMBOLS}
    years = range(2020, 2027)
    picks, oos = [], {s: [] for s in E.SYMBOLS}
    for y in years:
        train_end = pd.Timestamp(f"{y - 1}-12-31").date()
        scores = []
        for k in range(len(configs)):
            sh = []
            for s in E.SYMBOLS:
                d = daily[s][k]
                d = d[d.index <= train_end]
                sh.append(d.mean() / d.std() * np.sqrt(365))
            scores.append(min(sh))          # must work on BOTH coins, not just the better one
        best = int(np.nanargmax(scores))
        picks.append({"year": y, **configs[best], "train_min_sharpe": scores[best]})
        for s in E.SYMBOLS:
            d = daily[s][best]
            oos[s].append(d[(d.index > train_end) & (d.index <= pd.Timestamp(f"{y}-12-31").date())])
    tag = f"vol target {vol_target:.0%}" if vol_target else "100 % equity"
    print(f"\n### B. Anchored walk-forward, {tag} (refit each January, objective = min(BTC, ETH) Sharpe)")
    print(pd.DataFrame(picks).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out = {}
    for s in E.SYMBOLS:
        wf = pd.concat(oos[s])
        fixed = E.simulate(DATA[s], V32, vol_target=vol_target).daily()
        fixed = fixed[(fixed.index >= wf.index[0]) & (fixed.index <= wf.index[-1])]
        bh = BH[s].daily()
        bh = bh[(bh.index >= wf.index[0]) & (bh.index <= wf.index[-1])]
        for name, d in (("walk-forward OOS", wf), ("fixed config", fixed), ("buy & hold", bh)):
            eq = (1 + d).cumprod()
            print(f"  {s} {name:<18} 2020-26  Sharpe {d.mean()/d.std()*np.sqrt(365):5.2f}  "
                  f"CAGR {(eq.iloc[-1]**(365/len(d))-1)*100:6.1f}%  MDD {(1-eq/eq.cummax()).max()*100:5.1f}%")
        out[s] = wf
    return out


# ---------------------------------------------------------------- C. random-entry null
def random_null(n_draws: int = 500) -> None:
    print("\n### C. Random-entry null (v3.2 exits, random entries, 2018-26)")
    for sym in E.SYMBOLS:
        df = DATA[sym]
        s = E.sig.compute(df, V32)
        real = E.stats(E.simulate(df, V32))
        n_trades = real["trades"]
        trend_ok = ((s["Close"] > s["trend_ma"]) & (s["Close"] > s["trend_ma2"])).to_numpy()
        orig = E.sig.compute
        results = {}
        for name, pool in (("any bar", np.arange(250, len(df) - 1)),
                           ("bars passing trend filter", np.flatnonzero(trend_ok[:-1]))):
            sh, nt = [], []
            for _ in range(n_draws):
                pick = np.zeros(len(df), dtype=bool)
                pick[RNG.choice(pool, size=min(len(pool), n_trades * 3), replace=False)] = True
                E.sig.compute = lambda d, p, _m=pick: s.assign(signal=_m)
                st = E.stats(E.simulate(df, V32))
                sh.append(st["sharpe"] if st else np.nan)
                nt.append(st["trades"] if st else 0)
            E.sig.compute = orig
            sh = np.array(sh)
            results[name] = sh
            pct = (sh < real["sharpe"]).mean() * 100
            print(f"  {sym} random entry ({name:<26}) median Sharpe {np.nanmedian(sh):5.2f}, "
                  f"95th pct {np.nanpercentile(sh, 95):5.2f}, avg {np.mean(nt):.0f} trades | strategy {real['sharpe']:5.2f} "
                  f"beats {pct:5.1f}% of draws")


# ---------------------------------------------------------------- D. vol targeting
def vol_targeting() -> None:
    for sym in E.SYMBOLS:
        header(f"D. Volatility-targeted sizing — {sym} (weight = min(1, target / 30-day realised vol))")
        print(row("v3.2, 100 % equity", period_stats(E.simulate(DATA[sym], V32))))
        for vt in (0.2, 0.3, 0.4, 0.6, 0.8):
            tag = "  <- adopted (v3.3)" if vt == V33["vol_target"] else ""
            print(row(f"vol target {int(vt*100)}%{tag}", period_stats(E.simulate(DATA[sym], V32, vol_target=vt))))


def vol_lookbacks() -> None:
    header("D2. Vol target 40%, realised-vol lookback (Sharpe only matters here)")
    for sym in E.SYMBOLS:
        for lb, days in ((60, 10), (180, 30), (360, 60)):
            print(row(f"{sym[:3]} lookback {days} days", period_stats(
                E.simulate(DATA[sym], V32, vol_target=0.4, vol_lookback=lb))))


# ---------------------------------------------------------------- G. rejected alternatives
def rejected() -> None:
    import binance_data as bd
    for sym in E.SYMBOLS:
        header(f"G. Alternatives not adopted — {sym} (100 % equity)")
        print(row("v3.2 baseline (4H)", period_stats(E.simulate(DATA[sym], V32))))
        print(row("breakout on candle high", period_stats(E.simulate(DATA[sym], {**V32, "break_on_high": True}))))
        print(row("volume_mult 1.0", period_stats(E.simulate(DATA[sym], {**V32, "volume_mult": 1.0}))))
        h1 = bd.load(sym, "1h", E.START, E.END, verbose=False)
        print(row("same rules on 1H", period_stats(E.simulate(h1, V32))))


# ---------------------------------------------------------------- E. portfolio
def portfolio(vol_target: float | None = None) -> pd.Series:
    d = pd.concat([E.simulate(DATA[s], V33, vol_target=vol_target).daily() for s in E.SYMBOLS], axis=1)
    return d.fillna(0).mean(axis=1)


def portfolio_table() -> None:
    header("E. BTC + ETH, 50/50, rebalanced daily")
    for label, vt in (("v3.2 BTC+ETH", 0), ("v3.3 BTC+ETH (vol target 40%)", None)):
        d = portfolio(vt)
        per = {}
        for k, (a, b) in E.PERIODS.items():
            dd = d[(d.index >= pd.Timestamp(a).date()) & (d.index <= pd.Timestamp(b).date())]
            eq = (1 + dd).cumprod()
            per[k] = {"sharpe": dd.mean() / dd.std() * np.sqrt(365), "cagr": eq.iloc[-1] ** (365 / len(dd)) - 1,
                      "mdd": (1 - eq / eq.cummax()).max(), "trades": 0}
        print(row(label, per))


# ---------------------------------------------------------------- F. bootstrap
def bootstrap_sharpe(d: pd.Series, n: int = 2000, block: int = 20) -> tuple[float, float]:
    x = d.to_numpy()
    m = len(x)
    out = np.empty(n)
    for k in range(n):
        idx = []
        while len(idx) < m:
            start = RNG.integers(m)
            length = RNG.geometric(1 / block)
            idx.extend((start + np.arange(length)) % m)
        s = x[np.array(idx[:m])]
        out[k] = s.mean() / s.std() * np.sqrt(365)
    return tuple(np.percentile(out, [5, 95]))


def bootstrap_table() -> None:
    print("\n### F. Sharpe 90% confidence interval (stationary block bootstrap, mean block 20 days, 2018-26)")
    for label, d in (("BTC v3.2", E.simulate(DATA["BTCUSDT"], V32).daily()),
                     ("BTC v3.3", E.simulate(DATA["BTCUSDT"], V33).daily()),
                     ("ETH v3.3", E.simulate(DATA["ETHUSDT"], V33).daily()),
                     ("BTC+ETH v3.3", portfolio()),
                     ("BTC buy & hold", BH["BTCUSDT"].daily())):
        lo, hi = bootstrap_sharpe(d)
        print(f"  {label:<16} Sharpe {d.mean()/d.std()*np.sqrt(365):5.2f}   90% CI [{lo:5.2f}, {hi:5.2f}]")


if __name__ == "__main__":
    print(f"Data: Binance spot 4H, {E.START} ~ last closed bar {DATA['BTCUSDT'].index[-1]:%Y-%m-%d %H:%M} UTC; "
          f"cost {E.COMMISSION*100:.2f}%/side")
    headline()
    plateau()
    walk_forward(0)
    walk_forward(0.4)
    random_null()
    vol_targeting()
    vol_lookbacks()
    portfolio_table()
    bootstrap_table()
    rejected()
