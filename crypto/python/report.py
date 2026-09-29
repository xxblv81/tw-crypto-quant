"""
Render crypto/equity.png for crypto/README.md.

    python report.py

Top: growth of $1 (log scale). Bottom: drawdown. Same x-axis, separate y-axes
(two panels, never a dual axis on one panel).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

import engine as E               # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
OUT = Path(__file__).resolve().parents[1] / "equity.png"


def daily_index(d: pd.Series) -> pd.Series:
    d.index = pd.to_datetime(d.index)
    return d


def main() -> None:
    data = {s: E.load(s) for s in E.SYMBOLS}
    btc = daily_index(E.simulate(data["BTCUSDT"], E.BASE).daily())
    eth = daily_index(E.simulate(data["ETHUSDT"], E.BASE).daily())
    port = pd.concat([btc, eth], axis=1).fillna(0).mean(axis=1)
    bh = daily_index(E.buy_hold(data["BTCUSDT"]).daily())
    series = [("BTC buy & hold", bh, GRAY, 1.5),
              ("BTC+ETH v3.3", port, ORANGE, 2.0),
              ("BTC v3.3", btc, BLUE, 2.0)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)

    ends = []
    for label, d, color, lw in series:
        eq = (1 + d).cumprod()
        dd = eq / eq.cummax() - 1
        ax1.plot(eq.index, eq.values, color=color, linewidth=lw, label=label)
        ax2.plot(dd.index, dd.values * 100, color=color, linewidth=lw * 0.75)
        ends.append((eq.iloc[-1], label, eq.index[-1]))
    # direct end labels, dodged vertically (in points) so equal endings don't collide
    ends.sort()
    for k, (v, label, t) in enumerate(ends):
        ax1.annotate(f"{label}  {v:.1f}x", (t, v), xytext=(6, (k - (len(ends) - 1) / 2) * 13),
                     textcoords="offset points", va="center", color=INK, fontsize=9)

    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)", color=INK2)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.set_ylabel("Drawdown (%)", color=INK2)
    ax1.legend(loc="upper left", frameon=False, labelcolor=INK)
    ax1.set_title("Bollinger squeeze breakout, Binance 4H, "
                  f"{btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d} (backtest, 0.05%/side cost)",
                  color=INK, loc="left", fontsize=11)
    ax2.axvline(pd.Timestamp("2024-01-01"), color=INK2, linewidth=0.8, linestyle=":")
    ax2.text(pd.Timestamp("2024-01-15"), ax2.get_ylim()[0] * 0.92, "2024-26 period", color=INK2, fontsize=8)
    fig.subplots_adjust(right=0.8, hspace=0.08, left=0.08, top=0.93, bottom=0.06)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
