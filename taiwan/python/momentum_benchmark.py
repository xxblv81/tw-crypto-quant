"""
與大盤對照 —— 這套策略到底有沒有贏過「買進持有加權指數」。

為什麼一定要做：
    先前所有報告都只講絕對績效（CAGR 69.9%、PF 1.715）。但 2019~2026 台股大漲，
    **絕對賺錢不等於贏過大盤**。而且策略開 1.5~2.5x 槓桿，拿去比不加槓桿的
    指數是不公平的 —— 必須在同槓桿、或同風險下比。

三種對照，全部列出來讓人自己判斷：
    ① 同槓桿    策略 vs 指數，兩邊用一樣的倍數。回答「同樣的槓桿，選股有沒有加分」
    ② 同風險    把指數的槓桿調到與策略相同的最大回撤。回答「承擔同樣的痛，誰賺得多」
    ③ 曝險調整  策略平均只有約 50% 時間在場內，把閒置資金當現金（0% 報酬）已內含在
                權益曲線裡，所以①②本來就是公平的 —— 這裡只把曝險印出來當說明

用法：python momentum_benchmark.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import chips as ch
import config
import market
import momentum as mom
import momentum_backtest as mb
import twse


def curve_stats(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = float((1 - eq / eq.cummax()).max() * 100)
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    return {"總報酬%": round((eq.iloc[-1] - 1) * 100, 1), "CAGR%": round(cagr, 1),
            "最大回撤%": round(dd, 1), "Sharpe": round(sharpe, 2),
            "報酬/回撤": round(cagr / dd, 2) if dd > 0 else float("nan")}


def index_curve(ic: pd.Series, idx, lev: float) -> pd.Series:
    """買進持有加權指數，固定槓桿。逐日複利，貼合實際的槓桿部位。"""
    s = ic.reindex(idx).ffill()
    r = s.pct_change().fillna(0.0) * lev
    return (1 + r).cumprod()


def main() -> None:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    fp = config.params("taiwan_futures")
    rp = config.params("taiwan_market")
    COST = config.costs("taiwan_futures")["commission_pct_per_side"]

    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    idx = panel["Close"].index
    p = {**p0, "exit_ma": fp["exit_ma"], "max_hold_days": fp["max_hold_days"]}
    sc = mom.compute_scores(panel, p, ch.buy_streak(ch.load_trust_net()))
    msk = mom.liquidity_mask(panel["Turnover"], p0)
    pit = (panel["Turnover"].rolling(60, min_periods=60).mean()
           .rank(axis=1, ascending=False) <= fp["pit_universe_rank"])
    hi = panel["High"].rolling(250, min_periods=250).max()
    m = pit & (((hi - panel["Close"]) / panel["Close"] * 100) <= fp["near_high_pct"])

    brd = market.breadth(panel["Close"], msk, rp["breadth_ma"])
    lvl = market.regime(market.load_index(), brd, rp, idx)["level"]
    levmap = {int(k): float(v) for k, v in fp["regime_leverage"].items()}
    ic = market.load_index()

    tr = mb.extract_trades(panel, sc["score"], sc["ma"][fp["exit_ma"]], p,
                           p0["entry_score"], True, entry_mask=m)
    tr = tr.assign(ret_net=tr["ret_gross"] - 2 * COST / 100,
                   lev=[levmap[int(lvl.iloc[i])] for i in tr["entry_i"]])
    eq_s, port = mb.simulate_portfolio(tr, idx, fp["slots"], lev_col="lev")
    avg_lev = float(np.average([levmap[int(x)] for x in lvl], weights=None))

    print(f"期間 {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}"
          f"｜策略平均曝險 {port['avg_exposure_pct']:.0f}%"
          f"｜大盤狀態加權後的平均目標槓桿 {avg_lev:.2f}x\n")

    print("① 同槓桿對照")
    rows = []
    rows.append({"對象": "策略（弱0/中1.5/強2.5）", **curve_stats(eq_s)})
    for L in (1.0, 1.5, 2.0):
        rows.append({"對象": f"大盤買進持有 {L:g}x", **curve_stats(index_curve(ic, idx, L))})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n② 同風險對照（把大盤槓桿調到與策略相同的最大回撤）")
    target_dd = curve_stats(eq_s)["最大回撤%"]
    best = None
    for L in np.arange(0.2, 3.01, 0.05):
        st = curve_stats(index_curve(ic, idx, float(L)))
        d = abs(st["最大回撤%"] - target_dd)
        if best is None or d < best[0]:
            best = (d, float(L), st)
    print(f"  策略最大回撤 {target_dd:.1f}% ↔ 大盤需開 {best[1]:.2f}x 才有同樣回撤")
    print(pd.DataFrame([{"對象": "策略", **curve_stats(eq_s)},
                        {"對象": f"大盤 {best[1]:.2f}x", **best[2]}]).to_string(index=False))

    print("\n③ 分年對照（策略 vs 大盤 1.0x）")
    ix = index_curve(ic, idx, 1.0)
    out = []
    for y in sorted(set(idx.year)):
        mk = idx.year == y
        if mk.sum() < 50:
            continue
        s_, i_ = eq_s[mk], ix[mk]
        sr = (s_.iloc[-1] / s_.iloc[0] - 1) * 100
        ir = (i_.iloc[-1] / i_.iloc[0] - 1) * 100
        out.append({"年": y, "策略%": round(sr, 1), "大盤%": round(ir, 1),
                    "超額%": round(sr - ir, 1), "勝": "✓" if sr > ir else "✗",
                    "弱勢天數%": round(float((lvl[mk] == 0).mean() * 100)),
                    "策略回撤%": round(float((1 - s_ / s_.cummax()).max() * 100), 1)})
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    w = (df["策略%"] > df["大盤%"]).sum()
    print(f"\n  勝過大盤 {w}/{len(df)} 年　超額報酬中位數 {df['超額%'].median():+.1f}%")
    bad = df[df["策略%"] < 0]
    if len(bad):
        print(f"  ★ 策略虧損的年份：" + "、".join(
            f"{r['年']}（策略 {r['策略%']:+.1f}% vs 大盤 {r['大盤%']:+.1f}%）"
            for _, r in bad.iterrows()))


if __name__ == "__main__":
    main()
