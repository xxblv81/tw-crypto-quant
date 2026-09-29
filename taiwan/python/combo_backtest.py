"""
動能 × 崩盤反彈 合併回測 —— 兩個策略共用同一筆權益。

為什麼要合在一起測、而不是各自測完相加：
    兩者設計上互補（動能在大盤中／強出手，反彈在大盤崩盤時出手），但「互補」要驗證。
    弱勢時動能不進新倉，可是手上舊部位還沒出場 —— 那段時間兩邊會同時持倉，
    總槓桿會疊上去。各自測完相加看不到這件事。

模擬：
    各 sleeve 有自己的部位上限與槓桿（動能用大盤狀態動態槓桿，反彈用固定槓桿）。
    每筆部位權重 = 槓桿 ÷ 該 sleeve 部位上限，出場時 equity *= 1 + 報酬 × 權重（同 momentum_backtest）。
    另外逐日記錄「總槓桿」＝所有持倉權重加總，回報最大值與平均值。

★ 判準依 CLAUDE.md：先看超額報酬（相對加權指數），再看 PF。
★ 同 momentum_backtest：未模擬追繳斷頭，總槓桿高的日子實際風險比表上大。

用法：
    python combo_backtest.py
    python combo_backtest.py --rebound-lev 1.0 --rebound-slots 5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import chips as ch
import config
import market
import momentum as mom
import momentum_backtest as mb
import rebound as rb
import twse
from momentum_benchmark import curve_stats, index_curve


def load():
    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    return panel, ch.buy_streak(ch.load_trust_net()), market.load_index()


def momentum_trades(panel, streak, ic) -> pd.DataFrame:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    fp = config.params("taiwan_futures")
    rp = config.params("taiwan_market")
    cost = config.costs("taiwan_futures")["commission_pct_per_side"]
    idx = panel["Close"].index
    p = {**p0, "exit_ma": fp["exit_ma"], "max_hold_days": fp["max_hold_days"]}
    sc = mom.compute_scores(panel, p, streak)
    msk = mom.liquidity_mask(panel["Turnover"], p0)
    pit = (panel["Turnover"].rolling(60, min_periods=60).mean()
           .rank(axis=1, ascending=False) <= fp["pit_universe_rank"])
    hi = panel["High"].rolling(250, min_periods=250).max()
    m = pit & (((hi - panel["Close"]) / panel["Close"] * 100) <= fp["near_high_pct"])
    lvl = market.regime(ic, market.breadth(panel["Close"], msk, rp["breadth_ma"]), rp, idx)["level"]
    levmap = {int(k): float(v) for k, v in fp["regime_leverage"].items()}
    tr = mb.extract_trades(panel, sc["score"], sc["ma"][fp["exit_ma"]], p,
                           p0["entry_score"], True, entry_mask=m)
    return tr.assign(ret_net=tr["ret_gross"] - 2 * cost / 100,
                     lev=[levmap[int(lvl.iloc[i])] for i in tr["entry_i"]],
                     regime=[int(lvl.iloc[i]) for i in tr["entry_i"]])


def rebound_trades(panel, ic, over: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    p = {**config.params("taiwan_rebound"), **(over or {})}
    cost = config.costs("taiwan_rebound")["commission_pct_per_side"]
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    ok = mom.liquidity_mask(panel["Turnover"], p0)
    pick, score, ibias, ma = rb.signals(panel, ic, p, ok)
    tr = rb.extract_trades(panel, pick, score, ma, p)
    if tr.empty:
        return tr, ibias
    return tr.assign(ret_net=tr["ret_gross"] - 2 * cost / 100, lev=float(p["leverage"])), ibias


def simulate(sleeves: list[dict], idx: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """sleeves: [{"name", "trades", "slots"}]，trades 需有 entry_i/exit_i/ret_net/score/lev。"""
    books = []
    for s in sleeves:
        t = s["trades"]
        books.append({"name": s["name"], "slots": s["slots"],
                      "by": {i: g.sort_values("score", ascending=False)
                             for i, g in t.groupby("entry_i")} if len(t) else {},
                      "open": [], "taken": 0})
    eq, curve, gross = 1.0, np.ones(len(idx)), np.zeros(len(idx))
    for i in range(len(idx)):
        for b in books:
            for q in [x for x in b["open"] if x[0] == i]:
                eq *= 1 + q[1] * q[2]
                b["open"].remove(q)
        for b in books:
            if i in b["by"] and len(b["open"]) < b["slots"]:
                held = {x[3] for x in b["open"]}
                for _, r in b["by"][i].iterrows():
                    if len(b["open"]) >= b["slots"]:
                        break
                    if r["lev"] <= 0 or r["code"] in held:
                        continue
                    b["open"].append((int(r["exit_i"]), float(r["ret_net"]),
                                      float(r["lev"]) / b["slots"], r["code"]))
                    b["taken"] += 1
        gross[i] = sum(x[2] for b in books for x in b["open"])
        if eq <= 0:
            curve[i:] = 0.0
            break
        curve[i] = eq
    e = pd.Series(curve, index=idx)
    st = curve_stats(e)
    st["最大總槓桿"] = round(float(gross.max()), 2)
    st["平均總槓桿"] = round(float(gross.mean()), 2)
    st["進場數"] = {b["name"]: b["taken"] for b in books}
    return e, st


def yearly(e: pd.Series, ic: pd.Series) -> pd.DataFrame:
    ix = index_curve(ic, e.index, 1.0)
    rows = []
    for y in sorted(set(e.index.year)):
        mk = e.index.year == y
        if mk.sum() < 50:
            continue
        s, i_ = e[mk], ix[mk]
        rows.append({"年": y, "策略%": round((s.iloc[-1] / s.iloc[0] - 1) * 100, 1),
                     "大盤%": round((i_.iloc[-1] / i_.iloc[0] - 1) * 100, 1),
                     "回撤%": round(float((1 - s / s.cummax()).max() * 100), 1)})
    return pd.DataFrame(rows).set_index("年")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebound-lev", type=float, default=None)
    ap.add_argument("--rebound-slots", type=int, default=None)
    args = ap.parse_args()
    rp = config.params("taiwan_rebound")
    fp = config.params("taiwan_futures")
    over = {}
    if args.rebound_lev is not None:
        over["leverage"] = args.rebound_lev
    slots_r = args.rebound_slots or rp["slots"]

    panel, streak, ic = load()
    idx = panel["Close"].index
    tm = momentum_trades(panel, streak, ic)
    tr, _ = rebound_trades(panel, ic, over)
    print(f"期間 {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}｜動能 {len(tm)} 筆｜反彈 {len(tr)} 筆")

    runs = {
        "動能單獨": [{"name": "動能", "trades": tm, "slots": fp["slots"]}],
        "反彈單獨": [{"name": "反彈", "trades": tr, "slots": slots_r}],
        "合併": [{"name": "動能", "trades": tm, "slots": fp["slots"]},
                 {"name": "反彈", "trades": tr, "slots": slots_r}],
    }
    curves, rows = {}, []
    for k, s in runs.items():
        e, st = simulate(s, idx)
        curves[k] = e
        rows.append({"配置": k, **{c: v for c, v in st.items() if c != "進場數"}})
    rows.append({"配置": "大盤 1.0x", **curve_stats(index_curve(ic, idx, 1.0))})
    print(pd.DataFrame(rows).to_string(index=False))

    ya = pd.concat({k: yearly(curves[k], ic)["策略%"] for k in ("動能單獨", "反彈單獨", "合併")}, axis=1)
    ya["大盤%"] = yearly(curves["合併"], ic)["大盤%"]
    ya["合併回撤%"] = yearly(curves["合併"], ic)["回撤%"]
    print("\n分年報酬%"); print(ya.to_string())


if __name__ == "__main__":
    main()
