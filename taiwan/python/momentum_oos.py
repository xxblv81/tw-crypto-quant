"""
樣本外驗證 —— 這套策略到目前為止的所有參數都是在同一段 2019~2026 上挑的。

本專案累積下來的 in-sample 選擇（依決定順序）：
    entry_score=8.5、exit_ma=20、max_hold_days=14、rs_lookbacks=60/120/240、
    chips_weight=0.15、breadth_weak=0.40 + 動態槓桿、near_high_pct=4.0
單一決定都有跨門檻／跨持股數的旁證，但七個疊起來就有過度配適風險。

做法（兩件事分開回答，不要混為一談）：
  ① 重新選參數 —— 只用 2019~2023 掃一遍，看「當時會不會選到同樣的值」。
     選到 = 這個參數穩定；選不到 = 它是被後半段資料帶出來的。
  ② 前推績效 —— 把①選出來的配置拿到 2024~2026 跑，那段完全沒參與過選擇。

★ 誠實聲明：這仍不是完美的樣本外。這些參數我在對話過程中已經看過全期結果，
  知識已經污染。真正乾淨的樣本外只能靠「未來的資料」。本檢驗能抓到的是
  「參數對期間極度敏感」這種明顯的過度配適，抓不到選擇者的偏誤。

用法：python momentum_oos.py
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

IS_END = pd.Timestamp("2023-12-31")
COST = config.costs("taiwan_futures")["commission_pct_per_side"]


def load_all():
    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    return panel, ch.buy_streak(ch.load_trust_net())


def score_cache(panel, streak, p0, fp):
    """rs 回看期 × 籌碼權重 的分數矩陣只算一次，後面重複用。"""
    out = {}
    for rs_lab, rs_lb in (("20/60/120", {20: .40, 60: .35, 120: .25}),
                          ("60/120/240", {60: 1/3, 120: 1/3, 240: 1/3})):
        for cw in (0.0, 0.15, 0.30):
            p = {**p0, "rs_lookbacks": rs_lb, "use_chips": cw > 0, "chips_weight": cw,
                 "exit_ma": fp["exit_ma"], "max_hold_days": fp["max_hold_days"],
                 "ma_set": sorted(set(p0["ma_set"]) | {10, 20, 60})}
            out[(rs_lab, cw)] = mom.compute_scores(panel, p, streak if cw > 0 else None)
    return out


def evaluate(panel, sc, p0, fp, ov, lo, hi, slots=15):
    """跑一段期間。ov 是要覆寫的參數。回傳逐筆與投組兩組指標。"""
    idx = panel["Close"].index
    p = {**p0, "exit_ma": ov["exit_ma"], "max_hold_days": ov["max_hold"],
         "ma_set": sorted(set(p0["ma_set"]) | {10, 20, 60})}
    msk = mom.liquidity_mask(panel["Turnover"], p0)
    pit = (panel["Turnover"].rolling(60, min_periods=60).mean()
           .rank(axis=1, ascending=False) <= fp["pit_universe_rank"])
    m = pit
    if ov["near_high"]:
        h = panel["High"].rolling(250, min_periods=250).max()
        m = m & (((h - panel["Close"]) / panel["Close"] * 100) <= ov["near_high"])

    rp = {**config.params("taiwan_market"), "breadth_weak": ov["breadth_weak"]}
    brd = market.breadth(panel["Close"], msk, rp["breadth_ma"])
    lvl = market.regime(market.load_index(), brd, rp, idx)["level"]
    levmap = {int(k): float(v) for k, v in fp["regime_leverage"].items()}

    tr = mb.extract_trades(panel, sc["score"], sc["ma"][ov["exit_ma"]], p,
                           ov["entry_score"], True, entry_mask=m)
    if tr.empty:
        return None
    tr = tr.assign(ret_net=tr["ret_gross"] - 2 * COST / 100,
                   lev=[levmap[int(lvl.iloc[i])] for i in tr["entry_i"]])
    sel = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] <= hi)]
    if len(sel) < 30:
        return None
    net = sel["ret_net"].to_numpy()

    sub = idx[(idx >= lo) & (idx <= hi)]
    s0 = int(np.searchsorted(idx, sub[0]))
    sim = sel.assign(entry_i=sel["entry_i"] - s0, exit_i=(sel["exit_i"] - s0).clip(upper=len(sub) - 1))
    _, port = mb.simulate_portfolio(sim, sub, slots, lev_col="lev")
    return {"筆數": len(sel),
            "PF": round(float(net[net > 0].sum() / max(-net[net < 0].sum(), 1e-9)), 3),
            "勝率": round(float((net > 0).mean() * 100), 1),
            "平均%": round(float(net.mean() * 100), 2),
            "CAGR": port["cagr_pct"], "回撤": port["max_dd_pct"]}


BASE = {"entry_score": 8.5, "exit_ma": 20, "max_hold": 14, "rs": "60/120/240",
        "chips_w": 0.15, "breadth_weak": 0.40, "near_high": 4.0}

GRID = {
    "entry_score": [7.5, 8.0, 8.5, 9.0],
    "exit_ma": [10, 20, 60],
    "max_hold": [0, 14],
    "rs": ["20/60/120", "60/120/240"],
    "chips_w": [0.0, 0.15, 0.30],
    "breadth_weak": [0.30, 0.35, 0.40, 0.45, 0.50],
    "near_high": [None, 10.0, 6.0, 4.0, 2.0],
}


def main() -> None:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    fp = config.params("taiwan_futures")
    panel, streak = load_all()
    idx = panel["Close"].index
    print(f"資料 {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}")
    print(f"樣本內 IS：2019-01-02 ~ {IS_END:%Y-%m-%d}"
          f"　樣本外 OOS：2024-01-01 ~ {idx[-1]:%Y-%m-%d}\n")
    caches = score_cache(panel, streak, p0, fp)

    # ── ① 只用 IS 重選每個參數（逐一座標掃描，與當初的決定方式一致）──
    print("① 只用 2019~2023 重新選參數")
    print(f"  {'參數':<14}{'IS 最佳':<14}{'現行值':<14}{'一致?':<6}  IS 的 PF 曲線")
    is_best = dict(BASE)
    for key, vals in GRID.items():
        rows = []
        for v in vals:
            ov = {**BASE, key: v}
            sc = caches[(ov["rs"], ov["chips_w"])]
            r = evaluate(panel, sc, p0, fp, ov, idx[0], IS_END)
            rows.append((v, r["PF"] if r else float("nan")))
        best = max((r for r in rows if np.isfinite(r[1])), key=lambda x: x[1])
        is_best[key] = best[0]
        same = "✓" if best[0] == BASE[key] else "✗"
        curve = " ".join(f"{v}:{pf:.2f}" for v, pf in rows)
        print(f"  {key:<14}{str(best[0]):<14}{str(BASE[key]):<14}{same:<6}  {curve}")

    print(f"\n  IS 重選的配置：{is_best}")
    changed = {k: (BASE[k], is_best[k]) for k in BASE if BASE[k] != is_best[k]}
    print(f"  與現行不同的：{changed if changed else '無 —— 七個參數在 IS 上會做出同樣選擇'}")

    # ── ② 前推：兩個配置各自在 IS / OOS 的表現 ──
    print("\n② 前推績效（OOS 完全沒參與參數選擇）")
    print(f"  {'配置':<16}{'期間':<8}{'筆數':>7}{'PF':>8}{'勝率':>8}{'平均%':>8}{'CAGR':>9}{'回撤':>8}")
    for lab, ov in (("現行（全期挑）", BASE), ("IS 重選", is_best)):
        for pname, lo, hi in (("IS", idx[0], IS_END),
                              ("OOS", pd.Timestamp("2024-01-01"), idx[-1])):
            sc = caches[(ov["rs"], ov["chips_w"])]
            r = evaluate(panel, sc, p0, fp, ov, lo, hi)
            if not r:
                print(f"  {lab:<16}{pname:<8}樣本不足")
                continue
            print(f"  {lab:<16}{pname:<8}{r['筆數']:>7}{r['PF']:>8.3f}{r['勝率']:>7.1f}%"
                  f"{r['平均%']:>7.2f}%{r['CAGR']:>8.1f}%{r['回撤']:>7.1f}%")


if __name__ == "__main__":
    main()
