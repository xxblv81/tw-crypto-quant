"""
RS 敏感度分析 —— 三個維度全掃。

★ 這支解決一個公平比較的陷阱：
  四個分項各自是 0~10，但**加權平均的分布會隨權重改變**。
  純 rs（權重 1.0）是均勻分布，四項混合則向中間集中。
  所以「分數 ≥ 8.5」在不同權重下代表不同的嚴格度 —— 直接比會比到分布差異，不是比到績效。

  解法：把最終分數再做一次橫斷面百分位，用「當天前 N%」當進場條件。
  這樣每個配置的選股數量一致，比較的才是「排序品質」本身。

三個維度：
  A. 四項權重 —— rs 佔比 0.0 ~ 1.0，其餘三項按現行 1:1:1 比例分配
  B. rs 內部的多期權重 —— 20/60/120 日報酬怎麼配
  C. rs 的回看期間本身 —— 要不要換成更短或更長的組合

用法：python momentum_rs_sweep.py
輸出：momentum_rs_sweep.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import momentum as mom
import momentum_backtest as mb
import twse

EXIT_MA = 20
SLOTS = 8
TOP_PCT = None          # 由 baseline 校準決定，見 main()


def rs_from(close, mask, lookbacks) -> pd.DataFrame:
    raw = None
    for k, w in lookbacks.items():
        r = close / close.shift(int(k)) - 1
        raw = r * float(w) if raw is None else raw + r * float(w)
    return raw.where(mask).rank(axis=1, pct=True) * 10


def evaluate(score, ma_exit, panel, p, thr_pct, cost, idx, label) -> dict:
    """score → 橫斷面百分位（0~100）→ 以『前 N%』為進場門檻，跑完整回測。"""
    pct = score.rank(axis=1, pct=True) * 100
    tr = mb.extract_trades(panel, pct, ma_exit, p, thr_pct, limit_up_guard=True)
    if tr.empty:
        return {}
    net = tr["ret_gross"].to_numpy() - 2 * cost / 100
    tr["ret_net"] = net
    _, port = mb.simulate_portfolio(tr, idx, SLOTS)
    return {
        "config": label, "trades": len(tr),
        "win_pct": round(float((net > 0).mean() * 100), 1),
        "pf_net": round(float(net[net > 0].sum() / max(-net[net < 0].sum(), 1e-9)), 3),
        "avg_net_pct": round(float(net.mean() * 100), 2),
        "median_hold": int(tr["hold_days"].median()),
        "port_cagr_pct": port["cagr_pct"], "port_dd_pct": port["max_dd_pct"],
    }


def main() -> None:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    cost = config.costs("taiwan_momentum")["commission_pct_per_side"]
    p = {**p0, "exit_ma": EXIT_MA}

    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    close, high, turnover = panel["Close"], panel["High"], panel["Turnover"]
    idx = close.index
    mask = mom.liquidity_mask(turnover, p)

    # 四項裡與 rs 無關的三項只算一次
    vol = mom.volume_score(turnover, mask, p)
    pos = mom.position_score(close, high, p)
    align, ma = mom.align_score(close, p)
    ma_exit = ma[EXIT_MA]
    base_rs = rs_from(close, mask, p["rs_lookbacks"])

    def combine(rs, w_rs, rs_mat=None):
        """rs 佔 w_rs，其餘三項按現行 1:1:1 分配。"""
        rest = (1 - w_rs) / 3
        s = (rs_mat if rs_mat is not None else rs) * w_rs + (vol + align + pos) * rest
        return s.where(mask)

    # ── 校準：找出讓交易筆數貼近現行配置（8.5 分、6,621 筆）的「前 N%」 ──
    base_score = mom.compute_scores(panel, p, None)["score"]
    n_ref = len(mb.extract_trades(panel, base_score, ma_exit, p, p["entry_score"], True))
    cand = [98.5, 98.0, 97.5, 97.0, 96.0, 95.0]
    picks = []
    for q in cand:
        n = len(mb.extract_trades(panel, base_score.rank(axis=1, pct=True) * 100,
                                  ma_exit, p, q, True))
        picks.append((abs(n - n_ref), q, n))
    _, thr_pct, n_at = sorted(picks)[0]
    print(f"現行配置（分數 ≥ {p['entry_score']}）= {n_ref} 筆"
          f"　→　校準為『當天前 {100 - thr_pct:.1f}%』= {n_at} 筆，"
          f"以下所有配置都用同一個選股嚴格度\n")

    rows = []

    print("── A. 四項權重：rs 佔比 ──")
    for w_rs in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
        r = evaluate(combine(base_rs, w_rs), ma_exit, panel, p, thr_pct, cost, idx,
                     f"A rs={w_rs:.2f} 其餘各{(1-w_rs)/3:.3f}")
        rows.append({**r, "dim": "A", "rs_weight": w_rs}); print("  " + fmt(r), flush=True)

    print("\n── B. rs 內部多期權重（回看期固定 20/60/120，四項權重固定 0.4）──")
    mixes = {
        "純 R20": {20: 1.0}, "純 R60": {60: 1.0}, "純 R120": {120: 1.0},
        "等權 1/3": {20: 1/3, 60: 1/3, 120: 1/3},
        "現行 .40/.35/.25": {20: .40, 60: .35, 120: .25},
        "偏短 .60/.30/.10": {20: .60, 60: .30, 120: .10},
        "偏長 .10/.30/.60": {20: .10, 60: .30, 120: .60},
        "去掉最短 0/.50/.50": {60: .50, 120: .50},
    }
    for lab, lb in mixes.items():
        r = evaluate(combine(None, 0.40, rs_from(close, mask, lb)), ma_exit, panel, p,
                     thr_pct, cost, idx, f"B {lab}")
        rows.append({**r, "dim": "B"}); print("  " + fmt(r), flush=True)

    print("\n── C. 回看期間組合（內部等權，四項權重固定 0.4）──")
    sets = {
        "5/20/60 超短": [5, 20, 60], "10/20/60 短": [10, 20, 60],
        "20/60/120 現行": [20, 60, 120], "20/60/240": [20, 60, 240],
        "60/120/240 長": [60, 120, 240], "120/240 純長": [120, 240],
    }
    for lab, ks in sets.items():
        lb = {k: 1 / len(ks) for k in ks}
        r = evaluate(combine(None, 0.40, rs_from(close, mask, lb)), ma_exit, panel, p,
                     thr_pct, cost, idx, f"C {lab}")
        rows.append({**r, "dim": "C"}); print("  " + fmt(r), flush=True)

    out = pd.DataFrame([r for r in rows if r.get("config")])
    out.to_csv("momentum_rs_sweep.csv", index=False, encoding="utf-8-sig")
    print("\n" + out.to_string(index=False))
    print("\n→ momentum_rs_sweep.csv")


def fmt(r: dict) -> str:
    if not r:
        return "(無交易)"
    return (f"{r['config']:<26} PF {r['pf_net']:.3f}  勝率 {r['win_pct']:.1f}%  "
            f"平均 {r['avg_net_pct']:+.2f}%  {r['trades']:>5} 筆  "
            f"投組 CAGR {r['port_cagr_pct']:>6.1f}%  回撤 {r['port_dd_pct']:.1f}%")


if __name__ == "__main__":
    main()
