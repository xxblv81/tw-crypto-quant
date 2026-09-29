"""
籌碼（投信連買）評估 —— 該不該把它放進動能分數。

愛德恩的四步驟第二步是「看投信是否連買」，但他沒說是當分數還是當過濾。
兩種用法績效可以差很多，所以兩種都測，外加一個「它自己有沒有訊號」的對照。

四個維度：
  A. 當分數的一部分 —— chips_weight 0 ~ 0.30，其餘四項等比例縮小
  B. 連買幾天算滿分 —— chips_max_days 2/3/5/10
  C. 當進場過濾 —— 要求進場當天投信連買 ≥ N 天
     ★ 這維一定要配對照組：過濾會讓筆數變少，而「更嚴格」本身就會提高 PF。
       所以同時算「把價量門檻拉到同樣筆數」的基準，比的是誰的篩選比較聰明。
  D. 單獨用籌碼選股（不看動能分數）—— 它自己到底有沒有預測力

比較口徑：A/B 用「當天前 N%」控制選股嚴格度（理由同 momentum_rs_sweep.py）。

用法：python momentum_chips_sweep.py     （需先跑 python chips.py --start 2019-01-01）
輸出：momentum_chips_sweep.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import chips
import config
import momentum as mom
import momentum_backtest as mb
import twse

SLOTS = (8, 15, 25)


def stats(tr, cost, idx, label, extra=None) -> dict:
    net = tr["ret_gross"].to_numpy() - 2 * cost / 100
    tr = tr.assign(ret_net=net)
    r = {"config": label, "trades": len(tr),
         "win_pct": round(float((net > 0).mean() * 100), 1),
         "pf_net": round(float(net[net > 0].sum() / max(-net[net < 0].sum(), 1e-9)), 3),
         "avg_net_pct": round(float(net.mean() * 100), 2),
         "median_hold": int(tr["hold_days"].median())}
    for s in SLOTS:
        _, port = mb.simulate_portfolio(tr, idx, s)
        r[f"cagr@{s}"] = port["cagr_pct"]
        r[f"dd@{s}"] = port["max_dd_pct"]
    return {**r, **(extra or {})}


def line(r) -> str:
    return (f"{r['config']:<28} PF {r['pf_net']:.3f}  勝率 {r['win_pct']:.1f}%  "
            f"平均 {r['avg_net_pct']:+.2f}%  {r['trades']:>5} 筆  "
            f"CAGR@15 {r['cagr@15']:>6.1f}%  回撤 {r['dd@15']:.1f}%")


def main() -> None:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    cost = config.costs("taiwan_momentum")["commission_pct_per_side"]

    trust = chips.load_trust_net()
    if trust.empty:
        print("籌碼快取是空的。先跑：python chips.py --start 2019-01-01")
        return
    streak_raw = chips.buy_streak(trust)
    print(f"籌碼快取：{streak_raw.shape[0]} 天 × {streak_raw.shape[1]} 檔"
          f"｜{streak_raw.index[0]:%Y-%m-%d} ~ {streak_raw.index[-1]:%Y-%m-%d}")

    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    close, high, to = panel["Close"], panel["High"], panel["Turnover"]
    idx = close.index
    mask = mom.liquidity_mask(to, p0)

    # 籌碼快取的日期範圍可能短於行情 → 只在兩者都有的區間比較，否則是在比資料長度
    common = idx.intersection(streak_raw.index)
    if len(common) < len(idx) * 0.9:
        print(f"[warn] 籌碼只覆蓋 {len(common)}/{len(idx)} 個交易日，"
              f"結果僅供參考，不要當定論")
    streak = streak_raw.reindex(index=idx, columns=close.columns).fillna(0)

    rs = mom.rs_score(close, mask, p0)
    vol = mom.volume_score(to, mask, p0)
    pos = mom.position_score(close, high, p0)
    align, ma = mom.align_score(close, p0)
    ma_exit = ma[p0["exit_ma"]]
    base4 = {"rs": rs, "volume": vol, "align": align, "position": pos}
    w4 = p0["weights"]

    def score_with(cw: float, max_days: int):
        """籌碼佔 cw，其餘四項按現行比例縮到 (1-cw)。"""
        s = sum(base4[k] * (w4[k] * (1 - cw)) for k in w4)
        if cw > 0:
            s = s + (streak.clip(0, max_days) / max_days * 10) * cw
        return s.where(mask)

    base_pct = (score_with(0.0, 5)).rank(axis=1, pct=True) * 100

    def run(score_pct, thr_pct, label, entry_mask=None, extra=None):
        tr = mb.extract_trades(panel, score_pct, ma_exit, p0, thr_pct,
                               limit_up_guard=True, entry_mask=entry_mask)
        if tr.empty:
            return None
        return stats(tr, cost, idx, label, extra)

    def baseline_at(n_target):
        """
        找出讓筆數最接近 n_target 的價量門檻 —— C 維的對照組。
        粗掃再細掃：門檻與筆數單調，不必線性掃 40 個點。
        """
        def n_at(q):
            tr = mb.extract_trades(panel, base_pct, ma_exit, p0, float(q), True)
            return len(tr), tr
        best = None
        for q in np.arange(90.0, 99.6, 1.0):
            n, tr = n_at(q)
            if best is None or abs(n - n_target) < best[0]:
                best = (abs(n - n_target), float(q), tr)
        lo, hi = max(90.0, best[1] - 1.0), min(99.6, best[1] + 1.0)
        for q in np.arange(lo, hi + 1e-9, 0.2):
            n, tr = n_at(q)
            if abs(n - n_target) < best[0]:
                best = (abs(n - n_target), float(q), tr)
        return best[1], best[2]

    THR = 95.0                      # 當天前 5%，與 RS 敏感度分析同一嚴格度
    rows = []

    print(f"\n── A. 籌碼當分數的一部分（門檻：當天前 {100-THR:.0f}%，滿分 5 天）──")
    for cw in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        r = run(score_with(cw, 5).rank(axis=1, pct=True) * 100, THR,
                f"A 籌碼權重 {cw:.2f}")
        if r: rows.append({**r, "dim": "A"}); print("  " + line(r), flush=True)

    print("\n── B. 連買幾天算滿分（籌碼權重固定 0.15）──")
    for md in (2, 3, 5, 10):
        r = run(score_with(0.15, md).rank(axis=1, pct=True) * 100, THR,
                f"B 滿分 {md} 天")
        if r: rows.append({**r, "dim": "B"}); print("  " + line(r), flush=True)

    print("\n── C. 籌碼當進場過濾（vs 同筆數的純價量對照組）──")
    for k in (1, 2, 3, 5):
        r = run(base_pct, THR, f"C 需投信連買≥{k}天", entry_mask=(streak >= k))
        if not r:
            continue
        q, tr_ctrl = baseline_at(r["trades"])
        ctrl = stats(tr_ctrl, cost, idx, f"   對照：純價量前{100-q:.2f}%")
        r["ctrl_pf"] = ctrl["pf_net"]; r["ctrl_trades"] = ctrl["trades"]
        r["edge_vs_ctrl"] = round(r["pf_net"] - ctrl["pf_net"], 3)
        rows.append({**r, "dim": "C"})
        print("  " + line(r), flush=True)
        print("  " + line(ctrl) + f"   → 籌碼過濾的淨效果 {r['edge_vs_ctrl']:+.3f} PF", flush=True)

    print("\n── D. 單獨用籌碼選股（完全不看動能分數）──")
    for k in (3, 5, 8):
        fake = (streak.clip(0, 20) / 20 * 100).where(mask)
        r = run(fake, float(k) / 20 * 100, f"D 只看連買≥{k}天")
        if r: rows.append({**r, "dim": "D"}); print("  " + line(r), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("momentum_chips_sweep.csv", index=False, encoding="utf-8-sig")
    print("\n→ momentum_chips_sweep.csv")


if __name__ == "__main__":
    main()
