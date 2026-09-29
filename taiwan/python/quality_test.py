import config, data as d, twse, backtest, signals as sig, numpy as np, pandas as pd
p0 = config.params("taiwan"); cost = config.costs("taiwan")["commission_pct_per_side"]
bars, _ = twse.load_bars(include_otc=True, adjust=True, min_rows=260)
bars = {c: v for c, v in bars.items() if not c.startswith("00")}
to = {c: float(v["Turnover"].tail(250).mean()) for c, v in bars.items()}
print(f"載入 {len(bars)} 檔\n", flush=True)
print("設定                        門檻   檔數  總訊號   PF    勝率%  中位報酬%  回撤%")
def run(lbl, over, minto):
    p = dict(p0, **over)
    keep = {c: v for c, v in bars.items() if to[c] >= minto}
    tot = 0; rows = []
    for c, df in keep.items():
        try:
            tot += int(sig.compute(df, p)["signal"].sum())
            r = backtest.run_one(df, p, cost)
        except Exception:
            continue
        if r and r["trades"] >= 5: rows.append(r)
    if not rows:
        print(f"{lbl:<26} {minto/1e8:>4.1f}億  樣本不足", flush=True); return
    f = lambda k: np.median([r[k] for r in rows])
    print(f"{lbl:<26} {minto/1e8:>4.1f}億 {len(rows):>5} {tot:>7}  {f('profit_factor'):>5.3f}"
          f"  {f('win_rate_pct'):>5.1f}  {f('total_return_pct'):>8.1f}  {f('max_dd_pct'):>5.1f}", flush=True)
run("① 現行", {}, 2e7)
run("② 成交金額門檻拉高", {}, 1e8)
for v in (1.5, 2.0, 2.5):
    run(f"③ ATR% ≥ {v}", {"min_atr_pct": v}, 2e7)
for v in (0.80, 0.85, 0.90):
    run(f"④ 距52週高 ≤ {int((1-v)*100)}%", {"near_high_pct": v}, 2e7)
run("⑤ ATR2.0 + 近高0.85", {"min_atr_pct": 2.0, "near_high_pct": 0.85}, 2e7)
run("⑥ ⑤ + 成交金額1億", {"min_atr_pct": 2.0, "near_high_pct": 0.85}, 1e8)
