import config, data as d, backtest, signals as sig, numpy as np
p0 = config.params('taiwan'); cost = config.costs('taiwan')['commission_pct_per_side']
bars,_ = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c:v for c,v in bars.items() if not c.startswith('00')}
keep = {c:v for c,v in bars.items() if d.avg_turnover(v,250) >= 2e7}
print(f"樣本 {len(keep)} 檔｜基礎：候選期30根 + 多頭排列20>60>120 + 量>5日均×1.2\n", flush=True)
print("穿越以來放量倍數 │ 總訊號 有效檔  PF    勝率%  報酬%  回撤%  交易")
BASE = dict(mid_cross_window=30, ma_stack=[20,60,120])
for m in (0, 1.0, 1.2, 1.3, 1.5, 2.0):
    p = dict(p0, **BASE, volume_since_cross_mult=m); tot=0; rows=[]
    for c,df in keep.items():
        try:
            s = sig.compute(df,p); tot += int(s['signal'].sum())
            r = backtest.run_one(df,p,cost)
        except Exception: continue
        if r and r['trades']>=5: rows.append(r)
    if not rows: print(f"  ×{m:<13}│ {tot:>6}     0  樣本不足", flush=True); continue
    f=lambda k: np.median([r[k] for r in rows])
    lbl = "關閉" if m==0 else f"×{m}"
    print(f"  {lbl:<13}│ {tot:>6} {len(rows):>5}  {f('profit_factor'):>5.3f}  {f('win_rate_pct'):>5.1f}"
          f"  {f('total_return_pct'):>7.1f}  {f('max_dd_pct'):>5.1f}  {f('trades'):>4.0f}", flush=True)
