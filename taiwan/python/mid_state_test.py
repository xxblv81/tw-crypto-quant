import config, data as d, backtest, signals as sig, numpy as np
p0 = config.params('taiwan'); cost = config.costs('taiwan')['commission_pct_per_side']
bars,_ = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c:v for c,v in bars.items() if not c.startswith('00')}
keep = {c:v for c,v in bars.items() if d.avg_turnover(v,250) >= 2e7}
print(f"樣本 {len(keep)} 檔\n", flush=True)
print("設定                          │ 總訊號 有效檔  PF    勝率%  報酬%  回撤%  交易 持有天")
def run(lbl, over):
    p = dict(p0, **over); tot=0; rows=[]
    for c,df in keep.items():
        try:
            s = sig.compute(df,p); tot += int(s['signal'].sum())
            r = backtest.run_one(df,p,cost)
        except Exception: continue
        if r and r['trades']>=5: rows.append(r)
    if not rows: print(f"{lbl:<30}│ {tot:>6}     0  樣本不足", flush=True); return
    f=lambda k: np.median([r[k] for r in rows])
    print(f"{lbl:<30}│ {tot:>6} {len(rows):>5}  {f('profit_factor'):>5.3f}  {f('win_rate_pct'):>5.1f}"
          f"  {f('total_return_pct'):>7.1f}  {f('max_dd_pct'):>5.1f}  {f('trades'):>4.0f}  {f('median_hold_bars'):>4.0f}", flush=True)

run("① 現行 上軌突破", {})
B = dict(entry_mode="mid_confirm", require_bull_candle=True,
         volume_filter=True, volume_ma_length=5)
for em in (1.0, 1.2, 1.5):
    for vm in (1.0, 1.2, 1.5):
        run(f"帶寬×{em} 量×{vm}", dict(B, expand_mult=em, volume_mult=vm))
