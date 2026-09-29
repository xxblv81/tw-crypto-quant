import config, data as d, backtest, signals as sig, numpy as np
p0 = config.params('taiwan'); cost = config.costs('taiwan')['commission_pct_per_side']
bars,_ = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c:v for c,v in bars.items() if not c.startswith('00')}
keep = {c:v for c,v in bars.items() if d.avg_turnover(v,250) >= 2e7}
print(f"樣本 {len(keep)} 檔\n", flush=True)

MID = dict(entry_mode="mid_reclaim")
V5  = dict(volume_filter=True, volume_ma_length=5)
K   = dict(require_bull_candle=True)

cfgs = [
 ("① 現行：上軌突破、無量能無K棒", {}),
 ("② 中線反轉（只換進場點）",       MID),
 ("③ ②+紅K(實體>0)",              {**MID, **K}),
 ("④ ②+量>5日均×1.5",             {**MID, **V5, "volume_mult":1.5}),
 ("⑤ ②+紅K+量×1.5  ←你要的",      {**MID, **K, **V5, "volume_mult":1.5}),
 ("⑥ ⑤ 但量×1.2（放寬）",          {**MID, **K, **V5, "volume_mult":1.2}),
 ("⑦ ⑤ 但加實體>50%",             {**MID, **K, "min_body_ratio":0.5, **V5, "volume_mult":1.5}),
 ("⑧ ⑤ 但加實體>50%+收高檔>60%",   {**MID, **K, "min_body_ratio":0.5, "min_close_pos":0.6, **V5, "volume_mult":1.5}),
 ("⑨ 上軌突破+紅K+量×1.5",         {**K, **V5, "volume_mult":1.5}),
]
print("設定                         │ 總訊號  有效檔數  PF     勝率%  中位報酬%  回撤%  中位交易")
for lbl, over in cfgs:
    p = dict(p0, **over)
    tot=0; rows=[]
    for c,df in keep.items():
        try:
            s = sig.compute(df,p); tot += int(s['signal'].sum())
            r = backtest.run_one(df,p,cost)
        except Exception: continue
        if r and r['trades']>=5: rows.append(r)
    if not rows:
        print(f"{lbl:<28} │ {tot:>6}       0   —（樣本不足）", flush=True); continue
    pf=np.median([r['profit_factor'] for r in rows]); wr=np.median([r['win_rate_pct'] for r in rows])
    rt=np.median([r['total_return_pct'] for r in rows]); dd=np.median([r['max_dd_pct'] for r in rows])
    tr=np.median([r['trades'] for r in rows])
    print(f"{lbl:<28} │ {tot:>6}  {len(rows):>6}  {pf:>5.3f}  {wr:>5.1f}  {rt:>8.1f}  {dd:>5.1f}  {tr:>6.0f}", flush=True)
