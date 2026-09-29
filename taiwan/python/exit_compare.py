import config, data as d, backtest, numpy as np
p0 = config.params('taiwan'); cost = config.costs('taiwan')['commission_pct_per_side']
bars,names = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c:v for c,v in bars.items() if not c.startswith('00')}
keep = {c:v for c,v in bars.items() if d.avg_turnover(v,250) >= 2e7}
bh = {c: (float(v['Close'].iloc[-1])/float(v['Close'].iloc[0])-1)*100 for c,v in keep.items()}
print(f"樣本 {len(keep)} 檔｜2019~2026｜買進持有中位數 {np.median(list(bh.values())):.1f}%\n", flush=True)

cfgs = [("① 固定停利 rr2（現行）", {}),
        ("② 只停損，不停利",       {"take_profit": False}),
        ("③ ATR 移動停利",         {"take_profit": False, "trailing": True}),
        ("④ 跌破 MA20 出場",       {"take_profit": False, "exit_ma": 20}),
        ("⑤ 跌破 MA60 出場",       {"take_profit": False, "exit_ma": 60})]

print("設定                  │  PF    勝率%  中位報酬%  平均報酬%  第90百分位%  回撤%  交易數 持有天")
res_all = {}
for lbl, over in cfgs:
    p = dict(p0, **over); rows=[]
    for c,df in keep.items():
        try: r = backtest.run_one(df,p,cost)
        except Exception: continue
        if r and r['trades']>=5: rows.append((c,r))
    res_all[lbl] = dict(rows)
    pf=np.median([r['profit_factor'] for _,r in rows]); wr=np.median([r['win_rate_pct'] for _,r in rows])
    rt=np.median([r['total_return_pct'] for _,r in rows]); mn=np.mean([r['total_return_pct'] for _,r in rows])
    p90=np.percentile([r['total_return_pct'] for _,r in rows],90)
    dd=np.median([r['max_dd_pct'] for _,r in rows]); tr=np.median([r['trades'] for _,r in rows])
    hd=np.median([r['median_hold_bars'] for _,r in rows])
    print(f"{lbl:<21} │ {pf:>5.3f}  {wr:>5.1f}  {rt:>8.1f}  {mn:>8.1f}  {p90:>9.1f}  {dd:>5.1f}  {tr:>5.0f}  {hd:>4.0f}", flush=True)

# 右尾捕捉率：在買進持有前 10% 的暴漲股上，策略抓到多少
top = [c for c,_ in sorted(bh.items(), key=lambda x:-x[1])[:100]]
print(f"\n=== 右尾捕捉：買進持有報酬前 100 名的股票 ===")
print(f"這 100 檔買進持有中位數 {np.median([bh[c] for c in top]):.0f}%")
for lbl,_ in cfgs:
    sub = [res_all[lbl][c]['total_return_pct'] for c in top if c in res_all[lbl]]
    if sub:
        print(f"  {lbl:<21} 策略中位報酬 {np.median(sub):>8.1f}%  （捕捉率 {np.median(sub)/np.median([bh[c] for c in top])*100:>5.1f}%）")
