import config, data as d, backtest, numpy as np
p0 = config.params('taiwan')
bars,_ = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c:v for c,v in bars.items() if not c.startswith('00')}
keep = {c:v for c,v in bars.items() if d.avg_turnover(v,250) >= 1e8}
LEV = 1/0.135   # 股票期貨保證金約 13.5% → 槓桿約 7.4 倍
for lbl, mh, sa in [("8天 stop2.5",8,2.5), ("14天 stop2.5",14,2.5), ("20天 stop2.5",20,2.5)]:
    p = dict(p0, max_hold_bars=mh, stop_atr=sa)
    pf=[];dd=[];rt=[];tr=[];wr=[]
    for c,df in keep.items():
        try: r = backtest.run_one(df,p,0.025)
        except Exception: continue
        if r and r['trades']>=5:
            pf.append(r['profit_factor']); dd.append(r['max_dd_pct'])
            rt.append(r['total_return_pct']); tr.append(r['trades']); wr.append(r['win_rate_pct'])
    m_dd, m_rt = np.median(dd), np.median(rt)
    print(f"{lbl:<14} PF {np.median(pf):.3f}｜勝率 {np.median(wr):.1f}%｜交易 {np.median(tr):.0f} 筆")
    print(f"{'':14} 標的層級：報酬 {m_rt:>6.1f}%  回撤 {m_dd:>5.1f}%")
    print(f"{'':14} ★ 7.4 倍槓桿後：報酬 {m_rt*LEV:>6.0f}%  回撤 {m_dd*LEV:>5.0f}%  ← 回撤 >100% 代表會被斷頭")
    print()
