"""逐根盯市的 Sharpe 評估（2026-09-24 版）。

★ 2026-09-30 起已由 engine.py / optimize.py 取代，保留是為了能重現 SHARPE_RESULTS.md 的舊數字。
"""
import sys, datetime as d, numpy as np, pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R/"taiwan"/"python"),str(R/"crypto"/"python")]
import config, signals as sig, binance_data as bd
BASE=config.params("crypto"); COST=config.costs("crypto")["commission_pct_per_side"]/100
DATA={s:bd.load(s,"4h",d.date(2018,1,1),d.date(2026,9,24),verbose=False) for s in ["BTCUSDT","ETHUSDT"]}

def sim(df,p):
    s=sig.compute(df,p); o,h,l,c=(s[k].to_numpy() for k in ["Open","High","Low","Close"])
    sigl=s["signal"].to_numpy(); atr=s["atr"].to_numpy(); n=len(s)
    xma=s["Close"].rolling(int(p.get("exit_ma",0))).mean().to_numpy() if p.get("exit_ma") else None
    mh=int(p.get("max_hold_bars",0) or 0)
    r=np.zeros(n); trades=[]; i=0
    while i<n-1:
        if not sigl[i] or not np.isfinite(atr[i]): i+=1; continue
        e=o[i+1]; a=atr[i]; st=e-p["stop_atr"]*a; tg=e+p["stop_atr"]*a*p["rr"] if p["take_profit"] else np.inf
        j=i+1; px=None
        while j<n:
            if l[j]<=st: px=min(st,o[j]) if j>i+1 else st; break
            if h[j]>=tg: px=max(tg,o[j]) if j>i+1 else tg; break
            if xma is not None and c[j]<xma[j]:
                if p.get('exit_next_open') and j+1<n: j+=1; px=o[j]
                else: px=c[j]
                break
            if mh and j>=i+1+mh: px=c[j]; break
            j+=1
        if px is None: j=n-1; px=c[j]
        for k in range(i+1,j+1):
            prev=e if k==i+1 else c[k-1]; cur=px if k==j else c[k]
            r[k]+=cur/prev-1
        r[i+1]-=COST; r[j]-=COST
        trades.append((s.index[i+1],px/e-1-2*COST)); i=j+1
    return pd.Series(r,index=s.index), pd.Series([t for _,t in trades],index=[k for k,_ in trades],dtype=float)

def stats(r,tr):
    dr=(1+r).groupby(r.index.date).prod()-1
    eq=(1+dr).cumprod(); yrs=len(dr)/365
    sh=dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan
    neg=dr[dr<0]; so=dr.mean()/np.sqrt((np.minimum(dr,0)**2).mean())*np.sqrt(365)
    mdd=(1-eq/eq.cummax()).max()
    gp,gl=tr[tr>0].sum(),-tr[tr<0].sum()
    return dict(sharpe=round(sh,2),sortino=round(so,2),cagr=round((eq.iloc[-1]**(1/yrs)-1)*100,1),
                mdd=round(mdd*100,1),n=len(tr),pf=round(gp/gl,2) if gl>0 else np.inf,
                win=round((tr>0).mean()*100,0) if len(tr) else 0,
                expo=round((r!=0).mean()*100,0))

P=(("2018-01-01","2023-12-31","IS"),("2024-01-01","2026-09-24","OOS"))
def per(p,sym):
    r,tr=sim(DATA[sym],p); out={}
    for a,b,tag in P:
        rr=r.loc[a:b]; dr=(1+rr).groupby(rr.index.date).prod()-1; eq=(1+dr).cumprod()
        t=tr.loc[a:b].to_numpy(); gp,gl=t[t>0].sum(),-t[t<0].sum()
        out[tag]=dict(sh=dr.mean()/dr.std()*np.sqrt(365),n=len(t),pf=gp/gl if gl>0 else 9,dd=(1-eq/eq.cummax()).max()*100,
                      cagr=(eq.iloc[-1]**(365/len(dr))-1)*100)
    return out
def show(V,syms=("BTCUSDT","ETHUSDT")):
    for sym in syms:
        print(f"\n{sym:<22}{'IS 2018-23: Sharpe  筆  PF  回撤  CAGR':<40}{'OOS 2024-26: Sharpe 筆  PF  回撤  CAGR'}")
        for lab,d in V:
            o=per({**BASE,**d},sym)
            f=lambda x:f"{x['sh']:>6.2f} {x['n']:>4} {x['pf']:>5.2f} {x['dd']:>5.1f} {x['cagr']:>6.1f}"
            print(f"  {lab:<20}{f(o['IS']):<40}{f(o['OOS'])}")
if __name__ == "__main__":
    N = {"break_on_high": False, "take_profit": False, "exit_next_open": True, "exit_ma": 20}
    show([("舊 2R 停利", {"take_profit": True, "exit_ma": 0}), ("新 MA20 出場", N),
          ("新出場＋最高價判定", {**N, "break_on_high": True})]
         + [(f"出場 MA{m}", {**N, "exit_ma": m}) for m in (10, 15, 30, 50, 100)])
