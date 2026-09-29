"""
「抱得住」量化分析。

使用者的論點：買在起漲點 → 一開始就是浮盈 → 不會被套牢 → 抱得住 → 資金效率高。
這無法用「未來報酬」衡量（前一個分析做的），要用「買進後最深虧多少」。

指標：
  MAE  最大逆行幅度 = 買進後 N 天內最低點 vs 進場價（負數，越接近 0 越好抱）
  套牢率 = 買進後曾經虧超過 10% / 20% 的比例
  MFE  最大順行幅度 = 同期最高點（對照用）
"""
import config, data as d, signals as sig, numpy as np, pandas as pd

p = config.params('taiwan')
bars, names = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c: v for c, v in bars.items() if not c.startswith('00')}
keep = {c: v for c, v in bars.items() if d.avg_turnover(v, 250) >= 2e7}
print(f"樣本 {len(keep)} 檔", flush=True)

HOR = [20, 60, 120, 250]
S = {h: {'mae': [], 'mfe': []} for h in HOR}
B = {h: {'mae': [], 'mfe': []} for h in HOR}

for c, df in keep.items():
    try:
        s = sig.compute(df, p)
    except Exception:
        continue
    op = s['Open'].to_numpy(); lo = s['Low']; hi = s['High']
    n = len(op)
    idx = np.where(s['signal'].to_numpy())[0]
    idx = idx[idx < n - 1]
    for h in HOR:
        # 進場價 = 隔日開盤；MAE = 之後 h 根內最低點 / 進場價 - 1
        fmin = lo.rolling(h).min().shift(-h).to_numpy()
        fmax = hi.rolling(h).max().shift(-h).to_numpy()
        entry = np.full(n, np.nan); entry[:-1] = op[1:]
        mae = fmin / entry - 1
        mfe = fmax / entry - 1
        ok = ~np.isnan(mae)
        B[h]['mae'].extend(mae[ok].tolist()); B[h]['mfe'].extend(mfe[ok].tolist())
        si = idx[~np.isnan(mae[idx])]
        if len(si):
            S[h]['mae'].extend(mae[si].tolist()); S[h]['mfe'].extend(mfe[si].tolist())

print(f"訊號樣本 {len(S[20]['mae']):,}｜基準樣本 {len(B[20]['mae']):,}\n")
print("買進後最深虧損（MAE）—— 數字越接近 0 越抱得住")
print("期間 │ 訊號日中位  基準中位   差   │ 訊號P25   基準P25  │ 虧>10%比例(訊號/基準) │ 虧>20%(訊號/基準)")
for h in HOR:
    a, b = np.array(S[h]['mae'])*100, np.array(B[h]['mae'])*100
    print(f"{h:>4} │ {np.median(a):>8.2f}% {np.median(b):>8.2f}% {np.median(a)-np.median(b):>+6.2f} │"
          f" {np.percentile(a,25):>7.1f}% {np.percentile(b,25):>7.1f}% │"
          f"   {(a<-10).mean()*100:>5.1f}% / {(b<-10).mean()*100:>5.1f}%      │"
          f" {(a<-20).mean()*100:>5.1f}% / {(b<-20).mean()*100:>5.1f}%", flush=True)

print("\n最大順行幅度（MFE）—— 買進後最高漲到哪")
print("期間 │ 訊號日中位  基準中位   差")
for h in HOR:
    a, b = np.array(S[h]['mfe'])*100, np.array(B[h]['mfe'])*100
    print(f"{h:>4} │ {np.median(a):>8.2f}% {np.median(b):>8.2f}% {np.median(a)-np.median(b):>+6.2f}")

print("\n報酬風險比（MFE 中位 ÷ |MAE 中位|）")
for h in HOR:
    sa = np.median(S[h]['mfe'])/abs(np.median(S[h]['mae']))
    ba = np.median(B[h]['mfe'])/abs(np.median(B[h]['mae']))
    print(f"{h:>4} 日 │ 訊號 {sa:>5.2f}   基準 {ba:>5.2f}   {'訊號較佳' if sa>ba else '基準較佳'}")
