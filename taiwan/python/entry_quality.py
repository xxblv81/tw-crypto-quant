"""
進場品質分析 —— 這策略當「選股進場點」用，值不值得。

使用者的用法是：用訊號選股 → 買進 → 長抱，沒有出場規則。
所以要測的不是 PF，是「訊號日買進 vs 任意一天買進」的前瞻報酬差距。

作法：對每檔股票算「未來 N 日報酬」，比較兩個分布：
  A. 所有訊號日的未來報酬
  B. 所有交易日的未來報酬（基準，代表隨便挑一天買）
兩者受同樣的大盤漲跌影響，差額才是訊號本身的價值。
"""
import config, data as d, signals as sig, numpy as np, pandas as pd

p = config.params('taiwan')
bars, names = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c: v for c, v in bars.items() if not c.startswith('00')}
keep = {c: v for c, v in bars.items() if d.avg_turnover(v, 250) >= 2e7}
print(f"樣本 {len(keep)} 檔\n", flush=True)

HOR = [20, 60, 120, 250]
sig_r = {h: [] for h in HOR}
base_r = {h: [] for h in HOR}
n_sig = 0

for c, df in keep.items():
    try:
        s = sig.compute(df, p)
    except Exception:
        continue
    close = s['Close'].to_numpy()
    n = len(close)
    idx = np.where(s['signal'].to_numpy())[0]
    n_sig += len(idx)
    for h in HOR:
        fwd = np.full(n, np.nan)
        fwd[:n-h] = close[h:] / close[:n-h] - 1
        ok = ~np.isnan(fwd)
        base_r[h].extend(fwd[ok].tolist())
        si = idx[(idx < n - h)]
        if len(si):
            sig_r[h].extend(fwd[si].tolist())

print(f"訊號總數 {n_sig:,}\n")
print("未來期間 │ 訊號日買進          任意日買進(基準)      超額")
print("         │ 中位數    平均       中位數    平均      中位  平均")
for h in HOR:
    a, b = np.array(sig_r[h])*100, np.array(base_r[h])*100
    print(f"  {h:>3} 日  │ {np.median(a):>6.2f}% {a.mean():>7.2f}%   "
          f"{np.median(b):>6.2f}% {b.mean():>7.2f}%   "
          f"{np.median(a)-np.median(b):>+5.2f} {a.mean()-b.mean():>+6.2f}", flush=True)

print("\n=== 勝率（未來報酬 > 0 的比例）===")
print("未來期間 │ 訊號日   任意日   差")
for h in HOR:
    a, b = np.array(sig_r[h]), np.array(base_r[h])
    print(f"  {h:>3} 日  │ {(a>0).mean()*100:>5.1f}%  {(b>0).mean()*100:>5.1f}%  {((a>0).mean()-(b>0).mean())*100:>+5.1f}")

print("\n=== 右尾：未來報酬第 90 / 95 百分位 ===")
print("未來期間 │ 訊號日 P90   基準 P90 │ 訊號日 P95   基準 P95")
for h in HOR:
    a, b = np.array(sig_r[h])*100, np.array(base_r[h])*100
    print(f"  {h:>3} 日  │ {np.percentile(a,90):>8.1f}% {np.percentile(b,90):>9.1f}% │"
          f" {np.percentile(a,95):>8.1f}% {np.percentile(b,95):>9.1f}%")
