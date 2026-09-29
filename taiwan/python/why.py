"""為什麼這檔今天沒訊號 —— 逐條列出每個進場條件的實際值。"""
import sys, config, data as d, signals as sig, pandas as pd, numpy as np

code = sys.argv[1]; date = sys.argv[2] if len(sys.argv) > 2 else None
p = config.params('taiwan')
bars, names = d.fetch_twse(codes=[code], include_otc=True)
df = bars[code]
s = sig.compute(df, p)
i = s.index.get_loc(pd.Timestamp(date)) if date else len(s) - 1
row = s.iloc[i]
print(f"=== {code} {names.get(code,'')}  {s.index[i]:%Y-%m-%d} ===\n")

c, o = float(row['Close']), float(row['Open'])
basis, upper, lower = float(row['basis']), float(row['upper']), float(row['lower'])
bbw = float(row['bbw']); bs_sq = row['bars_since_squeeze']
vol = float(row['Volume']); v5 = float(s['Volume'].iloc[i-4:i+1].mean())
ma = float(row['trend_ma'])
cross = (s['Close'] > s['basis']) & (s['Close'].shift(1) <= s['basis'].shift(1))
bs_cross = sig.barssince(cross).iloc[i]

def chk(name, ok, detail):
    print(f"  {'✅' if ok else '❌'}  {name:<26} {detail}")

print(f"收盤 {c}｜開盤 {o}｜中線 {basis:.1f}｜上軌 {upper:.1f}｜帶寬 {bbw:.2f}%\n")
chk("① 近 20 根內縮口過", bool(row['bars_since_squeeze'] <= p['squeeze_window']) if pd.notna(bs_sq) else False,
    f"距上次縮口 {bs_sq if pd.notna(bs_sq) else '從未'} 根（需 ≤{p['squeeze_window']}）")
chk("② 帶寬正在擴張", bool(bbw > float(s['bbw'].iloc[i-1])), f"今 {bbw:.2f}% vs 昨 {float(s['bbw'].iloc[i-1]):.2f}%")
chk("③ 曾向上穿越中線", pd.notna(bs_cross) and bs_cross <= p.get('mid_cross_window',10),
    f"距上次穿越 {bs_cross if pd.notna(bs_cross) else '從未'} 根（需 {p.get('mid_confirm_min_gap',1)}~{p.get('mid_cross_window',10)}）")
chk("④ 收盤仍在中線之上", c > basis, f"收 {c} vs 中線 {basis:.1f}")
chk("⑤ 表態紅K（收>開）", c > o, f"收 {c} vs 開 {o}")
chk("⑥ 量 > 5日均量 ×1.2", vol > v5 * p['volume_mult'],
    f"今量 {vol/1000:,.0f} 張 vs 5日均 {v5/1000:,.0f} × {p['volume_mult']} = {v5*p['volume_mult']/1000:,.0f}")
chk("⑦ 收盤 > 趨勢均線", c > ma, f"收 {c} vs MA{p['trend_ma_length']} {ma:.1f}")
print(f"\n最終訊號: {'✅ 有' if bool(row['signal']) else '❌ 無'}  ｜待突破: {bool(row['watch'])}")
