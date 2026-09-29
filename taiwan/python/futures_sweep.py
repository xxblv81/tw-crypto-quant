"""
股票期貨版短線掃描 —— 重算「4~14 天」在期貨成本下可不可行。

現股來回 0.585%（手續費 0.1425%x2 + 證交稅 0.3%）之前把短線邊際吃光。
股票期貨：期交稅 0.002%/邊 + 手續費約 NT$20-40/口（契約 2000 股）。
以股價 100 元計，一口契約 20 萬，來回成本約 0.05%，是現股的 1/12。

★ 但期貨有槓桿（保證金約 13.5%，約 7.4 倍），回撤會被同倍數放大。
  本掃描算的是「標的報酬」，不含槓桿。實際下單要自己乘上去。
"""
import config, data as d, backtest, numpy as np

p0 = config.params('taiwan')
COST_STOCK = 0.3      # %/邊，現股
COST_FUT = 0.025      # %/邊，股票期貨（含滑價保守估）

bars, names = d.fetch_twse(codes=None, include_otc=True, adjust=True)
bars = {c: v for c, v in bars.items() if not c.startswith('00')}
# 期貨只有流動性好的標的有掛牌，門檻拉高到日均 1 億
keep = {c: v for c, v in bars.items() if d.avg_turnover(v, 250) >= 1e8}
print(f"樣本 {len(keep)} 檔（日均成交 >1 億，近似有股期掛牌的標的）", flush=True)
print(f"現股來回 {COST_STOCK*2}%　vs　期貨來回 {COST_FUT*2}%\n", flush=True)

print("持有上限 stop │ 中位持有 │ 現股淨PF  期貨淨PF │ 期貨勝率 期貨中位報酬  交易數")
for mh in (4, 6, 8, 10, 14, 20):
    for sa in (1.5, 2.5):
        p = dict(p0, max_hold_bars=mh, stop_atr=sa)
        sp=[];fp=[];wr=[];rt=[];tr=[];hd=[]
        for c, df in keep.items():
            try:
                rs = backtest.run_one(df, p, COST_STOCK)
                rf = backtest.run_one(df, p, COST_FUT)
            except Exception:
                continue
            if rs and rs['trades'] >= 5:
                sp.append(rs['profit_factor']); fp.append(rf['profit_factor'])
                wr.append(rf['win_rate_pct']); rt.append(rf['total_return_pct'])
                tr.append(rf['trades']); hd.append(rf['median_hold_bars'])
        mark = "  ★" if np.median(fp) > 1.1 else ""
        print(f"  {mh:>2}天  {sa:<4} │  {np.median(hd):>4.0f}天  │  {np.median(sp):>6.3f}   {np.median(fp):>6.3f}  │"
              f"  {np.median(wr):>5.1f}%  {np.median(rt):>8.1f}%  {np.median(tr):>5.0f}{mark}", flush=True)
