"""
崩盤反彈 —— 今天有沒有觸發、觸發的話買哪幾檔。

大部分日子會顯示「未觸發」—— 這是正常的。回測 2019~2026 只有 6 次崩盤達到門檻，
約每年 0.8 次。它的角色是在動能策略停手（大盤弱勢）時接手，不是天天出手。

用法：python rebound_screen.py [--margin 300000]
"""
from __future__ import annotations

import argparse

import pandas as pd

import combo_backtest as cb
import config
import futures as fut
import momentum as mom
import rebound as rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=300_000, help="可用保證金預算")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    p = config.params("taiwan_rebound")
    panel, _, ic = cb.load()
    idx = panel["Close"].index
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    pick, score, ibias, ma = rb.signals(panel, ic, p, mom.liquidity_mask(panel["Turnover"], p0))
    day = idx[-1] if args.date is None else idx[idx <= pd.Timestamp(args.date)][-1]
    ib = float(ibias.loc[day])

    print(f"══ 崩盤反彈　{day:%Y-%m-%d} ══")
    print(f"加權指數 BIAS{p['index_bias_ma']} = {ib:+.2f}%　觸發門檻 ≤ {p['index_trigger']:g}%")
    recent = ibias.loc[:day].tail(60)
    print(f"近 60 日最低 {recent.min():+.2f}%（{recent.idxmin():%Y-%m-%d}）")
    if ib > p["index_trigger"]:
        print(f"\n→ 未觸發，距門檻還差 {ib - p['index_trigger']:.2f} 個百分點。今天沒有反彈單。")
        return

    codes = list(pick.loc[day][pick.loc[day]].index)
    ssf = fut.load(min_volume=0).set_index("code")
    C = panel["Close"].loc[day]
    rows = []
    for c in sorted(codes, key=lambda x: -float(score.loc[day, x])):
        px = float(C[c])
        cs = int(ssf.loc[c, "contract_size"]) if c in ssf.index else 2000
        mr = float(ssf.loc[c, "margin_pct"]) / 100 if c in ssf.index else float("nan")
        rows.append({"code": c, "契約": ssf.loc[c, "Contract"] if c in ssf.index else "（無個股期）",
                     "乖離%": round(-float(score.loc[day, c]), 1), "現價": px,
                     "一口名目": int(px * cs), "一口保證金": int(px * cs * mr) if mr == mr else None,
                     "停損價": round(px * (1 - p["stop_loss_pct"] / 100), 2)})
    print(f"\n★ 觸發！最超跌的 {len(rows)} 檔（隔日開盤進、持有 {p['hold_days']} 個交易日）：")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n建議：{p['slots']} 個部位、槓桿 {p['leverage']:g}x（保證金預算 {args.margin:,.0f}）")
    print("★ 回測只有 6 次崩盤樣本，全部賺錢但樣本極薄；慢性空頭（如 2022）效果會差很多。")


if __name__ == "__main__":
    main()
