"""
台股參數掃描 —— 對全市場掃單一參數，看「整體分布」是否提升。

方法論（CLAUDE.md 守則 2）：判斷一個改動有沒有效，看的是同一批標的的
**中位數 PF 是否整體提升**，不是某幾檔變好。所以這支輸出的是分布統計
（中位數 PF、PF>1 佔比、各規模分組），不是最佳個股。

效率：行情只載入一次，之後在記憶體裡重跑不同參數。
      載入約 1 分鐘，每組參數約 40 秒。

用法：
    python tw_sweep.py --param trend_ma_length --values 20,60,120,200,240
    python tw_sweep.py --param stop_atr --values 2,3,4,5,6
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
import data as data_mod
from backtest import run_one
from universe import fetch_universe


def summarize(rows: list[dict], buckets: int = 4) -> dict:
    res = pd.DataFrame(rows)
    if res.empty:
        return {}
    res["bucket"] = pd.qcut(res["avg_turnover"], buckets,
                            labels=[f"Q{i+1}" for i in range(buckets)])
    med = res.groupby("bucket", observed=True)["profit_factor"].median()
    return {
        "n": len(res),
        "median_pf": round(float(res["profit_factor"].median()), 3),
        "pct_above_1": round(float((res["profit_factor"] > 1).mean() * 100), 1),
        "median_dd": round(float(res["max_dd_pct"].median()), 1),
        "median_trades": int(res["trades"].median()),
        "median_return": round(float(res["total_return_pct"].median()), 1),
        "median_win_rate": round(float(res["win_rate_pct"].median()), 1),
        **{f"PF_{b}": round(float(v), 3) for b, v in med.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, help="要掃的 markets.yaml 參數名")
    ap.add_argument("--values", required=True, help="逗號分隔的值")
    ap.add_argument("--market", default="taiwan")
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--min-turnover", type=float, default=2e7)
    ap.add_argument("--include-otc", action="store_true", default=True)
    ap.add_argument("--buckets", type=int, default=4)
    args = ap.parse_args()

    base = config.params(args.market)
    cost = config.costs(args.market)["commission_pct_per_side"]

    uni = fetch_universe(include_otc=args.include_otc)
    meta = {c: n for c, n in zip(uni["code"], uni["name"])}
    print(f"載入行情（含下市標的，一次載入重複使用）…", flush=True)
    bars, names = data_mod.fetch_twse(codes=None, include_otc=args.include_otc, adjust=True)
    bars = {c: d for c, d in bars.items() if not c.startswith("00")}
    print(f"  {len(bars)} 檔\n")

    # 先算好成交金額，避免每組重算
    turnover = {c: data_mod.avg_turnover(d, 250) for c, d in bars.items()}
    keep = {c: d for c, d in bars.items() if turnover[c] >= args.min_turnover}
    print(f"過 {args.min_turnover:.0e} 成交金額門檻：{len(keep)} 檔")
    print(f"掃描 {args.param}：{args.values}\n", flush=True)

    out = []
    for raw in args.values.split(","):
        v = float(raw) if "." in raw else int(raw)
        p = dict(base)
        p[args.param] = v
        rows = []
        for c, df in keep.items():
            try:
                r = run_one(df, p, cost)
            except Exception:
                continue
            if r and r["trades"] >= args.min_trades:
                rows.append({"code": c, "avg_turnover": turnover[c], **r})
        s = summarize(rows, args.buckets)
        s[args.param] = v
        out.append(s)
        bucket_str = "  ".join(f"{b} {s.get(f'PF_{b}', float('nan')):.3f}"
                               for b in [f"Q{i+1}" for i in range(args.buckets)])
        print(f"{args.param}={v:<5} 樣本 {s['n']:>3}｜PF {s['median_pf']:.3f}"
              f"｜PF>1 {s['pct_above_1']:>5.1f}%｜勝率 {s['median_win_rate']:>4.1f}%"
              f"｜報酬 {s['median_return']:>7.1f}%｜回撤 {s['median_dd']:>5.1f}%"
              f"｜交易 {s['median_trades']:>2}", flush=True)

    df = pd.DataFrame(out)
    cols = [args.param] + [c for c in df.columns if c != args.param]
    fn = f"tw_sweep_{args.param}.csv"
    df[cols].to_csv(fn, index=False, encoding="utf-8-sig")
    print(f"\n→ {fn}")
    best = df.loc[df["median_pf"].idxmax()]
    print(f"最高中位數 PF：{args.param}={best[args.param]} → {best['median_pf']:.3f}")
    print("\n★ 判讀：看的是有沒有一片平緩的高原，不是最高那一格。")
    print("  相鄰值差異很大＝在擬合雜訊；平緩單調＝真的結構性效果。")


if __name__ == "__main__":
    main()
