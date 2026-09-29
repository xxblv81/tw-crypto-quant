"""
全市場回測 — 這才是回答「這套策略在中小型股有沒有用」的工具。

對 universe 裡每一檔跑完整回測，輸出每檔的 PF / 勝率 / 交易數，
再依市值（用成交金額近似）分組，看「中小型股的中位數 PF」是否高於大型股。

★ 目前所有正面證據都來自 9 檔大型權值股。中小型股完全未經測試。
  跑完這支才知道 stop_atr=4.0 的結論能不能外推。

用法：
    python backtest.py --group-by turnover --buckets 4
輸出：backtest_results.csv, backtest_summary.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

import config
import data as data_mod
import signals as sig
from universe import fetch_universe, yf_ticker


def run_one(df: pd.DataFrame, p: dict, cost_pct: float,
            limit_up_guard: bool = True, limit_pct: float = 0.095) -> dict:
    """
    單檔回測：訊號隔日開盤進場，ATR 停損 / 風報比停利。

    limit_up_guard（P1-3）：台股漲跌幅 ±10%。若隔日「全天最低價」都貼在漲停附近，
    代表整天鎖死買不到，這筆單實務上不存在 —— 照樣成交會高估績效。
    判定用比例（low / 前一日收盤），不受除權息還原影響：還原是等比例縮放，
    當日內與跨日的漲跌幅比例都保持不變。
    """
    s = sig.compute(df, p)
    o, h, l, c = s["Open"], s["High"], s["Low"], s["Close"]
    n = len(s)
    # 出場用均線（選配）：收盤跌破就出場，是「讓贏家跑」的經典做法
    exit_ma_len = int(p.get("exit_ma", 0) or 0)
    exit_ma = c.rolling(exit_ma_len).mean().to_numpy() if exit_ma_len else None
    trail_on = bool(p.get("trailing", False))
    trades: list[float] = []
    skipped_limit_up = 0
    n_long = n_short = 0
    # 出場原因統計：停損 / 停利 / 到期未出場（用最後收盤價結算）
    n_stop = n_target = n_openend = n_timeout = 0
    hold_bars: list[int] = []          # 每筆持有幾根 K 棒（日線＝交易日）
    has_short = "signal_short" in s.columns
    i = 0
    while i < n - 1:
        is_long = bool(s["signal"].iloc[i])
        is_short = has_short and bool(s["signal_short"].iloc[i])
        if (not is_long and not is_short) or not np.isfinite(s["atr"].iloc[i]):
            i += 1
            continue
        if is_long and is_short:            # 同根多空同時觸發，方向矛盾，跳過
            i += 1
            continue
        # 漲停鎖死只擋做多（買不到）；做空無此問題，加密貨幣也無漲跌幅
        if is_long and limit_up_guard:
            prev_close = float(c.iloc[i])
            if prev_close > 0 and float(l.iloc[i + 1]) >= prev_close * (1 + limit_pct):
                skipped_limit_up += 1
                i += 1
                continue
        entry = float(o.iloc[i + 1])                      # 隔根開盤進場（貼近實務）
        a = float(s["atr"].iloc[i])
        ret = None
        j = i + 1
        if is_long:
            stop = entry - p["stop_atr"] * a
            target = entry + p["stop_atr"] * a * p["rr"] if p["take_profit"] else np.inf
            # 時間出場（選配）：抱滿 N 根就用收盤價出場，不管有沒有到停損停利。
            # 這是唯一能「直接」控制持有天數的方式 —— stop_atr 只能間接影響。
            max_hold = int(p.get("max_hold_bars", 0) or 0)
            deadline = (i + 1 + max_hold) if max_hold else n
            trig = entry + p.get("trail_trigger_atr", 2.0) * a   # 移動停利啟動價
            off = p.get("trail_offset_atr", 1.0) * a             # 回撤幅度
            cur_stop, peak = stop, entry
            while j < n:
                lo_j, hi_j = float(l.iloc[j]), float(h.iloc[j])
                if lo_j <= cur_stop:                     # 保守：同根同時觸及先算停損
                    ret = cur_stop / entry - 1
                    break
                if p["take_profit"] and hi_j >= target:
                    ret = target / entry - 1
                    break
                if exit_ma is not None and float(c.iloc[j]) < exit_ma[j]:
                    if p.get("exit_next_open", False) and j + 1 < n:
                        j += 1                            # 跌破均線，下一根開盤出場（同 TV strategy.close）
                        ret = float(o.iloc[j]) / entry - 1
                    else:
                        ret = float(c.iloc[j]) / entry - 1    # 跌破均線，收盤出場
                    break
                if trail_on:                             # 停損往上棘輪，不往下
                    peak = max(peak, hi_j)
                    if peak >= trig:
                        cur_stop = max(cur_stop, peak - off)
                if j >= deadline:                        # 時間到，收盤出場
                    ret = float(c.iloc[j]) / entry - 1
                    n_timeout += 1
                    break
                j += 1
            if ret is None:
                ret = float(c.iloc[-1]) / entry - 1
                j = n - 1
                n_openend += 1
            elif j < deadline and ret < 0:
                n_stop += 1
            elif j < deadline:
                n_target += 1
            n_long += 1
        else:                                             # 做空：停損在上、停利在下
            stop = entry + p["stop_atr"] * a
            target = entry - p["stop_atr"] * a * p["rr"] if p["take_profit"] else 0.0
            while j < n:
                if float(h.iloc[j]) >= stop:              # 同樣保守，先算停損
                    ret = (entry - stop) / entry
                    break
                if float(l.iloc[j]) <= target:
                    ret = (entry - target) / entry
                    break
                j += 1
            if ret is None:
                ret = (entry - float(c.iloc[-1])) / entry
                j = n - 1
            n_short += 1
        trades.append(ret - 2 * cost_pct / 100)           # 來回成本
        hold_bars.append(j - i)                           # 進場到出場的根數
        i = j + 1                                         # 出場後才找下一個訊號

    if not trades:
        return {}
    tr = np.array(trades)
    skip_note = {"skipped_limit_up": skipped_limit_up,
                 "trades_long": n_long, "trades_short": n_short,
                 "exit_stop": n_stop, "exit_target": n_target,
                 "exit_openend": n_openend, "exit_timeout": n_timeout,
                 "median_hold_bars": int(np.median(hold_bars)) if hold_bars else 0,
                 "mean_hold_bars": round(float(np.mean(hold_bars)), 1) if hold_bars else 0.0}
    gp, gl = tr[tr > 0].sum(), -tr[tr < 0].sum()
    eq = np.cumprod(1 + tr)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)
    return {
        "trades": len(tr),
        "win_rate_pct": round(float((tr > 0).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else np.inf,
        "total_return_pct": round(float((eq[-1] - 1) * 100), 2),
        "max_dd_pct": round(dd, 2),
        "avg_win_pct": round(float(tr[tr > 0].mean() * 100), 2) if (tr > 0).any() else 0.0,
        "avg_loss_pct": round(float(tr[tr < 0].mean() * 100), 2) if (tr < 0).any() else 0.0,
        **skip_note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="taiwan")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--min-turnover", type=float, default=2e7)
    ap.add_argument("--buckets", type=int, default=4, help="依成交金額分幾組（近似市值分組）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="twse", choices=["twse", "yf"],
                    help="twse=證交所官方快取（預設）；yf=yfinance 備援")
    ap.add_argument("--include-otc", action="store_true",
                    help="★ 不建議：上櫃無除權息還原資料，混入會讓中小型組被系統性低估")
    ap.add_argument("--no-adjust", action="store_true", help="不做除權息還原（僅供對照）")
    ap.add_argument("--no-limit-up-guard", action="store_true",
                    help="關掉「隔日漲停鎖死買不到」的過濾（僅供對照，會高估績效）")
    ap.add_argument("--exclude-delisted", action="store_true",
                    help="★ 只用目前仍在市的標的 —— 會產生倖存者偏差、高估績效。僅供對照")
    args = ap.parse_args()

    p = config.params(args.market)
    cost = config.costs(args.market)["commission_pct_per_side"]
    print(f"參數 markets.yaml[{args.market}]：stop_atr={p['stop_atr']}, "
          f"volume_filter={p['volume_filter']}, cost={cost}%/邊")

    uni = fetch_universe(include_otc=args.include_otc)
    if args.limit:
        uni = uni.head(args.limit)

    if args.source == "twse":
        meta = {c: (c, n) for c, n in zip(uni["code"], uni["name"])}
        # ★ 倖存者偏差：uni 只有「目前仍在市」的公司。快取裡其實保留了期間內
        #   下市公司的行情，用 codes=None 全載才不會把失敗案例篩掉。
        codes = list(uni["code"]) if args.exclude_delisted else None
        bars, cache_names = data_mod.fetch_twse(codes=codes, include_otc=args.include_otc,
                                                adjust=not args.no_adjust)
        bars = {c: d for c, d in bars.items() if not c.startswith("00")}   # 濾掉 ETF
        for c, n in cache_names.items():
            meta.setdefault(c, (c, n))
        delisted = [c for c in bars if c not in set(uni["code"])]
        print(f"資料源：證交所快取"
              f"｜除權息還原：{'否 ★' if args.no_adjust else '是'}"
              f"｜上櫃：{'含' if args.include_otc else '不含 ★'}"
              f"｜已下市標的：{'排除 ★' if args.exclude_delisted else f'納入 {len(delisted)} 檔'}")
    else:
        tickers = [yf_ticker(cc, m) for cc, m in zip(uni["code"], uni["market"])]
        meta = {yf_ticker(cc, m): (cc, nn) for cc, nn, m in zip(uni["code"], uni["name"], uni["market"])}
        bars = data_mod.fetch_yf(tickers, period=args.period)
    print(f"取得 {len(bars)} 檔，開始回測…")

    rows = []
    for t, df in bars.items():
        turnover = data_mod.avg_turnover(df, 250)
        if turnover < args.min_turnover:
            continue
        try:
            r = run_one(df, p, cost, limit_up_guard=not args.no_limit_up_guard)
        except Exception:
            continue
        if not r or r["trades"] < args.min_trades:
            continue
        code, name = meta.get(t, (t, ""))
        rows.append({"code": code, "name": name, "avg_turnover": int(turnover), **r})

    res = pd.DataFrame(rows)
    if res.empty:
        print("沒有足夠樣本")
        return
    res = res.sort_values("profit_factor", ascending=False)
    res.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")

    res["size_bucket"] = pd.qcut(res["avg_turnover"], args.buckets,
                                 labels=[f"Q{i+1}" for i in range(args.buckets)])
    summ = res.groupby("size_bucket", observed=True).agg(
        n=("code", "count"),
        median_pf=("profit_factor", "median"),
        pct_above_1=("profit_factor", lambda x: round((x > 1).mean() * 100, 1)),
        median_trades=("trades", "median"),
        median_dd=("max_dd_pct", "median"),
    ).reset_index()
    summ.to_csv("backtest_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n樣本 {len(res)} 檔｜全體中位數 PF = {res['profit_factor'].median():.3f}"
          f"｜PF>1 佔 {(res['profit_factor'] > 1).mean()*100:.1f}%")
    if "skipped_limit_up" in res.columns:
        tot_sk, tot_tr = int(res["skipped_limit_up"].sum()), int(res["trades"].sum())
        if args.no_limit_up_guard:
            print(f"★ 漲停過濾已關閉 —— 這批數字含買不到的單，會高估績效")
        else:
            print(f"漲停鎖死買不到而略過：{tot_sk} 筆"
                  f"（佔訊號 {tot_sk / max(tot_sk + tot_tr, 1) * 100:.1f}%）")
    print("\n依成交金額分組（Q1 最小 = 中小型股，Q4 最大 = 權值股）")
    print(summ.to_string(index=False))
    print("\n→ backtest_results.csv / backtest_summary.csv")
    print("\n★ 判讀：若 Q1/Q2 的 median_pf 明顯高於 Q4，代表策略在中小型股更有效；")
    print("  若全部落在 1.0 附近，代表大型股那組 1.094 只是雜訊。")


if __name__ == "__main__":
    main()
