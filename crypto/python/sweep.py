"""
加密貨幣掃描器 —— 週期掃描 + stop_atr × rr 敏感度。

訊號與出場邏輯**直接 import taiwan/python 的實作**，不另外複製一份。
理由：兩個市場用的是同一套進場邏輯，只有參數不同（見 markets.yaml）。
複製一份遲早會漂移，漂移之後兩邊算出來的東西就不能互相比較了。

★ 週期比較的公平性問題（重要）：
    策略的參數全部是「幾根 K 棒」，不是「多少時間」。
    trend_ma_length=200 在 4H 是約 33 天的趨勢過濾，在 15m 只剩約 2 天。
    直接拿 200 根去跑不同週期，比的其實是六個不同的策略，不是同一個策略的六個週期。

    所以本工具提供兩種模式，兩個都跑才有意義：
      --mode timeframe              固定「根數」：策略定義原封不動
      --mode timeframe --normalize  固定「時間」：把根數依週期換算，維持約當 4H 的時間尺度

用法：
    python sweep.py --mode parity                  先對答案（必做）
    python sweep.py --mode timeframe
    python sweep.py --mode timeframe --normalize
    python sweep.py --mode grid
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

# 重用台股那邊的訊號與回測實作（單一事實來源，不複製）
TW_PY = Path(__file__).resolve().parents[2] / "taiwan" / "python"
sys.path.insert(0, str(TW_PY))
import config                     # noqa: E402
import signals as sig             # noqa: E402
from backtest import run_one      # noqa: E402

import binance_data               # noqa: E402

# 已驗證基準（crypto/PARAMS.md、markets.yaml benchmark 區塊）
BENCHMARK = {"timeframe": "4h", "profit_factor": 1.713, "trades": 48,
             "win_rate_pct": 50.00, "net_profit_pct": 60.20, "max_drawdown_pct": 11.86}

# 各週期一根等於幾分鐘（用來做時間尺度換算）
MINUTES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}

BAR_PARAMS = ["bbw_lookback", "squeeze_window", "trend_ma_length",
              "volume_ma_length", "atr_length", "touch_window"]


def scaled_params(base: dict, interval: str, normalize: bool) -> dict:
    """normalize=True 時，把「根數」參數依週期換算，維持與 4H 相同的時間長度。"""
    p = dict(base)
    if not normalize or interval == "4h":
        return p
    ratio = MINUTES["4h"] / MINUTES[interval]
    for k in BAR_PARAMS:
        if k in p:
            p[k] = max(2, int(round(p[k] * ratio)))
    return p


def run(df: pd.DataFrame, p: dict, cost: float) -> dict:
    # 加密貨幣沒有漲跌幅限制，關掉台股專用的漲停過濾
    return run_one(df, p, cost, limit_up_guard=False)


def _fmt(r: dict) -> str:
    if not r:
        return "無交易"
    return (f"PF {r['profit_factor']:>6.3f}｜{r['trades']:>4} 筆"
            f"｜勝率 {r['win_rate_pct']:>5.1f}%｜報酬 {r['total_return_pct']:>8.2f}%"
            f"｜回撤 {r['max_dd_pct']:>5.1f}%")


def mode_parity(args, base_p, cost) -> None:
    print("=== 對答案：Python 版 vs TradingView 已驗證基準 ===")
    print(f"基準（crypto/PARAMS.md）：PF {BENCHMARK['profit_factor']}、{BENCHMARK['trades']} 筆、"
          f"勝率 {BENCHMARK['win_rate_pct']}%、報酬 {BENCHMARK['net_profit_pct']}%\n")
    df = binance_data.load(args.symbol, "4h", args.start, args.end)
    r = run(df, base_p, cost)
    print(f"Python  ：{_fmt(r)}")
    print(f"K 棒數  ：{len(df):,}")
    print()
    print("兩邊不會完全相同，已知的三個差異：")
    print("  1. Pine 用進場當根的 ATR 算停損；Python 用訊號當根的 ATR（差一根）")
    print("  2. Pine 的 slippage=1 tick；Python 只扣手續費，沒有滑價")
    print("  3. 同一根同時觸及停損與停利時，Python 一律算停損（保守），Pine 由 TV 撮合順序決定")
    print()
    if r:
        d_pf = abs(r["profit_factor"] - BENCHMARK["profit_factor"]) / BENCHMARK["profit_factor"]
        d_tr = abs(r["trades"] - BENCHMARK["trades"]) / BENCHMARK["trades"]
        print(f"落差：PF {d_pf*100:.1f}%、交易數 {d_tr*100:.1f}%")
        if d_pf < 0.25 and d_tr < 0.25:
            print("→ 在可接受範圍，後面的掃描結果可信（看相對排序，不要看絕對值）")
        else:
            print("★ 落差偏大，掃描結果只能看相對趨勢，絕對值不要拿去做決策")


def mode_timeframe(args, base_p, cost) -> None:
    tag = "固定時間尺度（參數依週期換算）" if args.normalize else "固定根數（策略定義原封不動）"
    print(f"=== 週期掃描 — {tag} ===")
    print(f"{args.symbol}  {args.start} ~ {args.end}  手續費 {cost}%/邊\n")

    rows = []
    for iv in args.intervals.split(","):
        iv = iv.strip()
        df = binance_data.load(args.symbol, iv, args.start, args.end, verbose=False)
        if df.empty:
            print(f"{iv:>4}  無資料")
            continue
        p = scaled_params(base_p, iv, args.normalize)
        r = run(df, p, cost)
        extra = f"  (MA{p['trend_ma_length']}, bbw{p['bbw_lookback']})" if args.normalize else ""
        print(f"{iv:>4}  {_fmt(r)}{extra}")
        if r:
            rows.append({"interval": iv, "bars": len(df), **r,
                         "trend_ma": p["trend_ma_length"], "bbw_lookback": p["bbw_lookback"]})

    if rows:
        out = pd.DataFrame(rows)
        fn = f"sweep_timeframe_{args.symbol}{'_normalized' if args.normalize else ''}.csv"
        out.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"\n→ {fn}")
        best = out.loc[out["profit_factor"].idxmax()]
        print(f"最高 PF：{best['interval']}（PF {best['profit_factor']:.3f}、{int(best['trades'])} 筆）")
        thin = out[out["trades"] < 30]
        if not thin.empty:
            print(f"★ 樣本不足 30 筆的週期：{', '.join(thin['interval'])} —— 這幾格的 PF 不要當真")


def mode_grid(args, base_p, cost) -> None:
    print(f"=== stop_atr × rr 敏感度（{args.grid_interval}）===")
    print(f"{args.symbol}  {args.start} ~ {args.end}\n")
    df = binance_data.load(args.symbol, args.grid_interval, args.start, args.end, verbose=False)
    if df.empty:
        print("無資料")
        return

    stops = [float(x) for x in args.stops.split(",")]
    rrs = [float(x) for x in args.rrs.split(",")]

    rows = []
    for s in stops:
        for rr in rrs:
            p = dict(base_p, stop_atr=s, rr=rr)
            r = run(df, p, cost)
            if r:
                rows.append({"stop_atr": s, "rr": rr, **r})

    if not rows:
        print("無交易")
        return
    out = pd.DataFrame(rows)
    grid_fn = f"sweep_stop_rr_{args.symbol}_{args.grid_interval}.csv"
    out.to_csv(grid_fn, index=False, encoding="utf-8-sig")

    for metric, label in (("profit_factor", "獲利因子 PF"), ("trades", "交易數"),
                          ("max_dd_pct", "最大回撤 %")):
        piv = out.pivot(index="stop_atr", columns="rr", values=metric)
        print(f"── {label}（列=stop_atr，欄=rr）")
        print(piv.round(3).to_string())
        print()

    cur = out[(out.stop_atr == base_p["stop_atr"]) & (out.rr == base_p["rr"])]
    if not cur.empty:
        c = cur.iloc[0]
        print(f"目前設定 stop_atr={base_p['stop_atr']} rr={base_p['rr']}："
              f"PF {c['profit_factor']:.3f}、{int(c['trades'])} 筆")
    best = out.loc[out["profit_factor"].idxmax()]
    print(f"最高 PF：stop_atr={best['stop_atr']} rr={best['rr']} "
          f"→ PF {best['profit_factor']:.3f}、{int(best['trades'])} 筆")
    print("\n★ 判讀：看的是「有沒有一片穩定高原」，不是最高那一格。")
    print("  孤立高點是雜訊；周圍都好才是真的參數穩健區。")
    print(f"→ {grid_fn}")


def mode_filters(args, base_p, cost) -> None:
    """
    測兩個新濾網：第二層均線、做空。

    ★ 60「日」均線的單位問題：策略參數都是「根數」。
      4H 圖上 60 根只等於 10 天，要真的等於 60 天得用 360 根。兩個都測。
    """
    print(f"=== 濾網測試（{args.grid_interval}）===")
    print(f"{args.symbol}  {args.start} ~ {args.end}  手續費 {cost}%/邊\n")
    df = binance_data.load(args.symbol, args.grid_interval, args.start, args.end, verbose=False)
    if df.empty:
        print("無資料"); return

    bars_per_day = 1440 // MINUTES[args.grid_interval]
    ma60d = 60 * bars_per_day                      # 60 個「日」換算成根數

    variants = [
        ("① 現行（僅 MA200 多單）",          dict(base_p)),
        (f"② + MA60 根（≈{60/bars_per_day:.1f} 天）", dict(base_p, trend_filter_2=True, trend_ma2_length=60)),
        (f"③ + MA{ma60d} 根（＝60 天）",      dict(base_p, trend_filter_2=True, trend_ma2_length=ma60d)),
        ("④ 現行 + 做空",                    dict(base_p, allow_short=True)),
        (f"⑤ MA60 根 + 做空",                dict(base_p, trend_filter_2=True, trend_ma2_length=60, allow_short=True)),
        (f"⑥ MA{ma60d} 根 + 做空",           dict(base_p, trend_filter_2=True, trend_ma2_length=ma60d, allow_short=True)),
        ("⑦ 純做空（現行濾網）",              dict(base_p, allow_short=True, allow_long=False)),
        (f"⑧ 純做空 + MA{ma60d} 根",         dict(base_p, allow_short=True, allow_long=False,
                                                 trend_filter_2=True, trend_ma2_length=ma60d)),
    ]

    rows = []
    for label, p in variants:
        r = run(df, p, cost)
        if not r:
            print(f"{label:<26} 無交易"); continue
        print(f"{label:<26} PF {r['profit_factor']:>6.3f}｜{r['trades']:>3} 筆"
              f"（多 {r['trades_long']:>3}／空 {r['trades_short']:>3}）"
              f"｜勝率 {r['win_rate_pct']:>5.1f}%｜報酬 {r['total_return_pct']:>8.2f}%"
              f"｜回撤 {r['max_dd_pct']:>5.1f}%")
        rows.append({"variant": label, **r})

    if rows:
        fn = f"filters_{args.symbol}_{args.grid_interval}.csv"
        pd.DataFrame(rows).to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"\n→ {fn}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parity", "timeframe", "grid", "filters"], required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--intervals", default="15m,1h,2h,4h,6h,1d")
    ap.add_argument("--normalize", action="store_true",
                    help="把根數參數依週期換算，維持約當 4H 的時間尺度")
    ap.add_argument("--grid-interval", default="4h")
    ap.add_argument("--stops", default="1.0,1.5,2.0,2.5,3.0,4.0")
    ap.add_argument("--rrs", default="1.0,1.5,2.0,2.5,3.0")
    args = ap.parse_args()

    args.start = dt.date.fromisoformat(args.start)
    args.end = dt.date.fromisoformat(args.end)

    base_p = config.params("crypto")
    cost = config.costs("crypto")["commission_pct_per_side"]

    {"parity": mode_parity, "timeframe": mode_timeframe, "grid": mode_grid,
     "filters": mode_filters}[args.mode](
        args, base_p, cost)


if __name__ == "__main__":
    main()
