"""
低位階籌碼清洗回測 —— 在下任何結論前，先讓它跑出數字。

與 momentum_backtest.py 的差別只有一處：**出場**。
動能策略的出場（跌破 MA）在這裡直接套會壞掉 —— 打底期的價格本來就在均線
上下震盪，一買進就會被洗出去。所以這裡把出場做成三個模式，掃過再定：

    time      到期就賣。最乾淨的基準，回答「等下去到底有沒有用」，
              也是唯一不含任何事後選擇的一種。
    ma        收盤跌破 MA_n → 隔日開盤賣。預期會被打底期的震盪洗掉，當對照組。
    armed_ma  獲利先達 arm_pct 才啟用「跌破 MA_n」出場。
              貼合策略意圖（等發酵、發酵後才管理），但**最像事後配適**，
              要對照另外兩組才知道改善是真的還是挑出來的。

停損一律先判（保守），漲停買不到的處理與其餘台股回測一致。

用法：
    python washout_backtest.py                          # 用 markets.yaml 的設定
    python washout_backtest.py --exit-mode time --max-hold 120
    python washout_backtest.py --sweep-exit             # 三種出場一次掃完
    python washout_backtest.py --slots 10 --min-score 8.5
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import config
import twse
import washout as wo
from momentum_backtest import pooled_stats, simulate_portfolio


def extract_trades(panel: dict[str, pd.DataFrame], score: pd.DataFrame,
                   ma_exit: pd.DataFrame, p: dict, thr: float,
                   limit_up_guard: bool, limit_pct: float = 0.095) -> pd.DataFrame:
    """對每一檔掃出所有交易。逐筆明細含進場分數，投組模擬要用它排序。"""
    idx = panel["Close"].index
    mode = p["exit_mode"]
    max_hold = int(p.get("max_hold_days") or 0)
    if mode == "time" and not max_hold:
        raise ValueError("exit_mode=time 必須設 max_hold_days")
    stop_frac = 1 - p["stop_loss_pct"] / 100.0
    arm = 1 + float(p.get("arm_pct", 0))
    rows = []

    for code in score.columns:
        sc = score[code].to_numpy(dtype=float)
        o = panel["Open"][code].to_numpy(dtype=float)
        h = panel["High"][code].to_numpy(dtype=float)
        l = panel["Low"][code].to_numpy(dtype=float)
        c = panel["Close"][code].to_numpy(dtype=float)
        ma = ma_exit[code].to_numpy(dtype=float)

        # 面板是全市場外連結，個股上市前／下市後是 NaN。不限制掃描範圍會算出 NaN 報酬。
        valid = np.flatnonzero(np.isfinite(c))
        if valid.size < 2:
            continue
        lo, hi = int(valid[0]), int(valid[-1])

        # ★ 進場語意與 momentum_backtest 刻意不同 —— 這裡是「**首次**達標」不是「上穿」。
        #   動能分數幾乎每天都算得出來，NaN 只出現在歷史開頭，所以那邊用
        #   「昨天在門檻下、今天站上」沒問題。
        #   本策略的分數只在通過四道硬門檻時才有值，一檔股票是「今天才進入
        #   低位階＋未發酵＋營收成長這個狀態」的，昨天必然是 NaN。
        #   若沿用上穿語意（NaN 不算在門檻下），這種最典型的候選會全部被跳過，
        #   只剩「本來就合格但分數偏低、後來變高」的少數情形 —— 那不是要找的東西。
        ok = np.isfinite(sc) & (sc >= thr)
        prev_ok = np.roll(ok, 1)
        prev_ok[0] = False
        cross = ok & ~prev_ok

        i = lo
        while i < hi:
            if not cross[i] or not np.isfinite(o[i + 1]):
                i += 1
                continue
            if limit_up_guard and np.isfinite(c[i]) and c[i] > 0 \
                    and l[i + 1] >= c[i] * (1 + limit_pct):
                i += 1
                continue
            entry = o[i + 1]
            if not (entry > 0):
                i += 1
                continue

            stop = entry * stop_frac
            armed = (mode == "ma")                   # ma 模式一開始就啟用；armed_ma 要先漲夠
            exit_px, reason, j = None, "", i + 1
            while j <= hi:
                if np.isfinite(l[j]) and l[j] <= stop:
                    exit_px, reason = stop, "stop"
                    break
                if mode == "armed_ma" and not armed and np.isfinite(h[j]) \
                        and h[j] >= entry * arm:
                    armed = True
                if armed and j > i + 1 and np.isfinite(ma[j]) and np.isfinite(c[j]) \
                        and c[j] < ma[j] and j + 1 <= hi and np.isfinite(o[j + 1]):
                    exit_px, reason = o[j + 1], "ma"
                    j += 1
                    break
                if max_hold and (j - i) >= max_hold and j + 1 <= hi and np.isfinite(o[j + 1]):
                    exit_px, reason = o[j + 1], "time"
                    j += 1
                    break
                j += 1
            if exit_px is None or not np.isfinite(exit_px):   # 走到有效區間末端仍在場內
                j, exit_px, reason = hi, c[hi], "eod"
            if not (exit_px > 0):
                i += 1
                continue

            rows.append({"code": code, "entry_i": i + 1, "exit_i": j,
                         "entry_date": idx[i + 1], "exit_date": idx[j],
                         "entry": entry, "exit": exit_px, "score": sc[i],
                         "ret_gross": exit_px / entry - 1,
                         "hold_days": j - (i + 1), "reason": reason})
            i = j + 1
    return pd.DataFrame(rows)


def market_index(panel: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.Series:
    """
    流動性池的**等權買進持有指數**。

    ★ 這是本專案 P2 #6 留下的方法論缺口：stop_atr 掃描當初單調上升到 12 都不收斂，
      最後是靠「同批股票買進持有中位報酬 367%，策略最好只做到 150.9%」才判定
      策略其實在退化成買進持有。那次是事後補算的，這裡直接內建。
      沒有這條線，「持有期越長 PF 越高」根本無法分辨是策略有效還是單純吃到多頭。

    用等權而不是加權指數：策略是等權買 10 檔，對照組也該是等權，
    否則比到的是「大型股 vs 小型股」而不是「選股 vs 不選股」。
    """
    ret = panel["Close"].where(mask).pct_change()
    daily = ret.mean(axis=1, skipna=True).fillna(0.0)
    return (1 + daily).cumprod()


def add_benchmark(tr: pd.DataFrame, midx: pd.Series,
                  idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    每筆交易加上「同期間買進持有等權市場」的報酬與超額。
    ★ entry_i/exit_i 是對 idx 的位置索引，--start 之後 idx 會被切過，
      所以 midx 必須先 reindex 到同一個 idx 再取位置，不能直接用原始序列。
    """
    v = midx.reindex(idx).to_numpy()
    b = v[tr["exit_i"].to_numpy()] / v[tr["entry_i"].to_numpy()] - 1
    return tr.assign(bench_ret=b, excess=tr["ret_net"] - b)


def run_one(panel, sc, p, thr, guard, cost, slots, start=None, midx=None):
    """跑一組設定，回傳 (逐筆明細, pooled, portfolio)。"""
    ma_exit = sc["ma"][p["exit_ma"]]
    tr = extract_trades(panel, sc["score"], ma_exit, p, thr, guard)
    if tr.empty:
        return tr, None, None
    tr["ret_net"] = tr["ret_gross"] - 2 * cost / 100

    idx = panel["Close"].index
    if start:
        tr = tr[tr["entry_date"] >= pd.Timestamp(start)].reset_index(drop=True)
        if tr.empty:
            return tr, None, None
        s = int(np.searchsorted(idx, pd.Timestamp(start)))
        idx, tr = idx[s:], tr.assign(entry_i=tr["entry_i"] - s, exit_i=tr["exit_i"] - s)
    ps = pooled_stats(tr["ret_net"].to_numpy())
    _, port = simulate_portfolio(tr, idx, slots)
    if midx is not None:
        tr = add_benchmark(tr, midx, idx)
        ps["bench_avg_pct"] = round(float(tr["bench_ret"].mean() * 100), 3)
        ps["excess_avg_pct"] = round(float(tr["excess"].mean() * 100), 3)
        ps["beat_bench_pct"] = round(float((tr["excess"] > 0).mean() * 100), 1)
        mv = midx.reindex(idx)
        yrs = (idx[-1] - idx[0]).days / 365.25
        port["bench_cagr_pct"] = round(
            (float(mv.iloc[-1] / mv.iloc[0]) ** (1 / yrs) - 1) * 100, 2)
    return tr, ps, port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--slots", type=int, default=10, help="同時最多持有幾檔")
    ap.add_argument("--start", default=None, help="回測起日（分數仍用完整歷史計算）")
    ap.add_argument("--exit-mode", choices=["time", "ma", "armed_ma"], default=None)
    ap.add_argument("--exit-ma", type=int, default=None)
    ap.add_argument("--arm-pct", type=float, default=None)
    ap.add_argument("--max-hold", type=int, default=None)
    ap.add_argument("--stop-pct", type=float, default=None)
    ap.add_argument("--sweep-exit", action="store_true", help="三種出場模式一次掃完")
    ap.add_argument("--no-limit-up-guard", action="store_true")
    args = ap.parse_args()

    p = wo.normalize_params(config.params("taiwan_washout"))
    cost = config.costs("taiwan_washout")["commission_pct_per_side"]
    thr = args.min_score if args.min_score is not None else p["entry_score"]
    guard = p.get("limit_up_guard", True) and not args.no_limit_up_guard
    for key, val in (("exit_mode", args.exit_mode), ("exit_ma", args.exit_ma),
                     ("arm_pct", args.arm_pct), ("max_hold_days", args.max_hold),
                     ("stop_loss_pct", args.stop_pct)):
        if val is not None:
            p[key] = val
    p["ma_set"] = sorted(set(p["ma_set"]) | {p["exit_ma"]})

    bars, names = twse.load_bars(adjust=True, min_rows=300)
    panel, mb, trust, foreign, rev = wo.load_inputs(bars)
    sc = wo.compute_scores(panel, p, mb, trust, foreign, rev)

    el = sc["eligible"]
    print(f"universe {panel['Close'].shape[1]} 檔"
          f"｜{panel['Close'].index[0]:%Y-%m-%d} ~ {panel['Close'].index[-1]:%Y-%m-%d}"
          f"｜門檻 {thr} 分｜成本 {cost}%/邊｜漲停過濾 {'開' if guard else '關 ★'}")
    print(f"通過流動性 {int(sc['mask'].sum().sum()):,} 個(日,股)"
          f"｜再通過四道硬門檻 {int(el.sum().sum()):,} 個"
          f"（{el.sum().sum() / max(sc['mask'].sum().sum(), 1) * 100:.1f}%）")
    for k, g in sc["gates"].items():
        keep = (sc["mask"] & g.fillna(False)).sum().sum() / max(sc["mask"].sum().sum(), 1)
        print(f"    單獨看 {k:>8} 門檻通過率 {keep * 100:5.1f}%")

    combos = ([("time", 0), ("ma", 0), ("armed_ma", 0)] if args.sweep_exit
              else [(p["exit_mode"], 0)])
    for mode, _ in combos:
        q = {**p, "exit_mode": mode}
        if mode == "time" and not q.get("max_hold_days"):
            q["max_hold_days"] = 120
        tr, ps, port = run_one(panel, sc, q, thr, guard, cost, args.slots, args.start,
                               midx=market_index(panel, sc["mask"]))
        head = (f"\n══ exit_mode={mode}"
                f"｜MA{q['exit_ma']}｜arm {q.get('arm_pct')}｜停損 {q['stop_loss_pct']}%"
                f"｜上限 {q['max_hold_days']} 日 ══")
        print(head)
        if tr.empty:
            print("  沒有任何交易")
            continue
        print("  【1】逐筆彙總 " + "｜".join(f"{k}={v}" for k, v in ps.items()))
        print(f"      median_hold_days={tr['hold_days'].median():.0f}"
              f"｜出場原因 {tr['reason'].value_counts().to_dict()}")
        print(f"  【2】投組 {args.slots} 檔 " + "｜".join(f"{k}={v}" for k, v in port.items()))
        if not args.sweep_exit:
            tr.to_csv("washout_trades.csv", index=False, encoding="utf-8-sig")
            print("\n分年（逐筆彙總口徑）")
            yr = tr.groupby(tr["entry_date"].dt.year)["ret_net"].agg(
                n="size", win=lambda x: round((x > 0).mean() * 100, 1),
                avg=lambda x: round(x.mean() * 100, 2),
                pf=lambda x: round(x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9), 2))
            print(yr.to_string())
            print("\n→ washout_trades.csv")


if __name__ == "__main__":
    main()
