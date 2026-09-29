"""
低位階籌碼清洗 —— 掃描與拆解。

三個問題，三種模式。順序不要顛倒：

  --thresholds   分數有沒有鑑別力？
                 門檻由低到高掃過去，看 PF / 勝率 / 平均報酬是不是**單調**上升。
                 這是動能策略當初唯一站得住腳的證據形式（見 MOMENTUM_RESULTS.md）：
                 單一格子的高 PF 可能是運氣，跨門檻單調就不是。
                 ★ 若低分組跟高分組差不多，代表分數只是在挑「符合狀態的股票」，
                   排名本身沒有資訊 —— 那這套策略就只剩門檻，沒有分數。

  --parts        哪一個分項在出力？
                 把權重全押在單一分項各跑一次。四項若表現接近，代表真正在作用
                 的是四道硬門檻（低位階、未發酵、止跌、營收正），不是分數組成。
                 這種情形下調權重是浪費時間，該做的是調門檻。

  --exits        出場怎麼設才對？
                 time / ma / armed_ma × 幾種持有上限。
                 ★ 本專案已有三次獨立證據顯示台股短持有必虧（見 CLAUDE.md），
                   所以這裡只掃 60 天以上。

用法：
    python washout_sweep.py --thresholds
    python washout_sweep.py --parts
    python washout_sweep.py --exits
    python washout_sweep.py --all --slots 10
"""
from __future__ import annotations

import argparse
import pandas as pd

import config
import twse
import washout as wo
from washout_backtest import market_index, run_one


def _row(tag: dict, tr, ps, port) -> dict:
    if ps is None:
        return {**tag, "trades": 0}
    return {**tag, "trades": ps["trades"], "win_pct": ps["win_rate_pct"],
            "pf": ps["profit_factor"], "avg_pct": ps["avg_ret_pct"],
            # ★ 同期間等權買進持有的對照，以及策略贏它的比例。
            #   「PF > 1」不等於有用 —— 多頭裡什麼都不做也是 PF > 1。
            "bench_avg_pct": ps.get("bench_avg_pct"),
            "excess_avg_pct": ps.get("excess_avg_pct"),
            "beat_bench_pct": ps.get("beat_bench_pct"),
            "median_pct": ps["median_ret_pct"],
            "median_hold": int(tr["hold_days"].median()),
            "pct_time_exit": round((tr["reason"] == "time").mean() * 100, 1),
            "cagr_pct": port["cagr_pct"], "max_dd_pct": port["max_dd_pct"],
            "taken": port["trades_taken"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", action="store_true")
    ap.add_argument("--parts", action="store_true")
    ap.add_argument("--exits", action="store_true")
    ap.add_argument("--gates", action="store_true", help="門檻消融：逐一拆掉看誰在扯後腿")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    if args.all:
        args.thresholds = args.parts = args.exits = args.gates = True
    if not any((args.thresholds, args.parts, args.exits, args.gates)):
        ap.error("至少指定一種：--thresholds / --parts / --exits / --gates / --all")

    p = wo.normalize_params(config.params("taiwan_washout"))
    cost = config.costs("taiwan_washout")["commission_pct_per_side"]
    guard = p.get("limit_up_guard", True)

    bars, _ = twse.load_bars(adjust=True, min_rows=300)
    panel, mb, trust, foreign, rev = wo.load_inputs(bars)
    base_sc = wo.compute_scores(panel, p, mb, trust, foreign, rev)
    midx = market_index(panel, base_sc["mask"])
    print(f"universe {panel['Close'].shape[1]} 檔"
          f"｜{panel['Close'].index[0]:%Y-%m-%d} ~ {panel['Close'].index[-1]:%Y-%m-%d}"
          f"｜投組 {args.slots} 檔｜成本 {cost}%/邊")

    def show(name: str, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        print(f"\n═══ {name} ═══")
        print(df.to_string(index=False))
        fn = f"washout_sweep_{name}.csv"
        df.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"→ {fn}")

    if args.thresholds:
        rows = []
        for thr in (5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0):
            tr, ps, port = run_one(panel, base_sc, p, thr, guard, cost,
                                   args.slots, args.start, midx=midx)
            rows.append(_row({"entry_score": thr}, tr, ps, port))
        show("thresholds", rows)

    if args.parts:
        rows = []
        for part in ("washout", "accum", "fund", "base"):
            q = {**p, "weights": {k: (1.0 if k == part else 0.0) for k in p["weights"]}}
            sc = wo.compute_scores(panel, q, mb, trust, foreign, rev, verbose=False)
            tr, ps, port = run_one(panel, sc, q, p["entry_score"], guard, cost,
                                   args.slots, args.start, midx=midx)
            rows.append(_row({"only_part": part}, tr, ps, port))
        # 對照組：四道硬門檻都在，但分數完全不用（門檻一過就進場）
        q = {**p, "weights": p["weights"]}
        tr, ps, port = run_one(panel, base_sc, q, 0.0, guard, cost, args.slots,
                               args.start, midx=midx)
        rows.append(_row({"only_part": "gates_only(門檻過就買)"}, tr, ps, port))
        show("parts", rows)

    if args.gates:
        # 每次拆掉一道門檻，與「四道全開」對照。判準是 excess_avg_pct 不是 pf ——
        # 多頭裡 pf > 1 太容易了，要看的是有沒有贏過同期間買進持有。
        rows = []
        for off in ([], ["posband"], ["stopfall"], ["dormant"], ["fund"],
                    ["posband", "dormant"], ["posband", "stopfall", "dormant"],
                    ["posband", "stopfall", "dormant", "fund"]):
            q2 = {**p, "disable_gates": off}
            sc = wo.compute_scores(panel, q2, mb, trust, foreign, rev, verbose=False)
            tr, ps, port = run_one(panel, sc, q2, p["entry_score"], guard, cost,
                                   args.slots, args.start, midx=midx)
            rows.append(_row({"disabled": "+".join(off) or "(四道全開)"}, tr, ps, port))
        show("gates", rows)

    if args.exits:
        rows = []
        for mode in ("time", "ma", "armed_ma"):
            for hold in (60, 120, 250):
                q = {**p, "exit_mode": mode, "max_hold_days": hold}
                tr, ps, port = run_one(panel, base_sc, q, p["entry_score"], guard, cost,
                                       args.slots, args.start, midx=midx)
                rows.append(_row({"exit_mode": mode, "max_hold": hold}, tr, ps, port))
        show("exits", rows)


if __name__ == "__main__":
    main()
