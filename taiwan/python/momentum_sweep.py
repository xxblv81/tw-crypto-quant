"""
動能策略的結構掃描 —— 出場均線 × 進場門檻。

為什麼掃這兩個而不是逐檔調參：
    首跑結果 PF 0.947、中位持有 4 天、平均每筆 −0.148%，
    但來回成本就佔 0.6% —— 也就是毛邊際其實是正的，被成本吃光。
    「出場太緊 → 換手太高 → 成本吃掉邊際」是首要嫌疑，所以掃出場均線。
    符合工作守則 #2：看的是同一批標的的整體分布，不是挑贏家。

gross 欄是「零成本」的同一批交易，用來把「訊號有沒有邊際」
和「成本吃掉多少」分開看。實務上不可能零成本，那一欄只做歸因用。

用法：python momentum_sweep.py
輸出：momentum_sweep.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import momentum as mom
import momentum_backtest as mb
import twse

EXIT_MAS = [5, 10, 20, 60]
THRESHOLDS = [7.5, 8.5, 9.0]
SLOTS = 8


def main() -> None:
    p0 = mom.normalize_params(config.params("taiwan_momentum"))
    cost = config.costs("taiwan_momentum")["commission_pct_per_side"]

    bars, _ = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    p_all = {**p0, "ma_set": sorted(set(p0["ma_set"]) | set(EXIT_MAS))}
    sc = mom.compute_scores(panel, p_all, None)
    idx = panel["Close"].index
    print(f"universe {panel['Close'].shape[1]} 檔｜{idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}"
          f"｜成本 {cost}%/邊")

    rows = []
    for ema in EXIT_MAS:
        for thr in THRESHOLDS:
            p = {**p_all, "exit_ma": ema}
            tr = mb.extract_trades(panel, sc["score"], sc["ma"][ema], p, thr,
                                   limit_up_guard=True)
            if tr.empty:
                continue
            g = tr["ret_gross"].to_numpy()
            tr["ret_net"] = g - 2 * cost / 100
            net = tr["ret_net"].to_numpy()
            _, port = mb.simulate_portfolio(tr, idx, SLOTS)
            rows.append({
                "exit_ma": ema, "min_score": thr, "trades": len(tr),
                "median_hold": int(tr["hold_days"].median()),
                "win_pct": round(float((net > 0).mean() * 100), 1),
                "pf_gross": round(float(g[g > 0].sum() / max(-g[g < 0].sum(), 1e-9)), 3),
                "pf_net": round(float(net[net > 0].sum() / max(-net[net < 0].sum(), 1e-9)), 3),
                "avg_net_pct": round(float(net.mean() * 100), 3),
                "port_total_pct": port["total_return_pct"],
                "port_cagr_pct": port["cagr_pct"],
                "port_dd_pct": port["max_dd_pct"],
                "port_taken": port["trades_taken"],
            })
            print(f"  MA{ema:<2} thr {thr}  →  pf_net {rows[-1]['pf_net']:.3f}"
                  f"  (gross {rows[-1]['pf_gross']:.3f})"
                  f"  hold {rows[-1]['median_hold']}d"
                  f"  投組 CAGR {rows[-1]['port_cagr_pct']}%", flush=True)

    out = pd.DataFrame(rows).sort_values("pf_net", ascending=False)
    out.to_csv("momentum_sweep.csv", index=False, encoding="utf-8-sig")
    print("\n" + out.to_string(index=False))
    print("\n→ momentum_sweep.csv")


if __name__ == "__main__":
    main()
