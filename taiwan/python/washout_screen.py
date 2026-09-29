"""
低位階籌碼清洗選股 —— 產出今日候選清單。

用法：
    python washout_screen.py                     # 用 markets.yaml 的 entry_score
    python washout_screen.py --top 30
    python washout_screen.py --min-score 7.0     # 放寬（榜單會變長）
    python washout_screen.py --date 2026-08-15   # 回看某一天（不會用到未來資料）
    python washout_screen.py --show-gates        # 印出每道硬門檻各刷掉多少

輸出：washout_YYYY-MM-DD.csv

★ 這是「雷達」不是下單清單，而且比另外兩套更該這樣看：
  策略選的是「還沒發酵」的股票，本質上不知道要等多久、也可能永遠不發酵。
  清單只回答「誰同時具備低位階＋融資退潮＋法人吸籌＋營收成長」，
  不回答「什麼時候會動」。

★ 本專案自己的證據站在這套策略的反面（P0：低位階冷門股是全市場最差的一區）。
  在 taiwan/WASHOUT_RESULTS.md 跑出數字之前，把這份清單當成假設而不是結論。
"""
from __future__ import annotations

import argparse
import pandas as pd

import config
import twse
import washout as wo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None,
                    help="分數下限，預設取 markets.yaml 的 entry_score")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD，預設最後一個交易日")
    ap.add_argument("--show-gates", action="store_true", help="印出每道硬門檻的通過檔數")
    ap.add_argument("--relaxed", action="store_true",
                    help="拆掉「位階帶」與「未發酵」兩道門檻 —— 回測中唯一贏過買進持有的組合"
                         "（見 taiwan/WASHOUT_RESULTS.md）。★ 拆掉後選出來的就不是低位階股了")
    ap.add_argument("--ignore-gates", action="store_true",
                    help="★ 只看分數、不套硬門檻（診斷用，選出來的不是本策略的標的）")
    args = ap.parse_args()

    p = wo.normalize_params(config.params("taiwan_washout"))
    min_score = args.min_score if args.min_score is not None else p["entry_score"]

    bars, names = twse.load_bars(adjust=True, min_rows=300)
    panel, mb, trust, foreign, rev = wo.load_inputs(bars)
    print(f"universe {panel['Close'].shape[1]} 檔"
          f"｜{panel['Close'].index[0]:%Y-%m-%d} ~ {panel['Close'].index[-1]:%Y-%m-%d}")

    if args.relaxed:
        p = {**p, "disable_gates": ["posband", "dormant"]}
        print("[relaxed] 已拆掉 位階帶／未發酵 兩道門檻")
    sc = wo.compute_scores(panel, p, mb, trust, foreign, rev)
    idx = panel["Close"].index
    day = idx[-1] if args.date is None else pd.Timestamp(args.date)
    if day not in idx:
        day = idx[idx <= day][-1]
        print(f"[note] 指定日非交易日，改用 {day:%Y-%m-%d}")

    # ★ 融資／法人／營收快取的最後一天可能落後行情快取。落後時分項會是 NaN，
    #   清單會莫名其妙變短或失真 —— 寧可吵，也不要安靜地給錯清單。
    for label, src in (("融資", mb), ("法人", trust)):
        if src is not None and not src.empty and src.index.max() < day:
            print(f"[warn] {label}快取只到 {src.index.max():%Y-%m-%d}，"
                  f"落後行情 {day:%Y-%m-%d} —— 該分項今日為 NaN")

    score = sc["score_all"] if args.ignore_gates else sc["score"]
    row = pd.DataFrame({
        "score": score.loc[day],
        "washout": sc["washout"].loc[day],
        "accum": sc["accum"].loc[day],
        "fund": sc["fund"].loc[day],
        "base": sc["base"].loc[day],
    })

    if args.show_gates:
        m = sc["mask"].loc[day]
        print(f"\n{day:%Y-%m-%d} 各道門檻（分母＝通過流動性的 {int(m.sum())} 檔）")
        for k, g in sc["gates"].items():
            print(f"  {k:>8} 通過 {int((m & g.loc[day].fillna(False)).sum()):4d} 檔")
        print(f"  {'全部':>8} 通過 {int(sc['eligible'].loc[day].sum()):4d} 檔")

    row = row[row["score"] >= min_score].sort_values("score", ascending=False)
    if row.empty:
        print(f"\n{day:%Y-%m-%d} 無標的達 {min_score} 分"
              f"{'（已忽略硬門檻）' if args.ignore_gates else ''}")
        return

    close = panel["Close"].loc[day]
    hi250 = panel["High"].rolling(p["high_window"]).max().loc[day]
    to20 = panel["Turnover"].rolling(p["turnover_window"]).mean().loc[day]
    m_chg = sc["diag"].get("margin_chg")
    intens = sc["diag"].get("accum_intensity")

    out = row.head(args.top).round(2).reset_index().rename(columns={"index": "code"})
    out.insert(1, "name", out["code"].map(names).fillna(""))
    out["close"] = close.reindex(out["code"]).round(2).values
    # 位階：距一年高點多遠。這是策略的定義性欄位，放前面。
    out["vs_52w_high_pct"] = ((close / hi250 - 1) * 100).reindex(out["code"]).round(1).values
    out["pos"] = sc["pos"].loc[day].reindex(out["code"]).round(3).values
    out["vol_ratio_5_60"] = sc["vol_ratio"].loc[day].reindex(out["code"]).round(2).values
    if m_chg is not None:
        out[f"margin_chg_{p['margin_window']}d_pct"] = (
            m_chg.loc[day].reindex(out["code"]) * 100).round(1).values
    if intens is not None:
        out[f"inst_buy_share_{p['accum_window']}d_pct"] = (
            intens.loc[day].reindex(out["code"]) * 100).round(2).values
    if rev is not None:
        out["rev_yoy_3m_avg_pct"] = (
            rev["yoy_avg"].loc[day].reindex(out["code"]) * 100).round(1).values
    out["avg_turnover_20d"] = to20.reindex(out["code"]).fillna(0).astype("int64").values
    out["lot_cost"] = (close.reindex(out["code"]) * 1000).round().astype("int64").values
    out["stop"] = (close.reindex(out["code"]) * (1 - p["stop_loss_pct"] / 100)).round(2).values

    fn = f"washout_{day:%Y-%m-%d}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")
    print(f"\n{day:%Y-%m-%d}｜{len(row)} 檔 ≥ {min_score} 分，輸出前 {len(out)} 檔 → {fn}")
    print(out.to_string(index=False))
    print(f"\n出場規則（markets.yaml）：exit_mode={p['exit_mode']}"
          f"｜獲利達 {p.get('arm_pct', 0) * 100:.0f}% 後啟用跌破 MA{p['exit_ma']} 出場"
          f"｜硬停損 −{p['stop_loss_pct']}%｜最長持有 {p['max_hold_days']} 個交易日")


if __name__ == "__main__":
    main()
