"""
大戶籌碼輪動選股 —— 每週看一次的清單。

找的是同時滿足兩件事的標的：
    1. 籌碼集中度在上升 —— >400 張大戶持股比例升、總股東人數減
    2. 成交量在放大     —— 5 日均額 / 60 日均額 > 門檻

為什麼「集中 + 量增」要一起看：
    只看集中度會抓到一堆冷門股 —— 沒人交易的股票股東人數自然慢慢減少，
    那是流動性枯竭不是有人收貨。
    只看量增則抓到的是已經在動的股票，那是動能策略的守備範圍。
    兩個一起看才是「有人正在收，而且開始有人注意到」。

★★ 這份清單**完全沒有回測過**。
   本專案的規矩是「先讓它跑出數字」——低位階籌碼清洗策略當初看起來也很合理，
   實測是每筆輸給等權買進持有 0.37%（見 taiwan/WASHOUT_RESULTS.md）。
   在同樣的超額報酬基準下驗過之前，把這份清單當**觀察名單**，不是買進清單。
   資料有 170 週，要回測是做得到的 —— 只是還沒做。

資料來源：
    籌碼  pyramid.py（神秘金字塔整理的 TDCC 集保週資料，2023-05 起約 170 週）
    行情  twse.py 快取（除權息還原）

用法：
    python chip_rotation.py                      # 用 markets.yaml 的門檻
    python chip_rotation.py --top 30
    python chip_rotation.py --min-score 7.0
    python chip_rotation.py --weeks 8            # 改看 8 週的集中度變化
    python chip_rotation.py --show-gates         # 每道門檻各刷掉多少
"""
from __future__ import annotations

import argparse
import pandas as pd

import config
import pyramid
import twse
from momentum import build_panel, liquidity_mask


def _rank(s: pd.Series, mask: pd.Series) -> pd.Series:
    """
    單日橫斷面百分位 → 0~10 分。

    ★ 這裡刻意不重用 momentum._pct_score。那支吃的是「日期 × 代號」的二維寬表，
      想拿它處理單一天的 Series 就得包成 pd.DataFrame([s]) ——
      而 pd.DataFrame([s]) 會拿 **s.name 當 index**（不是 0），
      跟遮罩的 index 對不上，where() 對齊後整列變 NaN。
      不會報錯，只會安靜地讓所有分數變成空的。實測踩過一次。
      單日就用單日的寫法，不要硬套二維的工具。
    """
    return s.where(mask).rank(pct=True) * 10


def weekly_features(p: dict) -> dict[str, pd.DataFrame]:
    """從 pyramid 快取算出各項週頻特徵（日期 × 代號）。"""
    n = int(p["lookback_weeks"])
    big400 = pyramid.load_wide("big400_pct")
    big1000 = pyramid.load_wide("big1000_pct")
    holders = pyramid.load_wide("holders")
    if big400.empty:
        return {}
    return {
        # 持股比例用**百分點差**：40%→42% 是 +2 個百分點（有意義），
        # 1%→1.5% 是 +50%（比率）但只有 +0.5 個百分點（多半是雜訊）。
        "big400_chg": big400 - big400.shift(n),
        "big1000_chg": big1000 - big1000.shift(n),
        # 股東人數改用**變化率**：人數的絕對值跨標的差幾個數量級，差值不可比較。
        # 取負號讓「人數減少」＝高分。
        "holders_chg": -(holders / holders.shift(n) - 1),
        "big400_pct": big400,
        "holders": holders,
        "avg_lots": pyramid.load_wide("avg_lots"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--weeks", type=int, default=None, help="集中度回看週數")
    ap.add_argument("--show-gates", action="store_true")
    args = ap.parse_args()

    p = dict(config.params("taiwan_chip_rotation"))
    if args.weeks:
        p["lookback_weeks"] = args.weeks
    thr = args.min_score if args.min_score is not None else p["entry_score"]
    w = p["weights"]
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError(f"weights 加總 {sum(w.values())}，應為 1.0")

    feat = weekly_features(p)
    if not feat:
        print("[error] 沒有籌碼快取 —— 先跑：python pyramid.py --build")
        return

    bars, names = twse.load_bars(adjust=True, min_rows=250)
    panel = build_panel(bars, fields=("Close", "High", "Turnover"))
    close, turnover = panel["Close"], panel["Turnover"]

    week = feat["big400_chg"].index.max()
    day = close.index[close.index <= week].max()
    if pd.isna(day):
        print("[error] 行情快取比籌碼資料還舊，先更新 twse.py")
        return
    # ★ 集保每週五更新，行情是日頻。用「不晚於該週次」的最後一個交易日對齊，
    #   不要用最新交易日 —— 否則等於用未來幾天的量能去配上週的籌碼。
    print(f"籌碼週次 {week:%Y-%m-%d}｜對齊行情日 {day:%Y-%m-%d}"
          f"｜回看 {p['lookback_weeks']} 週")
    if (close.index.max() - week).days > 12:
        print(f"[warn] 籌碼資料落後行情 {(close.index.max() - week).days} 天 —— "
              f"先跑 pyramid.py --build 更新")

    mask_d = liquidity_mask(turnover, p)
    liquid = mask_d.loc[day]
    codes = close.columns

    # 量能：5 日均額 / 60 日均額，取對齊日當天
    fast = turnover.rolling(p["vol_fast"] if "vol_fast" in p else 5, min_periods=5).mean()
    slow = turnover.rolling(60, min_periods=60).mean()
    vol_ratio = (fast / slow.replace(0, pd.NA)).loc[day]

    row = pd.DataFrame(index=codes)
    for k in ("big400_chg", "big1000_chg", "holders_chg"):
        row[k] = feat[k].loc[week].reindex(codes)
    row["vol_ratio"] = vol_ratio.reindex(codes)
    row["big400_pct"] = feat["big400_pct"].loc[week].reindex(codes)
    row["avg_lots"] = feat["avg_lots"].loc[week].reindex(codes)

    # ── 硬門檻 ──
    gates = {"流動性": liquid.reindex(codes).fillna(False)}
    if p["require_big400_up"]:
        gates["大戶比例上升"] = row["big400_chg"] > 0
    if p["require_holders_down"]:
        gates["股東人數減少"] = row["holders_chg"] > 0        # 已取負號
    gates["量能放大"] = row["vol_ratio"] >= p["vol_ratio_min"]
    # 公開收購／合併會讓比例單週跳數十個百分點 —— 真資料，但不是「有人在收貨」
    gates["非公司事件"] = row["big400_chg"].abs() <= p["max_abs_change"]

    if args.show_gates:
        print(f"\n各道門檻（全市場 {len(codes)} 檔）")
        keep = pd.Series(True, index=codes)
        for k, g in gates.items():
            g = g.reindex(codes).fillna(False)
            keep &= g
            print(f"  {k:>8} 單獨通過 {int(g.sum()):4d} 檔｜累計 {int(keep.sum()):4d} 檔")

    ok = pd.Series(True, index=codes)
    for g in gates.values():
        ok &= g.reindex(codes).fillna(False)

    # ── 分數：在「通過流動性」的池子裡做橫斷面百分位 ──
    #   ★ 分母用流動性池而非「通過所有門檻的池」：否則門檻一改，分數尺度就跟著變，
    #     跨設定不可比較。理由同 washout.py。
    lm = liquid.reindex(codes).fillna(False)
    parts = {k: _rank(row[k], lm) for k in ("big400_chg", "big1000_chg", "holders_chg")}
    parts["volume"] = _rank(row["vol_ratio"], lm)
    row["score"] = sum(parts[k] * w[k] for k in w)
    for k, v in parts.items():
        row[f"s_{k}"] = v

    # ★ 分數全空是「對齊出錯」的典型症狀，而且不會報錯 —— 寧可吵也不要給空清單。
    if not row["score"].notna().any():
        raise RuntimeError("所有分數皆為 NaN —— 多半是遮罩與資料的 index 沒對齊")

    out = row[ok & (row["score"] >= thr)].sort_values("score", ascending=False)
    if out.empty:
        print(f"\n{week:%Y-%m-%d}｜無標的達 {thr} 分")
        return

    hi250 = panel["High"].rolling(250, min_periods=250).max().loc[day]
    to20 = turnover.rolling(p["turnover_window"]).mean().loc[day]
    chg = (close.pct_change().loc[day] * 100)

    o = out.head(args.top).reset_index().rename(columns={"index": "code"})
    o.insert(1, "name", o["code"].map(names).fillna(""))
    o["close"] = close.loc[day].reindex(o["code"]).round(2).values
    o["chg_pct"] = chg.reindex(o["code"]).round(2).values
    o["vs_52w_high_pct"] = ((close.loc[day] / hi250 - 1) * 100).reindex(o["code"]).round(1).values
    o["avg_turnover_20d"] = to20.reindex(o["code"]).fillna(0).astype("int64").values
    o["lot_cost"] = (close.loc[day].reindex(o["code"]) * 1000).round().astype("int64").values

    cols = ["code", "name", "score", "big400_pct", "big400_chg", "big1000_chg",
            "holders_chg", "avg_lots", "vol_ratio", "close", "chg_pct",
            "vs_52w_high_pct", "avg_turnover_20d", "lot_cost"]
    o = o[cols].round({"score": 2, "big400_pct": 2, "big400_chg": 2, "big1000_chg": 2,
                       "holders_chg": 4, "avg_lots": 1, "vol_ratio": 2})
    o = o.rename(columns={"big400_chg": f"大戶400↑{p['lookback_weeks']}週",
                          "big1000_chg": f"千張↑{p['lookback_weeks']}週",
                          "holders_chg": "股東人數↓率"})

    fn = f"chip_rotation_{week:%Y-%m-%d}.csv"
    o.to_csv(fn, index=False, encoding="utf-8-sig")
    print(f"\n{week:%Y-%m-%d}｜{len(out)} 檔 ≥ {thr} 分，輸出前 {len(o)} 檔 → {fn}")
    print(o.to_string(index=False))
    print("\n★ 這份清單未經回測。集保每週五更新，週間看到的是上週五的狀態。"
          "\n  『大戶』含保管銀行帳戶，外資換託管行也會讓比例跳動 —— 不等於有人在收貨。")


if __name__ == "__main__":
    main()
