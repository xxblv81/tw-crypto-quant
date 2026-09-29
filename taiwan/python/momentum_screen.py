"""
動能選股 —— 產出今日動能分數排行。

用法：
    python momentum_screen.py                    # 預設門檻 7.5，全市場
    python momentum_screen.py --top 30           # 只看前 30 名
    python momentum_screen.py --min-score 8.5    # 提高門檻
    python momentum_screen.py --date 2026-08-15  # 回看某一天（不會用到未來資料）

輸出：momentum_YYYY-MM-DD.csv

★ 這是「雷達」不是下單清單。分數高只代表當天市場上動能相對最強的一批，
  不代表值得買 —— 動能股回檔起來也是最快的那批。
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import config
import momentum as mom
import twse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None,
                    help="動能分數下限，預設取 markets.yaml 的 entry_score")
    ap.add_argument("--top", type=int, default=40, help="最多輸出幾檔")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD，預設最後一個交易日")
    ap.add_argument("--include-otc", action="store_true",
                    help="★ 上櫃無除權息還原，動能分數會被除息假跌破扭曲")
    ap.add_argument("--futures", action="store_true",
                    help="個股期貨模式：只留有個股期且有量的標的，改列槓桿與保證金")
    ap.add_argument("--ssf-min-volume", type=float, default=None,
                    help="個股期契約日成交口數下限，預設取 markets.yaml 的 ssf_min_volume")
    ap.add_argument("--ignition", action="store_true",
                    help="只列今天出現布林縮口突破的（啟動觸發），需搭配 --futures 或單獨用")
    args = ap.parse_args()

    p = mom.normalize_params(config.params("taiwan_momentum"))
    min_score = args.min_score if args.min_score is not None else p["entry_score"]

    bars, names = twse.load_bars(include_otc=args.include_otc, adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    print(f"universe {panel['Close'].shape[1]} 檔"
          f"｜{panel['Close'].index[0]:%Y-%m-%d} ~ {panel['Close'].index[-1]:%Y-%m-%d}")

    streak = None
    if p.get("use_chips"):
        import chips
        streak = chips.buy_streak(chips.load_trust_net())
        cov = chips.last_date()
        need = panel["Close"].index[-1]
        if streak.empty:
            print("[warn] use_chips=true 但籌碼快取是空的 → 本次不計籌碼分。"
                  "先跑 python chips.py --start 2019-01-01")
            p, streak = {**p, "use_chips": False}, None
        elif cov is not None and cov < need:
            # ★ 籌碼落後於行情時，全市場當天籌碼分都是 0，總分上限被壓到 8.5，
            #   門檻 8.5 會篩出零檔 —— 這是靜默失效，一定要擋。
            print(f"[warn] 籌碼快取只到 {cov:%Y-%m-%d}，行情已到 {need:%Y-%m-%d}"
                  f" → 本次改用「不含籌碼」計分（分數會比平常高，不要跨日比較）。\n"
                  f"        補資料：./.venv/bin/python chips.py --start 2019-01-01")
            p, streak = {**p, "use_chips": False}, None

    # ── 大盤狀態：決定今天該用多少槓桿、要不要出手 ──
    import market
    rp = config.params("taiwan_market")
    _msk = mom.liquidity_mask(panel["Turnover"], p)
    _brd = market.breadth(panel["Close"], _msk, rp["breadth_ma"])
    regime = market.regime(market.load_index(), _brd, rp, panel["Close"].index)

    sc = mom.compute_scores(panel, p, streak)
    idx = panel["Close"].index
    day = idx[-1] if args.date is None else pd.Timestamp(args.date)
    if day not in idx:
        day = idx[idx <= day][-1]
        print(f"[note] 指定日非交易日，改用 {day:%Y-%m-%d}")

    # 大盤狀態要在選股結果之前印 —— 它決定「今天要不要出手」，比看誰上榜更優先
    _r = regime.loc[day]
    _nm = {0: "弱", 1: "中", 2: "強"}
    _fp = config.params("taiwan_futures")
    _lm = {int(k): float(v) for k, v in _fp["regime_leverage"].items()}
    _lv = int(_r["level"])
    print(f"\n━━ 大盤狀態：{_nm[_lv]} ━━")
    print(f"   寬度（全市場站上 MA{rp['breadth_ma']} 的比例）{_r['breadth']*100:.0f}%"
          f"　弱<{rp['breadth_weak']:.0%} 強≥{rp['breadth_strong']:.0%}")
    if pd.notna(_r["index_close"]):
        print(f"   加權指數 {_r['index_close']:,.0f}"
              f"（MA{rp['index_ma']} {_r['index_ma']:,.0f}，"
              f"{'站上' if _r['idx_above_ma'] else '★ 跌破'}"
              f"、均線{'上揚' if _r['idx_slope_up'] else '★ 走平或下彎'}）")
    else:
        print("   加權指數：無快取（python market.py --refresh），本次只用寬度判斷")
    if _lm[_lv] == 0:
        print(f"   → 建議槓桿 {_lm[_lv]:g}x　★★ 弱勢：實測此狀態下進場 PF 0.848（虧損），建議停手觀望")
    else:
        print(f"   → 建議槓桿 {_lm[_lv]:g}x（弱{_lm[0]:g} / 中{_lm[1]:g} / 強{_lm[2]:g}）")

    close = panel["Close"].loc[day]
    row = pd.DataFrame({
        "score": sc["score"].loc[day],
        "rs": sc["rs"].loc[day],
        "volume": sc["volume"].loc[day],
        "align": sc["align"].loc[day],
        "position": sc["position"].loc[day],
    })
    if "chips" in sc:
        row["chips"] = sc["chips"].loc[day]

    # ── 啟動觸發：今天是否剛出現布林縮口突破 ──
    #   動能分數看的是「已經在動」，縮口突破看的是「剛啟動」，兩件事。
    fired = pd.Series(False, index=row.index)
    if args.ignition or args.futures:
        import signals as sig_mod
        pt = config.params("taiwan")
        for code in row.index:
            if code not in bars:
                continue
            try:
                srow = sig_mod.compute(bars[code], pt)
            except Exception:
                continue
            if day in srow.index:
                fired[code] = bool(srow.loc[day, "signal"])
        row["啟動"] = np.where(fired.reindex(row.index).fillna(False), "★突破", "")

    if args.ignition:
        row = row[fired.reindex(row.index).fillna(False)]

    # ── 個股期貨模式：與可交易契約取交集 ──
    if args.futures:
        import futures as fut
        fp = config.params("taiwan_futures")
        minvol = args.ssf_min_volume if args.ssf_min_volume is not None else fp["ssf_min_volume"]
        ssf = fut.load(min_volume=minvol).set_index("code")
        keep = row.index.intersection(ssf.index)
        print(f"個股期可交易（日成交 ≥ {minvol:.0f} 口）{len(ssf)} 檔"
              f"　→　與動能榜取交集後 {len(keep)} 檔")
        row = row.loc[keep]
        # ★ ssf 也有 volume 欄，直接併會蓋掉動能的「量能分項」→ 改名
        for c, dst in (("Contract", "Contract"), ("leverage", "leverage"),
                       ("margin_pct", "margin_pct"), ("volume", "ssf_volume")):
            row[dst] = ssf[c].reindex(row.index).values

    row = row[row["score"] >= min_score].sort_values("score", ascending=False)
    if row.empty:
        print(f"{day:%Y-%m-%d} 無標的達 {min_score} 分")
        return

    ma_exit = sc["ma"][p["exit_ma"]].loc[day]
    chg = (panel["Close"].pct_change().loc[day] * 100)
    hi250 = panel["High"].rolling(p["high_window"]).max().loc[day]
    to20 = panel["Turnover"].rolling(p["turnover_window"]).mean().loc[day]

    out = row.head(args.top).round(2).reset_index().rename(columns={"index": "code"})
    out.insert(1, "name", out["code"].map(names).fillna(""))
    out["close"] = close.reindex(out["code"]).round(2).values
    out["chg_pct"] = chg.reindex(out["code"]).round(2).values
    out["vs_52w_high_pct"] = ((close / hi250 - 1) * 100).reindex(out["code"]).round(1).values
    out["avg_turnover_20d"] = to20.reindex(out["code"]).astype("int64").values
    # 一張的錢 —— 台股最小交易單位 1000 股，這決定你買不買得動（同 screen.py）
    out["lot_cost"] = (close.reindex(out["code"]) * 1000).round().astype("int64").values
    # 出場參考線。★ 均線用 markets.yaml 的 exit_ma（20），不是作者原本的 5 ——
    #   MA5 出場實測是虧的，理由見 taiwan/MOMENTUM_RESULTS.md 第二節。
    out[f"exit_ma{p['exit_ma']}"] = ma_exit.reindex(out["code"]).round(2).values
    out["stop_20pct"] = (close.reindex(out["code"]) * (1 - p["stop_loss_pct"] / 100)).round(2).values

    if args.futures:
        cs = config.params("taiwan_futures")["contract_size"]
        px = close.reindex(out["code"])
        out["契約"] = out.pop("Contract")
        out["槓桿"] = out.pop("leverage")
        # 一口要準備多少錢 —— 這決定你的資金能同時開幾個部位
        out["一口保證金"] = (px * cs * out["margin_pct"].values / 100).round().astype("int64").values
        out["一口名目"] = (px * cs).round().astype("int64").values
        out["期貨口數"] = out.pop("ssf_volume").astype("int64")
        # 持有上限對應的最晚出場日：3~14 天窗口的硬邊界，先算好省得自己數交易日
        mh = int(config.params("taiwan_futures").get("max_hold_days") or 0)
        if mh:
            fut_days = idx[idx > day]
            last = fut_days[mh - 1] if len(fut_days) >= mh else None
            out["最晚出場"] = (f"{last:%m-%d}" if last is not None
                            else f"進場後 {mh} 個交易日")
        out.pop("margin_pct")
        out.pop("lot_cost")          # 現股一張的錢，期貨模式用不到

    fn = (f"momentum_futures_{day:%Y-%m-%d}.csv" if args.futures
          else f"momentum_{day:%Y-%m-%d}.csv")
    out.to_csv(fn, index=False, encoding="utf-8-sig")
    print(f"\n{day:%Y-%m-%d}｜{len(row)} 檔 ≥ {min_score} 分，輸出前 {len(out)} 檔 → {fn}")
    print(out.to_string(index=False))
    if args.futures:
        fpp = config.params("taiwan_futures")
        print(f"\n短線窗口：持有上限 {fpp['max_hold_days']} 個交易日"
              f"（實測 93% 的交易落在 3~14 天）"
              f"｜建議 {fpp['slots']} 檔等權"
              f"｜槓桿依大盤狀態動態調整，今天是 {_lm[_lv]:g}x"
              f"\n★ 級距1 滿倉 7.4x 實測回撤 98.8%＝歸零；"
              f"模擬未含追繳斷頭，2.5x 以上實際更差")
    print(f"\n出場規則：收盤跌破 exit_ma{p['exit_ma']} → 隔日開盤出場；"
          f"或虧損達 {p['stop_loss_pct']}% 硬停損。"
          f"\n（作者原規則是跌破 5 日線，實測在台股成本結構下是虧的 —— "
          f"見 taiwan/MOMENTUM_RESULTS.md）")


if __name__ == "__main__":
    main()
