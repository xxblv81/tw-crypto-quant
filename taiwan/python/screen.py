"""
台股全市場日線選股器 — 產出今日出現「突破」或「待突破」的清單。

用法：
    python universe.py                    # 先產生 universe.csv
    python screen.py                      # 掃全市場
    python screen.py --small-cap          # 排除市值前 100 大，專掃中小型
    python screen.py --min-turnover 2e7   # 日均成交金額門檻（過濾流動性不足）

輸出：signals_YYYY-MM-DD.csv
"""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

import config
import data as data_mod
import signals as sig
from universe import fetch_universe, yf_ticker


def exit_levels(close: float, atr: float, profs: dict) -> dict:
    """
    每個設定檔各給一組停損／停利。
    持有天數只由 stop_atr 決定，與股票大小無關（Q1~Q4 實測皆同），
    所以這裡不分股票，直接套設定檔。
    """
    out: dict[str, float] = {}
    for name, cfg in profs.items():
        sa, rr = float(cfg["stop_atr"]), float(cfg["rr"])
        out[f"{name}_stop"] = round(close - sa * atr, 2)
        out[f"{name}_target"] = round(close + sa * atr * rr, 2)
        out[f"{name}_risk_pct"] = round(sa * atr / close * 100, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="taiwan")
    ap.add_argument("--period", default="3y", help="下載期間，選股只需近期資料")
    ap.add_argument("--min-turnover", type=float, default=5e7,
                    help="20 日均成交金額下限 (TWD)。2026-09-23 由 2000 萬改 5000 萬 —— "
                         "全市場實測 PF 1.480→1.496、中位報酬 14.3%%→17.0%%")
    ap.add_argument("--top", type=int, default=20,
                    help="每天最多輸出幾檔（突破優先，其餘用成交金額補滿）。0 = 不限")
    ap.add_argument("--small-cap", action="store_true",
                    help="排除成交金額前 100 大（近似排除權值股）")
    ap.add_argument("--exclude-top", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="只掃前 N 檔，除錯用")
    ap.add_argument("--source", default="twse", choices=["twse", "yf"],
                    help="twse=證交所官方快取（預設）；yf=yfinance 備援")
    ap.add_argument("--include-otc", action="store_true",
                    help="含上櫃（★ 上櫃無除權息還原資料，選股用可接受，回測不建議）")
    ap.add_argument("--no-adjust", action="store_true", help="不做除權息還原")
    ap.add_argument("--review-days", type=int, default=5,
                    help="選股後附上前幾天的追蹤（0 = 不附）")
    ap.add_argument("--review-horizon", type=int, default=5, help="追蹤到 D+N")
    args = ap.parse_args()

    p = config.params(args.market)
    profs = config.profiles(args.market)
    print(f"參數來源 markets.yaml[{args.market}]：trend_ma={p['trend_ma_length']} "
          f"volume_filter={p['volume_filter']} bbw_lookback={p['bbw_lookback']}")
    if profs:
        desc = "｜".join(
            f"{k}: stop {v['stop_atr']}ATR / rr {v['rr']}（約抱 {v.get('median_hold_days','?')} 交易日）"
            for k, v in profs.items())
        print(f"出場設定檔  {desc}")

    uni = fetch_universe(include_otc=args.include_otc)
    if args.limit:
        uni = uni.head(args.limit)

    if args.source == "twse":
        codes = list(uni["code"])
        name_of = dict(zip(uni["code"], uni["name"]))
        code_of = {c: c for c in codes}
        print(f"universe: {len(codes)} 檔，讀證交所快取…")
        bars, cache_names = data_mod.fetch_twse(codes=codes, include_otc=args.include_otc,
                                                adjust=not args.no_adjust, min_rows=60)
        name_of = {**cache_names, **name_of}
    else:
        tickers = [yf_ticker(c, m) for c, m in zip(uni["code"], uni["market"])]
        name_of = dict(zip(tickers, uni["name"]))
        code_of = dict(zip(tickers, uni["code"]))
        print(f"universe: {len(tickers)} 檔，開始下載…")
        bars = data_mod.fetch_yf(tickers, period=args.period)
    print(f"取得 {len(bars)} 檔有效資料")

    rows = []
    for t, df in bars.items():
        try:
            s = sig.compute(df, p)
        except Exception:
            continue
        last = s.iloc[-1]
        turnover = data_mod.avg_turnover(s, 20)
        if turnover < args.min_turnover:
            continue
        if not (bool(last["signal"]) or bool(last["watch"])):
            continue
        chg = float(s["Close"].pct_change().iloc[-1] * 100)
        # 收盤鎖漲停 → 明天開盤大機率買不到（回測端已用 limit_up_guard 排除這類單）
        locked = chg >= 9.5 and float(last["Close"]) >= float(last["High"]) * 0.999
        rows.append({
            "code": code_of.get(t, t),
            "name": name_of.get(t, ""),
            "ticker": t,
            "status": "突破" if last["signal"] else "待突破",
            "buyable": "" if not locked else "★鎖漲停可能買不到",
            "close": round(float(last["Close"]), 2),
            "chg_pct": round(chg, 2),
            "dist_to_upper_pct": round(float(last["dist_to_upper_pct"]), 2),
            "bbw_pct": round(float(last["bbw"]), 2),
            "bars_since_squeeze": int(last["bars_since_squeeze"]) if pd.notna(last["bars_since_squeeze"]) else -1,
            "atr_pct": round(float(last["atr"] / last["Close"] * 100), 2),
            "avg_turnover_20d": int(turnover),
            # 台股最小交易單位 1000 股（P1-5）。一張的錢決定這檔你買不買得動，
            # 但不影響 PF —— 回測用的是百分比報酬，張數是資金配置問題不是報酬率問題。
            "lot_cost": int(round(float(last["Close"]) * 1000)),
            # 兩組出場價位，由使用者自行決定這檔要做中線還是長線（見 markets.yaml profiles）
            **exit_levels(float(last["Close"]), float(last["atr"]), profs),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("今日無標的符合條件")
        return

    # 突破排前面，同類別內用成交金額排序（P0：規模越大 PF 越高）
    out["_rank"] = (out["status"] != "突破").astype(int)
    out = out.sort_values(["_rank", "avg_turnover_20d"], ascending=[True, False])
    if args.small_cap:
        big = out.nlargest(min(args.exclude_top, len(out)), "avg_turnover_20d")["code"]
        out = out[~out["code"].isin(big)]
    if args.top:
        out = out.head(args.top)
    out = out.drop(columns=["_rank"])

    # ★ 檔名用「資料日」不是日曆日。手動跑的時候這兩者經常不同 ——
    #   假日開、或行情快取還沒補上，日曆日就沒有對應的 K 棒，
    #   bb_review 找不到該日的收盤價會把整份存檔靜靜略過（實際發生過：2026-09-27）。
    #   用資料日另外解決重複計數：同一天的資料跑兩次只會有一份存檔，
    #   「被 call 幾次」才不會被「跑了幾次」污染。
    data_day = max(df.index[-1] for df in bars.values()).date()
    today = dt.date.today()
    fn = f"signals_{data_day}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")

    if data_day != today:
        print(f"\n★ 行情最新一天是 {data_day}，不是今天（{today}）—— "
              f"這批是 {data_day} 收盤的盤面。假日或快取未更新時屬正常。")

    # ★ 每日存檔供事後追蹤（bb_review.py 用）。存進去就不要改 —— 這是真實的樣本外紀錄。
    arch_dir = Path(__file__).resolve().parent / "cache" / "bb_picks"
    arch_dir.mkdir(parents=True, exist_ok=True)
    arch = arch_dir / f"{data_day}.csv"
    dup = arch.exists()
    if not dup:
        out.to_csv(arch, index=False, encoding="utf-8-sig")

    print(f"\n{len(out)} 檔 → {fn}"
          f"（存檔 {arch.name}{'，已存在故不覆蓋' if dup else ''}）")
    print(out.to_string(index=False))

    # 選股完直接附上前幾天的追蹤 —— 共用已載入的 bars，不重新載一次行情
    if args.review_days and args.source == "twse":
        print("\n" + "=" * 70)
        print(f"前 {args.review_days} 個選股日的後續表現")
        print("=" * 70)
        try:
            import bb_review
            bb_review.review(days=args.review_days, horizon=args.review_horizon,
                             bars=bars, names=name_of)
        except Exception as e:
            print(f"[warn] 追蹤失敗（選股結果不受影響）：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
