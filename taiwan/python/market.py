"""
大盤狀態（market regime）—— 讓策略依大盤動能與方向動態調整。

為什麼需要：
    動能策略在空頭會被反覆巴掌。實測分年 PF：2020 1.70、2021 1.41、**2022 0.57**、
    2023 1.63、2024 1.03、2025 1.88、2026 1.78 —— 唯一虧損的 2022 正是空頭年。
    這不是選股選錯，是「強者恆強」在全市場都在跌的時候本來就不成立。

兩個獨立的大盤量測，刻意不合成單一分數（合成會把兩者的失效模式混在一起）：

  trend   發行量加權股價指數 vs 自身均線。方向的直接量測。
          資料：TWSE FMTQIK（每日市場成交資訊），一個月一次請求，免 token。

  breadth 全市場（過流動性門檻者）站上自身 MA60 的比例。
          ★ 這個完全不用新資料，用現成的行情快取就算得出來。
          寬度常常比指數本身早轉向 —— 指數被權值股撐住、但多數股票已經走弱時，
          動能策略會先受傷，指數卻還沒跌。2022 那種年份就是這樣開始的。

用法：
    python market.py --refresh                # 建/更新加權指數快取（約 5 分鐘）
    python market.py --status
    python market.py --show                   # 印出近期的大盤狀態分級
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from twse import CACHE, UA, _get_json, _num, _roc_date

FMTQIK = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
INDEX_CSV = CACHE / "taiex.csv"


def fetch_index(start: dt.date, end: dt.date, pause: float = 3.0,
                verbose: bool = True) -> pd.DataFrame:
    """
    發行量加權股價指數日收盤。FMTQIK 一次回一個月，所以逐月抓。
    已有的月份不重抓（除了最後一個月，它可能還沒收完）。
    """
    import time
    old = pd.DataFrame()
    if INDEX_CSV.exists():
        old = pd.read_csv(INDEX_CSV, parse_dates=["date"])
    have = set()
    if not old.empty:
        # 最後一個月一定重抓 —— 上次抓的時候那個月可能還沒過完
        last_month = old["date"].max().to_period("M")
        have = {p for p in old["date"].dt.to_period("M").unique() if p != last_month}

    rows = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        if pd.Period(cur, "M") in have:
            cur = (pd.Timestamp(cur) + pd.offsets.MonthBegin(1)).date()
            continue
        js = _get_json(FMTQIK, {"date": cur.strftime("%Y%m01"), "response": "json"})
        if isinstance(js, dict) and js.get("stat") == "OK" and js.get("data"):
            f = js["fields"]
            i_d = f.index("日期")
            i_c = f.index("發行量加權股價指數")
            i_amt = f.index("成交金額")
            for r in js["data"]:
                d = _roc_date(r[i_d])
                c = _num(r[i_c])
                if d and c > 0:
                    rows.append({"date": pd.Timestamp(d), "close": c,
                                 "turnover": _num(r[i_amt])})
        if verbose:
            print(f"  {cur:%Y-%m}：累計 {len(rows)} 天", flush=True)
        cur = (pd.Timestamp(cur) + pd.offsets.MonthBegin(1)).date()
        time.sleep(pause)

    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if df.empty:
        return df
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)
    return df


def load_index() -> pd.Series:
    """加權指數日收盤，index 為日期。快取不存在時回空 Series。"""
    if not INDEX_CSV.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(INDEX_CSV, parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def breadth(close: pd.DataFrame, mask: pd.DataFrame, ma_len: int = 60) -> pd.Series:
    """
    大盤寬度：過流動性門檻的標的裡，收盤站上自身 MA_n 的比例（0~1）。
    ★ 只用現成的行情面板，不需要任何新資料源。
    """
    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    above = (close > ma).where(mask & close.notna() & ma.notna())
    return above.mean(axis=1, skipna=True)


def regime(index_close: pd.Series, brd: pd.Series, p: dict,
           calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    回傳每日大盤狀態，欄位：
        idx_above_ma  指數是否站上均線
        idx_slope_up  均線是否較 N 日前上揚（避免「剛跌破又彈回」反覆切換）
        breadth       寬度 0~1
        level         0 弱 / 1 中 / 2 強

    分級規則刻意用「兩個條件的計數」而不是加權分數 ——
    加權會產生一堆需要調的權重，計數只有門檻要調，過度配適的空間小很多。
    """
    if calendar is None:
        calendar = brd.index
    b = brd.reindex(calendar)

    # 指數是「輔助」訊號 —— 沒有它時整套要還能跑，只靠寬度。
    # ★ 早期版本在指數缺資料時把 level 一律設成「中」，結果把主訊號（寬度）也吃掉了。
    #   指數只在「有資料」的日子才參與判斷。
    # ★ use_index 預設 false —— 實測指數趨勢不但沒加分，還會讓「弱勢」那組
    #   混進賺錢的交易（弱勢 PF 由 0.848 升到 1.091 ＝ 開始錯過好單）：
    #       寬度only   弱勢PF 0.848  CAGR 72.9%  回撤 43.1%  比值 1.69
    #       寬度+指數  弱勢PF 1.091  CAGR 63.2%  回撤 39.5%  比值 1.60
    #       指數only   弱勢PF 1.133  CAGR 72.1%  回撤 43.2%  比值 1.67
    #   指數仍然抓、仍然顯示（人看盤要），但不參與分級。
    if p.get("use_index", False) and index_close is not None and len(index_close):
        ic = index_close.reindex(calendar).ffill()
        ma = ic.rolling(p["index_ma"], min_periods=p["index_ma"]).mean()
        idx_above = ic > ma
        idx_slope = ma > ma.shift(p["index_slope_lookback"])
        have_idx = ic.notna() & ma.notna()
    else:
        # 不參與分級，但仍算出來給畫面顯示用
        if index_close is not None and len(index_close):
            ic = index_close.reindex(calendar).ffill()
            ma = ic.rolling(p["index_ma"], min_periods=p["index_ma"]).mean()
            idx_above, idx_slope = ic > ma, ma > ma.shift(p["index_slope_lookback"])
        else:
            ic = pd.Series(float("nan"), index=calendar)
            ma = ic.copy()
            idx_above = pd.Series(False, index=calendar)
            idx_slope = pd.Series(False, index=calendar)
        have_idx = pd.Series(False, index=calendar)

    # 弱：寬度不足 —— 或者（指數有資料時）指數已在均線之下
    weak = (b < p["breadth_weak"]) | (have_idx & ~idx_above)
    # 強：寬度充足，且指數若有資料則須同時站上均線並上揚
    strong = (b >= p["breadth_strong"]) & (~have_idx | (idx_above & idx_slope))

    level = pd.Series(1, index=calendar, dtype="int64")
    level[strong] = 2
    level[weak] = 0
    # 只有「寬度本身」算不出來（暖身期）才視為中性，不要因為指數缺資料就停手
    level[b.isna()] = 1

    return pd.DataFrame({"index_close": ic, "index_ma": ma,
                         "idx_above_ma": idx_above, "idx_slope_up": idx_slope,
                         "breadth": b, "level": level})


def main() -> None:
    ap = argparse.ArgumentParser(description="大盤狀態")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="印出最近 N 天的大盤狀態")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--pause", type=float, default=3.0)
    args = ap.parse_args()

    if args.refresh:
        df = fetch_index(dt.date.fromisoformat(args.start),
                         dt.date.fromisoformat(args.end), pause=args.pause)
        print(f"加權指數 {len(df)} 天 → {INDEX_CSV}")

    ic = load_index()
    if ic.empty:
        print("尚未建立加權指數快取，先跑：python market.py --refresh")
        return
    print(f"加權指數快取：{len(ic)} 天｜{ic.index[0]:%Y-%m-%d} ~ {ic.index[-1]:%Y-%m-%d}")

    if args.show:
        import config, momentum as mom, twse
        p0 = mom.normalize_params(config.params("taiwan_momentum"))
        rp = config.params("taiwan_market")
        bars, _ = twse.load_bars(adjust=True, min_rows=260)
        panel = mom.build_panel(bars)
        msk = mom.liquidity_mask(panel["Turnover"], p0)
        brd = breadth(panel["Close"], msk, rp["breadth_ma"])
        rg = regime(ic, brd, rp, panel["Close"].index)
        name = {0: "弱", 1: "中", 2: "強"}
        out = rg.tail(args.show).copy()
        out["狀態"] = out["level"].map(name)
        out["breadth"] = (out["breadth"] * 100).round(1)
        print(out[["index_close", "index_ma", "breadth", "狀態"]].to_string())
        print(f"\n全期分布：" + "　".join(
            f"{name[k]} {v} 天（{v/len(rg)*100:.0f}%）"
            for k, v in rg["level"].value_counts().sort_index().items()))


if __name__ == "__main__":
    main()
