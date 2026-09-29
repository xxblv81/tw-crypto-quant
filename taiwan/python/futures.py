"""
台股個股期貨（SSF）—— 標的清單、槓桿倍數、流動性。

為什麼要這支：
    用個股期當交易主體時，「能不能買」的限制跟現股完全不同：
      1. 只有約 300 檔有個股期，現股 1,083 檔的宇宙直接砍到 1/4
      2. 保證金分三個級距（13.5% / 16.2% / 20.25%）→ 槓桿 7.4x / 6.2x / 4.9x，逐檔不同
      3. ★ 流動性極度集中 —— 實測 2026-09-01，284 檔有成交但只有 86 檔 > 1,000 口。
         個股期真正的成本是買賣價差，不是手續費。掃出一檔沒量的期貨等於掃了個寂寞。

    所以動能選股在期貨模式下要先跟這份清單取交集，再談分數。

資料源：期交所 OpenAPI（免 token，無明顯限流，但一樣自律）
    /SSFLists                     股票期貨交易標的（契約代碼 ↔ 股票代號）
    /SingleStockFuturesMargining  保證金級距 → 槓桿
    /DailyMarketReportFut         每日行情（成交量、未平倉）→ 流動性

★ 限制：OpenAPI 的每日行情只給「最近一個交易日」，沒有歷史。
  所以流動性是「現況快照」，回測時無法還原當時哪些契約有量。
  這代表期貨模式的回測**天生帶有前視偏誤**（用今天的流動性篩過去的標的），
  規模比倖存者偏差小但確實存在，結論要打折。

用法：
    python futures.py --refresh      # 更新清單與流動性快照
    python futures.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import requests

from twse import CACHE

API = "https://openapi.taifex.com.tw/v1"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SSF_CSV = CACHE / "ssf_universe.csv"

# ★ 契約規模不是單一值：
#     一般股票期貨   一口 = 2,000 股
#     小型股票期貨   一口 =   100 股   ← 契約名稱含「小型」，實測 249 檔裡有 45 檔
#   差 20 倍。早期版本一律用 2,000 算，把小型契約的名目與保證金全部灌大 20 倍
#   （大立光小型期一口被算成 1,516 萬，實際是 75.8 萬）。
CONTRACT_SIZE = 2000
CONTRACT_SIZE_MINI = 100


def _get(path: str):
    r = requests.get(API + path, headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()


def _num(s) -> float:
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def refresh() -> pd.DataFrame:
    """抓標的清單 + 保證金 + 當日流動性，合併後落地。"""
    CACHE.mkdir(parents=True, exist_ok=True)

    ssf = pd.DataFrame(_get("/SSFLists"))
    ssf = ssf[ssf["Type"].str.contains("普通股", na=False)]
    ssf = ssf.rename(columns={"StockCode": "code", "StockName": "name"})
    ssf["code"] = ssf["code"].astype(str).str.strip()

    mg = pd.DataFrame(_get("/SingleStockFuturesMargining"))
    mg = mg.rename(columns={"UnderlyingSecurityCode": "code"})
    mg["code"] = mg["code"].astype(str).str.strip()
    mg["margin_pct"] = mg["InitialMarginRate"].map(_num)
    mg["is_mini"] = mg["ContractName"].str.contains("小型", na=False)
    mg["contract_size"] = mg["is_mini"].map({True: CONTRACT_SIZE_MINI,
                                             False: CONTRACT_SIZE})
    mg["leverage"] = (100 / mg["margin_pct"]).round(2)
    # ★ 不要在這裡依 code 去重。一檔股票可能同時有一般契約與小型契約，兩者的
    #   契約規模差 20 倍，砍掉一個會讓後面用 Contract 併不到、規模抓錯。
    #   同一個 Contract 若有多列（不同生效日），取保證金最高的那筆（最保守）。
    mg = (mg.sort_values("margin_pct", ascending=False)
            .drop_duplicates("Contract", keep="first"))

    day = pd.DataFrame(_get("/DailyMarketReportFut"))
    day = day[day["TradingSession"] == "一般"]
    for c in ("Volume", "OpenInterest"):
        day[c] = day[c].map(_num)
    liq = (day.groupby("Contract", as_index=False)
              .agg(volume=("Volume", "sum"), open_interest=("OpenInterest", "sum")))
    snap_date = str(day["Date"].iloc[0]) if len(day) else ""

    # ★ 用 Contract 併保證金表，不要用 code —— 一檔股票可能同時有一般與小型兩種契約，
    #   用 code 併會把兩者混在一起，契約規模就抓錯了。
    out = (ssf[["Contract", "code", "name"]]
           .merge(mg[["Contract", "margin_pct", "leverage", "GroupLevel",
                      "is_mini", "contract_size"]], on="Contract", how="left")
           .merge(liq, on="Contract", how="left"))
    out["contract_size"] = out["contract_size"].fillna(CONTRACT_SIZE).astype(int)
    out["is_mini"] = out["is_mini"].fillna(False)
    out["volume"] = out["volume"].fillna(0)
    out["open_interest"] = out["open_interest"].fillna(0)
    out["liq_date"] = snap_date
    # 同一股票可能有多個契約（一般 + 小型 + 調整型）→ 保留成交量最大的那個。
    # 小型契約往往比一般契約更活躍，對小資金也更好用（一口只要 1/20 的錢）。
    out = (out.sort_values("volume", ascending=False)
              .drop_duplicates("code", keep="first"))
    out.to_csv(SSF_CSV, index=False, encoding="utf-8-sig")
    return out


def basis(spot: pd.Series, universe: pd.DataFrame) -> pd.DataFrame:
    """
    今日期現價差 = (近月期貨結算價 − 現貨收盤) / 現貨收盤。

    負值＝期貨折價，做多的人到期收斂會賺到這段；正值＝溢價，要倒貼。
    ★★ 現貨與期貨必須同一天。期交所給的是「最近一個交易日」，行情快取若落後，
       算出來的不是價差而是這幾天的漲跌 —— 呼叫端要比對 attrs["fut_date"]。

    ★ 兩個限制，用之前要知道：
      1. 期交所 OpenAPI 只給「最近一個交易日」（實測 date 參數無效），**無法回測**。
         所以這欄只能當即時加分項，不能宣稱它有統計上的邊際。
      2. 折價常常只是反映結算前的除息 —— 期貨不配息，所以除息前必然折價。
         那不是免費的錢，是你本來就領不到的股息。要判斷是否為真折價，
         需要「結算前是否除息」的資訊，本專案的 exright.csv 只有歷史沒有未來。
    """
    day = pd.DataFrame(_get("/DailyMarketReportFut"))
    day = day[day["TradingSession"] == "一般"].copy()
    fut_date = str(day["Date"].iloc[0]) if len(day) else ""
    for c in ("Last", "SettlementPrice"):
        day[c] = day[c].map(_num)
    # 同一契約有多個到期月 → 取最近月
    day = day.sort_values("ContractMonth(Week)").drop_duplicates("Contract", keep="first")
    px = day.set_index("Contract").apply(
        lambda r: r["SettlementPrice"] if pd.notna(r["SettlementPrice"]) else r["Last"], axis=1)

    rows = []
    for code, r in universe.iterrows():
        ct = r["Contract"]
        if ct not in px.index or code not in spot.index:
            continue
        f_, s_ = float(px[ct]), float(spot[code])
        if not (f_ > 0 and s_ > 0):
            continue
        rows.append({"code": code, "期貨": round(f_, 2),
                     "價差%": round((f_ - s_) / s_ * 100, 2),
                     "到期月": day.set_index("Contract").loc[ct, "ContractMonth(Week)"],
                     "期貨日": fut_date})
    out = pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()
    out.attrs["fut_date"] = fut_date
    return out


def load(min_volume: float = 500) -> pd.DataFrame:
    """
    讀快取。min_volume 是「最近一個交易日成交口數」下限。

    為什麼預設 500：實測分布 —— >0 口 284 檔、>100 口 194 檔、>500 口 116 檔、
    >1000 口 86 檔。500 口大約是「掛單進得去、價差還能忍」的分界。
    要更保守就調到 1000。
    """
    if not SSF_CSV.exists():
        raise FileNotFoundError("尚未建立個股期清單，先跑：python futures.py --refresh")
    df = pd.read_csv(SSF_CSV, dtype={"code": str})
    df["code"] = df["code"].str.zfill(4)
    return df[df["volume"] >= min_volume].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="個股期貨標的清單與流動性")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--min-volume", type=float, default=500)
    args = ap.parse_args()

    if args.refresh:
        df = refresh()
        print(f"個股期標的 {len(df)} 檔 → {SSF_CSV}（流動性快照 {df['liq_date'].iloc[0]}）")
    if not SSF_CSV.exists():
        print("尚未建立，跑 --refresh")
        return

    df = pd.read_csv(SSF_CSV, dtype={"code": str})
    print(f"\n個股期標的 {len(df)} 檔｜流動性快照 {df['liq_date'].iloc[0]}")
    print("\n成交口數分布：")
    for th in (0, 100, 500, 1000, 5000):
        print(f"  > {th:>5} 口：{(df['volume'] > th).sum():>3} 檔")
    print("\n保證金級距（槓桿）：")
    print(df.groupby("GroupLevel").agg(檔數=("code", "count"),
                                       保證金=("margin_pct", "first"),
                                       槓桿=("leverage", "first")).to_string())
    sel = load(args.min_volume)
    print(f"\n以 >{args.min_volume:.0f} 口為門檻 → {len(sel)} 檔可用")
    print(f"其中小型契約（一口 100 股）{int(sel['is_mini'].sum())} 檔、"
          f"一般契約（一口 2,000 股）{int((~sel['is_mini']).sum())} 檔")
    print(sel.head(12)[["code", "name", "Contract", "contract_size",
                        "leverage", "volume"]].to_string(index=False))


if __name__ == "__main__":
    main()
