"""
產業「延續性」分析 —— 哪些類股漲了之後通常不會續漲。

三個獨立量尺，全部依產業彙總：
  A. 市場延續性（與任何策略無關）：每 20 個交易日，挑全市場過去 20 日報酬前 20% 的股票，
     看它們「接下來 20 日」相對當期全市場平均的超額報酬。超額為負＝漲完就熄火。
     兩段時間分開算（2019-2022 / 2023-2026），避免把單一年份的運氣當成產業特性。
  B. 布林縮口（signals + backtest.run_one）各股 PF 的產業中位數。
  C. 動能策略逐筆交易（momentum_trades.csv）的產業平均淨報酬，同樣分兩段。

判準：A 在兩段都為負、且 B、C 至少一個也偏弱，才列為「建議排除」。
只看單一量尺或單一時段挑出來的，是 data mining 不是產業特性。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
import twse
from backtest import run_one

HERE = Path(__file__).resolve().parent
NAMES = {
    "01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維", "05": "電機機械", "06": "電器電纜",
    "08": "玻璃陶瓷", "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造",
    "15": "航運", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨", "20": "其他", "21": "化學",
    "22": "生技醫療", "23": "油電燃氣", "24": "半導體", "25": "電腦週邊", "26": "光電",
    "27": "通信網路", "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子",
    "32": "文化創意", "33": "農業科技", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒",
    "38": "居家生活", "91": "存託憑證",
}
SPLIT = pd.Timestamp("2023-01-01")


def industry_map() -> dict[str, str]:
    m = {}
    for name, ck, ik in (("twse", "公司代號", "產業別"), ("tpex", "SecuritiesCompanyCode", "SecuritiesIndustryCode")):
        for r in json.loads((HERE / "cache" / "lists" / f"{name}.json").read_text(encoding="utf-8")):
            m[str(r.get(ck, "")).strip()] = str(r.get(ik, "")).strip()
    return m


def main() -> None:
    ind = industry_map()
    bars, _ = twse.load_bars(include_otc=True, adjust=True, min_rows=260)
    bars = {c: d for c, d in bars.items() if not c.startswith("00") and c in ind}
    liquid = {c: d for c, d in bars.items() if float(d["Turnover"].tail(250).mean()) >= 2e7}
    print(f"樣本 {len(liquid)} 檔（已排除金融電信、日均成交 ≥2000 萬）", flush=True)

    # ── A. 市場延續性 ──
    close = pd.DataFrame({c: d["Close"] for c, d in liquid.items()}).sort_index()
    past = close / close.shift(20) - 1
    fwd = close.shift(-20) / close - 1
    rows = []
    for t in close.index[20:-20:20]:                       # 每 20 日取樣一次，不重疊
        p, f = past.loc[t].dropna(), fwd.loc[t].dropna()
        common = p.index.intersection(f.index)
        if len(common) < 100:
            continue
        p, f = p[common], f[common]
        winners = p[p >= p.quantile(0.8)].index
        mkt = f.mean()
        for c in winners:
            rows.append({"date": t, "code": c, "ind": ind[c], "excess": f[c] - mkt})
    A = pd.DataFrame(rows)
    A["half"] = np.where(A["date"] < SPLIT, "前段", "後段")

    # ── B. 布林縮口各股 PF ──
    p = config.params("taiwan"); cost = config.costs("taiwan")["commission_pct_per_side"]
    brows = []
    for c, d in liquid.items():
        try:
            r = run_one(d, p, cost)
        except Exception:
            continue
        if r and r["trades"] >= 5:
            brows.append({"code": c, "ind": ind[c], "pf": r["profit_factor"]})
    B = pd.DataFrame(brows)

    # ── C. 動能逐筆 ──
    C = pd.read_csv(HERE / "momentum_trades.csv", dtype={"code": str}, parse_dates=["entry_date"])
    C["ind"] = C["code"].map(ind)
    C = C.dropna(subset=["ind"])
    C["half"] = np.where(C["entry_date"] < SPLIT, "前段", "後段")

    # ── 彙總 ──
    out = []
    for code in sorted(set(A["ind"])):
        a = A[A["ind"] == code]
        a1, a2 = a[a.half == "前段"]["excess"], a[a.half == "後段"]["excess"]
        b = B[B["ind"] == code]["pf"]
        c = C[C["ind"] == code]
        out.append({
            "代碼": code, "產業": NAMES.get(code, code),
            "檔數": int(a["code"].nunique()),
            "A前段超額%": round(a1.mean() * 100, 2) if len(a1) else np.nan,
            "A後段超額%": round(a2.mean() * 100, 2) if len(a2) else np.nan,
            "A樣本": len(a),
            "B布林PF": round(b.median(), 3) if len(b) >= 5 else np.nan,
            "B檔數": len(b),
            "C動能平均%": round(c["ret_net"].mean() * 100, 2) if len(c) >= 20 else np.nan,
            "C筆數": len(c),
        })
    df = pd.DataFrame(out)
    b_all = B["pf"].median()
    c_all = C["ret_net"].mean() * 100
    both_neg = (df["A前段超額%"] < 0) & (df["A後段超額%"] < 0)
    b_weak = df["B布林PF"] < b_all
    c_weak = df["C動能平均%"] < c_all
    df["建議排除"] = np.where(both_neg & (b_weak | c_weak) & (df["A樣本"] >= 100), "★", "")
    df = df.sort_values(["建議排除", "A後段超額%"], ascending=[False, True])
    df.to_csv(HERE / "industry_continuity.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(f"\n全市場基準：布林中位 PF {b_all:.3f}｜動能平均 {c_all:.2f}%/筆\n")
    print(df.to_string(index=False))
    print("\n★ = 兩段時間的「漲完續漲超額」都為負，且布林或動能至少一項低於全市場")


if __name__ == "__main__":
    main()
