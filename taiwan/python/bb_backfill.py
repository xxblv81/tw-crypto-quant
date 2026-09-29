"""
回填布林選股存檔 —— 用「現在的參數」重算過去 N 個交易日的清單。

★ 這不是「當時真的會看到的清單」（當時參數不同），但它是**點時間**的：
  每一天只用該日（含）之前的資料算訊號，沒有用到未來資訊。
  用來回答「這套現行設定最近幾天選的東西表現如何」是有效的。
  真正的樣本外紀錄是之後每天 screen.py 自動存的那些。
"""
import argparse, datetime as dt
from pathlib import Path
import pandas as pd
import config, twse, signals as sig, data as data_mod
from screen import exit_levels
from universe import fetch_universe

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=8)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--min-turnover", type=float, default=5e7)
a = ap.parse_args()

p = config.params("taiwan"); profs = config.profiles("taiwan")
uni = fetch_universe(include_otc=True)
codes = list(uni["code"]); name_of = dict(zip(uni["code"], uni["name"]))
bars, cache_names = twse.load_bars(include_otc=True, adjust=True, min_rows=60, codes=codes)
name_of = {**cache_names, **name_of}
print(f"{len(bars)} 檔", flush=True)

all_dates = sorted({d for b in bars.values() for d in b.index[-a.days-1:]})[-a.days:]
out_dir = Path(__file__).resolve().parent / "cache" / "bb_picks"
out_dir.mkdir(parents=True, exist_ok=True)

for day in all_dates:
    rows = []
    for c, b in bars.items():
        if day not in b.index:
            continue
        d = b.loc[:day]                      # ★ 只用當日(含)之前
        if len(d) < 60:
            continue
        try:
            s = sig.compute(d, p)
        except Exception:
            continue
        last = s.iloc[-1]
        to = data_mod.avg_turnover(s, 20)
        if to < a.min_turnover or not (bool(last["signal"]) or bool(last["watch"])):
            continue
        chg = float(s["Close"].pct_change().iloc[-1] * 100)
        locked = chg >= 9.5 and float(last["Close"]) >= float(last["High"]) * 0.999
        rows.append({"code": c, "name": name_of.get(c, ""), "ticker": c,
                     "status": "突破" if last["signal"] else "待突破",
                     "buyable": "" if not locked else "★鎖漲停可能買不到",
                     "close": round(float(last["Close"]), 2), "chg_pct": round(chg, 2),
                     "atr_pct": round(float(last["atr"] / last["Close"] * 100), 2),
                     "avg_turnover_20d": int(to),
                     "lot_cost": int(round(float(last["Close"]) * 1000)),
                     **exit_levels(float(last["Close"]), float(last["atr"]), profs)})
    if not rows:
        print(f"{day:%Y-%m-%d}  0 檔", flush=True); continue
    df = pd.DataFrame(rows)
    df["_r"] = (df["status"] != "突破").astype(int)
    df = df.sort_values(["_r", "avg_turnover_20d"], ascending=[True, False]).head(a.top).drop(columns=["_r"])
    f = out_dir / f"{day:%Y-%m-%d}.csv"
    df.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"{day:%Y-%m-%d}  {len(df)} 檔（突破 {(df.status=='突破').sum()}）→ {f.name}", flush=True)
