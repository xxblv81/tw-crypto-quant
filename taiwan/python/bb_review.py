"""
布林縮口選股的事後追蹤。

資料來自 `cache/bb_picks/YYYY-MM-DD.csv`（screen.py 每天自動存，存了就不改）。
這是真正的樣本外紀錄：當天只看得到當天的資訊，不受事後調參數影響。

三個層次：
  ① 逐日彙總 —— 每一天選出的那批股票，平均／中位／上漲比例，並對照同期大盤
  ② 逐檔彙整（主要看這個）—— 同一檔合併：被 call 過幾次、哪幾天、從第一次 call
     到「最新一天」的表現。★ 這裡不受 horizon 限制，會一直追下去。
  ③ 逐檔逐日明細 —— 加 --detail 才印

★ 報酬一律用「首次被 call 當天收盤」到目標日收盤計算。實際進場是隔天開盤，
  所以數字與真實成交略有差距，但用於比較相對好壞沒問題。

用法：
    python bb_review.py                     # 最近 10 個選股日
    python bb_review.py --days 5 --detail    # 含逐日明細
    python bb_review.py --sort ret           # 逐檔彙整改按報酬排序
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import twse

PICKS = Path(__file__).resolve().parent / "cache" / "bb_picks"


def review(days: int = 5, horizon: int = 5, status: str = "all",
           bars: dict | None = None, names: dict | None = None,
           detail: bool = False, sort: str = "first") -> None:
    """印出追蹤報告。bars 可從外部傳進來共用，避免重複載入行情（省約 8 秒）。"""
    _run(days, horizon, status, bars, names, detail, sort)


def _run(days: int, horizon: int, status: str,
         bars: dict | None = None, names: dict | None = None,
         detail: bool = False, sort: str = "first") -> None:
    files = sorted(PICKS.glob("*.csv"))[-days:]
    if not files:
        print(f"還沒有任何存檔。跑過 screen.py 之後，清單會存進 {PICKS}")
        return

    if bars is None:
        bars, names = twse.load_bars(include_otc=True, adjust=True, min_rows=60)
    closes = pd.DataFrame({c: d["Close"] for c, d in bars.items()})
    mkt_cum = (1 + closes.pct_change().mean(axis=1).fillna(0)).cumprod()

    hs = list(range(1, horizon + 1))
    rows = []
    for f in files:
        pick_date = pd.Timestamp(f.stem)
        df = pd.read_csv(f, dtype={"code": str})
        if status != "all":
            df = df[df["status"] == status]
        for _, r in df.iterrows():
            b = bars.get(r["code"])
            if b is None or pick_date not in b.index:
                continue
            i = b.index.get_loc(pick_date)
            base = float(b["Close"].iloc[i])
            rec = {"日期": pick_date, "代號": r["code"], "名稱": r["name"],
                   "狀態": r["status"], "選出價": round(base, 2),
                   "停損": float(r["mid_stop"]), "_i": i}
            for h in hs:
                rec[f"D+{h}"] = (round((float(b["Close"].iloc[i + h]) / base - 1) * 100, 2)
                                 if i + h < len(b) else np.nan)
            last_i = min(i + horizon, len(b) - 1)
            rec["目前"] = round((float(b["Close"].iloc[last_i]) / base - 1) * 100, 2)
            rec["天數"] = last_i - i
            rows.append(rec)

    if not rows:
        print("存檔裡的標的都還沒有後續行情（可能今天才剛選）。")
        return
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 240)

    # ── ① 逐日彙總 ──
    daily = []
    for d, g in R.groupby("日期"):
        v = g["目前"].dropna()
        if v.empty:
            continue
        i0 = mkt_cum.index.get_loc(d)
        i1 = min(i0 + int(g["天數"].max()), len(mkt_cum) - 1)
        bench = (mkt_cum.iloc[i1] / mkt_cum.iloc[i0] - 1) * 100
        daily.append({"選股日": d.date(), "檔數": len(g), "追蹤天數": int(g["天數"].max()),
                      "平均%": round(v.mean(), 2), "中位%": round(v.median(), 2),
                      "上漲比例%": round((v > 0).mean() * 100, 1),
                      "大盤%": round(bench, 2), "超額%": round(v.mean() - bench, 2)})
    print(f"=== ① 逐日彙總（{len(files)} 個選股日，追蹤到 D+{horizon}）===")
    print(pd.DataFrame(daily).to_string(index=False))

    # ── ② 逐檔彙整：同一檔合併，追蹤到最新一天 ──
    agg = []
    for code, g in R.groupby("代號"):
        g = g.sort_values("日期")
        b = bars[code]
        i0 = int(g["_i"].iloc[0])
        base = float(g["選出價"].iloc[0])
        seg = b.iloc[i0:]                       # 首次 call 之後的全部 K 棒
        now = float(seg["Close"].iloc[-1])
        stop = float(g["停損"].iloc[0])
        broke = float(seg["Low"].min()) < stop
        dates = list(g["日期"].dt.strftime("%m/%d"))
        sts = g["狀態"].tolist()
        agg.append({
            "代號": code, "名稱": g["名稱"].iloc[0],
            "call次數": len(g),
            "被call日期": "、".join(dates),
            "首次狀態": sts[0],
            "轉突破": "✓" if ("突破" in sts and sts[0] == "待突破") else "",
            "首次價": round(base, 2), "現價": round(now, 2),
            "至今%": round((now / base - 1) * 100, 2),
            "最高%": round((float(seg["High"].max()) / base - 1) * 100, 2),
            "最低%": round((float(seg["Low"].min()) / base - 1) * 100, 2),
            "已過天數": len(seg) - 1,
            "跌破停損": "✗" if broke else "",
        })
    A = pd.DataFrame(agg)
    A = A.sort_values("至今%", ascending=False) if sort == "ret" else A.sort_values(
        ["call次數", "至今%"], ascending=[False, False])

    print(f"\n=== ② 逐檔彙整（{len(A)} 檔，同一檔合併，追蹤到最新一天）===")
    print(A.to_string(index=False))

    v = A["至今%"]
    print(f"\n合計 {len(A)} 檔｜平均 {v.mean():+.2f}%｜中位 {v.median():+.2f}%"
          f"｜上漲 {(v > 0).sum()} 檔／下跌 {(v < 0).sum()} 檔"
          f"｜跌破停損 {(A['跌破停損'] == '✗').sum()} 檔")

    multi = A[A["call次數"] >= 2]
    if not multi.empty:
        print(f"\n被 call 過 2 次以上的 {len(multi)} 檔：平均 {multi['至今%'].mean():+.2f}%"
              f"（只 call 過 1 次的 {A[A['call次數'] == 1]['至今%'].mean():+.2f}%）")
    turned = A[A["轉突破"] == "✓"]
    if not turned.empty:
        print(f"從「待突破」轉成「突破」的 {len(turned)} 檔：平均 {turned['至今%'].mean():+.2f}%")
    print("\n依首次狀態拆分：")
    print(A.groupby("首次狀態")["至今%"].agg(["count", "mean", "median"]).round(2).to_string())

    # ── ③ 逐檔逐日明細（選配）──
    if detail:
        print(f"\n=== ③ 逐檔逐日明細 ===")
        show = ["日期", "代號", "名稱", "狀態", "選出價"] + [f"D+{h}" for h in hs] + ["目前"]
        D = R.copy()
        D["日期"] = D["日期"].dt.date
        print(D.sort_values(["代號", "日期"])[show].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="回顧最近幾個存檔日")
    ap.add_argument("--horizon", type=int, default=5, help="逐日彙總看到 D+N")
    ap.add_argument("--status", default="all", choices=["all", "突破", "待突破"])
    ap.add_argument("--detail", action="store_true", help="加印逐檔逐日明細")
    ap.add_argument("--sort", default="first", choices=["first", "ret"],
                    help="逐檔彙整排序：first=call 次數優先，ret=報酬")
    args = ap.parse_args()
    _run(args.days, args.horizon, args.status, detail=args.detail, sort=args.sort)


if __name__ == "__main__":
    main()
