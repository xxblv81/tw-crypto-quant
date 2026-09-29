"""
假日例行：更新集保大戶資料 → 產生本週選股 → 存檔備查。

為什麼要存檔：
    集保官網只保留 51 週滾動視窗，而且大戶換手訊號目前只有一年樣本、單一多頭年。
    每週把「當下的判斷」原封不動存下來，時間一久就會累積出**真正的樣本外紀錄** ——
    這是唯一能驗證這套策略的方法（回測永遠有選擇者的知識污染）。
    ★ 存的是「當週看到什麼、選了什麼」，事後不要修改，否則紀錄就沒有意義。

排程：週六／週日跑（集保週五資料在週末已可取得，實測週日抓得到週五那批）。

用法：
    python weekly_pick.py              # 更新 + 選股 + 存檔
    python weekly_pick.py --no-update  # 不更新資料，只重算
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from twse import CACHE

PICK_DIR = CACHE / "picks"

STATES = ("續買", "回頭車", "下車", "續賣")
NOTE = {"續買": "上週買 → 這週買　實測 4 週 +15.66%／勝率 63%（唯一明顯優於基準 +8.19%）",
        "回頭車": "上週賣 → 這週買　實測 4 週 +3.69%／勝率 54%（最差，反指標）",
        "下車": "上週買 → 這週賣　實測 4 週 +7.55%／勝率 61%（略低於基準）",
        "續賣": "上週賣 → 這週賣　實測 4 週 +6.46%／勝率 60%"}


def _nonempty(p) -> bool:
    import os
    return os.path.getsize(p) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--thr", type=float, default=1.0)
    ap.add_argument("--tradeable", action="store_true", default=True)
    args = ap.parse_args()

    if not args.no_update:
        import tdcc
        df = tdcc.fetch_latest()
        d = str(df["date"].iloc[0])
        PICK_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE / "tdcc").mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE / "tdcc" / f"{d}.parquet", index=False)
        print(f"集保更新：{d}　{df['code'].nunique()} 檔")

    import chip_state
    import futures as fut
    st = chip_state.build(thr=args.thr)
    if st.empty:
        print("集保快取是空的")
        return
    day = st["date"].max()
    cur = st[st["date"] == day].copy()

    import glob
    from twse import DAY_DIR
    names = {}
    for p in sorted(glob.glob(str(DAY_DIR / "*.parquet")))[-5:]:
        try:
            x = pd.read_parquet(p, columns=["code", "name"])
            names.update(dict(zip(x["code"], x["name"])))
        except Exception:
            pass
    cur["name"] = cur["code"].map(names).fillna("")

    # 本週漲跌：上一個集保週別 → 本週別的收盤（行情日檔、未還原，除權息週會失真）
    dates = sorted(st["date"].unique())
    prev_day = dates[-2] if len(dates) > 1 else None

    def close_at(d):
        fs = sorted(p for p in glob.glob(str(DAY_DIR / "*.parquet"))
                    if pd.Timestamp(p.split("/")[-1][:10]) <= d and _nonempty(p))
        if not fs:
            return pd.Series(dtype=float), None
        x = pd.read_parquet(fs[-1], columns=["code", "Close"])
        return x.set_index("code")["Close"], fs[-1].split("/")[-1][:10]

    c1, d1 = close_at(day)
    c0, d0 = close_at(prev_day) if prev_day is not None else (pd.Series(dtype=float), None)
    cur["本週漲跌%"] = (cur["code"].map(c1) / cur["code"].map(c0) - 1) * 100

    try:
        ssf = fut.load(min_volume=500).set_index("code")
        cur["契約"] = cur["code"].map(ssf["Contract"])
        cur["有個股期"] = cur["code"].isin(ssf.index)
    except Exception:
        cur["契約"], cur["有個股期"] = "", False

    picks = cur[cur["狀態"] != ""].copy()
    picks["pick_date"] = day
    PICK_DIR.mkdir(parents=True, exist_ok=True)
    fn = PICK_DIR / f"{day:%Y-%m-%d}.csv"
    cols = ["pick_date", "code", "name", "契約", "有個股期", "狀態", "升格",
            "d_big400_prev", "d_big400", "d_big1000", "d_retail",
            "本週漲跌%", "融資增減%", "融資增減張", "券資比"]
    # 已存在就不覆蓋：存檔是當週的樣本外紀錄，重跑也不能改
    if fn.exists():
        saved = False
    else:
        picks[cols].to_csv(fn, index=False, encoding="utf-8-sig")
        saved = True

    n = picks["狀態"].value_counts()
    n_ok = int(cur["d_big400_prev"].notna().sum())
    print(f"\n══ {day:%Y-%m-%d} 大戶換手（全市場・400 張以上）══")
    # 近兩週出現過無效快照或股本異動、但本週是正常股票 → 減資等事件，本週無法判定
    recent = st[st["date"].isin(dates[-3:])]
    hit = recent[(recent["股本異動"] != "") | recent["holders"].le(1)]["code"]
    n_jump = int(cur[cur["code"].isin(hit) & cur["holders"].gt(1)]["code"].nunique())
    n_stock = int(cur["holders"].gt(1).sum())
    print(f"母體 {n_stock} 檔（股東 >1 人）｜可判定 {n_ok} 檔｜門檻 ±{args.thr:g}pp"
          f"｜減資／股本異動無法判定 {n_jump} 檔")
    print(f"本週漲跌＝收盤 {d0} → {d1}（未還原）｜融資含上市＋上櫃")
    print("　".join(f"{s} {int(n.get(s,0))}" for s in STATES))

    for s in STATES:
        g = picks[picks["狀態"] == s]
        print(f"\n── {s}（{len(g)} 檔）── {NOTE[s]}")
        if g.empty:
            print("  本週無")
            continue
        # 本週買的依增幅排、本週賣的依減幅排
        g = g.sort_values("d_big400", ascending=s in ("下車", "續賣"))
        for _, r in g.iterrows():
            mark = "★" if r["升格"] else "　"
            fu = r["契約"] if r.get("有個股期") else "—"
            px = "   —  " if pd.isna(r["本週漲跌%"]) else f"{r['本週漲跌%']:+6.2f}%"
            mg = ("—" if pd.isna(r["融資增減%"])
                  else f"{r['融資增減張']:+,.0f}張({r['融資增減%']:+.1f}%)")
            print(f"  {mark} {r['code']:<5} {r['name']:<6} 上週Δ {r['d_big400_prev']:+.2f}"
                  f"　本週Δ {r['d_big400']:+.2f}　千張Δ {r['d_big1000']:+.2f}"
                  f"　散戶Δ {r['d_retail']:+.2f}　漲跌 {px}　融資 {mg}　個股期 {fu}")
    print("\n★ = 中大戶升格（千張Δ > 400張Δ）")
    print(f"\n→ {fn}" + ("" if saved else "（已存在，未覆蓋）"))
    print("★ 這份紀錄是事後驗證用的，不要修改。平日跑 pick_review.py 看累積成績。")


if __name__ == "__main__":
    main()
