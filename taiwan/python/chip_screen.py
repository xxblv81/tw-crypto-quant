"""
籌碼選股 —— 對標旺來籌碼卡的四個名單，並依實測結果排序。

四個狀態（定義見 chip_state.py，門檻預設 1.0pp）：
    續買    上週買 → 這週再加碼
    回頭車  上週賣 → 這週買回
    下車    上週買 → 這週賣
    續賣    上週賣 → 這週續賣

★ 實測前瞻報酬（107 檔個股期標的 × 51 週、集保公布後起算、避免前視）：

    狀態        n     1週平均   4週平均   4週中位   4週勝率
    續買      144     +4.33    +15.66    +8.57     63%    ← 唯一明顯優於基準
    （無狀態） 4850    +1.77     +8.19    +3.39     61%
    下車      176     +2.84     +7.55    +6.77     61%
    續賣      169     +1.35     +6.46    +2.26     60%
    回頭車    115     -0.36     +3.69    +1.85     54%    ← 最差

    續買 × 中大戶升格   n=63   4週 +18.83%  中位 +12.32%  勝率 73%
    續買 × 未升格       n=81   4週 +12.81%  中位  +2.80%  勝率 55%

  → **只有「續買」值得追，「回頭車」是反指標**（與卡片「回頭車不再溢價」一致）。
    加上「中大戶升格」還能再拉一段，而且中位數同步上升（不是離群值造成的）。

★★ 限制：51 週、單一多頭年、107 檔；4 週窗口重疊會高估顯著性；
   而且我測了 4 狀態 × 2 升格 × 3 期間共 24 格，未做多重比較校正。
   訊號的**方向**可信，**幅度**不要當期望值。

用法：
    python chip_screen.py                 # 四個名單全列（同卡片）
    python chip_screen.py --state 續買     # 只看續買
    python chip_screen.py --tradeable     # 只留有個股期且有量的
"""
from __future__ import annotations

import argparse

import pandas as pd

import chip_state


def twse_day_dir():
    from twse import DAY_DIR
    return DAY_DIR


ORDER = ["續買", "回頭車", "下車", "續賣"]
NOTE = {"續買": "上週買 → 這週再加碼　★ 實測唯一明顯優於基準（4週 +15.7%）",
        "回頭車": "上週賣 → 這週買回　★ 實測最差（4週 +3.7%、勝率 54%），反指標",
        "下車": "上週買 → 這週賣　（4週 +7.6%，略低於基準 +8.2%）",
        "續賣": "上週賣 → 這週續賣　（4週 +6.5%）"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=1.0, help="大戶買賣的門檻（百分點）")
    ap.add_argument("--state", default=None, choices=ORDER)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--date", default=None, help="指定集保週別 YYYY-MM-DD")
    ap.add_argument("--tradeable", action="store_true", help="只留有個股期且有量的標的")
    ap.add_argument("--no-margin", action="store_true", help="不併融資（快一點）")
    args = ap.parse_args()

    df = chip_state.build(thr=args.thr, with_margin=not args.no_margin)
    if df.empty:
        print("集保快取是空的。先跑：python tdcc.py --latest")
        return

    day = df["date"].max() if args.date is None else pd.Timestamp(args.date)
    cur = df[df["date"] == day].copy()

    # 名稱：直接讀行情日檔，不要用 load_bars —— 後者會被 min_rows 過濾掉標的，
    # 造成部分個股沒有名字（實際踩過：雙鴻、精材、威剛都變空白）。
    import glob
    names = {}
    for p in sorted(glob.glob(str(twse_day_dir() / "*.parquet")))[-5:]:
        try:
            d = pd.read_parquet(p, columns=["code", "name"])
            names.update(dict(zip(d["code"], d["name"])))
        except Exception:
            pass
    cur["name"] = cur["code"].map(names).fillna("")

    if args.tradeable:
        import futures as fut
        ssf = fut.load(min_volume=500).set_index("code")
        cur = cur[cur["code"].isin(ssf.index)]
        cur["契約"] = cur["code"].map(ssf["Contract"])

    n_prev = df[df["date"] == day]["d_big400_prev"].notna().sum()
    print(f"══ 大戶換手　集保週別 {day:%Y-%m-%d} ══")
    print(f"母體 {len(cur)} 檔（{'僅個股期可交易' if args.tradeable else '全市場'}）"
          f"｜可判定狀態 {n_prev} 檔｜門檻 ±{args.thr:g}pp")
    counts = cur["狀態"].value_counts()
    print("　".join(f"{s} {int(counts.get(s, 0))}" for s in ORDER))

    cols = ["code", "name", "狀態", "升格", "d_big400_prev", "d_big400",
            "d_big1000", "d_retail", "融資增減%", "券資比"]
    if args.tradeable:
        cols.insert(2, "契約")
    ren = {"d_big400_prev": "上週Δ", "d_big400": "本週Δ",
           "d_big1000": "千張Δ", "d_retail": "散戶Δ"}

    for s in ([args.state] if args.state else ORDER):
        g = cur[cur["狀態"] == s].copy()
        if g.empty:
            print(f"\n── {s} ── 本週無標的")
            continue
        # 續買/回頭車 依本週增幅排序；下車/續賣 依減幅排序
        g = g.sort_values("d_big400", ascending=s in ("下車", "續賣"))
        print(f"\n── {s}（{len(g)} 檔）── {NOTE[s]}")
        out = g[cols].head(args.top).rename(columns=ren).round(2)
        print(out.to_string(index=False))

    if not args.state:
        best = cur[(cur["狀態"] == "續買") & (cur["升格"] == "中大戶升格")]
        print(f"\n★ 續買 × 中大戶升格（實測 4 週 +18.8%、中位 +12.3%、勝率 73%，n=63）："
              + ("、".join(f"{r['code']} {r['name']}" for _, r in best.iterrows())
                 if len(best) else "本週無"))
    print("\n★ 資料為公開籌碼統計，非投資建議。集保次週才公布，實務上要留公布時滯。")


if __name__ == "__main__":
    main()
