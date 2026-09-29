"""
平日例行：回顧假日選股的實際表現 —— 累積中的樣本外紀錄。

與回測的差別（也是它的價值）：
    回測是「事後用今天的知識挑參數再跑歷史」，永遠有選擇者的知識污染。
    這支讀的是 weekly_pick.py **當週存下、事後未改**的清單，
    所以每過一週就多一筆真正的樣本外證據。

前瞻報酬從「集保公布後」起算（預設 +6 天），與回測口徑一致，不會有前視。

用法：
    python pick_review.py                # 全部
    python pick_review.py --weeks 8      # 只看最近 8 週
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

import momentum as mom
import twse
from twse import CACHE

PICK_DIR = CACHE / "picks"
LAG = pd.Timedelta(days=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=0)
    ap.add_argument("--tradeable-only", action="store_true")
    args = ap.parse_args()

    fs = sorted(glob.glob(str(PICK_DIR / "*.csv")))
    if not fs:
        print("還沒有任何選股紀錄。先跑：python weekly_pick.py")
        return
    if args.weeks:
        fs = fs[-args.weeks:]
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in fs], ignore_index=True)
    df["pick_date"] = pd.to_datetime(df["pick_date"])
    if args.tradeable_only:
        df = df[df["有個股期"] == True]           # noqa: E712

    bars, _ = twse.load_bars(adjust=True, min_rows=60)
    panel = mom.build_panel(bars)
    close, idx = panel["Close"], panel["Close"].index

    def fwd(d, c, n):
        if c not in close.columns:
            return np.nan
        st = idx[idx >= d + LAG]
        if len(st) < 2:
            return np.nan
        a = close.at[st[0], c]
        b = close.at[st[min(n, len(st) - 1)], c]
        done = len(st) > n                        # 是否已滿 n 根（未滿的標成未到期）
        if not (pd.notna(a) and pd.notna(b) and a > 0):
            return np.nan
        return (b / a - 1) * 100 if done else np.nan

    for n, lab in ((5, "1週"), (10, "2週"), (20, "4週")):
        df[lab] = [fwd(d, c, n) for d, c in zip(df["pick_date"], df["code"])]

    print(f"══ 假日選股累積成績 ══")
    print(f"紀錄 {df['pick_date'].nunique()} 週"
          f"（{df['pick_date'].min():%Y-%m-%d} ~ {df['pick_date'].max():%Y-%m-%d}）"
          f"｜{len(df)} 筆")
    ready = df["4週"].notna().sum()
    print(f"已滿 4 週可評估：{ready} 筆"
          + ("（樣本還太少，數字僅供追蹤）" if ready < 60 else ""))

    print(f"\n{'狀態':<8}{'n':>5}" + "".join(f"{l+'平均':>9}{l+'中位':>9}{l+'勝率':>8}"
                                            for l in ("1週", "2週", "4週")))
    for s in ("續買", "回頭車", "下車", "續賣"):
        g = df[df["狀態"] == s]
        if g.empty:
            continue
        row = f"{s:<8}{len(g):>5}"
        for l in ("1週", "2週", "4週"):
            x = g[l].dropna()
            if x.empty:
                row += f"{'—':>9}{'—':>9}{'—':>8}"
            else:
                row += f"{x.mean():>9.2f}{x.median():>9.2f}{(x > 0).mean()*100:>7.0f}%"
        print(row)

    up = df[(df["狀態"] == "續買") & (df["升格"].notna()) & (df["升格"] != "")]
    if len(up):
        x = up["4週"].dropna()
        print(f"\n續買 × 中大戶升格：{len(up)} 筆"
              + (f"，已滿 4 週 {len(x)} 筆　平均 {x.mean():+.2f}%　中位 {x.median():+.2f}%"
                 f"　勝率 {(x>0).mean()*100:.0f}%" if len(x) else "，尚無滿 4 週的"))
        print(f"  （回測基準：4 週 +18.83%、中位 +12.32%、勝率 73%，n=63）")

    print("\n★ 這是逐週累積的真實樣本外紀錄 —— 與回測不同，沒有事後挑參數的污染。")
    print("  樣本要兩三年才夠下定論；在那之前只當追蹤，不要據此改參數。")


if __name__ == "__main__":
    main()
