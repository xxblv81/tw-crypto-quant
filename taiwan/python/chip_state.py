"""
大戶籌碼分層與換手狀態 —— 用集保股權分散表做「續買／回頭車／下車／續賣」分類。

對標使用者提供的旺來籌碼卡做法。定義（口徑寫死在這裡，改了要同步改回測）：

    大戶400   持股 400 張以上的比例 = 集保分級 12~15 的 pct 合計
    千張      持股 1,000 張以上的比例 = 分級 15
    散戶      未達 400 張 = 分級 1~11 的 pct 合計
    Δ         與前一週的變化（百分點，pp）

    狀態（以大戶400 的 Δ 判定，門檻 thr 預設 1.0pp，同卡片的「≧1pp」）：
        續買    上週 Δ ≥ +thr　且　本週 Δ ≥ +thr
        回頭車  上週 Δ ≤ −thr　且　本週 Δ ≥ +thr
        下車    上週 Δ ≥ +thr　且　本週 Δ ≤ −thr
        續賣    上週 Δ ≤ −thr　且　本週 Δ ≤ −thr

    中大戶升格  千張 Δ > 大戶400 Δ（增量集中在最高一層＝中實戶升格為千張大戶）
                ★ 這是我對卡片標籤的解讀，不是官方定義。

    融資增減%   融資餘額在兩個集保週別之間的變化率（2026-09-27 起含上櫃）
    融資增減張  同上，張數差

    ★ 資料異常（2026-09-27 發現，實際把 8 檔減資股誤判成「下車」）：
      減資換發股票那一週，集保會出現「股東 1 人、100% 在分級 15」的暫時快照，
      下一週股東數大減（零股被消掉）。兩週的 Δ 都不是大戶買賣。
      處理：(1) 總股東數 ≤ 1 的快照視為無效（值設 NaN，前後兩週的 Δ 自然變 NaN）；
            (2) 集保總股數與前一週相差 > SHARE_JUMP（減資／大額增資／合併），
                該週 Δ 設 NaN，標「股本異動」。
    券資比      融券餘額 ÷ 融資餘額 × 100

★ 資料限制（決定了能做什麼）：
    集保 opendata 只給最新一週；官網查詢只有 51 週且要逐檔逐週抓。
    本專案目前的歷史是「107 檔個股期可交易標的 × 51 週」+「全市場 × 最新一週」。
    → 全市場的週對週比較需要至少兩週全市場資料，要另外補抓。
      個股期標的則有 51 週，足以做初步驗證（但仍只有一年、單一多頭年）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIG400 = (12, 13, 14, 15)
BIG1000 = (15,)
RETAIL = tuple(range(1, 12))          # 1~11 = 未達 400 張
SHARE_JUMP = 0.10                     # 總股數週變動超過 10% → 股本異動，Δ 不可比


def layers(raw: pd.DataFrame) -> pd.DataFrame:
    """長表（date, code, level, pct）→ 每 (date, code) 的三層比例。"""
    if raw.empty:
        return raw
    d = raw[raw["level"].between(1, 15)].copy()
    d["pct"] = pd.to_numeric(d["pct"], errors="coerce")

    def agg(levels, name):
        return (d[d["level"].isin(levels)].groupby(["date", "code"])["pct"].sum()
                .rename(name))

    tot = raw[raw["level"] == 17].groupby(["date", "code"])[["people", "shares"]].sum()
    tot.columns = ["holders", "total_shares"]
    out = pd.concat([agg(BIG400, "big400"), agg(BIG1000, "big1000"),
                     agg(RETAIL, "retail"), tot], axis=1).reset_index()
    # 無效快照（減資換發期間：股東 1 人、全部在分級 15）→ 比例設 NaN，保留列讓 diff 斷開
    bad = out["holders"] <= 1
    out.loc[bad, ["big400", "big1000", "retail"]] = np.nan
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d")
    return out.sort_values(["code", "date"])


def with_state(lay: pd.DataFrame, thr: float = 1.0) -> pd.DataFrame:
    """加上週變動與換手狀態。thr 是「算不算買/賣」的門檻（百分點）。"""
    if lay.empty:
        return lay
    df = lay.sort_values(["code", "date"]).copy()
    for c in ("big400", "big1000", "retail"):
        df[f"d_{c}"] = df.groupby("code")[c].diff()
    # 股本異動：總股數跳動（減資、大額增資、合併）→ 前後比例不可比
    jump = (df.groupby("code")["total_shares"].pct_change().abs() > SHARE_JUMP)
    df["股本異動"] = np.where(jump, "股本異動", "")
    df.loc[jump, ["d_big400", "d_big1000", "d_retail"]] = np.nan
    df["d_big400_prev"] = df.groupby("code")["d_big400"].shift(1)

    cur, prev = df["d_big400"], df["d_big400_prev"]
    df["狀態"] = np.select(
        [(prev >= thr) & (cur >= thr),
         (prev <= -thr) & (cur >= thr),
         (prev >= thr) & (cur <= -thr),
         (prev <= -thr) & (cur <= -thr)],
        ["續買", "回頭車", "下車", "續賣"], default="")
    # 中大戶升格：增量集中在千張那一層（我的解讀，非官方定義）
    df["升格"] = np.where((df["d_big1000"] > df["d_big400"]) & (df["d_big1000"] > 0),
                          "中大戶升格", "")
    return df


def attach_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    併入融資增減%與券資比。融資是「日」資料，取每個集保週別當天（或之前最近一日）的值。
    """
    import margin
    mb = margin.load_wide("margin_bal")
    sb = margin.load_wide("short_bal")
    if mb.empty:
        df["融資增減%"], df["融資增減張"], df["券資比"] = np.nan, np.nan, np.nan
        return df

    def at(wide, dates):
        idx = wide.index
        pos = np.searchsorted(idx, dates, side="right") - 1
        pos = np.clip(pos, 0, len(idx) - 1)
        return wide.iloc[pos]

    dates = sorted(df["date"].unique())
    m_at, s_at = at(mb, dates), at(sb, dates)
    m_at.index, s_at.index = dates, dates

    def lookup(w, d, c):
        try:
            v = w.at[d, c]
            return float(v) if pd.notna(v) else np.nan
        except (KeyError, TypeError):
            return np.nan

    prev = {d: (dates[i - 1] if i else None) for i, d in enumerate(dates)}
    mv, mz, sv = [], [], []
    for d, c in zip(df["date"], df["code"]):
        cur_m = lookup(m_at, d, c)
        p = prev.get(d)
        pre_m = lookup(m_at, p, c) if p is not None else np.nan
        mv.append((cur_m / pre_m - 1) * 100 if (pre_m and pre_m > 0) else np.nan)
        mz.append(cur_m - pre_m)
        cur_s = lookup(s_at, d, c)
        sv.append(cur_s / cur_m * 100 if (cur_m and cur_m > 0) else np.nan)
    df["融資增減%"], df["融資增減張"], df["券資比"] = mv, mz, sv
    return df


def build(thr: float = 1.0, with_margin: bool = True) -> pd.DataFrame:
    """一步到位：讀集保快取 → 分層 → 狀態 → 併融資。"""
    import tdcc
    df = with_state(layers(tdcc.load()), thr=thr)
    return attach_margin(df) if (with_margin and not df.empty) else df
