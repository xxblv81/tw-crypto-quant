"""
訊號計算 — 必須與 Pine 版 (taiwan/strategy.pine) 邏輯逐行一致。

★ 關鍵：縮口判斷用 <= 而非 <。
  Pine 的 ta.lowest() 含當根，用 < 會恆為 false（原始版本一筆單都不進的 bug）。
  pandas 的 rolling().min() 同樣含當根，所以這裡也必須用 <=。

兩個選配功能（2026-09-02 新增，**預設關閉**，台股行為完全不變）：

  trend_filter_2 / trend_ma2_length
      第二層趨勢濾網。原本只有一條 MA200，開啟後要求同時站上第二條均線。
      注意單位是「根數」不是「天數」：4H 圖上 60 根 = 10 天，60 天 = 360 根。

  require_bull_candle / min_body_ratio / min_close_pos
      「表態紅K」條件。收紅 + 實體佔振幅比例 + 收盤在當日區間的相對位置。
      預設關閉；台股 mid_reclaim 進場時建議開啟（見 taiwan/CANDLE_VOLUME.md）。

  allow_short
      做空訊號（多空鏡像：縮口後跌破下軌 + 在均線之下）。
      ★ CLAUDE.md 把做空列為已排除方向（v3 含做空 PF 0.848 vs 關掉 1.464）。
        這裡重新提供是為了在「加了第二層濾網」的新條件下重測，不是推翻舊結論。
        測完的數據見 crypto/SHORT_AND_MA60.md。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()      # Pine 的 ta.atr 用 RMA


def barssince(cond: pd.Series) -> pd.Series:
    """等同 Pine 的 ta.barssince：距離上次 True 幾根，從未發生為 NaN。"""
    idx = np.arange(len(cond))
    last = np.where(cond.values, idx, np.nan)
    last = pd.Series(last, index=cond.index).ffill()
    return pd.Series(idx, index=cond.index) - last


def compute(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """df 需含 Open/High/Low/Close/Volume，index 為日期，由舊到新。"""
    out = df.copy()
    c = out["Close"]

    basis = c.rolling(p["bb_length"]).mean()
    dev = p["bb_mult"] * c.rolling(p["bb_length"]).std(ddof=0)   # Pine ta.stdev 為母體標準差
    upper, lower = basis + dev, basis - dev
    bbw = (upper - lower) / basis * 100

    # 縮口：當根帶寬為區間最低（含當根，故用 <=）
    narrow = bbw <= bbw.rolling(p["bbw_lookback"]).min()
    bs_sq = barssince(narrow)
    squeeze_recent = bs_sq.le(p["squeeze_window"]).fillna(False)

    expanding = (bbw > bbw.shift(1)) if p["require_expanding"] else pd.Series(True, index=c.index)

    cross_up_mid = (c > basis) & (c.shift(1) <= basis.shift(1))

    if p["entry_mode"] == "upper_break":
        # break_on_high（2026-09-24）：改用最高價（含上影線）穿過上軌判定，
        # 仍在收盤後確認、隔根開盤進場。預設 false = 收盤穿越。
        src = out["High"] if p.get("break_on_high", False) else c
        trigger = (src > upper) & (src.shift(1) <= upper.shift(1))
        trigger_short = (c < lower) & (c.shift(1) >= lower.shift(1))   # 鏡像：跌破下軌
    elif p["entry_mode"] == "mid_confirm":
        # 兩段式進場：
        #   ① 縮口後，收盤向上穿越中線 → 進入候選狀態
        #   ② 之後出現「帶量表態紅K」才進場（穿越當根不算，見 mid_confirm_min_gap）
        #   候選狀態在 mid_cross_window 根內有效，且期間收盤須維持在中線之上；
        #   跌回中線下方即作廢，等下一次穿越重新計算。
        #
        # expand_mult（選配，預設 0 = 關閉）：額外要求帶寬已張開到縮口時的 N 倍。
        # ★ 實測有害：1.0→PF 1.288、1.2→1.247、1.5→1.233（單調下降），
        #   因為等帶寬明顯張開時，價格通常已經噴完一段。故預設不啟用。
        bs_cross = barssince(cross_up_mid)
        win = p.get("mid_cross_window", 10)
        min_gap = p.get("mid_confirm_min_gap", 1)
        trigger = (bs_cross.ge(min_gap) & bs_cross.le(win) & (c > basis)).fillna(False)

        expand_mult = p.get("expand_mult", 0)
        if expand_mult:
            bbw_at_sq = bbw.where(narrow).ffill()
            trigger = trigger & (bbw > bbw_at_sq * expand_mult).fillna(False)
        trigger_short = (c < basis) & (c.shift(1) >= basis.shift(1))
    else:                                                # mid_reclaim
        trigger = cross_up_mid
        trigger_short = (c < basis) & (c.shift(1) >= basis.shift(1))

    # ── 表態紅K（選配，2026-09-09）──
    # 「表態」= 不只是收紅，而是實體要夠大、收在當日高檔，代表買方真的掌控全場。
    #   body_ratio  = |收-開| / (高-低)   實體佔全日振幅多少
    #   close_pos   = (收-低) / (高-低)   收盤落在當日區間的相對位置
    o_, h_, l_ = out["Open"], out["High"], out["Low"]
    rng = (h_ - l_).replace(0, np.nan)
    body_ratio = (c - o_).abs() / rng
    close_pos = (c - l_) / rng
    if p.get("require_bull_candle", False):
        bull = (c > o_)
        if p.get("min_body_ratio", 0):
            bull = bull & (body_ratio >= p["min_body_ratio"])
        if p.get("min_close_pos", 0):
            bull = bull & (close_pos >= p["min_close_pos"])
        candle_ok = bull.fillna(False)
    else:
        candle_ok = pd.Series(True, index=c.index)
    out["body_ratio"] = body_ratio
    out["close_pos"] = close_pos

    # ── 多頭排列均線（選配）──
    # ma_stack: [5,10,20,60] 之類，要求 MA5 > MA10 > MA20 > MA60 且收盤在最上面。
    # 這是「趨勢結構完整」的濾網，比單一條均線嚴格得多。
    stack = p.get("ma_stack") or []
    if stack:
        mas = [c.rolling(int(n)).mean() for n in stack]
        stack_ok = (c > mas[0])
        for a, b in zip(mas, mas[1:]):
            stack_ok = stack_ok & (a > b)
        if p.get("ma_stack_rising", False):     # 再要求最長那條本身在走揚
            stack_ok = stack_ok & (mas[-1] > mas[-1].shift(5))
        stack_ok = stack_ok.fillna(False)
        out["ma_stack_ok"] = stack_ok
    else:
        stack_ok = pd.Series(True, index=c.index)

    trend_ma = c.rolling(p["trend_ma_length"]).mean()
    trend_ok = (c > trend_ma) if p["trend_filter"] else pd.Series(True, index=c.index)
    trend_dn = (c < trend_ma) if p["trend_filter"] else pd.Series(True, index=c.index)

    # 第二層趨勢濾網（選配）。單位是根數：4H 圖 60 根 = 10 天、360 根 = 60 天
    if p.get("trend_filter_2", False):
        ma2 = c.rolling(p.get("trend_ma2_length", 60)).mean()
        trend_ok = trend_ok & (c > ma2)
        trend_dn = trend_dn & (c < ma2)
        out["trend_ma2"] = ma2

    v = out["Volume"]
    if p["volume_filter"]:
        vma = v.rolling(p["volume_ma_length"]).mean()
        vol_ok = v > vma * p["volume_mult"]
    else:
        vol_ok = pd.Series(True, index=c.index)

    # ── 相對「穿越中線以來」的放量（選配）──
    # 只跟 5 日均量比不夠：如果整段候選期都在放量，那這根就不算表態。
    # 比的是「穿越那根之後、到昨天為止」的平均量（不含當根，避免自我比較）。
    # 全向量化：用 cumsum 差分算區間和，不跑迴圈。
    since_mult = p.get("volume_since_cross_mult", 0)
    if since_mult:
        pos = pd.Series(np.arange(len(c)), index=c.index)
        cross_pos = pos.where(cross_up_mid).ffill()            # 最近一次穿越的位置
        csum = v.cumsum().shift(1)                             # 到昨天為止的累計量
        csum_at_cross = csum.where(cross_up_mid).ffill()       # 穿越當根之前的累計量
        n_since = pos - cross_pos                              # 穿越至今幾根
        mean_since = (csum - csum_at_cross) / n_since.where(n_since > 0)
        vol_ok = vol_ok & (v > mean_since * since_mult).fillna(False)
        out["vol_mean_since_cross"] = mean_since

    if p["require_touch_lower"]:
        bs_touch = barssince(out["Low"] <= lower)
        touch_ok = bs_touch.le(p["touch_window"]).fillna(False)
    else:
        touch_ok = pd.Series(True, index=c.index)

    out["basis"], out["upper"], out["lower"], out["bbw"] = basis, upper, lower, bbw
    out["atr"] = atr(out, p["atr_length"])
    out["narrow"] = narrow
    out["bars_since_squeeze"] = bs_sq
    out["trend_ma"] = trend_ma
    out["bars_since_mid_cross"] = barssince(cross_up_mid)
    out["dist_to_upper_pct"] = (upper - c) / c * 100
    # ── 品質濾網（選配，2026-09-23）──
    # min_atr_pct：排除牛皮股。ATR 佔股價太低＝就算訊號對也沒有幅度可賺，
    #              而來回成本 0.585% 是固定的。
    # near_high_pct：要求收盤已接近 252 日高點，排除「還在低檔區掙扎」的標的。
    quality = pd.Series(True, index=c.index)
    if p.get("min_atr_pct", 0):
        quality = quality & (out["atr"] / c * 100 >= p["min_atr_pct"])
    if p.get("near_high_pct", 0):
        hi252 = c.rolling(252, min_periods=60).max()
        quality = quality & (c >= hi252 * p["near_high_pct"])
    quality = quality.fillna(False)

    _long = (squeeze_recent & expanding & trigger & trend_ok & vol_ok
             & touch_ok & candle_ok & stack_ok & quality).fillna(False)
    # allow_long 預設 True；設 False 是為了做「純做空」的拆解分析
    out["signal"] = _long if p.get("allow_long", True) else pd.Series(False, index=c.index)
    # 做空訊號（選配）。多空共用縮口／擴張／量能條件，只有方向相反
    if p.get("allow_short", False):
        out["signal_short"] = (squeeze_recent & expanding & trigger_short
                               & trend_dn & vol_ok).fillna(False)
    else:
        out["signal_short"] = pd.Series(False, index=c.index)
    # 待突破 = 「其他條件都到位，只差最後那根帶量表態紅K」。
    # ★ 必須跟 signal 用同一套條件，只放掉最後的觸發，否則兩邊在量不同的東西。
    #   舊版是「離上軌 0~3%」，那是上軌突破時代的定義，與兩段式無關。
    if p["entry_mode"] == "mid_confirm":
        out["watch"] = (squeeze_recent & expanding & trigger & trend_ok
                        & stack_ok & touch_ok & ~out["signal"]).fillna(False)
    else:
        out["watch"] = (squeeze_recent & trend_ok & ~out["signal"]
                        & out["dist_to_upper_pct"].between(0, 3)).fillna(False)
    return out
