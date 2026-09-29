"""
低位階籌碼清洗評分引擎 —— 本專案第 3 套策略。

想抓的東西（使用者原話的拆解）：
    「低位階」          價格離高點還遠，但已經止跌、在打底
    「籌碼清洗得差不多」融資（散戶槓桿）退潮，浮額洗掉了
    「基本面不錯」      月營收年增率是正的、而且不是單月僥倖
    「市場未發酵」      量能還沒放大、近期還沒起漲
    「零星法人或單一投資人大幅度看好」
                        法人在這種沒量的盤裡默默吃貨 ——
                        關鍵不是買超「金額大」，而是買超**佔當日成交量的比重高**。
                        沒量的股票被吃掉三成量能，那就是有人在收。

★ 這套策略與本專案的動能選股**方向相反**，而且與既有證據相衝：
    · 動能策略的 position 分項是「越貼近 52 週高點越高分」，本策略反過來。
    · P0 全市場回測（taiwan/P0_RESULTS.md）顯示 PF 隨市值單調遞增，
      低位階、冷門、小型股是最差的那一區。
  所以本策略的先驗是**負面的**，不是中性的。寫它是因為使用者要驗證這個想法，
  不是因為既有證據支持它。任何結果出來前不要當它可用。

架構刻意與 momentum.py 對齊（同樣的面板、同樣先濾流動性再做橫斷面百分位），
理由見 momentum.py 的說明：絕對門檻會變成擇時，橫斷面排名才是選股。

分數與硬門檻分工 —— 這是與動能策略最大的設計差異：
    動能策略幾乎全部用分數（因為「強」是連續的、可以排名）。
    本策略要抓的是一個**狀態**（在底部、沒發酵、有人在收），
    狀態不成立時排名再高也沒意義，所以位階帶、止跌、未發酵、營收為正
    這四項做成**硬門檻（gate）**，分數只在通過門檻的池子裡排名。

資料依賴：
    行情       twse.py 快取（必要）
    融資餘額   margin.py 快取（必要 —— 沒有它整個 washout 分項是空的）
    法人買賣超 chips.py 快取（必要 —— accumulation 分項）
    月營收     fundamentals.py 快取（必要 —— fundamental 分項與營收門檻）
    以上任一缺席時，對應分項會退化成 0 分並印出警告，總分仍算得出來但意義不同。
"""
from __future__ import annotations

import pandas as pd

from momentum import build_panel, liquidity_mask, _pct_score  # noqa: F401  同一套面板/百分位


# ─────────────────────────── 硬門檻 ───────────────────────────
def position_band(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                  p: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    位階帶 + 止跌。回傳 (位階帶, 止跌, pos) 三者分開。

    pos = 收盤 / 一年最高。門檻要的是一個**帶狀區間**不是「越低越好」：
      · pos 太高（> pos_hi）＝已經漲上去了，那是動能策略的守備範圍
      · pos 太低（< pos_lo）＝還在跌，接刀。低位階不等於便宜

    止跌條件：近 low_window 日的最低價要比一年最低價高出 low_margin。
    這一條是「打底」與「還在破底」的分界，沒有它整組會塞滿下跌中的股票。
    """
    hi = high.rolling(p["high_window"], min_periods=p["high_window"]).max()
    lo_year = low.rolling(p["high_window"], min_periods=p["high_window"]).min()
    lo_recent = low.rolling(p["low_window"], min_periods=p["low_window"]).min()

    pos = close / hi
    in_band = (pos >= p["pos_lo"]) & (pos <= p["pos_hi"])
    stopped_falling = lo_recent >= lo_year * (1 + p["low_margin"])
    # ★ 兩半分開回傳。消融實驗必須能分辨「位階帶沒用」與「止跌沒用」——
    #   綁在一起測，拆掉 band 會同時拿掉兩者，結論就說不清是哪一半的功勞。
    return in_band, stopped_falling, pos


def dormant_gate(close: pd.DataFrame, turnover: pd.DataFrame,
                 p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    「市場未發酵」。回傳 (gate, vol_ratio)。

    兩條都要成立：
      · 量能還沒放大 —— vol_fast/vol_slow 均額比 ≤ vol_ratio_max
      · 近期還沒起漲 —— recent_window 日報酬 ≤ recent_run_max

    ★ 這是整套策略的定義性條件，也是它最大的執行風險：
      未發酵代表**不知道要等多久**，可能永遠不發酵。回測的時間出場
      （max_hold_days）就是在量測這件事的代價。
    """
    fast = turnover.rolling(p["vol_fast"], min_periods=p["vol_fast"]).mean()
    slow = turnover.rolling(p["vol_slow"], min_periods=p["vol_slow"]).mean()
    vr = fast / slow.replace(0, pd.NA)
    run = close / close.shift(p["recent_window"]) - 1
    return ((vr <= p["vol_ratio_max"]) & (run <= p["recent_run_max"])), vr


# ─────────────────────────── 分項 ───────────────────────────
def washout_score(margin_bal: pd.DataFrame, close: pd.DataFrame,
                  mask: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    籌碼清洗強度。回傳 (分數, 融資變化率)。

    ★ 這一項的定義是本策略最關鍵的判斷，不要簡化成「融資減少就好」：
      融資 −30% 而股價也 −30%，那是**一起崩**，籌碼沒有換手，只是大家一起賠錢跑。
      融資 −30% 而股價只 −5%，才是**清洗** —— 賣壓被不用槓桿的人接走了。
    所以量測的是「融資跌幅**超出**股價跌幅多少」：
        washout = (股價變化) − (融資變化)     ← 兩者都取同一個回看期
    數字越大代表融資退得比股價兇，越像洗籌。再做全市場百分位。

    融資餘額為 0 或極小（幾乎沒有信用交易）的股票會被 min_margin_lots 濾掉：
    分母太小時變化率是雜訊，且「沒有融資」本來就談不上「清洗融資」。
    """
    n = p["margin_window"]
    m_chg = margin_bal / margin_bal.shift(n).replace(0, pd.NA) - 1
    p_chg = close / close.shift(n) - 1
    raw = p_chg - m_chg

    enough = margin_bal.shift(n) >= p["min_margin_lots"]
    return _pct_score(raw, mask & enough.reindex_like(mask).fillna(False)), m_chg


def accumulation_score(trust: pd.DataFrame, foreign: pd.DataFrame,
                       volume: pd.DataFrame, mask: pd.DataFrame,
                       p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    「零星法人／單一投資人大幅度看好」。回傳 (分數, 吸籌比重)。

    量測方式 —— 法人淨買佔成交量的比重，不是金額：
        intensity = Σ(投信+外資淨買股數, N 日) / Σ(成交股數, N 日)
    為什麼是比重：使用者要的是「單一投資人**大幅度**看好」，
    在一檔沒量的冷門股上，能吃掉一成、兩成量能就是大幅度；
    同樣的金額放到台積電上什麼都不是。比重會自動把規模正規化掉。

    ★ 這是代理不是實證。真正想看的是集保大戶持股比例（TDCC 千張大戶），
      但 TDCC 的 opendata 只回最新一週、沒有歷史（2026-09-03 實測），
      無法回測。法人買超佔量比是唯一有完整歷史又免 token 的替代品。
      它抓得到「投信/外資默默吃貨」，抓不到「非法人的單一大戶」。
    """
    n = p["accum_window"]
    net = trust.fillna(0) + foreign.fillna(0)
    num = net.rolling(n, min_periods=n // 2).sum()
    den = volume.rolling(n, min_periods=n // 2).sum().replace(0, pd.NA)
    intensity = num / den
    return _pct_score(intensity, mask), intensity


def fundamental_score(rev: dict[str, pd.DataFrame], mask: pd.DataFrame,
                      p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基本面。回傳 (分數, gate)。

    用近 3 個月營收年增率平均（yoy_avg）排名 —— 單月營收受工作天數、
    拉貨節奏影響很大，平均後才是趨勢。
    gate：yoy_avg > min_yoy 且近 3 月至少 min_pos_months 個月正成長。

    ★ 涵蓋範圍限制：只有成長性，沒有獲利品質（毛利率／EPS／負債）。
      原因是 MOPS 改版後季報端點連不上，只有月營收靜態檔可取，
      見 fundamentals.py 的說明。「基本面不錯」在這裡＝「營收在成長」，
      一家虧損但營收成長的公司同樣會通過。這是已知的缺口。
    """
    yoy_avg = rev["yoy_avg"].reindex_like(mask)
    pos_m = rev["pos_months"].reindex_like(mask)
    gate = (yoy_avg > p["min_yoy"]) & (pos_m >= p["min_pos_months"])
    return _pct_score(yoy_avg, mask), gate.fillna(False)


def base_score(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
               mask: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    打底品質 —— 波動收斂程度。

    近 low_window 日的振幅（高低差 / 收盤）相對於一年振幅越小，代表價格
    在窄幅整理、賣壓消化完了。這與布林縮口策略的直覺是同一件事，
    只是這裡不要求突破，要的是「還在縮」。分數越高＝縮得越緊。
    """
    n, N = p["low_window"], p["high_window"]
    rng_s = (high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min())
    rng_l = (high.rolling(N, min_periods=N).max() - low.rolling(N, min_periods=N).min())
    tight = 1 - (rng_s / rng_l.replace(0, pd.NA))
    return _pct_score(tight, mask)


# ─────────────────────────── 總分 ───────────────────────────
def compute_scores(panel: dict[str, pd.DataFrame], p: dict,
                   margin_bal: pd.DataFrame | None = None,
                   trust: pd.DataFrame | None = None,
                   foreign: pd.DataFrame | None = None,
                   rev: dict[str, pd.DataFrame] | None = None,
                   verbose: bool = True) -> dict:
    """
    回傳 {"score", "gate", "eligible", 各分項, "ma", "diag"}。

    score  0~10，只在 eligible（流動性 + 四道硬門檻）為 True 處有值，其餘 NaN。
    ★ 分數的百分位是在「流動性池」裡算的，不是在「通過門檻的池」裡算 ——
      否則門檻一變，分數尺度就跟著變，跨設定不可比較。
    """
    close, high, low = panel["Close"], panel["High"], panel["Low"]
    turnover, volume = panel["Turnover"], panel["Volume"]
    mask = liquidity_mask(turnover, p)

    posband_gate, stopfall_gate, pos = position_band(close, high, low, p)
    dorm_gate, vr = dormant_gate(close, turnover, p)

    parts: dict[str, pd.DataFrame] = {}
    zero = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    diag: dict[str, object] = {}

    if margin_bal is not None and not margin_bal.empty:
        mb = margin_bal.reindex(index=close.index, columns=close.columns)
        parts["washout"], m_chg = washout_score(mb, close, mask, p)
        diag["margin_chg"] = m_chg
    else:
        if verbose:
            print("[warn] 沒有融資快取 → washout 分項為 0（先跑 margin.py）")
        parts["washout"] = zero

    if trust is not None and not trust.empty:
        t = trust.reindex(index=close.index, columns=close.columns)
        f = (foreign if foreign is not None else trust * 0).reindex(
            index=close.index, columns=close.columns)
        parts["accum"], intensity = accumulation_score(t, f, volume, mask, p)
        diag["accum_intensity"] = intensity
    else:
        if verbose:
            print("[warn] 沒有法人快取 → accum 分項為 0（先跑 chips.py）")
        parts["accum"] = zero

    if rev is not None and not rev["yoy_avg"].empty:
        parts["fund"], fund_gate = fundamental_score(rev, mask, p)
    else:
        if verbose:
            print("[warn] 沒有營收快取 → fund 分項為 0、營收門檻不生效"
                  "（先跑 fundamentals.py）")
        parts["fund"] = zero
        fund_gate = pd.DataFrame(True, index=close.index, columns=close.columns)

    parts["base"] = base_score(close, high, low, mask, p)

    w = dict(p["weights"])
    total = sum(parts[k] * w[k] for k in w)

    # disable_gates：門檻消融實驗用。拆掉某一道，看它是幫忙還是扯後腿。
    #   「整個想法錯」與「其中一個零件錯」是兩種完全不同的結論，
    #   沒有這個開關就分不出來。
    off = set(p.get("disable_gates") or ())
    if "band" in off:                                  # 舊寫法：一次關掉位階帶與止跌
        off |= {"posband", "stopfall"}
    ones = pd.DataFrame(True, index=close.index, columns=close.columns)
    gate = ((ones if "posband" in off else posband_gate.fillna(False))
            & (ones if "stopfall" in off else stopfall_gate.fillna(False))
            & (ones if "dormant" in off else dorm_gate.fillna(False))
            & (ones if "fund" in off else fund_gate))
    eligible = mask & gate

    ma = {n: close.rolling(n, min_periods=n).mean() for n in p["ma_set"]}
    out = {"score": total.where(eligible), "score_all": total.where(mask),
           "eligible": eligible, "mask": mask, "ma": ma, "weights": w,
           "pos": pos, "vol_ratio": vr,
           "gates": {"posband": posband_gate, "stopfall": stopfall_gate,
                     "dormant": dorm_gate, "fund": fund_gate},
           "diag": diag}
    out.update(parts)
    return out


def normalize_params(p: dict) -> dict:
    """YAML 讀進來的型別統一，順便檢查權重加總。"""
    p = dict(p)
    p["ma_set"] = [int(n) for n in p["ma_set"]]
    s = sum(p["weights"].values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"markets.yaml[taiwan_washout].params.weights 加總 {s}，應為 1.0")
    return p


def load_inputs(bars: dict[str, pd.DataFrame], verbose: bool = True):
    """把四份快取一次載齊，回傳 (panel, margin_bal, trust, foreign, rev)。"""
    panel = build_panel(bars, fields=("Open", "High", "Low", "Close",
                                      "Volume", "Turnover"))
    close = panel["Close"]

    try:
        import margin as mg
        margin_bal = mg.load_wide("margin_bal")
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"[warn] 融資快取讀取失敗：{e}")
        margin_bal = pd.DataFrame()

    try:
        import chips
        raw = chips.load_trust_net()
        trust = raw
        foreign = _load_foreign()
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"[warn] 法人快取讀取失敗：{e}")
        trust = foreign = pd.DataFrame()

    try:
        import fundamentals as fd
        rev = fd.load_panel(close.index, close.columns)
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"[warn] 營收快取讀取失敗：{e}")
        rev = None

    return panel, margin_bal, trust, foreign, rev


def _load_foreign() -> pd.DataFrame:
    """chips.py 的快取裡本來就存了外資欄，但它只導出投信，這裡自己讀。"""
    import chips
    files = sorted(p for p in chips.CHIP_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        return pd.DataFrame()
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        if "foreign_net" not in df.columns:
            continue
        df["date"] = pd.Timestamp(p.stem)
        parts.append(df[["date", "code", "foreign_net"]])
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    return raw.pivot_table(index="date", columns="code", values="foreign_net", aggfunc="sum")
