"""
動能選股回測 —— 在下任何結論前，先讓它跑出數字。

進出場完全照愛德恩公開講過的規則，沒有額外調參：
    進場  動能分數由下往上穿過 entry_score（預設 7.5）→ 隔日開盤買
    出場  收盤跌破 5 日線 → 隔日開盤賣
    停損  盤中觸及 −20% → 該價出場（作者說「停損不超過 20%」）
    漲停  隔日全天最低價已在漲停 → 買不到，略過（同 backtest.py 的 limit_up_guard）

兩種績效口徑，兩個都看：
  1. 逐筆彙總（pooled）—— 所有標的所有交易混在一起算 PF / 勝率 / 期望值。
     這回答「這個訊號本身有沒有邊際」。
  2. 投組模擬（portfolio）—— 同時最多持有 N 檔、等權重、分數高者優先。
     這回答「實際只能拿 N 個位置時，剩下多少」。
     ★ 一定要看第 2 個：某些日子有 80 檔同時達標，逐筆彙總等於假設你全買，
       那不是可執行的績效。

用法：
    python momentum_backtest.py                          # 預設 8 個持股位置
    python momentum_backtest.py --slots 5 --min-score 8
    python momentum_backtest.py --start 2022-01-01
    python momentum_backtest.py --no-limit-up-guard      # 對照組，會高估
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import config
import momentum as mom
import twse


def extract_trades(panel: dict[str, pd.DataFrame], score: pd.DataFrame,
                   ma_exit: pd.DataFrame, p: dict, thr: float,
                   limit_up_guard: bool, limit_pct: float = 0.095,
                   entry_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    對每一檔掃出所有交易。回傳逐筆明細（含進場分數，投組模擬要用它排序）。

    entry_mask：額外的進場條件（同形狀布林表），例如「投信當天仍在連買」。
    只擋進場，不影響出場 —— 用來測「把籌碼當過濾條件」而不是「當分數的一部分」。
    """
    idx = panel["Close"].index
    max_hold = int(p.get("max_hold_days") or 0)
    stop_frac = 1 - p["stop_loss_pct"] / 100.0
    rows = []

    for code in score.columns:
        sc = score[code].to_numpy(dtype=float)
        o = panel["Open"][code].to_numpy(dtype=float)
        l = panel["Low"][code].to_numpy(dtype=float)
        c = panel["Close"][code].to_numpy(dtype=float)
        ma = ma_exit[code].to_numpy(dtype=float)
        n = len(sc)

        # ★ 面板是全市場外連結，個股上市前／下市後那些日子是 NaN。
        #   不把掃描範圍限制在有效區間，出場會走到 NaN 去，算出 NaN 報酬。
        valid = np.flatnonzero(np.isfinite(c))
        if valid.size < 2:
            continue
        lo, hi = int(valid[0]), int(valid[-1])

        # 上穿門檻：前一根在門檻下（NaN 不算），這一根站上
        prev = np.roll(sc, 1)
        prev[0] = np.nan
        cross = (sc >= thr) & (prev < thr)
        if entry_mask is not None:
            cross &= entry_mask[code].to_numpy(dtype=bool)

        i = lo
        while i < hi:
            if not cross[i] or not np.isfinite(o[i + 1]):
                i += 1
                continue
            # 隔日鎖漲停買不到 —— 用比例判定，不受除權息還原影響
            if limit_up_guard and np.isfinite(c[i]) and c[i] > 0 \
                    and l[i + 1] >= c[i] * (1 + limit_pct):
                i += 1
                continue

            entry = o[i + 1]
            if not (entry > 0):
                i += 1
                continue
            stop = entry * stop_frac
            exit_px, j = None, i + 1
            while j <= hi:
                if np.isfinite(l[j]) and l[j] <= stop:        # 保守：先判停損
                    exit_px = stop
                    break
                # 收盤跌破出場均線 → 隔日開盤出場（進場當根不判，否則買進即賣出）
                if j > i + 1 and np.isfinite(ma[j]) and np.isfinite(c[j]) and c[j] < ma[j] \
                        and j + 1 <= hi and np.isfinite(o[j + 1]):
                    exit_px = o[j + 1]
                    j += 1
                    break
                if max_hold and (j - i) >= max_hold and j + 1 <= hi and np.isfinite(o[j + 1]):
                    exit_px = o[j + 1]
                    j += 1
                    break
                j += 1
            if exit_px is None or not np.isfinite(exit_px):    # 走到有效區間末端仍在場內
                j = hi
                exit_px = c[hi]
            if not (exit_px > 0):
                i += 1
                continue

            rows.append({"code": code, "entry_i": i + 1, "exit_i": j,
                         "entry_date": idx[i + 1], "exit_date": idx[j],
                         "entry": entry, "exit": exit_px, "score": sc[i],
                         "ret_gross": exit_px / entry - 1,
                         "hold_days": j - (i + 1)})
            i = j + 1                                         # 出場後才找下一個訊號
    return pd.DataFrame(rows)


def pooled_stats(ret: np.ndarray) -> dict:
    gp, gl = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return {
        "trades": len(ret),
        "win_rate_pct": round(float((ret > 0).mean() * 100), 2),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else float("inf"),
        "avg_ret_pct": round(float(ret.mean() * 100), 3),
        "median_ret_pct": round(float(np.median(ret) * 100), 3),
        "avg_win_pct": round(float(ret[ret > 0].mean() * 100), 2) if (ret > 0).any() else 0.0,
        "avg_loss_pct": round(float(ret[ret < 0].mean() * 100), 2) if (ret < 0).any() else 0.0,
    }


def simulate_portfolio(tr: pd.DataFrame, index: pd.DatetimeIndex,
                       slots: int, leverage: float = 1.0,
                       lev_col: str | None = None) -> tuple[pd.Series, dict]:
    """
    等權重、最多同時 slots 檔、同日多個候選取分數高者。
    部位規模用固定比例 leverage/slots：出場時 equity *= (1 + ret × leverage/slots)。

    lev_col：逐筆槓桿的欄名（大盤狀態動態調整時用）。給了就以該欄為準、忽略 leverage；
             該欄為 0 的交易直接不進場（弱勢停手）。槓桿在**進場當下**決定，之後不變。

    ★ 這是簡化：真實個股期在權益跌破維持保證金時會被強制平倉，本模擬沒有這段，
      權益數學上不會歸零。所以 2.5x 以上的數字要當上限看，不是預期值。
    """
    by_entry = {i: g for i, g in tr.groupby("entry_i")}
    equity, curve = 1.0, np.ones(len(index))
    open_pos: list[tuple[int, float, float]] = []   # (exit_i, ret, 權重)
    taken = 0
    filled = 0                                      # 累計「已用倉位數」，用來算平均曝險
    w = leverage / slots

    for i in range(len(index)):
        for pos in [q for q in open_pos if q[0] == i]:
            equity *= 1 + pos[1] * pos[2]
            open_pos.remove(pos)
        if i in by_entry and len(open_pos) < slots:
            cand = by_entry[i].sort_values("score", ascending=False)
            for _, r in cand.iterrows():
                if len(open_pos) >= slots:
                    break
                lv = float(r[lev_col]) if lev_col else leverage
                if lv <= 0:                      # 弱勢：略過，但不佔用倉位
                    continue
                open_pos.append((int(r["exit_i"]), float(r["ret_net"]), lv / slots))
                taken += 1
        filled += len(open_pos)
        if equity <= 0:                          # 權益歸零 = 出局
            curve[i:] = 0.0
            break
        curve[i] = equity

    eq = pd.Series(curve, index=index)
    dd = float((1 - eq / eq.cummax()).max() * 100)
    years = (index[-1] - index[0]).days / 365.25
    cagr = ((eq.iloc[-1] ** (1 / years) - 1) * 100
            if years > 0 and eq.iloc[-1] > 0 else -100.0)
    # ★ 平均曝險：空倉的日子是 0% 報酬。同樣的 CAGR，曝險 60% 與 100% 意義完全不同 ——
    #   低回撤有一部分只是「常常空手」，不是選股選得好。看回撤時一定要一起看這個。
    exposure = filled / max(len(index) * slots, 1) * 100
    return eq, {"slots": slots, "trades_taken": taken,
                "avg_exposure_pct": round(float(exposure), 1),
                "trades_skipped_no_slot": len(tr) - taken,
                "total_return_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
                "cagr_pct": round(float(cagr), 2),
                "max_dd_pct": round(dd, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--slots", type=int, default=8, help="同時最多持有幾檔")
    ap.add_argument("--start", default=None, help="回測起日（分數仍用完整歷史計算）")
    ap.add_argument("--no-limit-up-guard", action="store_true",
                    help="關掉漲停買不到的過濾（僅供對照，會高估）")
    ap.add_argument("--exit-ma", type=int, default=None,
                    help="出場均線，預設取 markets.yaml 的 exit_ma（5）")
    ap.add_argument("--stop-pct", type=float, default=None, help="硬停損百分比")
    ap.add_argument("--max-hold", type=int, default=None, help="最長持有天數，0=不限")
    ap.add_argument("--include-otc", action="store_true")
    ap.add_argument("--futures", action="store_true",
                    help="個股期貨短線模式：改用 markets.yaml[taiwan_futures] 的成本、"
                         "持有上限與 point-in-time 宇宙，並印出槓桿表")
    ap.add_argument("--leverage", type=float, default=None,
                    help="投組模擬的槓桿倍數，預設取 taiwan_futures 的 max_leverage")
    args = ap.parse_args()

    p = mom.normalize_params(config.params("taiwan_momentum"))
    cost = config.costs("taiwan_momentum")["commission_pct_per_side"]
    fp = None
    if args.futures:
        # 期貨模式：成本、出場、持有上限、部位數全部改吃 taiwan_futures
        fp = config.params("taiwan_futures")
        cost = config.costs("taiwan_futures")["commission_pct_per_side"]
        p = {**p, "exit_ma": fp["exit_ma"], "max_hold_days": fp["max_hold_days"],
             "stop_loss_pct": fp["stop_loss_pct"], "entry_score": fp["entry_score"]}
        if args.slots == 8:                      # 使用者沒指定就用設定檔的
            args.slots = fp["slots"]
    thr = args.min_score if args.min_score is not None else p["entry_score"]
    guard = p.get("limit_up_guard", True) and not args.no_limit_up_guard

    bars, names = twse.load_bars(include_otc=args.include_otc, adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    streak = None
    if p.get("use_chips"):
        import chips
        streak = chips.buy_streak(chips.load_trust_net())
        cov = chips.last_date()
        need = panel["Close"].index[-1]
        if streak.empty:
            print("[warn] use_chips=true 但籌碼快取是空的 → 本次不計籌碼分")
            p, streak = {**p, "use_chips": False}, None
        elif cov is not None and cov < need:
            # 籌碼落後時全市場當天都是 0 分，總分上限被壓到 (1-w)×10 → 訊號會憑空消失。
            # 回測只影響最後幾天，但仍然要講，否則數字會被誤讀。
            print(f"[warn] 籌碼快取只到 {cov:%Y-%m-%d}、行情到 {need:%Y-%m-%d}"
                  f" → 最後 {(need - cov).days} 天的籌碼分為 0，該區間訊號會偏少")

    if args.exit_ma is not None:
        p = {**p, "exit_ma": args.exit_ma, "ma_set": sorted(set(p["ma_set"]) | {args.exit_ma})}
    if args.stop_pct is not None:
        p = {**p, "stop_loss_pct": args.stop_pct}
    if args.max_hold is not None:
        p = {**p, "max_hold_days": args.max_hold}

    sc = mom.compute_scores(panel, p, streak)
    ma_exit = sc["ma"][p["exit_ma"]]

    # 期貨模式的可交易宇宙：★ 用 point-in-time 代理，不能用「今天有量的個股期清單」
    #   —— 後者是前視偏誤，實測會把 MA20 的 PF 從 1.711 灌水到 2.428。
    entry_mask = None
    if fp is not None:
        to60 = panel["Turnover"].rolling(60, min_periods=60).mean()
        entry_mask = to60.rank(axis=1, ascending=False) <= fp["pit_universe_rank"]
        # ★ 進場位置濾網：距 52 週高點 ≤ near_high_pct
        nh = fp.get("near_high_pct")
        if nh:
            hi = panel["High"].rolling(p["high_window"], min_periods=p["high_window"]).max()
            entry_mask = entry_mask & (((hi - panel["Close"]) / panel["Close"] * 100) <= nh)
        if fp.get("require_squeeze_breakout"):
            import signals as sig_mod
            pt = config.params("taiwan")
            brk = pd.DataFrame(False, index=panel["Close"].index,
                               columns=panel["Close"].columns)
            for code, df_ in bars.items():
                if code in brk.columns:
                    try:
                        brk.loc[:, code] = (sig_mod.compute(df_, pt)["signal"]
                                            .reindex(brk.index).fillna(False).values)
                    except Exception:
                        pass
            entry_mask = entry_mask & brk

    print(f"universe {panel['Close'].shape[1]} 檔"
          f"｜{panel['Close'].index[0]:%Y-%m-%d} ~ {panel['Close'].index[-1]:%Y-%m-%d}"
          f"｜門檻 {thr} 分｜出場 MA{p['exit_ma']}｜停損 {p['stop_loss_pct']}%"
          + (f"｜持有上限 {p['max_hold_days']} 天｜PIT 前 {fp['pit_universe_rank']} 大"
             if fp else "")
          + (f"｜距高 ≤{fp['near_high_pct']:g}%" if fp and fp.get("near_high_pct") else "")
          + f"｜成本 {cost}%/邊"
          f"｜漲停過濾 {'開' if guard else '關 ★'}")

    tr = extract_trades(panel, sc["score"], ma_exit, p, thr, guard, entry_mask=entry_mask)
    if tr.empty:
        print("沒有任何交易")
        return
    tr["ret_net"] = tr["ret_gross"] - 2 * cost / 100

    idx = panel["Close"].index
    if args.start:
        keep = tr["entry_date"] >= pd.Timestamp(args.start)
        tr = tr[keep].reset_index(drop=True)
        start_i = int(np.searchsorted(idx, pd.Timestamp(args.start)))
        idx_sim, tr = idx[start_i:], tr.assign(entry_i=tr["entry_i"] - start_i,
                                               exit_i=tr["exit_i"] - start_i)
    else:
        idx_sim = idx

    # ── 大盤狀態：動態調整槓桿與「弱勢停手」 ──
    lev_col = None
    if fp is not None and fp.get("use_regime"):
        import market
        rp = config.params("taiwan_market")
        msk = mom.liquidity_mask(panel["Turnover"], p)
        brd = market.breadth(panel["Close"], msk, rp["breadth_ma"])
        rg = market.regime(market.load_index(), brd, rp, panel["Close"].index)
        lv = rg["level"].to_numpy()
        levmap = {int(k): float(v) for k, v in fp["regime_leverage"].items()}
        base_i = tr["entry_i"].to_numpy() + (start_i if args.start else 0)
        tr["regime"] = lv[base_i]
        tr["lev"] = [levmap[int(x)] for x in tr["regime"]]
        lev_col = "lev"
        nm = {0: "弱", 1: "中", 2: "強"}
        print("\n大盤狀態分組（依進場當天）")
        for k in sorted(nm):
            g = tr[tr["regime"] == k]["ret_net"].to_numpy()
            if not len(g):
                continue
            pf_k = g[g > 0].sum() / max(-g[g < 0].sum(), 1e-9)
            print(f"  {nm[k]}　{len(g):>5} 筆　PF {pf_k:.3f}　"
                  f"勝率 {(g > 0).mean() * 100:.1f}%　平均 {g.mean() * 100:+.2f}%　"
                  f"→ 槓桿 {levmap[k]}x" + ("（停手）" if levmap[k] == 0 else ""))

    tr.to_csv("momentum_trades.csv", index=False, encoding="utf-8-sig")

    ps = pooled_stats(tr["ret_net"].to_numpy())
    print("\n【1】逐筆彙總（假設每個訊號都買得到、沒有部位上限）")
    for k, v in ps.items():
        print(f"  {k:>18} = {v}")
    print(f"  {'median_hold_days':>18} = {tr['hold_days'].median():.0f}")
    if fp is not None:
        inwin = tr["hold_days"].between(3, 14).mean() * 100
        print(f"  {'落在 3~14 天':>18} = {inwin:.0f}%")

    eq, port = simulate_portfolio(tr, idx_sim, args.slots,
                                  leverage=(1.0 if lev_col else 1.0), lev_col=lev_col)
    print(f"\n【2】投組模擬（最多同時 {args.slots} 檔、等權、分數高者優先）")
    for k, v in port.items():
        print(f"  {k:>22} = {v}")

    if fp is not None:
        lev0 = args.leverage if args.leverage is not None else fp["max_leverage"]
        print(f"\n【3】槓桿表（{args.slots} 檔等權）"
              f"　★ 未模擬追繳／斷頭，2.5x 以上實際會更差")
        if lev_col:
            print("  （固定槓桿對照組 —— 不分大盤狀態、全程同一倍數）")
        for lev in (1.0, 1.5, 2.0, 2.5, 3.0):
            eqL, portL = simulate_portfolio(tr, idx_sim, args.slots, lev)
            star = "  ← 舊的固定 max_leverage" if abs(lev - lev0) < 1e-9 else ""
            print(f"  {lev:>4.1f}x   CAGR {portL['cagr_pct']:>6.1f}%"
                  f"   最大回撤 {portL['max_dd_pct']:>5.1f}%{star}")
        if lev_col:
            lm = {int(k): float(v) for k, v in fp["regime_leverage"].items()}
            print(f"\n  ★ 動態（弱{lm[0]:g} / 中{lm[1]:g} / 強{lm[2]:g}）"
                  f"   CAGR {port['cagr_pct']:>6.1f}%"
                  f"   最大回撤 {port['max_dd_pct']:>5.1f}%   ← 實際採用")

    print("\n分年（逐筆彙總口徑）")
    yr = tr.groupby(tr["entry_date"].dt.year)["ret_net"].agg(
        n="size", win=lambda x: round((x > 0).mean() * 100, 1),
        avg=lambda x: round(x.mean() * 100, 2),
        pf=lambda x: round(x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9), 2))
    print(yr.to_string())

    eq.to_frame("equity").to_csv("momentum_equity.csv", encoding="utf-8-sig")
    print("\n→ momentum_trades.csv / momentum_equity.csv")


if __name__ == "__main__":
    main()
