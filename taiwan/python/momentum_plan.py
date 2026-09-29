"""
執行表 —— 把選股結果算成「照著就能下單」的口數與保證金。

★ 這支不會、也不能幫你下單。它只把 markets.yaml 的規則套到今天的行情，
  算出「照這套規則做，部位會長什麼樣」。要不要做、實際下多少，是你的決定。

以**保證金預算**為主軸，不是以「總資金」——
個股期真正卡住你的是「今天能押多少保證金」，不是帳戶裡有多少錢。

兩個一定要看懂的數字：
    名目 / 權益     ＝ 實際槓桿。這決定你賺賠的放大倍數。
    保證金 / 預算   ＝ 保證金使用率。這決定你離追繳有多遠。
  兩者不是同一件事：保證金率 13.5% 的標的，用掉 27% 的預算就已經是 2 倍槓桿。

★ 契約規模有兩種，差 20 倍：
    一般股票期貨  一口 = 2,000 股
    小型股票期貨  一口 =   100 股   （契約名稱含「小型」，107 檔可用標的裡有 32 檔）
  逐檔取用 futures.py 快取裡的 contract_size，不要用單一常數。

用法：
    python momentum_plan.py --margin 250000
    python momentum_plan.py --margin 250000 --slots 5
    python momentum_plan.py --margin 250000 --date 2026-09-03
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
import futures as fut
import market
import momentum as mom
import twse

NAME = {0: "弱", 1: "中", 2: "強"}

# 25 萬保證金預算、每日最多新進 1 檔、契約規模已修正的實測（2019~2026）：
#   同時上限   CAGR    回撤    平均曝險
#       1     42.1%   98.3%    68.9%   ★★ 等於歸零，絕對不要
#       2     88.1%   80.8%    65.0%
#       3     71.9%   79.7%    60.2%
#       5     77.4%   58.0%    46.6%
#       8     63.5%   25.2%    24.9%   ← 風險調整最佳
#      12     31.3%   12.8%    12.2%
#      15     21.6%    7.6%     7.8%
# ★ 「一天最多進一檔」是安全的（中位持有 13 天，會自然累積到十幾檔）；
#   「同時只持有一檔」才是危險的 —— 單一部位在 1.5~2.5x 槓桿下沒有任何分散。
CONCURRENT_TABLE = [(1, 42.1, 98.3), (2, 88.1, 80.8), (3, 71.9, 79.7),
                    (5, 77.4, 58.0), (8, 63.5, 25.2), (12, 31.3, 12.8), (15, 21.6, 7.6)]


def suggest_slots(budget: float, lev: float, notionals: pd.Series,
                  hard_cap: int) -> tuple[int, str]:
    """
    依「今天候選的一口名目」決定實際能開幾個部位。

    ★ 真正的限制是**名目**不是保證金。個股期一口名目中位數約 16 萬，
      而目標名目配額 = 預算 × 槓桿 ÷ 部位數。25 萬 × 1.5x ÷ 5 = 7.5 萬 ——
      連中位數的一半都不到，五個部位裡有四個會填不滿。
      保證金反而寬鬆（一口保證金中位數僅約 3 萬），所以用保證金去推部位數會高估。
    """
    med = float(notionals.median())
    # 用「一口名目最小的那幾檔」估上限比用中位數合理 —— 實際下單會優先填得進去的。
    cheap = float(notionals.nsmallest(max(1, len(notionals) // 3)).median())
    n = int(budget * lev / cheap) if cheap > 0 else 1
    n = max(1, min(n, hard_cap))
    return n, (f"預算 × {lev:g}x = {budget*lev:,.0f}，"
               f"較便宜那批的一口名目約 {cheap:,.0f} → 上限 {n} 個部位")


def main() -> None:
    fp = config.params("taiwan_futures")
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=250_000,
                    help="今天可動用的保證金預算 (TWD)")
    ap.add_argument("--capital", type=float, default=None,
                    help="等同 --margin（舊名，保留相容）")
    ap.add_argument("--slots", type=int, default=None,
                    help="同時最多持有幾檔；不給就依保證金預算自動建議")
    ap.add_argument("--leverage", type=float, default=None,
                    help="手動指定槓桿；不給就依大盤狀態自動決定")
    ap.add_argument("--min-score", type=float, default=8.0)
    ap.add_argument("--max-new-per-day", type=int, default=1,
                    help="今天最多新進幾檔（預設 1）。★ 這與「同時持有上限」是兩回事")
    ap.add_argument("--no-near-high", action="store_true",
                    help="關掉「距 52 週高點」濾網（僅供對照，PF 會由 1.715 掉回 1.465）")
    ap.add_argument("--fit-only", action="store_true",
                    help="只列「一口就是一個完整部位」的標的（一口曝險落在目標槓桿附近）")
    ap.add_argument("--fit-range", type=float, nargs=2, default=(0.6, 1.4),
                    metavar=("LO", "HI"),
                    help="適配區間（一口曝險 ÷ 目標曝險），預設 0.6~1.4")
    ap.add_argument("--date", default=None)
    ap.add_argument("--ssf-min-volume", type=float, default=None)
    args = ap.parse_args()

    budget = args.capital if args.capital is not None else args.margin
    p = mom.normalize_params(config.params("taiwan_momentum"))
    rp = config.params("taiwan_market")

    bars, names = twse.load_bars(adjust=True, min_rows=260)
    panel = mom.build_panel(bars)
    idx = panel["Close"].index

    streak = None
    if p.get("use_chips"):
        import chips
        streak = chips.buy_streak(chips.load_trust_net())
        cov = chips.last_date()
        if streak.empty or (cov is not None and cov < idx[-1]):
            print(f"[warn] 籌碼快取只到 {cov:%Y-%m-%d} → 本次不計籌碼分。"
                  f"補：./.venv/bin/python chips.py --start 2019-01-01")
            p, streak = {**p, "use_chips": False}, None

    msk = mom.liquidity_mask(panel["Turnover"], p)
    brd = market.breadth(panel["Close"], msk, rp["breadth_ma"])
    rg = market.regime(market.load_index(), brd, rp, idx)
    sc = mom.compute_scores(panel, p, streak)

    day = idx[-1] if args.date is None else pd.Timestamp(args.date)
    if day not in idx:
        day = idx[idx <= day][-1]
    r = rg.loc[day]
    lvl = int(r["level"])
    levmap = {int(k): float(v) for k, v in fp["regime_leverage"].items()}
    lev = args.leverage if args.leverage is not None else levmap[lvl]

    print(f"══ 執行表　{day:%Y-%m-%d} ══")
    print(f"保證金預算 {budget:,.0f}　大盤 {NAME[lvl]}（寬度 {r['breadth']*100:.0f}%）"
          f"　目標槓桿 {lev:g}x"
          + ("（手動指定）" if args.leverage is not None else ""))

    if lev <= 0:
        print("\n★★ 大盤弱勢 —— 依設定不進新倉。")
        print("   實測此狀態下進場：245 筆、PF 0.848、平均 −0.63%（見 taiwan/MOMENTUM_REGIME.md）")
        print("   手上既有部位仍照原出場規則管理（跌破 MA20 或 −20% 或滿 14 個交易日）。")
        return

    minvol = args.ssf_min_volume if args.ssf_min_volume is not None else fp["ssf_min_volume"]
    ssf = fut.load(min_volume=minvol).set_index("code")
    row = sc["score"].loc[day].dropna()
    row = row[row >= args.min_score]
    cand = row.index.intersection(ssf.index)
    row = row.loc[cand].sort_values(ascending=False)
    close = panel["Close"].loc[day]
    ma_exit = sc["ma"][fp["exit_ma"]].loc[day]

    # ★ 進場位置：距 52 週高點 ≤ near_high_pct。回測顯示這是最乾淨的增額來源
    #   （PF 1.465→1.715、回撤 43.1%→30.5%，CAGR 幾乎不變）。
    hi250 = panel["High"].rolling(p["high_window"], min_periods=p["high_window"]).max().loc[day]
    dist_hi = ((hi250 - close) / close * 100)
    nh = fp.get("near_high_pct")
    if nh and not args.no_near_high:
        keep = [c for c in row.index if pd.notna(dist_hi.get(c)) and dist_hi[c] <= nh]
        dropped = [c for c in row.index if c not in keep]
        if dropped:
            print(f"\n距 52 週高點 >{nh:g}% 而濾掉 {len(dropped)} 檔："
                  + "、".join(f"{c} {names.get(c,'')}({dist_hi[c]:.1f}%)" for c in dropped[:8]))
        row = row.loc[keep]
        if row.empty:
            print(f"\n今天沒有「≥{args.min_score} 分、且距 52 週高點 ≤{nh:g}%」的標的。")
            print(f"　放寬：--no-near-high（會回到 PF 1.465 / 回撤 43% 的水準）"
                  f"　或 --min-score 7.0")
            return

    # 大戶持股（集保股權分散表）。★ 歷史只有 51 週，樣本不足以驗證 ——
    #   當即時加分欄位看，不要當已驗證的因子。細節見 tdcc.py 的說明。
    big_now, big_chg, tdcc_weeks = {}, {}, 0
    try:
        import tdcc
        raw = tdcc.load()
        if not raw.empty:
            h = tdcc.changes(tdcc.holders(raw), weeks=1)
            tdcc_weeks = h["date"].nunique()
            last = h[h["date"] == h["date"].max()].set_index("code")
            big_now = last["big"].to_dict()
            if tdcc_weeks >= 2:
                big_chg = last["big_chg1w"].to_dict()
    except Exception as e:
        print(f"[warn] 集保資料讀取失敗（{type(e).__name__}）")

    # ★ 期現價差已移除（2026-09-07，使用者要求）——
    #   期交所只給當日資料無法回測，且折價多半只反映結算前除息，價值有限。
    #   籌碼面改看 chip_screen.py 的大戶換手狀態。
    bs = pd.DataFrame()

    # 今天候選的一口名目 —— 部位數要由它決定，不是由保證金決定
    n1_all = pd.Series({c: float(close[c]) * int(ssf.loc[c, "contract_size"])
                        for c in row.index})
    if args.slots:
        slots, why = args.slots, "手動指定"
    else:
        slots, why = suggest_slots(budget, lev, n1_all, fp["slots"])
    print(f"同時持有上限 {slots}（{why}）　今天最多新進 {args.max_new_per_day} 檔")
    if slots <= 2:
        print(f"　★★ 同時只持有 {slots} 檔：25 萬預算下實測最大回撤 "
              f"{dict((c, d) for c, _, d in CONCURRENT_TABLE).get(slots, 98.3):.1f}% —— "
              f"1 檔 98.3%、2 檔 80.8%，等於出局。")
        print(f"　　「一天最多進一檔」沒問題（中位持有 13 天，會自然累積到十幾檔），"
              f"但別把同時持有也壓到 1~2 檔。25 萬建議 --slots 8（回撤 25.2%）。")

    # ★ 25 萬這種規模，分散與槓桿是直接衝突的，把取捨攤開來讓人自己選
    _cheap = float(n1_all.nsmallest(max(1, len(n1_all) // 3)).median())
    print(f"\n可行組合（預算 {budget:,.0f}，較便宜那批的一口名目約 {_cheap:,.0f}）：")
    print(f"  {'槓桿':>5}{'總名目':>12}{'放得下部位':>11}   說明")
    for L in (1.0, 1.5, 2.5, 4.0):
        n = max(1, int(budget * L / _cheap))
        note = ("回測預設" if abs(L - 1.5) < 1e-9 else
                "大盤強勢時的設定" if abs(L - 2.5) < 1e-9 else
                "分散夠但槓桿高，追繳風險大" if L >= 4 else "最保守")
        print(f"  {L:>4.1f}x{budget*L:>12,.0f}{n:>11}   {note}")
    print(f"  ★ {budget:,.0f} 這個規模，「分散」與「低槓桿」二選一"
          f" —— 個股期一口太大，兩者無法兼得。")

    target_notional = budget * lev            # 這次要建立的總名目
    # 單一部位的名目上限：不讓一檔吃掉整個額度。25 萬這種規模常常只放得下 1~2 檔，
    # 所以上限放寬到總額的 60%，否則會變成「什麼都買不了」。
    # fit-only 的定義就是「一口 ≈ 目標總曝險」，所以單一部位上限不能再壓到 60%，
    # 否則合身標的必然被自己擋掉。這是刻意放寬，代價是集中度上升（上面已警告）。
    # fit-only 的定義就是「一口 ≈ 目標總曝險」，單一部位上限要放到適配區間的上緣，
    # 否則合身標的（適配可到 1.4）會被自己擋掉。代價是集中度上升，上面已警告。
    pos_cap = target_notional * (args.fit_range[1] if args.fit_only
                                 else (1.0 if slots == 1 else 0.6))
    fair = target_notional / slots            # 公平份額，用來決定買幾口

    # ── 候選明細：每一檔「單獨買」的話能買幾口、要多少保證金 ──
    #   一天只做一檔時，自動配置沒有意義 —— 要看的是每個候選各自的規格，自己挑。
    detail = []
    for code in row.index:
        px = float(close[code])
        cs = int(ssf.loc[code, "contract_size"])
        mr = float(ssf.loc[code, "margin_pct"]) / 100
        n1, m1 = px * cs, px * cs * mr
        lots_solo = int(budget // m1)                     # 只買這一檔、用滿預算的口數
        detail.append({
            "code": code, "name": names.get(code, ""),
            "契約": ssf.loc[code, "Contract"],
            "型": "小型" if bool(ssf.loc[code, "is_mini"]) else "一般",
            "分數": round(float(row[code]), 2),
            "rs": round(float(sc["rs"].loc[day, code]), 1),
            "量能": round(float(sc["volume"].loc[day, code]), 1),
            "排列": round(float(sc["align"].loc[day, code]), 1),
            "位階": round(float(sc["position"].loc[day, code]), 1),
            "現價": round(px, 2),
            "一口名目": int(n1), "一口保證金": int(m1),
            "最多口數": lots_solo,
            "槓桿/口": round(n1 / budget, 2),
            # 適配 = 一口曝險 ÷ 目標曝險。1.0 代表「買一口剛好就是一個完整部位」。
            # 太小要買很多口才有意義；太大則一口就超重、想減也減不了（最小單位就是一口）。
            "適配": round(n1 / target_notional, 2),
            "距高%": round(float(dist_hi[code]), 1),
            "大戶%": round(big_now[code], 1) if code in big_now else float("nan"),
            "大戶Δ": round(big_chg[code], 2) if code in big_chg else float("nan"),
            # ★ 警示而非計分：訊號不對稱 —— 減持要避開，增持沒有明顯加分。
            #   實測 188 筆：減持<−0.25pp 那組 PF 0.859／中位 −4.78%，
            #   其餘 PF 2.665／中位 +4.50%。詳見 taiwan/MOMENTUM_HOLDERS.md
            "⚠": ("大戶減持" if (code in big_chg and big_chg[code] < -0.25) else ""),
            "停損價": round(px * (1 - fp["stop_loss_pct"] / 100), 2),
            f"出場MA{fp['exit_ma']}": round(float(ma_exit[code]), 2),
        })
    dt = pd.DataFrame(detail)
    lo, hi = args.fit_range
    dt["合身"] = ["★" if lo <= v <= hi else "" for v in dt["適配"]]
    shown = dt[dt["合身"] == "★"] if args.fit_only else dt
    print(f"\n── 候選明細（單獨買、預算 {budget:,.0f}、目標曝險 {target_notional:,.0f}）──")
    if shown.empty:
        print(f"　今天沒有適配落在 {lo}~{hi} 的標的。放寬：--fit-range 0.4 2.0，"
              f"或降門檻 --min-score 7.5 讓候選變多。")
    else:
        print(shown.to_string(index=False))
    print(f"\n　「適配」＝ 一口名目 ÷ 目標曝險（{target_notional:,.0f}）。")
    print(f"　　★ = 落在 {lo}~{hi}，**買一口就是一個完整部位** ← 你要的那種")
    print(f"　　< {lo}：一口太小，要買好幾口才有意義（部位會變成一堆零碎口數）")
    print(f"　　> {hi}：一口就超重，而且最小單位就是一口，想減也減不了")
    if tdcc_weeks:
        print(f"　「大戶%」＝集保 400 張以上持股比例，"
              f"「大戶Δ」＝較前一週的變動（百分點，正＝大戶在增持）。")
        if tdcc_weeks < 2:
            print(f"　　★ 目前只有 {tdcc_weeks} 週集保資料，還算不出變動 —— "
                  f"跑 ./.venv/bin/python tdcc.py --history 補歷史（約 2 小時）")
        else:
            print(f"　　⚠ 標記 = 大戶週變動 < −0.25 個百分點。實測 188 筆（僅 2025-09~2026-09、"
                  f"單一多頭年）：\n　　　　該組 PF 0.859／勝率 34%／中位 −4.78%，"
                  f"其餘 PF 2.665／勝率 55%／中位 +4.50%。")
            print(f"　　★ 但只有 {tdcc_weeks} 週、188 筆樣本，且做過多重比較 —— "
                  f"當警示看，不要當硬條件。增持那側沒有明顯加分（2.421 vs 基準 2.031）。")

    # --fit-only 時，自動配置也只從「合身」的候選裡挑，否則會挑到分數高但一口太小的
    alloc_order = list(shown["code"]) if args.fit_only and not shown.empty else list(row.index)

    # ★ 「一口就是一個完整部位」與「同時持有多檔」在小預算下直接衝突：
    #   合身標的一口名目 ≈ 目標總曝險，要同時持有 N 檔就需要 N 倍的權益。
    if not shown.empty:
        med_fit = float(shown["一口名目"].median())
        need = med_fit * slots / lev
        if need > budget * 1.2:
            print(f"\n★★ 這裡有個沒辦法兩全的地方：")
            print(f"   「合身」標的一口名目中位數 {med_fit:,.0f}，"
                  f"要同時持有 {slots} 檔需要權益約 {need:,.0f}。")
            print(f"   你目前 {budget:,.0f} —— 所以只能二選一：")
            print(f"     (a) 做南亞這種「一口＝一個部位」的，同時只持有 1~2 檔"
                  f"　→ 實測回撤 98.3% / 80.8%，等於出局")
            print(f"     (b) 同時持有 {slots} 檔以分散　→ 只能買一口很小的標的"
                  f"（小型契約、低價股），那就不是南亞這型")
            print(f"   要兩者兼得，權益要放大到約 {need:,.0f}。")

    rows, skipped = [], []
    left_n, left_m = target_notional, budget

    def fits(lots_: int, n1_: float, m1_: float) -> bool:
        """
        能不能放得下。兩道限制：
          名目 —— 總名目不得超過目標的 120%。★ 容忍度算在「總目標」上，
                  不是算在「剩餘額度」上，後者會越填越緊、讓後面的標的一律進不來。
          保證金 —— 不得超過剩餘預算，這條沒有容忍空間。
        """
        return ((target_notional - left_n) + lots_ * n1_ <= target_notional * 1.20
                and lots_ * m1_ <= left_m)

    for code in alloc_order:
        if len(rows) >= min(slots, args.max_new_per_day):
            break
        px = float(close[code])
        cs = int(ssf.loc[code, "contract_size"])
        mr = float(ssf.loc[code, "margin_pct"]) / 100
        n1, m1 = px * cs, px * cs * mr        # 一口名目、一口保證金

        lots = int(fair // n1)                # 先照公平份額
        if lots < 1:
            # 一口就超過公平份額 —— 只要沒超過單一部位上限、總額也還放得下，仍可拿 1 口。
            # 這是小資金的現實：不容許超過公平份額就會一檔都買不到。
            if n1 <= pos_cap and fits(1, n1, m1):
                lots = 1
            else:
                skipped.append((code, names.get(code, ""), n1, m1))
                continue
        while lots > 1 and (lots * n1 > pos_cap or not fits(lots, n1, m1)):
            lots -= 1
        if not fits(lots, n1, m1):
            skipped.append((code, names.get(code, ""), n1, m1))
            continue

        left_n -= lots * n1
        left_m -= lots * m1
        rows.append({
            "code": code, "name": names.get(code, ""),
            "契約": ssf.loc[code, "Contract"],
            "型": "小型" if bool(ssf.loc[code, "is_mini"]) else "一般",
            "分數": round(float(row[code]), 2),
            "現價": round(px, 2),
            "口數": int(lots),
            "名目": int(lots * n1),
            "保證金": int(round(lots * m1)),
            "停損價": round(px * (1 - fp["stop_loss_pct"] / 100), 2),
            f"出場MA{fp['exit_ma']}": round(float(ma_exit[code]), 2),
        })

    if not rows:
        print("\n★ 自動配置放不進任何部位 —— 但上面的候選明細仍可自己挑一檔做。")
    else:
        out = pd.DataFrame(rows)
        print(f"\n── 今天建議新進（最多 {args.max_new_per_day} 檔，"
              f"同時持有上限 {slots}）──")
        print(f"\n目標總名目 {target_notional:,.0f}（＝預算 × {lev:g}x）"
              f"　單一部位名目上限 {pos_cap:,.0f}\n")
        print(out.to_string(index=False))
        tot_n, tot_m = out["名目"].sum(), out["保證金"].sum()
        print(f"\n合計　名目 {tot_n:,.0f}　保證金 {tot_m:,.0f}")
        print(f"　　　實際槓桿 {tot_n/budget:.2f}x（目標 {lev:g}x）"
              f"　保證金使用率 {tot_m/budget*100:.1f}%")
        if len(rows) < slots:
            print(f"　　★ 只填了 {len(rows)}/{slots} 個部位 —— 今天符合條件的標的不夠，"
                  f"槓桿因此低於目標。這是正常的，不要為了填滿而放寬門檻。")
        if tot_m > budget * 0.8:
            print("　　★★ 保證金使用率超過 80%，逆走時幾乎沒有追繳緩衝。")

    if skipped:
        print(f"\n★ 放不進剩餘額度的 {len(skipped)} 檔：")
        for c, n, n1, m1 in skipped[:8]:
            print(f"   {c} {n}　一口名目 {n1:,.0f}　一口保證金 {m1:,.0f}")

    print(f"\n出場（三選一先到者）：收盤跌破 MA{fp['exit_ma']} → 隔日開盤出；"
          f"　跌到停損價 −{fp['stop_loss_pct']}%；　滿 {fp['max_hold_days']} 個交易日。")
    print("★ 本表是把 markets.yaml 的規則套到今天的行情，不是投資建議；下不下單由你決定。")


if __name__ == "__main__":
    main()
