"""
融資融券餘額快取 —— TWSE MI_MARGN 每日全市場。

為什麼需要它：
    「籌碼清洗」在台股最可驗證的代理變數就是**融資餘額**。
    融資是散戶槓桿的直接量測 —— 融資減少而股價沒有同步破底，
    代表浮額被洗掉、籌碼從散戶手上換到不用槓桿的人手上。
    這是本專案第 3 套策略（低位階吸籌）的核心欄位，沒有它整套策略無從驗證。

    ★ 注意這是「代理」不是「實證」。真正想看的是集保大戶持股比例（TDCC），
      但 TDCC 的 opendata 只給最新一週（實測 getOD.ashx?id=1-5 只回 20260828 一天），
      沒有歷史就無法回測。融資是唯一免 token 又有完整歷史的籌碼面欄位。

架構與 chips.py 完全一致：逐交易日抓全市場，一天一個 parquet，可中斷可續跑。
    端點：https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=YYYYMMDD&selectType=ALL
    只涵蓋上市（TWSE），與本專案其餘部分一致。

欄位單位是「交易單位（張）」不是股數 —— 與 chips.py 的股數不同，不要混用。
本策略只用它的相對變化率，單位不影響。

用法：
    python margin.py --start 2019-01-01     # 建快取（已有的日期自動跳過，約 1.5 小時）
    python margin.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from twse import CACHE, _get_json, _num, trading_days  # noqa: F401

MI_MARGN = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
# 上櫃融資融券（2026-09-27 加）。欄名不重複，用名稱找 index。
TPEX_MARGIN = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
MARGIN_DIR = CACHE / "margin"

# MI_MARGN 的「融資融券彙總」表欄位名有重複（融資與融券各一組「今日餘額」），
# 用名稱找 index 會全部指到融資那組，所以這裡固定用位置，並在解析時驗證表頭。
I_CODE, I_FIN_BAL, I_SHORT_BAL = 0, 6, 12
EXPECT_HEAD = ["代號", "名稱", "買進", "賣出"]


def _day_path(d: dt.date) -> Path:
    return MARGIN_DIR / f"{d:%Y-%m-%d}.parquet"


def fetch_day(d: dt.date) -> pd.DataFrame | None:
    """單日全市場融資／融券餘額（張）。"""
    js = _get_json(MI_MARGN, {"date": d.strftime("%Y%m%d"),
                              "selectType": "ALL", "response": "json"})
    if not isinstance(js, dict) or js.get("stat") != "OK":
        return None
    tabs = [t for t in js.get("tables", []) if "彙總" in str(t.get("title", ""))]
    if not tabs or not tabs[0].get("data"):
        return None
    tab = tabs[0]
    head = [str(x).strip() for x in tab["fields"][:4]]
    if head != EXPECT_HEAD:                     # 欄位改版了 → 寧可不存，也不要存錯的
        raise RuntimeError(f"MI_MARGN 欄位與預期不符：{tab['fields']}")

    rows = []
    for r in tab["data"]:
        code = str(r[I_CODE]).strip()
        if len(code) != 4 or not code.isdigit():   # 只留 4 碼普通股
            continue
        rows.append({"code": code,
                     "margin_bal": _num(r[I_FIN_BAL]),
                     "short_bal": _num(r[I_SHORT_BAL])})
    return pd.DataFrame(rows) if rows else None


def fetch_tpex_day(d: dt.date) -> pd.DataFrame | None:
    """單日上櫃融資／融券餘額（張）。無資料（假日或未公布）回 None。"""
    js = _get_json(TPEX_MARGIN, {"date": d.strftime("%Y/%m/%d"), "response": "json"})
    tabs = js.get("tables") if isinstance(js, dict) else None
    if not tabs or not tabs[0].get("data"):
        return None
    f = [str(x).strip() for x in tabs[0]["fields"]]
    if "資餘額" not in f or "券餘額" not in f:
        raise RuntimeError(f"TPEx 融資融券欄位與預期不符：{f}")
    i_fin, i_short = f.index("資餘額"), f.index("券餘額")
    rows = [{"code": str(r[0]).strip(), "margin_bal": _num(r[i_fin]),
             "short_bal": _num(r[i_short])}
            for r in tabs[0]["data"]
            if len(str(r[0]).strip()) == 4 and str(r[0]).strip().isdigit()]
    return pd.DataFrame(rows) if rows else None


def _with_otc(d: dt.date, tw: pd.DataFrame) -> pd.DataFrame:
    """上市那天有資料才併上櫃；上櫃抓不到就只存上市（下次 --otc-backfill 可補）。"""
    tw = tw.assign(market="TWSE")
    try:
        tp = fetch_tpex_day(d)
    except Exception:
        tp = None
    return tw if tp is None else pd.concat([tw, tp.assign(market="TPEX")], ignore_index=True)


def otc_backfill(start: dt.date, end: dt.date, pause: float = 3.0) -> int:
    """把上櫃融資補進已存在的日檔（已含 TPEX 的跳過，可中斷續跑）。"""
    n = 0
    for p in sorted(MARGIN_DIR.glob("*.parquet")):
        d = dt.date.fromisoformat(p.stem)
        if not (start <= d <= end) or p.stat().st_size == 0:
            continue
        df = pd.read_parquet(p)
        if "market" in df.columns and (df["market"] == "TPEX").any():
            continue
        if "market" not in df.columns:
            df["market"] = "TWSE"
        tp = fetch_tpex_day(d)
        time.sleep(pause)
        if tp is None:
            print(f"  {d} 上櫃無資料，略過", flush=True)
            continue
        pd.concat([df, tp.assign(market="TPEX")], ignore_index=True).to_parquet(p, index=False)
        n += 1
        print(f"  {d} +上櫃 {len(tp)} 檔", flush=True)
    return n


def build_cache(start: dt.date, end: dt.date, pause: float = 3.0,
                verbose: bool = True) -> int:
    MARGIN_DIR.mkdir(parents=True, exist_ok=True)
    from twse import DAY_DIR

    def todo(d):
        p = _day_path(d)
        if not p.exists():
            return True
        # 空檔原本代表假日；但若行情快取那天有資料，代表當時只是抓失敗／還沒公布 → 重抓
        # （2026-09-27 發現 15 天被錯佔位，包括 2026-09-03/07/14/21）
        q = DAY_DIR / f"{d:%Y-%m-%d}.parquet"
        return p.stat().st_size == 0 and q.exists() and q.stat().st_size > 0

    days = [d for d in trading_days(start, end) if todo(d)]
    if verbose:
        print(f"待抓 {len(days)} 個交易日（已快取的自動跳過）｜pause={pause}s"
              f"｜預估 {len(days) * pause / 60:.0f} 分鐘", flush=True)
    n_ok = 0
    for k, d in enumerate(days, 1):
        df = fetch_day(d)
        if df is not None:
            df = _with_otc(d, df)
            df.to_parquet(_day_path(d), index=False)
            n_ok += 1
        else:
            _day_path(d).write_bytes(b"")          # 空檔＝假日，佔位避免重抓
        time.sleep(pause)
        if verbose and (k % 20 == 0 or k == len(days)):
            print(f"  [{k}/{len(days)}] {d} 累計有效 {n_ok} 天", flush=True)
    return n_ok


def load_wide(field: str = "margin_bal") -> pd.DataFrame:
    """日期 × 代號 的寬表。快取不存在時回空表。field: margin_bal / short_bal"""
    if not MARGIN_DIR.exists():
        return pd.DataFrame()
    files = sorted(p for p in MARGIN_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        return pd.DataFrame()
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        df["date"] = pd.Timestamp(p.stem)
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    return raw.pivot_table(index="date", columns="code", values=field, aggfunc="sum")


def main() -> None:
    ap = argparse.ArgumentParser(description="建立融資融券餘額快取（TWSE MI_MARGN）")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--pause", type=float, default=3.0, help="每次請求間隔秒數，別調低")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--otc-backfill", action="store_true",
                    help="把上櫃融資補進 --start~--end 已存在的日檔")
    args = ap.parse_args()

    if args.status:
        files = sorted(MARGIN_DIR.glob("*.parquet")) if MARGIN_DIR.exists() else []
        good = [p for p in files if p.stat().st_size > 0]
        print(f"融資融券快取：{len(files)} 個日期檔，其中 {len(good)} 天有資料")
        if good:
            print(f"範圍：{good[0].stem} ~ {good[-1].stem}"
                  f"｜{sum(p.stat().st_size for p in good) / 1e6:.1f} MB")
        else:
            print("尚未建立 —— washout 策略無法執行")
        return

    if args.otc_backfill:
        n = otc_backfill(dt.date.fromisoformat(args.start),
                         dt.date.fromisoformat(args.end), pause=args.pause)
        print(f"完成，補上櫃 {n} 天。")
        return

    n = build_cache(dt.date.fromisoformat(args.start),
                    dt.date.fromisoformat(args.end), pause=args.pause)
    print(f"完成，{n} 個交易日。")


if __name__ == "__main__":
    main()
