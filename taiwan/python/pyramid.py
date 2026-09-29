"""
集保大戶週資料 —— 神秘金字塔（twsthr.info）。

為什麼用第三方而不是 TDCC 官方：
    TDCC 官方兩條路都不夠用（實測見 tdcc.py）：
      · opendata 整批檔只有**最新一週**，沒有歷史
      · 官網查詢頁能回溯約一年，但**只能單檔查** → 全市場要 56,000 次請求
    神秘金字塔把同一份 TDCC 資料整理成**單檔一次請求給 170 週**（約 3.3 年），
    全市場只要約 1,100 次請求。差了 50 倍。

    ★ 資料源頭仍是 TDCC 公開資料，但**經過第三方重新整理**。
      本檔已用 2330 對過帳：>1000張持有百分比 20260828 = 84.76（神秘金字塔）
      vs TDCC 官方 84.75，差 0.01 個百分點（四捨五入差異）。可用，但不是原始資料。

    ★ 這是第三方網站，不是官方 API：
      1. 版面隨時可能改，欄位對應會壞掉 —— 所以 parse 時驗表頭，壞了就炸。
      2. 網站靠廣告營運（首頁會偵測 AdBlock）。程式取用等於跳過廣告，
         如果這份資料對你有價值，站方有 donate.aspx，值得贊助。
      3. robots.txt 只有樣板前言，**沒有任何 Disallow、也沒設 content-signal 值**
         （2026-09-05 實測），依其自身說明等於「既未授權也未限制」。
         沒有明文禁止，但仍請維持低頻率：pause 預設 2 秒，不要調低。

欄位（比 TDCC 原始的 15 級距更好用）：
    holders        總股東人數          ← 人數減少＝籌碼集中，最直觀的指標
    avg_lots       平均張數/人         ← 同上，且已對股本正規化
    big400_pct     >400張大股東持有 %   ← 台股慣用的「大戶」定義
    big1000_pct    >1000張大股東持有 %  ← 千張大戶
    big400_holders >400張大股東人數
    n400_600 / n600_800 / n800_1000 / n1000up   各級距人數
    total_lots     集保總張數
    close          收盤價（站方自己的，本專案回測仍用 twse.py 的還原價）

★ 「大戶」含**保管銀行帳戶**（外資託管都掛在這裡），不等於自然人大戶。
  大戶比例跳動有時只是外資調整託管行，不是有人在收貨。

用法：
    python pyramid.py --build              # 建全市場快取，約 1,100 檔 / 37 分鐘
    python pyramid.py --codes 2330,2317    # 只抓指定標的
    python pyramid.py --top-week           # 全市場本週大戶增減排行（單一請求，不需快取）
    python pyramid.py --status
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

from twse import CACHE, UA, TIMEOUT

BASE = "https://norway.twsthr.info"
STOCK_URL = BASE + "/StockHolders.aspx?stock={code}"
TOPWEEK_URL = BASE + "/StockHoldersTopWeek.aspx"
PY_DIR = CACHE / "pyramid"

# 站方表頭 → 本專案欄名。★ 驗表頭用：對不上就是版面改了，寧可炸掉也不要默默存錯欄位。
COLMAP = {
    "資料日期": "date",
    "集保總張數": "total_lots",
    "總股東 人數": "holders",
    "平均張數/人": "avg_lots",
    ">400張大股東 持有張數": "big400_lots",
    ">400張大股東 持有百分比": "big400_pct",
    ">400張大股東 人數": "big400_holders",
    "400~600張人數": "n400_600",
    "600~800張人數": "n600_800",
    "800~1000張人數": "n800_1000",
    ">1000張人數": "n1000up",
    ">1000張大股東 持有百分比": "big1000_pct",
    "收盤價": "close",
}
NUMERIC = [v for v in COLMAP.values() if v != "date"]


def _path(code: str) -> Path:
    return PY_DIR / f"{code}.parquet"


def fetch_stock(code: str, session: requests.Session | None = None) -> pd.DataFrame | None:
    """單檔完整週歷史（實測約 170 週）。查無資料回 None。"""
    s = session or requests
    r = s.get(STOCK_URL.format(code=code), headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    try:
        tabs = [t for t in pd.read_html(io.StringIO(r.text))
                if t.shape[1] >= 14 and t.shape[0] > 20]
    except ValueError:
        return None
    if not tabs:
        return None

    t = tabs[0]
    head = [str(x).strip() for x in t.iloc[0].tolist()]
    missing = [k for k in COLMAP if k not in head]
    if missing:
        raise RuntimeError(f"神秘金字塔版面已變更，找不到欄位 {missing}；實際表頭 {head}")

    df = t.iloc[1:].copy()
    df.columns = head
    df = df[[k for k in COLMAP]].rename(columns=COLMAP)
    df = df[df["date"].notna()]
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip().str.slice(0, 8),
                                format="%Y%m%d", errors="coerce")
    df = df[df["date"].notna()]
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = code
    return df.sort_values("date").reset_index(drop=True) if len(df) else None


def build_cache(codes: list[str], pause: float = 2.0, refresh: bool = True,
                verbose: bool = True) -> int:
    """
    逐檔抓取。已有快取且最後一週不早於「目前最新週」的自動跳過。

    ★ refresh 預設 True：這份資料每週更新，只看檔案存不存在會永遠停在舊資料。
      判斷方式是拿第一檔抓到的最新日期當基準，其餘檔案落後就重抓。
    """
    PY_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    latest: pd.Timestamp | None = None
    n_ok = 0

    for k, code in enumerate(codes, 1):
        p = _path(code)
        if p.exists() and refresh and latest is not None:
            try:
                if pd.read_parquet(p, columns=["date"])["date"].max() >= latest:
                    continue
            except Exception:                      # noqa: BLE001  壞檔就重抓
                pass
        elif p.exists() and not refresh:
            continue

        try:
            df = fetch_stock(code, s)
        except RuntimeError:
            raise                                  # 版面變更要立刻知道，不要吞掉
        except requests.RequestException as e:
            if verbose:
                print(f"  [{k}/{len(codes)}] {code} 連線失敗：{e}", flush=True)
            time.sleep(pause * 3)
            continue

        if df is not None and len(df):
            df.to_parquet(p, index=False)
            n_ok += 1
            if latest is None:
                latest = df["date"].max()
                if verbose:
                    print(f"最新週次 {latest:%Y-%m-%d}（其餘檔案落後此日期就重抓）", flush=True)
        time.sleep(pause)
        if verbose and (k % 25 == 0 or k == len(codes)):
            print(f"  [{k}/{len(codes)}] 累計更新 {n_ok} 檔", flush=True)
    return n_ok


def fetch_top_week() -> pd.DataFrame | None:
    """
    全市場「本週大戶持股增減」排行 —— **一次請求**就有 1,100 檔，不需要建快取。
    想每週看一次輪動清單而不想維護快取的話，用這個就夠。
    回傳 code / name / cat / 最近 6 週增減 / 總增減 / 上週持有% / 收盤 / 漲跌。
    """
    r = requests.get(TOPWEEK_URL, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    tabs = [t for t in pd.read_html(io.StringIO(r.text)) if t.shape[0] > 100]
    if not tabs:
        return None
    t = tabs[0]
    t.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip()
                 if isinstance(c, tuple) else str(c) for c in t.columns]
    return t.dropna(how="all", axis=1)


def load_wide(field: str = "big400_pct") -> pd.DataFrame:
    """日期 × 代號 的寬表（週頻）。快取不存在時回空表。"""
    if not PY_DIR.exists():
        return pd.DataFrame()
    files = sorted(PY_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    parts = [pd.read_parquet(p, columns=["date", "code", field]) for p in files]
    raw = pd.concat(parts, ignore_index=True)
    return raw.pivot_table(index="date", columns="code", values=field,
                           aggfunc="last").sort_index()


def pct_point_change(wide: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """N 週的變化，單位是百分點（理由同 tdcc.py / qfiis.py）。"""
    return wide - wide.shift(weeks)


def last_week() -> pd.Timestamp | None:
    w = load_wide("big400_pct")
    return None if w.empty else w.index.max()


def main() -> None:
    ap = argparse.ArgumentParser(description="集保大戶週資料（神秘金字塔）")
    ap.add_argument("--build", action="store_true", help="建全市場快取（約 37 分鐘）")
    ap.add_argument("--codes", default=None, help="逗號分隔代號，只抓這些")
    ap.add_argument("--top-week", action="store_true",
                    help="全市場本週大戶增減排行（單一請求，不需快取）")
    ap.add_argument("--pause", type=float, default=2.0, help="請求間隔，別調低")
    ap.add_argument("--no-refresh", action="store_true", help="已有的檔案一律跳過")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        files = sorted(PY_DIR.glob("*.parquet")) if PY_DIR.exists() else []
        print(f"神秘金字塔快取：{len(files)} 檔")
        if files:
            w = load_wide("big400_pct")
            print(f"週次 {len(w)} 個｜{w.index.min():%Y-%m-%d} ~ {w.index.max():%Y-%m-%d}"
                  f"｜{sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
        return

    if args.top_week:
        df = fetch_top_week()
        if df is None:
            print("抓取失敗")
            return
        fn = "pyramid_topweek.csv"
        df.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"{len(df)} 檔 → {fn}")
        print(df.head(20).to_string(index=False))
        return

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.build:
        import twse
        bars, _ = twse.load_bars(adjust=True, min_rows=250)
        codes = sorted(c for c in bars if not c.startswith("00"))
        print(f"全市場 {len(codes)} 檔｜預估 {len(codes) * args.pause / 60:.0f} 分鐘")
    else:
        ap.error("請指定 --build / --codes / --top-week / --status")

    n = build_cache(codes, pause=args.pause, refresh=not args.no_refresh)
    print(f"完成，更新 {n} 檔。")


if __name__ == "__main__":
    main()
