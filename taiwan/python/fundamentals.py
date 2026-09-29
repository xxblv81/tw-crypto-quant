"""
基本面快取 —— 上市公司月營收（MOPS 靜態彙總檔）。

為什麼是月營收而不是 EPS／毛利率：
    「基本面不錯」最理想的量測是毛利率、EPS、ROE，但**拿不到歷史**：
      · mops.twse.com.tw 改版後的 /mops/api/*、openapi.twse.com.tw 於實測時
        TLS 連線直接被 reset（2026-09-03 實測，非本機限流 —— 同時間 mopsov 的
        靜態檔正常）。
      · 舊站 mopsov.twse.com.tw/mops/web/ajax_t163sb04（綜合損益表）同樣 reset，
        只有 /nas/ 底下的**靜態彙總檔**可取。
    月營收剛好是靜態檔，而且是台股最即時的基本面訊號（次月 10 日前公告），
    點時間乾淨、一個月一次請求就能拿到全市場。
    ★ 這是實質取捨：本策略的「基本面」只涵蓋成長性，不涵蓋獲利品質。
      要補毛利率／EPS 需要 FinMind token，見 CLAUDE.md。

點時間（point-in-time）處理 —— 這是本檔最重要的一件事：
    Y 年 M 月的營收，法規要求次月 10 日前公告。
    因此 available_date = (M+1 月 10 日)，在那之前的任何一天都**不可以**看到它。
    load_panel() 會把資料對齊到 available_date 之後才 forward-fill。
    弄錯這步整套回測就是前視偏誤，數字會漂亮但不可執行。

端點（一個月一個檔，全市場）：
    https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{民國年}_{月}_0.html   上市
    （sotc 目錄是上櫃，本專案預設不含上櫃）

用法：
    python fundamentals.py --start 2018-01     # 建快取，約 90 次請求
    python fundamentals.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests

from twse import CACHE, UA, TIMEOUT, _session

REV_URL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{y}_{m}_0.html"
REV_DIR = CACHE / "revenue"

# read_html 出來的欄位是兩層 MultiIndex，取第二層即可
COL_MAP = {
    "公司 代號": "code", "公司代號": "code",
    "公司名稱": "name",
    "當月營收": "rev",
    "上月營收": "rev_prev_m",
    "去年當月營收": "rev_prev_y",
    "當月累計營收": "cum_rev",
    "去年累計營收": "cum_rev_prev_y",
}


def _path(y: int, m: int) -> Path:
    return REV_DIR / f"{y:04d}-{m:02d}.parquet"


def avail_date(y: int, m: int) -> pd.Timestamp:
    """該月營收最早可以被看到的日期＝次月 10 日（法定公告期限）。"""
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return pd.Timestamp(ny, nm, 10)


def fetch_month(y: int, m: int) -> pd.DataFrame | None:
    """單月全市場營收。y/m 是西元年月。"""
    url = REV_URL.format(y=y - 1911, m=m)
    r = _session.get(url, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200 or len(r.content) < 5000:      # 未公告的月份是空殼頁
        return None
    r.encoding = "big5"
    try:
        tabs = pd.read_html(io.StringIO(r.text))
    except ValueError:
        return None

    parts = []
    for t in tabs:
        if t.shape[1] < 9:                                 # 產業別標題列，跳過
            continue
        t = t.copy()
        if isinstance(t.columns, pd.MultiIndex):           # 取第二層欄名
            t.columns = [str(c[-1]).strip() for c in t.columns]
        else:
            t.columns = [str(c).strip() for c in t.columns]
        cols = {c: COL_MAP[c] for c in t.columns if c in COL_MAP}
        if "code" not in cols.values():
            continue
        t = t[list(cols)].rename(columns=cols)
        parts.append(t)
    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.fullmatch(r"\d{4}")]            # 濾掉「合計」等彙總列
    for c in ("rev", "rev_prev_m", "rev_prev_y", "cum_rev", "cum_rev_prev_y"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("code").reset_index(drop=True)
    df["year"], df["month"] = y, m
    return df if len(df) else None


def build_cache(start: str, end: str | None = None, pause: float = 3.0,
                verbose: bool = True) -> int:
    """start/end 格式 'YYYY-MM'。已存在的月份自動跳過。"""
    REV_DIR.mkdir(parents=True, exist_ok=True)
    sy, sm = (int(x) for x in start.split("-"))
    today = dt.date.today()
    ey, em = (int(x) for x in end.split("-")) if end else (today.year, today.month)

    # ★ 尚未過公告期限的月份不能只抓一次就定案 —— 例如 9/3 抓 8 月營收只會拿到
    #   20~30 檔（早報的公司）。這種檔案每次都要重抓，直到過了次月 10 日為止，
    #   否則快取會永久停在一份殘缺的月份上。
    now = pd.Timestamp(today)
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        stale = _path(y, m).exists() and avail_date(y, m) > now
        if not _path(y, m).exists() or stale:
            months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    if verbose:
        print(f"待抓 {len(months)} 個月份（已快取的自動跳過）｜pause={pause}s", flush=True)
    n_ok = 0
    for k, (y, m) in enumerate(months, 1):
        try:
            df = fetch_month(y, m)
        except requests.RequestException as e:
            print(f"  [{k}] {y}-{m:02d} 失敗：{e}", flush=True)
            df = None
            time.sleep(pause * 3)
        if df is not None:
            df.to_parquet(_path(y, m), index=False)
            n_ok += 1
            if verbose:
                print(f"  [{k}/{len(months)}] {y}-{m:02d} {len(df)} 檔", flush=True)
        elif (y, m) < (today.year, today.month):
            _path(y, m).write_bytes(b"")                   # 確定不會再有的月份，佔位
        time.sleep(pause)
    return n_ok


def load_long() -> pd.DataFrame:
    """全部月份的長表，多一欄 avail_date（點時間可見日）。"""
    if not REV_DIR.exists():
        return pd.DataFrame()
    files = sorted(p for p in REV_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["period"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["avail_date"] = [avail_date(y, m) for y, m in zip(df["year"], df["month"])]
    return df.sort_values(["code", "period"]).reset_index(drop=True)


def revenue_features(long: pd.DataFrame, n_growth: int = 3) -> pd.DataFrame:
    """
    由長表算出各月的營收特徵（仍以 period 為列，之後才對齊到交易日）。

      yoy         當月營收年增率
      yoy_avg     近 3 個月年增率平均 —— 單月營收雜訊大（工作天數、拉貨節奏），
                  平均後才看得出趨勢
      cum_yoy     累計營收年增率
      pos_months  近 n_growth 個月裡年增率為正的月數（0~n）
    """
    if long.empty:
        return long
    d = long.copy()
    # ★ 這幾欄從 parquet 讀進來是 int64。int64 上做 .replace(0, pd.NA) 在 pandas 3.0
    #   會變成 object dtype，後面的 rolling 直接 DataError。先轉 float 再除。
    num = d[["rev", "rev_prev_y", "cum_rev", "cum_rev_prev_y"]].astype("float64")
    d["yoy"] = num["rev"] / num["rev_prev_y"].replace(0.0, float("nan")) - 1
    d["cum_yoy"] = num["cum_rev"] / num["cum_rev_prev_y"].replace(0.0, float("nan")) - 1
    # ★ 極端值要夾：實測 yoy 最大 94,578 倍、最小 −1,280 倍（極小基期或營收沖銷）。
    #   分數是百分位排名，本來就對離群值不敏感，但夾住可以讓「成長 300%」與
    #   「成長 900 萬 %」並列 —— 後者是資料假象不是基本面。
    d["yoy"] = d["yoy"].clip(-1.0, 3.0)
    d["cum_yoy"] = d["cum_yoy"].clip(-1.0, 3.0)
    g = d.groupby("code", sort=False)["yoy"]
    d["yoy_avg"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    d["pos_months"] = g.transform(
        lambda s: (s > 0).rolling(n_growth, min_periods=n_growth).sum())
    return d


def load_panel(index: pd.DatetimeIndex, columns, fields=("yoy", "yoy_avg", "cum_yoy",
                                                         "pos_months")
               ) -> dict[str, pd.DataFrame]:
    """
    對齊到交易日的日頻面板 {欄位: 日期 × 代號}。

    ★ 點時間：每筆資料從 avail_date（次月 10 日）當天起才可見，之前一律 NaN。
      這是整份檔案唯一不能寫錯的地方。
    """
    long = revenue_features(load_long())
    if long.empty:
        return {f: pd.DataFrame(index=index, columns=columns, dtype=float) for f in fields}
    out = {}
    for f in fields:
        wide = long.pivot_table(index="avail_date", columns="code", values=f, aggfunc="last")
        wide = wide.reindex(index=wide.index.union(index)).ffill().reindex(index)
        out[f] = wide.reindex(columns=columns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="建立上市公司月營收快取（MOPS 靜態檔）")
    ap.add_argument("--start", default="2018-01", help="YYYY-MM。要算年增率請比回測起點早一年")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        files = sorted(REV_DIR.glob("*.parquet")) if REV_DIR.exists() else []
        good = [p for p in files if p.stat().st_size > 0]
        print(f"月營收快取：{len(files)} 個月份檔，其中 {len(good)} 個有資料")
        if good:
            long = load_long()
            print(f"範圍：{good[0].stem} ~ {good[-1].stem}"
                  f"｜{long['code'].nunique()} 檔｜{len(long):,} 筆")
        else:
            print("尚未建立 —— washout 策略的基本面項無法計算")
        return

    n = build_cache(args.start, args.end, pause=args.pause)
    print(f"完成，{n} 個月份。")


if __name__ == "__main__":
    main()
