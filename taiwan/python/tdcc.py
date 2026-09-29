"""
集保股權分散表（TDCC）—— 大戶持股比例與週變動。

為什麼要它：
    融資餘額（margin.py）是散戶槓桿的代理，但真正想看的是「籌碼往誰手上跑」。
    集保每週公布持股分級，400 張以上的比例就是大戶持股 —— 這是直接量測，不是代理。

★★ 兩個資料限制，決定了它只能怎麼用：
    1. opendata（getOD.ashx?id=1-5）**只給最新一週**，一次全市場，免 token。
    2. 網頁查詢（qryStock）有歷史，但**只有 51 週（約一年）**，而且要逐檔逐週查
       （需 CSRF token）。107 檔 × 51 週 ≈ 5,457 次請求。
    → 一年的歷史配上本策略每年約 200 筆交易，**樣本太薄，做不了可信的回測**。
      所以定位是「即時選股的加分欄位」，不是「已驗證的因子」。
      要能驗證，得從現在開始逐週累積，等兩三年後才有足夠樣本。

持股分級（實測與網頁版對照無誤）：
    1~15  各級距（15 = 1,000,001 股以上）
    16    差異數
    17    合計
    大戶(400張以上) = 級距 12~15；大戶(1000張以上) = 級距 15

用法：
    python tdcc.py --latest          # 抓最新一週（1 次請求，秒殺）
    python tdcc.py --history         # 抓 51 週歷史（僅個股期可交易標的，約 2 小時）
    python tdcc.py --status
"""
from __future__ import annotations

import argparse
import io
import re
import time

import pandas as pd
import requests

from twse import CACHE

OPENDATA = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
QRY = "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TDCC_DIR = CACHE / "tdcc"

BIG_LEVELS = (12, 13, 14, 15)      # 400 張以上
HUGE_LEVEL = 15                    # 1,000 張以上


def _day_path(d: str):
    return TDCC_DIR / f"{d}.parquet"


def fetch_latest() -> pd.DataFrame:
    """opendata：最新一週全市場。回傳長表 (date, code, level, people, shares, pct)。"""
    r = requests.get(OPENDATA, headers=UA, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "code", "level", "people", "shares", "pct"]
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.len() == 4]          # 只留 4 碼普通股
    return df


def _session():
    s = requests.Session()
    s.headers.update(UA)
    return s


def _form(s, tries: int = 5):
    """
    取回 CSRF token 與可查詢的週別清單。

    ★ 一定要有重試：這支要連續跑上萬次，一次 ChunkedEncodingError／連線重置
      就會把幾小時的工作整個打斷（實測發生過）。
    """
    delay = 2.0
    for k in range(tries):
        try:
            r = s.get(QRY, timeout=60)
            r.raise_for_status()
            break
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    else:
        raise RuntimeError("取不到表單")
    tok = re.search(r'name="SYNCHRONIZER_TOKEN"\s+value="([^"]*)"', r.text)
    uri = re.search(r'name="SYNCHRONIZER_URI"\s+value="([^"]*)"', r.text)
    dates = re.findall(r'<option[^>]*value="(\d{8})"', r.text)
    return (tok.group(1) if tok else ""), (uri.group(1) if uri else "/portal/zh/smWeb/qryStock"), dates


def fetch_one(s, tok, uri, code: str, date: str) -> pd.DataFrame | None:
    """網頁查詢：單一個股單一週。"""
    payload = {"SYNCHRONIZER_TOKEN": tok, "SYNCHRONIZER_URI": uri, "method": "submit",
               "firDate": "", "scaDate": date, "sqlMethod": "StockNo",
               "stockNo": code, "stockName": ""}
    for k in range(3):
        try:
            rr = s.post(QRY, data=payload, timeout=60)
            tbs = pd.read_html(io.StringIO(rr.text))
            break
        except Exception:
            if k == 2:
                return None
            time.sleep(2.0 * (k + 1))
    for t in tbs:
        if t.shape[0] >= 16 and "占集保庫存數比例 (%)" in " ".join(map(str, t.columns)):
            t = t.iloc[:, :5]
            t.columns = ["level", "band", "people", "shares", "pct"]
            t = t[pd.to_numeric(t["level"], errors="coerce").notna()]
            t["level"] = t["level"].astype(int)
            t["code"], t["date"] = code, date
            return t[["date", "code", "level", "people", "shares", "pct"]]
    return None


def build_history(codes: list[str], pause: float = 1.0, verbose: bool = True) -> int:
    """
    逐檔逐週抓，一週一個 parquet（累積各股）。可中斷可續跑。

    ★★ SYNCHRONIZER_TOKEN 是**一次性**的：同一個 token 送第二次查詢會回空表，
       而且 HTTP 仍是 200、不報錯 —— 靜默失效。
       實測：同 token 連三次 → (17,6)/None/None；每次換 token → 三次都成功。
       所以**每一筆查詢前都要重新 GET 表單拿新 token**，成本是 2 個請求換 1 筆資料。
       （初版每 200 次才換一次，結果 5,350 次請求只抓到每週 1 檔。）
    """
    TDCC_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    tok, uri, dates = _form(s)
    if verbose:
        print(f"可查週別 {len(dates)} 個：{dates[-1]} ~ {dates[0]}"
              f"｜標的 {len(codes)} 檔"
              f"｜預估 {len(codes)*len(dates)*(pause+0.6)/3600:.1f} 小時"
              f"（每筆 2 個請求：換 token + 查詢）")
    n = 0
    for di, d in enumerate(dates, 1):
        p = _day_path(d)
        have = set()
        if p.exists():
            try:
                have = set(pd.read_parquet(p, columns=["code"])["code"])
            except Exception:
                have = set()
        todo = [c for c in codes if c not in have]
        if not todo:
            if verbose:
                print(f"  [{di}/{len(dates)}] {d} 已完整，跳過", flush=True)
            continue
        def flush(parts):
            """把已抓到的落地。★ 每 20 檔存一次 —— 不然中途掛掉整週白做。"""
            if not parts:
                return
            old = pd.read_parquet(p) if p.exists() else None
            out = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
            out.drop_duplicates(["date", "code", "level"], keep="last").to_parquet(p, index=False)

        parts = []
        for k, c in enumerate(todo, 1):
            try:
                tok, uri, _ = _form(s)      # ★ 每筆都要新 token，見 docstring
                t = fetch_one(s, tok, uri, c, d)
            except Exception:
                flush(parts)
                parts = []
                time.sleep(10.0)            # 連 _form 都重試失敗 → 對方可能在限流
                continue
            if t is not None:
                parts.append(t)
            n += 1
            if len(parts) >= 20:
                flush(parts)
                parts = []
            time.sleep(pause)
        flush(parts)
        if verbose:
            print(f"  [{di}/{len(dates)}] {d} 新增 {len(parts)}/{len(todo)} 檔"
                  f"（累計請求 {n}）", flush=True)
    return n


def load() -> pd.DataFrame:
    """讀所有已快取的週別，回傳長表。"""
    if not TDCC_DIR.exists():
        return pd.DataFrame()
    fs = sorted(TDCC_DIR.glob("*.parquet"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["code"] = df["code"].astype(str).str.strip()
    for c in ("level", "pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def holders(df: pd.DataFrame) -> pd.DataFrame:
    """
    長表 → 每 (date, code) 的大戶比例。
        big   400 張以上（級距 12~15）
        huge  1,000 張以上（級距 15）
    """
    if df.empty:
        return df
    d = df[df["level"].between(1, 15)]
    big = (d[d["level"].isin(BIG_LEVELS)].groupby(["date", "code"])["pct"].sum()
           .rename("big"))
    huge = (d[d["level"] == HUGE_LEVEL].groupby(["date", "code"])["pct"].sum()
            .rename("huge"))
    out = pd.concat([big, huge], axis=1).reset_index()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d")
    return out.sort_values(["code", "date"])


def changes(h: pd.DataFrame, weeks: int = 1) -> pd.DataFrame:
    """加上 N 週變動（百分點）。這才是「大戶在吸籌還是在出貨」。"""
    if h.empty:
        return h
    h = h.sort_values(["code", "date"]).copy()
    for col in ("big", "huge"):
        h[f"{col}_chg{weeks}w"] = h.groupby("code")[col].diff(weeks)
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description="集保股權分散表（大戶持股）")
    ap.add_argument("--latest", action="store_true", help="抓最新一週（全市場，1 次請求）")
    ap.add_argument("--history", action="store_true", help="抓 51 週歷史（個股期標的）")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--min-volume", type=float, default=500)
    args = ap.parse_args()

    if args.latest:
        df = fetch_latest()
        d = str(df["date"].iloc[0])
        TDCC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_day_path(d), index=False)
        print(f"最新一週 {d}：{df['code'].nunique()} 檔 → {_day_path(d).name}")

    if args.history:
        import futures as fut
        codes = sorted(set(fut.load(min_volume=args.min_volume)["code"]))
        n = build_history(codes, pause=args.pause)
        print(f"完成，共 {n} 次請求")

    if args.status or not (args.latest or args.history):
        fs = sorted(TDCC_DIR.glob("*.parquet")) if TDCC_DIR.exists() else []
        print(f"集保快取：{len(fs)} 個週別")
        if fs:
            df = load()
            print(f"　範圍 {fs[0].stem} ~ {fs[-1].stem}｜{df['code'].nunique()} 檔"
                  f"｜{sum(f.stat().st_size for f in fs)/1e6:.1f} MB")
            h = holders(df)
            print(f"　大戶(400張)比例：中位 {h['big'].median():.1f}%")


if __name__ == "__main__":
    main()
