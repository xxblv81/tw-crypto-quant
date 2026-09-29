"""
Binance 公開 K 線下載 + 本地快取。

為什麼是 Binance 而不是 Toobit（使用者實際交易的交易所）：
    Toobit 的 4H K 線只到 2025-01-26，再往前 endTime 一律回空（實測）。
    19 個月的資料比 TradingView Basic 給的 32 個月還少，樣本會更薄。
    Binance 的 startTime 參數正常運作，4H 可回溯到 2017-08-17。
    且已驗證的 PF 1.713 基準就是 BINANCE:BTCUSDT，用同一個來源才能對答案。

    ★ 實單可以在 Toobit 執行（現價實測差 0.0 bp），但「量能過濾」必須用 Binance 的量 ——
      Toobit 的 24h 量只有 Binance 的 1/3，量能分布完全不同，訊號不可移植。

免 API key，只用公開行情端點。
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"
UA = {"User-Agent": "Mozilla/5.0"}
CACHE = Path(__file__).resolve().parent / "cache"

# Binance 單次上限 1000 根
MAX_LIMIT = 1000

INTERVAL_MS = {
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _ms(d: dt.date | dt.datetime) -> int:
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        d = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def fetch(symbol: str, interval: str, start: dt.date, end: dt.date,
          pause: float = 0.25, verbose: bool = True) -> pd.DataFrame:
    """
    往前翻頁抓滿整個區間。Binance 的 startTime 有效，所以可以順著時間拉。
    回傳 index=開盤時間(UTC) 的 DataFrame[Open,High,Low,Close,Volume,Turnover]。
    """
    step = INTERVAL_MS[interval]
    cur, end_ms = _ms(start), _ms(end)
    rows: list[list] = []
    while cur < end_ms:
        r = requests.get(BASE, params={"symbol": symbol, "interval": interval,
                                       "startTime": cur, "endTime": end_ms,
                                       "limit": MAX_LIMIT}, headers=UA, timeout=30)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        rows.extend(k)
        nxt = k[-1][0] + step
        if nxt <= cur:                       # 保險：避免無限迴圈
            break
        cur = nxt
        if verbose and len(rows) % 10000 < MAX_LIMIT:
            print(f"    {symbol} {interval}: {len(rows):,} 根…", flush=True)
        time.sleep(pause)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
        "Turnover", "trades", "tb_base", "tb_quote", "ignore"])
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume", "Turnover"]]
    for c in ("Open", "High", "Low", "Close", "Volume", "Turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"]).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="first")].dropna()


def load(symbol: str, interval: str, start: dt.date, end: dt.date,
         refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """有快取讀快取，沒有就下載。快取檔名含區間，換區間不會誤用。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if verbose:
        print(f"  下載 {symbol} {interval} {start}~{end} …", flush=True)
    df = fetch(symbol, interval, start, end, verbose=verbose)
    if not df.empty:
        df.to_parquet(path)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--intervals", default="15m,1h,2h,4h,6h,1d")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    for iv in a.intervals.split(","):
        df = load(a.symbol, iv.strip(), s, e, refresh=a.refresh)
        if df.empty:
            print(f"  {iv}: 無資料")
        else:
            print(f"  {iv}: {len(df):,} 根  {df.index[0]:%Y-%m-%d %H:%M} ~ {df.index[-1]:%Y-%m-%d %H:%M}")
