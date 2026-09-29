"""
跨交易所 BTC 逐筆成交下載 → 壓成 100ms 格子，一天一檔，可中斷可續跑。

cache/{ex}/{YYYY-MM-DD}.parquet  欄位：bin(=UTC ms // 100), last, ask, bid, vol, n

★ ask／bid 是「主動買／主動賣的最後成交價」，當作賣一／買一的代理。
  只用 last 算價差會把 bid-ask 來回彈跳誤算成收斂利潤（假 edge）。
  side 一律轉成 taker 方向：+1 主動買（成交在 ask）、−1 主動賣（成交在 bid）。

資料來源（2026-09-21 逐一實測）：
    binance   data.binance.vision spot aggTrades 日檔（時間戳是微秒）
    okx       static.okx.com 日檔 ★ 日界是 UTC+8，要多抓一天再用時間戳切
    bybit     public.bybit.com/spot 日檔
    gate      download.gatedata.org 月檔（只有已結束的月份）＋ 30 天內用 REST
    coinbase  REST /trades，只能用 trade_id 往回翻（after=），1000 筆/頁
    kraken    REST /Trades since=ns，1000 筆/頁，限速最嚴
    bitfinex  REST /trades/hist，10000 筆/頁
    bitget    REST fills-history，1000 筆/頁，idLessThan 往回翻，只保留 90 天
    cryptocom REST get-trades，150 筆/頁，用 end_ts(ns) 往回翻

沒有公開逐筆歷史而排除的：KuCoin、HTX、MEXC（只給最近幾百筆）、Bitstamp（最多一天）、
Upbit（KRW 報價且 USDT 盤只給 7 天）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "cache"
UA = {"User-Agent": "Mozilla/5.0"}
BIN_MS = 100

# 各所主力盤。USD 報價的三家（coinbase/kraken/bitfinex）與 USDT 之間有匯差，回測端用滾動均值吸收
SYMBOL = {
    "binance": "BTCUSDT", "okx": "BTC-USDT", "bybit": "BTCUSDT", "gate": "BTC_USDT",
    "coinbase": "BTC-USD", "kraken": "XBTUSD", "bitfinex": "tBTCUSD",
    "bitget": "BTCUSDT", "cryptocom": "BTC_USD",
}

_sess = threading.local()


def S() -> requests.Session:
    if not hasattr(_sess, "s"):
        _sess.s = requests.Session()
        _sess.s.headers.update(UA)
    return _sess.s


def get(url, params=None, tries=8, timeout=30):
    for k in range(tries):
        try:
            r = S().get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except Exception as e:  # noqa: BLE001  連線中斷、限速都退避重試
            if k == tries - 1:
                raise
            if k >= 2:
                print(f"  重試 {url.split('/')[2]} 第 {k+1} 次: {e!r}", flush=True)
            time.sleep(min(60, 2 ** k))


def day_ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def to_bins(ts_ms, price, qty, side) -> pd.DataFrame:
    o = np.argsort(ts_ms, kind="stable")
    df = pd.DataFrame({"bin": (ts_ms[o] // BIN_MS).astype(np.int64),
                       "p": price[o].astype(np.float64), "q": qty[o].astype(np.float64),
                       "s": side[o].astype(np.int8)})
    g = df.groupby("bin", sort=True)
    out = pd.DataFrame({"last": g["p"].last(), "vol": g["q"].sum(),
                        "n": g["p"].size().astype(np.int32)})
    out["ask"] = df[df.s > 0].groupby("bin")["p"].last()
    out["bid"] = df[df.s < 0].groupby("bin")["p"].last()
    return out.reset_index()


def save_day(ex: str, d: dt.date, ts_ms, price, qty, side):
    ts_ms = np.asarray(ts_ms, dtype=np.float64)
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    m = (ts_ms >= lo) & (ts_ms < hi)
    side = np.asarray(side, dtype=np.int8)
    out = to_bins(ts_ms[m].astype(np.int64), np.asarray(price, float)[m], np.asarray(qty, float)[m], side[m])
    p = CACHE / ex / f"{d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    print(f"[{ex}] {d} trades={int(m.sum()):,} bins={len(out):,}", flush=True)


def done(ex, d):
    return (CACHE / ex / f"{d}.parquet").exists()


# ───────────────────────── 日檔型 ─────────────────────────

def binance(d):
    u = f"https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip"
    z = zipfile.ZipFile(io.BytesIO(get(u, timeout=300).content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[1, 2, 5, 6])
    ts = df[5].to_numpy(np.float64)
    ts = ts / 1000 if ts[0] > 1e14 else ts  # 2025 起改微秒
    side = np.where(df[6].astype(str).str.lower() == "true", -1, 1)
    save_day("binance", d, ts, df[1].to_numpy(), df[2].to_numpy(), side)


def _okx_file(d):
    u = (f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{d:%Y%m%d}/"
         f"BTC-USDT-trades-{d}.zip")
    r = get(u, timeout=300)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return pd.read_csv(z.open(z.namelist()[0]), usecols=["side", "price", "size", "created_time"])


def okx(d):
    # OKX 日檔是 UTC+8 的一天：UTC 的 d 橫跨檔 d 與檔 d+1
    parts = [_okx_file(d), _okx_file(d + dt.timedelta(days=1))]
    if any(x is None for x in parts):
        # 隔天的檔還沒上架時硬存，會得到只有 16 小時的一天且不報錯（2026-09-20 踩過）
        raise RuntimeError(f"OKX 日檔 {d} 或 {d + dt.timedelta(days=1)} 尚未上架")
    df = pd.concat(parts)
    save_day("okx", d, df["created_time"].to_numpy(), df["price"].to_numpy(), df["size"].to_numpy(),
             np.where(df["side"] == "buy", 1, -1))


def bybit(d):
    u = f"https://public.bybit.com/spot/BTCUSDT/BTCUSDT_{d}.csv.gz"
    df = pd.read_csv(io.BytesIO(gzip.decompress(get(u, timeout=300).content)),
                     usecols=["timestamp", "price", "volume", "side"])
    save_day("bybit", d, df["timestamp"].to_numpy(), df["price"].to_numpy(), df["volume"].to_numpy(),
             np.where(df["side"].str.lower() == "buy", 1, -1))


_gate_month: dict = {}


def gate(d):
    today = dt.datetime.now(dt.timezone.utc).date()
    if (today - d).days >= 29:  # REST 只給 30 天內，其餘用月檔
        key = f"{d:%Y%m}"
        if key not in _gate_month:
            u = f"https://download.gatedata.org/spot/deals/{key}/BTC_USDT-{key}.csv.gz"
            df = pd.read_csv(io.BytesIO(gzip.decompress(get(u, timeout=600).content)),
                             header=None, usecols=[0, 2, 3, 4])
            _gate_month[key] = df
        df = _gate_month[key]
        # 月檔第 5 欄：2＝主動買、1＝主動賣（2026-09-21 對照滾動中位數實測）
        save_day("gate", d, df[0].to_numpy() * 1000, df[2].to_numpy(), df[3].to_numpy(),
                 np.where(df[4] == 2, 1, -1))
        return
    # REST：10 分鐘一窗，窗內翻頁
    T, P, Q, SD = [], [], [], []
    lo = day_ms(d) // 1000
    for w in range(lo, lo + 86400, 600):
        page = 1
        while True:
            r = get("https://api.gateio.ws/api/v4/spot/trades",
                    {"currency_pair": "BTC_USDT", "limit": 1000, "from": w, "to": w + 599, "page": page})
            js = r.json()
            if not isinstance(js, list):
                raise RuntimeError(js)
            for x in js:
                T.append(float(x["create_time_ms"])); P.append(float(x["price"])); Q.append(float(x["amount"]))
                SD.append(1 if x["side"] == "buy" else -1)
            if len(js) < 1000:
                break
            page += 1
            time.sleep(0.06)
        time.sleep(0.06)
    save_day("gate", d, T, P, Q, SD)


# ───────────────────────── REST 翻頁型 ─────────────────────────

def kraken(d):
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    since = str(lo * 1_000_000)
    T, P, Q, SD = [], [], [], []
    while True:
        js = get("https://api.kraken.com/0/public/Trades", {"pair": "XBTUSD", "since": since, "count": 1000}).json()
        if js.get("error"):
            if any("Too many" in e for e in js["error"]):
                time.sleep(5); continue
            raise RuntimeError(js["error"])
        rows = js["result"]["XXBTZUSD"]
        for x in rows:
            T.append(x[2] * 1000); P.append(float(x[0])); Q.append(float(x[1])); SD.append(1 if x[3] == "b" else -1)
        since = js["result"]["last"]
        if len(T) % 50000 < 1000:
            print(f"  [kraken] {d} 已 {len(T):,} 筆", flush=True)
        if not rows or rows[-1][2] * 1000 >= hi:
            break
        time.sleep(1.0)
    save_day("kraken", d, T, P, Q, SD)


def bitfinex(d):
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    T, P, Q, SD = [], [], [], []
    start = lo
    while True:
        js = get(f"https://api-pub.bitfinex.com/v2/trades/tBTCUSD/hist",
                 {"start": start, "end": hi - 1, "limit": 10000, "sort": 1}).json()
        if isinstance(js, list) and js and js[0] == "error":
            time.sleep(30); continue
        for x in js:
            T.append(x[1]); P.append(x[3]); Q.append(abs(x[2])); SD.append(1 if x[2] > 0 else -1)
        if len(js) < 10000:
            break
        start = js[-1][1]  # 同毫秒可能重複，存檔前去重
        time.sleep(5.0)  # 端點上限 15 次/分，超過會被鎖 60 秒
    df = pd.DataFrame({"t": T, "p": P, "q": Q, "s": SD}).drop_duplicates()
    save_day("bitfinex", d, df.t.to_numpy(), df.p.to_numpy(), df.q.to_numpy(), df.s.to_numpy())


def bitget(d):
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    T, P, Q, SD = [], [], [], []
    params = {"symbol": "BTCUSDT", "limit": 1000, "startTime": lo, "endTime": hi - 1}
    while True:
        js = get("https://api.bitget.com/api/v2/spot/market/fills-history", params).json()
        if js.get("code") != "00000":
            if "429" in str(js.get("code")) or "limit" in str(js.get("msg", "")).lower():
                time.sleep(2); continue
            raise RuntimeError(js)
        rows = js["data"]
        for x in rows:
            T.append(int(x["ts"])); P.append(float(x["price"])); Q.append(float(x["size"]))
            SD.append(1 if x["side"].lower() == "buy" else -1)
        if len(rows) < 1000:
            break
        params["idLessThan"] = rows[-1]["tradeId"]
        time.sleep(0.12)
    save_day("bitget", d, T, P, Q, SD)


def cryptocom(d):
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    T, P, Q, SD = [], [], [], []
    end_ns = hi * 1_000_000
    seen = set()
    while True:
        js = get("https://api.crypto.com/exchange/v1/public/get-trades",
                 {"instrument_name": "BTC_USD", "count": 150,
                  "start_ts": lo * 1_000_000, "end_ts": end_ns}).json()
        if js.get("code") != 0:
            time.sleep(2); continue
        rows = [x for x in js["result"]["data"] if x["d"] not in seen]
        if not rows:
            break
        for x in rows:
            seen.add(x["d"]); T.append(x["t"]); P.append(float(x["p"])); Q.append(float(x["q"]))
            SD.append(1 if x["s"].lower() == "buy" else -1)
        end_ns = min(int(x["tn"]) for x in rows)  # 含端點，靠 seen 去重
        if len(T) % 30000 < 150:
            print(f"  [cryptocom] {d} 已 {len(T):,} 筆，翻到 {pd.Timestamp(end_ns, tz='UTC'):%H:%M}", flush=True)
        time.sleep(0.05)
    save_day("cryptocom", d, T, P, Q, SD)


# Coinbase 只能照 trade_id 翻頁 → 先用二分搜找每天第一筆的 id
def _cb_page(after):
    r = get("https://api.exchange.coinbase.com/products/BTC-USD/trades", {"limit": 1000, "after": after})
    return r.json()


def _cb_ts(x):
    return pd.Timestamp(x["time"]).value / 1e6


def _cb_id_at(ms: int) -> int:
    hi = get("https://api.exchange.coinbase.com/products/BTC-USD/trades", {"limit": 1}).json()[0]["trade_id"]
    lo = hi - 60_000_000
    while hi - lo > 1000:
        mid = (lo + hi) // 2
        pg = _cb_page(mid + 1)  # 回傳 id <= mid
        if _cb_ts(pg[0]) < ms:
            lo = mid
        else:
            hi = mid
        time.sleep(0.15)
    return hi


def coinbase(d):
    lo, hi = day_ms(d), day_ms(d + dt.timedelta(days=1))
    top = _cb_id_at(hi) + 1000
    T, P, Q, SD = [], [], [], []
    after = top + 1
    while True:
        pg = _cb_page(after)
        if not pg:
            break
        for x in pg:
            T.append(_cb_ts(x)); P.append(float(x["price"])); Q.append(float(x["size"]))
            SD.append(1 if x["side"] == "sell" else -1)  # ★ Coinbase 的 side 是 maker 方向，要反過來
        after = pg[-1]["trade_id"]
        if len(T) % 100000 < 1000:
            print(f"  [coinbase] {d} 已 {len(T):,} 筆，翻到 {pg[-1]['time'][11:16]}", flush=True)
        if _cb_ts(pg[-1]) < lo:
            break
        time.sleep(0.12)
    save_day("coinbase", d, T, P, Q, SD)


FETCH = {"binance": binance, "okx": okx, "bybit": bybit, "gate": gate, "kraken": kraken,
         "bitfinex": bitfinex, "bitget": bitget, "cryptocom": cryptocom, "coinbase": coinbase}
# 同一交易所可平行幾天（REST 的限速是每 IP 共用）
WORKERS = {"binance": 4, "okx": 4, "bybit": 4, "gate": 2, "kraken": 1, "bitfinex": 1,
           "bitget": 3, "cryptocom": 3, "coinbase": 2}


def run(ex, days):
    todo = [d for d in days if not done(ex, d)]

    def one(d):
        if done(ex, d):  # 可能被另一支平行的程式先抓完
            return
        for k in range(3):
            try:
                return FETCH[ex](d)
            except Exception as e:  # noqa: BLE001
                print(f"[{ex}] {d} 失敗第 {k+1} 次: {e!r}", flush=True)
                time.sleep(30)

    if ex == "gate":  # 月檔要依序讀才能共用
        for d in todo:
            one(d)
        return
    with ThreadPoolExecutor(WORKERS[ex]) as pool:
        list(pool.map(one, todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-21")
    ap.add_argument("--end", default="2026-09-20")
    ap.add_argument("--ex", nargs="*", default=list(FETCH))
    ap.add_argument("--reverse", action="store_true", help="從最後一天往回抓（給第二支平行程式用）")
    a = ap.parse_args()
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    if a.reverse:
        days = days[::-1]
    ths = [threading.Thread(target=run, args=(ex, days)) for ex in a.ex]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


if __name__ == "__main__":
    main()
