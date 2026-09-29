"""
實測兩件回測資料給不了的事：
  1. 掛單簿深度 → 下 Q 顆 BTC 市價單，相對最佳價要多付幾 bp（吃單滑價）
  2. 從這台電腦到各所 API 的網路延遲（TCP 連線、TLS、首位元組）

★ 掛單簿只能拿「現在」的快照，歷史掛單簿要付費。所以滑價數字代表的是抓取當下的市況，
  急殺時深度會變薄，實際滑價只會更大。
"""
from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "results"
UA = {"User-Agent": "Mozilla/5.0"}

BOOK = {
    "binance": ("https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000",
                lambda j: (j["bids"], j["asks"])),
    "okx": ("https://www.okx.com/api/v5/market/books?instId=BTC-USDT&sz=400",
            lambda j: (j["data"][0]["bids"], j["data"][0]["asks"])),
    "bybit": ("https://api.bybit.com/v5/market/orderbook?category=spot&symbol=BTCUSDT&limit=200",
              lambda j: (j["result"]["b"], j["result"]["a"])),
    "coinbase": ("https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
                 lambda j: (j["bids"], j["asks"])),
    "kraken": ("https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=500",
               lambda j: (j["result"]["XXBTZUSD"]["bids"], j["result"]["XXBTZUSD"]["asks"])),
    "bitfinex": ("https://api-pub.bitfinex.com/v2/book/tBTCUSD/P0?len=250",
                 lambda j: ([[x[0], x[2]] for x in j if x[2] > 0], [[x[0], -x[2]] for x in j if x[2] < 0])),
    "bitget": ("https://api.bitget.com/api/v2/spot/market/orderbook?symbol=BTCUSDT&type=step0&limit=150",
               lambda j: (j["data"]["bids"], j["data"]["asks"])),
    "gate": ("https://api.gateio.ws/api/v4/spot/order_book?currency_pair=BTC_USDT&limit=100",
             lambda j: (j["bids"], j["asks"])),
    "cryptocom": ("https://api.crypto.com/exchange/v1/public/get-book?instrument_name=BTC_USD&depth=150",
                  lambda j: (j["result"]["data"][0]["bids"], j["result"]["data"][0]["asks"])),
}
SIZES = [0.01, 0.1, 0.5, 1.0, 5.0]


def impact(levels, q):
    """吃掉 q 顆的成交均價，相對最佳價的 bp。深度不夠回 NaN。"""
    px = np.array([float(x[0]) for x in levels]); sz = np.array([float(x[1]) for x in levels])
    cum = np.cumsum(sz)
    if cum[-1] < q:
        return np.nan
    k = np.searchsorted(cum, q)
    filled = np.concatenate([sz[:k], [q - (cum[k - 1] if k else 0)]])
    avg = (px[: k + 1] * filled).sum() / q
    return abs(avg / px[0] - 1) * 1e4


def books(rounds: int, gap: float):
    rows = []
    for r in range(rounds):
        for ex, (u, parse) in BOOK.items():
            try:
                bids, asks = parse(requests.get(u, headers=UA, timeout=15).json())
                bids = sorted(bids, key=lambda x: -float(x[0])); asks = sorted(asks, key=lambda x: float(x[0]))
                bb, ba = float(bids[0][0]), float(asks[0][0])
                row = dict(ex=ex, round=r, spread_bp=(ba / bb - 1) * 1e4,
                           top_btc=(float(bids[0][1]) + float(asks[0][1])) / 2)
                for q in SIZES:
                    row[f"imp_{q}"] = (impact(asks, q) + impact(bids, q)) / 2
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                print(ex, "失敗", e)
        print(f"掛單簿第 {r+1}/{rounds} 輪", flush=True)
        time.sleep(gap)
    return pd.DataFrame(rows)


PING = {ex: u for ex, (u, _) in BOOK.items()}


def latency(n: int):
    rows = []
    for ex, u in PING.items():
        c, t, f = [], [], []
        fails = 0
        for _ in range(n):
            try:
                o = subprocess.run(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w",
                                    "%{time_connect} %{time_appconnect} %{time_starttransfer}", u],
                                   capture_output=True, text=True, timeout=15).stdout.split()
            except subprocess.TimeoutExpired:
                o = []
            if len(o) != 3 or float(o[0]) == 0:
                fails += 1
                continue
            a, b, s = map(float, o)
            c.append(a * 1000); t.append(b * 1000); f.append((s - b) * 1000)
        rows.append(dict(ex=ex, tcp_rtt_ms=statistics.median(c), tls_ms=statistics.median(t) - statistics.median(c),
                         server_ms=statistics.median(f), tcp_p90=np.percentile(c, 90), tcp_min=min(c),
                         total_p90=np.percentile(np.array(t) + np.array(f), 90), fails=fails, tries=n))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    L = latency(15)
    L.to_csv(OUT / "latency.csv", index=False)
    print(L.round(1).to_string(index=False))
    B = books(10, 30)
    B.to_csv(OUT / "books_raw.csv", index=False)
    s = B.groupby("ex").median(numeric_only=True).drop(columns="round")
    s.to_csv(OUT / "books.csv")
    print(s.round(2).to_string())
