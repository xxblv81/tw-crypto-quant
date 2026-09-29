"""
把 backtest.py 的逐筆明細彙整成成本拆解表，用各所「無優惠」費率逐筆扣。

每筆的損益拆成（全部 bp，名目 = 單邊部位）：
  訊號偏離    零延遲、以中價計算的收斂幅度（策略本身抓到的東西）
  − 延遲損失  從看到訊號到成交之間，偏離已經回去的部分
  − 買賣價差  賣在 bid、買在 ask（進出場各一次，兩所都付）
  − 吃單滑價  下單量超過最佳一檔的部分（用 probe.py 的掛單簿實測）
  − 手續費    對沖 4 次成交、單所 2 次，逐所套 FEES_TAKER
  ＝ 淨利
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"

# 現貨、最低等級、不持平台幣、不做 maker（套利要立刻成交，只能吃單）。單位 bp（0.01%）
# 2026-09-21 查證，來源見 ARB_RESULTS.md。Coinbase 9/16 宣布過調降，但各地區版本不一，取官方 Intro 1。
FEES_TAKER = {"binance": 10, "okx": 10, "bybit": 10, "bitget": 10, "gate": 20,
              "coinbase": 120, "kraken": 80, "bitfinex": 0, "cryptocom": 50}
FEES_MAKER = {"binance": 10, "okx": 8, "bybit": 10, "bitget": 10, "gate": 20,
              "coinbase": 60, "kraken": 40, "bitfinex": 0, "cryptocom": 25}


def load_impact(size: float) -> dict:
    p = OUT / "books.csv"
    if not p.exists() or size <= 0:
        return {}
    b = pd.read_csv(p, index_col=0)
    col = f"imp_{size}"
    return b[col].to_dict() if col in b else {}


def enrich(tr: pd.DataFrame, size: float) -> pd.DataFrame:
    imp = load_impact(size)
    t = tr.copy()
    pair = t.kind == "pair"
    fa = t.a.map(FEES_TAKER); fb = t.b.map(FEES_TAKER).fillna(0)
    t["fee"] = np.where(pair, 2 * (fa + fb), 2 * fa)
    ia = t.a.map(imp).fillna(0); ib = t.b.map(imp).fillna(0)
    t["impact"] = np.where(pair, 2 * (ia + ib), 2 * ia)
    t["lat_loss"] = t.mid0 - t.mid1
    t["spread"] = t.mid1 - t.gross
    t["net"] = t.gross - t.fee - t.impact
    return t


def grid(t: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    for (th, lat), g in t[t.kind == kind].groupby(["theta", "lat_ms"]):
        rows.append(dict(theta=th, lat_ms=lat, n=len(g), per_day=len(g) / 31,
                         signal=g.mid0.mean(), lat_loss=g.lat_loss.mean(), spread=g.spread.mean(),
                         gross=g.gross.mean(), impact=g.impact.mean(), fee=g.fee.mean(), net=g.net.mean(),
                         net_win=(g.net > 0).mean(), gross_win=(g.gross > 0).mean(),
                         usd_month=(g.net * 1e-4 * 10_000).sum()))
    return pd.DataFrame(rows)


def by_pair(t: pd.DataFrame, lat: int, theta: float) -> pd.DataFrame:
    g = t[(t.kind == "pair") & (t.lat_ms == lat) & (t.theta == theta)]
    r = g.groupby(["a", "b"]).agg(n=("net", "size"), signal=("mid0", "mean"), lat_loss=("lat_loss", "mean"),
                                  spread=("spread", "mean"), gross=("gross", "mean"), fee=("fee", "mean"),
                                  impact=("impact", "mean"), net=("net", "mean"),
                                  hold_s=("hold_s", "median"), days=("day", "nunique"))
    return r.sort_values("net", ascending=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--size", type=float, default=0.1, help="每邊下單顆數（查掛單簿滑價用）")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    t = enrich(pd.read_parquet(OUT / f"trades{a.tag}.parquet"), a.size)
    t.to_parquet(OUT / f"trades_enriched{a.tag}.parquet", index=False)
    for kind in ("pair", "solo"):
        if (t.kind == kind).any():
            G = grid(t, kind)
            G.to_csv(OUT / f"grid_{kind}{a.tag}.csv", index=False)
            print(f"\n=== {kind}（每邊 {a.size} BTC、無優惠 taker 費率；usd_month＝每筆 1 萬美元名目的月淨利） ===")
            print(G.round(2).to_string(index=False))
    for lat in sorted(t.lat_ms.unique()):
        for th in sorted(t.theta.unique()):
            P = by_pair(t, lat, th)
            P.to_csv(OUT / f"pairs_lat{lat}_th{th:g}{a.tag}.csv", index=False)
    lat = sorted(t.lat_ms.unique())[min(1, t.lat_ms.nunique() - 1)]
    th = sorted(t.theta.unique())[min(1, t.theta.nunique() - 1)]
    print(f"\n=== 延遲 {lat}ms、門檻 {th}bp，逐對拆解（淨利最高 15 對） ===")
    print(by_pair(t, lat, th).head(15).round(2).to_string(index=False))
    pos = t[(t.kind == "pair")].groupby(["a", "b", "theta", "lat_ms"]).net.mean()
    print(f"\n所有（配對 × 門檻 × 延遲）組合共 {len(pos)} 個，平均淨利 > 0 的有 {(pos > 0).sum()} 個")


if __name__ == "__main__":
    main()
