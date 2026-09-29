"""
跨交易所 BTC 價差收斂回測。資料來自 fetch.py 的 100ms 格子。

兩種策略：
  pair   兩所對沖：A 所相對 B 所「偏貴」超過門檻 → A 賣、B 買（兩邊都預先放好 BTC 與 USD(T)，
         不需要轉帳），價差收斂回 0 就反向平倉。4 次成交。市場風險中性。
  solo   單所追基準：某所低於全市場基準超過門檻 → 在該所買進，回到基準就賣。2 次成交，
         不對沖，賺的是「落後者追上」（lead-lag），但要承擔持有期間的漲跌。

價格口徑（★ 這是結果可信與否的關鍵）：
  賣用 bid、買用 ask（主動成交方向的最後成交價當代理），所以買賣價差是實付的。
  只用 last 會把 bid-ask 來回彈跳誤算成收斂 → 假 edge。
  基差：各所中價 − 全市場中位數 的 EMA（半衰期 --basis-hl 分鐘，只用過去資料），
  用來吸收 USD vs USDT 匯差、各所長期溢價。訊號是「偏離自己的基差」，損益用原始價格算。
  延遲：第 t 格看到訊號，第 t+L 格的價格成交（平倉同理）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
EXS = ["binance", "okx", "bybit", "coinbase", "bitget", "gate", "kraken", "bitfinex", "cryptocom"]


def load_grid(days: list[dt.date], res_ms: int, exs: list[str], warm_min: int = 90):
    """回傳 dict：t0(ms), ask/bid/age 矩陣 (T × E)。第一天前面多載 warm_min 分鐘給基差暖身。"""
    k = res_ms // 100
    first = days[0]
    t0 = int(dt.datetime(first.year, first.month, first.day, tzinfo=dt.timezone.utc).timestamp() * 1000) \
        - warm_min * 60_000
    t1 = int(dt.datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=dt.timezone.utc)
             .timestamp() * 1000) + 86_400_000
    T = (t1 - t0) // res_ms
    E = len(exs)
    ask = np.full((T, E), np.nan); bid = np.full((T, E), np.nan)
    lastt = np.full((T, E), np.nan)
    load_days = [first - dt.timedelta(days=1)] + days
    for e, ex in enumerate(exs):
        parts = [pd.read_parquet(CACHE / ex / f"{d}.parquet") for d in load_days
                 if (CACHE / ex / f"{d}.parquet").exists()]
        if not parts:
            continue
        df = pd.concat(parts)
        g = (df["bin"].to_numpy() * 100 - t0) // res_ms
        m = (g >= 0) & (g < T)
        df = df[m].assign(g=g[m])
        a = df.dropna(subset=["ask"]).groupby("g")["ask"].last()
        b = df.dropna(subset=["bid"]).groupby("g")["bid"].last()
        ask[a.index.to_numpy(), e] = a.to_numpy()
        bid[b.index.to_numpy(), e] = b.to_numpy()
        # 兩邊都更新過才算「新鮮」：取 ask/bid 各自最後更新時間的較舊者
        ta = np.full(T, np.nan); ta[a.index.to_numpy()] = a.index.to_numpy()
        tb = np.full(T, np.nan); tb[b.index.to_numpy()] = b.index.to_numpy()
        ta = pd.Series(ta).ffill().to_numpy(); tb = pd.Series(tb).ffill().to_numpy()
        lastt[:, e] = np.minimum(ta, tb)
    ask = pd.DataFrame(ask).ffill().to_numpy()
    bid = pd.DataFrame(bid).ffill().to_numpy()
    idx = np.arange(T)[:, None]
    age_s = (idx - lastt) * res_ms / 1000.0
    return dict(t0=t0, res=res_ms, ask=ask, bid=bid, age=age_s, exs=exs, warm=warm_min * 60_000 // res_ms)


def prepare(G, basis_hl_min: float, fresh_s: float):
    la, lb = np.log(G["ask"]), np.log(G["bid"])
    # 代理 ask 可能因為上一筆主動買比較舊而低於 bid（交叉）；中價仍取平均
    mid = (la + lb) / 2
    fresh = G["age"] <= fresh_s
    mid_f = np.where(fresh, mid, np.nan)
    raw_med = np.nanmedian(mid_f, axis=1)
    steps = basis_hl_min * 60_000 / G["res"]
    basis = pd.DataFrame(mid_f - raw_med[:, None]).ewm(halflife=steps, ignore_na=True).mean()
    basis = basis.shift(1).to_numpy()  # 只用過去
    adj = mid_f - basis
    comp = np.nanmedian(adj, axis=1)  # 全市場基準（已扣各所基差）
    G.update(la=la, lb=lb, basis=basis, fresh=fresh, comp=comp, adj=adj)
    return G


def _first_at_or_after(cands: np.ndarray, t: int) -> int:
    k = np.searchsorted(cands, t)
    return cands[k] if k < len(cands) else -1


def run_pair(G, i, j, theta_bp, lat, max_hold, exit_bp=0.0):
    """i 貴 → 賣 i 買 j。

    每筆回傳 (訊號格, 持有格數, 毛利, 零延遲中價損益, 實際中價損益)，全部 bp：
      零延遲中價損益 − 實際中價損益 ＝ 延遲損失（等待成交時偏離已經回去了）
      實際中價損益 − 毛利           ＝ 買賣價差（賣在 bid、買在 ask 的代價，進出場各一次）
      毛利 − 4 次手續費             ＝ 淨利
    """
    la, lb, bs, fr = G["la"], G["lb"], G["basis"], G["fresh"]
    mid = (la + lb) / 2
    T = len(la)
    db = bs[:, i] - bs[:, j]
    md = mid[:, i] - mid[:, j]
    edge_in = (lb[:, i] - la[:, j] - db) * 1e4          # 現在就賣 i 買 j 可以鎖到的偏離
    edge_out = (la[:, i] - lb[:, j] - db) * 1e4         # 反向平倉的成本（>0 表示 i 仍偏貴）
    ok = fr[:, i] & fr[:, j] & np.isfinite(edge_in)
    ok[: G["warm"]] = False
    ent = np.flatnonzero(ok & (edge_in > theta_bp))
    ext = np.flatnonzero(np.isfinite(edge_out) & (edge_out <= exit_bp))
    out = []
    t = 0
    while True:
        s = _first_at_or_after(ent, t)
        if s < 0 or s + lat >= T:
            break
        te = s + lat
        x = _first_at_or_after(ext, te + 1)
        if x < 0 or x > te + max_hold:
            x = min(te + max_hold, T - 1 - lat)
        tx = x + lat
        if tx >= T or tx <= te:
            break
        g = (lb[te, i] - la[te, j]) - (la[tx, i] - lb[tx, j])
        out.append((s, tx - te, g * 1e4, (md[s] - md[x]) * 1e4, (md[te] - md[tx]) * 1e4))
        t = tx + 1
    return out


def run_solo(G, i, theta_bp, lat, max_hold):
    """i 的 ask 低於全市場基準 θ → 在 i 買，bid 回到基準就賣。不對沖。"""
    la, lb, bs, fr, comp = G["la"], G["lb"], G["basis"], G["fresh"], G["comp"]
    T = len(la)
    dev_buy = (la[:, i] - bs[:, i] - comp) * 1e4
    dev_sell = (lb[:, i] - bs[:, i] - comp) * 1e4
    ok = fr[:, i] & np.isfinite(dev_buy)
    ok[: G["warm"]] = False
    ent = np.flatnonzero(ok & (dev_buy < -theta_bp))
    ext = np.flatnonzero(np.isfinite(dev_sell) & (dev_sell >= 0))
    out = []
    t = 0
    while True:
        s = _first_at_or_after(ent, t)
        if s < 0 or s + lat >= T:
            break
        te = s + lat
        x = _first_at_or_after(ext, te + 1)
        if x < 0 or x > te + max_hold:
            x = min(te + max_hold, T - 1 - lat)
        tx = x + lat
        if tx >= T or tx <= te:
            break
        mid = (la[:, i] + lb[:, i]) / 2
        out.append((s, tx - te, (lb[tx, i] - la[te, i]) * 1e4, (mid[x] - mid[s]) * 1e4, (mid[tx] - mid[te]) * 1e4))
        t = tx + 1
    return out


def describe(G):
    """各所偏離基準的分布與收斂速度（不涉及交易，純描述）。"""
    rows = []
    for e, ex in enumerate(G["exs"]):
        d = (G["adj"][:, e] - G["comp"]) * 1e4
        d = d[G["warm"]:]
        m = np.isfinite(d)
        if m.sum() < 1000:
            continue
        dd = d[m]
        # 半衰期：偏離序列的一階自相關 → ln(0.5)/ln(ρ)
        x = pd.Series(d)
        rho = x.autocorr(1)
        hl = np.log(0.5) / np.log(rho) * G["res"] / 1000 if 0 < rho < 1 else np.nan
        sp = ((G["la"][:, e] - G["lb"][:, e]) * 1e4)[G["warm"]:]
        rows.append(dict(ex=ex, fresh_pct=100 * m.mean(), dev_abs_med=np.median(np.abs(dd)),
                         dev_p95=np.percentile(np.abs(dd), 95), dev_p99=np.percentile(np.abs(dd), 99),
                         gt5bp_pct=100 * (np.abs(dd) > 5).mean(), gt10bp_pct=100 * (np.abs(dd) > 10).mean(),
                         halflife_s=hl, spread_proxy_med=np.nanmedian(sp)))
    return pd.DataFrame(rows)


def summarize(trades: dict, res_ms: int, t0s: dict) -> pd.DataFrame:
    rows = []
    for (kind, a, b, th, lat_ms), tr in trades.items():
        if not tr:
            continue
        g = np.array([x[2] for x in tr]); h = np.array([x[1] for x in tr])
        rows.append(dict(kind=kind, a=a, b=b, theta=th, lat_ms=lat_ms, n=len(g),
                         gross_mean=g.mean(), gross_med=np.median(g), win=(g > 0).mean(),
                         p05=np.percentile(g, 5), hold_med_s=np.median(h) * res_ms / 1000,
                         fills=4 if kind == "pair" else 2,
                         days_active=len({x[3] for x in tr})))
    R = pd.DataFrame(rows)
    R["breakeven_fee_bp"] = R["gross_mean"] / R["fills"]   # 損益兩平的單邊費率
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-21")
    ap.add_argument("--end", default="2026-09-20")
    ap.add_argument("--res-ms", type=int, default=1000)
    ap.add_argument("--chunk-days", type=int, default=31, help="一次載入幾天；100ms 格子請用 1")
    ap.add_argument("--basis-hl", type=float, default=30.0, help="基差 EMA 半衰期（分鐘）")
    ap.add_argument("--fresh", type=float, default=2.0, help="幾秒內有雙向成交才算報價有效")
    ap.add_argument("--theta", type=float, nargs="*", default=[2, 3, 5, 8, 12, 20])
    ap.add_argument("--lat-ms", type=int, nargs="*", default=[0, 1000, 2000, 5000])
    ap.add_argument("--max-hold-s", type=int, default=600)
    ap.add_argument("--exs", nargs="*", default=EXS)
    ap.add_argument("--no-solo", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    exs = [x for x in a.exs if any((CACHE / x).glob("*.parquet"))]
    print(f"交易所 {exs}  {s}~{e}  格子 {a.res_ms}ms", flush=True)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)

    mh = a.max_hold_s * 1000 // a.res_ms
    trades: dict = {}
    per_trade = []
    descs = []
    for c in range(0, len(days), a.chunk_days):
        chunk = days[c:c + a.chunk_days]
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            G = prepare(load_grid(chunk, a.res_ms, exs), a.basis_hl, a.fresh)
            descs.append(describe(G).assign(chunk=str(chunk[0])))
        day_of = lambda k: str(dt.datetime.fromtimestamp((G["t0"] + k * a.res_ms) / 1000, dt.timezone.utc).date())
        for lat_ms in a.lat_ms:
            lat = lat_ms // a.res_ms
            for th in a.theta:
                for i, j in itertools.permutations(range(len(exs)), 2):
                    tr = [(s_, h, g, day_of(s_), m0, m1) for s_, h, g, m0, m1 in run_pair(G, i, j, th, lat, mh)]
                    trades.setdefault(("pair", exs[i], exs[j], th, lat_ms), []).extend(tr)
                if not a.no_solo:
                    for i in range(len(exs)):
                        tr = [(s_, h, g, day_of(s_), m0, m1) for s_, h, g, m0, m1 in run_solo(G, i, th, lat, mh)]
                        trades.setdefault(("solo", exs[i], "基準", th, lat_ms), []).extend(tr)
        print(f"  {chunk[0]}~{chunk[-1]} 完成", flush=True)
        del G

    desc = pd.concat(descs)
    if len(descs) > 1:  # 分段時取各段中位數當代表
        desc = desc.drop(columns="chunk").groupby("ex", sort=False).median().reset_index()
    print(desc.round(2).to_string(index=False), flush=True)
    desc.to_csv(out / f"describe{a.tag}.csv", index=False)

    R = summarize(trades, a.res_ms, {})
    R.to_csv(out / f"trades_summary{a.tag}.csv", index=False)
    # 逐筆明細（給分日統計用）
    rows = [dict(kind=k[0], a=k[1], b=k[2], theta=k[3], lat_ms=k[4], day=x[3], hold_s=x[1] * a.res_ms / 1000,
                 gross=x[2], mid0=x[4], mid1=x[5])
            for k, tr in trades.items() for x in tr]
    pd.DataFrame(rows).to_parquet(out / f"trades{a.tag}.parquet", index=False)
    print(R.sort_values("gross_mean", ascending=False).head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
