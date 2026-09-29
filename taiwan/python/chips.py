"""
籌碼快取 —— 三大法人買賣超日報（TWSE T86），目前只用「投信買賣超股數」。

為什麼需要它：
    愛德恩的四步驟第二步是「看投信是否連買」。那是他方法裡最不可省的一項 ——
    價量動能全市場都看得到，投信連續買超才是「有人真的在收貨」的證據。

架構同 twse.py：逐交易日抓全市場，一天一個 parquet，可中斷可續跑。
    端點：https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALL
    只涵蓋上市（TWSE）。上櫃有另一個端點，但本專案預設不含上櫃（理由見 CLAUDE.md）。

★ 這份快取要另外建，跑一次約 1.5 小時（1,861 個交易日 × pause 3 秒）：
      python chips.py --start 2019-01-01
  只想先試最近一年就 --start 2025-09-01，幾分鐘跑完。
  沒建快取時 use_chips 必須維持 false，動能分數只用價量四項。

用法：
    python chips.py --start 2019-01-01     # 建快取（已有的日期自動跳過）
    python chips.py --status               # 看快取狀態
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from twse import CACHE, UA, _get_json, _num, trading_days, _pick, _session  # noqa: F401

T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
CHIP_DIR = CACHE / "chips"


def _day_path(d: dt.date) -> Path:
    return CHIP_DIR / f"{d:%Y-%m-%d}.parquet"


def fetch_day(d: dt.date) -> pd.DataFrame | None:
    """單日全市場投信買賣超（股數）。"""
    js = _get_json(T86, {"date": d.strftime("%Y%m%d"),
                         "selectType": "ALL", "response": "json"})
    if not isinstance(js, dict) or js.get("stat") != "OK" or not js.get("data"):
        return None
    f = js["fields"]
    i_code = _pick(f, "證券代號")
    i_it = _pick(f, "投信買賣超股數")
    i_fo = _pick(f, "外陸資買賣超股數")          # 一併存著，之後想加外資條件不必重抓
    rows = []
    for r in js["data"]:
        code = str(r[i_code]).strip()
        if len(code) != 4 or not code.isdigit():   # 只留 4 碼普通股，濾掉 ETF/權證
            continue
        rows.append({"code": code,
                     "trust_net": _num(r[i_it]),
                     "foreign_net": _num(r[i_fo])})
    return pd.DataFrame(rows) if rows else None


def build_cache(start: dt.date, end: dt.date, pause: float = 3.0,
                verbose: bool = True) -> int:
    CHIP_DIR.mkdir(parents=True, exist_ok=True)
    days = [d for d in trading_days(start, end) if not _day_path(d).exists()]
    if verbose:
        print(f"待抓 {len(days)} 個交易日（已快取的自動跳過）｜pause={pause}s"
              f"｜預估 {len(days) * pause / 60:.0f} 分鐘")
    n_ok = 0
    for k, d in enumerate(days, 1):
        df = fetch_day(d)
        if df is not None:
            df.to_parquet(_day_path(d), index=False)
            n_ok += 1
        else:
            # ★ 不要對「最近幾天」寫空檔佔位。空檔＝假日，會被永遠跳過；
            #   但當天資料常常只是還沒公布（T86 約 16:00 後、行情約 14:30 後），
            #   寫了佔位就再也補不回來。實際踩過：15:48 跑 → 09-07 被當成假日。
            if (dt.date.today() - d).days >= 3:
                _day_path(d).write_bytes(b"")      # 舊日期仍無資料 → 真的是假日
            elif verbose:
                print(f"      {d} 尚無資料（可能還沒公布），不佔位，下次再試", flush=True)
        time.sleep(pause)
        if verbose and (k % 20 == 0 or k == len(days)):
            print(f"  [{k}/{len(days)}] {d} 累計有效 {n_ok} 天", flush=True)
    return n_ok


def load_trust_net() -> pd.DataFrame:
    """日期 × 代號 的投信買賣超股數寬表。快取不存在時回空表。"""
    if not CHIP_DIR.exists():
        return pd.DataFrame()
    files = sorted(p for p in CHIP_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        return pd.DataFrame()
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        df["date"] = pd.Timestamp(p.stem)
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    return raw.pivot_table(index="date", columns="code", values="trust_net", aggfunc="sum")


def last_date() -> pd.Timestamp | None:
    """籌碼快取覆蓋到哪一天。給呼叫端判斷資料是否落後於行情用。"""
    if not CHIP_DIR.exists():
        return None
    good = [p for p in CHIP_DIR.glob("*.parquet") if p.stat().st_size > 0]
    return pd.Timestamp(max(p.stem for p in good)) if good else None


def buy_streak(trust_net: pd.DataFrame) -> pd.DataFrame:
    """
    連續買超天數。當日買超（>0）則累加，否則歸零。
    沒有資料的日子（該股當天不在 T86 名單中＝法人沒動作）視為 0，中斷連買。
    """
    if trust_net.empty:
        return trust_net
    buy = (trust_net.fillna(0) > 0)
    # 經典的「連續 True 計數」：累加後減去每段開頭時的累加值
    cum = buy.cumsum()
    reset = cum.where(~buy).ffill().fillna(0)
    return (cum - reset).where(buy, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="建立三大法人（投信）買賣超快取")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--pause", type=float, default=3.0, help="每次請求間隔秒數，別調低")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        files = sorted(CHIP_DIR.glob("*.parquet")) if CHIP_DIR.exists() else []
        good = [p for p in files if p.stat().st_size > 0]
        print(f"籌碼快取：{len(files)} 個日期檔，其中 {len(good)} 天有資料")
        if good:
            print(f"範圍：{good[0].stem} ~ {good[-1].stem}"
                  f"｜{sum(p.stat().st_size for p in good) / 1e6:.1f} MB")
        else:
            print("尚未建立 —— markets.yaml 的 use_chips 請維持 false")
        return

    n = build_cache(dt.date.fromisoformat(args.start),
                    dt.date.fromisoformat(args.end), pause=args.pause)
    print(f"完成，{n} 個交易日。接著把 markets.yaml 的 use_chips 改成 true。")


if __name__ == "__main__":
    main()
