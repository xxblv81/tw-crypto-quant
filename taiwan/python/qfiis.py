"""
外資及陸資持股快取 —— TWSE MI_QFIIS 每日全市場。

為什麼值得單獨抓一份（已經有 T86 了）：
    T86 是**流量**（今天買賣超幾股），MI_QFIIS 是**存量**（外資總共持有幾 %）。
    「有人默默把持股比從 22% 一路堆到 30%」這種吸籌行為，看流量會被雜訊蓋掉，
    看存量一目了然。低位階策略的回測裡 `accum`（法人買超佔量比）是唯一
    明顯正貢獻的分項（PF 1.615、中位 +2.39%），這份資料是它的存量版升級。

    ★ 涵蓋的是「外資及陸資」，不含投信與自營商。投信要繼續用 chips.py 的 T86。

順帶解決另一件事 —— `發行股數`：
    本專案至今用「成交金額」當規模代理（P0 的四分組就是這樣分的）。
    有了發行股數就能算真市值 = 收盤價 × 發行股數，
    **P0 的分組值得用真市值重做一次**（見 CLAUDE.md）。

★ 端點的坑：`selectType=ALL` 會回一個 `stat:"OK"` 但 `data` 是空陣列的結果 ——
  看起來成功、實際沒資料。必須用 `selectType=ALLBUT0999`。
  這個失敗是靜默的，所以下面的解析特地把「表頭欄位」驗一次。

架構同 chips.py / margin.py：逐交易日抓全市場，一天一個 parquet，可中斷可續跑。
只涵蓋上市（TWSE），與本專案其餘部分一致。資料實測回溯到 2018-12-28。

用法：
    python qfiis.py --start 2019-01-01     # 建快取（已有的日期自動跳過，約 95 分鐘）
    python qfiis.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from twse import CACHE, _get_json, _num, trading_days  # noqa: F401

MI_QFIIS = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
QFIIS_DIR = CACHE / "qfiis"

# 欄位位置。名稱沒有重複，但仍固定用位置 + 驗表頭，理由同 margin.py：
# 改版時寧可炸掉，也不要安靜地存進錯欄位。
I_CODE, I_SHARES, I_FOREIGN_SH, I_FOREIGN_PCT = 0, 3, 5, 7
EXPECT_HEAD = ["證券代號", "證券名稱", "國際證券編碼", "發行股數"]


def _day_path(d: dt.date) -> Path:
    return QFIIS_DIR / f"{d:%Y-%m-%d}.parquet"


def fetch_day(d: dt.date) -> pd.DataFrame | None:
    """
    單日全市場外資持股。回傳 None 代表當天沒有資料（假日或早於資料起點）。

    欄位：
        shares_out    發行股數
        foreign_sh    全體外資及陸資持有股數
        foreign_pct   全體外資及陸資持股比率（%，端點直接給 float）
    """
    js = _get_json(MI_QFIIS, {"date": d.strftime("%Y%m%d"),
                              "selectType": "ALLBUT0999", "response": "json"})
    if not isinstance(js, dict) or js.get("stat") != "OK" or not js.get("data"):
        return None
    head = [str(x).strip() for x in js.get("fields", [])[:4]]
    if head != EXPECT_HEAD:
        raise RuntimeError(f"MI_QFIIS 欄位與預期不符：{js.get('fields')}")

    rows = []
    for r in js["data"]:
        code = str(r[I_CODE]).strip()
        if len(code) != 4 or not code.isdigit():    # 只留 4 碼普通股，濾掉 ETF/DR/特別股
            continue
        rows.append({"code": code,
                     "shares_out": _num(r[I_SHARES]),
                     "foreign_sh": _num(r[I_FOREIGN_SH]),
                     "foreign_pct": _num(r[I_FOREIGN_PCT])})
    return pd.DataFrame(rows) if rows else None


def build_cache(start: dt.date, end: dt.date, pause: float = 3.0,
                verbose: bool = True) -> int:
    QFIIS_DIR.mkdir(parents=True, exist_ok=True)
    days = [d for d in trading_days(start, end) if not _day_path(d).exists()]
    if verbose:
        print(f"待抓 {len(days)} 個交易日（已快取的自動跳過）｜pause={pause}s"
              f"｜預估 {len(days) * pause / 60:.0f} 分鐘", flush=True)
    n_ok = 0
    for k, d in enumerate(days, 1):
        df = fetch_day(d)
        if df is not None:
            df.to_parquet(_day_path(d), index=False)
            n_ok += 1
        else:
            _day_path(d).write_bytes(b"")           # 空檔＝假日，佔位避免重抓
        time.sleep(pause)
        if verbose and (k % 20 == 0 or k == len(days)):
            print(f"  [{k}/{len(days)}] {d} 累計有效 {n_ok} 天", flush=True)
    return n_ok


def load_wide(field: str = "foreign_pct") -> pd.DataFrame:
    """
    日期 × 代號 的寬表。快取不存在時回空表。
    field: foreign_pct（持股比率 %）／ foreign_sh（持股數）／ shares_out（發行股數）
    """
    if not QFIIS_DIR.exists():
        return pd.DataFrame()
    files = sorted(p for p in QFIIS_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        return pd.DataFrame()
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        df["date"] = pd.Timestamp(p.stem)
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    return raw.pivot_table(index="date", columns="code", values=field, aggfunc="last")


def pct_point_change(pct: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    持股比率的 N 日變化，單位是**百分點**不是百分比。

    ★ 用差不用比率是刻意的：持股比從 2% 升到 3% 是 +50%（比率）但只有 +1 個百分點，
      在籌碼意義上遠不如從 28% 升到 30%（+2 個百分點）。比率會讓低基期的雜訊放大。
    """
    return pct - pct.shift(window)


def market_cap(shares_out: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    真市值 = 收盤價 × 發行股數。

    ★ close 請傳**未還原**價，或自行確認口徑：本專案的行情快取預設是除權息還原價，
      還原價 × 今日發行股數在歷史日期上會低估市值。要精確算歷史市值，
      應該用 twse.load_bars(adjust=False) 的收盤價。
      當「規模分組」的相對排序用時（例如 P0 重做），還原與否影響很小。
    """
    idx = close.index.union(shares_out.index)
    so = shares_out.reindex(index=idx).ffill().reindex(close.index)
    return close * so.reindex(columns=close.columns)


def last_date() -> pd.Timestamp | None:
    """快取最後一個有資料的日期。呼叫端應該拿它跟行情快取比對，落後就要吵。"""
    if not QFIIS_DIR.exists():
        return None
    files = sorted(p for p in QFIIS_DIR.glob("*.parquet") if p.stat().st_size > 0)
    return pd.Timestamp(files[-1].stem) if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="建立外資及陸資持股快取（TWSE MI_QFIIS）")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--pause", type=float, default=3.0, help="每次請求間隔秒數，別調低")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        files = sorted(QFIIS_DIR.glob("*.parquet")) if QFIIS_DIR.exists() else []
        good = [p for p in files if p.stat().st_size > 0]
        print(f"外資持股快取：{len(files)} 個日期檔，其中 {len(good)} 天有資料")
        if good:
            pct = load_wide("foreign_pct")
            print(f"範圍：{good[0].stem} ~ {good[-1].stem}"
                  f"｜{pct.shape[1]} 檔｜{sum(p.stat().st_size for p in good) / 1e6:.1f} MB")
        else:
            print("尚未建立")
        return

    n = build_cache(dt.date.fromisoformat(args.start),
                    dt.date.fromisoformat(args.end), pause=args.pause)
    print(f"完成，{n} 個交易日。")


if __name__ == "__main__":
    main()
