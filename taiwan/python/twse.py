"""
證交所 / 櫃買中心官方 API 行情後端。

取代 yfinance：資料直接來自 TWSE / TPEx，免 token、免限流風險（但要自律 pause）。

架構重點 —— 為什麼是「逐日抓全市場」而不是「逐檔抓歷史」：
    TWSE 的 STOCK_DAY 是「一檔一個月」，1000 檔 × 5 年 = 60,000 次請求，跑不完。
    MI_INDEX 是「一天全市場」，5 年只要約 1,250 次請求。
    所以快取以「交易日」為單位，一天一個 parquet，可中斷可續跑。

除權息還原（重要）：
    TWSE 的行情是「未還原」原始價。長期回測若不還原，除息當天會出現假跳空，
    在 stop_atr=4.0 的日線策略上會觸發假停損 → 系統性低估績效。
    本模組用 TWT49U（除權除息計算結果表）算還原因子：
        factor = 除權息參考價 / 除權息前收盤價
    再由今往回累乘，回頭調整除權息日之前的所有 OHLC（等同 yfinance 的 auto_adjust）。

    ★ 限制：TWT49U 只涵蓋上市（TWSE）。櫃買（TPEx）的除權息 openapi 只有兩個月
      滾動窗口、沒有歷史，因此上櫃股無法還原。預設 include_otc=False，
      理由見 README / CLAUDE.md：混入未還原的上櫃股會污染「中小型 vs 權值股」的比較。

用法：
    python twse.py --start 2019-01-01              # 建快取（可重跑，已有的日期會跳過）
    python twse.py --start 2019-01-01 --include-otc
"""
from __future__ import annotations

import argparse
import datetime as dt
import ssl
import sys
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 30

TWSE_QUOTES = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_QUOTES = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"
TWSE_EXRIGHT = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DAY_DIR = CACHE / "day"
EXRIGHT_CSV = CACHE / "exright.csv"

COLS = ["Open", "High", "Low", "Close", "Volume"]
DAY_COLS = ["code", "name", "market", "Open", "High", "Low", "Close", "Volume", "Turnover"]


# ─────────────────────────── 小工具 ───────────────────────────
def _num(v) -> float:
    """把 '1,234.50' / '--' / 'X0.00' 這種欄位轉成 float，轉不動回 nan。"""
    if v is None:
        return float("nan")
    s = str(v).replace(",", "").replace("+", "").strip()
    if s in ("", "--", "---", "X", "N/A", "null"):
        return float("nan")
    if s.startswith("X"):          # X 開頭是註記價，不是有效成交價
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _roc_date(v) -> dt.date | None:
    """民國日期轉西元。TWSE 各端點格式不一：『115年07月01日』與『115/07/01』都要吃。"""
    s = str(v).strip()
    for a, b, c in (("年", "月", "日"), ("/", "/", "")):
        if a in s:
            try:
                y, rest = s.split(a, 1)
                m, rest2 = rest.split(b, 1)
                d = rest2.replace(c, "").strip() if c else rest2
                return dt.date(int(y) + 1911, int(m), int(d.split("/")[0]))
            except Exception:
                return None
    return None


def _pick(fields: list[str], *names: str) -> int:
    """依欄位名找 index（TPEx 欄名帶空白／全形，所以要 strip 後比對）。"""
    norm = [str(f).replace(" ", "").replace("　", "").strip() for f in fields]
    for n in names:
        key = n.replace(" ", "")
        for i, f in enumerate(norm):
            if f.startswith(key):
                return i
    raise KeyError(f"找不到欄位 {names}，實際欄位：{fields}")


class _RelaxedStrictAdapter(requests.adapters.HTTPAdapter):
    """
    TPEx 的伺服器憑證缺 Subject Key Identifier 擴充欄位（不符 RFC 5280）。
    Python 3.13 起 ssl.create_default_context() 預設開啟 VERIFY_X509_STRICT，
    會直接拒連（SSLError: CERTIFICATE_VERIFY_FAILED）。curl 比較寬鬆所以測得過。

    這裡只關掉「嚴格擴充欄位檢查」這一個旗標，**憑證鏈本身仍然完整驗證**，
    不是 verify=False。只影響 TPEx，TWSE 走一般連線。
    """
    def init_poolmanager(self, *a, **kw):
        ctx = ssl.create_default_context()
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kw["ssl_context"] = ctx
        return super().init_poolmanager(*a, **kw)


_session = requests.Session()
_session.mount("https://www.tpex.org.tw", _RelaxedStrictAdapter())


def _get_json(url: str, params: dict, tries: int = 4) -> dict | list | None:
    delay = 3.0
    for k in range(tries):
        try:
            r = _session.get(url, params=params, headers=UA, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):        # 被限流，退避後重試
                time.sleep(delay)
                delay *= 2
                continue
            return None
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None


def trading_days(start: dt.date, end: dt.date):
    """週一到週五。國定假日不預先排除 —— API 回空就跳過，比維護假日表可靠。"""
    d = start
    one = dt.timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += one


# ─────────────────────────── 單日抓取 ───────────────────────────
def fetch_twse_day(d: dt.date) -> pd.DataFrame | None:
    """上市單日全市場。type=ALLBUT0999 排除權證，回應量只有 ALL 的 1/19。"""
    js = _get_json(TWSE_QUOTES, {"date": d.strftime("%Y%m%d"),
                                 "type": "ALLBUT0999", "response": "json"})
    if not isinstance(js, dict) or js.get("stat") != "OK":
        return None
    tbl = next((t for t in js.get("tables", []) if "每日收盤行情" in (t.get("title") or "")), None)
    if not tbl or not tbl.get("data"):
        return None

    f = tbl["fields"]
    i_code, i_name = _pick(f, "證券代號"), _pick(f, "證券名稱")
    i_vol, i_amt = _pick(f, "成交股數"), _pick(f, "成交金額")
    i_o, i_h = _pick(f, "開盤價"), _pick(f, "最高價")
    i_l, i_c = _pick(f, "最低價"), _pick(f, "收盤價")

    rows = []
    for r in tbl["data"]:
        code = str(r[i_code]).strip()
        if len(code) != 4 or not code.isdigit():      # 只留 4 碼普通股
            continue
        rows.append({
            "code": code, "name": str(r[i_name]).strip(), "market": "TWSE",
            "Open": _num(r[i_o]), "High": _num(r[i_h]),
            "Low": _num(r[i_l]), "Close": _num(r[i_c]),
            "Volume": _num(r[i_vol]), "Turnover": _num(r[i_amt]),
        })
    return pd.DataFrame(rows, columns=DAY_COLS) if rows else None


def fetch_tpex_day(d: dt.date) -> pd.DataFrame | None:
    """上櫃單日全市場。★ 無法除權息還原，見模組 docstring。"""
    js = _get_json(TPEX_QUOTES, {"date": d.strftime("%Y/%m/%d"),
                                 "type": "EW", "response": "json"})
    if not isinstance(js, dict) or not js.get("tables"):
        return None
    tbl = js["tables"][0]
    if not tbl.get("data"):
        return None

    f = tbl["fields"]
    i_code, i_name = _pick(f, "代號"), _pick(f, "名稱")
    i_o, i_h = _pick(f, "開盤"), _pick(f, "最高")
    i_l, i_c = _pick(f, "最低"), _pick(f, "收盤")
    i_vol, i_amt = _pick(f, "成交股數"), _pick(f, "成交金額")

    rows = []
    for r in tbl["data"]:
        code = str(r[i_code]).strip()
        if len(code) != 4 or not code.isdigit():
            continue
        rows.append({
            "code": code, "name": str(r[i_name]).strip(), "market": "TPEX",
            "Open": _num(r[i_o]), "High": _num(r[i_h]),
            "Low": _num(r[i_l]), "Close": _num(r[i_c]),
            "Volume": _num(r[i_vol]), "Turnover": _num(r[i_amt]),
        })
    return pd.DataFrame(rows, columns=DAY_COLS) if rows else None


# ─────────────────────────── 快取建置 ───────────────────────────
def _day_path(d: dt.date) -> Path:
    return DAY_DIR / f"{d:%Y-%m-%d}.parquet"


def build_cache(start: dt.date, end: dt.date, include_otc: bool = False,
                pause: float = 3.0, verbose: bool = True) -> int:
    """
    逐交易日抓取並落地。已存在的日期直接跳過 → 可中斷、可續跑。
    pause 是自律節流：TWSE 對高頻請求會短暫封 IP，3 秒是安全值。
    """
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    days = [d for d in trading_days(start, end) if not _day_path(d).exists()]
    if verbose:
        print(f"待抓 {len(days)} 個交易日（已快取的自動跳過）｜pause={pause}s"
              f"｜預估 {len(days) * pause * (2 if include_otc else 1) / 60:.0f} 分鐘")

    n_ok = 0
    for k, d in enumerate(days, 1):
        parts = []
        tw = fetch_twse_day(d)
        if tw is not None:
            parts.append(tw)
        time.sleep(pause)

        if include_otc:
            tp = fetch_tpex_day(d)
            if tp is not None:
                parts.append(tp)
            time.sleep(pause)

        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(_day_path(d), index=False)
            n_ok += 1
        else:
            # ★ 不要對「最近幾天」寫空檔佔位。空檔＝假日，會被永遠跳過；
            #   但當天資料常常只是還沒公布（T86 約 16:00 後、行情約 14:30 後），
            #   寫了佔位就再也補不回來。實際踩過：15:48 跑 → 09-07 被當成假日。
            if (dt.date.today() - d).days >= 3:
                _day_path(d).write_bytes(b"")      # 舊日期仍無資料 → 真的是假日
            elif verbose:
                print(f"      {d} 尚無資料（可能還沒公布），不佔位，下次再試", flush=True)

        if verbose and (k % 20 == 0 or k == len(days)):
            print(f"  [{k}/{len(days)}] {d} 累計有效 {n_ok} 天", flush=True)
    return n_ok


def backfill_otc(start: dt.date, end: dt.date, pause: float = 3.0,
                 verbose: bool = True) -> int:
    """
    把上櫃資料補進「已存在」的日期檔。

    build_cache 會跳過已存在的日期，所以先前只抓上市、之後才想加上櫃時，
    用這支回填。已經含 TPEX 的日期會自動跳過 → 一樣可中斷可續跑。
    """
    files = sorted(p for p in DAY_DIR.glob("*.parquet") if p.stat().st_size > 0)
    todo = []
    for p in files:
        d = dt.date.fromisoformat(p.stem)
        if not (start <= d <= end):
            continue
        try:
            if "TPEX" in set(pd.read_parquet(p, columns=["market"])["market"]):
                continue
        except Exception:
            continue
        todo.append((d, p))

    if verbose:
        print(f"待回填上櫃 {len(todo)} 個交易日｜pause={pause}s"
              f"｜預估 {len(todo) * pause / 60:.0f} 分鐘")

    n_ok = 0
    for k, (d, p) in enumerate(todo, 1):
        tp = fetch_tpex_day(d)
        if tp is not None and len(tp):
            pd.concat([pd.read_parquet(p), tp], ignore_index=True).to_parquet(p, index=False)
            n_ok += 1
        time.sleep(pause)
        if verbose and (k % 20 == 0 or k == len(todo)):
            print(f"  [{k}/{len(todo)}] {d} 成功 {n_ok} 天", flush=True)
    return n_ok


def fetch_adjust_factors(start: dt.date, end: dt.date, pause: float = 3.0,
                         verbose: bool = True, incremental: bool = True) -> pd.DataFrame:
    """
    TWT49U 除權除息計算結果表 → 還原因子。
    factor = 除權息參考價 / 除權息前收盤價（除息日當天以前的價格要乘上它）
    一次可查一段期間，但區間太長會被截斷，所以按季切。
    """
    CACHE.mkdir(parents=True, exist_ok=True)

    # 增量：歷史除權息紀錄不可變（2019 年的除息不會今天才改），
    # 每次從 2019 重抓 = 31 次請求、約 1.3 分鐘，全是白工。
    # 只補「最後一筆日期往前 30 天」到今天 —— 留 30 天緩衝以防證交所補登。
    existing = pd.DataFrame()
    if incremental and EXRIGHT_CSV.exists():
        try:
            existing = pd.read_csv(EXRIGHT_CSV, parse_dates=["date"])
            if not existing.empty:
                last = existing["date"].max().date()
                new_start = max(start, last - dt.timedelta(days=30))
                if new_start > start and verbose:
                    print(f"  增量模式：已有 {len(existing):,} 筆（最新 {last}），"
                          f"只補 {new_start} 之後", flush=True)
                start = new_start
        except Exception:
            existing = pd.DataFrame()

    rows = []
    cur = start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=90), end)
        js = _get_json(TWSE_EXRIGHT, {"startDate": cur.strftime("%Y%m%d"),
                                      "endDate": nxt.strftime("%Y%m%d"),
                                      "response": "json"})
        if isinstance(js, dict) and js.get("stat") == "OK" and js.get("data"):
            f = js["fields"]
            i_d, i_code = _pick(f, "資料日期"), _pick(f, "股票代號")
            i_pre, i_ref = _pick(f, "除權息前收盤價"), _pick(f, "除權息參考價")
            for r in js["data"]:
                code = str(r[i_code]).strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                pre, ref = _num(r[i_pre]), _num(r[i_ref])
                if not (pre > 0 and ref > 0):
                    continue
                date = _roc_date(r[i_d])                   # 民國「115年07月01日」
                if date is None:
                    continue
                rows.append({"date": date, "code": code, "factor": ref / pre})
        if verbose:
            print(f"  除權息 {cur}~{nxt}：累計 {len(rows)} 筆", flush=True)
        cur = nxt + dt.timedelta(days=1)
        time.sleep(pause)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    if not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    if not df.empty:
        # ★ code 必須統一型別再去重：從 CSV 讀回來是 int64（1203），
        #   新抓的是 str（"1203"），不統一就比對不到 → 重複列 → 還原因子被平方。
        df["code"] = df["code"].astype(str).str.strip().str.zfill(4)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])
    if not df.empty:
        df.to_csv(EXRIGHT_CSV, index=False)
    return df


# ─────────────────────────── 讀取 ───────────────────────────
def _load_raw(include_otc: bool = False) -> pd.DataFrame:
    files = sorted(p for p in DAY_DIR.glob("*.parquet") if p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"快取是空的，先跑：python twse.py --start 2019-01-01")
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        df["date"] = pd.Timestamp(p.stem)
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    if not include_otc:
        raw = raw[raw["market"] == "TWSE"]
    return raw


def _apply_adjust(df: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """
    往回累乘還原。factors 是這一檔的 {除權息日: factor}。
    做法：除權息日「當天及之後」不動，之前的全部乘上該因子（多次除權息會疊乘）。
    """
    if factors.empty:
        return df
    mult = pd.Series(1.0, index=df.index)
    for ex_date, f in factors.items():
        mult.loc[df.index < pd.Timestamp(ex_date)] *= f
    out = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        out[c] = out[c] * mult
    return out


def load_bars(include_otc: bool = False, adjust: bool = True,
              min_rows: int = 250, codes: list[str] | None = None,
              apply_exclusions: bool = True
              ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    回傳 ({code: DataFrame[Open,High,Low,Close,Volume,Turnover]}, {code: name})
    DataFrame 以日期為 index，已（可選）除權息還原。
    """
    raw = _load_raw(include_otc)
    if codes:
        raw = raw[raw["code"].isin(codes)]

    adj: pd.DataFrame | None = None
    if adjust and EXRIGHT_CSV.exists():
        adj = pd.read_csv(EXRIGHT_CSV, parse_dates=["date"])
    elif adjust:
        print("[warn] 找不到 exright.csv，這批資料未做除權息還原 —— "
              "長期回測報酬會被除息假跳空低估。先跑 python twse.py --exright", file=sys.stderr)

    # 全域排除（markets.yaml taiwan_exclude）：金融保險業、電信業者
    if apply_exclusions:
        import exclusions
        raw = raw[~exclusions.mask(raw["code"]).to_numpy()]

    names = dict(zip(raw["code"], raw["name"]))
    out: dict[str, pd.DataFrame] = {}
    for code, g in raw.groupby("code", sort=False):
        g = g.set_index("date").sort_index()
        g = g[["Open", "High", "Low", "Close", "Volume", "Turnover"]].dropna(
            subset=["Open", "High", "Low", "Close"])
        g = g[g["Close"] > 0]
        if len(g) < min_rows:
            continue
        if adj is not None:
            sub = adj[adj["code"].astype(str).str.zfill(4) == code]
            if not sub.empty:
                g = _apply_adjust(g, pd.Series(sub["factor"].values, index=sub["date"].values))
        out[code] = g
    return out, names


# ─────────────────────────── CLI ───────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="建立證交所行情快取")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--include-otc", action="store_true",
                    help="一併抓上櫃（★ 上櫃無法除權息還原，會污染跨市值比較）")
    ap.add_argument("--pause", type=float, default=3.0, help="每次請求間隔秒數")
    ap.add_argument("--exright", action="store_true", help="只更新除權息還原因子（預設增量）")
    ap.add_argument("--exright-full", action="store_true",
                    help="除權息從頭重抓（約 1.3 分鐘）。平常不需要，資料疑似有缺才用")
    ap.add_argument("--backfill-otc", action="store_true",
                    help="把上櫃補進已存在的日期檔（先前只抓上市時用這個）")
    ap.add_argument("--status", action="store_true", help="看目前快取狀態")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    if args.status:
        files = sorted(DAY_DIR.glob("*.parquet")) if DAY_DIR.exists() else []
        good = [p for p in files if p.stat().st_size > 0]
        print(f"快取：{len(files)} 個日期檔，其中 {len(good)} 天有資料")
        if good:
            print(f"範圍：{good[0].stem} ~ {good[-1].stem}")
            print(f"大小：{sum(p.stat().st_size for p in good) / 1e6:.1f} MB")
        print(f"除權息因子：{'有' if EXRIGHT_CSV.exists() else '無'}"
              + (f"（{len(pd.read_csv(EXRIGHT_CSV))} 筆）" if EXRIGHT_CSV.exists() else ""))
        return

    if args.exright or args.exright_full:
        df = fetch_adjust_factors(start, end, pause=args.pause,
                                  incremental=not args.exright_full)
        print(f"除權息因子 {len(df)} 筆 → {EXRIGHT_CSV}")
        return

    if args.backfill_otc:
        n = backfill_otc(start, end, pause=args.pause)
        print(f"上櫃回填完成，{n} 個交易日")
        return

    build_cache(start, end, include_otc=args.include_otc, pause=args.pause)
    print("\n接著跑除權息還原因子：python twse.py --exright --start "
          f"{args.start} --end {args.end}")


if __name__ == "__main__":
    main()
