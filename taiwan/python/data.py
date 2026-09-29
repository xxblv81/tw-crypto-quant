"""
行情下載 — 三後端。

後端 T：TWSE 證交所官方 API（預設，見 twse.py）
        逐日抓全市場 → 本地 parquet 快取 → 除權息還原。免 token。
後端 A：yfinance（免 token，備援）
後端 B：FinMind（需 token；上櫃股若要還原只能靠它）

★ 除權息還原很重要：未還原的長期回測報酬率不可信。
  台股一次除息中位數就吃掉 3.7% 價格（實測 2019~2026、7007 筆），
  在 stop_atr=4.0 的日線策略上，未還原的假跳空會觸發假停損。
  twse 後端用 TWT49U 算還原因子；yfinance 用 auto_adjust=True。
"""
from __future__ import annotations
import time
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_twse(codes: list[str] | None = None, include_otc: bool = False,
               adjust: bool = True, min_rows: int = 250
               ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    證交所後端。讀本地快取，不連網 —— 快取要先用 `python twse.py --start ...` 建好。
    回傳 ({code: df}, {code: name})，df 多一個 Turnover 欄（官方成交金額，未受還原影響）。
    """
    import twse
    return twse.load_bars(include_otc=include_otc, adjust=adjust,
                          min_rows=min_rows, codes=codes)


def avg_turnover(df: pd.DataFrame, window: int) -> float:
    """
    近 N 日均成交金額。
    優先用官方 Turnover 欄：還原後的 Close×Volume 會系統性低估歷史成交金額。
    """
    if "Turnover" in df.columns and df["Turnover"].notna().any():
        return float(df["Turnover"].tail(window).mean())
    return float((df["Close"] * df["Volume"]).tail(window).mean())


def fetch_yf(tickers: list[str], period: str = "10y", pause: float = 0.0) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    B = 40                                              # 分批，避免被限流
    for i in range(0, len(tickers), B):
        batch = tickers[i:i + B]
        df = yf.download(batch, period=period, interval="1d",
                         auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for t in batch:
            try:
                d = df[t] if len(batch) > 1 else df
                d = d[COLS].dropna()
                if len(d) >= 250:                       # 至少一年資料才收
                    out[t] = d
            except Exception:
                pass
        if pause:
            time.sleep(pause)
    return out


def fetch_finmind(codes: list[str], token: str, start: str = "2015-01-01") -> dict[str, pd.DataFrame]:
    from FinMind.data import DataLoader
    dl = DataLoader()
    dl.login_by_token(api_token=token)
    out = {}
    for c in codes:
        try:
            d = dl.taiwan_stock_daily_adj(stock_id=c, start_date=start)
            if d is None or len(d) < 250:
                continue
            d = d.rename(columns={"open": "Open", "max": "High", "min": "Low",
                                  "close": "Close", "Trading_Volume": "Volume"})
            d["date"] = pd.to_datetime(d["date"])
            out[c] = d.set_index("date")[COLS].dropna()
        except Exception:
            pass
    return out
