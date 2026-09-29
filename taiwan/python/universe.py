"""
抓取台股交易標的清單。

輸出 DataFrame：code, name, market, industry
資料源：證交所 / 櫃買中心公開 OpenAPI（免費、免 token）

涵蓋範圍與限制（2026-09-01 實測）：

  上市 TWSE      ✓ 完整。**創新板已含在內**（代號同為 4 碼，名稱帶「-創」，
                   例如 7803 雲象科技-創），不需要另外處理。
  上櫃 TPEx      ✓ 行情可取，但**無歷史除權息資料**可還原，見 twse.py。
  興櫃           ✗ **無法納入回測**。TPEx 只提供「當日」報價
                   （openapi/v1/tpex_esb_latest_statistics），
                   批次歷史日行情端點全部 404（emerging/dailyQuotes、
                   emerging/historical、舊站 EMdaily_result.php 皆試過）。
                   沒有歷史就算不出布林通道與 200 日均線，策略無法運作。
                   另外興櫃是議價（非集中競價）市場，流動性極薄，
                   就算有資料，突破訊號也多半成交不了。

  ★ 這支回傳的是「目前仍在市」的清單。下市公司不在裡面 —— 那會造成倖存者偏差。
    回測請改用 twse.py 快取裡出現過的所有代號（含已下市），
    見 backtest.py 的 --exclude-delisted 說明。
"""
from __future__ import annotations
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

from twse import _session, UA, TIMEOUT      # 共用含 TPEx SSL 修正的 session

TWSE_LIST = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"      # 上市公司基本資料
TPEX_LIST = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"   # 上櫃公司基本資料


LIST_CACHE = Path(__file__).resolve().parent / "cache" / "lists"


def _fetch_json(url: str, tries: int = 4) -> list[dict]:
    """
    ★ TPEx 伺服器會偶發切斷連線（ConnectionResetError／IncompleteRead，下載到一半斷）。
      2026-09-14 實測連抓 5 次：第 1 次斷、後 4 次都正常 —— 是偶發，重試就過。
      舊版只試一次、失敗就默默退回只用上市，當天選股會少掉 890 檔上櫃股。
    """
    delay = 2.0
    last: Exception | None = None
    for k in range(tries):
        try:
            r = _session.get(url, headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:          # 斷線、截斷、JSON 不完整都算
            last = e
            if k < tries - 1:
                time.sleep(delay)
                delay *= 2
    raise last  # type: ignore[misc]


def _cached_list(name: str, url: str) -> tuple[list[dict], str | None]:
    """
    抓清單；成功就存一份。重試全失敗時退回上次成功的那份（公司清單很少變動）。
    回傳 (資料, 若用了舊檔則為舊檔日期，否則 None)。
    """
    LIST_CACHE.mkdir(parents=True, exist_ok=True)
    path = LIST_CACHE / f"{name}.json"
    try:
        data = _fetch_json(url)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data, None
    except Exception as e:
        if path.exists():
            stamp = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
            print(f"[warn] {name} 清單連線失敗（{type(e).__name__}），改用 {stamp} 存的舊清單",
                  file=sys.stderr)
            return json.loads(path.read_text(encoding="utf-8")), stamp
        raise


def fetch_universe(include_otc: bool = True) -> pd.DataFrame:
    rows = []
    twse_data, _ = _cached_list("twse", TWSE_LIST)
    for rec in twse_data:
        rows.append({
            "code": str(rec.get("公司代號", "")).strip(),
            "name": str(rec.get("公司簡稱", "")).strip(),
            "market": "TWSE",
            "industry": str(rec.get("產業別", "")).strip(),
        })
    if include_otc:
        try:
            tpex_data, _ = _cached_list("tpex", TPEX_LIST)
            for rec in tpex_data:
                rows.append({
                    "code": str(rec.get("SecuritiesCompanyCode", rec.get("公司代號", ""))).strip(),
                    "name": str(rec.get("CompanyAbbreviation", rec.get("公司簡稱", ""))).strip(),
                    "market": "TPEX",
                    "industry": str(rec.get("SecuritiesIndustryCode", rec.get("產業別", ""))).strip(),
                })
        except Exception as e:
            # 重試全敗 且 從未成功存過清單 —— 只有第一次跑才可能走到這裡。講清楚，不要默默少一半。
            print("\n" + "!" * 60, file=sys.stderr)
            print(f"[錯誤] 上櫃清單抓不到且沒有舊清單可用：{e}", file=sys.stderr)
            print("       本次只含上市，上櫃約 890 檔不在結果內。稍後重跑即可。", file=sys.stderr)
            print("!" * 60 + "\n", file=sys.stderr)

    df = pd.DataFrame(rows)
    df = df[df["code"].str.fullmatch(r"\d{4}")]        # 只留 4 碼普通股，濾掉 ETF/權證/特別股
    df = df.drop_duplicates("code").reset_index(drop=True)
    return df


def yf_ticker(code: str, market: str) -> str:
    """轉成 yfinance 代號：上市 .TW，上櫃 .TWO"""
    return f"{code}.TW" if market == "TWSE" else f"{code}.TWO"


if __name__ == "__main__":
    u = fetch_universe()
    u["yf"] = [yf_ticker(c, m) for c, m in zip(u["code"], u["market"])]
    u.to_csv("universe.csv", index=False, encoding="utf-8-sig")
    print(f"共 {len(u)} 檔（上市 {sum(u.market=='TWSE')} / 上櫃 {sum(u.market=='TPEX')}）→ universe.csv")
