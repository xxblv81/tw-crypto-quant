"""
台股全域排除清單 —— 設定在 markets.yaml 的 taiwan_exclude。

產業別來自 universe.py 存下的公司清單（cache/lists/twse.json、tpex.json）。
★ 已下市公司不在清單裡，查不到產業別；金融股代號絕大多數是 28xx，
  所以另外把「28 開頭」一律視為金融（上市金融保險業代碼 2801~2892 全在這段）。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import config

LISTS = Path(__file__).resolve().parent / "cache" / "lists"


@lru_cache(maxsize=1)
def excluded_codes() -> frozenset[str]:
    try:
        cfg = config.load_raw().get("taiwan_exclude", {}) or {}
    except Exception:
        cfg = {}
    industries = {str(x) for x in cfg.get("industries", [])}
    codes = {str(x) for x in cfg.get("codes", [])}

    for name, code_key, ind_key in (("twse", "公司代號", "產業別"),
                                    ("tpex", "SecuritiesCompanyCode", "SecuritiesIndustryCode")):
        path = LISTS / f"{name}.json"
        if not path.exists():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            ind = str(rec.get(ind_key, rec.get("產業別", ""))).strip()
            if ind in industries:
                codes.add(str(rec.get(code_key, rec.get("公司代號", ""))).strip())
    return frozenset(codes)


@lru_cache(maxsize=1)
def _industries() -> frozenset[str]:
    try:
        cfg = config.load_raw().get("taiwan_exclude", {}) or {}
    except Exception:
        cfg = {}
    return frozenset(str(x) for x in cfg.get("industries", []))


def is_excluded(code: str) -> bool:
    """單檔判斷（選股清單、why.py 用）。大量資料請用 mask()。"""
    code = str(code)
    return code in excluded_codes() or ("17" in _industries() and code.startswith("28"))


def mask(codes) -> "pd.Series":
    """
    整批判斷，回傳 True＝要排除。
    ★ 行情有數百萬列，不能逐列呼叫 is_excluded —— 那會讓 load_bars 從幾十秒變成十幾分鐘。
    """
    import pandas as pd
    codes = pd.Series(codes).astype(str)
    m = codes.isin(excluded_codes())
    if "17" in _industries():
        m = m | codes.str.startswith("28")
    return m
