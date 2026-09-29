"""
參數同步檢查器 — 驗證 Pine 檔的 input 預設值與 markets.yaml 一致。

存在理由：markets.yaml 是單一事實來源，但 Pine 的 input 預設值只能手動同步。
手動的東西一定會漂移，漂移之後 TradingView 上看到的績效就不是 markets.yaml
描述的那組參數，兩邊對不起來卻沒人發現。這支把「手動」變成「可驗證」。

用法：
    python check_sync.py            # 檢查全部，有落差回傳 exit code 1
    python check_sync.py --market taiwan
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[2]

# markets.yaml 的鍵 → Pine 變數名
PARAM_MAP = {
    "bb_length": "length",
    "bb_mult": "mult",
    "bbw_lookback": "bbwLen",
    "squeeze_window": "sqWin",
    "require_expanding": "useExp",
    "entry_mode": "entryMode",
    "break_on_high": "useHigh",
    "require_touch_lower": "useTouch",
    "touch_window": "tcWin",
    "trend_filter": "useTrend",
    "trend_ma_length": "trendLen",
    "mid_cross_window": "midWin",
    "volume_since_cross_mult": "volSince",
    "mid_confirm_min_gap": "midGap",
    "require_bull_candle": "useBullK",
    "trend_filter_2": "useMA2",
    "trend_ma2_length": "ma2Len",
    "allow_short": "useShort",
    "allow_long": "useLong",
    "volume_filter": "useVol",
    "volume_ma_length": "volLen",
    "volume_mult": "volMult",
    "atr_length": "atrLen",
    "stop_atr": "slATR",
    "take_profit": "useTP",
    "rr": "rr",
    "exit_ma": "exitMa",
    "trailing": "useTrail",
    "trail_trigger_atr": "trigATR",
    "trail_offset_atr": "trailATR",
}

# markets.yaml 的 entry_mode 值 → Pine 的中文選項字串
ENTRY_MODE = {"upper_break": "上軌突破", "mid_reclaim": "中線反轉", "mid_confirm": "中線確認"}

# 2026-09 新增的選配參數。只在 crypto 導入，台股 Pine 尚未加，
# 兩邊都沒有時視為一致（有一邊有、另一邊沒有才算漂移）。
OPTIONAL = {"trend_filter_2", "trend_ma2_length", "allow_short", "allow_long",
            "mid_cross_window", "mid_confirm_min_gap", "require_bull_candle",
            "volume_since_cross_mult", "ma_stack", "ma_stack_rising", "expand_mult",
            "break_on_high", "exit_ma"}

# costs 區塊 → strategy() 標頭參數
COST_MAP = {
    "commission_pct_per_side": "commission_value",
    "slippage_ticks": "slippage",
    "initial_capital": "initial_capital",
}

PINE_FILES = {"crypto": "crypto/strategy.pine", "taiwan": "taiwan/strategy.pine"}


def parse_pine_inputs(text: str) -> dict[str, object]:
    """抓 `名稱 = input.型別(預設值, ...)` 的預設值。"""
    out: dict[str, object] = {}
    pat = re.compile(r"^\s*(\w+)\s*=\s*input\.(int|float|bool|string)\(\s*"
                     r"(\"[^\"]*\"|[^,()]+?)\s*[,)]", re.M)
    for name, typ, raw in pat.findall(text):
        raw = raw.strip()
        if typ == "bool":
            out[name] = raw == "true"
        elif typ == "string":
            out[name] = raw.strip('"')
        else:
            try:
                out[name] = float(raw) if typ == "float" else int(raw)
            except ValueError:
                out[name] = raw
    return out


def parse_strategy_header(text: str) -> dict[str, float]:
    """抓 strategy(...) 標頭裡的成本設定。"""
    out: dict[str, float] = {}
    for key in ("initial_capital", "commission_value", "slippage"):
        m = re.search(rf"\b{key}\s*=\s*([\d.]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a) == str(b)


def check(market: str) -> list[str]:
    path = ROOT / PINE_FILES[market]
    if not path.exists():
        return [f"✗ 找不到 {path}"]

    text = path.read_text(encoding="utf-8")
    pine = parse_pine_inputs(text)
    header = parse_strategy_header(text)
    yml = config.params(market)
    costs = config.costs(market)

    problems: list[str] = []

    for ykey, pname in PARAM_MAP.items():
        if ykey not in yml:
            # 選配參數：兩邊都沒有就算一致（台股尚未導入第二層濾網與做空）
            if ykey in OPTIONAL and pname not in pine:
                continue
            problems.append(f"  ! markets.yaml[{market}].params 缺少 {ykey}")
            continue
        if pname not in pine:
            problems.append(f"  ! Pine 找不到 input 變數 {pname}（對應 {ykey}）")
            continue
        want = yml[ykey]
        if ykey == "entry_mode":
            want = ENTRY_MODE.get(want, want)
        if not _eq(want, pine[pname]):
            problems.append(f"  ✗ {ykey}: markets.yaml={want!r}  vs  Pine {pname}={pine[pname]!r}")

    for ckey, hname in COST_MAP.items():
        if ckey in costs and hname in header and not _eq(costs[ckey], header[hname]):
            problems.append(f"  ✗ costs.{ckey}: markets.yaml={costs[ckey]!r}  vs  Pine {hname}={header[hname]!r}")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "taiwan", "all"], default="all")
    args = ap.parse_args()

    markets = ["crypto", "taiwan"] if args.market == "all" else [args.market]
    bad = 0
    for m in markets:
        problems = check(m)
        print(f"=== {m}  ({PINE_FILES[m]}) ===")
        if problems:
            bad += len(problems)
            print("\n".join(problems))
        else:
            print("  ✓ Pine 預設值與 markets.yaml 完全一致")
        print()

    if bad:
        print(f"共 {bad} 處不一致 —— 改 markets.yaml 或同步 Pine 預設值。")
        sys.exit(1)
    print("全部同步。")


if __name__ == "__main__":
    main()
