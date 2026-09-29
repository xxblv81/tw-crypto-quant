#!/usr/bin/env bash
# 台股全流程一鍵執行。第一次跑請改用 CLAUDE.md 的 checklist 逐步確認。
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
[ -x "$PY" ] || { echo "找不到 .venv，先跑：python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"; exit 1; }

echo "=== ① 確認參數讀取 ==="
$PY config.py

echo -e "\n=== ② 抓取全台股清單 ==="
$PY universe.py

echo -e "\n=== ③ 更新證交所行情快取（已有的日期會跳過）==="
$PY twse.py --start 2019-01-01
$PY twse.py --exright --start 2019-01-01

echo -e "\n=== ③b 更新籌碼快取（投信買賣超；落後會讓動能榜歸零）==="
$PY chips.py --start 2019-01-01

echo -e "\n=== ④ 今日選股（排除成交金額前 100 大）==="
$PY screen.py --small-cap

echo -e "\n=== ⑤ 全市場回測（依成交金額分組）==="
$PY backtest.py --buckets 4

echo -e "\n=== ⑥ 期貨短線動能選股（3~14 天）==="
$PY momentum_screen.py --futures --min-score 8.0

echo -e "\n=== ⑦ 期貨短線回測（含槓桿表）==="
$PY momentum_backtest.py --futures

echo -e "\n完成。產出：universe.csv / signals_*.csv / backtest_results.csv / backtest_summary.csv"
