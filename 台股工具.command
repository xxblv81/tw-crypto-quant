#!/bin/bash
# ============================================================
# 台股選股工具 — 點兩下就能跑的選單
#
# 這是給 Finder 雙擊用的啟動器，不是主程式。
# 真正的邏輯在 taiwan/python/ 底下，指令行用法見 README.md。
#
# ★ 桌面上那份請放「轉呼叫這支」的兩行小殼，不要放複本 ——
#   複本會過期，實際踩過：桌面版停在只有 1~4 個選項的舊版本。
#   重建桌面捷徑：
#       printf '#!/bin/bash\nexec "%s"\n' "此檔絕對路徑" > ~/Desktop/台股工具.command
#       chmod +x ~/Desktop/台股工具.command
#
# 專案內有三套獨立策略，共用 taiwan/python/ 的資料層：
#   布林縮口突破／動能選股（個股期短線）
# ============================================================

PROJ="/Users/august/Claude/Projects/投資"
PYDIR="$PROJ/taiwan/python"
PY="$PYDIR/.venv/bin/python"

echo -ne "\033]0;台股選股工具\007"          # 設定終端機視窗標題
[ -t 1 ] && clear                            # 只在真的終端機裡清畫面

# ── 收尾：不管成功失敗都停住，讓使用者看得到訊息 ──
pause_exit() {
    echo
    echo "────────────────────────────────────────"
    echo "按任意鍵關閉這個視窗…"
    read -n 1 -s
    exit "${1:-0}"
}

# ── 環境檢查 ──
if [ ! -d "$PYDIR" ]; then
    echo "✗ 找不到專案資料夾："
    echo "  $PYDIR"
    echo
    echo "  專案可能被移動或改名了。把這個檔案裡的 PROJ 路徑改成新位置即可。"
    pause_exit 1
fi

if [ ! -x "$PY" ]; then
    echo "✗ 找不到 Python 環境（.venv）"
    echo
    echo "  在終端機執行這行重建："
    echo "  cd \"$PYDIR\" && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
    pause_exit 1
fi

cd "$PYDIR" || pause_exit 1

# ── 共用函式 ──
update_price_cache() {
    echo "→ 行情與除權息（只補缺的日期，每天約 4 秒）…"
    "$PY" twse.py --start 2019-01-01 --include-otc || return 1
    "$PY" twse.py --exright --start 2019-01-01 || return 1
}

update_index_cache() {
    # 加權指數：大盤狀態的輔助訊號（主訊號「寬度」用現成行情算，不需要它）。
    # FMTQIK 一個月一次請求，只補缺的月份，日常增量約 1~2 次。
    echo "→ 加權指數（大盤狀態）…"
    "$PY" market.py --refresh || return 1
}

update_chips_cache() {
    # ★ 籌碼只要落後行情一天，動能分數就會全市場一起少掉 0.15×10 分、榜單歸零。
    #   所以凡是要跑動能的選項都先補這個。已有的日期自動跳過，增量通常 1 天（3 秒）。
    echo "→ 三大法人買賣超…"
    "$PY" chips.py --start 2019-01-01 || return 1
}

open_latest() {
    latest=$(ls -t "$1"*.csv 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo
        echo "→ 開啟 $(basename "$latest")"
        open "$latest"
    fi
}

die() { echo; echo "✗ $1"; pause_exit 1; }

echo "═══════════════════════════════════════════════════"
echo "   台股選股工具"
echo "═══════════════════════════════════════════════════"
echo

# ── 開場自動補行情（2026-09-27 使用者指示）──
# 使用者不跑排程、只手動開這支，所以行情一定要在「選單之前」補完，
# 不要求使用者自己先選 3。已有的日期自動跳過，日常增量約 4 秒。
# 失敗不中斷 —— 舊快取還能跑回測，只是資料停在上次成功的日期。
echo "【自動更新行情】"
if update_price_cache; then
    PRICE_FRESH=1
else
    PRICE_FRESH=0
    echo
    echo "⚠ 行情更新失敗（網路或證交所暫時擋連線）。"
    echo "  選單照開，但資料只到上次成功的日期 —— 選股結果會是舊的。"
fi
echo

# ── 快取狀態 ──
echo "【行情】"
"$PY" twse.py --status 2>/dev/null || echo "（行情快取狀態讀取失敗）"
echo "【籌碼】"
"$PY" chips.py --status 2>/dev/null | head -2
echo "  ★ 籌碼落後行情時，動能榜會自動退回不含籌碼計分（選項 2 會先補）"
echo

# ── 主選單（精簡版）──
# 平常只顯示每天會用到的四個。其餘功能仍在，輸入 m 展開。
# 舊的編號全部保留可用 —— 直接打 7、12、14 也能跑，不必先按 m。
echo "───────────────────────────────────────────────────"
echo "   1  布林縮口突破選股"
echo "   2  動能選股（期貨短線 3~14 天）"
echo
echo "   3  更新所有資料"
echo "   4  回測（兩套策略一起跑，數分鐘）"
echo "   0  離開"
echo "───────────────────────────────────────────────────"
echo

read -r -p "請輸入數字後按 Enter： " choice
echo

case "$choice" in
    # ─────────── 選股 ───────────
    2)
        # 動能選股。三個參數的由來（完整依據見 taiwan/MOMENTUM_FUTURES.md）：
        #   --futures     與「有個股期且日成交 ≥500 口」的清單取交集。237 檔有個股期，
        #                 但實際有量的只有約 103 檔 —— 個股期真正的成本是買賣價差。
        #   --min-score   用 8.0 而不是設定檔的 8.5：籌碼開啟後總分上限是
        #                 (1-0.15)×10 = 8.5，多數個股當天投信沒動作＝籌碼 0 分，
        #                 8.5 會篩到剩十來檔。實測 2026-09-01：8.5→11 檔、8.0→22 檔。
        #   先 futures.py --refresh：期交所 OpenAPI 只給「最近一個交易日」的量，
        #                 沒有歷史，所以流動性一定要當天更新才準。
        [ "$PRICE_FRESH" = "1" ] || echo "⚠ 行情是舊的，這批選股不代表今天的盤面。"
        update_chips_cache || echo "  （籌碼更新失敗，選股會自動退回不含籌碼計分）"
        update_index_cache || echo "  （指數更新失敗，大盤狀態會只用寬度判斷）"
        echo "→ 更新個股期清單與流動性…"
        "$PY" futures.py --refresh || die "期交所連線失敗。"
        echo
        echo "→ 計算動能分數並與可交易個股期取交集…"
        echo
        "$PY" momentum_screen.py --futures --min-score 8.0 --top 40 || die "選股失敗。"
        open_latest "momentum_futures_"
        # 選股只給分數排行，還不知道「幾口、保證金多少、買不買得動」。
        # 契約規模兩種：一般 2,000 股、小型 100 股。一口名目中位數約 16 萬、
        # 一口保證金中位數約 3 萬，但分布極廣，不算口數等於沒法執行。
        echo
        echo "───────────────────────────────────────────────────"
        read -r -p "要算執行表嗎？輸入今天可用的保證金預算（直接 Enter 跳過）： " cap
        if [ -n "$cap" ]; then
            echo
            "$PY" momentum_plan.py --margin "$cap" --slots 8 --min-score 7.5 --fit-only \
                || die "執行表計算失敗。"
        fi
        ;;
    1)
        # 行情已在開場自動補過，這裡不再重複。
        [ "$PRICE_FRESH" = "1" ] || echo "⚠ 行情是舊的，這批選股不代表今天的盤面。"
        echo "→ 掃描中…"
        echo
        # 這裡可以加 --include-otc：screen.py 自己的說明是「選股用可接受，回測不建議」。
        # 上櫃無除權息還原，但選股只看最近幾根 K 棒，還原與否影響很小；
        # 回測跨七年就會被除息假跳空系統性低估（所以選項 9 沒有加）。
        if [ "$choice" = "5" ]; then
            "$PY" screen.py --include-otc --small-cap || die "選股失敗。"
        else
            "$PY" screen.py --include-otc || die "選股失敗。"
        fi
        open_latest "signals_"
        ;;



    # ─────────── 回測 ───────────
    4)
        echo "→ ① 布林縮口突破全市場回測（依成交金額分四組）…"
        echo
        # ★ 沒有 --include-otc：backtest.py 自己的說明就寫「上櫃無除權息還原資料，
        #   混入會讓中小型組被系統性低估」，而這支的重點正是比大小型股。
        "$PY" backtest.py --buckets 4 || die "縮口突破回測失敗。"
        open_latest "backtest_summary"
        echo
        echo "→ ② 期貨短線動能回測（3~14 天，含槓桿表）…"
        echo
        # 宇宙用 point-in-time 前 150 大，不能用「今天有量的個股期清單」——
        # 那是前視偏誤，實測會把 MA20 的 PF 從 1.711 灌水到 2.428。
        "$PY" momentum_backtest.py --futures || die "動能回測失敗。"
        open_latest "momentum_trades"
        ;;

    # ─────────── 維護 ───────────
    3)
        # 順序有意義：行情要先補，因為籌碼／融資的「落後與否」是拿行情最後一天當基準。
        # 個股期清單放最後且失敗不中斷 —— 它只影響選項 1 的流動性過濾，
        # 舊清單頂多是流動性資訊過期，不會讓其他選項失效。
        # 行情由開場自動處理，這個選項的價值剩下「籌碼／指數／個股期一次補齊」。
        # 只更新兩套策略真正會讀的：
        #   行情＋除權息 → 布林縮口、動能  ｜ 籌碼(投信) → 動能計分
        #   指數 → 動能的大盤狀態          ｜ 個股期清單 → 動能的流動性過濾
        # 集保(tdcc)、融資(margin)、營收(fundamentals) 是已移除策略用的，不再更新。
        # 行情已在開場自動補過。這裡若仍是舊的就再試一次，成功了就不重跑。
        [ "$PRICE_FRESH" = "1" ] || update_price_cache || die "行情更新失敗。"
        update_chips_cache  || die "籌碼更新失敗。"
        update_index_cache  || echo "  （指數更新失敗，大盤狀態會只用寬度判斷）"
        echo "→ 個股期清單與流動性…"
        "$PY" futures.py --refresh || echo "  （期交所連線失敗，期貨選股會用上次的清單）"
        echo
        echo "═══ 各快取現況 ═══"
        "$PY" twse.py --status
        "$PY" chips.py --status
        ;;

    0|"")
        echo "掰掰。"
        ;;
    *)
        echo "✗ 沒有這個選項：$choice"
        ;;
esac

pause_exit 0
