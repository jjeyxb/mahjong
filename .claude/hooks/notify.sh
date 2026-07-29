#!/bin/bash
# Claude Code 結束回合時發一則 macOS 通知。
#
# 由 .claude/settings.json 的 Stop hook 觸發。hook 從 stdin 收到一個 JSON,
# 其中 cwd 是專案目錄 —— 用它當標題,同時開多個專案時才分得出是哪一個。
#
# 刻意不解析 transcript 產生摘要:hook 是同步阻塞的,每多花一秒都會讓回合
# 感覺變慢。標題 + 專案名 + 時間就夠回答「可以回去看了嗎」。

payload=$(cat)
project=$(printf '%s' "$payload" | /usr/bin/python3 -c \
    'import json,sys,os; print(os.path.basename(json.load(sys.stdin).get("cwd") or os.getcwd()))' \
    2>/dev/null || basename "$PWD")

osascript -e "display notification \"完成於 $(date '+%H:%M:%S')\" with title \"Claude Code\" subtitle \"$project\" sound name \"Glass\"" >/dev/null 2>&1

exit 0
