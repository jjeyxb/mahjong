"""假的引擎子程序,用來測 SubprocessEngine 而不必扯進 torch。

真正的 Mortal 子程序要 Python 3.12、130 MB 權重與幾秒的載入時間,拿它跑單元
測試會讓整組測試變成分鐘級,而且在沒有建 venv 的機器上直接掛掉。這個腳本用
測試自己的直譯器就能跑,並且可以**故意壞掉** —— 崩潰、逾時、亂輸出這些路徑
用真的引擎反而難重現。

各種壞法由命令列參數指定,預設是乖乖照協定回話。
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hello", action="store_true", help="不送握手就直接開始")
    parser.add_argument("--wrong-hello", action="store_true", help="第一行送別的 JSON 而非握手")
    parser.add_argument("--hello-delay", type=float, default=0.0, help="握手前先睡幾秒")
    parser.add_argument("--reply-delay", type=float, default=0.0, help="每次回覆前睡幾秒")
    parser.add_argument("--die-after", type=int, default=-1, help="收到第 N 個事件後直接結束")
    parser.add_argument("--noise", action="store_true", help="握手前往 stdout 印非 JSON 的雜訊")
    parser.add_argument("--reply-error", action="store_true", help="一律回 type=error")
    parser.add_argument("--reply-garbage", action="store_true", help="一律回不是 JSON 的東西")
    parser.add_argument("--stderr", default="", help="啟動時往 stderr 寫一行")
    args = parser.parse_args()

    if args.stderr:
        print(args.stderr, file=sys.stderr, flush=True)
    if args.noise:
        print("loading some third-party thing...", flush=True)
    if args.hello_delay:
        time.sleep(args.hello_delay)
    if args.wrong_hello:
        emit({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
    elif not args.no_hello:
        emit({"type": "hello", "name": "fake", "detail": "test"})

    seen = 0
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        seen += 1
        if args.die_after >= 0 and seen > args.die_after:
            return 3
        if args.reply_delay:
            time.sleep(args.reply_delay)
        if args.reply_garbage:
            sys.stdout.write("not json at all\n")
            sys.stdout.flush()
            continue
        if args.reply_error:
            emit({"type": "error", "message": "壞掉了"})
            continue

        event = json.loads(line)
        # 只在 tsumo 時動作,其餘回 none —— 與真正的引擎同樣是一進一出
        if event.get("type") == "tsumo":
            emit(
                {
                    "type": "dahai",
                    "actor": event["actor"],
                    "pai": event["pai"],
                    "tsumogiri": True,
                    "meta": {"seen": seen},
                }
            )
        else:
            emit({"type": "none"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
