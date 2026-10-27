#!/usr/bin/env python
"""把錄影餵進真正的 TableWorker,量「跑起來」對不對::

    python tools/replay_table.py data/recordings/20261024-123746 data/gt/ws.jsonl

與 ``tools/eval_rivers.py`` 吃**同一組素材**,但問的是不同的問題:

* ``eval_rivers.py``:讀圖的函式對不對。它自己挑動畫演完的幀、每幀從頭認。
* 這一支:app 裡那條 1 Hz 的迴圈對不對。它拿到的是當下最新的那一張
  (可能正好是動畫中途),而且只認新增的格子 —— 錯誤會留在狀態裡。

先跑 ``--dry-run``:那一步不辨識,幾秒就知道兩份檔案對不對得上、
照這個節奏會醒幾輪。

這份報告最該看哪一欄
--------------------
**「假現物」。** 其他每一欄錯了都還有症狀:棄權會被當成危險(保守),
少讀一格會讓張數對不上。只有「tracker 的河裡有、桌上沒有」那幾張是
**無症狀的**:放銃分析會說那張一定安全,而它可能正是別家的待牌。

為什麼是單行程
--------------
``eval_rivers.py`` 開多行程衝吞吐量,因為它每一幀彼此獨立。這裡不行:
增量讀取的狀態是**一條鏈**,第 n 輪看到什麼取決於前 n-1 輪。而且要量的
正是「一輪來不來得及」—— 四個工人互搶核心的話那個數字沒有意義。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.config.loader import load_config
from mia.eval.align import SETTLE
from mia.eval.material import parse_table_rect, require_one_calibration
from mia.eval.replay import replay, schedule
from mia.eval.rivers import build_river_timeline
from mia.recorder import SessionReader
from mia.utils.geometry import Rect
from mia.utils.logging import setup_logging
from mia.vision.tiles.classify import DEFAULT_SKIN


def _table_rect(session: SessionReader, args: argparse.Namespace) -> Rect:
    """這份素材要用哪個牌桌矩形 —— 與 ``tools/eval_rivers.py`` 同一套三道檢查。

    四家角點與 ROI 全部相對它正規化,所以它錯掉的時候報告**看起來完全正常**,
    只是每一個數字都是假的。
    """
    if args.table_rect is not None:
        rect = parse_table_rect(args.table_rect, session.image_size)
        print(f"手動指定牌桌矩形:{rect}(蓋過錄製時存的 {session.table_rect})")
        return rect
    if session.table_rect is None:
        raise SystemExit(
            "session 沒有 table_rect —— 錄製時校正沒鎖定。"
            "可以用 tools/frame_table.py 框一個,再用 --table-rect 傳進來。"
        )
    require_one_calibration(session, anyway=args.anyway)
    return session.table_rect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path, help="tools/record.py 錄的 session 目錄")
    parser.add_argument("dump", type=Path, help="tools/gt.py 錄的 WebSocket .jsonl")
    parser.add_argument("--skin", default=None, help="牌面模板皮膚,預設用內建的")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="worker 的節奏(秒),預設 1.0"
    )
    parser.add_argument("--settle", type=float, default=None, help="動畫穩定秒數")
    parser.add_argument("--limit", type=int, default=0, help="最多跑幾輪,0 表示全部")
    parser.add_argument("--dry-run", action="store_true", help="只檢查對得上對不上")
    parser.add_argument(
        "--anyway", action="store_true", help="校正中途換過也照跑(數字量的會是校正)"
    )
    parser.add_argument(
        "--table-rect",
        default=None,
        metavar="X,Y,寬,高",
        help="手動指定牌桌矩形(0~1 的比例),蓋過錄製時存的。見 tools/frame_table.py",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level)

    session = SessionReader(args.session)
    table = _table_rect(session, args)
    print(f"畫面 {args.session}:{len(session.manifest.frames)} 幀")
    timeline = build_river_timeline(args.dump)
    print(f"封包 {args.dump}:{len(timeline)} 個牌河狀態,自己坐 {timeline.seat}")

    settle = args.settle if args.settle is not None else SETTLE

    if args.dry_run:
        print("\n節奏 → 會醒幾輪 / 其中幾輪手上沒有新畫面")
        for value in (0.5, 1.0, 2.0):
            ticks = list(schedule(session, interval=value))
            starved = sum(1 for t in ticks if t.record is None)
            print(f"  {value:.1f}s  {len(ticks):5d} 輪,沒畫面 {starved}")
        return 0

    ticks = len(list(schedule(session, interval=args.interval)))
    if args.limit:
        ticks = min(ticks, args.limit)
    if not ticks:
        print("\n一輪都排不出來 —— 這份錄影是空的?")
        return 1
    print(f"\n節奏 {args.interval:.1f}s,共 {ticks} 輪,單行程,開始…")

    started = time.monotonic()

    def tick(done: int, total: int) -> None:
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0.0
        left = (total - done) / rate if rate else 0.0
        print(
            f"\r  {done}/{total} 輪  {rate:.1f} 輪/秒  剩約 {left / 60:.1f} 分 ",
            end="",
            flush=True,
        )

    report = replay(
        session,
        timeline,
        table,
        config=load_config().roi,
        interval=args.interval,
        skin=args.skin or DEFAULT_SKIN,
        settle=settle,
        limit=args.limit,
        on_tick=tick,
    )
    print("\n")
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
