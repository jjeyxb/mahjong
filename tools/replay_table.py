#!/usr/bin/env python
"""把錄影餵進真正的 TableWorker,量「跑起來」對不對::

    python tools/replay_table.py data/recordings/20261024-123746 data/gt/ws.jsonl

與 ``tools/eval_rivers.py`` 吃**同一組素材**,但問的是不同的問題:

* ``eval_rivers.py``:讀圖的函式對不對。它自己挑動畫演完的幀、每幀從頭認。
* 這一支:app 裡那條 1 Hz 的迴圈對不對。它拿到的是當下最新的那一張
  (可能正好是動畫中途),而且只認新增的格子 —— 錯誤會留在狀態裡。

先跑 ``--dry-run``:那一步不辨識,幾秒就知道兩份檔案對不對得上、
照這個節奏會醒幾輪。

``--sequence``:答案改用單調對齊
--------------------------------
預設的答案是 ``stable_at``:狀態維持 ``--settle`` 秒以上才評分。那會**跳過**
出牌快的那幾輪與局尾最後一張 —— 而那裡有它量不到的錯(``docs/cv_mjai.md``
§3.4)。牌譜素材則根本沒有時間,只能用這個::

    python tools/replay_table.py <session> <dump> --sequence data/reads.pkl

第一次會先把整場每一張影像讀一遍(多行程,幾分鐘)存到那個檔案,之後重用。
檔案裡帶著辨識設定的指紋,角點或門檻改過就自動重讀。

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
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.config.loader import load_config
from mia.eval.align import SETTLE
from mia.eval.material import parse_table_rect, require_one_calibration
from mia.eval.replay import replay, schedule
from mia.eval.rivers import RiverFrameResult, build_river_timeline, read_rivers
from mia.eval.sequence import align_session, fingerprint
from mia.recorder import SessionReader
from mia.utils.geometry import Rect
from mia.utils.logging import setup_logging
from mia.vision.tiles.classify import DEFAULT_SKIN, TemplateSet

_READER: tuple[SessionReader, TemplateSet, Rect, dict[int, object]] | None = None


def _reader_setup(session_dir: Path, table: Rect, skin: str) -> None:
    global _READER
    import cv2

    cv2.setNumThreads(1)
    session = SessionReader(session_dir)
    _READER = (
        session,
        TemplateSet.load(skin),
        table,
        {r.index: r for r in session.manifest.frames},
    )


def _reader_work(indices: list[int]) -> list[tuple[int, list[RiverFrameResult]]]:
    assert _READER is not None, "工人沒有經過 _reader_setup"
    session, templates, table, by_index = _READER
    return [
        (i, read_rivers(session.load_image(by_index[i]), i, templates, table))  # type: ignore[arg-type]
        for i in indices
    ]


def _reads(
    args: argparse.Namespace, session: SessionReader, table: Rect, skin: str
) -> dict[int, list[RiverFrameResult]]:
    """整場每一張寫過檔的影像讀一次。存檔重用,但**指紋對不上就重讀**。"""
    path: Path = args.sequence
    want = fingerprint(table, skin)
    if path.exists():
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved.get("fingerprint") == want and saved.get("session") == str(args.session):
            print(f"讀取結果:沿用 {path}(指紋 {want})")
            return saved["reads"]
        print(f"讀取結果:{path} 是別的設定或別的錄影讀的,重讀")

    from concurrent.futures import ProcessPoolExecutor

    changed = [r.index for r in session.manifest.frames if r.changed]
    chunks = [changed[i : i + 24] for i in range(0, len(changed), 24)]
    reads: dict[int, list[RiverFrameResult]] = {}
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=os.process_cpu_count() or 1,
        initializer=_reader_setup,
        initargs=(args.session, table, skin),
    ) as pool:
        for block in pool.map(_reader_work, chunks):
            reads.update(block)
            print(
                f"\r  讀取 {len(reads)}/{len(changed)} 張  {time.monotonic() - started:.0f}s ",
                end="",
                flush=True,
            )
    print()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"fingerprint": want, "session": str(args.session), "reads": reads}, handle)
    return reads


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
    parser.add_argument(
        "--sequence",
        type=Path,
        default=None,
        metavar="檔案",
        help="答案改用單調對齊(牌譜素材必須)。整場讀取結果存在這個檔案重用",
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
            plan = list(schedule(session, interval=value))
            starved = sum(1 for t in plan if t.record is None)
            print(f"  {value:.1f}s  {len(plan):5d} 輪,沒畫面 {starved}")
        return 0

    ticks = len(list(schedule(session, interval=args.interval)))
    if args.limit:
        ticks = min(ticks, args.limit)
    if not ticks:
        print("\n一輪都排不出來 —— 這份錄影是空的?")
        return 1
    skin = args.skin or DEFAULT_SKIN
    answers = None
    if args.sequence is not None:
        answers = align_session(session, timeline, _reads(args, session, table, skin))
        print(f"單調對齊:{len(answers)} / {len(session.manifest.frames)} 幀有答案")

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
        skin=skin,
        settle=settle,
        limit=args.limit,
        on_tick=tick,
        answers=answers,
    )
    print("\n")
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
