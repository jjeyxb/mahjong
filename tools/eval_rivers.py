#!/usr/bin/env python
"""量四家牌河的 CV 準確率:拿同時錄下的封包當標準答案。

與 ``tools/evaluate.py`` 吃**同一組素材**,差別只在量的是牌河而不是手牌::

    python tools/eval_rivers.py data/recordings/20260801-110639 data/gt/ws.jsonl

先跑 ``--dry-run`` 確認兩份檔案真的對得上 —— 那一步不跑 CV,幾秒就有結果。

為什麼牌河要單獨量
------------------
手牌認錯了使用者一眼就發現(他正在看自己的手牌),牌河沒人會去核對 ——
放銃分析憑它算危險度,算錯了沒有任何症狀。所以這裡最該看的不是總體準確率,
是**靜默錯誤**那一欄:有信心卻認錯的比例。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.eval.align import SETTLE
from mia.eval.rivers import (
    RiverFrameResult,
    align_rivers,
    build_river_timeline,
    evaluate_rivers,
    summarize_rivers,
)
from mia.recorder import SessionReader
from mia.utils.geometry import Rect
from mia.utils.logging import setup_logging
from mia.vision.tiles.classify import TemplateSet

#: 每個工人一次拿幾幀。四家一起跑,所以比手牌那邊小一點。
CHUNK = 32

_STATE: tuple[SessionReader, TemplateSet, Rect, dict[int, object]] | None = None


def _setup(session_dir: Path, table: Rect, skin: str | None) -> None:
    global _STATE
    import cv2

    # 不設回 1 的話 N 個行程各開十幾條執行緒互搶核心,比單行程還慢。
    cv2.setNumThreads(1)
    session = SessionReader(session_dir)
    _STATE = (
        session,
        TemplateSet.load(skin) if skin else TemplateSet.load(),
        table,
        {r.index: r for r in session.manifest.frames},
    )


def _work(chunk: list) -> list[RiverFrameResult]:
    assert _STATE is not None, "工人沒有經過 _setup"
    session, templates, table, by_index = _STATE
    out: list[RiverFrameResult] = []
    for frame in chunk:
        image = session.load_image(by_index[frame.index])  # type: ignore[arg-type]
        out.extend(evaluate_rivers(image, frame, templates, table))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path, help="tools/record.py 錄的 session 目錄")
    parser.add_argument("dump", type=Path, help="tools/gt.py 錄的 WebSocket .jsonl")
    parser.add_argument("--skin", default=None, help="牌面模板皮膚,預設用內建的")
    parser.add_argument("--settle", type=float, default=None, help="動畫穩定秒數")
    parser.add_argument("--limit", type=int, default=0, help="最多評幾幀,0 表示全部")
    parser.add_argument("--jobs", type=int, default=None, help="幾個工人行程")
    parser.add_argument("--dry-run", action="store_true", help="只檢查對得上對不上")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level)

    session = SessionReader(args.session)
    if session.table_rect is None:
        raise SystemExit("session 沒有 table_rect —— 錄製時校正沒鎖定")
    print(f"畫面 {args.session}:{len(session.manifest.frames)} 幀")
    timeline = build_river_timeline(args.dump)
    print(f"封包 {args.dump}:{len(timeline)} 個牌河狀態,自己坐 {timeline.seat}")

    settle = args.settle if args.settle is not None else SETTLE
    if args.dry_run:
        print("\n穩定秒數 → 可評分的幀數")
        for value in (0.4, 0.6, SETTLE, 1.3, 2.0):
            print(f"  {value:.1f}s  {len(list(align_rivers(session, timeline, settle=value)))} 幀")
        return 0

    paired = list(align_rivers(session, timeline, settle=settle))
    if args.limit:
        paired = paired[: args.limit]
    if not paired:
        print("\n配不到任何幀。用 --dry-run 看看是不是穩定秒數設得太嚴。")
        return 1

    jobs = args.jobs if args.jobs is not None else (os.process_cpu_count() or 1)
    print(f"\n穩定秒數 {settle:.1f}s,配到 {len(paired)} 幀,{jobs} 個工人,開始辨識…")

    chunks = [paired[i : i + CHUNK] for i in range(0, len(paired), CHUNK)]
    results: list[RiverFrameResult] = []
    started = time.monotonic()

    def tick(done: int) -> None:
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0.0
        left = (len(paired) - done) / rate if rate else 0.0
        print(
            f"\r  {done}/{len(paired)} 幀  {rate:.1f} 幀/秒  剩約 {left / 60:.1f} 分 ",
            end="",
            flush=True,
        )

    if jobs <= 1:
        _setup(args.session, session.table_rect, args.skin)
        for done, chunk in enumerate(chunks, 1):
            results.extend(_work(chunk))
            tick(min(done * CHUNK, len(paired)))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_setup,
            initargs=(args.session, session.table_rect, args.skin),
        ) as pool:
            for done, block in enumerate(pool.map(_work, chunks), 1):
                results.extend(block)
                tick(min(done * CHUNK, len(paired)))
    print("\n")
    print(summarize_rivers(results).summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
