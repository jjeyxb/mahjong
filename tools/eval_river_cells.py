#!/usr/bin/env python
"""每張捨牌停穩之後,照即時路徑認那一格,對封包的答案。

``tools/selfplay.py replay`` 量的是整條事件流,照真實時間跑,同一局重播兩次結果不一樣,模板尺寸、
搜尋範圍這類設定在上面比不出小的差別。這支只問一件事:**牌停穩之後,那一格讀不讀得對**。
每張捨牌只看一次(打出來 2 秒後的畫面),不經過牌桌的排序與時機,同一批錄影每次跑的數字都相同。

認法與 ``TableWorker._read_cell`` 相同:各家的模板尺寸、左右放寬、第三列往上多看,先比素材,
沒把握再比牌河的第二份模板。改那邊的認法時這裡要跟著改。

被鳴走的牌不算;立直那一列起格線會挪,也不算(答案取自 ``tools/build_river_faces.py`` 的
``labels``)。所以這裡量不到宣言牌、動畫中途、漏牌與多牌,那些要看重播。

用法::

    python tools/eval_river_cells.py data/recordings/selfplay-* data/recordings/human-*
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cv2

from build_river_faces import labels
from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.recorder import SessionReader
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import MIN_BRIGHTNESS, Match, TemplateSet
from mia.vision.tiles.river import (
    CONFIDENT,
    DEFAULT_CORNERS,
    ROWS,
    SEATS,
    classify_river_tile,
    face_size,
    read_river,
    search_window,
    slot,
)

#: 一張捨牌的結果:第幾家、第幾列、讀對了嗎、有把握嗎。切不出來的是 ``None``。
_Read = tuple[int, int, bool, bool]

_STATE: dict[str, Any] = {}


def _setup(session: Path) -> None:
    cv2.setNumThreads(1)
    reader = SessionReader(session)
    _STATE["reader"] = reader
    _STATE["frames"] = {frame.image: frame for frame in reader.manifest.frames}
    _STATE["base"] = TemplateSet.load()
    _STATE["river"] = TemplateSet.load_river()


def _sure(match: Match) -> bool:
    return (
        match.score >= CONFIDENT[0]
        and match.margin >= CONFIDENT[1]
        and match.brightness >= MIN_BRIGHTNESS
    )


def _work(chunk: list[tuple[list[int], int, int, str, str]]) -> list[_Read | None]:
    reader, frames = _STATE["reader"], _STATE["frames"]
    base, river_templates = _STATE["base"], _STATE["river"]
    out: list[_Read | None] = []
    loaded: tuple[str, Any] | None = None
    for rect, seat, cell, pai, name in chunk:
        if loaded is None or loaded[0] != name:
            loaded = (name, reader.load_image(frames[name]))
        view = DEFAULT_CORNERS[SEATS[seat]]
        flat = view.rectify(loaded[1], Rect(*rect))
        river = read_river(flat)
        if not river.is_plausible() or cell >= len(river.cells) or not river.occupied[cell]:
            out.append(None)
            continue
        row = slot(cell)[1]
        window = search_window(
            flat, river.cells[cell], left=view.left_slack, right=view.right_slack, row=row
        )
        size = face_size(seat=SEATS[seat])
        match = classify_river_tile(window, base, size=size)
        if not _sure(match):
            match = classify_river_tile(window, river_templates, size=size)
        right = normalize_red(ms_to_mjai(match.label)) == pai
        out.append((seat, row, right, _sure(match)))
    return out


def evaluate(session: Path, workers: int) -> list[_Read | None]:
    """一場錄影的每張捨牌。"""
    rect = SessionReader(session).manifest.table_rect
    if rect is None:
        raise ValueError(f"{session} 沒有存牌桌矩形")
    jobs = sorted(((list(rect), *label) for label in labels(session)), key=lambda job: job[4])
    chunks = [jobs[i : i + 40] for i in range(0, len(jobs), 40)]
    reads: list[_Read | None] = []
    with ProcessPoolExecutor(workers, initializer=_setup, initargs=(session,)) as pool:
        for part in pool.map(_work, chunks):
            reads += part
    return reads


def summary(reads: list[_Read]) -> str:
    lines = ["家        列    張數  有把握且對  有把握卻錯  讀錯"]
    total: Counter[str] = Counter()
    for seat in range(len(SEATS)):
        for row in range(ROWS):
            mine = [(right, sure) for s, r, right, sure in reads if s == seat and r == row]
            counts = {
                "n": len(mine),
                "sure": sum(right and sure for right, sure in mine),
                "silent": sum(sure and not right for right, sure in mine),
                "wrong": sum(not right for right, _ in mine),
            }
            total.update(counts)
            lines.append(
                f"{SEATS[seat]:9s} {row}  {counts['n']:6d}  {counts['sure']:10d}  "
                f"{counts['silent']:10d}  {counts['wrong']:4d}"
            )
    lines.append(
        f"合計         {total['n']:6d}  {total['sure']:10d}  {total['silent']:10d}  "
        f"{total['wrong']:4d}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    reads: list[_Read] = []
    for session in args.sessions:
        found = evaluate(session, args.workers)
        kept = [read for read in found if read is not None]
        wrong = sum(not right for _, _, right, _ in kept)
        print(f"{session.name}: 捨牌 {len(found)},切得出來 {len(kept)},讀錯 {wrong}", flush=True)
        reads += kept
    print()
    print(summary(reads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
