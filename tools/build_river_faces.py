#!/usr/bin/env python
"""從 ``tools/selfplay.py`` 的錄影做牌河用的第二份模板。

原本的模板是遊戲素材的牌面,牌河那邊看到的是攤平後的 3D 牌:比較糊,而且有斜面與光。同一張牌
拿真的畫面平均起來,比素材更像河裡看到的樣子。答案來自封包,所以不用人工標。

只當第二份用,不取代素材:比對時每種牌取分數高的那份,見 :meth:`TemplateSet.load_river`。這批
都是網頁版的畫面,Steam 的算圖不一定一樣,有素材墊著就不會比原本差。

留一場交叉驗證(五場 2357 格):有信心的格子 84.6% → 95.2%,其中讀錯 6 → 4 張,全部格子正確率
97.8% → 98.3%。

用法::

    python tools/build_river_faces.py data/recordings/selfplay-*
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from mia.eval.streamdiff import load_truth
from mia.mjai.events import Chi, Dahai, Daiminkan, Pon, Reach
from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.recorder import SessionReader
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import DEFAULT_SKIN, TemplateSet, river_faces_dir
from mia.vision.tiles.river import (
    DEFAULT_CORNERS,
    SEATS,
    face_size,
    read_river,
    search_window,
)

#: 打出來之後幾秒的畫面。動畫演完,又不會被和了的畫面蓋住。
_SETTLED = 2.0

#: 存檔的尺寸。之後都會縮放,大一點留細節。
_FACE = (66, 84)

#: 對齊分數低於這個的格子不算:多半是格線對錯了,不是那張牌。
_ALIGNED = 0.5

#: 一種牌至少要有幾張才做模板,太少就是雜訊。
_ENOUGH = 8


def labels(session: Path) -> list[tuple[int, int, str, str]]:
    """``(第幾家, 第幾格, 牌, 畫面檔)``。鳴走的不算,立直那一列起格線會挪,也不算。"""
    reader = SessionReader(session)
    frames = reader.manifest.frames
    stamps = [frame.timestamp for frame in frames]
    first = reader.manifest.first_frame_wall or 0.0
    out = []
    for kyoku in load_truth(session / "ws.jsonl"):
        rivers: dict[int, list[tuple[str, float]]] = {seat: [] for seat in range(4)}
        declared: dict[int, int] = {}
        pending = None
        for wall, event in kyoku.events:
            if isinstance(event, Reach):
                pending = event.actor
            elif isinstance(event, Dahai):
                rivers[event.actor].append((event.pai, wall))
                if pending == event.actor:
                    declared[event.actor] = len(rivers[event.actor]) - 1
                    pending = None
            elif isinstance(event, Chi | Pon | Daiminkan) and rivers[event.target]:
                rivers[event.target].pop()
        for seat, river in rivers.items():
            usable = declared[seat] // 6 * 6 if seat in declared else len(river)
            for cell, (pai, wall) in enumerate(river[:usable]):
                at = bisect.bisect_left(stamps, wall + _SETTLED - first)
                image = frames[min(at, len(frames) - 1)].image
                out.append((seat, cell, normalize_red(pai), image))
    return out


def _crop(job: tuple[Path, list[int], int, int, str, str]) -> tuple[str, np.ndarray] | None:
    """切出一格,用對的那張模板對齊,回傳牌面。"""
    session, rect, seat, cell, pai, name = job
    reader = _reader(session)
    table = Rect(*rect)
    record = _FRAMES.setdefault(session, {f.image: f for f in reader.manifest.frames})[name]
    view = DEFAULT_CORNERS[SEATS[seat]]
    flat = view.rectify(reader.load_image(record), table)
    river = read_river(flat)
    if not river.is_plausible() or cell >= len(river.cells) or not river.occupied[cell]:
        return None
    window = search_window(flat, river.cells[cell], left=view.left_slack)
    size = face_size(seat=SEATS[seat])
    faces = _scaled(size)
    if pai not in faces:
        return None
    _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(window, faces[pai], cv2.TM_CCOEFF_NORMED))
    if score < _ALIGNED:
        return None
    face = window[at[1] : at[1] + size[1], at[0] : at[0] + size[0]]
    return pai, cv2.resize(face, _FACE, interpolation=cv2.INTER_AREA)


_READERS: dict[Path, SessionReader] = {}
_FRAMES: dict[Path, dict[str, Any]] = {}
_TEMPLATES: dict[tuple[int, int], dict[str, np.ndarray]] = {}


def _reader(session: Path) -> SessionReader:
    if session not in _READERS:
        _READERS[session] = SessionReader(session)
    return _READERS[session]


def _scaled(size: tuple[int, int]) -> dict[str, np.ndarray]:
    """縮好的素材模板,鍵是 MJAI 記法,赤五併進普通五。"""
    if size not in _TEMPLATES:
        pairs = TemplateSet.load().scaled(size)
        _TEMPLATES[size] = {
            ms_to_mjai(label): face for label, face in pairs if not label.startswith("0")
        }
    return _TEMPLATES[size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--skin", default=DEFAULT_SKIN)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    jobs = []
    for session in args.sessions:
        rect = SessionReader(session).manifest.table_rect
        jobs += [(session, list(rect), *label) for label in labels(session)]
    print(f"{len(args.sessions)} 場,{len(jobs)} 格", flush=True)

    pool: dict[str, list[np.ndarray]] = defaultdict(list)
    cv2.setNumThreads(1)
    with ProcessPoolExecutor(args.workers) as executor:
        for done in executor.map(_crop, jobs, chunksize=20):
            if done is not None:
                pool[done[0]].append(done[1])

    labels_seen = TemplateSet.load(args.skin).labels
    names = {ms_to_mjai(label): label for label in labels_seen if not label.startswith("0")}
    target = river_faces_dir(args.skin)
    target.mkdir(parents=True, exist_ok=True)
    for pai, faces in sorted(pool.items()):
        if len(faces) < _ENOUGH:
            print(f"{pai} 只有 {len(faces)} 張,不做")
            continue
        mean = np.mean(faces, axis=0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(target / f"{names[pai]}.png"), mean)
    print(f"{sum(map(len, pool.values()))} 格,寫到 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
