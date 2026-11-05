"""Fuse tile observations, never independently produced MJAI event streams.

The template score remains a correlation coefficient, not a YOLO probability.
`conf` is the upstream tracker's compatibility gate; raw scores and the decision
are retained alongside it whenever multi-source evidence changes that gate.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.tracker import Stream
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet, classify
from mia.vision.tiles.river import (
    CELL,
    COLUMNS,
    DEFAULT_CORNERS,
    ROWS,
    classify_river_tile,
    face_size,
)

_SEATS = ("own", "shimocha", "toimen", "kamicha")
_STRONG_SCORE = 0.75
_STRONG_MARGIN = 0.10
_YOLO_TRUST = 0.70


def annotate(payload: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    payload["fusion"] = fusion
    if fusion["pending"]:
        payload["hand"] = {**payload.get("hand", {}), "stable": False, "confident": False}
    return payload


def guard_stream(stream: Stream, payload: dict[str, Any]) -> Stream:
    if (payload.get("fusion") or {}).get("pending"):
        return replace(stream, stalled=True, doubt="混合辨識：等待手牌歧異釐清")
    return stream


@dataclass(frozen=True)
class Evidence:
    pai: str
    score: float
    margin: float
    brightness: float = 1.0
    sideways: bool = False

    @property
    def strong(self) -> bool:
        return (
            not self.sideways
            and self.score >= _STRONG_SCORE
            and self.margin >= _STRONG_MARGIN
            and self.brightness >= 0.65
        )


@dataclass
class _Track:
    center: tuple[float, float]
    wall: float
    checked: int = -1
    label: str | None = None
    votes: int = 0
    frame: int = -1
    vote_wall: float = float("-inf")


class TemplateReader:
    """Use MIA's existing classifiers and projection geometry without editing them."""

    def __init__(self, skin: str, *, capacity: int = 8192) -> None:
        self.hand = TemplateSet.load(skin)
        self.river = TemplateSet.load_river(skin)
        self.capacity = capacity
        self.cache: OrderedDict[bytes, Evidence] = OrderedDict()
        target = np.array(
            [
                [0, 0],
                [COLUMNS * CELL[0], 0],
                [COLUMNS * CELL[0], ROWS * CELL[1]],
                [0, ROWS * CELL[1]],
            ],
            dtype=np.float32,
        )
        self.matrices = [
            cv2.getPerspectiveTransform(DEFAULT_CORNERS[s].quad(Rect(0, 0, 1920, 1080)), target)
            for s in _SEATS
        ]
        self.calls = self.hits = 0

    def patch(
        self, frame: np.ndarray, detection: dict[str, Any], kind: str, seat: int
    ) -> tuple[np.ndarray | None, bool]:
        x1, y1, x2, y2 = (int(v) for v in detection["box"])
        if kind == "hand":
            patch = frame[max(0, y1) : min(1080, y2), max(0, x1) : min(1920, x2)]
            # Same width cap as MIA's hand classifier; preserve aspect ratio.
            if patch.size and patch.shape[1] > 80:
                height = max(1, round(patch.shape[0] * 80 / patch.shape[1]))
                patch = cv2.resize(patch, (80, height), interpolation=cv2.INTER_AREA)
            return patch, False
        matrix = self.matrices[seat]
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], detection["center"]]], dtype=np.float32
        )
        points = cv2.perspectiveTransform(corners, matrix)[0]
        span = np.ptp(points[:4], axis=0)
        sideways = bool(span[0] > span[1] * 1.2)
        # Existing sideways-template matches are deliberately untrusted in MIA.
        # Leave these to YOLO/temporal evidence rather than undoing that policy.
        if sideways:
            return None, True
        cx, cy = points[4]
        width, height = 110, 175
        move = np.array(
            [[1, 0, width / 2 - cx], [0, 1, height / 2 - cy], [0, 0, 1]], dtype=np.float64
        )
        return cv2.warpPerspective(frame, move @ matrix, (width, height)), False

    def read(
        self,
        frame: np.ndarray,
        detection: dict[str, Any],
        kind: str,
        seat: int,
        *,
        allow_new: bool = True,
    ) -> Evidence | None:
        patch, sideways = self.patch(frame, detection, kind, seat)
        if patch is None or patch.size == 0 or sideways:
            return None
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{kind}:{seat}:{patch.shape}".encode())
        digest.update(patch.tobytes())
        key = digest.digest()
        known = self.cache.get(key)
        if known is not None:
            self.cache.move_to_end(key)
            self.hits += 1
            return known
        if not allow_new:
            return None
        try:
            match = (
                classify(patch, self.hand)
                if kind == "hand"
                else classify_river_tile(patch, self.river, size=face_size())
            )
        except ValueError:
            return None
        evidence = Evidence(ms_to_mjai(match.label), match.score, match.margin, match.brightness)
        self.calls += 1
        self.cache[key] = evidence
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return evidence


class Fusion:
    def __init__(
        self, reader: TemplateReader, *, max_new: int = 24, budget_ms: float = 50.0
    ) -> None:
        self.reader = reader
        self.max_new, self.budget_ms = max_new, budget_ms
        self.tracks: dict[tuple[str, int], list[_Track]] = {}
        self.totals: Counter[str] = Counter()

    def reset(self) -> None:
        self.tracks.clear()

    def _track(
        self, group: tuple[str, int], det: dict[str, Any], wall: float, used: set[int]
    ) -> _Track:
        center = (float(det["center"][0]), float(det["center"][1]))
        candidates = self.tracks.setdefault(group, [])
        candidates[:] = [t for t in candidates if 0 <= wall - t.wall <= 1500]
        radius = max(8, min(28, (det["box"][2] - det["box"][0]) * 0.45))
        nearby = [
            t
            for t in candidates
            if id(t) not in used
            and np.hypot(center[0] - t.center[0], center[1] - t.center[1]) <= radius
        ]
        track = (
            min(nearby, key=lambda t: np.hypot(center[0] - t.center[0], center[1] - t.center[1]))
            if nearby
            else _Track(center, wall)
        )
        if not nearby:
            candidates.append(track)
        track.center, track.wall = center, wall
        used.add(id(track))
        return track

    def apply(self, frame: np.ndarray, observation: dict[str, Any]) -> dict[str, Any]:
        obs = deepcopy(observation)
        stats: Counter[str] = Counter()
        pending = []
        jobs = []
        index, wall = int(obs["frame_index"]), float(obs["t_ms"])
        ph = obs.get("p0_hand")
        groups = [("hand", 0, ph["tiles"])] if ph is not None else []
        for kind in ("river", "meld"):
            groups += [
                (kind, i, part.get("tiles") or [])
                for i in range(4)
                if (part := (obs.get(kind) or {}).get(f"P{i}")) is not None
            ]
        for kind, seat, tiles in groups:
            used: set[int] = set()
            for det in tiles:
                track = self._track((kind, seat), det, wall, used)
                priority = (float(det["conf"]) >= _YOLO_TRUST, kind != "hand", track.checked)
                jobs.append((priority, kind, seat, det, track))
        jobs.sort(key=lambda item: item[0])
        started = time.perf_counter()
        before_calls, before_hits = self.reader.calls, self.reader.hits
        for _, kind, seat, det, track in jobs:
            allow = (
                self.reader.calls - before_calls < self.max_new
                and (time.perf_counter() - started) * 1000 < self.budget_ms
            )
            ev = self.reader.read(frame, det, kind, seat, allow_new=allow)
            if ev is None:
                stats["template_abstain"] += 1
                continue
            track.checked = index
            original, conf = det["pai"], float(det["conf"])
            det["template"] = dict(
                pai=ev.pai,
                score=round(ev.score, 5),
                margin=round(ev.margin, 5),
                brightness=round(ev.brightness, 5),
            )
            if not ev.strong:
                track.label, track.votes = None, 0
                stats["template_weak"] += 1
                continue
            if not 0 <= wall - track.vote_wall <= 1500:
                track.label, track.votes = None, 0
            if track.frame == index and track.label != ev.pai:
                track.label, track.votes = ev.pai, 0
            if track.frame != index:
                track.votes = track.votes + 1 if track.label == ev.pai else 1
                track.label, track.frame = ev.pai, index
                track.vote_wall = wall
            state = "agree" if original == ev.pai else "disagree"
            stats[state] += 1
            det["fusion"] = state
            # Red/plain differences require separate color evidence; never fold
            # them into a base-label correction by the river's merge-red margin.
            red_dispute = original != ev.pai and normalize_red(original) == normalize_red(ev.pai)
            if track.votes < 2 or red_dispute:
                continue
            if conf < _YOLO_TRUST:
                det["raw_yolo"] = dict(pai=original, conf=conf)
                det["pai"], det["conf"] = ev.pai, _YOLO_TRUST
                det["fusion"] = "template_confirmed" if original != ev.pai else "joint_confirmed"
                stats["corrected" if original != ev.pai else "joint_confirmed"] += 1
            elif original != ev.pai and kind == "hand":
                det["raw_yolo"] = dict(pai=original, conf=conf)
                det["conf"] = min(conf, 0.69)
                det["fusion"] = "pending_conflict"
                pending.append(
                    dict(kind=kind, seat=seat, center=det["center"], yolo=original, template=ev.pai)
                )
        stats["template_calls"] = self.reader.calls - before_calls
        stats["template_cache_hits"] = self.reader.hits - before_hits
        # Do not apply YOLO-only learned hand fixes on top of changed evidence.
        if ph is not None:
            for key in ("hand_fix", "drawn_lib"):
                fix = ph.get(key)
                if fix and any(
                    d.get("fusion") in ("template_confirmed", "pending_conflict")
                    for d in ph["tiles"]
                ):
                    ph[key] = None
        self.totals.update(stats)
        obs["fusion"] = dict(
            counts=dict(stats), pending=pending, elapsed_ms=(time.perf_counter() - started) * 1000
        )
        return obs
