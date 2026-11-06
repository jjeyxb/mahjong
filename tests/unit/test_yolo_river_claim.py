"""YOLO 代理的牌河:被鳴走的牌多留幾幀、還沒進牌桌時的假牌河。

錄影 selfplay-20261106-110134 東1局 f1464:自己吃上家的 8s,8s 在牌河上留到 f1469。
封存之後同位置同一張又被看成新的一格 -> 上家又打一張 -> 狀態機再「推定」一次吃。
同一批錄影開局前的載入畫面,四家牌河都讀到假的白(live_v7/pipeline.py)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "engines" / "yolo" / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "live_v7"), str(AGENT / "live_v7" / "ui")]
river_adapter = pytest.importorskip("river_adapter")
pipeline = pytest.importorskip("pipeline")

SEAT = "P3"
SPOTS = [(734, 345), (732, 383), (726, 431)]  # 上家牌河往下長


def _tile(pai, center):
    x, y = center
    return {"pai": pai, "conf": 0.95, "box": (x - 39, y - 21, x + 39, y + 21), "center": center}


def _feed(adapter, frames, tiles):
    out = []
    for f in frames:
        out += adapter.feed(f, f * 66.0, {SEAT: {"tiles": [_tile(p, c) for p, c in tiles]}})
    return [e for e in out if e.kind == "candidate"]


def test_a_claimed_tile_still_on_screen_is_not_a_new_discard():
    a = river_adapter.RiverAdapter("H001")
    _feed(a, range(1, 20), [("3p", SPOTS[0]), ("5p", SPOTS[1])])
    river = [("3p", SPOTS[0]), ("5p", SPOTS[1]), ("8s", SPOTS[2])]
    cands = _feed(a, range(20, 40), river)
    assert len(cands) == 1
    eight = cands[0]
    assert a.claim(SEAT, eight.slot_id, eight.generation, 40).ok
    assert _feed(a, range(40, 46), river) == []  # 吃完 8s 還留在牌河上
    _feed(a, range(46, 100), river[:2])
    # 上家下一張打到同一格:還是一次打牌
    assert len(_feed(a, range(100, 110), [*river[:2], ("9s", SPOTS[2])])) == 1


def test_the_next_discard_soon_after_on_the_claimed_spot_is_still_a_discard():
    """selfplay-20261102-155238 東4局4本場:下家碰白,上家 20 幀後打 3s 到同一格。"""
    a = river_adapter.RiverAdapter("H001")
    _feed(a, range(1, 20), [("3p", SPOTS[0]), ("5p", SPOTS[1])])
    river = [("3p", SPOTS[0]), ("5p", SPOTS[1]), ("P", SPOTS[2])]
    white = _feed(a, range(20, 40), river)[0]
    assert a.claim(SEAT, white.slot_id, white.generation, 40).ok
    _feed(a, range(40, 44), river)  # 白還在
    _feed(a, range(44, 56), river[:2])
    assert len(_feed(a, range(56, 66), [*river[:2], ("3s", SPOTS[2])])) == 1


def _obs(frame, wall, river=None, hand=0):
    tiles = [{"pai": "1m", "conf": 0.9, "box": (220 + 94 * i, 930, 310 + 94 * i, 1050),
              "center": (265 + 94 * i, 990)} for i in range(hand)]
    return {"frame_index": frame, "t_ms": frame * 66.0, "region_detections": [],
            "river": {s: {"tiles": [_tile(p, c) for p, c in (river or {}).get(s, [])]}
                      for s in ("P0", "P1", "P2", "P3")},
            "p0_hand": {"tiles": tiles} if hand else None, "meld": None,
            "dora_observation": None, "ui_context": None, "hud": {"wall": wall, "turn": None}}


def _discards(p):
    return [e for e in p.ledger.events if e["type"] == "dahai"]


def test_river_tiles_before_the_table_ever_appears_are_ignored():
    p = pipeline.Pipeline("t")
    for f in range(1, 30):  # 載入畫面:沒有余牌、沒有手牌,牌河讀到白
        p.feed(_obs(f, None, {"P1": [("P", (1188, 466))]}))
    # 整幀都不看(selfplay-20261102-170335:載入畫面一個假的摸牌就讓莊家第一打被丟掉)
    assert p.ledger.events == []


def test_river_tiles_count_once_a_hand_was_seen_even_without_a_wall_reading():
    p = pipeline.Pipeline("t")
    for f in range(1, 10):
        p.feed(_obs(f, None, hand=13))
    for f in range(10, 30):
        p.feed(_obs(f, None, {"P1": [("S", (1188, 466))]}, hand=13))
    assert _discards(p)
