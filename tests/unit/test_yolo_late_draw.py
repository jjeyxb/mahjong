"""YOLO 代理:P0 漏看的摸牌,用前後的穩定暗牌補牌面(live_v7/pipeline.py _late_p0_draw)。

情境照錄影 selfplay-20261106-113550 東4局1本場:摸牌沒看到,牌河出現新棄牌才推出一筆
摸牌(INFERRED,牌面不知道)。摸牌前那份暗牌少讀一張、打牌後那份讀錯一張;更早、
更晚的暗牌是對的,要用已知的動作推過去才比得起來。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "engines" / "yolo" / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "live_v7"), str(AGENT / "live_v7" / "ui")]
pipeline = pytest.importorskip("pipeline")

A = ["1m", "2m", "3m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "C"]


def _without(tiles, *gone):
    out = list(tiles)
    for t in gone:
        out.remove(t)
    return out


B = [*_without(A, "C"), "4s"]  # 摸 4s 打 C
AFTER = [*_without(B, "3m"), "6s"]  # 摸 6s(沒看到)打 3m
D = [*_without(AFTER, "4p"), "5m"]  # 摸 5m 打 4p


def _ev(p, typ, frame, status="CONFIRMED", **kw):
    return p.ledger.add(typ, p.hand_id, frame * 10.0, frame, status, actor="P0", **kw)


def _pipeline(after_snapshots):
    p = pipeline.Pipeline("t")
    _ev(p, "tsumo", 20, pai="4s")
    _ev(p, "dahai", 25, pai="C", tsumogiri=False)
    ts = _ev(p, "tsumo", 40, "INFERRED", pai=None)
    _ev(p, "dahai", 40, pai="3m", tsumogiri=False, p0_tsumo=ts["event_id"])
    _ev(p, "tsumo", 60, pai="5m")
    _ev(p, "dahai", 65, pai="4p", tsumogiri=False)
    p.signals.p0_stable = [
        (10, 22, A),  # 讀對,但在上一次打牌之前
        (26, 35, _without(B, "9s")),  # 少讀 9s
        *after_snapshots,
    ]
    p._late_p0_draw(100.0)  # noqa: SLF001
    return p.ledger.get(ts["event_id"])


def test_a_missed_draw_is_named_from_earlier_and_later_hands_moved_to_it():
    misread = [*_without(AFTER, "2m"), "7m"]  # 2m 讀成 7m
    ts = _pipeline([(41, 55, misread), (66, 90, D)])
    assert ts["pai"] == "6s"
    assert ts["export_eligible"]


def test_a_missed_draw_stays_unnamed_when_no_hand_explains_it():
    misread = [*_without(AFTER, "2m"), "7m"]  # 2m 讀成 7m
    ts = _pipeline([(41, 55, misread)])
    assert ts["pai"] is None
