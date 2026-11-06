"""YOLO 代理:P0 打的牌,牌河跟手牌差分不一致時聽比較有把握的那一邊(live_v7/fsm.py)。

十段錄影兩邊不一致 4 次:手牌 0.97 > 牌河 0.88 那次手牌對;手牌 0.81 < 0.96、
0.62 < 0.92 那兩次牌河對。判成摸切時「差分」那張只是摸入牌的牌名,不算手牌證據。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "engines" / "yolo" / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "live_v7"), str(AGENT / "live_v7" / "ui")]
pipeline = pytest.importorskip("pipeline")
turn_signals = pytest.importorskip("turn_signals")
river_adapter = pytest.importorskip("river_adapter")


def _discard():
    fsm = pipeline.Pipeline("t").fsm
    ev = fsm._ev("dahai", 1000.0, 10, actor="P0", pai=None, tsumogiri=None,  # noqa: SLF001
                 slot_id="S1")
    fsm.slot_event["S1"] = ev["event_id"]
    return fsm, ev


def _river(fsm, pai, conf):
    fsm.on_confirmed(river_adapter.RiverConfirmed("P0", "S1", 1, 20, pai, 8, False, False, conf),
                     1500.0, 20)


def _cut(fsm, pai, conf, *, tsumogiri=False, why="13張 -> 13張"):
    v = turn_signals.CutVerdict("P0", "S1", 30, tsumogiri, why, "p0_hand_diff", False,
                                pai_guess=pai, pai_conf=conf)
    fsm.on_cut(v, 2000.0, 30)


@pytest.mark.parametrize("river_first", [True, False])
def test_a_surer_river_beats_a_weaker_hand_diff(river_first):
    fsm, ev = _discard()
    steps = [lambda: _river(fsm, "1s", 0.92), lambda: _cut(fsm, "5m", 0.62)]
    for step in steps if river_first else reversed(steps):
        step()
    assert fsm.ledger.get(ev["event_id"])["pai"] == "1s"


@pytest.mark.parametrize("river_first", [True, False])
def test_a_surer_hand_diff_still_beats_the_river(river_first):
    fsm, ev = _discard()
    steps = [lambda: _river(fsm, "6p", 0.88), lambda: _cut(fsm, "4p", 0.97)]
    for step in steps if river_first else reversed(steps):
        step()
    assert fsm.ledger.get(ev["event_id"])["pai"] == "4p"


def test_a_tsumogiri_guess_is_the_drawn_name_not_hand_evidence():
    fsm, ev = _discard()
    _river(fsm, "E", 0.95)
    _cut(fsm, "N", None, tsumogiri=True, why="11張沒變、內容也一樣")
    assert fsm.ledger.get(ev["event_id"])["pai"] == "E"
