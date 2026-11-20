"""YOLO 代理:換局時只有「場況確定之後」真的打過的局,才記住它的局數(live_v7/pipeline.py _new_hand)。

情境照錄影 selfplay-20261109-182732:開局前的畫面讀到兩張假捨牌(白、9m)而且定案了。
真的東1局開始後,場況在頭幾秒讀齊「東1/0」,莊家第一打讓牌河從 2 張變 1 張而換局。
那一局如果被記成「上一局 = 東1/0」,真的東1局就會被當成舊面板一直擋掉,之後每局晚一局。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "engines" / "yolo" / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "live_v7"), str(AGENT / "live_v7" / "ui")]
pipeline = pytest.importorskip("pipeline")

ROUND_INFO = dict(bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0, scores=[25000] * 4,
                  dora_marker="1m", tehais=[["?"] * 13] * 4)


def _hand_completed_at(t_ms: float) -> pipeline.Pipeline:
    """一局:start_kyoku 一開始是待定,``t_ms`` 時場況讀齊。"""
    p = pipeline.Pipeline("t")
    p._ensure_start_event(0.0, 1)  # noqa: SLF001
    ev = p.ledger.update(p.start_eid, t_ms=t_ms, reason="場況", **ROUND_INFO)
    assert ev["export_eligible"]
    assert ev["t_confirm_ms"] == t_ms
    return p


def _discard(p: pipeline.Pipeline, t_ms: float, actor: str, pai: str) -> None:
    ev = p.ledger.add("dahai", p.hand_id, t_ms, int(t_ms // 100), "CONFIRMED",
                      actor=actor, pai=pai, tsumogiri=False)
    assert ev["export_eligible"]


def _remembered(p: pipeline.Pipeline, monkeypatch: pytest.MonkeyPatch, t_ms: float) -> bool:
    seen = []
    monkeypatch.setattr(p.kyoku, "reset", lambda remember=True: seen.append(remember))
    p._new_hand(t_ms, int(t_ms // 100))  # noqa: SLF001
    return seen[-1]


def test_confirmed_discards_read_before_the_round_info_do_not_count(monkeypatch):
    p = _hand_completed_at(40_000.0)
    _discard(p, 1_000.0, "P1", "P")  # 開局前畫面的假牌
    _discard(p, 2_000.0, "P2", "9m")
    assert _remembered(p, monkeypatch, 44_500.0) is False


def test_a_hand_that_played_after_its_round_info_is_remembered(monkeypatch):
    p = _hand_completed_at(3_000.0)
    _discard(p, 4_500.0, "P0", "W")
    _discard(p, 5_500.0, "P1", "5m")
    assert _remembered(p, monkeypatch, 110_000.0) is True


def test_a_hand_whose_round_info_never_completed_is_not_remembered(monkeypatch):
    p = pipeline.Pipeline("t")
    p._ensure_start_event(0.0, 1)  # noqa: SLF001
    _discard(p, 4_500.0, "P0", "W")
    assert _remembered(p, monkeypatch, 110_000.0) is False
