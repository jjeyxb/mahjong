"""YOLO 代理:換局時要有確實的牌局證據,才記住上一局的局數。

情境照錄影 selfplay-20261109-182732:開局前的畫面讀到兩張假捨牌(白、9m)而且定案了。
真的東1局開始後,場況在頭幾秒讀齊「東1/0」,莊家第一打讓牌河從 2 張變 1 張而換局。
那一局如果被記成「上一局 = 東1/0」,真的東1局就會被當成舊面板一直擋掉,之後每局晚一局。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
    _complete_start(p, t_ms)
    return p


def _complete_start(p: pipeline.Pipeline, t_ms: float) -> None:
    ev = p.ledger.update(p.start_eid, t_ms=t_ms, reason="場況", **ROUND_INFO)
    assert ev["export_eligible"]
    assert ev["t_confirm_ms"] == t_ms


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
    p.fsm.on_declaration(SimpleNamespace(cls="ryukyoku", seat=None), 90_000.0, 900)
    assert _remembered(p, monkeypatch, 110_000.0) is False


def _read_east_one_again(p: pipeline.Pipeline) -> None:
    for _ in range(6):
        p.kyoku.feed({"panel": {"kyoku": ("E1", 0.99), "honba": (0, 0.99)}}, None)


@pytest.mark.parametrize("declaration", ["tsumo", "ryukyoku"])
def test_a_seen_end_preserves_stale_panel_filter_when_start_completes_late(declaration):
    p = pipeline.Pipeline("t")
    p._ensure_start_event(0.0, 1)  # noqa: SLF001
    p.kyoku.value.update(kyoku_label="E1", honba=0)
    _discard(p, 85_200.0, "P1", "P")
    p.fsm.on_declaration(SimpleNamespace(cls=declaration, seat="P2"), 88_500.0, 885)
    _complete_start(p, 90_000.0)
    p._new_hand(100_000.0, 1000)  # noqa: SLF001

    assert p.kyoku.previous == ("E1", 0)
    _read_east_one_again(p)
    assert p.kyoku.value["kyoku_label"] is None


def test_an_inferred_drawn_game_cannot_turn_phantom_discards_into_round_evidence():
    p = _hand_completed_at(40_000.0)
    p.kyoku.value.update(kyoku_label="E1", honba=0)
    _discard(p, 1_500.0, "P1", "P")
    _discard(p, 2_500.0, "P2", "9m")
    p.fsm.on_draw_seen(SimpleNamespace(seat="P0"), 41_000.0, 410)
    p._new_hand(44_500.0, 445)  # noqa: SLF001

    previous = p.ledger.ordered("H001")
    assert any(
        e["type"] == "ryukyoku" and e["status"] == "INFERRED" and e["t_first_ms"] > 40_000
        for e in previous
    )
    assert p.kyoku.previous is None
    _read_east_one_again(p)
    assert p.kyoku.value["kyoku_label"] == "E1"
    assert p.kyoku.value["honba"] == 0
