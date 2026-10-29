"""沒有時間基準的對齊。

這裡測的是對齊的**語意**:不倒退、被蓋住的那一家不算證據、讀不出來的
格子不算衝突。實際素材上的驗證(與真時間對齊的一致率)在
``docs/cv_mjai.md`` §3.4。
"""

from __future__ import annotations

import numpy as np

from mia.eval.rivers import RiverFrameResult, RiverState
from mia.eval.sequence import (
    COUNT_WEIGHT,
    Observation,
    align_monotone,
    cost_matrix,
    interior,
    observe,
)
from mia.tracker import UNKNOWN
from mia.vision.tiles.river import SEATS

SURE = (0.90, 0.30, 1.0)
UNSURE = (0.90, 0.01, 1.0)


def read(seat: str, tiles: str, *, unsure: tuple[int, ...] = (), contiguous: bool = True):
    labels = tuple(tiles.split())
    picked = [UNSURE if i in unsure else SURE for i in range(len(labels))]
    return RiverFrameResult(
        index=0,
        seat=seat,
        truth=(),
        predicted=labels,
        scores=tuple(v[0] for v in picked),
        margins=tuple(v[1] for v in picked),
        brightness=tuple(v[2] for v in picked),
        contiguous=contiguous,
    )


def state(*rivers: str) -> RiverState:
    """四家,依視角順序。``""`` 是空河。"""
    seats = tuple(tuple(r.split()) for r in rivers)
    return RiverState(0.0, seats, seats)


def seen(*rivers: str | None) -> Observation:
    return Observation(tuple(None if r is None else tuple(r.split()) for r in rivers))


class TestObserve:
    def test_an_unsure_cell_becomes_unknown(self) -> None:
        results = [read(s, "1m 2m", unsure=(1,)) for s in SEATS]
        assert observe(results).rivers[0] == ("1m", UNKNOWN)

    def test_a_broken_river_is_no_evidence_not_an_empty_one(self) -> None:
        """被立繪蓋住的河不是「零張」—— 零張是新局開始的強證據,拿錯會把
        對齊拉回局首。"""
        results = [read(s, "1m", contiguous=(s != "toimen")) for s in SEATS]
        observation = observe(results)
        assert observation.rivers[SEATS.index("toimen")] is None
        assert observation.rivers[0] == ("1m",)

    def test_a_frame_with_nothing_readable_is_uninformative(self) -> None:
        assert not observe([read(s, "1m", contiguous=False) for s in SEATS]).informative


class TestCost:
    def test_each_missing_tile_costs_the_count_weight(self) -> None:
        cost = cost_matrix([seen("1m", "", "", "")], [state("1m 2m 3m", "", "", "")])
        assert cost[0, 0] == 2 * COUNT_WEIGHT

    def test_an_unknown_cell_never_conflicts(self) -> None:
        cost = cost_matrix([seen(f"1m {UNKNOWN}", "", "", "")], [state("1m 9p", "", "", "")])
        assert cost[0, 0] == 0

    def test_a_confident_wrong_cell_costs_one(self) -> None:
        cost = cost_matrix([seen("1m 2m", "", "", "")], [state("1m 9p", "", "", "")])
        assert cost[0, 0] == 1

    def test_an_unreadable_seat_costs_nothing_against_any_state(self) -> None:
        cost = cost_matrix(
            [seen(None, None, None, None)], [state("", "", "", ""), state("1m", "2m", "", "")]
        )
        assert cost.tolist() == [[0, 0]]


class TestAlign:
    def test_each_frame_lands_on_the_state_it_shows(self) -> None:
        states = [state("", "", "", ""), state("1m", "", "", ""), state("1m", "2m", "", "")]
        frames = [seen("", "", "", ""), seen("1m", "", "", ""), seen("1m", "2m", "", "")]
        assert align_monotone(cost_matrix(frames, states)).tolist() == [0, 1, 2]

    def test_it_never_goes_back(self) -> None:
        """中間那一幀讀短了一張,看起來像上一個狀態 —— 但前後都已經是下一個,
        而畫面不會倒退。這正是靠順序抓得到讀錯的地方。"""
        states = [state("1m", "", "", ""), state("1m 2m", "", "", "")]
        frames = [seen("1m 2m", "", "", ""), seen("1m", "", "", ""), seen("1m 2m", "", "", "")]
        assert align_monotone(cost_matrix(frames, states)).tolist() == [1, 1, 1]

    def test_states_may_be_skipped_between_frames(self) -> None:
        """畫面只在變化時存檔,兩幀之間可能發生了不只一件事。"""
        states = [state("", "", "", ""), state("1m", "", "", ""), state("1m", "2m", "", "")]
        frames = [seen("", "", "", ""), seen("1m", "2m", "", "")]
        assert align_monotone(cost_matrix(frames, states)).tolist() == [0, 2]

    def test_a_tie_goes_forward(self) -> None:
        cost = np.zeros((1, 3))
        assert align_monotone(cost).tolist() == [2]

    def test_empty_inputs(self) -> None:
        assert align_monotone(np.zeros((0, 3))).tolist() == []
        assert align_monotone(np.zeros((2, 0))).tolist() == [0, 0]


class TestInterior:
    def test_edges_of_each_run_are_dropped(self) -> None:
        path = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        kept = interior(path, 1).tolist()
        assert kept == [False, True, True, False, False, True, True, True, False]

    def test_a_short_run_disappears_entirely(self) -> None:
        assert not interior(np.array([0, 1, 1, 2]), 1)[0]

    def test_zero_keeps_everything(self) -> None:
        assert interior(np.array([0, 1]), 0).all()
