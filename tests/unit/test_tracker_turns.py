"""四家捨牌 → 依序的動作。規則表見 ``mia/tracker/turns.py``。"""

from __future__ import annotations

from mia.tracker import Action, Step, TurnTracker
from mia.tracker.turns import WAIT

T = Action.TSUMO
C = Action.CALL
D = Action.DAHAI


def shape(steps: list[Step]) -> list[tuple[Action, int]]:
    return [(s.action, s.actor) for s in steps]


class TestNormalPlay:
    def test_the_dealer_draws_first(self) -> None:
        turns = TurnTracker(oya=2)
        turns.tick(1, wall=69)
        assert shape(turns.tick(2, [(2, "1m")], wall=69)) == [(T, 2), (D, 2)]

    def test_the_tile_is_kept(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        assert turns.tick(2, [(0, "1m")])[-1] == Step(D, 0, "1m")

    def test_the_turn_goes_around(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=68)
        assert shape(turns.tick(3, [(1, "2m")], wall=67)) == [(T, 1), (D, 1)]

    def test_two_in_one_tick_follow_the_rotation(self) -> None:
        """每秒讀一次,一輪常常跨兩個動作;傳進來的順序不算數。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=68)
        steps = turns.tick(3, [(2, "3m"), (1, "2m")], wall=67)
        # 第二張的摸牌還沒在上一輪的余牌裡,要等一下才確定。
        for tick in range(4, 4 + WAIT):
            steps += turns.tick(tick, wall=67)
        assert [s.actor for s in steps if s.action is D] == [1, 2]


class TestCalls:
    def _after_first_discard(self) -> TurnTracker:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        return turns

    def test_a_skipped_seat_waits_for_the_meld(self) -> None:
        """對家碰了莊家的牌:下家沒摸,輪到的不是他。"""
        turns = self._after_first_discard()
        assert turns.tick(3, [(2, "5p")], wall=69) == []
        assert shape(turns.tick(4, melds=[2], wall=69)) == [(C, 2), (D, 2)]

    def test_the_next_seat_with_no_draw_waits_for_the_meld(self) -> None:
        """下家吃:看起來就像輪到他,差別只有那一筆副露。"""
        turns = self._after_first_discard()
        assert turns.tick(3, [(1, "5p")], wall=69) == []
        assert shape(turns.tick(4, melds=[1], wall=69)) == [(C, 1), (D, 1)]

    def test_a_plain_draw_that_outran_the_wall(self) -> None:
        """動作很快的時候,摸牌與打牌落在同一輪,上一輪的余牌還沒看到。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(1, "5p")], wall=68)
        for tick in range(4, 4 + WAIT):
            steps += turns.tick(tick, wall=68)
        assert shape(steps) == [(T, 1), (D, 1)]

    def test_a_tile_that_was_called_at_once_is_filled_in(self) -> None:
        """下家打的牌被上家馬上鳴走,河裡從沒出現過 —— 余牌減過,所以知道。"""
        turns = self._after_first_discard()
        turns.tick(3, wall=68)
        steps = turns.tick(4, [(3, "5p")], melds=[3], wall=68)
        assert shape(steps) == [(T, 1), (D, 1), (C, 3), (D, 3)]
        assert steps[1].pai is None

    def test_a_closed_kan_draws_twice(self) -> None:
        turns = self._after_first_discard()
        turns.tick(3, wall=67)
        steps = turns.tick(4, [(1, "5p")], melds=[1], wall=67)
        assert shape(steps) == [(T, 1), (C, 1), (T, 1), (D, 1)]


class TestWhatIsNotADiscard:
    def test_the_result_screen_leaves_nothing(self) -> None:
        """和了之後結算畫面被讀成一排 ``?``:沒有摸牌、也沒有鳴牌。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=68)
        turns.tick(3, [(1, "2m")], wall=68)
        steps = turns.tick(4, [(3, None), (3, None), (3, None)], wall=68)
        for tick in range(5, 6 + WAIT):
            steps += turns.tick(tick, wall=68)
        assert steps == []
        assert turns.last == 1


class TestItDoesNotDrift:
    def test_an_extra_draw_is_absorbed(self) -> None:
        """加槓看不見,但嶺上讓余牌多減一張。不吸收的話之後每次跳家都多補一張。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.tick(3, wall=67)
        turns.tick(4, [(1, "2m")], wall=67)
        for tick in range(5, 5 + WAIT):
            turns.tick(tick, wall=67)
        assert turns.drawn == 3
