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

    def test_a_slow_human_still_gets_the_call(self) -> None:
        """真人鳴完想很久才打。CPU 馬上打,所以這在 CPU 素材裡看不出來。"""
        turns = self._after_first_discard()
        turns.tick(3, melds=[2], wall=69)
        steps = turns.tick(3 + 15, [(2, "5p")], wall=69)
        assert shape(steps) == [(C, 2), (D, 2)]
        assert steps[0].target == 0

    def test_own_turn_does_not_turn_a_call_into_a_draw(self) -> None:
        """對家碰完打牌,下一刻就輪到自己,副露還沒確認。沒有人摸牌,所以是鳴牌。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(2, "5p")], wall=68, hurry=True)
        assert shape(steps) == [(C, 2), (D, 2)]

    def test_a_call_counted_early_is_not_counted_again(self) -> None:
        turns = self._after_first_discard()
        turns.tick(3, [(2, "5p")], wall=68, hurry=True)
        turns.tick(4, melds=[2], wall=68)
        assert shape(turns.tick(5, [(3, "1p")], wall=67)) == [(T, 3), (D, 3)]
        assert shape(turns.tick(6, [(0, "2p")], wall=66)) == [(T, 0), (D, 0)]
        assert shape(turns.tick(7, [(1, "3p")], wall=65)) == [(T, 1), (D, 1)]
        assert shape(turns.tick(8, [(2, "4p")], wall=64)) == [(T, 2), (D, 2)]

    def test_the_next_seat_with_no_draw_waits_for_the_meld(self) -> None:
        """下家吃:看起來就像輪到他,差別只有那一筆副露。"""
        turns = self._after_first_discard()
        assert turns.tick(3, [(1, "5p")], wall=69) == []
        assert shape(turns.tick(4, melds=[1], wall=69)) == [(C, 1), (D, 1)]

    def test_a_growing_meld_lets_the_call_through_at_once(self) -> None:
        """下家吃:余牌沒減,副露區也已經長了一組,不必等第二次讀到一樣的長度。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(1, "5p")], wall=69, growing=[1])
        assert shape(steps) == [(C, 1), (D, 1)]
        assert steps[0].target == 0

    def test_the_meld_confirmed_later_is_not_another_call(self) -> None:
        turns = self._after_first_discard()
        turns.tick(3, [(1, "5p")], wall=69, growing=[1])
        turns.tick(4, melds=[1], wall=68)
        assert turns._melds == []  # noqa: SLF001
        assert shape(turns.tick(5, [(2, "1p")], wall=67)) == [(T, 2), (D, 2)]
        assert shape(turns.tick(6, [(3, "2p")], wall=66)) == [(T, 3), (D, 3)]

    def test_a_growing_meld_without_a_wall_reading_still_waits(self) -> None:
        """余牌讀不到時 ``unseen`` 一律是 0,不能拿來當沒人摸牌的證據。"""
        turns = TurnTracker(oya=0)
        turns.tick(1)
        turns.tick(2, [(0, "1m")])
        assert turns.tick(3, [(1, "5p")], growing=[1]) == []

    def test_a_growing_meld_on_another_seat_does_not_count(self) -> None:
        turns = self._after_first_discard()
        assert turns.tick(3, [(1, "5p")], wall=69, growing=[3]) == []

    def test_an_undecided_seat_waits_a_little(self) -> None:
        """摸切手切還沒看出來:多等一下再排,也先不接在尾巴。"""
        turns = self._after_first_discard()
        turns.hold = 0.5
        turns.tick(3, wall=68)
        assert turns.tick(4, [(1, "5p")], wall=68, undecided=[1]) == []
        assert turns.next_up is None
        assert turns.held(4.6)
        assert shape(turns.tick(4.6, wall=68, undecided=[1])) == [(T, 1), (D, 1)]

    def test_a_decided_seat_does_not_wait(self) -> None:
        turns = self._after_first_discard()
        turns.hold = 0.5
        turns.tick(3, wall=68)
        assert shape(turns.tick(4, [(1, "5p")], wall=68, undecided=[2])) == [(T, 1), (D, 1)]

    def test_own_turn_does_not_wait_for_it(self) -> None:
        turns = self._after_first_discard()
        turns.hold = 0.5
        turns.tick(3, wall=68)
        steps = turns.tick(4, [(1, "5p")], wall=68, undecided=[1], hurry=True)
        assert shape(steps) == [(T, 1), (D, 1)]

    def test_a_plain_draw_that_outran_the_wall(self) -> None:
        """動作很快的時候,摸牌與打牌落在同一輪,上一輪的余牌還沒看到。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(1, "5p")], wall=68)
        for tick in range(4, 4 + WAIT):
            steps += turns.tick(tick, wall=68)
        assert shape(steps) == [(T, 1), (D, 1)]

    def test_an_open_kan_draws_from_the_dead_wall(self) -> None:
        """對家大明槓莊家的牌:嶺上讓余牌減一張,不是有人被跳過。"""
        turns = self._after_first_discard()
        turns.tick(3, wall=68)
        steps = turns.tick(4, [(2, "5p")], melds=[2], wall=68, kans=[2])
        assert shape(steps) == [(C, 2), (T, 2), (D, 2)]
        assert steps[0] == Step(C, 2, target=0, kan=True)

    def test_the_next_seat_can_open_kan_too(self) -> None:
        """下家大明槓:余牌減一張,跟摸打一樣,只有副露的寬度分得開。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(1, "5p")], melds=[1], wall=68, kans=[1])
        assert shape(steps) == [(C, 1), (T, 1), (D, 1)]
        assert steps[0] == Step(C, 1, target=0, kan=True)

    def test_an_open_kan_with_a_stale_wall_still_counts_the_dead_wall(self) -> None:
        """余牌還沒更新也一樣:寬度說了算,嶺上的那一張要算進去。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(1, "5p")], melds=[1], wall=69, kans=[1])
        assert shape(steps) == [(C, 1), (T, 1), (D, 1)]
        assert turns.drawn == 2

    def test_a_late_river_keeps_its_turn(self) -> None:
        """下家的河這一輪讀不到,對家的牌先到:不是跳家,不要補假的隱藏捨牌。"""
        turns = self._after_first_discard()
        steps = turns.tick(3, [(2, "5p")], wall=67)
        steps += turns.tick(4, [(1, "2p")], wall=67)
        for tick in range(5, 5 + WAIT):
            steps += turns.tick(tick, wall=67)
        assert shape(steps) == [(T, 1), (D, 1), (T, 2), (D, 2)]
        assert [s.pai for s in steps if s.action is D] == ["2p", "5p"]

    def test_own_turn_flushes_what_is_waiting(self) -> None:
        """自己摸到牌了:前面的事都發生過,Mortal 要的是完整的局面。"""
        turns = TurnTracker(oya=1)
        turns.tick(1, wall=69)
        assert turns.tick(2, [(1, "1m")]) != []
        steps = turns.tick(3, [(2, "5p")], wall=68)
        assert steps == []
        assert shape(turns.tick(4, hurry=True, wall=66)) == [(T, 2), (D, 2)]

    def test_the_very_first_tile_called_at_once(self) -> None:
        """莊家第一張就被鳴走:河裡沒出現過,而開局沒有上一輪的余牌可以比。"""
        turns = TurnTracker(oya=2)
        steps = turns.tick(1, [(3, "9s")], melds=[3], wall=69)
        assert shape(steps) == [(T, 2), (D, 2), (C, 3), (D, 3)]
        assert steps[2].target == 2

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


class TestAWallThatIsAhead:
    """牌讀得晚,再下一家已經摸了:余牌多減一張。"""

    def test_ones_own_discard_does_not_wait(self) -> None:
        """自己有沒有槓手牌看得到。等的話實測下家、對家的牌全部晚 3 秒。"""
        turns = TurnTracker(oya=3)
        turns.tick(1, wall=69)
        turns.tick(2, [(3, "1m")], wall=69)
        turns.see_wall(3, 67)
        assert shape(turns.tick(4, [(0, "2m")], wall=67)) == [(T, 0), (D, 0)]

    def test_the_dealers_first_discard_does_not_wait(self) -> None:
        """那時候還沒讀過余牌。實測莊家第一張每局都晚 4 秒,後面全部跟著晚。"""
        turns = TurnTracker(oya=0)
        assert shape(turns.tick(1, [(0, "1m")])) == [(T, 0), (D, 0)]

    def test_a_call_of_ones_own_still_counts(self) -> None:
        turns = TurnTracker(oya=3)
        turns.tick(1, wall=69)
        turns.tick(2, [(3, "1m")], wall=69)
        assert (C, 0) in shape(turns.tick(3, [(0, "2m")], melds=[0], wall=69))

    def test_a_skipped_seat_before_ones_own_tile_waits_a_moment(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.see_wall(2.5, 68)
        turns.tick(3, [(1, "2m")], wall=68)
        turns.see_wall(4, 66)
        assert turns.tick(5, [(0, "3m")], wall=66) == []
        steps = turns.tick(6.1, wall=66)
        assert shape(steps)[-2:] == [(T, 0), (D, 0)]

    def test_another_seat_waits_a_moment_for_a_kan(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.see_wall(3, 67)
        assert turns.tick(4, [(1, "2m")], wall=67) == []

    def test_then_it_is_a_plain_discard(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.see_wall(3, 67)
        turns.tick(4, [(1, "2m")], wall=67)
        assert shape(turns.tick(5.1, wall=67)) == [(T, 1), (D, 1)]

    def test_a_kan_that_shows_up_in_time_is_still_a_kan(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.see_wall(3, 67)
        turns.tick(4, [(1, "2m")], wall=67)
        steps = turns.tick(4.5, melds=[1], wall=67)
        assert (C, 1) in shape(steps)

    def test_the_tiles_behind_it_are_not_held_up(self) -> None:
        turns = TurnTracker(oya=3)
        turns.tick(1, wall=69)
        turns.tick(2, [(3, "1m")], wall=69)
        turns.see_wall(3, 67)
        steps = turns.tick(4, [(0, "2m"), (1, "3m")], wall=67)
        assert shape(steps) == [(T, 0), (D, 0), (T, 1), (D, 1)]


class TestABatchOfTiles:
    """中途才開始看,或畫面被蓋了好幾秒。"""

    def test_it_goes_around_the_table(self) -> None:
        turns = TurnTracker(oya=1)
        steps = turns.jump(5, [(1, "1m"), (1, "2m"), (2, "3m"), (3, "4m"), (0, "5m")], wall=65)
        assert shape(steps)[1::2] == [(D, 1), (D, 2), (D, 3), (D, 0), (D, 1)]

    def test_a_seat_with_fewer_tiles_is_skipped(self) -> None:
        turns = TurnTracker(oya=0)
        steps = turns.jump(5, [(0, "1m"), (0, "2m"), (0, "3m"), (2, "4m")])
        assert shape(steps)[1::2] == [(D, 0), (D, 2), (D, 0), (D, 0)]

    def test_each_seat_keeps_its_own_order(self) -> None:
        turns = TurnTracker(oya=0)
        steps = turns.jump(5, [(0, "1m", None, 0), (0, "2m", None, 1), (1, "3m", None, 0)])
        assert [s.pai for s in steps if s.action is D and s.actor == 0] == ["1m", "2m"]

    def test_the_wall_is_believed(self) -> None:
        turns = TurnTracker(oya=0)
        turns.jump(5, [(0, "1m"), (1, "2m"), (2, "3m"), (3, "4m")], wall=60)
        assert turns.drawn == 10

    def test_it_carries_on_afterwards(self) -> None:
        turns = TurnTracker(oya=0)
        turns.jump(5, [(0, "1m"), (1, "2m"), (2, "3m"), (3, "4m")], wall=66)
        turns.tick(6, wall=65)
        assert shape(turns.tick(7, [(0, "5m")], wall=65)) == [(T, 0), (D, 0)]

    def test_tiles_still_waiting_go_first(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(2, "1m")], wall=69)
        steps = turns.jump(3, [(0, "2m"), (1, "3m"), (3, "4m")])
        assert [s.pai for s in steps if s.action is D] == ["2m", "3m", "1m", "4m"]


class TestTheWallBetweenRounds:
    def test_a_fresh_wall_keeps_a_tile_from_waiting(self) -> None:
        """看的是牌落下之前的余牌。太舊的話自己那次摸牌沒算進去,看起來像鳴牌。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        assert shape(turns.tick(2, [(0, "1m")], wall=69)) == [(T, 0), (D, 0)]
        turns.see_wall(2.5, 68)
        assert shape(turns.tick(3, [(1, "2m")], wall=68)) == [(T, 1), (D, 1)]

    def test_an_unread_wall_changes_nothing(self) -> None:
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.see_wall(2, None)
        assert shape(turns.tick(3, [(0, "1m")], wall=69)) == [(T, 0), (D, 0)]


class TestNextUp:
    @staticmethod
    def _waiting(*discard: tuple[int, str | None]) -> TurnTracker:
        """莊家打完,下一張讀到了,余牌還沒更新,所以還排不出來。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.tick(3, wall=69)
        assert turns.tick(4, list(discard), wall=69) == []
        return turns

    def test_the_next_seat_is_shown_before_it_is_settled(self) -> None:
        assert self._waiting((1, "2m")).next_up == Step(D, 1, "2m")

    def test_a_skipped_seat_is_not_shown(self) -> None:
        """跳家多半有人鳴牌,順序還不知道。"""
        assert self._waiting((2, "2m")).next_up is None

    def test_a_tile_we_cannot_read_is_not_shown(self) -> None:
        """讀不出來的格子可能根本不是牌:結算畫面就長這樣。"""
        assert self._waiting((1, None)).next_up is None

    def test_nothing_waiting_nothing_shown(self) -> None:
        assert TurnTracker(oya=0).next_up is None


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
    def test_a_late_river_does_not_absorb_the_next_draw(self) -> None:
        """下家的牌晚讀到,余牌已經連下家的下家都減了:那一張不能算給下家。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.tick(3, wall=67)
        turns.tick(4, [(1, "2m"), (2, "3m")], wall=67)
        assert turns.drawn <= 3
        assert shape(turns.tick(4 + WAIT, hurry=True)) != []
        assert turns.drawn == 3

    def test_a_wide_reach_tile_is_not_two_discards(self) -> None:
        """橫放的宣言牌佔兩格:同一家沒摸牌又打一張,第二張不是真的。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m", None, 0)], wall=68)
        turns.tick(3, [(1, "2m", None, 0)], wall=68)
        assert turns.tick(4, [(1, "P", None, 1)], wall=68) == []
        assert turns.tick(4 + WAIT, wall=68) == []
        assert turns.doubled == {(1, 1): 0}
        turns.tick(4 + 2 * WAIT, [(2, "3m", None, 0)], wall=67)
        assert shape(turns.tick(5 + 2 * WAIT, hurry=True)) == [(T, 2), (D, 2)]

    def test_an_extra_draw_is_absorbed(self) -> None:
        """加槓不變長,但嶺上讓余牌多減一張。不吸收的話之後每次跳家都多補一張。"""
        turns = TurnTracker(oya=0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        turns.tick(3, wall=67)
        turns.tick(4, [(1, "2m")], wall=67)
        for tick in range(5, 5 + WAIT):
            turns.tick(tick, wall=67)
        assert turns.drawn == 3


class TestTheTurnLight:
    """中央面板的燈說輪到誰。見 ``mia/tracker/turns.py``「中央的燈」。"""

    def _after_first_discard(self) -> TurnTracker:
        """莊家(自己)摸打了第一張,燈亮在莊家那邊。"""
        turns = TurnTracker(oya=0)
        turns.see_light(0.5, 0)
        turns.tick(1, wall=69)
        turns.tick(2, [(0, "1m")], wall=69)
        return turns

    def test_a_draw_the_wall_missed_is_still_a_draw(self) -> None:
        """余牌多算了一張:下家正常摸打,看起來卻像沒人摸牌。真人鳴牌前猶豫好幾秒,余牌一直不動,
        沒有燈的話這張牌等不到副露,被當成假的丟掉。"""
        turns = self._after_first_discard()
        turns.see_light(2.5, 1)
        assert shape(turns.tick(3, [(1, "2m")], wall=69)) == [(T, 1), (D, 1)]

    def test_without_the_light_it_waits(self) -> None:
        turns = self._after_first_discard()
        assert turns.tick(3, [(1, "2m")], wall=69) == []

    def test_a_draw_counted_twice_is_absorbed(self) -> None:
        """燈換過來之後的余牌說這次摸牌已經算過了:之前多算了一張,不再加。"""
        turns = self._after_first_discard()
        turns.drawn += 1
        turns.see_light(2.5, 1)
        turns.see_wall(2.6, 68)
        turns.tick(3, [(1, "2m")], wall=68)
        assert turns.drawn == 2

    def test_a_plain_draw_still_counts(self) -> None:
        turns = self._after_first_discard()
        turns.see_light(2.5, 1)
        turns.tick(3, [(1, "2m")], wall=68)
        assert turns.drawn == 2

    def test_a_jump_is_a_call_at_once(self) -> None:
        """燈從莊家直接跳到對家:對家碰了。不必等副露確認。"""
        turns = self._after_first_discard()
        turns.see_light(2.5, 2)
        assert shape(turns.tick(3, [(2, "5p")], wall=69)) == [(C, 2), (D, 2)]

    def test_the_meld_confirmed_later_is_not_another_call(self) -> None:
        turns = self._after_first_discard()
        turns.see_light(2.5, 2)
        turns.tick(3, [(2, "5p")], wall=69)
        turns.tick(4, melds=[2], wall=69)
        turns.see_light(4.5, 3)
        assert shape(turns.tick(5, [(3, "7s")], wall=68)) == [(T, 3), (D, 3)]

    def test_a_growing_meld_on_the_next_seat_is_still_a_chi(self) -> None:
        turns = self._after_first_discard()
        turns.see_light(2.5, 1)
        assert shape(turns.tick(3, [(1, "2m")], wall=69, growing=[1])) == [(C, 1), (D, 1)]

    def test_a_light_that_has_not_moved_yet_changes_nothing(self) -> None:
        """電腦打得快,牌常常比燈先到:照余牌推。"""
        turns = self._after_first_discard()
        turns.see_light(2.5, 0)
        turns.tick(3, wall=68)
        assert shape(turns.tick(4, [(1, "2m")], wall=68)) == [(T, 1), (D, 1)]

    def test_another_seat_in_between_is_left_to_the_wall(self) -> None:
        """燈經過了別家:中間有看不到的動作,照余牌推。"""
        turns = self._after_first_discard()
        turns.see_light(2.5, 1)
        turns.see_light(2.8, 2)
        assert turns.tick(3, [(2, "5p")], wall=69) == []

    def test_an_old_light_is_not_this_turn(self) -> None:
        """這一家上一輪的燈:這一輪的還沒換過來,照余牌推。"""
        turns = self._after_first_discard()
        turns.see_light(2.5, 1)
        turns.tick(3, [(1, "2m")], wall=68)
        for seat, tick in ((2, 4), (3, 5), (0, 6)):
            turns.see_light(tick - 0.5, seat)
            turns.tick(tick, [(seat, "9p")], wall=68 - seat)
        assert turns.last == 0
        assert turns.tick(10, [(1, "3m")], wall=65) == []

    def test_an_unread_light_is_not_a_change(self) -> None:
        turns = self._after_first_discard()
        turns.see_light(2.4, 1)
        turns.see_light(2.5, None)
        assert shape(turns.tick(3, [(1, "2m")], wall=69)) == [(T, 1), (D, 1)]
