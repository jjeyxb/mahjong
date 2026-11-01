"""手牌的變化 → 自己的動作。規則表見 ``mia/tracker/hand.py``。"""

from __future__ import annotations

from mia.tracker import HandWatcher, Move, OwnMove

TEHAI = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "S"]


def dealt() -> HandWatcher:
    watcher = HandWatcher()
    watcher.observe(TEHAI)
    return watcher


class TestDeal:
    def test_thirteen_tiles_are_the_deal(self) -> None:
        assert dealt().tehai == tuple(TEHAI)

    def test_the_dealer_starts_with_a_draw(self) -> None:
        watcher = HandWatcher()
        moves = watcher.observe([*TEHAI, "9p"], "9p")
        assert watcher.tehai == tuple(TEHAI)
        assert moves == [OwnMove(Move.DRAW, ("9p",))]

    def test_a_new_deal_replaces_the_stale_hand(self) -> None:
        """上一局的手牌還留在畫面上時就收下了,真的配牌隨後才到。"""
        watcher = dealt()
        fresh = ["5s", "6s", *TEHAI[2:]]
        assert watcher.observe(fresh) == []
        assert watcher.tehai == tuple(fresh)
        assert watcher.moves == []


class TestMoves:
    def test_draw_and_discard(self) -> None:
        watcher = dealt()
        assert watcher.observe([*TEHAI, "9p"], "9p") == [OwnMove(Move.DRAW, ("9p",))]
        assert watcher.observe(["9p", *TEHAI[1:]]) == [OwnMove(Move.DISCARD, ("1m",))]

    def test_discarding_the_drawn_tile_is_tsumogiri(self) -> None:
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        assert watcher.observe(TEHAI) == [OwnMove(Move.DISCARD, ("9p",), tsumogiri=True)]

    def test_a_pon_takes_two_tiles(self) -> None:
        watcher = dealt()
        moves = watcher.observe(TEHAI[:9] + TEHAI[11:])
        assert moves == [OwnMove(Move.CALL, ("E", "E"))]

    def test_a_pon_and_a_discard_between_two_reads(self) -> None:
        """手牌一次少三張,兩張一樣一張不同:是碰完打牌,不是大明槓。"""
        watcher = dealt()
        rest = [tile for tile in TEHAI if tile not in ("S", "1m")]
        assert watcher.observe(rest) == [
            OwnMove(Move.CALL, ("S", "S")),
            OwnMove(Move.DISCARD, ("1m",)),
        ]

    def test_a_chi_and_a_discard_between_two_reads(self) -> None:
        watcher = dealt()
        rest = [tile for tile in TEHAI if tile not in ("7s", "8s", "S")]
        rest.append("S")
        assert watcher.observe(rest) == [
            OwnMove(Move.CALL, ("7s", "8s")),
            OwnMove(Move.DISCARD, ("S",)),
        ]

    def test_a_pon_a_discard_and_the_next_draw_between_two_reads(self) -> None:
        watcher = dealt()
        rest = [tile for tile in TEHAI if tile not in ("S", "1m")]
        assert watcher.observe([*rest, "9p"], "9p") == [
            OwnMove(Move.CALL, ("S", "S")),
            OwnMove(Move.DISCARD, ("1m",)),
            OwnMove(Move.DRAW, ("9p",)),
        ]

    def test_three_of_a_kind_gone_is_still_an_open_kan(self) -> None:
        watcher = HandWatcher()
        tehai = ["1m", "1m", "1m", *TEHAI[3:]]
        watcher.observe(tehai)
        assert watcher.observe([*tehai[3:], "9p"], "9p") == [
            OwnMove(Move.CALL, ("1m", "1m", "1m")),
            OwnMove(Move.DRAW, ("9p",)),
        ]

    def test_a_tsumogiri_too_fast_to_see_is_caught_up(self) -> None:
        """立直後自動摸切,手牌從來沒讀到 14 張,但河裡多了一張。"""
        watcher = dealt()
        assert watcher.catch_up(1, "W") == [
            OwnMove(Move.DRAW, ("W",)),
            OwnMove(Move.DISCARD, ("W",), tsumogiri=True),
        ]
        assert watcher.hand == TEHAI

    def test_nothing_to_catch_up_when_the_river_agrees(self) -> None:
        assert dealt().catch_up(0, "W") == []

    def test_an_unread_tile_cannot_be_caught_up(self) -> None:
        assert dealt().catch_up(1, None) == []

    def test_not_while_the_drawn_tile_is_still_in_hand(self) -> None:
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        assert watcher.catch_up(1, "9p") == []

    def test_an_ankan_comes_with_the_replacement_draw(self) -> None:
        hand = ["E"] * 4 + TEHAI[:9]
        watcher = HandWatcher()
        watcher.observe(hand)
        watcher.observe([*hand, "9p"], "9p")
        moves = watcher.observe([*TEHAI[:9], "9p", "1p"], "1p")
        assert moves == [OwnMove(Move.CALL, ("E",) * 4), OwnMove(Move.DRAW, ("1p",))]


class TestASwapWithACall:
    def test_a_discard_called_at_once_never_reaches_the_river(self) -> None:
        """河沒有多,但別家的副露多一組:是摸打,牌被鳴走了。"""
        watcher = dealt()
        hand = ["9p", *TEHAI[1:]]
        watcher.observe(hand, thrown=0, called=1)
        watcher.observe(hand, thrown=0, called=1)
        moves = watcher.observe(hand, thrown=0, called=1)
        assert [move.move for move in moves] == [Move.DRAW, Move.DISCARD]

    def test_an_old_call_is_not_evidence(self) -> None:
        """那組副露是上一手之前的事,跟這次換牌無關。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p", called=1)
        watcher.observe(TEHAI, called=1)
        hand = ["8p", *TEHAI[1:]]
        for _ in range(3):
            watcher.observe(hand, thrown=1, called=1)
        assert len(watcher.moves) == 2


class TestAddedKan:
    """畫面沒變就不會重讀,所以這裡每副手牌都只給一次。"""

    #: 碰了東、打掉 9 索之後的手牌。
    BASE = ("1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "S", "S")

    def _with_fourth(self) -> HandWatcher:
        watcher = HandWatcher()
        watcher.observe(TEHAI)
        watcher.observe([*self.BASE, "9s"])
        watcher.observe(self.BASE)
        watcher.observe([*self.BASE, "E"], "E")
        return watcher

    def test_kan_and_dead_wall_read_at_once(self) -> None:
        watcher = self._with_fourth()
        moves = watcher.observe([*self.BASE, "2p"], "2p", thrown=1)
        assert moves == [OwnMove(Move.CALL, ("E",)), OwnMove(Move.DRAW, ("2p",))]

    def test_the_dead_wall_tile_can_come_later(self) -> None:
        """加槓的動畫中間只少一張:不是打牌,等嶺上。"""
        watcher = self._with_fourth()
        assert watcher.observe(self.BASE, thrown=1) == []
        moves = watcher.observe([*self.BASE, "2p"], "2p", thrown=1)
        assert moves == [OwnMove(Move.CALL, ("E",)), OwnMove(Move.DRAW, ("2p",))]

    def test_what_follows_stays_in_step(self) -> None:
        watcher = self._with_fourth()
        watcher.observe(self.BASE, thrown=1)
        watcher.observe([*self.BASE, "2p"], "2p", thrown=1)
        moves = watcher.observe(self.BASE, thrown=2)
        assert moves == [OwnMove(Move.DISCARD, ("2p",), tsumogiri=True)]
        assert watcher.stuck == 0

    def test_throwing_the_fourth_is_not_a_kan(self) -> None:
        """第四張也可以打掉。河裡多一張才分得出來。"""
        watcher = self._with_fourth()
        moves = watcher.observe(self.BASE, thrown=2)
        assert moves == [OwnMove(Move.DISCARD, ("E",), tsumogiri=True)]

    def test_the_river_settles_it_when_it_is_late(self) -> None:
        watcher = self._with_fourth()
        assert watcher.observe(self.BASE, thrown=1) == []
        moves = watcher.catch_up(2, "E")
        assert moves == [OwnMove(Move.DISCARD, ("E",), tsumogiri=True)]
        assert sorted(watcher.hand) == sorted(self.BASE)

    def test_only_a_tile_we_ponned(self) -> None:
        """別的牌換掉是認錯,不是加槓。"""
        watcher = self._with_fourth()
        hand = [*self.BASE, "E"]
        hand[0] = "9m"
        assert watcher.observe(hand, "E", thrown=1) == []
        assert not any(len(m.tiles) == 1 and m.move is Move.CALL for m in watcher.moves)

    def test_a_tile_cannot_be_added_twice(self) -> None:
        watcher = self._with_fourth()
        watcher.observe([*self.BASE, "2p"], "2p", thrown=1)
        assert not watcher._can_add_kan("E")  # noqa: SLF001


class TestAnOldTileReadWrong:
    """配牌把 6 索認成 3 索,之後才認對。實測建議從那裡卡到局尾。"""

    SEEN = ("1m", "2m", "3m", "4p", "5p", "6p", "3s", "8s", "9s", "E", "E", "S", "S")
    REAL = ("1m", "2m", "3m", "4p", "5p", "6p", "6s", "8s", "9s", "E", "E", "S", "S")

    def _watcher(self) -> HandWatcher:
        watcher = HandWatcher()
        watcher.observe(self.SEEN)
        return watcher

    def test_it_is_fixed_at_the_next_draw(self) -> None:
        watcher = self._watcher()
        assert watcher.observe([*self.REAL, "2p"], "2p") == [OwnMove(Move.DRAW, ("2p",))]
        assert "3s" not in watcher.hand
        assert "6s" in watcher.tehai

    def test_it_is_fixed_at_the_next_discard(self) -> None:
        """讀到 14 張的那次認錯還留著,打牌時才認對。"""
        watcher = self._watcher()
        watcher.observe([*self.SEEN, "2p"], "2p")
        moves = watcher.observe([*self.REAL[1:], "2p"])
        assert moves == [OwnMove(Move.DISCARD, ("1m",))]
        assert watcher.stuck == 0

    def test_the_game_goes_on(self) -> None:
        watcher = self._watcher()
        hand = list(self.REAL)
        for drawn, thrown in (("2p", "1m"), ("7p", "2m"), ("3p", "3m"), ("4s", "4p")):
            hand.append(drawn)
            watcher.observe(hand, drawn)
            hand.remove(thrown)
            watcher.observe(hand)
        kinds = [move.move for move in watcher.moves]
        assert kinds == [Move.DRAW, Move.DISCARD] * 4
        assert watcher.stuck == 0

    def test_a_wrong_draw_is_still_fixed_first(self) -> None:
        """摸到的牌認錯最常見:打牌時少的兩張裡,先懷疑它。"""
        watcher = HandWatcher()
        watcher.observe(self.REAL)
        watcher.observe([*self.REAL, "3s"], "3s")
        moves = watcher.observe([*self.REAL[1:], "6s"])
        assert moves == [OwnMove(Move.DISCARD, ("1m",))]
        assert watcher.moves[0] == OwnMove(Move.DRAW, ("6s",))

    def _played(self) -> HandWatcher:
        """打過一手。還沒動過手時差很多會當成重新發牌,見 ``_only_dealt``。"""
        watcher = self._watcher()
        watcher.observe([*self.SEEN, "2p"], "2p")
        watcher.observe(self.SEEN)
        return watcher

    def test_two_tiles_need_two_bad_hands(self) -> None:
        """差兩張的讀數多半是動畫,不能一次就信。"""
        watcher = self._played()
        watcher.observe(["9m", "9p", *self.SEEN[2:]])
        assert sorted(watcher.hand) == sorted(self.SEEN)
        assert watcher.stuck == 1

    def test_it_never_stays_stuck(self) -> None:
        """差好幾張也一樣,連續解釋不了就接受讀到的。"""
        watcher = self._played()
        third = ["7m", "7p", *self.SEEN[2:]]
        for hand in (["9m", "9p", *self.SEEN[2:]], ["8m", "8p", *self.SEEN[2:]], third):
            watcher.observe(hand)
        assert sorted(watcher.hand) == sorted(third)
        assert watcher.stuck == 0


class TestMisreads:
    def test_the_same_bad_read_counts_once(self) -> None:
        """畫面沒變就是同一個問題,重讀幾次都不會變成「對不上」。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        watcher.observe(TEHAI)
        watcher.observe([*TEHAI, "9p"], "9p")
        bad = ["1p", "2p", *TEHAI[2:], "9p"]
        for _ in range(5):
            watcher.observe(bad, "9p")
        assert watcher.stuck == 1

    def test_a_fifth_copy_is_never_a_hand(self) -> None:
        """開局前的介面被讀成一副手牌,裡面有五張白。"""
        watcher = HandWatcher()
        assert watcher.observe(["P"] * 5 + TEHAI[:9], "P") == []
        assert watcher.tehai == ()

    def test_a_misread_draw_is_repaired_at_the_discard(self) -> None:
        """摸到的牌讀錯了,打牌時差兩張:改回去,不要讓假牌留在手上。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        moves = watcher.observe(["7p", *TEHAI[1:]])
        assert moves == [OwnMove(Move.DISCARD, ("1m",))]
        assert watcher.moves[0] == OwnMove(Move.DRAW, ("7p",))
        assert sorted(watcher.hand) == sorted(["7p", *TEHAI[1:]])

    def test_one_flickering_tile_is_not_a_turn(self) -> None:
        watcher = dealt()
        assert watcher.observe(["9p", *TEHAI[1:]], thrown=0) == []
        assert watcher.moves == []

    def test_a_swap_with_a_new_river_tile_is_a_turn(self) -> None:
        watcher = dealt()
        moves = watcher.observe(["9p", *TEHAI[1:]], thrown=1)
        assert [move.move for move in moves] == [Move.DRAW, Move.DISCARD]

    def test_a_discard_and_a_draw_between_two_reads(self) -> None:
        """14 張換了一張,河多一張:打完又摸了,不是加槓。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        fresh = watcher.observe([*TEHAI[1:], "9p", "2p"], "2p", thrown=1)
        assert fresh == [OwnMove(Move.DISCARD, ("1m",)), OwnMove(Move.DRAW, ("2p",))]

    def test_a_swap_with_no_meld_cannot_be_kakan(self) -> None:
        """14 張換了一張,而沒有碰過:是配牌認錯了,不是加槓。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        hand = ["7p", *TEHAI[1:], "9p"]
        for _ in range(3):
            assert watcher.observe(hand, "9p", thrown=0) == []
        assert watcher.tehai[0] == "7p"
        assert watcher.moves == [OwnMove(Move.DRAW, ("9p",))]

    def test_a_deal_read_mid_animation_is_read_again(self) -> None:
        """莊家第一副 14 張抓在翻牌途中,牌面差好幾張。"""
        watcher = HandWatcher()
        watcher.observe(["P", "P", *TEHAI[2:], "9p"], "9p")
        watcher.observe([*TEHAI, "9p"], "9p")
        assert watcher.tehai == tuple(TEHAI)
        assert watcher.moves == [OwnMove(Move.DRAW, ("9p",))]

    def test_a_swap_seen_again_and_again_is_a_correction(self) -> None:
        """河沒有多:一直讀到另一張是之前認錯了,不是摸打。當成摸打的話自己多一手,之後都對不上。"""
        watcher = dealt()
        hand = ["9p", *TEHAI[1:]]
        assert watcher.observe(hand, thrown=0) == []
        assert watcher.observe(hand, thrown=0) == []
        assert watcher.observe(hand, thrown=0) == []
        assert watcher.moves == []
        assert sorted(watcher.hand) == sorted(hand)
        assert watcher.tehai[0] == "9p"

    def test_a_bad_read_is_not_an_action(self) -> None:
        """差太多就不是一個動作:不收,追蹤的手牌不動。"""
        watcher = dealt()
        watcher.observe([*TEHAI, "9p"], "9p")
        watcher.observe(TEHAI)
        watcher.observe([*TEHAI, "9p"], "9p")
        before = list(watcher.hand)
        assert watcher.observe(["1p", "2p", *TEHAI[2:], "9p"], "9p") == []
        assert watcher.hand == before
        assert watcher.rejected == 1

    def test_the_drawn_slot_settles_a_misread_old_tile(self) -> None:
        watcher = dealt()
        moves = watcher.observe(["1p", *TEHAI[1:], "9p"], "9p")
        assert moves == [OwnMove(Move.DRAW, ("9p",))]
