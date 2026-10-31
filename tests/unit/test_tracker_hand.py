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

    def test_an_ankan_comes_with_the_replacement_draw(self) -> None:
        hand = ["E"] * 4 + TEHAI[:9]
        watcher = HandWatcher()
        watcher.observe(hand)
        watcher.observe([*hand, "9p"], "9p")
        moves = watcher.observe([*TEHAI[:9], "9p", "1p"], "1p")
        assert moves == [OwnMove(Move.CALL, ("E",) * 4), OwnMove(Move.DRAW, ("1p",))]


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

    def test_a_swap_seen_again_and_again_is_believed(self) -> None:
        """打出去馬上被鳴走的牌進不了河。"""
        watcher = dealt()
        hand = ["9p", *TEHAI[1:]]
        assert watcher.observe(hand, thrown=0) == []
        assert watcher.observe(hand, thrown=0) == []
        assert len(watcher.observe(hand, thrown=0)) == 2

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
