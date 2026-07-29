"""從事件流追蹤自己的手牌。

這份邏輯同時是規則式 baseline 的輸入,也是 CV 準確率報告的**標準答案** ——
它錯了,準確率那個數字就完全沒有意義。所以測試比一般的追蹤器嚴。
"""

from __future__ import annotations

import pytest

from majsoul_copilot.mjai import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Kakan,
    Kita,
    Pon,
    StartGame,
    StartKyoku,
    Tsumo,
)
from majsoul_copilot.mjai.handstate import HandTracker, UnknownSeatError

TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]


def kyoku(hand: list[str], *, seat: int = 0) -> StartKyoku:
    tehais = [["?"] * 13 for _ in range(4)]
    tehais[seat] = hand
    return StartKyoku("E", 1, 0, 0, 0, "2s", tehais, [25000] * 4)


def tracker(hand: list[str] = TENPAI, *, seat: int = 0) -> HandTracker:
    tracked = HandTracker()
    tracked.feed(StartGame(id=seat))
    tracked.feed(kyoku(hand, seat=seat))
    return tracked


class TestDeal:
    def test_the_starting_hand_is_taken_from_tehais(self) -> None:
        assert tracker().tiles == TENPAI

    def test_a_non_zero_seat_reads_its_own_tehai(self) -> None:
        """自己不一定坐 0 —— 讀錯索引會拿到別人的配牌,而且看起來完全正常。"""
        assert tracker(seat=2).tiles == TENPAI

    def test_a_new_kyoku_resets_everything(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        tracked.feed(kyoku(["1z"] * 4 + ["2z"] * 4 + ["3z"] * 4 + ["4z"]))
        assert tracked.tiles.count("1z") == 4
        assert tracked.drawn is None

    def test_a_kyoku_without_a_game_is_an_error(self) -> None:
        with pytest.raises(UnknownSeatError, match="不知道自己是誰"):
            HandTracker().feed(kyoku(TENPAI))


class TestDrawAndDiscard:
    def test_a_draw_is_added_and_remembered(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        assert len(tracked.tiles) == 14
        assert tracked.drawn == "9s"

    def test_a_discard_clears_the_draw(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        tracked.feed(Dahai(actor=0, pai="9s", tsumogiri=True))
        assert len(tracked.tiles) == 13
        assert tracked.drawn is None

    def test_other_players_actions_are_ignored(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=1, pai="?"))
        tracked.feed(Dahai(actor=1, pai="1z", tsumogiri=False))
        assert tracked.tiles == TENPAI


class TestMelds:
    def test_a_pon_removes_only_my_own_tiles(self) -> None:
        """被碰的那張來自別家,本來就不在我手上 —— 扣掉它會讓手牌少一張。"""
        hand = ["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"]
        tracked = tracker(hand)
        tracked.feed(Pon(actor=0, target=2, pai="1m", consumed=["1m", "1m"]))
        assert len(tracked.tiles) == 11
        assert tracked.melds == 1

    def test_a_chi_removes_only_my_own_tiles(self) -> None:
        tracked = tracker()
        tracked.feed(Chi(actor=0, target=3, pai="4m", consumed=["2m", "3m"]))
        assert len(tracked.tiles) == 11

    def test_a_meld_leaves_no_draw(self) -> None:
        """吃碰不從牌山補牌 —— 鳴完直接輪到自己打,不可能是摸切。"""
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        tracked.feed(Dahai(actor=0, pai="9s", tsumogiri=True))
        tracked.feed(Chi(actor=0, target=3, pai="4m", consumed=["2m", "3m"]))
        assert tracked.drawn is None

    def test_an_ankan_removes_all_four(self) -> None:
        hand = ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"]
        tracked = tracker(hand)
        tracked.feed(Tsumo(actor=0, pai="1m"))
        tracked.feed(Ankan(actor=0, consumed=["1m"] * 4))
        assert len(tracked.tiles) == 10

    def test_a_daiminkan_removes_three(self) -> None:
        """大明槓的第四張來自別家,與碰同理。"""
        hand = ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"]
        tracked = tracker(hand)
        tracked.feed(Daiminkan(actor=0, target=1, pai="1m", consumed=["1m"] * 3))
        assert len(tracked.tiles) == 10

    def test_a_kakan_removes_the_added_tile(self) -> None:
        """加槓是在已經碰過的三張上加一張,那三張早就不在暗手牌裡了。"""
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        tracked.feed(Kakan(actor=0, pai="9s", consumed=["9s"] * 3))
        assert len(tracked.tiles) == 13
        assert tracked.drawn is None

    def test_a_kita_removes_the_north(self) -> None:
        """MJAI 的北是 ``N``,不是雀魂記法的 ``4z``。"""
        hand = ["N", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]
        tracked = tracker(hand)
        tracked.feed(Kita(actor=0))
        assert "N" not in tracked.tiles

    def test_other_players_melds_do_not_touch_my_hand(self) -> None:
        tracked = tracker()
        tracked.feed(Pon(actor=1, target=2, pai="1z", consumed=["1z", "1z"]))
        assert len(tracked.tiles) == 13


class TestMeldCount:
    @pytest.mark.parametrize(("tiles", "melds"), [(13, 0), (10, 1), (7, 2), (4, 3), (1, 4)])
    def test_melds_come_from_the_tile_count(self, tiles: int, melds: int) -> None:
        """副露內容不必記 —— 少掉的每 3 張就是一組面子。"""
        tracked = tracker()
        tracked.tiles = TENPAI[:tiles]
        assert tracked.melds == melds

    def test_an_illegal_count_reports_minus_one(self) -> None:
        """逢 3 的倍數不是合法的暗手牌張數,回 -1 而不是一個看似合理的數字。"""
        tracked = tracker()
        tracked.tiles = TENPAI[:12]
        assert tracked.melds == -1

    def test_a_drawn_hand_counts_against_fourteen(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        assert tracked.melds == 0

    def test_a_melded_hand_counts_against_fourteen_too(self) -> None:
        """吃碰之後沒有摸牌,但手上是 11 張、還欠一張沒打 —— 狀態等同 14 張。

        用 ``drawn is None`` 判斷該對照 13 還是 14 的話,這裡會算成 -1
        (13-11=2 不能被 3 整除),每次鳴牌之後副露組數都會壞掉。
        """
        tracked = tracker()
        tracked.feed(Chi(actor=0, target=3, pai="4m", consumed=["2m", "3m"]))
        assert tracked.drawn is None
        assert len(tracked.tiles) == 11
        assert tracked.melds == 1


class TestDesync:
    def test_discarding_a_tile_not_held_is_recorded(self) -> None:
        """事件流漏了一段。不拋例外 —— 錄影其餘部分仍然有用,但要標記出來。"""
        tracked = tracker()
        tracked.feed(Dahai(actor=0, pai="9s", tsumogiri=False))
        assert tracked.desyncs == 1
        assert not tracked.in_sync

    def test_a_clean_stream_stays_in_sync(self) -> None:
        tracked = tracker()
        tracked.feed(Tsumo(actor=0, pai="9s"))
        tracked.feed(Dahai(actor=0, pai="9s", tsumogiri=True))
        assert tracked.in_sync

    def test_a_new_kyoku_clears_the_desync(self) -> None:
        """上一局壞掉不代表下一局也壞 —— 配牌是重新給的。"""
        tracked = tracker()
        tracked.feed(Dahai(actor=0, pai="9s", tsumogiri=False))
        tracked.feed(kyoku(TENPAI))
        assert tracked.in_sync
