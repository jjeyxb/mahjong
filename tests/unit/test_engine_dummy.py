"""規則式 baseline 的手牌追蹤與打牌選擇。

這個引擎的價值在於「不會亂」—— 它是 Mortal 的對照組,自己算錯的話,比出來的
差距就沒有意義。所以測試重點在手牌追蹤:摸、打、鳴牌之後手上還剩什麼。
"""

from __future__ import annotations

import pytest

from majsoul_copilot.engine import DummyEngine, EngineError
from majsoul_copilot.mjai import (
    Ankan,
    Chi,
    Dahai,
    Pon,
    Reach,
    StartGame,
    StartKyoku,
    Tsumo,
)

# 123m 456m 789m + 11p 對子 + 23p 兩面 = 聽 1p/4p,進張 6 枚。
#
# 刻意**不用**單騎聽牌當素材:單騎手上留 A 聽 B 與留 B 聽 A 的進張枚數往往
# 一樣多,是真正的平手,排序只好退回按牌名排。那種牌型測不出「有沒有挑對」,
# 只測得出排序規則。這手的最佳打牌是唯一的。
TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]


def kyoku(hand: list[str], *, oya: int = 0) -> StartKyoku:
    unknown = [["?"] * 13 for _ in range(3)]
    return StartKyoku("E", 1, 0, 0, oya, "2s", [hand, *unknown], [25000] * 4)


def started(hand: list[str] = TENPAI, *, seat: int = 0) -> DummyEngine:
    bot = DummyEngine()
    bot.start()
    bot.react(StartGame(id=seat))
    bot.react(kyoku(hand))
    return bot


class TestDiscard:
    def test_it_discards_the_useless_draw(self) -> None:
        """聽牌時摸到無關牌,該摸切 —— 留它會拆掉現成的聽牌。"""
        advice = started().react(Tsumo(actor=0, pai="9s"))
        assert advice.action == Dahai(actor=0, pai="9s", tsumogiri=True)

    def test_tsumogiri_is_false_when_it_keeps_the_draw(self) -> None:
        """摸到 1p 湊成暗刻,改切 3p —— 留下的是新摸的牌,不是摸切。"""
        advice = started().react(Tsumo(actor=0, pai="1p"))
        assert advice.action is not None
        assert advice.action.tsumogiri is False  # type: ignore[attr-defined]

    def test_other_players_draws_are_ignored(self) -> None:
        """別家摸牌與我無關 —— 而且我根本看不到他們摸了什麼。"""
        assert started().react(Tsumo(actor=1, pai="?")).action is None

    def test_non_turn_events_produce_no_action(self) -> None:
        bot = started()
        assert bot.react(Reach(actor=2)).action is None
        assert bot.react(Dahai(actor=2, pai="1z", tsumogiri=False)).action is None


class TestRedFives:
    def test_it_keeps_the_red_five_when_both_are_held(self) -> None:
        """向聽計算把赤五與普通五當同一種,但切錯那張會白丟一番。"""
        hand = ["5m", "5mr", "1m", "2m", "3m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "5p"]
        advice = started(hand).react(Tsumo(actor=0, pai="4m"))
        assert advice.action is not None
        assert advice.action.pai != "5mr", "不該切掉赤五"  # type: ignore[attr-defined]

    def test_it_can_discard_the_red_five_when_it_is_the_only_one(self) -> None:
        """留赤五是偏好,不是硬規則 —— 它是唯一的孤張時還是得切。

        123m 456m 789m + 11p + 2p + 5sr,摸 3p 之後 5sr 是唯一沒有用的那張。
        向聽計算回報的是「切 5s」,對應到手上實際的牌只有 ``5sr``。
        """
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "5sr"]
        advice = started(hand).react(Tsumo(actor=0, pai="3p"))
        assert advice.action is not None
        assert advice.action.pai == "5sr"  # type: ignore[attr-defined]


class TestHandTracking:
    def test_a_pon_removes_only_my_own_tiles(self) -> None:
        """被碰的那張來自別家,本來就不在我手上 —— 扣掉它會讓手牌少一張。"""
        hand = ["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"]
        bot = started(hand)
        bot.react(Pon(actor=0, target=2, pai="1m", consumed=["1m", "1m"]))
        assert len(bot._hand) == 11  # noqa: SLF001

    def test_a_chi_removes_only_my_own_tiles(self) -> None:
        hand = ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "5p"]
        bot = started(hand)
        bot.react(Chi(actor=0, target=3, pai="1m", consumed=["2m", "3m"]))
        assert len(bot._hand) == 11  # noqa: SLF001

    def test_an_ankan_removes_all_four(self) -> None:
        hand = ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p"]
        bot = started(hand)
        bot.react(Tsumo(actor=0, pai="1m"))
        bot.react(Ankan(actor=0, consumed=["1m"] * 4))
        assert len(bot._hand) == 10  # noqa: SLF001

    def test_other_players_melds_do_not_touch_my_hand(self) -> None:
        bot = started()
        bot.react(Pon(actor=1, target=2, pai="1z", consumed=["1z", "1z"]))
        assert len(bot._hand) == 13  # noqa: SLF001

    def test_calling_a_chi_immediately_requires_a_discard(self) -> None:
        """吃碰**不從牌山補牌**,鳴完直接輪到自己打 —— 建議要在鳴牌事件當下就給。

        先前只在 tsumo 時才給建議,鳴牌之後會整整少一手沒有任何輸出。
        """
        hand = ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "5p"]
        bot = started(hand)
        advice = bot.react(Chi(actor=0, target=3, pai="1m", consumed=["2m", "3m"]))
        assert advice.action is not None, "鳴牌之後應該立刻給打牌建議"
        assert advice.action.tsumogiri is False, "沒有摸牌,不可能是摸切"  # type: ignore[attr-defined]

    def test_a_meld_by_someone_else_needs_no_discard(self) -> None:
        chi = Chi(actor=2, target=1, pai="1m", consumed=["2m", "3m"])
        assert started().react(chi).action is None

    def test_it_still_advises_after_a_meld(self) -> None:
        """副露之後手牌變少,向聽計算靠張數反推副露組數 —— 不必另外記。"""
        hand = ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "5p"]
        bot = started(hand)
        bot.react(Chi(actor=0, target=3, pai="1m", consumed=["2m", "3m"]))
        bot.react(Dahai(actor=0, pai="9m", tsumogiri=False))
        assert bot.react(Tsumo(actor=0, pai="9s")).action is not None

    def test_a_new_kyoku_resets_the_hand(self) -> None:
        bot = started()
        bot.react(Tsumo(actor=0, pai="9s"))
        bot.react(kyoku(["1z"] * 4 + ["2z"] * 4 + ["3z"] * 4 + ["4z"]))
        assert bot._hand.count("1z") == 4  # noqa: SLF001

    def test_a_non_zero_seat_reads_its_own_tehai(self) -> None:
        """自己不一定坐 0 —— 讀錯 tehais 的索引會拿到別人的配牌。"""
        bot = DummyEngine()
        bot.start()
        bot.react(StartGame(id=2))
        tehais = [["?"] * 13, ["?"] * 13, TENPAI, ["?"] * 13]
        bot.react(StartKyoku("E", 1, 0, 0, 0, "2s", tehais, [25000] * 4))
        assert bot._hand == TENPAI  # noqa: SLF001


class TestRobustness:
    def test_reacting_before_start_is_an_error(self) -> None:
        with pytest.raises(EngineError, match="尚未啟動"):
            DummyEngine().react(Tsumo(actor=0, pai="1m"))

    def test_a_kyoku_without_a_game_is_an_error(self) -> None:
        """沒有 start_game 就不知道自己是誰,拿不到正確的 tehai。"""
        bot = DummyEngine()
        bot.start()
        with pytest.raises(EngineError, match="不知道自己是誰"):
            bot.react(kyoku(TENPAI))

    def test_an_unknown_tile_in_hand_yields_no_advice(self) -> None:
        """手牌含 ``?`` 時算不出向聽。回「不動作」而不是猜一張。"""
        bot = started(["?"] * 13)
        assert bot.react(Tsumo(actor=0, pai="1m")).action is None

    def test_a_broken_event_stream_does_not_crash(self) -> None:
        """打了一張手上沒有的牌 —— 事件流漏了。之後的建議不能信,但不該崩。"""
        bot = started()
        bot.react(Dahai(actor=0, pai="9s", tsumogiri=False))
        assert bot.react(Tsumo(actor=0, pai="9s")).action is not None
