"""把 Mortal 的 Q 值對應回動作。

標籤錯開一格的話,UI 會顯示「建議切 3m」而引擎說的其實是 4m —— 那種錯誤在
畫面上完全看不出來,所以這裡的期望值全部取自**真實引擎輸出**,不是自己造的。
"""

from __future__ import annotations

import pytest

from mia.engine.actions import (
    ACTION_SPACE,
    TILE_ACTIONS,
    action_label,
    action_tiles,
    decode_candidates,
    is_decision,
    legal_count,
)
from mia.mjai import Chi, Dahai, Kakan, Pon, Reach

#: 2026-07-30 實測:手牌 123456789m 123p 5p 摸 9s,引擎回 reach。
#: 合法動作恰為那 14 張牌 + 立直,mask 的位元完全對應。
REACH_META = {
    "mask_bits": 137506074623,
    "q_values": [
        -7.349411, -7.4835396, -7.2532053, -6.0267124, -6.3264084, -7.1490574,
        -6.0205574, -6.8105097, -7.011588, -6.3229656, -7.6495876, -7.3316984,
        -1.4386911, -1.1807599, 1.2769496,
    ],
}

#: 同日實測:上家打 3m,手上三種吃法 + 一對 3m3m,引擎回 none。
CALL_META = {
    "mask_bits": (1 << 38) | (1 << 39) | (1 << 40) | (1 << 41) | (1 << 45),
    "q_values": [-6.848, -5.228, -3.789, -4.055, 0.063],
}


class TestActionSpace:
    def test_the_label_table_covers_the_whole_space(self) -> None:
        from mia.engine.actions import _LABELS

        assert len(_LABELS) == ACTION_SPACE == 46

    def test_the_first_thirty_seven_are_tiles(self) -> None:
        """34 種牌 + 3 種赤五。第 37 個開始才是立直、吃碰這些。"""
        assert TILE_ACTIONS == 37


@pytest.fixture(scope="module")
def candidates():
    """實測的立直局面解出來的候選。"""
    return decode_candidates(REACH_META)


class TestDecodeReach:
    def test_every_legal_action_gets_a_candidate(self, candidates) -> None:
        assert len(candidates) == 15

    def test_the_chosen_action_is_the_highest(self, candidates) -> None:
        """引擎實際回的是 reach,而它就是最高分那個。"""
        assert candidates[0].label == "reach"
        assert candidates[0].chosen

    def test_exactly_one_candidate_is_chosen(self, candidates) -> None:
        assert sum(c.chosen for c in candidates) == 1

    def test_results_are_sorted_by_q_descending(self, candidates) -> None:
        values = [c.q for c in candidates]
        assert values == sorted(values, reverse=True)

    def test_the_hand_tiles_are_labelled_correctly(self, candidates) -> None:
        """手牌是 123456789m 123p 5p 摸 9s —— 切牌選項應該恰好是這 14 張。"""
        tiles = {c.label for c in candidates if c.is_tile}
        expected = {f"{n}m" for n in range(1, 10)} | {"1p", "2p", "3p", "5p", "9s"}
        assert tiles == expected

    def test_reach_is_not_a_tile(self, candidates) -> None:
        reach = next(c for c in candidates if c.label == "reach")
        assert not reach.is_tile
        assert reach.display == "立直"


class TestDecodeCalls:
    def test_chi_pon_and_none_are_labelled(self) -> None:
        """38/39/40 是吃的三種、41 是碰、45 是跳過 —— 全部實測確認過。"""
        candidates = decode_candidates(CALL_META)
        labels = [c.label for c in candidates]
        assert labels.count("chi") == 3
        assert "pon" in labels
        assert "none" in labels

    def test_none_wins_here(self) -> None:
        """引擎實際回的是 none。"""
        assert decode_candidates(CALL_META)[0].label == "none"

    def test_calls_display_in_chinese(self) -> None:
        displays = {c.display for c in decode_candidates(CALL_META)}
        assert {"吃", "碰", "跳過"} <= displays


class TestRejection:
    def test_no_meta_gives_nothing(self) -> None:
        """規則式 baseline 沒有 meta,那不是錯誤。"""
        assert decode_candidates(None) == ()
        assert decode_candidates({}) == ()

    def test_a_length_mismatch_gives_nothing(self) -> None:
        """位元數與 Q 值個數不符,代表對引擎輸出的理解有誤。

        勉強 zip 會產生錯開一格的標籤 —— UI 上完全看不出來,所以寧可什麼都不顯示。
        """
        assert decode_candidates({"mask_bits": 0b111, "q_values": [1.0]}) == ()

    def test_missing_fields_give_nothing(self) -> None:
        assert decode_candidates({"mask_bits": 0b1}) == ()
        assert decode_candidates({"q_values": [1.0]}) == ()

    def test_wrong_types_give_nothing(self) -> None:
        assert decode_candidates({"mask_bits": "7", "q_values": [1.0]}) == ()

    def test_bits_beyond_the_action_space_are_ignored(self) -> None:
        """遮罩若出現不該有的高位元,那是理解錯了 —— 長度就會對不上而被擋掉。"""
        assert decode_candidates({"mask_bits": 1 << 60, "q_values": [1.0]}) == ()


class TestDecisionPoints:
    """「引擎回 none」有兩種完全不同的意思,靠合法動作數才分得開。

    這一段釘住一個實測到的 bug:遊戲跳出「碰 / 槓 / 跳過」三個按鈕,而畫面上
    還掛著上一巡那個已經打掉的切牌建議 —— 使用者要的答案(該不該鳴)恰好就是
    被丟掉的那一個。
    """

    def test_no_meta_means_no_decision(self) -> None:
        """規則式 baseline 沒有 meta。它本來就不鳴牌,不該被算成決策點。"""
        assert not is_decision(None)
        assert not is_decision({})
        assert legal_count(None) == 0

    def test_a_single_legal_action_is_not_a_decision(self) -> None:
        """立直之後強制摸切:只有一個選擇,沒有什麼要決定的。"""
        assert legal_count({"mask_bits": 1 << 5}) == 1
        assert not is_decision({"mask_bits": 1 << 5})

    def test_more_than_one_legal_action_is_a_decision(self) -> None:
        """上家打牌、手上有一對可以碰 —— 遮罩含碰(41)與跳過(45)。"""
        meta = {"mask_bits": (1 << 41) | (1 << 45)}
        assert legal_count(meta) == 2
        assert is_decision(meta)

    def test_the_pon_kan_skip_case_from_the_screenshot(self) -> None:
        """實測的那一手:碰 -6.30、槓 -6.55、跳過 -0.15。"""
        meta = {"mask_bits": (1 << 41) | (1 << 42) | (1 << 45)}
        assert legal_count(meta) == 3
        assert is_decision(meta)

    def test_a_zero_mask_is_not_a_decision(self) -> None:
        """別人在摸打,輪不到我 —— 一場東風戰裡有 469 個這種事件。"""
        assert not is_decision({"mask_bits": 0})

    def test_a_garbage_mask_does_not_raise(self) -> None:
        """解錯了要安靜地當成「沒有決策」,不要讓 UI 崩掉。"""
        for mask in ("八", None, -1, 3.5, [1]):
            assert legal_count({"mask_bits": mask}) == 0

    def test_the_skip_option_is_labelled_like_the_game_button(self) -> None:
        """候選清單裡的那一列要與遊戲上「跳過」那顆按鈕用同一個詞。"""
        meta = {"mask_bits": (1 << 41) | (1 << 45), "q_values": [-4.22, -0.13]}
        labels = [c.display for c in decode_candidates(meta)]
        assert labels == ["跳過", "碰"]


class TestCallLabels:
    """鳴牌要寫出「用手上哪幾張」。

    ``吃 3m`` 可以是 1m2m、2m4m 或 4m5m 三種吃法。只寫「吃 3m」的話:

    * 使用者不知道該點哪兩張(實機打一場才發現);
    * 兩個引擎選了不同吃法會被當成一致 —— action_label 也用來判斷分歧。
    """

    def test_chi_says_which_two_tiles_to_use(self) -> None:
        label = action_label(Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]))
        assert label == "吃 3m ← 1m 2m"

    def test_the_three_chi_variants_get_different_labels(self) -> None:
        """這是分歧判斷的前提:不同動作必須給出不同字串。"""
        variants = [["1m", "2m"], ["2m", "4m"], ["4m", "5m"]]
        labels = {
            action_label(Chi(actor=0, target=3, pai="3m", consumed=c)) for c in variants
        }
        assert len(labels) == 3

    def test_pon_says_which_two_tiles_to_use(self) -> None:
        """赤五用不用掉是兩個不同的決定。"""
        plain = action_label(Pon(actor=0, target=1, pai="5m", consumed=["5m", "5m"]))
        red = action_label(Pon(actor=0, target=1, pai="5m", consumed=["5mr", "5m"]))
        assert plain != red

    def test_kakan_does_not_repeat_itself(self) -> None:
        """加槓只從手上拿一張,就是被槓那張本身 —— 寫出來是重複的。"""
        label = action_label(Kakan(actor=0, pai="5p", consumed=["5pr", "5p", "5p"]))
        assert label == "槓 5p"

    def test_a_discard_is_unchanged(self) -> None:
        assert action_label(Dahai(actor=0, pai="E", tsumogiri=False)) == "切 E"


class TestActionTiles:
    """要畫成牌面圖的牌:``(動作在講的那張, 自己手上要拿出來的那幾張)``。"""

    def test_a_discard_has_no_own_tiles(self) -> None:
        assert action_tiles(Dahai(actor=0, pai="3s", tsumogiri=False)) == ("3s", ())

    def test_chi_separates_the_called_tile_from_the_hand_tiles(self) -> None:
        """桌上那張與自己手上那兩張要分得開 —— 三張一樣大小排在一起看不出來。"""
        subject, own = action_tiles(Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]))
        assert subject == "3m"
        assert own == ("1m", "2m")

    def test_pon_too(self) -> None:
        subject, own = action_tiles(Pon(actor=0, target=1, pai="P", consumed=["P", "P"]))
        assert (subject, own) == ("P", ("P", "P"))

    def test_reach_has_no_tiles_of_its_own(self) -> None:
        """立直本身不牽涉某一張牌 —— 之後要切的那張走 Advice.follow_up。"""
        assert action_tiles(Reach(actor=0)) == (None, ())
