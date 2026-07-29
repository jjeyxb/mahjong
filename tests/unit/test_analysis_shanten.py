"""向聽 / 進張 / 打牌建議測試。

期望值都是手算得出來的牌型 —— 刻意挑「一眼看得出答案」的牌,
這樣測試失敗時能立刻分辨是實作壞了還是期望值寫錯了。
"""

from __future__ import annotations

import pytest

from majsoul_copilot.analysis import (
    AGARI,
    TENPAI,
    HandError,
    analyse,
    suggest_discards,
)


def hand(spec: str) -> list[str]:
    """``"123m 456p 東"`` 這種寫法太容易寫錯,一律用空白分隔的完整牌名。"""
    return spec.split()


# 123m 456m 789m 123p + 5p 單騎
TENPAI_13 = hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p")
# 上面那手再摸一張 9s
DRAWN_14 = [*TENPAI_13, "9s"]


class TestShanten:
    def test_a_complete_hand_is_agari(self) -> None:
        # 123m 456m 789m 123p 55p
        assert analyse(hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p")).shanten == AGARI

    def test_tenpai_is_zero(self) -> None:
        result = analyse(TENPAI_13)
        assert result.shanten == TENPAI
        assert result.is_tenpai
        assert not result.is_agari

    def test_the_wait_is_reported_with_its_remaining_count(self) -> None:
        """單騎 5p:自己手上有一張,所以剩 3 張。"""
        result = analyse(TENPAI_13)
        assert [(u.tile, u.count) for u in result.ukeire] == [("5p", 3)]
        assert result.total_ukeire == 3

    def test_agari_has_no_ukeire(self) -> None:
        """已經和了就沒有「還要摸什麼」可言,回空的比回一堆沒意義的牌好。"""
        assert analyse(hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p")).ukeire == ()

    def test_seven_pairs_is_recognised(self) -> None:
        """七對子:六對 + 一張單張 = 聽牌。一般型算法給不出這個答案。"""
        result = analyse(hand("1m 1m 3m 3m 5m 5m 7m 7m 9m 9m 1p 1p 3p"))
        assert result.shanten == TENPAI

    def test_a_drawn_hand_reports_shanten_but_no_ukeire(self) -> None:
        """14 張時算進張會湊出 15 張,套件會拋錯 —— 而且那個問題本來就沒意義。

        這個 bug 是整條管線跑真實畫面時才炸出來的:單元測試裡唯一的 14 張
        牌型剛好是和了,提早跳過了進張計算,所以沒抓到。
        """
        result = analyse(DRAWN_14)
        assert result.shanten == TENPAI
        assert result.needs_discard
        assert result.ukeire == ()

    @pytest.mark.parametrize("count", [2, 5, 8, 11, 14])
    def test_every_drawn_count_survives_analysis(self, count: int) -> None:
        """五種「剛摸完」的張數各跑一次 —— 之前只有 14 張被測到。"""
        tiles = hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 9s")[-count:]
        assert analyse(tiles).needs_discard

    def test_a_waiting_hand_does_not_need_a_discard(self) -> None:
        assert not analyse(TENPAI_13).needs_discard

    def test_thirteen_orphans_is_recognised(self) -> None:
        """13 種么九各一張是**十三面聽牌**,不是和了 —— 和了要 14 張。

        一般型算法會說這手離聽牌很遠,所以這個測試同時證明國士型有被評估到。
        """
        result = analyse(hand("1m 9m 1p 9p 1s 9s 1z 2z 3z 4z 5z 6z 7z"))
        assert result.shanten == TENPAI
        assert len(result.ukeire) == 13, "十三面應該有 13 種進張"


class TestRedFives:
    def test_a_red_five_counts_as_a_normal_five(self) -> None:
        """赤五在向聽計算上與普通五完全等價 —— 它只影響打點。"""
        normal = analyse(hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p"))
        red = analyse(hand("1m 2m 3m 4m 0m 6m 7m 8m 9m 1p 2p 3p 5p"))
        assert normal.shanten == red.shanten

    def test_ukeire_is_reported_in_normal_notation(self) -> None:
        """進張回普通五 —— 「摸赤五才算」是錯的,兩種都能進張。"""
        result = analyse(hand("1m 2m 3m 4m 6m 7m 8m 9m 1p 2p 3p 5p 5p"))
        assert all(u.tile != "0m" for u in result.ukeire)

    def test_four_normal_fives_plus_a_red_five_is_rejected(self) -> None:
        """赤五也是那四張的其中一張,不是第五張。"""
        with pytest.raises(HandError, match="最多"):
            analyse(hand("5m 5m 5m 5m 0m 1p 2p 3p 4p 5p 6p 7p 8p"))


class TestMelds:
    @pytest.mark.parametrize(
        ("tiles", "melds"),
        [
            ("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p", 0),
            ("1m 2m 3m 4m 5m 6m 7m 8m 9m 5p", 1),
            ("1m 2m 3m 4m 5m 6m 5p", 2),
            ("1m 2m 3m 5p", 3),
            ("5p", 4),
        ],
    )
    def test_meld_count_comes_from_the_tile_count(self, tiles: str, melds: int) -> None:
        """副露**不需要當參數傳** —— 少掉的每 3 張就是一組面子。"""
        assert analyse(hand(tiles)).melds == melds

    def test_a_melded_hand_can_be_tenpai(self) -> None:
        """4 張暗牌 + 3 組副露:345m 聽 2m/5m 兩面。"""
        result = analyse(hand("2m 3m 4m 5m"))
        assert result.shanten == TENPAI
        assert {u.tile for u in result.ukeire} == {"2m", "5m"}

    def test_seven_pairs_cannot_be_claimed_with_melds(self) -> None:
        """10 張五對看起來像七對子一向聽,但有副露就不可能是七對子。

        ``mahjong`` 套件只在 13 張以上才評估七對子與國士,所以這件事
        結構上就不會發生 —— 這個測試釘住那個前提,套件換版時會被抓到。
        """
        result = analyse(hand("1m 1m 2m 2m 3p 3p 4p 4p 5s 5s"))
        assert result.shanten > 0, "有副露卻算出七對子聽牌"


class TestInvalidHands:
    @pytest.mark.parametrize("count", [3, 6, 9, 12])
    def test_multiples_of_three_are_rejected(self, count: int) -> None:
        """逢 3 的倍數代表切到一半或辨識抓在動畫中間,該重讀而不是硬算。"""
        with pytest.raises(HandError, match="合法的暗手牌張數"):
            analyse(["1m"] * count)

    def test_too_many_tiles_is_rejected(self) -> None:
        with pytest.raises(HandError):
            analyse(["1m"] * 15)

    def test_an_empty_hand_is_rejected(self) -> None:
        with pytest.raises(HandError):
            analyse([])

    @pytest.mark.parametrize("tile", ["10m", "8z", "0z", "1x", "m1", ""])
    def test_unknown_tiles_are_rejected(self, tile: str) -> None:
        with pytest.raises(HandError):
            analyse([tile, *["1m"] * 12])


class TestDiscards:
    def test_the_best_discard_reaches_tenpai(self) -> None:
        best = suggest_discards(DRAWN_14)[0]
        assert best.shanten == TENPAI

    def test_options_are_sorted_by_shanten_then_ukeire(self) -> None:
        options = suggest_discards(DRAWN_14)
        keys = [(o.shanten, -o.total_ukeire) for o in options]
        assert keys == sorted(keys)

    def test_each_distinct_tile_appears_once(self) -> None:
        """手上兩張一樣的牌,切哪張都一樣 —— 列兩次只是雜訊。"""
        tiles = hand("1m 1m 2m 2m 3m 3m 4m 4m 5m 5m 6m 6m 7m 7m")
        labels = [o.tile for o in suggest_discards(tiles)]
        assert len(labels) == len(set(labels))

    def test_no_discard_can_ever_be_agari(self) -> None:
        """切完只剩 13/10/7/4/1 張,而和了需要 14/11/8/5/2 張。

        這是個不變量,不是巧合 —— suggest_discards 因此不需要處理和了分支。
        測試釘住它,免得日後有人「順手」加回一段永遠跑不到的程式碼。
        """
        for tiles in (DRAWN_14, hand("2m 3m 4m 5m 9s"), hand("1m 1m")):
            assert all(o.shanten > AGARI for o in suggest_discards(tiles))

    def test_breaking_up_a_complete_hand_leaves_tenpai(self) -> None:
        """已經和了的 14 張,不管切哪張都還是聽牌 —— 沒有更差的選擇。"""
        complete = hand("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 5p 5p")
        assert analyse(complete).shanten == AGARI
        assert suggest_discards(complete)[0].shanten == TENPAI

    def test_waiting_counts_are_rejected(self) -> None:
        """13 張還沒摸牌,沒有「該切什麼」可言。"""
        with pytest.raises(HandError, match="要先摸牌"):
            suggest_discards(TENPAI_13)

    def test_works_with_melds(self) -> None:
        """5 張暗牌 + 3 組副露,摸完之後一樣要決定切什麼。"""
        options = suggest_discards(hand("2m 3m 4m 5m 9s"))
        assert options[0].tile == "9s"
        assert options[0].shanten == TENPAI


class TestUkeire:
    def test_a_tile_fully_held_is_not_counted(self) -> None:
        """四張都在自己手上就摸不到了,不能算進進張。"""
        result = analyse(hand("1m 1m 1m 1m 2m 3m 4m 5m 6m 7m 8m 9m 9m"))
        assert all(u.count > 0 for u in result.ukeire)
        assert all(u.tile != "1m" for u in result.ukeire)

    def test_a_two_sided_wait_beats_a_closed_wait(self) -> None:
        """兩面 8 枚 vs 坎張 4 枚 —— 進張枚數要能反映這個差別。"""
        two_sided = analyse(hand("2m 3m 1p 1p 1p 2p 2p 2p 3p 3p 3p 4p 4p"))
        closed = analyse(hand("2m 4m 1p 1p 1p 2p 2p 2p 3p 3p 3p 4p 4p"))
        assert two_sided.total_ukeire > closed.total_ukeire

    def test_ukeire_is_sorted_by_remaining_count(self) -> None:
        result = analyse(TENPAI_13)
        counts = [u.count for u in result.ukeire]
        assert counts == sorted(counts, reverse=True)
