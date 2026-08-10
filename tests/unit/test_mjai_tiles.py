"""牌表示法轉換測試。"""

from __future__ import annotations

import pytest

from mia.mjai.tiles import (
    HONOR_NAMES,
    HONOR_ORDER,
    UNKNOWN,
    TileError,
    is_red_five,
    mjai_to_ms,
    ms_to_mjai,
    normalize_red,
    sort_key,
    tile_name,
)


class TestMsToMjai:
    @pytest.mark.parametrize("tile", ["1m", "9m", "1p", "9p", "1s", "9s", "5m"])
    def test_number_tiles_unchanged(self, tile: str) -> None:
        assert ms_to_mjai(tile) == tile

    @pytest.mark.parametrize(("ms", "mjai"), [("0m", "5mr"), ("0p", "5pr"), ("0s", "5sr")])
    def test_red_fives(self, ms: str, mjai: str) -> None:
        assert ms_to_mjai(ms) == mjai

    def test_honors_follow_east_south_west_north_haku_hatsu_chun(self) -> None:
        assert [ms_to_mjai(f"{i + 1}z") for i in range(7)] == list(HONOR_ORDER)
        assert ms_to_mjai("1z") == "E"
        assert ms_to_mjai("7z") == "C"

    def test_unknown_passes_through(self) -> None:
        assert ms_to_mjai(UNKNOWN) == UNKNOWN

    @pytest.mark.parametrize("tile", ["", "0z", "8z", "10m", "5x", "m5", "5", "5mr"])
    def test_invalid_raises(self, tile: str) -> None:
        """壞牌一定要拋例外。回傳預設值會讓錯誤悄悄流進事件流,整局狀態全歪。"""
        with pytest.raises(TileError):
            ms_to_mjai(tile)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "tile", ["0m", "0p", "0s", "1m", "5p", "9s", "1z", "4z", "7z", UNKNOWN]
    )
    def test_ms_mjai_ms(self, tile: str) -> None:
        assert mjai_to_ms(ms_to_mjai(tile)) == tile

    def test_mjai_to_ms_rejects_garbage(self) -> None:
        with pytest.raises(TileError):
            mjai_to_ms("X")


class TestHelpers:
    def test_is_red_five_accepts_both_notations(self) -> None:
        assert is_red_five("0m") and is_red_five("5mr")
        assert not is_red_five("5m")
        assert not is_red_five("1z")

    def test_normalize_red(self) -> None:
        assert normalize_red("5mr") == "5m"
        assert normalize_red("5pr") == "5p"
        assert normalize_red("3p") == "3p"
        assert normalize_red("E") == "E"


class TestSortKey:
    """理牌順序。

    ``sorted()`` 直接排字串會把筒子插進萬子中間 —— 這個 bug 是把手牌畫出來
    才看到的,單元測試不會抱怨一個「排好序」的 list。
    """

    def test_suits_are_grouped_before_ranks_are_compared(self) -> None:
        hand = ["1m", "2m", "3m", "1p", "1p", "2p", "3p"]
        assert sorted(hand, key=sort_key) == ["1m", "2m", "3m", "1p", "1p", "2p", "3p"]

    def test_plain_string_sort_would_interleave(self) -> None:
        """釘住問題本身 —— 這是當初做錯的方式。"""
        hand = ["1m", "2m", "1p", "2p"]
        assert sorted(hand) == ["1m", "1p", "2m", "2p"]
        assert sorted(hand, key=sort_key) == ["1m", "2m", "1p", "2p"]

    def test_the_suit_order_is_man_pin_sou_then_honours(self) -> None:
        mixed = ["E", "1s", "1p", "1m"]
        assert sorted(mixed, key=sort_key) == ["1m", "1p", "1s", "E"]

    def test_honours_follow_the_conventional_order(self) -> None:
        """東南西北白發中,不是字母序。"""
        honours = ["C", "F", "P", "N", "W", "S", "E"]
        assert sorted(honours, key=sort_key) == ["E", "S", "W", "N", "P", "F", "C"]

    def test_a_red_five_sits_next_to_its_plain_five(self) -> None:
        """赤五點數與普通五相同,但兩者不能互換 —— 順序固定才不會每次更新都跳。"""
        hand = ["4m", "5m", "5mr", "6m"]
        assert sorted(hand, key=sort_key) == ["4m", "5mr", "5m", "6m"]

    def test_unknown_tiles_go_last(self) -> None:
        assert sorted(["?", "1m"], key=sort_key) == ["1m", "?"]

    def test_an_unrecognised_tile_does_not_raise(self) -> None:
        """排序是顯示用的,不該讓 UI 崩掉。"""
        assert sorted(["99z", "1m"], key=sort_key) == ["1m", "99z"]


class TestTileName:
    """給人看的牌名。``1z`` 與 ``E`` 都要在腦裡再 translate 一次才對得上畫面。"""

    def test_both_notations_give_the_same_name(self) -> None:
        assert tile_name("1z") == tile_name("E") == "東"

    def test_every_honour_has_a_name(self) -> None:
        names = [tile_name(f"{i}z") for i in range(1, 8)]
        assert names == ["東", "南", "西", "北", "白", "發", "中"]

    def test_the_two_notations_agree_on_all_seven(self) -> None:
        assert [tile_name(c) for c in HONOR_ORDER] == list(HONOR_NAMES)

    def test_number_tiles_are_left_alone(self) -> None:
        """點數本來就是阿拉伯數字,換成「三萬」反而比對不上畫面上的牌。"""
        assert tile_name("3m") == "3m"
        assert tile_name("0p") == "0p"
        assert tile_name("5sr") == "5sr"

    def test_an_unrecognised_tile_comes_back_unchanged(self) -> None:
        """這是顯示用的,不該讓 UI 因為多了一種牌就崩掉。"""
        assert tile_name("99z") == "99z"
        assert tile_name("?") == "?"
