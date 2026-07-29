"""牌表示法轉換測試。"""

from __future__ import annotations

import pytest

from majsoul_copilot.mjai.tiles import (
    HONOR_ORDER,
    UNKNOWN,
    TileError,
    is_red_five,
    mjai_to_ms,
    ms_to_mjai,
    normalize_red,
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
