"""切牌測試。

用真實截圖當 fixture 而不是合成影像 —— 這段程式碼要對付的正是真實畫面的雜訊
(索子的竹紋、萬子的筆畫、副露的透視角度),合成一張乾淨的色塊圖測不出任何
有意義的東西。fixture 是從 ``data/roi_ref/aligned/*.png`` 用 ``own_hand`` 這個
ROI 裁出來的。
"""

from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul_copilot.vision.tiles.segment import (
    TileStrip,
    find_drawn_index,
    segment_tiles,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"讀不到 fixture {name}"
    return image


@pytest.fixture
def hand14() -> np.ndarray:
    """13 張整理好的手牌 + 空一格 + 摸進來的第 14 張。"""
    return load("hand_14_with_draw.png")


@pytest.fixture
def hand13() -> np.ndarray:
    """13 張滿手,無副露、非自己的回合(所以沒有摸牌那張)。"""
    return load("hand_13_full.png")


@pytest.fixture
def hand4() -> np.ndarray:
    """4 張暗手牌 + 3 組副露。副露的斜面在下緣,不該被當成手牌切進來。"""
    return load("hand_4_with_melds.png")


class TestTileCount:
    def test_fourteen_tiles_including_the_drawn_one(self, hand14: np.ndarray) -> None:
        assert len(segment_tiles(hand14)) == 14

    def test_thirteen_tiles(self, hand13: np.ndarray) -> None:
        assert len(segment_tiles(hand13)) == 13

    def test_melds_are_not_counted_as_concealed_tiles(self, hand4: np.ndarray) -> None:
        """副露的斜面在下緣,找上緣時不該把它們算進來。"""
        assert len(segment_tiles(hand4, bevel="top")) == 4

    def test_bottom_bevel_finds_the_melds_instead(self, hand4: np.ndarray) -> None:
        """同一張影像換找下緣,切到的是副露 —— 這就是兩者的區分方式。"""
        melds = segment_tiles(hand4, bevel="bottom")
        assert len(melds) > 0
        # 副露在畫面右側,暗手牌在左側
        hand = segment_tiles(hand4, bevel="top")
        assert min(b.x for b in melds) > max(b.x for b in hand)


class TestGeometry:
    def test_tiles_are_evenly_spaced_and_equally_wide(self, hand14: np.ndarray) -> None:
        strip = segment_tiles(hand14)
        widths = [b.width for b in strip]
        assert max(widths) - min(widths) <= 2, f"寬度應該幾乎一致: {widths}"
        # 最後一段是摸牌的空隙,不列入均勻性檢查
        pitches = strip.pitches[:-1]
        assert max(pitches) - min(pitches) <= 2, f"間距應該均勻: {pitches}"

    def test_boxes_are_ordered_left_to_right_without_overlap(self, hand14: np.ndarray) -> None:
        boxes = segment_tiles(hand14).boxes
        for left, right in itertools.pairwise(boxes):
            assert left.right <= right.x

    def test_boxes_stay_inside_the_roi(self, hand14: np.ndarray) -> None:
        height, width = hand14.shape[:2]
        for box in segment_tiles(hand14):
            assert box.x >= 0 and box.right <= width
            assert box.y >= 0 and box.bottom <= height

    def test_crop_returns_a_single_tile(self, hand14: np.ndarray) -> None:
        strip = segment_tiles(hand14)
        tile = strip.crop(hand14, 0)
        assert tile.shape[0] == hand14.shape[0]
        assert tile.shape[1] == strip.boxes[0].width


class TestDrawnTile:
    def test_the_gap_identifies_the_drawn_tile(self, hand14: np.ndarray) -> None:
        strip = segment_tiles(hand14)
        assert find_drawn_index(strip) == 13  # 最後一張

    def test_no_drawn_tile_when_the_hand_is_evenly_spaced(self, hand13: np.ndarray) -> None:
        assert find_drawn_index(segment_tiles(hand13)) is None

    def test_too_few_tiles_to_judge(self) -> None:
        assert find_drawn_index(TileStrip((), (0, 0))) is None


class TestScaleIndependence:
    def test_same_tile_count_at_half_resolution(self, hand14: np.ndarray) -> None:
        """非 Retina 螢幕的擷取解析度只有一半,不能因此就切不出來。"""
        height, width = hand14.shape[:2]
        small = cv2.resize(hand14, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        assert len(segment_tiles(small)) == 14


class TestDegenerateInput:
    def test_empty_region_yields_no_tiles(self) -> None:
        """對手還沒副露時副露 ROI 就是一片桌面 —— 這是正常情況,不該拋例外。"""
        table_blue = np.full((200, 800, 3), (90, 60, 30), np.uint8)
        assert len(segment_tiles(table_blue)) == 0

    def test_grayscale_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="BGR"):
            segment_tiles(np.zeros((10, 10), np.uint8))
