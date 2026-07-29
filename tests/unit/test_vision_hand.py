"""手牌讀取測試。

用真實截圖當 fixture 而不是合成影像 —— 這段程式碼要對付的正是真實畫面的雜訊
(索子的竹紋、白板的整片空白、副露的透視角度),合成一張乾淨的色塊圖測不出
任何有意義的東西。fixture 是用 ``own_hand`` 這個 ROI 從真實對局畫面裁出來的。
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul_copilot.vision.tiles.hand import (
    DEFAULT_LAYOUT,
    MAX_CONCEALED,
    Hand,
    HandLayout,
    read_hand,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"讀不到 fixture {name}"
    return image


@pytest.fixture
def hand14() -> np.ndarray:
    """13 張整理好的手牌 + 空一格 + 摸進來的第 14 張(是白板)。"""
    return load("hand_14_with_draw.png")


@pytest.fixture
def hand13() -> np.ndarray:
    """13 張滿手,無副露、非自己的回合(所以沒有摸牌那張)。"""
    return load("hand_13_full.png")


@pytest.fixture
def hand4() -> np.ndarray:
    """4 張暗手牌 + 3 組副露。副露在畫面右側,不該被算進暗手牌。"""
    return load("hand_4_with_melds.png")


class TestCounting:
    def test_full_hand(self, hand13: np.ndarray) -> None:
        hand = read_hand(hand13)
        assert len(hand) == 13
        assert hand.drawn is None
        assert hand.total == 13

    def test_hand_with_a_drawn_tile(self, hand14: np.ndarray) -> None:
        hand = read_hand(hand14)
        assert len(hand) == 13, "摸牌那張不該混進暗手牌"
        assert hand.drawn is not None
        assert hand.total == 14

    def test_the_drawn_tile_here_is_a_blank_white_dragon(
        self, hand14: np.ndarray
    ) -> None:
        """這張 fixture 的第 14 張是白板,牌面整片空白。

        第一版的佔用判據只取牌面中段,在這裡量到標準差 0.0,直接把它判成空槽。
        含上下緣之後靠的是牌緣結構,與牌面圖案無關 —— 這個測試盯住那個回歸。
        """
        hand = read_hand(hand14)
        assert hand.drawn is not None
        centre = cv2.cvtColor(
            hand14[int(207 * 0.35) : int(207 * 0.85), hand.drawn.x + 20 : hand.drawn.x + 100],
            cv2.COLOR_BGR2GRAY,
        )
        assert centre.std() < 1.0, "這張 fixture 的第 14 張本來就該是空白牌面"

    def test_melds_are_not_counted_as_concealed_tiles(self, hand4: np.ndarray) -> None:
        """掃到第一個空槽就停 —— 副露在更右邊,不能跨過中間的空桌面把它們算進來。"""
        hand = read_hand(hand4)
        assert len(hand) == 4
        assert hand.drawn is None


class TestMeldCount:
    def test_meld_count_is_derived_from_the_tile_count(self, hand4: np.ndarray) -> None:
        """副露的**內容**不需要辨識 —— 向聽數只要組數,而組數由張數反推。"""
        assert read_hand(hand4).melds == 3

    def test_no_melds_when_the_hand_is_full(self, hand13: np.ndarray) -> None:
        assert read_hand(hand13).melds == 0

    @pytest.mark.parametrize(
        ("concealed", "melds"), [(13, 0), (10, 1), (7, 2), (4, 3), (1, 4)]
    )
    def test_every_legal_hand_length(self, concealed: int, melds: int) -> None:
        hand = Hand(tuple(DEFAULT_LAYOUT.slot(1813, 207, i) for i in range(concealed)))
        assert hand.melds == melds
        assert hand.is_plausible

    @pytest.mark.parametrize("concealed", [2, 3, 5, 6, 8, 9, 11, 12])
    def test_impossible_lengths_are_flagged(self, concealed: int) -> None:
        """張數不是 13/10/7/4/1 就代表這一幀讀錯了(通常抓在理牌動畫中間)。

        回傳 -1 而不是拋錯,是因為這在即時串流裡是**預期會發生**的常態,
        呼叫端該做的是丟掉這一幀,不是處理例外。
        """
        hand = Hand(tuple(DEFAULT_LAYOUT.slot(1813, 207, i) for i in range(concealed)))
        assert hand.melds == -1
        assert not hand.is_plausible

    def test_an_empty_hand_is_not_plausible(self) -> None:
        assert not Hand(()).is_plausible


class TestGeometry:
    def test_slots_are_evenly_spaced(self) -> None:
        boxes = [DEFAULT_LAYOUT.slot(1813, 207, i) for i in range(MAX_CONCEALED)]
        pitches = {b.x - a.x for a, b in pairwise(boxes)}
        assert pitches <= {126, 127}, f"間距應該均勻,量到 {sorted(pitches)}"

    def test_slot_positions_match_the_measured_values(self) -> None:
        """實測 13 張滿手時每張牌的左緣。槽位模型偏了就會在這裡被抓到。"""
        expected = [4, 131, 257, 384, 510, 637, 763, 890, 1016, 1143, 1269, 1396, 1522]
        actual = [DEFAULT_LAYOUT.slot(1813, 207, i).x for i in range(MAX_CONCEALED)]
        assert actual == expected

    def test_the_drawn_slot_sits_past_the_gap(self) -> None:
        """摸牌那張比下一個槽位再往右 gap 那麼多 —— 那個空隙就是它的識別特徵。"""
        next_slot = DEFAULT_LAYOUT.slot(1813, 207, 13)
        drawn = DEFAULT_LAYOUT.drawn_slot(1813, 207, 13)
        assert drawn.x - next_slot.x == 39

    def test_the_drawn_slot_moves_with_the_hand_length(self) -> None:
        """四組副露時暗手牌只剩 1 張,摸的那張就跟著往左跑 —— 沒有固定座標。"""
        no_melds = DEFAULT_LAYOUT.drawn_slot(1813, 207, 13)
        four_melds = DEFAULT_LAYOUT.drawn_slot(1813, 207, 1)
        assert four_melds.x < no_melds.x
        # 剛好差 12 個槽位(13 張暗牌 vs 1 張)
        assert no_melds.x - four_melds.x == round(12 * DEFAULT_LAYOUT.pitch * 1813)

    def test_geometry_scales_with_the_roi(self) -> None:
        """同一份幾何要能適用於不同視窗大小 —— 這就是用比例儲存的理由。"""
        small = DEFAULT_LAYOUT.slot(1813, 207, 7)
        large = DEFAULT_LAYOUT.slot(3626, 414, 7)
        assert large.x == pytest.approx(small.x * 2, abs=1)
        assert large.width == pytest.approx(small.width * 2, abs=1)


class TestRobustness:
    def test_works_at_half_resolution(self, hand14: np.ndarray) -> None:
        """視窗縮一半時張數要照樣讀對。"""
        small = cv2.resize(hand14, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        hand = read_hand(small)
        assert len(hand) == 13
        assert hand.drawn is not None

    def test_an_empty_table_yields_no_tiles(self, hand4: np.ndarray) -> None:
        """整片桌面(對局還沒開始)要回傳空手牌,不是報錯。"""
        empty = np.full_like(hand4, hand4[100, 900])  # 取一塊真實的桌面顏色鋪滿
        hand = read_hand(empty)
        assert len(hand) == 0
        assert hand.drawn is None

    def test_a_custom_layout_is_honoured(self, hand13: np.ndarray) -> None:
        """換皮膚若改了牌的尺寸,只要換 layout,不必動演算法。"""
        doubled = HandLayout(origin=DEFAULT_LAYOUT.origin, pitch=DEFAULT_LAYOUT.pitch * 2,
                             width=DEFAULT_LAYOUT.width, draw_gap=DEFAULT_LAYOUT.draw_gap)
        assert len(read_hand(hand13, layout=doubled)) < 13

    def test_a_greyscale_image_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="BGR"):
            read_hand(np.zeros((10, 10), np.uint8))
