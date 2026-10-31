"""副露帶的格線與連續性檢查。

第一階段只問「這一格是不是翻開的牌」,所以這裡測的是帶子的切法與棄權規則,
認牌面(第二階段)的端到端測試在 ``test_live_table``。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.melds import (
    _RED_MELD,
    CELL,
    COLUMNS,
    DEFAULT_VIEWS,
    EXTENT,
    GROUP_LENGTH,
    MeldStrip,
    MeldView,
    _red_score,
    occupied_length,
    read_melds,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TABLE = Rect(0, 0, 1000, 1000)
#: 整個牌桌都是副露區,方便直接畫格子進去。
WHOLE = MeldView(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))


def _table(faces: int) -> np.ndarray:
    """一張 1000x1000 的「桌子」,左邊 ``faces`` 格畫成翻開的牌。"""
    image = np.full((1000, 1000, 3), (90, 60, 30), dtype=np.uint8)  # 暗藍桌布
    for i in range(faces):
        x0 = round(i * 1000 / COLUMNS)
        x1 = round((i + 1) * 1000 / COLUMNS) - 2
        image[:, x0:x1] = (235, 240, 240)  # 象牙色牌面
    return image


class TestTheStrip:
    def test_it_counts_the_face_up_tiles(self) -> None:
        strip = read_melds(_table(9), TABLE, WHOLE)
        assert strip.count == 9
        assert strip.filled == tuple(range(9))

    def test_an_empty_meld_area_reads_zero(self) -> None:
        assert read_melds(_table(0), TABLE, WHOLE).count == 0

    def test_the_strip_is_long_enough_for_four_kans(self) -> None:
        """只對**加槓**成立:加的那張疊在橫放那張上面,一組仍是三張寬。

        暗槓與大明槓會更寬(``docs/cv_mjai.md`` §3.7),那由
        :func:`occupied_length` 的 EXTENT 處理;數格子這條路只保證碰與加槓。
        """
        assert COLUMNS >= 4 * 3


class TestItAbstainsRatherThanGuess:
    """副露從桌角往外長,中間不會有洞。有洞就是攤平歪了或被蓋住。"""

    def test_a_contiguous_run_is_plausible(self) -> None:
        assert MeldStrip((True, True, True, False, False)).is_plausible()

    def test_a_hole_means_the_whole_seat_is_dropped(self) -> None:
        assert not MeldStrip((True, False, True, False)).is_plausible()

    def test_nothing_at_all_is_plausible(self) -> None:
        """還沒有人鳴牌是正常狀態,不是讀失敗。"""
        assert MeldStrip((False,) * COLUMNS).is_plausible()


class TestTheMeasuredCorners:
    def test_all_four_seats_are_measured(self) -> None:
        """下家原本 0 樣本,自己原本以為跟手牌分不開 —— 牌譜換視角補上了。"""
        assert set(DEFAULT_VIEWS) == {"own", "shimocha", "toimen", "kamicha"}

    def test_every_seat_has_a_group_length(self) -> None:
        """長度換組數各用各的組長(角點框的比例不同)。"""
        assert set(GROUP_LENGTH) == set(DEFAULT_VIEWS)
        assert all(2.0 < unit < 5.0 for unit in GROUP_LENGTH.values())

    def test_the_corners_stay_on_the_table(self) -> None:
        for name, view in DEFAULT_VIEWS.items():
            for x, y in view.corners:
                assert 0.0 <= x <= 1.0, name
                assert 0.0 <= y <= 1.0, name

    def test_the_cell_is_small_enough_to_be_cheap(self) -> None:
        """只判有無,不比對模板 —— 一整條帶子要比牌河一格還便宜。"""
        assert CELL[0] * CELL[1] * COLUMNS < 90 * 120 * 6 * 3


#: 量長度用的帶子:13 格的角點框住整張桌子,外插到 EXTENT 格。
#: 所以桌上的 1000 px 對應 13 格,一格約 77 px。
_UNIT = 1000 / COLUMNS
FACE = (235, 240, 240)
BACK = (40, 140, 240)  # 橘色牌背(BGR)


def _strip(*runs: tuple[float, float, tuple[int, int, int]]) -> np.ndarray:
    """畫幾段牌,每段是 (起點格, 長度格, 顏色)。"""
    image = np.full((1000, 1000, 3), (90, 60, 30), dtype=np.uint8)
    for start, length, colour in runs:
        image[:, round(start * _UNIT) : round((start + length) * _UNIT)] = colour
    return image


class TestOccupiedLength:
    """副露一多,等寬切格會切偏、暗槓會有洞;量長度兩樣都不怕。"""

    def test_nothing_is_zero(self) -> None:
        assert occupied_length(_strip(), TABLE, WHOLE) == 0

    def test_it_measures_from_the_start(self) -> None:
        assert abs(occupied_length(_strip((0, 3.33, FACE)), TABLE, WHOLE) - 3.33) < 0.05

    def test_the_seam_between_groups_is_not_the_end(self) -> None:
        """組與組之間有一條細縫。"""
        image = _strip((0, 3.3, FACE), (3.4, 3.3, FACE))
        assert abs(occupied_length(image, TABLE, WHOLE) - 6.7) < 0.05

    def test_an_ankan_counts_its_face_down_ends(self) -> None:
        """暗槓是背、正、正、背 —— is_face_up 會在兩端各挖一個洞。"""
        image = _strip((0, 1, BACK), (1, 2, FACE), (3, 1, BACK))
        assert abs(occupied_length(image, TABLE, WHOLE) - 4) < 0.05

    def test_a_gap_ends_it(self) -> None:
        """結尾之後的東西(例如牌山)不算。"""
        image = _strip((0, 3, FACE), (5, 3, BACK))
        assert abs(occupied_length(image, TABLE, WHOLE) - 3) < 0.05

    def test_four_daiminkan_fit(self) -> None:
        """四組大明槓約 17.3 張寬,13 格裝不下。"""
        assert EXTENT * 1.0 >= 4 * (3 + 1.33)


class TestRedFiveInAMeld:
    """顏色不必認牌:赤五常常是模板認不到的那一張。素材都是實際畫面裁下來的。"""

    @staticmethod
    def _score(name: str) -> float:
        plane = cv2.imread(str(FIXTURES / f"{name}.png"))
        height, width = plane.shape[:2]
        inside = np.array(
            [[45, 45], [width - 45, 45], [width - 45, height - 45], [45, height - 45]],
            np.float32,
        )
        return _red_score(plane, inside)

    def test_a_red_five_man(self) -> None:
        assert self._score("meld_red_5m_own") >= _RED_MELD["m"]

    def test_a_red_five_sou_on_a_far_seat(self) -> None:
        assert self._score("meld_red_5s_shimocha") >= _RED_MELD["s"]

    def test_a_plain_five_man(self) -> None:
        assert self._score("meld_plain_5m_own") < _RED_MELD["m"]

    def test_the_red_half_of_man_tiles_is_not_a_red_tile(self) -> None:
        """萬字本來就是紅的,框滑到只蓋住牌的下半部時紅墨到得了 0.87。"""
        assert self._score("meld_plain_123m_toimen") < _RED_MELD["m"]
