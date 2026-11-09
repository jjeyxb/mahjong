"""別家手牌的牌背形狀。用合成畫面:藍桌布上照手牌那條線畫橘色的牌背。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mia.utils.geometry import Rect
from mia.vision.tiles.backs import _LINES, read_backs

TABLE = Rect(0, 0, 2560, 1440)
BLUE = (120, 60, 30)
ORANGE = (40, 150, 240)


def _canvas() -> np.ndarray:
    return np.full((TABLE.height, TABLE.width, 3), BLUE, np.uint8)


def _paint(image: np.ndarray, seat: str, spans: list[tuple[float, float]], width: int = 40) -> None:
    """沿這家的線,把 ``(起, 迄)``(線上的比例)那幾段塗成牌背。"""
    (fx0, fy0), (fx1, fy1) = _LINES[seat]
    start = np.array([fx0 * TABLE.width, fy0 * TABLE.height])
    end = np.array([fx1 * TABLE.width, fy1 * TABLE.height])
    along = end - start
    side = np.array([-along[1], along[0]]) / np.linalg.norm(along) * width / 2
    for a, b in spans:
        p, q = start + along * a, start + along * b
        corners = np.array([p - side, p + side, q + side, q - side]).round().astype(np.int32)
        cv2.fillPoly(image, [corners], ORANGE)


#: 13 張連成一排,摸到的那張隔一點空放在尾端。單位是線上的比例。
HAND = (0.02, 0.80)
DRAWN = (0.83, 0.88)


@pytest.mark.parametrize("seat", ["toimen", "kamicha"])
class TestAlongTheLine:
    def test_a_resting_hand(self, seat: str) -> None:
        image = _canvas()
        _paint(image, seat, [HAND])
        shape = read_backs(image, TABLE, seat)
        assert not shape.drawn
        assert shape.pieces == 1
        assert shape.body > 250

    def test_the_drawn_tile_is_apart(self, seat: str) -> None:
        image = _canvas()
        _paint(image, seat, [HAND, DRAWN])
        shape = read_backs(image, TABLE, seat)
        assert shape.drawn
        assert shape.pieces == 1

    def test_a_tile_taken_from_the_middle(self, seat: str) -> None:
        """手切:中間空一格,摸到的那張還在。本體比原本短。"""
        whole = _canvas()
        _paint(whole, seat, [HAND, DRAWN])
        holed = _canvas()
        _paint(holed, seat, [(0.02, 0.40), (0.46, 0.80), DRAWN])
        before, after = read_backs(whole, TABLE, seat), read_backs(holed, TABLE, seat)
        assert after.drawn
        assert after.clean
        assert after.body < before.body - 10


def test_sharp_seams_between_tiles_do_not_split_the_hand() -> None:
    """實打的原始畫面牌與牌之間有一條淡色的細縫,錄影的 JPEG 把它糊掉了。門檻是在錄影上調的,
    沒先把顏色降成一半解析度的話,上家的手牌會在縫上斷開。"""
    image = _canvas()
    _paint(image, "kamicha", [HAND])
    (fx0, fy0), (fx1, fy1) = _LINES["kamicha"]
    start = np.array([fx0 * TABLE.width, fy0 * TABLE.height])
    along = np.array([fx1 * TABLE.width, fy1 * TABLE.height]) - start
    side = np.array([-along[1], along[0]]) / np.linalg.norm(along) * 20
    for tile in range(1, 13):
        seam = start + along * (HAND[0] + (HAND[1] - HAND[0]) * tile / 13)
        a, b = (seam - side).round().astype(int), (seam + side).round().astype(int)
        cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (150, 170, 190), 1)
    shape = read_backs(image, TABLE, "kamicha")
    assert shape.pieces == 1
    assert not shape.drawn


class TestShimochaInPieces:
    def test_one_piece_at_rest(self) -> None:
        image = _canvas()
        _paint(image, "shimocha", [HAND], width=60)
        shape = read_backs(image, TABLE, "shimocha")
        assert shape.pieces == 1

    def test_a_hole_makes_two_pieces(self) -> None:
        image = _canvas()
        _paint(image, "shimocha", [(0.02, 0.40), (0.48, 0.88)], width=60)
        assert read_backs(image, TABLE, "shimocha").pieces == 2

    def test_an_empty_table_has_nothing(self) -> None:
        assert read_backs(_canvas(), TABLE, "shimocha").body == 0


class TestWithMelds:
    """副露從線的另一端往回排。暗槓的牌背、下家副露的橘色側面不能算成手牌。"""

    #: 三組副露,手牌剩 4 張;暗槓的兩張牌背落在線的後段。
    SHORT = (0.02, 0.26)
    SHORT_DRAWN = (0.28, 0.33)
    ANKAN = ((0.59, 0.65), (0.80, 0.86))

    @pytest.mark.parametrize("seat", ["toimen", "kamicha"])
    def test_the_ankan_backs_are_not_the_drawn_tile(self, seat: str) -> None:
        image = _canvas()
        _paint(image, seat, [self.SHORT, *self.ANKAN])
        shape = read_backs(image, TABLE, seat, melds=3)
        assert not shape.drawn
        assert shape.pieces == 1

    @pytest.mark.parametrize("seat", ["toimen", "kamicha"])
    def test_the_drawn_tile_is_still_seen(self, seat: str) -> None:
        image = _canvas()
        _paint(image, seat, [self.SHORT, self.SHORT_DRAWN, *self.ANKAN])
        shape = read_backs(image, TABLE, seat, melds=3)
        assert shape.drawn
        assert shape.pieces == 1

    def test_shimocha_meld_sides_are_not_a_second_piece(self) -> None:
        image = _canvas()
        _paint(image, "shimocha", [self.SHORT, (0.66, 0.74)], width=60)
        assert read_backs(image, TABLE, "shimocha").pieces == 2
        assert read_backs(image, TABLE, "shimocha", melds=3).pieces == 1

    @pytest.mark.parametrize("seat", ["shimocha", "toimen", "kamicha"])
    def test_without_melds_nothing_changes(self, seat: str) -> None:
        image = _canvas()
        _paint(image, seat, [HAND, DRAWN], width=60 if seat == "shimocha" else 40)
        assert read_backs(image, TABLE, seat, melds=0) == read_backs(image, TABLE, seat)
