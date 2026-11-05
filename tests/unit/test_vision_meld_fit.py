"""拿候選比副露的畫面。理由見 ``mia/vision/tiles/meld_fit.py``。

真實畫面的評估在 ``docs/cv_mjai.md``(十四份錄影 156 組);這裡把模板畫進攤平後的影像,
測排法、選法與赤五,不必帶整張截圖進 repo。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet
from mia.vision.tiles.meld_fit import SIDEWAYS, MeldPicture, pick
from mia.vision.tiles.melds import _SIZES

TABLE = Rect(0, 0, 2560, 1440)
SEAT = "toimen"
PITCH = 80
FELT = (90, 60, 30)


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    return TemplateSet.load_river()


def _picture(templates: TemplateSet, tiles: list[str], sideways: int) -> MeldPicture:
    """一組畫在桌布上的副露,由左而右 ``tiles``(雀魂記法),第 ``sideways`` 張橫放。"""
    picture = MeldPicture(np.zeros((1440, 2560, 3), np.uint8), TABLE, SEAT, end=6.0)
    picture.image[:] = FELT
    size = _SIZES[SEAT][0]
    centres = picture.layout(len(tiles), sideways, (0.0, PITCH, 20.0))
    for index, (label, (cx, cy)) in enumerate(zip(tiles, centres, strict=True)):
        face = next(face for name, face in templates.scaled(size) if name == label)
        if index == sideways:
            face = cv2.rotate(face, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width = face.shape[:2]
        x, y = round(cx - width / 2), round(cy - height / 2)
        picture.image[y : y + height, x : x + width] = face
    return picture


class TestPick:
    def test_the_opposite_seat_uses_five_and_seven_to_chi_six(
        self, templates: TemplateSet
    ) -> None:
        picture = _picture(templates, ["6s", "5s", "7s"], 0)
        shapes = [
            (["6s", "6s", "6s"], 0),
            (["6s", "4s", "5s"], 0),
            (["6s", "5s", "7s"], 0),
            (["6s", "7s", "8s"], 0),
        ]
        found = pick(picture, shapes, templates)
        assert found is not None
        assert found.index == 2
        assert found.quality == pytest.approx(picture.fit(*shapes[2], templates).score / 3)

    def test_it_picks_the_run_that_is_there(self, templates: TemplateSet) -> None:
        """吃 3 萬,手上出 2 4 萬:同一張被鳴的牌,另外三種排法都不像。"""
        picture = _picture(templates, ["3m", "2m", "4m"], 0)
        shapes = [
            (["3m", "3m", "3m"], 0),
            (["3m", "1m", "2m"], 0),
            (["3m", "2m", "4m"], 0),
            (["3m", "4m", "5m"], 0),
        ]
        found = pick(picture, shapes, templates)
        assert found is not None
        assert found.index == 2
        assert found.reds == (False, False, False)

    def test_a_pon_from_across_lies_in_the_middle(self, templates: TemplateSet) -> None:
        picture = _picture(templates, ["7p", "7p", "7p"], 1)
        shapes = [(["7p", "7p", "7p"], 1), (["7p", "7p", "7p"], 0)]
        found = pick(picture, shapes, templates)
        assert found is not None
        assert found.index == 0

    def test_a_red_five_in_the_hand(self, templates: TemplateSet) -> None:
        picture = _picture(templates, ["6s", "4s", "0s"], 0)
        found = pick(picture, [(["6s", "4s", "5s"], 0)], templates)
        assert found is not None
        assert found.reds == (False, False, True)

    def test_a_plain_five_is_not_red(self, templates: TemplateSet) -> None:
        picture = _picture(templates, ["6m", "4m", "5m"], 0)
        found = pick(picture, [(["6m", "4m", "5m"], 0)], templates)
        assert found is not None
        assert found.reds == (False, False, False)

    def test_too_close_to_call_is_no_answer(self, templates: TemplateSet) -> None:
        picture = _picture(templates, ["3m", "2m", "4m"], 0)
        shape = (["3m", "2m", "4m"], 0)
        assert pick(picture, [shape, shape], templates) is None

    def test_nothing_to_pick_from(self, templates: TemplateSet) -> None:
        assert pick(_picture(templates, ["3m", "2m", "4m"], 0), [], templates) is None

    def test_one_possible_pon_still_needs_a_visible_picture(self, templates: TemplateSet) -> None:
        """規則只剩碰,不代表被手蓋住的畫面能判斷赤五。"""
        picture = _picture(templates, ["5m", "5m", "5m"], 1)
        picture.image[:] = (140, 170, 190)
        assert pick(picture, [(["5m", "5m", "5m"], 1)], templates) is None

    def test_one_visible_tile_does_not_identify_the_whole_run(
        self, templates: TemplateSet
    ) -> None:
        """只剩一張直放二萬可見,不能因為二萬比三萬像就採信整組。"""
        picture = _picture(templates, ["3m", "2m", "4m"], 0)
        face = picture.image.copy()
        picture.image[:] = FELT
        cx, cy = picture.layout(3, 0, (0.0, PITCH, 20.0))[1]
        width, height = _SIZES[SEAT][0]
        x, y = round(cx - width / 2), round(cy - height / 2)
        picture.image[y : y + height, x : x + width] = face[y : y + height, x : x + width]
        shapes = [(["3m", "2m", "4m"], 0), (["3m", "3m", "3m"], 0)]
        assert pick(picture, shapes, templates) is None


class TestTheLayout:
    def test_the_sideways_tile_is_wider(self, templates: TemplateSet) -> None:
        picture = _picture(templates, ["3m", "2m", "4m"], 0)
        first, second, third = (np.array(c) for c in picture.layout(3, 0, (0.0, 80.0, 0.0)))
        assert np.linalg.norm(second - first) == pytest.approx((SIDEWAYS + 1) * 40)
        assert np.linalg.norm(third - second) == pytest.approx(80)


@pytest.mark.parametrize("seat", ["shimocha", "toimen", "kamicha"])
@pytest.mark.parametrize("sideways", [0, 1, 2])
def test_cropped_search_keeps_the_full_image_score(
    seat: str, sideways: int, templates: TemplateSet
) -> None:
    image = np.zeros((1440, 2560, 3), dtype=np.uint8)
    cropped = MeldPicture(image, TABLE, seat, end=6.0)
    full = MeldPicture(image, TABLE, seat, end=6.0)
    pixels = np.random.default_rng(42).integers(0, 256, cropped.image.shape, dtype=np.uint8)
    cropped.image[:] = pixels
    full.image[:] = pixels
    tiles = ("3m", "2m", "4m")
    for index, label in enumerate(tiles):
        # 強制參考版搜尋完整影像,驗證裁框與細搜餘裕,不改評分公式。
        full._ranges[(label, index == sideways)] = (  # noqa: SLF001
            0, 0, full.image.shape[1], full.image.shape[0]
        )
    want = full.fit(tiles, sideways, templates)
    got = cropped.fit(tiles, sideways, templates)
    assert got.score == pytest.approx(want.score, abs=1e-5)
