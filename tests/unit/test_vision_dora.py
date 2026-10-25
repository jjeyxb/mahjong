"""寶牌面板:切槽、判斷翻開沒、認牌。

素材 ``table_full.jpg`` 那一刻的指示牌是 **3s**,答案來自同一時刻的封包
(``20260801-110639`` 幀 12141),不是目視判讀的。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet
from mia.vision.tiles.dora import DORA_FACE, SLOTS, read_dora, slot_rect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 牌桌矩形與 ``dora_indicators`` 的 ROI(``data/roi_ref/roi.yaml``)。
TABLE = Rect(0, 174, 2536, 1430)
ROI = (0.0133, 0.0385, 0.1475, 0.1319)

#: 封包說那一刻的指示牌。
EXPECTED = "3s"


@pytest.fixture(scope="module")
def panel() -> np.ndarray:
    image = cv2.imread(str(FIXTURES / "table_full.jpg"))
    assert image is not None, "讀不到 table_full.jpg"
    x = round(TABLE.x + ROI[0] * TABLE.width)
    y = round(TABLE.y + ROI[1] * TABLE.height)
    return image[y : y + round(ROI[3] * TABLE.height), x : x + round(ROI[2] * TABLE.width)]


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    return TemplateSet.load()


class TestItReadsThePanel:
    def test_the_revealed_indicator(self, panel: np.ndarray, templates: TemplateSet) -> None:
        slots = read_dora(panel, templates)
        assert slots[0] is not None
        assert slots[0].label == EXPECTED

    def test_it_is_confident(self, panel: np.ndarray, templates: TemplateSet) -> None:
        """寶牌是正面朝鏡頭、沒有遮擋 —— 分數遠高於牌河的 0.6~0.7。

        實測 0.929。這裡要求 0.85 留餘裕,但**必須明顯高於牌河**,
        不然就代表切槽或尺寸跑掉了。
        """
        match = read_dora(panel, templates)[0]
        assert match is not None
        assert match.score > 0.85
        assert match.is_confident

    def test_the_face_down_slots_are_none(
        self, panel: np.ndarray, templates: TemplateSet
    ) -> None:
        """橘色牌背不是「認不出來的牌」,是「還沒翻開」。

        實測翻開的槽飽和度 37、牌背 169 —— 分得比牌河還開。
        """
        assert read_dora(panel, templates)[1:] == (None,) * (SLOTS - 1)


class TestTheGeometry:
    def test_five_slots_tile_the_panel(self) -> None:
        """1 張起始指示牌 + 最多 4 張槓寶,剛好填滿面板寬度。"""
        rects = [slot_rect((374, 189), i) for i in range(SLOTS)]
        assert rects[0].x == 0
        assert rects[-1].x + rects[-1].width == 374
        widths = {r.width for r in rects}
        assert len(widths) == 1

    def test_the_slots_stop_above_the_stick_counters(self) -> None:
        """面板下面那塊是「立直棒 xN」「本場棒 xN」,不是牌。

        實測面板高 189,牌的區域到 y≈112(亮度從 213 掉到 68 的那一列)。
        """
        assert slot_rect((374, 189), 0).height < 189
        assert 100 < slot_rect((374, 189), 0).height < 125

    def test_the_face_is_measured_not_assumed(self) -> None:
        """尖峰很銳利 —— 寬度差 8 px 答案就從 3s 變成 5z。

        立直橫放牌踩過同樣的坑(原本假設是寬高交換)。這裡釘住量出來的值。
        """
        rect = slot_rect((374, 189), 0)
        assert (round(rect.width * DORA_FACE[0]), round(rect.height * DORA_FACE[1])) == (58, 92)
