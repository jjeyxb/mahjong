"""「框選牌桌範圍」對話框裡的推導。

這裡測的是**不需要 GUI 的那部分**:從使用者拖的框算出手牌 ROI 與四家牌河,
以及長寬比的提醒。對話框本身的安全網就是把這些畫回畫面上 —— 算錯了,
畫出來的線就落在錯的地方,而那正是使用者唯一看得見的證據。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.config.models import RoiConfig
from mia.ui.panel.tablerect import (
    SEAT_NAMES,
    TableRectDialog,
    _aspect_note,
    _hand_roi,
    _river_quads,
)
from mia.utils.geometry import Rect, Size
from mia.vision.tiles.river import SEATS

#: 實測素材的牌桌矩形與畫面尺寸(``data/recordings/20260801-110639``)。
TABLE = Rect(0, 174, 2536, 1430)
IMAGE = Size(2536, 1608)


@pytest.fixture(scope="module")
def table_image():
    """真實對局的一整張擷取畫面。答案來自封包,不是目視 —— 見 test_vision_river。"""
    import cv2

    path = Path(__file__).resolve().parents[1] / "fixtures" / "table_full.jpg"
    image = cv2.imread(str(path))
    assert image is not None, "讀不到 table_full.jpg"
    return image


class TestHandRoi:
    def test_it_matches_what_the_evaluator_computes(self) -> None:
        """對話框算的必須與 ``tools/evaluate.py`` 算的是同一個。

        兩邊算不一樣的話,使用者在對話框裡看到線落在牌上、存下去,然後實際
        跑的是另一個位置 —— 畫面說一套、程式做另一套,最糟的那種。
        """
        roi = _hand_roi(_config(), TABLE, IMAGE)
        assert roi == Rect(292, 1394, 1796, 205)

    def test_it_scales_with_the_frame(self) -> None:
        """ROI 是相對牌桌矩形的比例 —— 框小一半,ROI 要跟著小一半。"""
        half = Rect(0, 87, 1268, 715)
        roi = _hand_roi(_config(), half, Size(1268, 804))
        assert roi is not None
        assert roi.width == pytest.approx(1796 / 2, abs=2)

    def test_a_config_without_own_hand_says_so(self) -> None:
        """設定檔沒有這個 ROI 時要回 None,不能拋 —— 對話框還是要開得起來,
        只是沒辦法當場驗給使用者看。
        """
        assert _hand_roi(RoiConfig(), TABLE, IMAGE) is None


class TestRiverQuads:
    def test_every_seat_gets_a_quad(self) -> None:
        quads = _river_quads(TABLE)
        assert len(quads) == len(SEATS)
        assert all(len(q) == 4 for q in quads)

    def test_the_quads_land_inside_the_table(self) -> None:
        """牌河在牌桌中央 —— 跑到框外面就是推導錯了。"""
        for quad in _river_quads(TABLE):
            for x, y in quad:
                assert TABLE.x <= x <= TABLE.x + TABLE.width
                assert TABLE.y <= y <= TABLE.y + TABLE.height

    def test_they_move_with_the_frame(self) -> None:
        """角點是比例 —— 框往下移,牌河要跟著往下移。

        不跟著動的話,使用者把框拖到別的地方,畫面上的綠線卻不動,
        於是他看不出自己框錯了。
        """
        moved = Rect(TABLE.x, TABLE.y + 100, TABLE.width, TABLE.height)
        before = _river_quads(TABLE)[0][0]
        after = _river_quads(moved)[0][0]
        assert after[1] - before[1] == pytest.approx(100, abs=2)


class TestAspectNote:
    def test_a_normal_frame_says_nothing(self) -> None:
        assert _aspect_note(TABLE) == ""

    def test_a_non_sixteen_nine_client_is_tolerated(self) -> None:
        """Steam 版的客戶區實測 1512x844 = 1.7915,本來就不是 16:9。

        擋下來的話那些使用者永遠按不了「確定」。
        """
        assert _aspect_note(Rect(0, 0, 1512, 844)) == ""

    def test_a_toolbar_left_in_the_frame_is_called_out(self) -> None:
        """實測 20260801-112437 自動剝除剝出 1798x1152 = 1.56,就是這種。"""
        note = _aspect_note(Rect(0, 0, 1798, 1152))
        assert "工具列" in note


def _config() -> RoiConfig:
    from mia.config.loader import load_config

    return load_config().roi


class TestSeatNames:
    def test_every_seat_has_a_name(self) -> None:
        """少一家的話訊息會丟 KeyError —— 而那發生在使用者剛拖完框的時候。"""
        assert set(SEAT_NAMES) == set(SEATS)


class TestRiverNote:
    """拖完之後對牌河的判語。

    **不叫使用者用眼睛判斷綠線正不正。** 容忍度比看得出來的還窄:實測牌桌
    矩形的上緣只要偏 0.5% 的高度(1608 px 的畫面上是 8 px),每張牌正確率
    就從 94.6% 掉到 79.2%,而綠線看起來仍然「大致落在牌上」。
    """

    def _note(self, image, table: Rect) -> str:
        """不開 QDialog —— 這段邏輯不需要視窗。"""

        class Stub:
            _image = image
            _templates = None
            _river_note = TableRectDialog._river_note  # noqa: SLF001

        return Stub()._river_note(table)  # noqa: SLF001

    def test_a_correct_frame_passes(self, table_image) -> None:
        note = self._note(table_image, TABLE)
        assert "可以按" in note
        assert "15/17/17/16" in note, "張數要報出來,那是使用者唯一對得起來的東西"

    @pytest.mark.parametrize(("offset", "label"), [(8, "0.5%"), (16, "1.0%")])
    def test_a_slightly_crooked_frame_is_caught(self, table_image, offset, label) -> None:
        """實測上緣偏 0.5% → 63% 有把握,偏 1% → 23%。門檻 65% 兩個都擋得下。"""
        note = self._note(table_image, Rect(0, 174 + offset, 2536, 1430))
        assert "框歪了" in note, f"偏 {label} 沒被擋下來"

    def test_the_count_alone_cannot_tell(self, table_image) -> None:
        """**這是這條檢查存在的理由。**

        即時路徑的採用條件是「連續 + 張數正確」,而那兩個在這些偏移下一個都
        沒反應 —— 三種偏移讀到的都是 15/17/17/16 張、四家都連續。1921 幀的
        全量也一樣:上緣偏 1% 時採用率仍有 93.3%,每張牌卻只剩 32.9%。
        """
        counts = [_counts(table_image, Rect(0, 174 + d, 2536, 1430)) for d in (0, 8, 16)]
        assert counts[0] == counts[1] == counts[2] == (15, 17, 17, 16)

    def test_an_empty_river_says_so_instead_of_guessing(self) -> None:
        """一局剛開始四家加起來只有兩三張,那時的比例是兩三個樣本的事。

        寧可說「太空了」也不要給一個會晃的數字 —— 使用者會照著它重框。
        """
        import numpy as np

        blank = np.full((1608, 2536, 3), 70, dtype=np.uint8)
        assert "太空" in self._note(blank, TABLE)


def _counts(image, table: Rect) -> tuple[int, ...]:
    from mia.vision.tiles.river import DEFAULT_CORNERS, read_river

    return tuple(read_river(DEFAULT_CORNERS[s].rectify(image, table)).count for s in SEATS)
