"""「框選牌桌範圍」對話框裡的推導。

這裡測的是**不需要 GUI 的那部分**:從使用者拖的框算出手牌 ROI 與四家牌河,
以及長寬比的提醒。對話框本身的安全網就是把這些畫回畫面上 —— 算錯了,
畫出來的線就落在錯的地方,而那正是使用者唯一看得見的證據。
"""

from __future__ import annotations

import pytest

from mia.config.models import RoiConfig
from mia.ui.panel.tablerect import _aspect_note, _hand_roi, _river_quads
from mia.utils.geometry import Rect, Size
from mia.vision.tiles.river import SEATS

#: 實測素材的牌桌矩形與畫面尺寸(``data/recordings/20260801-110639``)。
TABLE = Rect(0, 174, 2536, 1430)
IMAGE = Size(2536, 1608)


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
