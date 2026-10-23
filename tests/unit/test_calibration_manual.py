"""使用者自己框的手牌範圍。"""

from __future__ import annotations

from mia.calibration.manual import HandRoiChoice, TableRectChoice
from mia.utils.geometry import NormRect, Rect, Size


class TestHandRoiChoice:
    def test_nothing_framed_is_the_default(self) -> None:
        """沒框過時呼叫端要走校正那條路,所以得分得出來。"""
        choice = HandRoiChoice()
        assert choice.value is None
        assert choice.pixels(Size(100, 100)) is None
        assert choice.as_tuple() is None

    def test_it_converts_to_pixels(self) -> None:
        choice = HandRoiChoice((0.1, 0.8, 0.7, 0.15))
        rect = choice.pixels(Size(1000, 1000))
        assert rect is not None
        assert (rect.x, rect.y, rect.width, rect.height) == (100, 800, 700, 150)

    def test_the_same_box_reports_no_change(self) -> None:
        choice = HandRoiChoice((0.1, 0.8, 0.7, 0.15))
        assert choice.set((0.1, 0.8, 0.7, 0.15)) is False
        assert choice.set((0.1, 0.8, 0.7, 0.16)) is True

    def test_a_broken_value_is_treated_as_never_framed(self) -> None:
        """來源是磁碟上的舊設定 —— 為了一個記住的框讓程式開不起來不划算。"""
        assert HandRoiChoice((-1.0, 0.0, 1.0, 1.0)).value is None
        assert HandRoiChoice((0.0, 0.0, 0.0, 0.0)).value is None
        assert HandRoiChoice((0.5, 0.5, 0.9, 0.9)).value is None, "超出邊界"

    def test_a_click_without_a_drag_is_refused(self) -> None:
        """寬高是 0 —— 使用者在對話框裡點了一下沒拖。那是誤觸,不是錯誤。"""
        choice = HandRoiChoice((0.1, 0.8, 0.7, 0.15))
        assert choice.set((0.3, 0.3, 0.0, 0.0)) is False
        assert choice.value is not None, "誤觸不該把已經框好的清掉"

    def test_clearing_goes_back_to_calibration(self) -> None:
        choice = HandRoiChoice((0.1, 0.8, 0.7, 0.15))
        assert choice.clear() is True
        assert choice.value is None
        assert choice.clear() is False, "已經是空的了,沒有變"

    def test_it_takes_a_normrect_too(self) -> None:
        """對話框算出來的就是 NormRect,不該強迫呼叫端先拆成四個數字。"""
        choice = HandRoiChoice()
        assert choice.set(NormRect(0.1, 0.8, 0.7, 0.15)) is True
        assert choice.as_tuple() == (0.1, 0.8, 0.7, 0.15)


class TestTableRectChoice:
    """牌桌矩形的格子。與手牌範圍共用實作,但**代表的東西不同**。

    手牌範圍取代校正;牌桌矩形就是校正的結果。所以它們必須是兩個格子 ——
    共用一個的話,使用者框了手牌就等於改掉了牌桌,反之亦然。
    """

    def test_it_is_a_separate_cell_from_the_hand_roi(self) -> None:
        HandRoiChoice((0.1, 0.8, 0.8, 0.1))
        assert TableRectChoice().value is None, "框手牌不該順便改掉牌桌"

    def test_it_remembers_what_was_framed(self) -> None:
        table = TableRectChoice()
        assert table.set((0.0, 0.1, 1.0, 0.8))
        assert table.pixels(Size(1800, 1186)) == Rect(0, 119, 1800, 949)

    def test_clearing_goes_back_to_nothing(self) -> None:
        table = TableRectChoice((0.0, 0.1, 1.0, 0.8))
        assert table.clear()
        assert table.value is None

    def test_a_broken_remembered_value_is_treated_as_unframed(self) -> None:
        """來源是磁碟上的 ``ui_state.json``。為了一個記不得的框讓程式開不起來
        是不划算的交易。
        """
        assert TableRectChoice((0.5, 0.5, 0.9, 0.9)).value is None

    def test_its_repr_names_its_own_class(self) -> None:
        """共用實作的副作用:``__repr__`` 寫死成另一個類別名的話,
        日誌會說「HandRoiChoice」而使用者在調的是牌桌。
        """
        assert "TableRectChoice" in repr(TableRectChoice())
