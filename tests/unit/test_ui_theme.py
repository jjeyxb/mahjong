"""側邊視窗的控制項外觀。"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QComboBox, QWidget

from mia.ui.panel.window import PanelWindow
from mia.ui.theme import (
    COMBO_WIDTH,
    CONTROL_HEIGHT,
    POPUP_GAP,
    AlignedComboBox,
    ChoiceButtons,
)
from mia.ui.viewmodel import ViewModel


@pytest.fixture
def combo(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    box = AlignedComboBox(host)
    box.setFixedWidth(COMBO_WIDTH)
    box.addItems(["網頁版", "Steam 版"])
    host.move(200, 200)
    host.show()
    yield box
    box.hidePopup()


class TestAlignedPopup:
    """展開的清單要與框同寬、左緣對齊、緊貼在正下方。差幾個像素就看得出來。"""

    def test_same_width(self, combo) -> None:
        combo.showPopup()
        assert combo.view().window().width() == combo.width()

    def test_left_edges_line_up(self, combo) -> None:
        combo.showPopup()
        popup = combo.view().window()
        assert popup.geometry().left() == combo.mapToGlobal(QPoint(0, 0)).x()

    def test_right_below_the_box(self, combo) -> None:
        combo.showPopup()
        popup = combo.view().window()
        bottom = combo.mapToGlobal(QPoint(0, combo.height())).y()
        assert popup.geometry().top() == bottom + POPUP_GAP

    def test_it_opens_upwards_at_the_bottom_of_the_screen(self, combo) -> None:
        area = combo.screen().availableGeometry()
        combo.window().move(area.left() + 40, area.bottom() - combo.height() - 10)
        combo.showPopup()
        popup = combo.view().window()
        top = combo.mapToGlobal(QPoint(0, 0)).y()
        assert popup.geometry().bottom() < top
        assert popup.width() == combo.width()


def test_every_panel_combo_is_aligned_and_equally_wide(qtbot) -> None:
    window = PanelWindow(ViewModel())
    qtbot.addWidget(window)
    combos = window.findChildren(QComboBox)
    assert combos
    for box in combos:
        assert isinstance(box, AlignedComboBox), box
        assert box.width() == COMBO_WIDTH or box.maximumWidth() == COMBO_WIDTH


class TestChoiceButtons:
    @pytest.fixture
    def buttons(self, qtbot):
        widget = ChoiceButtons([("網頁版", "web"), ("Steam 版", "steam")])
        qtbot.addWidget(widget)
        widget.select("web")
        return widget

    def test_pressing_one_dims_the_other(self, buttons) -> None:
        buttons.button("steam").click()
        assert buttons.button("steam").isChecked()
        assert not buttons.button("web").isChecked()
        assert buttons.current() == "steam"

    def test_a_click_is_reported(self, buttons) -> None:
        got: list[object] = []
        buttons.chosen.connect(got.append)
        buttons.button("steam").click()
        assert got == ["steam"]

    def test_drawing_the_state_is_not_a_click(self, buttons) -> None:
        got: list[object] = []
        buttons.chosen.connect(got.append)
        buttons.select("steam")
        assert got == []
        assert buttons.button("steam").isChecked()

    def test_the_current_one_again_is_not_a_click(self, buttons) -> None:
        got: list[object] = []
        buttons.chosen.connect(got.append)
        buttons.button("web").click()
        assert got == []

    def test_as_wide_as_a_combo(self, buttons) -> None:
        """與底下的下拉選單上下排在一起,左右緣要對齊。"""
        assert buttons.width() == COMBO_WIDTH


def test_buttons_and_combos_are_the_same_height(qtbot) -> None:
    """遊戲版本(按鈕組)與遊戲視窗尺寸(下拉選單)上下排在一起。"""
    window = PanelWindow(ViewModel())
    qtbot.addWidget(window)
    window.show()
    combo = window._canvas_picker  # noqa: SLF001
    assert combo.height() == CONTROL_HEIGHT
    assert window._platform_picker.height() == CONTROL_HEIGHT  # noqa: SLF001
