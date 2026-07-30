"""iOS 風格開關。

畫出來的樣子測不了太多(那要看的),但**狀態與外觀的對應**要釘住:
滑塊在哪一邊、軌道是不是綠的,都不能與 ``isChecked()`` 脫節。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from mia.ui.widgets.toggle import ON_COLOR, TRACK_HEIGHT, TRACK_WIDTH, ToggleSwitch


@pytest.fixture
def switch(qapp) -> ToggleSwitch:  # noqa: ARG001 - 只需要 QApplication 存在
    # duration=0:沒有事件迴圈在跑,靠動畫是等不到終點的
    return ToggleSwitch(duration=0)


@pytest.mark.usefixtures("qapp")
class TestState:
    def test_it_starts_off(self, switch: ToggleSwitch) -> None:
        assert not switch.isChecked()
        assert switch.position == pytest.approx(0.0)

    def test_it_is_checkable(self, switch: ToggleSwitch) -> None:
        assert switch.isCheckable()

    def test_checking_moves_the_knob_to_the_right(self, switch: ToggleSwitch) -> None:
        switch.setChecked(True)
        assert switch.position == pytest.approx(1.0)

    def test_unchecking_moves_it_back(self, switch: ToggleSwitch) -> None:
        switch.setChecked(True)
        switch.setChecked(False)
        assert switch.position == pytest.approx(0.0)

    def test_clicking_toggles(self, switch: ToggleSwitch) -> None:
        switch.click()
        assert switch.isChecked()
        switch.click()
        assert not switch.isChecked()

    def test_toggled_is_emitted_with_the_new_state(self, switch: ToggleSwitch) -> None:
        seen: list[bool] = []
        switch.toggled.connect(seen.append)
        switch.click()
        switch.click()
        assert seen == [True, False]

    def test_setting_the_same_state_twice_emits_once(self, switch: ToggleSwitch) -> None:
        seen: list[bool] = []
        switch.toggled.connect(seen.append)
        switch.setChecked(True)
        switch.setChecked(True)
        assert seen == [True]

    def test_the_knob_ends_up_correct_even_without_an_event_loop(self) -> None:
        """duration>0 時動畫要靠事件迴圈推進。

        建構期或測試裡沒有迴圈,滑塊會停在錯的一邊 —— 而畫面上那就是
        「開關看起來是關的,但功能在跑」。這裡釘住 duration=0 這條路。
        """
        animated = ToggleSwitch(duration=0)
        animated.setChecked(True)
        assert animated.position == pytest.approx(1.0)


@pytest.mark.usefixtures("qapp")
class TestAppearance:
    def test_the_on_colour_is_apple_system_green(self) -> None:
        """使用者要的就是綠色。跟著系統色而不是自己調 —— 對比度已經被驗過。"""
        assert QColor(52, 199, 89) == ON_COLOR

    def test_the_track_is_green_when_on(self, switch: ToggleSwitch) -> None:
        switch.setChecked(True)
        assert switch._track_color() == ON_COLOR  # noqa: SLF001

    def test_the_track_is_not_green_when_off(self, switch: ToggleSwitch) -> None:
        assert switch._track_color() != ON_COLOR  # noqa: SLF001

    def test_the_size_hint_is_a_pill(self, switch: ToggleSwitch) -> None:
        """寬必須大於高,否則圓角會把它畫成一個圓,看不出是開關。"""
        hint = switch.sizeHint()
        assert (hint.width(), hint.height()) == (TRACK_WIDTH, TRACK_HEIGHT)
        assert hint.width() > hint.height()

    def test_a_disabled_switch_is_dimmed(self, switch: ToggleSwitch) -> None:
        switch.setChecked(True)
        opaque = switch._track_color().alphaF()  # noqa: SLF001
        switch.setEnabled(False)
        assert switch._track_color().alphaF() < opaque  # noqa: SLF001

    def test_painting_does_not_raise_in_either_state(self, switch: ToggleSwitch) -> None:
        """繪製會被呼叫在沒有 show 過的 widget 上(離屏測試、截圖)。"""
        from PySide6.QtGui import QPixmap

        for checked in (False, True):
            switch.setChecked(checked)
            switch.resize(switch.sizeHint())
            pixmap = QPixmap(switch.size())
            switch.render(pixmap)
            assert not pixmap.isNull()

    def test_the_rendered_pixels_differ_between_on_and_off(self, switch: ToggleSwitch) -> None:
        """光是「不會爆」不夠 —— 兩個狀態畫出來必須真的不一樣。"""
        from PySide6.QtGui import QPixmap

        switch.resize(switch.sizeHint())
        images = []
        for checked in (False, True):
            switch.setChecked(checked)
            pixmap = QPixmap(switch.size())
            switch.render(pixmap)
            images.append(pixmap.toImage())
        assert images[0] != images[1]


@pytest.mark.usefixtures("qapp")
class TestAccessibility:
    def test_it_takes_keyboard_focus(self, switch: ToggleSwitch) -> None:
        """開關是主要的操作元件,只能用滑鼠按到就太窄了。"""
        assert switch.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_it_shows_a_pointing_cursor(self, switch: ToggleSwitch) -> None:
        assert switch.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_repr_reports_the_state(self, switch: ToggleSwitch) -> None:
        switch.setChecked(True)
        assert "checked=True" in repr(switch)


@pytest.mark.usefixtures("qapp")
class TestInitialState:
    """``set_initial`` 是給「反映現況」用的,不是給「使用者改了」用的。

    這一段釘住一個渲染出來才看到的 bug:用預設的 140 ms 動畫呼叫
    ``setChecked(True)``,而事件迴圈還沒起來時動畫一格都不會走 —— 滑塊停在
    左邊,開關看起來是關的,但功能其實在跑。
    """

    def test_the_knob_is_immediately_in_place(self) -> None:
        switch = ToggleSwitch()  # 預設 140 ms 動畫,且沒有事件迴圈在跑
        switch.set_initial(True)
        assert switch.position == pytest.approx(1.0)

    def test_plain_set_checked_would_not_have_moved_it(self) -> None:
        """釘住問題本身 —— 這是當初做錯的方式。"""
        switch = ToggleSwitch()
        switch.setChecked(True)
        assert switch.isChecked()
        assert switch.position == pytest.approx(0.0), "沒有事件迴圈,動畫竟然走完了?"

    def test_it_sets_the_checked_state(self) -> None:
        switch = ToggleSwitch()
        switch.set_initial(True)
        assert switch.isChecked()

    def test_it_does_not_emit_toggled(self) -> None:
        """反映現況不該被誤當成「使用者要求改變」—— 那會再送一次 set_enabled。"""
        switch = ToggleSwitch()
        seen: list[bool] = []
        switch.toggled.connect(seen.append)
        switch.set_initial(True)
        assert seen == []

    def test_it_can_turn_things_off_too(self) -> None:
        switch = ToggleSwitch(duration=0)
        switch.setChecked(True)
        switch.set_initial(False)
        assert not switch.isChecked()
        assert switch.position == pytest.approx(0.0)
