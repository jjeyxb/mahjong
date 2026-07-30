"""側邊視窗的 widget。

只驗**接線**:狀態換了畫面有沒有跟著換、牌面圖找不找得到、widget 池會不會
隨手牌變短而留下上一手的殘影。像素長什麼樣不在這裡驗 —— 那要人眼看,
用 ``python tools/ui.py --demo``。

在無頭環境跑,所以 conftest 會設 ``QT_QPA_PLATFORM=offscreen``。
"""

from __future__ import annotations

import pytest

from majsoul_copilot.engine.base import Advice
from majsoul_copilot.mjai import Dahai, Reach
from majsoul_copilot.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import Qt

from majsoul_copilot.ui.panel.window import PanelWindow, present
from majsoul_copilot.ui.widgets.tiles import TileIcons

TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]


@pytest.fixture
def panel(qtbot):
    model = ViewModel()
    window = PanelWindow(model)
    model.subscribe(window.apply)
    qtbot.addWidget(window)
    return model, window


def shown(widget) -> bool:
    """這個 widget 有沒有被我們設成顯示。

    用 ``isHidden`` 而不是 ``isVisible``:後者對「沒有 show() 過的視窗」底下的
    子元件一律回 False,測不出我們自己的 setVisible 有沒有下對。實際要驗的是
    「殘影有沒有被藏起來」,那正是 isHidden 反映的東西。
    """
    return not widget.isHidden()


@pytest.mark.usefixtures("qapp")
class TestTileIcons:
    """QPixmap 需要 QApplication 才能建 —— 沒有的話是 segfault 而不是例外。

    ``qapp`` 是 pytest-qt 提供的,這裡只需要它的副作用(讓 QApplication 存在),
    所以用 usefixtures 而不是掛成參數。
    """

    def test_every_tile_has_an_image(self) -> None:
        """37 種牌都要找得到圖 —— 少一種在畫面上是一個空格,很容易漏看。"""
        icons = TileIcons()
        tiles = [f"{n}{s}" for s in "mps" for n in range(1, 10)]
        tiles += ["E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]
        assert len(tiles) == 37
        missing = [t for t in tiles if icons.pixmap(t, 40).isNull()]
        assert missing == [], f"這些牌沒有圖:{missing}"

    def test_scaling_keeps_the_aspect_ratio(self) -> None:
        icons = TileIcons()
        pixmap = icons.pixmap("1m", 109)
        assert pixmap.height() == 109
        assert pixmap.width() == 64

    def test_an_unknown_tile_gives_an_empty_pixmap(self) -> None:
        """認不得的牌回空的而不是拋例外 —— UI 在畫的時候不該因此崩掉。"""
        assert TileIcons().pixmap("?", 40).isNull()
        assert TileIcons().pixmap("99z", 40).isNull()

    def test_a_missing_skin_fails_loudly(self) -> None:
        with pytest.raises(FileNotFoundError, match="fetch_tiles"):
            TileIcons("這個皮膚不存在")


class TestPanel:
    def test_it_starts_without_data(self, panel) -> None:
        _, window = panel
        assert "等待" in window._notice.text()  # noqa: SLF001

    def test_a_hand_reaches_the_analysis_tab(self, panel) -> None:
        model, window = panel
        model.update_packet_hand(TENPAI)
        assert window._analysis._shanten.text() == "聽牌"  # noqa: SLF001

    def test_an_advice_reaches_the_headline(self, panel) -> None:
        model, window = panel
        meta = {"mask_bits": (1 << 8) | (1 << 27), "q_values": [0.1, 0.9]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="E", tsumogiri=False), meta, 14.0)]
        )
        headline = window._advice._headline  # noqa: SLF001
        assert headline._verb.text() == "切"  # noqa: SLF001
        assert "mortal" in headline._detail.text()  # noqa: SLF001

    def test_a_non_discard_action_shows_the_action_name(self, panel) -> None:
        """立直沒有對應的單張牌,大字要換成動作本身而不是留空。"""
        model, window = panel
        model.update_advices([Advice("mortal", Reach(actor=0), None, 10.0)])
        assert window._advice._headline._verb.text() == "立直"  # noqa: SLF001

    def test_a_shorter_hand_hides_the_leftover_tiles(self, panel) -> None:
        """13 張換成 4 張(副露之後)時,不該留著上一手的九張殘影。"""
        model, window = panel
        model.update_packet_hand(TENPAI)
        model.update_packet_hand(["2m", "3m", "4m", "5m"])
        strip = window._analysis._hand  # noqa: SLF001
        visible = [label for label in strip._labels if shown(label)]  # noqa: SLF001
        assert len(visible) == 4

    def test_fewer_candidates_hide_the_leftover_rows(self, panel) -> None:
        model, window = panel
        wide = {"mask_bits": 0b1111, "q_values": [0.1, 0.2, 0.3, 0.4]}
        narrow = {"mask_bits": 0b11, "q_values": [0.1, 0.2]}
        for meta in (wide, narrow):
            model.update_advices(
                [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
            )
        rows = window._advice._rows  # noqa: SLF001
        assert sum(shown(row) for row in rows) == 2

    def test_a_rule_based_engine_hides_the_candidate_section(self, panel) -> None:
        """baseline 沒有 Q 值,不該留著上一手的長條。"""
        model, window = panel
        meta = {"mask_bits": 0b11, "q_values": [0.1, 0.2]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
        )
        model.update_advices([Advice("baseline", Dahai(actor=0, pai="1m", tsumogiri=False))])
        assert not shown(window._advice._candidates_label)  # noqa: SLF001

    def test_a_conflict_shows_up_in_the_status_bar(self, panel) -> None:
        model, window = panel
        model.update_cv_hand(["1m"] * 13)
        model.update_packet_hand(TENPAI)
        assert "不一致" in window._notice.text()  # noqa: SLF001

    def test_a_single_candidate_does_not_divide_by_zero(self, panel) -> None:
        """立直之後強制摸切,合法動作只有一個 —— 實測真的會發生。"""
        model, window = panel
        meta = {"mask_bits": 0b1, "q_values": [0.27]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=True), meta, 1.0)]
        )
        assert shown(window._advice._rows[0])  # noqa: SLF001


class TestPresent:
    """顯示流程。

    這一段是為了釘住一個真實的 bug:原本 ``show()`` 之後才設
    ``WindowStaysOnTopHint``,Qt 會把視窗隱藏並要求重新 ``show()``。症狀是
    「Dock 上有圖示,但畫面上沒有視窗」—— 完全不像 flag 造成的,而且我當初
    截圖驗證時只呼叫了 ``show()``、沒走這條路,所以漏掉了。
    """

    def test_the_window_is_visible_after_present(self, qtbot) -> None:
        model = ViewModel()
        window = PanelWindow(model)
        qtbot.addWidget(window)
        present(window)
        assert not window.isHidden(), "present() 之後視窗竟然是隱藏的"

    def test_always_on_top_is_set_before_showing(self, qtbot) -> None:
        """flag 要在建構時就設好。show 之後再設會讓視窗被隱藏。"""
        model = ViewModel()
        window = PanelWindow(model)
        qtbot.addWidget(window)
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        present(window)
        assert not window.isHidden()

    def test_it_can_be_turned_off(self, qtbot) -> None:
        model = ViewModel()
        window = PanelWindow(model, always_on_top=False)
        qtbot.addWidget(window)
        assert not (window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        present(window)
        assert not window.isHidden()


class TestNavigation:
    """左側功能列。"""

    def test_there_are_three_pages(self, panel) -> None:
        _, window = panel
        assert window._nav.count() == 3  # noqa: SLF001
        assert window._stack.count() == 3  # noqa: SLF001

    def test_selecting_a_page_switches_the_stack(self, panel) -> None:
        _, window = panel
        window.show_page(1)
        assert window._stack.currentIndex() == 1  # noqa: SLF001

    def test_the_settings_strip_is_hidden_when_empty(self, panel) -> None:
        """向聽分析頁目前沒有設定 —— 不該留一條空白的橫線。"""
        _, window = panel
        analysis_page = window._stack.widget(1)  # noqa: SLF001
        assert not shown(analysis_page._settings)  # noqa: SLF001


class TestEnginePicker:
    """主要引擎下拉選單。引擎清單是跑起來才知道的,所以要跟著狀態長出來。"""

    def _advice(self, name: str, tile: str) -> Advice:
        return Advice(name, Dahai(actor=0, pai=tile, tsumogiri=False), None, 1.0)

    def test_it_starts_with_only_the_auto_entry(self, panel) -> None:
        _, window = panel
        assert window._engine_picker.count() == 1  # noqa: SLF001

    def test_engines_are_added_as_they_appear(self, panel) -> None:
        model, window = panel
        model.update_advices([self._advice("mortal", "1m"), self._advice("baseline", "9p")])
        picker = window._engine_picker  # noqa: SLF001
        assert [picker.itemData(i) for i in range(picker.count())] == [None, "mortal", "baseline"]

    def test_picking_an_engine_makes_it_primary(self, panel) -> None:
        model, window = panel
        model.update_advices([self._advice("mortal", "1m"), self._advice("baseline", "9p")])
        window._engine_picker.setCurrentIndex(2)  # noqa: SLF001
        assert model.state.preferred_engine == "baseline"
        assert model.state.primary is not None
        assert model.state.primary.name == "baseline"

    def test_rebuilding_the_list_does_not_clear_the_preference(self, panel) -> None:
        """重建下拉選單時要擋掉訊號,否則清空的瞬間會把偏好設成 None。"""
        model, window = panel
        model.update_advices([self._advice("mortal", "1m")])
        window._engine_picker.setCurrentIndex(1)  # noqa: SLF001
        model.update_advices([self._advice("mortal", "1m"), self._advice("baseline", "9p")])
        assert model.state.preferred_engine == "mortal"

    def test_the_candidate_count_is_adjustable(self, panel) -> None:
        model, window = panel
        meta = {"mask_bits": 0b1111, "q_values": [0.1, 0.2, 0.3, 0.4]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
        )
        window._candidate_count.setValue(2)  # noqa: SLF001
        assert sum(shown(row) for row in window._advice._rows) == 2  # noqa: SLF001


class TestAlwaysOnTopToggle:
    def test_toggling_it_off_keeps_the_window_visible(self, panel) -> None:
        """改 window flag 會讓已顯示的視窗被隱藏 —— 必須重新 show 一次。"""
        _, window = panel
        present(window)
        window._on_top.setChecked(False)  # noqa: SLF001
        assert not window.isHidden()
        assert not (window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

    def test_toggling_it_back_on_keeps_the_window_visible(self, panel) -> None:
        _, window = panel
        present(window)
        window._on_top.setChecked(False)  # noqa: SLF001
        window._on_top.setChecked(True)  # noqa: SLF001
        assert not window.isHidden()
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
