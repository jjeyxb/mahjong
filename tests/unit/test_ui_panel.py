"""側邊視窗的 widget。

只驗**接線**:狀態換了畫面有沒有跟著換、牌面圖找不找得到、widget 池會不會
隨手牌變短而留下上一手的殘影。像素長什麼樣不在這裡驗 —— 那要人眼看,
用 ``python tools/ui.py --demo``。

在無頭環境跑,所以 conftest 會設 ``QT_QPA_PLATFORM=offscreen``。
"""

from __future__ import annotations

import pytest

from mia import features
from mia.engine.base import Advice
from mia.mjai import Dahai, Reach
from mia.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import Qt

from mia.calibration.canvas import PRESETS as CANVAS_PRESETS
from mia.ui.panel.window import PanelWindow, present
from mia.ui.widgets.tiles import TileIcons

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


class TestEmptySkeleton:
    """功能關著時只留一句話,不留骨架。

    分隔線原本是寫死加進版面的,沒有任何參照 —— 它底下那一區藏起來之後,
    線還橫在一片空白上。畫面上看起來像「東西沒載出來」,而那是使用者最會
    去截圖回報的一種樣子。
    """

    def test_the_advice_page_keeps_no_rules_when_it_is_off(self, panel) -> None:
        _, window = panel
        advice = window._advice  # noqa: SLF001
        assert not shown(advice._candidates_rule)  # noqa: SLF001
        assert not shown(advice._others_rule)  # noqa: SLF001

    def test_the_analysis_page_keeps_no_rules_when_it_is_off(self, panel) -> None:
        _, window = panel
        analysis = window._analysis  # noqa: SLF001
        for rule in (analysis._hand_rule, analysis._ukeire_rule, analysis._discard_rule):  # noqa: SLF001
            assert not shown(rule)

    def test_the_rule_comes_back_with_its_section(self, panel) -> None:
        """藏掉容易,忘了放回來也一樣看不出來 —— 兩個方向都要釘。"""
        model, window = panel
        meta = {"mask_bits": 0b11, "q_values": [0.1, 0.2]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
        )
        assert shown(window._advice._candidates_rule)  # noqa: SLF001

    def test_the_second_rule_comes_back_with_the_other_engines(self, panel) -> None:
        model, window = panel
        model.update_advices(
            [
                Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), None, 1.0),
                Advice("baseline", Dahai(actor=0, pai="9p", tsumogiri=False), None, 1.0),
            ]
        )
        assert shown(window._advice._others_rule)  # noqa: SLF001

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

    def test_there_is_one_page_per_feature_plus_settings(self, panel) -> None:
        _, window = panel
        expected = len(features.NAMES) + 1
        assert window._nav.count() == expected  # noqa: SLF001
        assert window._stack.count() == expected  # noqa: SLF001

    def test_selecting_a_page_switches_the_stack(self, panel) -> None:
        _, window = panel
        window.show_page(1)
        assert window._stack.currentIndex() == 1  # noqa: SLF001

    def test_the_settings_strip_is_hidden_when_empty(self, panel) -> None:
        """設定頁的內容本身就是設定,頂端不該再留一條空白的橫線。"""
        _, window = panel
        settings_page = window._stack.widget(len(features.NAMES))  # noqa: SLF001
        assert not shown(settings_page._settings)  # noqa: SLF001

    def test_every_feature_page_shows_its_strip(self, panel) -> None:
        """功能頁都有開關,所以設定列一定看得到。"""
        _, window = panel
        for index in range(len(features.NAMES)):
            page = window._stack.widget(index)  # noqa: SLF001
            assert shown(page._settings), f"第 {index} 頁的設定列不見了"  # noqa: SLF001


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


class FakeSwitchboard:
    """記下被要求開關什麼。

    ``refuse`` 裡的 key 開不起來 —— 模擬引擎啟動失敗。
    ``missing`` 裡的 key 根本不存在 —— 模擬 --no-vision / --no-packets。
    """

    def __init__(
        self, refuse: set[str] | None = None, missing: set[str] | None = None
    ) -> None:
        self.state: dict[str, bool] = {}
        self.refuse = refuse or set()
        self.missing = missing or set()
        self.calls: list[tuple[str, bool]] = []

    def available(self, key: str) -> bool:
        return key not in self.missing

    def is_enabled(self, key: str) -> bool:
        return self.state.get(key, False)

    def set_enabled(self, key: str, on: bool) -> None:
        self.calls.append((key, on))
        self.state[key] = on and key not in self.refuse


class TestFeatureSwitches:
    """兩個功能的開關。預設關著,打開才會跑。"""

    @pytest.fixture
    def wired(self, qtbot):
        board = FakeSwitchboard()
        model = ViewModel()
        window = PanelWindow(model, switchboard=board)
        qtbot.addWidget(window)
        return board, window

    def test_every_feature_has_a_switch(self, wired) -> None:
        _, window = wired
        assert set(window._switches) == set(features.NAMES)  # noqa: SLF001

    def test_all_default_to_off(self, wired) -> None:
        """使用者要求的:預設都不執行。"""
        _, window = wired
        for switch in window._switches.values():  # noqa: SLF001
            assert not switch.isChecked()

    def test_no_feature_is_enabled_just_by_opening_the_window(self, wired) -> None:
        board, _ = wired
        assert board.calls == []
        assert board.state == {}

    def test_flipping_a_switch_enables_that_feature(self, wired) -> None:
        board, window = wired
        window._switches[features.ADVICE].click()  # noqa: SLF001
        assert board.calls == [(features.ADVICE, True)]
        assert board.is_enabled(features.ADVICE)

    def test_flipping_it_back_disables(self, wired) -> None:
        board, window = wired
        switch = window._switches[features.ADVICE]  # noqa: SLF001
        switch.click()
        switch.click()
        assert board.calls[-1] == (features.ADVICE, False)
        assert not board.is_enabled(features.ADVICE)

    def test_the_two_switches_are_independent(self, wired) -> None:
        board, window = wired
        window._switches[features.VISION].click()  # noqa: SLF001
        assert board.is_enabled(features.VISION)
        assert not board.is_enabled(features.ADVICE)

    def test_a_switch_springs_back_when_enabling_fails(self, qtbot) -> None:
        """權重不見了、找不到遊戲視窗 —— 開關不能停在「開」而底下沒東西在跑。"""
        board = FakeSwitchboard(refuse={features.ADVICE})
        window = PanelWindow(ViewModel(), switchboard=board)
        qtbot.addWidget(window)
        switch = window._switches[features.ADVICE]  # noqa: SLF001
        switch.click()
        assert not switch.isChecked(), "開啟失敗了,開關卻還停在開"

    def test_the_spring_back_does_not_re_enter_the_switchboard(self, qtbot) -> None:
        """彈回去不該再送一次 set_enabled —— 那會變成無窮遞迴。"""
        board = FakeSwitchboard(refuse={features.ADVICE})
        window = PanelWindow(ViewModel(), switchboard=board)
        qtbot.addWidget(window)
        window._switches[features.ADVICE].click()  # noqa: SLF001
        assert board.calls == [(features.ADVICE, True)]

    def test_switches_reflect_a_switchboard_that_is_already_on(self, qtbot) -> None:
        board = FakeSwitchboard()
        board.state[features.VISION] = True
        window = PanelWindow(ViewModel(), switchboard=board)
        qtbot.addWidget(window)
        assert window._switches[features.VISION].isChecked()  # noqa: SLF001
        assert not window._switches[features.ADVICE].isChecked()  # noqa: SLF001


class TestSwitchesWithoutASwitchboard:
    """重播與示範模式:命令列已經決定跑哪一條路。"""

    def test_they_are_disabled(self, panel) -> None:
        _, window = panel
        for switch in window._switches.values():  # noqa: SLF001
            assert not switch.isEnabled()

    def test_they_show_as_on(self, panel) -> None:
        """那條路真的在跑,顯示為關會誤導。"""
        _, window = panel
        for switch in window._switches.values():  # noqa: SLF001
            assert switch.isChecked()

    def test_the_tooltip_explains_why(self, panel) -> None:
        _, window = panel
        for switch in window._switches.values():  # noqa: SLF001
            assert "不適用" in switch.toolTip()


class TestOffStateDisplay:
    """功能關著的時候畫面要說「關著」,不能看起來像在載入。

    「等待引擎…」與「等待手牌…」是為「開著但還沒有資料」寫的。功能關著時
    沿用那兩句話,使用者會一直等下去 —— 這是渲染出來才看到的。
    """

    @pytest.fixture
    def wired(self, qtbot):
        board = FakeSwitchboard()
        model = ViewModel()
        window = PanelWindow(model, switchboard=board)
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        return board, model, window

    def test_advice_says_not_enabled_rather_than_waiting(self, wired) -> None:
        _, _, window = wired
        assert window._advice._headline._verb.text() == "未開啟"  # noqa: SLF001

    def test_advice_points_at_the_switch(self, wired) -> None:
        _, _, window = wired
        assert "開關" in window._advice._headline._detail.text()  # noqa: SLF001

    def test_analysis_says_not_enabled_rather_than_waiting(self, wired) -> None:
        _, _, window = wired
        assert window._analysis._shanten.text() == "未開啟"  # noqa: SLF001

    def test_the_status_bar_says_both_are_off(self, wired) -> None:
        _, _, window = wired
        assert "都關著" in window._notice.text()  # noqa: SLF001

    def test_turning_it_on_switches_to_the_waiting_message(self, wired) -> None:
        _, _, window = wired
        window._switches[features.ADVICE].click()  # noqa: SLF001
        assert window._advice._headline._verb.text() == "等待引擎…"  # noqa: SLF001

    def test_a_disabled_feature_does_not_show_stale_data(self, wired) -> None:
        """關掉之後畫面上不能留著最後一手 —— 那看起來像還在運作。"""
        _, model, window = wired
        switch = window._switches[features.VISION]  # noqa: SLF001
        switch.click()
        model.update_packet_hand(TENPAI)
        assert window._analysis._shanten.text() == "聽牌"  # noqa: SLF001

        switch.click()
        assert window._analysis._shanten.text() == "未開啟"  # noqa: SLF001

    def test_replay_mode_shows_content_without_a_switchboard(self, panel) -> None:
        """重播沒有 switchboard,但那條路真的在跑 —— 不該顯示「未開啟」。"""
        model, window = panel
        model.update_packet_hand(TENPAI)
        assert window._analysis._shanten.text() == "聽牌"  # noqa: SLF001


class TestUnavailableFeature:
    """``--no-vision`` / ``--no-packets``:那條路根本沒建起來。

    開關做成可按的話,使用者撥了它會自己彈回去 —— 看起來像壞掉,而實際上是
    他自己在命令列關掉的。
    """

    @pytest.fixture
    def half(self, qtbot):
        board = FakeSwitchboard(missing={features.VISION})
        window = PanelWindow(ViewModel(), switchboard=board)
        qtbot.addWidget(window)
        return board, window

    def test_the_missing_ones_switch_is_disabled(self, half) -> None:
        _, window = half
        assert not window._switches[features.VISION].isEnabled()  # noqa: SLF001

    def test_the_other_one_still_works(self, half) -> None:
        board, window = half
        switch = window._switches[features.ADVICE]  # noqa: SLF001
        assert switch.isEnabled()
        switch.click()
        assert board.is_enabled(features.ADVICE)

    def test_the_tooltip_says_it_was_disabled_at_startup(self, half) -> None:
        _, window = half
        assert "啟動參數" in window._switches[features.VISION].toolTip()  # noqa: SLF001

    def test_it_is_shown_as_off_not_on(self, half) -> None:
        """與重播模式不同:那條路真的沒在跑,顯示為開會誤導。"""
        _, window = half
        assert not window._switches[features.VISION].isChecked()  # noqa: SLF001


class TestOverlayControls:
    """設定頁上的三個勾選。

    **控制項在側邊視窗這一側,不在 Overlay 上。** Overlay 鎖定之後對滑鼠是
    透明的 —— 擺在它自己身上的按鈕會變成看得到卻按不到的死按鈕。
    """

    @pytest.fixture
    def wired(self, qtbot, tmp_path):
        from mia.ui.overlay.window import OverlayWindow
        from mia.ui.state import UiState

        model = ViewModel()
        overlay = OverlayWindow(model, ui_state=UiState.load(tmp_path / "ui.json"))
        window = PanelWindow(model, overlay=overlay)
        qtbot.addWidget(overlay)
        qtbot.addWidget(window)
        return overlay, window

    def test_opening_the_panel_does_not_show_the_overlay(self, wired) -> None:
        """建構時先設好勾選再接訊號 —— 反過來的話 setChecked 會立刻觸發一次
        handler,把「上次記住的狀態」當成使用者剛剛的操作重存一遍。
        """
        overlay, _ = wired
        assert overlay.isHidden()

    def test_checking_it_shows_the_overlay(self, wired) -> None:
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        assert overlay.shown

    def test_unchecking_it_hides_the_overlay(self, wired) -> None:
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        window._overlay_shown.setChecked(False)  # noqa: SLF001
        assert overlay.isHidden()

    def test_the_other_two_are_disabled_until_it_is_shown(self, wired) -> None:
        """收起來的時候「展開」與「鎖定」沒有東西可以作用。"""
        _, window = wired
        assert not window._overlay_expanded.isEnabled()  # noqa: SLF001
        assert not window._overlay_danger.isEnabled()  # noqa: SLF001
        assert not window._overlay_locked.isEnabled()  # noqa: SLF001

    def test_showing_it_enables_the_other_two(self, wired) -> None:
        _, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        assert window._overlay_expanded.isEnabled()  # noqa: SLF001
        assert window._overlay_danger.isEnabled()  # noqa: SLF001
        assert window._overlay_locked.isEnabled()  # noqa: SLF001

    def test_expanding_reaches_the_overlay(self, wired) -> None:
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        window._overlay_expanded.setChecked(True)  # noqa: SLF001
        assert overlay.expanded

    def test_the_danger_list_reaches_the_overlay(self, wired) -> None:
        """放銃分析上不上 HUD 是**另外問**的:它會把 Overlay 往下拉長一整手。"""
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        window._overlay_danger.setChecked(True)  # noqa: SLF001
        assert overlay.danger_shown

    def test_locking_reaches_the_overlay(self, wired) -> None:
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        window._overlay_locked.setChecked(True)  # noqa: SLF001
        assert overlay.locked

    def test_closing_the_panel_closes_the_overlay(self, wired) -> None:
        """少了這一段,顯示中的 Overlay 會是最後一個還開著的視窗,Qt 於是不
        結束程式 —— 畫面上只剩一個關不掉的浮動 HUD。
        """
        overlay, window = wired
        window._overlay_shown.setChecked(True)  # noqa: SLF001
        window.close()
        assert overlay.isHidden()

    def test_a_remembered_state_comes_back_checked(self, qtbot, tmp_path) -> None:
        from mia.ui.overlay.window import OverlayWindow
        from mia.ui.state import UiState

        state = UiState.load(tmp_path / "ui.json")
        state.overlay_visible = True
        state.overlay_expanded = True
        model = ViewModel()
        overlay = OverlayWindow(model, ui_state=state)
        window = PanelWindow(model, overlay=overlay)
        qtbot.addWidget(overlay)
        qtbot.addWidget(window)
        assert window._overlay_shown.isChecked()  # noqa: SLF001
        assert window._overlay_expanded.isChecked()  # noqa: SLF001


class TestWithoutAnOverlay:
    """重播與示範以外也可能沒有 Overlay(測試就是)—— 勾選要畫成停用。"""

    def test_the_boxes_are_disabled(self, panel) -> None:
        _, window = panel
        assert not window._overlay_shown.isEnabled()  # noqa: SLF001
        assert not window._overlay_expanded.isEnabled()  # noqa: SLF001
        assert not window._overlay_danger.isEnabled()  # noqa: SLF001
        assert not window._overlay_locked.isEnabled()  # noqa: SLF001

    def test_the_tooltip_explains_why(self, panel) -> None:
        _, window = panel
        assert "Overlay" in window._overlay_shown.toolTip()  # noqa: SLF001


class FakeLauncher:
    """滿足 :class:`~mia.ui.switchboard.GameLauncher` 的最小實作。"""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.running = False
        self.calls = 0

    def can_start_game(self) -> bool:
        return self.available

    def game_running(self) -> bool:
        return self.running

    def start_game(self) -> None:
        self.calls += 1
        self.running = True


class TestStartGameButton:
    """左下角那顆按鈕。瀏覽器不會自己開,要按了才開。"""

    @pytest.fixture
    def wired(self, qtbot):
        launcher = FakeLauncher()
        model = ViewModel()
        window = PanelWindow(model, switchboard=FakeSwitchboard(), launcher=launcher)
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        return launcher, window

    def test_opening_the_window_does_not_open_a_browser(self, wired) -> None:
        """開一個瀏覽器並開始往磁碟寫錄影檔該是明確的動作,不是視窗的副作用。"""
        launcher, _ = wired
        assert launcher.calls == 0

    def test_it_is_ready_to_press(self, wired) -> None:
        _, window = wired
        assert window._start_button.isEnabled()  # noqa: SLF001
        assert window._start_button.text() == "開始遊戲"  # noqa: SLF001

    def test_pressing_it_starts_the_game(self, wired) -> None:
        launcher, window = wired
        window._start_button.click()  # noqa: SLF001
        assert launcher.calls == 1

    def test_it_goes_dead_while_a_game_is_running(self, wired) -> None:
        """關掉瀏覽器等於把那一場丟掉 —— 不該是一個手滑就按得到的按鈕。"""
        _, window = wired
        window._start_button.click()  # noqa: SLF001
        assert not window._start_button.isEnabled()  # noqa: SLF001
        assert window._start_button.text() == "遊戲進行中"  # noqa: SLF001

    def test_it_comes_back_when_the_browser_is_closed(self, wired) -> None:
        """使用者把瀏覽器關掉之後要能再開一場,不必重開 MIA。"""
        launcher, window = wired
        window._start_button.click()  # noqa: SLF001
        launcher.running = False
        window.apply(ViewModel().state)
        assert window._start_button.isEnabled()  # noqa: SLF001

    def test_the_status_bar_points_at_the_button(self, wired) -> None:
        """功能開關的提示要讓位:打開了功能卻沒有遊戲,使用者會盯著空白等。"""
        _, window = wired
        assert "開始遊戲" in window._notice.text()  # noqa: SLF001

    def test_only_the_danger_feature_on_is_not_all_off(self, qtbot) -> None:
        """**實機踩到的。** 狀態列原本寫死問 ADVICE / VISION,所以只開放銃
        分析的時候照樣說「功能都關著」—— 而那一頁上明明有東西在跑。

        用 ``features.NAMES`` 逐一問,加第四個功能時也不會再壞一次。
        """
        launcher = FakeLauncher()
        launcher.running = True
        board = FakeSwitchboard()
        board.set_enabled(features.DANGER, True)
        model = ViewModel()
        window = PanelWindow(model, switchboard=board, launcher=launcher)
        qtbot.addWidget(window)
        window.apply(model.state)
        assert "功能都關著" not in window._notice.text()  # noqa: SLF001

    def test_the_status_bar_moves_on_once_a_game_is_running(self, wired) -> None:
        _, window = wired
        window._start_button.click()  # noqa: SLF001
        assert "功能都關著" in window._notice.text()  # noqa: SLF001


class TestStartGameWithoutALauncher:
    def test_replay_mode_disables_the_button(self, panel) -> None:
        _, window = panel
        assert not window._start_button.isEnabled()  # noqa: SLF001
        assert "重播" in window._start_button.toolTip()  # noqa: SLF001

    def test_tail_mode_says_it_is_not_ours_to_open(self, qtbot) -> None:
        """--tail 是跟著別人正在錄的檔案走,遊戲不是 MIA 開的。"""
        window = PanelWindow(ViewModel(), launcher=FakeLauncher(available=False))
        qtbot.addWidget(window)
        assert not window._start_button.isEnabled()  # noqa: SLF001
        assert "--tail" in window._start_button.toolTip()  # noqa: SLF001


class FakeCanvasPicker:
    """實作 :class:`~mia.ui.switchboard.CanvasPicker` 的假物件。"""

    def __init__(self, *, available: bool = True, current: str | None = None) -> None:
        self.available = available
        self.current = current
        self.calls: list[str | None] = []

    def can_pick_canvas(self) -> bool:
        return self.available

    def canvas(self) -> str | None:
        return self.current

    def set_canvas(self, key: str | None) -> None:
        self.calls.append(key)
        self.current = key


class TestCanvasPicker:
    """設定頁上的「遊戲視窗尺寸」。

    **在設定頁而不是向聽分析頁。** 它同時調整瀏覽器視窗與牌桌校正基準,而
    使用者心裡它就是「遊戲視窗要開多大」—— 那是應用程式層級的設定,不是
    某一頁的顯示選項。

    它存在的理由見 :mod:`mia.calibration.canvas` —— 自動偵測在某些視窗尺寸
    下會安靜地給出偏掉 0.7 張牌寬的矩形。
    """

    def _window(self, qtbot, picker):
        window = PanelWindow(ViewModel(), switchboard=FakeSwitchboard(), canvas=picker)
        qtbot.addWidget(window)
        return window

    def test_it_lives_on_the_settings_page(self, qtbot) -> None:
        """迴歸測試:一開始放在向聽分析頁,使用者指名要在設定頁。"""
        from PySide6.QtWidgets import QComboBox

        window = self._window(qtbot, FakeCanvasPicker())
        picker = window._canvas_picker  # noqa: SLF001
        on_settings = window._settings_page.findChildren(QComboBox)  # noqa: SLF001
        on_analysis = window._analysis_page.findChildren(QComboBox)  # noqa: SLF001
        assert picker in on_settings, "選單不在設定頁上"
        assert picker not in on_analysis, "選單還留在向聽分析頁"

    def test_auto_detect_is_not_offered(self, qtbot) -> None:
        """自動偵測是猜的,實測會安靜地鎖進偏掉 0.7 張牌寬的矩形。

        它仍然是螢幕放不下任何尺寸時的退路,但不該擺在選單上邀請使用者去選
        一個已知比較差的做法。
        """
        window = self._window(qtbot, FakeCanvasPicker())
        picker = window._canvas_picker  # noqa: SLF001
        assert all(picker.itemData(i) is not None for i in range(picker.count()))
        assert "自動" not in " ".join(picker.itemText(i) for i in range(picker.count()))

    def test_a_typed_size_is_accepted(self, qtbot) -> None:
        """預設清單只有四個,而使用者的螢幕未必剛好是那幾種。"""
        picker = FakeCanvasPicker()
        window = self._window(qtbot, picker)
        window._canvas_picker.setCurrentText("1440x810")  # noqa: SLF001
        window._on_canvas_typed()  # noqa: SLF001
        assert picker.calls == ["1440x810"]

    def test_a_typed_size_is_normalised_for_display(self, qtbot) -> None:
        window = self._window(qtbot, FakeCanvasPicker())
        window._canvas_picker.setCurrentText(" 1440 X 810 ")  # noqa: SLF001
        window._on_canvas_typed()  # noqa: SLF001
        assert window._canvas_picker.currentText() == "1440×810"  # noqa: SLF001

    def test_nonsense_reverts_instead_of_sticking(self, qtbot) -> None:
        """看不懂時**不能留著那串字** —— 選單顯示 1440x81o 而實際跑
        1280x720,那是最糟的一種:畫面說一套、程式做另一套。"""
        picker = FakeCanvasPicker(current="1280x720")
        window = self._window(qtbot, picker)
        window._canvas_picker.setCurrentText("大一點")  # noqa: SLF001
        window._on_canvas_typed()  # noqa: SLF001
        assert picker.calls == []
        assert window._canvas_picker.currentText() == "1280×720"  # noqa: SLF001

    def test_every_preset_is_offered(self, qtbot) -> None:
        window = self._window(qtbot, FakeCanvasPicker())
        offered = {
            window._canvas_picker.itemData(i)  # noqa: SLF001
            for i in range(window._canvas_picker.count())  # noqa: SLF001
        }
        assert {p.key for p in CANVAS_PRESETS} <= offered

    def test_it_starts_on_the_remembered_choice(self, qtbot) -> None:
        """記住的尺寸要**選起來**,不是只存著。

        選單停在「自動偵測」而實際上跑的是 1920x1080 的話,使用者會以為
        上次的設定沒存到而再選一次 —— 那次選擇會被當成「沒有改變」忽略掉。
        """
        window = self._window(qtbot, FakeCanvasPicker(current="1920x1080"))
        assert window._canvas_picker.currentText() == "1920×1080"  # noqa: SLF001

    def test_picking_forwards_the_key(self, qtbot) -> None:
        # 起始值刻意不是要選的那個 —— 選同一項不會觸發 currentIndexChanged,
        # 那樣測到的是 Qt 的去重,不是我們的接線。
        picker = FakeCanvasPicker(current="1280x720")
        window = self._window(qtbot, picker)
        index = window._canvas_picker.findData("1920x1080")  # noqa: SLF001
        window._canvas_picker.setCurrentIndex(index)  # noqa: SLF001
        assert picker.calls == ["1920x1080"]

    def test_it_is_disabled_when_nobody_can_act_on_it(self, qtbot) -> None:
        """重播與 ``--no-vision``:可按而按了沒事發生,看起來就是壞掉。"""
        window = self._window(qtbot, FakeCanvasPicker(available=False))
        assert not window._canvas_picker.isEnabled()  # noqa: SLF001

    def test_it_is_disabled_without_a_picker_at_all(self, qtbot) -> None:
        window = PanelWindow(ViewModel(), switchboard=FakeSwitchboard())
        qtbot.addWidget(window)
        assert not window._canvas_picker.isEnabled()  # noqa: SLF001
