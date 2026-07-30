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
        """設定頁的內容本身就是設定,頂端不該再留一條空白的橫線。"""
        _, window = panel
        settings_page = window._stack.widget(2)  # noqa: SLF001
        assert not shown(settings_page._settings)  # noqa: SLF001

    def test_both_feature_pages_show_their_strip(self, panel) -> None:
        """兩個功能頁都有開關,所以設定列一定看得到。"""
        _, window = panel
        for index in (0, 1):
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

    def test_both_switches_exist(self, wired) -> None:
        _, window = wired
        assert set(window._switches) == {features.ADVICE, features.VISION}  # noqa: SLF001

    def test_both_default_to_off(self, wired) -> None:
        """使用者要求的:預設兩個都不執行。"""
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
