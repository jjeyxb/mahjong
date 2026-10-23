"""側邊視窗的 widget。

只驗**接線**:狀態換了畫面有沒有跟著換、牌面圖找不找得到、widget 池會不會
隨手牌變短而留下上一手的殘影。好不好看不在這裡驗 —— 那要人眼看,
用 ``python tools/ui.py --demo``。

牌面那幾個例外**驗的是像素**,但驗的不是好不好看而是「有沒有東西被切掉」:
圓角削到圖案、外框線畫在圖案上,這些是看得出對錯的,而且只影響 37 張裡的
十幾張 —— 剛好是人眼最容易看漏、又最容易被當成素材本來就長這樣的那種。

在無頭環境跑,所以 conftest 會設 ``QT_QPA_PLATFORM=offscreen``。
"""

from __future__ import annotations

import math

import pytest

from mia import features, platforms
from mia.engine.base import Advice
from mia.mjai import Dahai, Reach
from mia.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QPushButton

from mia.calibration.canvas import PRESETS as CANVAS_PRESETS
from mia.ui.panel.window import PanelWindow, present
from mia.ui.widgets.tiles import (
    _PAD_FRACTION,
    _RADIUS_FRACTION,
    TILE_ASPECT,
    TileIcons,
)
from mia.utils.geometry import NormRect

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
        pixmap = icons.pixmap("1m", 105)
        assert pixmap.height() == 105
        assert pixmap.width() == round(105 * TILE_ASPECT)

    def test_the_grey_edge_is_cropped_off(self) -> None:
        """素材上面那條灰邊是遊戲裡牌的**上緣**,不是牌面。

        比對時每張都一樣所以無所謂,畫在 UI 上就是每張牌頭上一條髒灰條。
        切掉之後牌面變短,所以顯示用的寬高比比素材本身寬。
        """
        assert TILE_ASPECT > 64 / 109

    def test_it_renders_at_device_pixels(self) -> None:
        """2× 螢幕上要畫出兩倍的像素,不然是先縮小再被系統放大 —— 一排糊牌。

        在 1× 螢幕上測完全看不出這個問題,所以要釘著。
        """
        icons = TileIcons()
        one = icons.pixmap("1m", 40, 1.0)
        two = icons.pixmap("1m", 40, 2.0)
        assert two.height() == 2 * one.height()
        assert two.devicePixelRatio() == 2.0
        # 掛上 devicePixelRatio 之後版面上佔的高度仍然是 40
        assert two.deviceIndependentSize().height() == 40

    def test_the_padding_clears_the_rounded_corner(self) -> None:
        """留白要大到讓圖案整塊落在圓弧**裡面**。

        內矩形的角到圓心的距離是 √2·(r−pad),要 ≤ r —— 也就是
        ``pad ≥ r·(1 − 1/√2)``。有人調大圓角卻忘了調留白的話,症狀只出現在
        十幾張牌的角落(索子兩側的竹節被削掉一點),很容易被當成
        「素材本來就長這樣」而不是程式的問題。
        """
        assert _RADIUS_FRACTION * (1 - 1 / math.sqrt(2)) <= _PAD_FRACTION

    def test_no_tile_loses_ink_to_the_rounded_corner(self) -> None:
        """37 張牌,圖案都要完整落在圓角**裡面**。

        素材的 ROI 是貼著圖案切的 —— 量過,``3s`` ``6s`` ``7p`` ``1p`` 等十幾張
        的墨色直接碰到左右或下緣。沒有留白的話圓角會削掉索子兩側的竹節、
        外框線直接畫在圖案上。使用者的原話是「很多牌的顯示都被裁切到了」。

        驗法:取素材裡墨色的外接矩形,換算到畫出來的座標,要求它的**四個角**
        都落在圓角路徑內。外接矩形比墨色本身大,所以這是偏嚴的檢查 —— 它過了
        就保證沒有一個墨色像素被切到。

        這裡檢查素材而不只是檢查 ``_PAD_FRACTION``:留白夠不夠是幾何問題
        (另一條測試釘著),而圖案有沒有貼著邊是**素材**的問題 —— 換一套皮膚
        之後前者仍然成立,後者可能不成立。
        """
        icons = TileIcons()
        tiles = [f"{n}{s}" for s in "mps" for n in range(1, 10)]
        tiles += ["E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]
        height = 109
        pad = height * _PAD_FRACTION
        radius = height * _RADIUS_FRACTION
        width = height * TILE_ASPECT

        def inside(x: float, y: float) -> bool:
            """(x, y) 在圓角矩形內嗎。角落以外都是直邊,只有角落要算距離。"""
            cx = min(max(x, radius), width - radius)
            cy = min(max(y, radius), height - radius)
            return math.hypot(x - cx, y - cy) <= radius

        clipped = []
        for tile in tiles:
            face = icons._face(tile).toImage()  # noqa: SLF001
            box = _ink_box(face)
            if box is None:  # 白板沒有墨色
                continue
            scale_x = (width - 2 * pad) / face.width()
            scale_y = (height - 2 * pad) / face.height()
            left, top, right, bottom = box
            corners = [
                (pad + left * scale_x, pad + top * scale_y),
                (pad + right * scale_x, pad + top * scale_y),
                (pad + left * scale_x, pad + bottom * scale_y),
                (pad + right * scale_x, pad + bottom * scale_y),
            ]
            if not all(inside(x, y) for x, y in corners):
                clipped.append(tile)
        assert clipped == [], f"這些牌的圖案會被圓角切到:{clipped}"

    def test_the_corners_are_transparent(self) -> None:
        """圓角要真的是透明的,不是畫上去的白角 —— 深色模式下白角會很明顯。"""
        icons = TileIcons()
        image = icons.pixmap("1m", 60).toImage()
        assert image.pixelColor(0, 0).alpha() == 0

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
        assert not shown(window._advice._candidates_header)  # noqa: SLF001


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


class TestCandidateCount:
    """「列幾個」。它在候選清單的標題旁邊,不在頂端的設定列上 ——
    設定列只放會**重啟引擎**的東西(風格、比較對象),而這個只是少畫幾列。
    """

    def test_the_candidate_count_is_adjustable(self, panel) -> None:
        model, window = panel
        meta = {"mask_bits": 0b1111, "q_values": [0.1, 0.2, 0.3, 0.4]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
        )
        window._advice._candidate_count.setValue(2)  # noqa: SLF001
        assert sum(shown(row) for row in window._advice._rows) == 2  # noqa: SLF001

    def test_it_redraws_with_the_current_state(self, panel) -> None:
        """調了之後要**立刻**重畫,不是等下一個事件。

        等下一個事件的話,使用者拉了數字畫面不動 —— 那看起來像控制項壞了,
        而下一手打出來時它又突然對了,更難理解。
        """
        model, window = panel
        meta = {"mask_bits": 0b1111, "q_values": [0.1, 0.2, 0.3, 0.4]}
        model.update_advices(
            [Advice("mortal", Dahai(actor=0, pai="1m", tsumogiri=False), meta, 1.0)]
        )
        window._advice._candidate_count.setValue(1)  # noqa: SLF001
        assert sum(shown(row) for row in window._advice._rows) == 1  # noqa: SLF001
        window._advice._candidate_count.setValue(4)  # noqa: SLF001
        assert sum(shown(row) for row in window._advice._rows) == 4  # noqa: SLF001


class FakeStylePicker:
    """滿足 :class:`~mia.ui.switchboard.StylePicker` 的最小實作。"""

    def __init__(self, names: tuple[str, ...] = ("標準", "愛鳴牌"), current: str = "標準") -> None:
        self._names = names
        self._current = current
        self._compare: str | None = None
        self.asked: list[str] = []
        self.asked_compare: list[str | None] = []

    def can_pick_style(self) -> bool:
        return bool(self._names)

    def styles(self) -> tuple[str, ...]:
        return self._names

    def style(self) -> str | None:
        return self._current

    def set_style(self, name: str) -> None:
        self.asked.append(name)
        self._current = name
        if self._compare == name:
            # 真的那一份(StyleChoice.set)也是這樣:不能跟自己比
            self._compare = None

    def compare(self) -> str | None:
        return self._compare

    def set_compare(self, name: str | None) -> None:
        self.asked_compare.append(name)
        self._compare = None if name == self._current else name


class TestStylePicker:
    """風格下拉選單。

    使用者的原話:「主要只有自動能選,沒有風格讓我調整」—— 當時設定列上只有
    一個「主要引擎」下拉,而那個東西只決定已經在跑的那幾個裡誰當 headline,
    **換不了權重**。這個選單才是換權重的那一個。(「主要引擎」後來拿掉了。)
    """

    def test_the_configured_styles_are_listed(self, qtbot) -> None:  # noqa: ARG002
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        combo = window._style_picker  # noqa: SLF001
        assert [combo.itemData(i) for i in range(combo.count())] == ["標準", "愛鳴牌"]

    def test_it_starts_on_the_current_style(self, qtbot) -> None:  # noqa: ARG002
        """選單一打開就要顯示現在跑的那個,不然看起來像沒生效。"""
        window = PanelWindow(ViewModel(), style=FakeStylePicker(current="愛鳴牌"))
        assert window._style_picker.currentText() == "愛鳴牌"  # noqa: SLF001

    def test_picking_one_asks_for_the_change(self, qtbot) -> None:  # noqa: ARG002
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        window._style_picker.setCurrentIndex(1)  # noqa: SLF001
        assert picker.asked == ["愛鳴牌"]

    def test_showing_the_current_style_does_not_ask_for_a_change(self, qtbot) -> None:  # noqa: ARG002
        """初始化時把選單切到現在那個,不能被當成使用者按的 —— 那會在開窗的
        瞬間重啟一次引擎(重載 130MB 權重)。
        """
        picker = FakeStylePicker(current="愛鳴牌")
        PanelWindow(ViewModel(), style=picker)
        assert picker.asked == []

    def test_without_a_picker_the_combo_is_disabled(self, panel) -> None:
        """重播與 ``--no-packets`` 的引擎是命令列決定的。可按而按了沒事發生,
        看起來就是壞掉。
        """
        _, window = panel
        assert not window._style_picker.isEnabled()  # noqa: SLF001

    def test_no_available_styles_disables_the_combo(self, qtbot) -> None:  # noqa: ARG002
        """權重不隨專案散布 —— 別人的機器上可能一份都沒有。"""
        window = PanelWindow(ViewModel(), style=FakeStylePicker(names=()))
        assert not window._style_picker.isEnabled()  # noqa: SLF001


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

    def __init__(self, refuse: set[str] | None = None, missing: set[str] | None = None) -> None:
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
    """三個功能的開關。預設關著,打開才會跑。"""

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


class TestComparePicker:
    """「比較對象」選單。使用者的原話:「可以設計成能自己選擇要哪兩個模型」。"""

    def test_it_offers_everything_except_the_primary(self, qtbot) -> None:  # noqa: ARG002
        """主要風格要從清單裡排除,而不是選了再擋。

        選單上出現一個按下去沒反應的項目,比它根本不在那裡糟 ——
        使用者會以為程式壞了。
        """
        window = PanelWindow(ViewModel(), style=FakeStylePicker(current="標準"))
        combo = window._compare_picker  # noqa: SLF001
        assert [combo.itemData(i) for i in range(combo.count())] == [None, "愛鳴牌"]

    def test_it_starts_on_no_comparison(self, qtbot) -> None:  # noqa: ARG002
        window = PanelWindow(ViewModel(), style=FakeStylePicker())
        assert window._compare_picker.currentData() is None  # noqa: SLF001

    def test_picking_one_asks_for_it(self, qtbot) -> None:  # noqa: ARG002
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        window._compare_picker.setCurrentIndex(1)  # noqa: SLF001
        assert picker.asked_compare == ["愛鳴牌"]

    def test_going_back_to_none_asks_for_none(self, qtbot) -> None:  # noqa: ARG002
        """「不比較」是個合法的值,不能像換風格那樣把 None 擋掉。"""
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        window._compare_picker.setCurrentIndex(1)  # noqa: SLF001
        window._compare_picker.setCurrentIndex(0)  # noqa: SLF001
        assert picker.asked_compare == ["愛鳴牌", None]

    def test_switching_the_primary_rebuilds_the_list(self, qtbot) -> None:  # noqa: ARG002
        """換了主要風格之後,被排除的那一項要跟著換。"""
        window = PanelWindow(ViewModel(), style=FakeStylePicker(current="標準"))
        window._style_picker.setCurrentIndex(1)  # noqa: SLF001  → 愛鳴牌
        combo = window._compare_picker  # noqa: SLF001
        assert [combo.itemData(i) for i in range(combo.count())] == [None, "標準"]

    def test_rebuilding_the_list_does_not_restart_the_engine(self, qtbot) -> None:  # noqa: ARG002
        """**這一條是陷阱。**

        ``clear()`` 與 ``addItem()`` 都會發 ``currentIndexChanged``,而接在上面
        的 slot 會去 ``set_compare`` —— 沒擋訊號的話「重建一次選單」就等於
        「重啟一次引擎」,每份權重 130MB。
        """
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        window._style_picker.setCurrentIndex(1)  # noqa: SLF001
        assert picker.asked_compare == []

    def test_a_cleared_comparison_shows_as_none(self, qtbot) -> None:  # noqa: ARG002
        """主要風格換成目前的比較對象時,比較被關掉 —— 選單上的字要跟著改。

        留著就是畫面寫著一個已經不生效的選擇。
        """
        picker = FakeStylePicker()
        window = PanelWindow(ViewModel(), style=picker)
        window._compare_picker.setCurrentIndex(1)  # noqa: SLF001  → 愛鳴牌
        window._style_picker.setCurrentIndex(1)  # noqa: SLF001  → 主要也換成愛鳴牌
        assert window._compare_picker.currentData() is None  # noqa: SLF001

    def test_it_is_disabled_when_styles_are_fixed(self, qtbot) -> None:  # noqa: ARG002
        """重播 / --no-packets 時沒有引擎那條路,兩個選單都該停用。"""
        window = PanelWindow(ViewModel(), style=FakeStylePicker(names=()))
        assert not window._compare_picker.isEnabled()  # noqa: SLF001


def _ink_box(image: QImage) -> tuple[int, int, int, int] | None:
    """影像裡「明顯比牌面底色暗」的那些像素的外接矩形。

    門檻 190 是量出來的:牌面底色約 218,最淡的墨色(赤 5 的紅)遠低於它。
    """
    xs: list[int] = []
    ys: list[int] = []
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if (colour.red() + colour.green() + colour.blue()) / 3 < 190:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


class FakePlatformPicker:
    """滿足 :class:`~mia.ui.switchboard.PlatformPicker` 的最小實作。

    與真的那一條線(``LiveRuntime.set_platform``)一樣:換到 Steam 版會把
    沒有來源的功能**關掉**,所以這裡也要動 switchboard —— 不然測的就不是
    使用者會看到的那個狀態。
    """

    def __init__(self, switchboard: FakeSwitchboard, value: str = platforms.WEB) -> None:
        self._switchboard = switchboard
        self._value = value
        self.asked: list[str] = []

    def can_pick_platform(self) -> bool:
        return True

    def platform(self) -> str:
        return self._value

    def set_platform(self, value: str) -> None:
        self.asked.append(value)
        self._value = value
        for key in tuple(features.NAMES):
            if not platforms.supports(value, key):
                self._switchboard.missing.add(key)
                self._switchboard.state[key] = False
            else:
                self._switchboard.missing.discard(key)

    def unavailable_reason(self, key: str) -> str | None:
        return platforms.why_unsupported(self._value, key)


class TestPlatformPicker:
    """遊戲版本的選單。選了 Steam 版之後,沒有來源的功能要停用並說明原因。"""

    def _window(self, qtbot, value: str = platforms.WEB):
        board = FakeSwitchboard()
        picker = FakePlatformPicker(board, value)
        window = PanelWindow(ViewModel(), switchboard=board, platform=picker)
        qtbot.addWidget(window)
        return window, picker, board

    def test_both_versions_are_listed(self, qtbot) -> None:
        window, _, _ = self._window(qtbot)
        combo = window._platform_picker  # noqa: SLF001
        assert [combo.itemData(i) for i in range(combo.count())] == list(platforms.PLATFORMS)

    def test_it_starts_on_the_remembered_version(self, qtbot) -> None:
        """記住的是「我現在都玩 Steam 版」,不是臨時的調整。"""
        window, _, _ = self._window(qtbot, platforms.STEAM)
        combo = window._platform_picker  # noqa: SLF001
        assert combo.currentData() == platforms.STEAM

    def test_picking_steam_reaches_the_runtime(self, qtbot) -> None:
        window, picker, _ = self._window(qtbot)
        combo = window._platform_picker  # noqa: SLF001
        combo.setCurrentIndex(list(platforms.PLATFORMS).index(platforms.STEAM))
        assert picker.asked == [platforms.STEAM]

    def test_steam_disables_the_switches_it_cannot_feed(self, qtbot) -> None:
        """使用者要看到的是「停用」,不是一個撥了會自己彈回去的開關。"""
        window, _, _ = self._window(qtbot)
        combo = window._platform_picker  # noqa: SLF001
        combo.setCurrentIndex(list(platforms.PLATFORMS).index(platforms.STEAM))
        switches = window._switches  # noqa: SLF001
        assert switches[features.VISION].isEnabled()
        assert not switches[features.ADVICE].isEnabled()
        assert not switches[features.DANGER].isEnabled()

    def test_the_disabled_switch_says_why(self, qtbot) -> None:
        """「不支援」會讓人去找一個不存在的開關,所以要講為什麼。"""
        window, _, _ = self._window(qtbot)
        combo = window._platform_picker  # noqa: SLF001
        combo.setCurrentIndex(list(platforms.PLATFORMS).index(platforms.STEAM))
        tip = window._switches[features.ADVICE].toolTip()  # noqa: SLF001
        assert "封包" in tip

    def test_no_picker_means_the_menu_is_disabled(self, qtbot) -> None:
        """重播與示範模式裡「現在玩哪一版」是錄的時候就定了的。"""
        window = PanelWindow(ViewModel())
        qtbot.addWidget(window)
        assert not window._platform_picker.isEnabled()  # noqa: SLF001


class FakeTableRectPicker:
    """假的牌桌框選接線。不開對話框 —— 那需要一張真的遊戲畫面。"""

    def __init__(self, *, can_pick: bool = True, framed: NormRect | None = None) -> None:
        self._can_pick = can_pick
        self._rect = framed
        self.calls: list[NormRect | None] = []

    def can_pick_table_rect(self) -> bool:
        return self._can_pick

    def grab_frame(self):  # pragma: no cover - 這些測試不走到抓畫面
        raise AssertionError("這組測試不該抓畫面")

    def table_rect(self) -> NormRect | None:
        return self._rect

    def set_table_rect(self, rect: NormRect | None) -> None:
        self._rect = rect
        self.calls.append(rect)


class TestTableRectPicker:
    """向聽分析頁上的「框選牌桌範圍」。

    自動校正是從四邊剝除邊框,而它**錯的時候很安靜** —— 實測鎖在載入畫面上
    之後,整場的手牌辨識從 97% 掉到 26%,全程沒有任何錯誤訊息。
    """

    def _window(self, qtbot, picker):
        window = PanelWindow(ViewModel(), switchboard=FakeSwitchboard(), table_rect=picker)
        qtbot.addWidget(window)
        return window

    def test_it_lives_on_the_analysis_page(self, qtbot) -> None:
        """與「框選手牌範圍」並排 —— 兩個都是對著這一頁的內容在調的校正,
        而且框完這一頁當場就會變。設定頁放的是「整個程式的前提」。
        """
        window = self._window(qtbot, FakeTableRectPicker())
        on_analysis = window._analysis_page.findChildren(QPushButton)  # noqa: SLF001
        assert window._table_rect_button in on_analysis  # noqa: SLF001

    def test_clear_is_disabled_until_something_is_framed(self, qtbot) -> None:
        """沒框過的時候「改回自動」沒有意義 —— 按了什麼都不會發生,
        而一顆按了沒反應的按鈕讓人以為程式壞了。
        """
        window = self._window(qtbot, FakeTableRectPicker())
        assert not window._table_rect_clear.isEnabled()  # noqa: SLF001

    def test_a_remembered_frame_shows_as_reframe(self, qtbot) -> None:
        """上次框過的話按鈕要說「重新框選」—— 否則使用者看不出記住了沒。"""
        window = self._window(qtbot, FakeTableRectPicker(framed=NormRect(0, 0.1, 1, 0.8)))
        assert window._table_rect_button.text() == "重新框選"  # noqa: SLF001
        assert window._table_rect_clear.isEnabled()  # noqa: SLF001

    def test_clearing_forwards_none(self, qtbot) -> None:
        picker = FakeTableRectPicker(framed=NormRect(0, 0.1, 1, 0.8))
        window = self._window(qtbot, picker)
        window._on_clear_table_rect()  # noqa: SLF001
        assert picker.calls == [None]
        assert window._table_rect_button.text() == "框選牌桌範圍"  # noqa: SLF001

    def test_modes_without_a_screen_disable_it(self, qtbot) -> None:
        """重播、示範、--no-vision 都沒有畫面可以框。"""
        window = self._window(qtbot, FakeTableRectPicker(can_pick=False))
        assert not window._table_rect_button.isEnabled()  # noqa: SLF001

    def test_no_picker_at_all_disables_it(self, qtbot) -> None:
        window = PanelWindow(ViewModel(), switchboard=FakeSwitchboard())
        qtbot.addWidget(window)
        assert not window._table_rect_button.isEnabled()  # noqa: SLF001
