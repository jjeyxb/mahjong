"""Overlay 視窗。

驗的是**接線與狀態**,不是像素:哪些東西該顯示、哪些該藏起來、鎖定有沒有真的
讓滑鼠穿過去、拖完的位置有沒有存回去。長什麼樣要人眼看,用
``python tools/ui.py --demo`` 然後到設定頁勾起來。

跟側邊視窗一樣在無頭環境跑(conftest 設了 ``QT_QPA_PLATFORM=offscreen``),
所以一律用 ``isHidden()`` 而不是 ``isVisible()`` —— 後者對沒 show 過的視窗
底下的子元件一律回 False,測不出我們自己下的 setVisible。
"""

from __future__ import annotations

import pytest

from mia import features
from mia.engine.base import Advice
from mia.mjai import Chi, Dahai
from mia.mjai.tiles import UNKNOWN
from mia.ui.state import UiState
from mia.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import QPoint, Qt

from mia.ui.overlay.window import OverlayWindow

TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]

#: 碰 / 槓 / 跳過:三個位元,最高分落在「跳過」。
SKIP_META = {"mask_bits": (1 << 41) | (1 << 42) | (1 << 45), "q_values": [-6.3, -6.55, -0.15]}


class FakeSwitchboard:
    """只做 :class:`~mia.ui.switchboard.Switchboard` 要求的那三件事。"""

    def __init__(self, on: set[str] | None = None) -> None:
        self.on = on if on is not None else set()

    def available(self, key: str) -> bool:  # noqa: ARG002
        return True

    def is_enabled(self, key: str) -> bool:
        return key in self.on

    def set_enabled(self, key: str, on: bool) -> None:
        self.on.add(key) if on else self.on.discard(key)


@pytest.fixture
def overlay(qtbot, tmp_path):
    """一個沒有 switchboard 的 Overlay —— 重播與示範模式的樣子(一律視為開著)。

    狀態檔指到 tmp_path,免得測試把使用者真正的 ``data/ui_state.json`` 蓋掉。
    """
    model = ViewModel()
    window = OverlayWindow(model, ui_state=UiState.load(tmp_path / "ui.json"))
    model.subscribe(window.apply)
    qtbot.addWidget(window)
    return model, window


def shown(widget) -> bool:
    return not widget.isHidden()


class TestVisibility:
    def test_it_starts_hidden(self, overlay) -> None:
        """勾了才出現。Overlay 蓋在遊戲畫面上,不該是打開程式的副作用。"""
        _, window = overlay
        assert window.shown is False
        assert window.isHidden()

    def test_showing_it_remembers_the_choice(self, overlay) -> None:
        _, window = overlay
        window.set_shown(True)
        assert window.shown is True
        assert UiState.load(window._ui_state.path).overlay_visible is True  # noqa: SLF001

    def test_hiding_it_remembers_too(self, overlay) -> None:
        _, window = overlay
        window.set_shown(True)
        window.set_shown(False)
        assert window.isHidden()
        assert UiState.load(window._ui_state.path).overlay_visible is False  # noqa: SLF001

    def test_a_hidden_overlay_catches_up_when_shown(self, overlay) -> None:
        """隱藏期間不重畫(即時模式一秒 10 次,沒在看時是白花的),所以顯示
        之前一定要補畫 —— 少了那一步,勾開會看到上一次的殘影。
        """
        model, window = overlay
        model.update_advices([Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)])
        window.set_shown(True)
        assert window._verb.text() == "切"  # noqa: SLF001


class TestAdviceLine:
    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_a_discard_shows_the_verb_and_the_tile(self, overlay) -> None:
        model, window = overlay
        model.update_advices([Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)])
        assert window._verb.text() == "切"  # noqa: SLF001
        assert shown(window._subject)  # noqa: SLF001

    def test_a_chi_shows_both_of_our_tiles(self, overlay) -> None:
        """``吃 3m`` 可以用 1m2m、2m4m 或 4m5m —— 精簡版也不能省掉這個。"""
        model, window = overlay
        model.update_advices(
            [Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 1.0)]
        )
        assert window._verb.text() == "吃"  # noqa: SLF001
        assert window._joiner.text() == "用"  # noqa: SLF001
        assert [shown(t) for t in window._own] == [True, True, False]  # noqa: SLF001

    def test_a_shorter_action_hides_the_leftover_tiles(self, overlay) -> None:
        """吃完換成切牌,那兩張不收起來就變成殘影。"""
        model, window = overlay
        model.update_advices(
            [Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 1.0)]
        )
        model.update_advices([Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)])
        assert not any(shown(t) for t in window._own)  # noqa: SLF001

    def test_a_skip_is_an_answer(self, overlay) -> None:
        """遊戲跳出「碰 / 槓 / 跳過」時大字是「跳過」,不是一片空白。"""
        model, window = overlay
        model.update_advices([Advice("m", None, SKIP_META, 1.0)])
        assert window._verb.text() == "跳過"  # noqa: SLF001

    def test_a_played_out_advice_is_marked(self, overlay) -> None:
        """建議切 3s 但手上沒有 3s —— 那是剛剛打掉了,不是程式算錯。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        model.update_advices([Advice("m", Dahai(actor=0, pai="9s", tsumogiri=False), None, 1.0)])
        assert shown(window._mark)  # noqa: SLF001
        assert "已打出" in window._mark.text()  # noqa: SLF001


class TestShantenLine:
    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_a_tenpai_hand_says_so(self, overlay) -> None:
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert "聽牌" in window._shanten.text()  # noqa: SLF001

    def test_the_ukeire_count_is_there(self, overlay) -> None:
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert "進張" in window._shanten.text()  # noqa: SLF001

    def test_an_unreadable_hand_leaves_it_empty(self, overlay) -> None:
        """有一張 CV 認不出來就算不出向聽。空著就好,不要彈錯誤 ——
        抓在理牌或摸打的動畫中間是常態。"""
        model, window = overlay
        model.update_packet_hand([*TENPAI[:-1], UNKNOWN])
        assert not shown(window._shanten)  # noqa: SLF001

    def test_clearing_the_hand_clears_the_line(self, overlay) -> None:
        """關掉功能時 runtime 會清空手牌 —— 留著上一手的向聽看起來像還在運作。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        model.update_packet_hand(())
        assert not shown(window._shanten)  # noqa: SLF001

    def test_a_conflict_between_the_two_sources_is_flagged(self, overlay) -> None:
        """Overlay 沒有狀態列,所以「兩邊對不上」只能塞在這一行。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        model.update_cv_hand([*TENPAI[:-1], "9p"])
        assert "⚠" in window._shanten.text()  # noqa: SLF001


class TestUkeireTiles:
    """進張要畫成**牌面**,不能只寫枚數。

    Overlay 疊在遊戲上,手牌本來就看得到 —— 再畫一次不增加任何資訊。
    遊戲沒告訴你的是「哪幾張牌能讓你前進」,那才是這塊面積該換來的東西。
    使用者的原話:「overlay 沒有顯示圖示,太過無用」。
    """

    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_a_tenpai_hand_shows_the_waits_as_tiles(self, overlay) -> None:
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert shown(window._ukeire[0])  # noqa: SLF001

    def test_it_works_with_only_vision_running(self, qtbot, tmp_path) -> None:
        """這正是壞掉的那個情境:只開畫面辨識時,建議那半邊整段收起來,
        而進張以前只剩一行字 —— 整個 Overlay 一張圖都沒有。"""
        model = ViewModel()
        board = FakeSwitchboard({features.VISION})
        window = OverlayWindow(
            model, ui_state=UiState.load(tmp_path / "ui.json"), switchboard=board
        )
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        window.set_shown(True)

        model.update_cv_hand(TENPAI)
        assert shown(window._ukeire[0]), "只開畫面辨識時也該看得到進張的牌面"  # noqa: SLF001

    def test_unused_slots_are_hidden(self, overlay) -> None:
        """留著上一巡的殘影會讓人照著一個已經不成立的答案打。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        model.update_packet_hand(())
        assert not any(shown(label) for label in window._ukeire)  # noqa: SLF001

    def test_too_many_waits_are_summarised(self, overlay) -> None:
        """三向聽以上常有 8~13 種進張,全畫出來 HUD 會橫跨半個牌桌。"""
        model, window = overlay
        model.update_packet_hand(["1m", "4m", "7m", "1p", "4p", "7p", "1s", "4s", "7s",
                                  "E", "S", "W", "N"])
        visible = [label for label in window._ukeire if shown(label)]  # noqa: SLF001
        assert len(visible) <= 5
        assert window._ukeire_more.text().startswith("+")  # noqa: SLF001

    def test_a_short_wait_list_has_no_overflow_marker(self, overlay) -> None:
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert not shown(window._ukeire_more)  # noqa: SLF001


class TestExpanding:
    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_it_starts_collapsed(self, overlay) -> None:
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), SKIP_META, 1.0)]
        )
        assert not shown(window._candidates)  # noqa: SLF001

    def test_expanding_shows_the_candidates(self, overlay) -> None:
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), SKIP_META, 1.0)]
        )
        window.set_expanded(True)
        assert shown(window._candidates)  # noqa: SLF001

    def test_an_engine_without_q_values_shows_no_rows(self, overlay) -> None:
        """規則式 baseline 沒有 meta —— 展開了也不該留一塊空的黑條。"""
        model, window = overlay
        window.set_expanded(True)
        model.update_advices(
            [Advice("baseline", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)]
        )
        assert not shown(window._candidates)  # noqa: SLF001

    def test_the_choice_is_remembered(self, overlay) -> None:
        _, window = overlay
        window.set_expanded(True)
        assert UiState.load(window._ui_state.path).overlay_expanded is True  # noqa: SLF001


class TestLocking:
    def test_it_starts_unlocked(self, overlay) -> None:
        """第一次開起來一定要拖得動 —— 鎖住的視窗擺不了位置。"""
        _, window = overlay
        assert window.locked is False

    def test_the_drag_hint_is_only_there_while_unlocked(self, overlay) -> None:
        _, window = overlay
        assert shown(window._hint)  # noqa: SLF001
        window.set_locked(True)
        assert not shown(window._hint)  # noqa: SLF001

    def test_locking_makes_it_transparent_for_input(self, overlay) -> None:
        """這才是鎖定真正的用途:點擊要能穿過去給遊戲。"""
        _, window = overlay
        window.set_locked(True)
        assert window.windowFlags() & Qt.WindowType.WindowTransparentForInput

    def test_unlocking_takes_it_back(self, overlay) -> None:
        _, window = overlay
        window.set_locked(True)
        window.set_locked(False)
        assert not (window.windowFlags() & Qt.WindowType.WindowTransparentForInput)

    def test_a_locked_overlay_ignores_drags(self, overlay) -> None:
        """鎖定之後即使真的收到滑鼠事件也不能動 —— 不然使用者會拖到一個
        自己以為固定住的東西。
        """
        _, window = overlay
        window.set_locked(True)
        before = window.pos()
        window.mousePressEvent(_press(QPoint(500, 500)))
        window.mouseMoveEvent(_press(QPoint(600, 600)))
        assert window.pos() == before

    def test_the_choice_is_remembered(self, overlay) -> None:
        _, window = overlay
        window.set_locked(True)
        assert UiState.load(window._ui_state.path).overlay_locked is True  # noqa: SLF001


class TestPosition:
    def test_a_drag_moves_it(self, overlay) -> None:
        _, window = overlay
        window.mousePressEvent(_press(window.pos() + QPoint(10, 10)))
        window.mouseMoveEvent(_press(window.pos() + QPoint(110, 60)))
        window.mouseReleaseEvent(_press(window.pos()))
        assert window._ui_state.overlay_pos is not None  # noqa: SLF001

    def test_the_position_is_saved_on_release_not_during(self, overlay) -> None:
        """拖的過程中每個 pixel 都寫一次檔案是沒必要的磁碟流量。"""
        _, window = overlay
        window.mousePressEvent(_press(window.pos() + QPoint(10, 10)))
        window.mouseMoveEvent(_press(window.pos() + QPoint(110, 60)))
        assert UiState.load(window._ui_state.path).overlay_pos is None  # noqa: SLF001
        window.mouseReleaseEvent(_press(window.pos()))
        assert UiState.load(window._ui_state.path).overlay_pos is not None  # noqa: SLF001

    def test_a_position_off_every_screen_falls_back_to_the_default(self, qtbot, tmp_path) -> None:
        """外接螢幕拔掉之後,存下來的座標會落在看不見的地方 —— 那時 Overlay
        開了等於沒開,而使用者只會覺得勾了沒反應。
        """
        state = UiState.load(tmp_path / "ui.json")
        state.overlay_pos = (-9000, -9000)
        window = OverlayWindow(ViewModel(), ui_state=state)
        qtbot.addWidget(window)
        assert window.pos() != QPoint(-9000, -9000)


class TestFeatureSwitches:
    """開關是側邊視窗上的,但 Overlay 也要跟著說實話。"""

    @staticmethod
    def build(qtbot, tmp_path, on: set[str]):
        model = ViewModel()
        window = OverlayWindow(
            model, switchboard=FakeSwitchboard(on), ui_state=UiState.load(tmp_path / "ui.json")
        )
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        window.set_shown(True)
        return model, window

    def test_both_off_says_so_instead_of_showing_an_empty_box(self, qtbot, tmp_path) -> None:
        _, window = self.build(qtbot, tmp_path, set())
        assert window._verb.text() == "未開啟"  # noqa: SLF001
        assert "開關" in window._mark.text()  # noqa: SLF001

    def test_advice_off_hides_the_advice_line_entirely(self, qtbot, tmp_path) -> None:
        """只跑畫面辨識時,「未開啟」佔著半個 HUD 是浪費 —— 把位置讓給向聽。"""
        model, window = self.build(qtbot, tmp_path, {features.VISION})
        model.update_cv_hand(TENPAI)
        assert not shown(window._verb)  # noqa: SLF001
        assert shown(window._shanten)  # noqa: SLF001

    def test_advice_off_does_not_show_the_engine_output(self, qtbot, tmp_path) -> None:
        """關掉之後留著最後一手看起來像還在運作,而那比空白糟得多。"""
        model, window = self.build(qtbot, tmp_path, {features.VISION})
        model.update_advices([Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)])
        assert not shown(window._subject)  # noqa: SLF001

    def test_advice_on_but_no_engine_yet_says_waiting(self, qtbot, tmp_path) -> None:
        _, window = self.build(qtbot, tmp_path, {features.ADVICE})
        assert window._verb.text() == "等待引擎…"  # noqa: SLF001


def _press(point: QPoint):
    """造一個左鍵事件。座標一律用全域的 —— 拖曳算的是螢幕座標。"""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        QPointF(point),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
