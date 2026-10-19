"""Overlay 視窗。

驗的是**接線與狀態**,不是像素:哪些東西該顯示、哪些該藏起來、鎖定有沒有真的
讓滑鼠穿過去、拖完的位置有沒有存回去。長什麼樣要人眼看,用
``python tools/ui.py --demo`` 然後到設定頁勾起來。

跟側邊視窗一樣在無頭環境跑(conftest 設了 ``QT_QPA_PLATFORM=offscreen``),
所以一律用 ``isHidden()`` 而不是 ``isVisible()`` —— 後者對沒 show 過的視窗
底下的子元件一律回 False,測不出我們自己下的 setVisible。
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from mia import features
from mia.analysis import assess
from mia.engine.base import Advice
from mia.mjai import Chi, Dahai, Pon, Reach, StartGame, StartKyoku
from mia.mjai.table import TableTracker
from mia.mjai.tiles import UNKNOWN
from mia.ui.state import UiState
from mia.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from PySide6.QtCore import QPoint, Qt

from mia.ui.overlay.window import OverlayWindow

TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]

#: 危險度那一段用的手牌。一整手 13 張,MJAI 記法(封包那條路的原生記法)。
#: 刻意混著中張、么九與字牌 —— 三者能被打中的型數差很多,排序才有東西可排。
REACH_HAND = [
    "2m", "3m", "4m", "5m", "6m", "7p", "8p", "3s", "4s", "5s", "9s", "E", "S",
]


def _table(seat: int = 0) -> TableTracker:
    t = TableTracker()
    t.handle(StartGame(id=seat))
    t.handle(
        StartKyoku(
            bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0, dora_marker="1z",
            tehais=[["?"] * 13 for _ in range(4)], scores=[25000] * 4,
        )
    )
    return t


def _report(hand):
    """上家立直之後的一份報告。"""
    table = _table(seat=0)
    table.handle(Reach(actor=3))
    return assess(hand, table)

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

    def test_the_ukeire_count_is_gone(self, overlay) -> None:
        """使用者要求拿掉進張。那塊面積在 HUD 上太貴,而真正在決定下一手的是
        左半邊那個建議。完整的枚數表在側邊視窗的向聽分析頁。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert "進張" not in window._shanten.text()  # noqa: SLF001

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


class TestWhatTheShantenLineStillSays:
    """進張拿掉了,但這一行上有兩個判斷要留著。

    兩個都是當初為了進張才寫的,拿掉進張之後很容易順手刪掉 —— 而它們驗的
    是**向聽數有沒有前提**,與進張無關。
    """

    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_it_says_which_tile_the_shanten_assumes_discarding(self, overlay) -> None:
        """14 張的手牌 ``analysis.ukeire`` 依定義是空的 —— 向聽數講的是
        「切掉某張之後」。不寫是哪張的話那個數字沒有前提。"""
        model, window = overlay
        model.update_packet_hand([*TENPAI, "9s"], drawn="9s")
        assert "切" in window._shanten.text()  # noqa: SLF001

    def test_a_winning_hand_is_never_told_to_discard(self, overlay) -> None:
        """和了的手牌 ukeire 也是空的,但那時候該做的事是**和牌**。

        少了這個判斷會顯示「和了(切1m)」。進張不畫了,這個判斷還是得留著
        —— 它靠的是 ``_effective_ukeire``,而那個函式現在只剩這一個用途。
        """
        model, window = overlay
        model.update_packet_hand([*TENPAI, "4p"], drawn="4p")  # 這一手其實已經和了
        assert "和了" in window._shanten.text()  # noqa: SLF001
        assert "切" not in window._shanten.text()  # noqa: SLF001

    def test_a_thirteen_tile_hand_has_no_discard_qualifier(self, overlay) -> None:
        """還沒摸牌時向聽就是向聽,沒有「切某張之後」這個前提。"""
        model, window = overlay
        model.update_packet_hand(TENPAI)
        assert "切" not in window._shanten.text()  # noqa: SLF001


class TestShantenNeedsVision:
    """向聽**只在畫面辨識開著時**才顯示。

    ``analysis`` 是從手牌算的,而手牌是封包優先 —— 所以只開 AI 建議時它照樣
    算得出來,於是 HUD 上會冒出一個使用者沒有打開任何東西去要的數字。
    那一頁的開關關著,它的輸出就不該出現在另一個畫面上。
    """

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

    def test_advice_only_does_not_show_shanten(self, qtbot, tmp_path) -> None:
        model, window = self.build(qtbot, tmp_path, {features.ADVICE})
        model.update_packet_hand(TENPAI)
        assert not shown(window._shanten)  # noqa: SLF001

    def test_vision_on_shows_it(self, qtbot, tmp_path) -> None:
        """反面 —— 不然「永遠不顯示」也會過。"""
        model, window = self.build(qtbot, tmp_path, {features.VISION})
        model.update_cv_hand(TENPAI)
        assert shown(window._shanten)  # noqa: SLF001

    def test_a_packet_hand_still_counts_when_vision_is_on(self, qtbot, tmp_path) -> None:
        """封包的手牌比 CV 精確,開著畫面辨識時當然要用它算 —— 閘門管的是
        「使用者要不要看」,不是「資料從哪來」。"""
        model, window = self.build(qtbot, tmp_path, {features.VISION})
        model.update_packet_hand(TENPAI)
        assert shown(window._shanten)  # noqa: SLF001


class TestTopTiles:
    """主要引擎推薦的前三張,畫在右半邊(原本是進張的位置)。

    比進張值錢:進張講的是「摸到什麼會前進」,而手牌本來就看得到;
    這個講的是「模型第二、第三喜歡什麼」,是遊戲裡沒有的資訊。
    """

    #: 三個切牌候選 + 立直,Q 值刻意不照位元順序 —— 要驗它是按分數排,
    #: 不是按動作索引排。位元:1m(0)、2m(1)、3m(2)、立直(37)。
    META: ClassVar[dict] = {"mask_bits": 0b111 | (1 << 37), "q_values": [-2.0, 5.0, 1.0, 3.0]}

    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    def test_it_shows_three(self, overlay) -> None:
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="2m", tsumogiri=False), self.META, 1.0)]
        )
        assert sum(shown(t) for t in window._top) == 3  # noqa: SLF001

    def test_they_are_in_score_order(self, overlay) -> None:
        """2m(5.0)> 3m(1.0)> 1m(-2.0)。立直排第二但不是牌,跳過。"""
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="2m", tsumogiri=False), self.META, 1.0)]
        )
        assert [t._tile for t in window._top[:3]] == ["2m", "3m", "1m"]  # noqa: SLF001

    def test_non_tile_actions_are_skipped(self, overlay) -> None:
        """候選裡也會有立直 / 吃 / 碰,但那些沒有對應的單張牌 ——
        混進一排牌面裡會變成「第二名是空白」。"""
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="2m", tsumogiri=False), self.META, 1.0)]
        )
        assert "reach" not in [t.tile for t in window._top]  # noqa: SLF001

    def test_an_engine_without_q_values_shows_nothing(self, overlay) -> None:
        """規則式基準沒有 meta —— 不該留一排上一手的殘影。"""
        model, window = overlay
        model.update_advices(
            [Advice("baseline", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)]
        )
        assert not any(shown(t) for t in window._top)  # noqa: SLF001
        assert not shown(window._top_label)  # noqa: SLF001

    def test_advice_off_hides_them(self, qtbot, tmp_path) -> None:
        model = ViewModel()
        window = OverlayWindow(
            model,
            switchboard=FakeSwitchboard({features.VISION}),
            ui_state=UiState.load(tmp_path / "ui.json"),
        )
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        window.set_shown(True)
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="2m", tsumogiri=False), self.META, 1.0)]
        )
        assert not any(shown(t) for t in window._top)  # noqa: SLF001


class TestOtherEngines:
    """比較對象那幾行。

    挑了比較對象卻只看得到主要引擎,那個選單等於沒作用 —— 而兩份權重真正的
    差別只在**分歧的那幾手**顯現。
    """

    @pytest.fixture(autouse=True)
    def _shown(self, overlay):
        overlay[1].set_shown(True)

    @staticmethod
    def _two(primary: str, other: str):
        return [
            Advice("call_high", Dahai(actor=0, pai=primary, tsumogiri=False), None, 1.0),
            Advice("call_low", Dahai(actor=0, pai=other, tsumogiri=False), None, 1.0),
        ]

    def rows(self, window):
        return [r for r, _n, _v in window._other_rows if shown(r)]  # noqa: SLF001

    def test_the_second_engine_is_shown(self, overlay) -> None:
        model, window = overlay
        model.update_advices(self._two("3s", "7p"))
        assert len(self.rows(window)) == 1
        assert window._other_rows[0][1].text() == "call_low"  # noqa: SLF001

    def test_it_says_which_tile_the_other_would_discard(self, overlay) -> None:
        model, window = overlay
        model.update_advices(self._two("3s", "7p"))
        assert "7p" in window._other_rows[0][2].text()  # noqa: SLF001

    def test_disagreement_is_marked(self, overlay) -> None:
        """九成以上的手兩份權重會給同一個答案,看了也沒用。有價值的是剩下那幾手。"""
        model, window = overlay
        model.update_advices(self._two("3s", "7p"))
        assert "分歧" in window._other_rows[0][2].text()  # noqa: SLF001

    def test_agreement_is_not_marked(self, overlay) -> None:
        model, window = overlay
        model.update_advices(self._two("3s", "3s"))
        assert "分歧" not in window._other_rows[0][2].text()  # noqa: SLF001

    def test_a_declining_engine_is_still_a_disagreement(self, overlay) -> None:
        """「它鳴、我不鳴」正是副露率那條風格軸的全部內容。不標的話畫面上
        只有鳴的那一邊被標成分歧,看起來像它自己跟自己不一致。"""
        model, window = overlay
        model.update_advices([
            Advice("call_high", Pon(actor=0, target=3, pai="5m", consumed=["5m", "5m"]),
                   None, 1.0),
            # declined 是從 meta 推出來的:action 是 None 而 meta 是一個決策點。
            # SKIP_META 正好就是那種 meta(碰 / 槓 / 跳過三個位元)。
            Advice("call_low", None, SKIP_META, 1.0),
        ])
        assert "分歧" in window._other_rows[0][2].text()  # noqa: SLF001

    def test_a_single_engine_leaves_the_section_empty(self, overlay) -> None:
        """沒挑比較對象時不該留一條空的黑線佔著牌桌。"""
        model, window = overlay
        model.update_advices(
            [Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0)]
        )
        assert not shown(window._others)  # noqa: SLF001

    def test_advice_off_hides_it(self, qtbot, tmp_path) -> None:
        """關掉之後留著最後一手看起來像還在運作。"""
        model = ViewModel()
        window = OverlayWindow(
            model,
            switchboard=FakeSwitchboard({features.VISION}),
            ui_state=UiState.load(tmp_path / "ui.json"),
        )
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        window.set_shown(True)
        model.update_advices(self._two("3s", "7p"))
        assert not shown(window._others)  # noqa: SLF001


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


class TestDangerSection:
    """危險度那一段。

    使用者要的是**每一張都列出來、安全的在上** —— 挑要切哪張的時候,眼睛
    從上往下掃到第一張不影響進張的牌就停。所以這裡驗的是「有沒有全列」與
    「順序對不對」,還有那段話有沒有把「沒人立直 ≠ 安全」講清楚。
    """

    @staticmethod
    def build(qtbot, tmp_path, *, on=None, danger=True):
        model = ViewModel()
        window = OverlayWindow(
            model,
            switchboard=FakeSwitchboard({features.DANGER} if on is None else on),
            ui_state=UiState.load(tmp_path / "ui.json"),
        )
        model.subscribe(window.apply)
        qtbot.addWidget(window)
        window.set_shown(True)
        window.set_danger_shown(danger)
        return model, window

    def rows(self, window):
        return [r for r in window._danger_rows if shown(r)]  # noqa: SLF001

    def test_it_is_off_until_asked_for(self, qtbot, tmp_path) -> None:
        """預設不佔那段高度 —— 一整手 14 列是很大一塊牌桌。"""
        _, window = self.build(qtbot, tmp_path, danger=False)
        assert not window.danger_shown
        assert not shown(window._dangers)  # noqa: SLF001

    def test_every_tile_is_listed(self, qtbot, tmp_path) -> None:
        """不是「最安全的前幾張」。被截掉的正好是最危險的那幾張。"""
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(_report(REACH_HAND))
        assert len(self.rows(window)) == len(set(REACH_HAND))

    def test_the_safest_are_on_top(self, qtbot, tmp_path) -> None:
        model, window = self.build(qtbot, tmp_path)
        report = _report(REACH_HAND)
        model.update_dangers(report)
        levels = [d.level for d in report.tiles]
        assert levels == sorted(levels)
        assert self.rows(window)[0]._level.text() == report.tiles[0].level.label  # noqa: SLF001

    def test_a_row_says_who_it_is_dangerous_against(self, qtbot, tmp_path) -> None:
        """只寫「安全」是那個坑本身:一張牌對立直的家是現物、對別家全新。"""
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(_report(REACH_HAND))
        assert any("家" in r._note.text() for r in self.rows(window))  # noqa: SLF001

    def test_nobody_named_still_warns(self, qtbot, tmp_path) -> None:
        """實測那場三次榮和,兩次是沒立直的人和的。"""
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(assess(REACH_HAND, _table()))
        assert "不代表安全" in window._danger_head.text()  # noqa: SLF001

    def test_a_reach_is_named_in_the_headline(self, qtbot, tmp_path) -> None:
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(_report(REACH_HAND))
        assert "上家立直" in window._danger_head.text()  # noqa: SLF001

    def test_a_melded_seat_is_named_too(self, qtbot, tmp_path) -> None:
        """三副露與立直在標題上同一階 —— 只說「沒人立直」會讓人以為沒事。"""
        model, window = self.build(qtbot, tmp_path)
        table = _table(seat=0)
        for pai in ("1z", "2z", "3z"):
            table.handle(Pon(actor=1, target=0, pai=pai, consumed=[pai, pai]))
        model.update_dangers(assess(REACH_HAND, table))
        assert "下家3副露" in window._danger_head.text()  # noqa: SLF001

    def test_no_data_collapses_the_section(self, qtbot, tmp_path) -> None:
        """HUD 的高度是從牌桌上拿走的 —— 沒內容就該還回去,不是留一句「等待中」。"""
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(_report(REACH_HAND))
        model.update_dangers(None)
        assert not shown(window._dangers)  # noqa: SLF001

    def test_the_feature_being_off_hides_it_even_when_checked(self, qtbot, tmp_path) -> None:
        """勾了但功能沒開:沒有資料可畫,不該留上一局的殘影。"""
        model, window = self.build(qtbot, tmp_path, on=set())
        model.update_dangers(_report(REACH_HAND))
        assert not shown(window._dangers)  # noqa: SLF001

    def test_it_does_not_shout_not_enabled_over_the_list(self, qtbot, tmp_path) -> None:
        """只開放銃分析時,上面那行「未開啟」會與底下的清單互相矛盾。"""
        model, window = self.build(qtbot, tmp_path, on={features.DANGER})
        model.update_dangers(_report(REACH_HAND))
        assert not shown(window._verb)  # noqa: SLF001
        assert shown(window._dangers)  # noqa: SLF001

    def test_running_but_not_on_the_hud_does_not_say_not_enabled(
        self, qtbot, tmp_path
    ) -> None:
        """**實機第一次開起來踩到的。** 放銃分析開著、只是還沒勾上 HUD,
        Overlay 卻寫「未開啟 —— 用側邊視窗的開關打開」。使用者會去撥一個
        已經開著的開關,然後以為程式壞了。
        """
        _, window = self.build(qtbot, tmp_path, on={features.DANGER}, danger=False)
        assert window._verb.text() != "未開啟"  # noqa: SLF001
        assert "設定頁" in window._mark.text()  # noqa: SLF001

    def test_everything_off_still_says_not_enabled(self, qtbot, tmp_path) -> None:
        """反面:真的全關著時那句話還要在,不然是一個空黑框蓋在牌桌上。"""
        _, window = self.build(qtbot, tmp_path, on=set(), danger=False)
        assert window._verb.text() == "未開啟"  # noqa: SLF001

    def test_honour_tiles_do_not_blow_up(self, qtbot, tmp_path) -> None:
        """牌名已經是 MJAI 記法。再轉一次只有字牌會炸 —— 數牌剛好轉得過去。"""
        model, window = self.build(qtbot, tmp_path)
        model.update_dangers(_report(["S", "1z", "5m"]))
        assert len(self.rows(window)) == 3

    def test_the_choice_is_remembered(self, qtbot, tmp_path) -> None:
        self.build(qtbot, tmp_path, danger=True)
        assert UiState.load(tmp_path / "ui.json").overlay_danger


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
