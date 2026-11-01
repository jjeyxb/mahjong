"""Log 視窗的資料與畫面。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mia.mjai.events import Dahai, MjaiEvent, StartGame, StartKyoku, Tsumo
from mia.ui.eventlog import PACKET_SOURCE, EventLog
from mia.ui.overlay.log import EventLogWindow
from mia.ui.state import UiState

SCREEN = "圖像辨識"


def _kyoku(kyoku: int = 1) -> StartKyoku:
    return StartKyoku(
        bakaze="E",
        kyoku=kyoku,
        honba=0,
        kyotaku=0,
        oya=0,
        dora_marker="1m",
        tehais=[["1p"], ["?"], ["?"], ["?"]],
        scores=[25000] * 4,
    )


def _round(kyoku: int = 1, *rest: MjaiEvent) -> tuple[MjaiEvent, ...]:
    return (_kyoku(kyoku), *rest)


class Counter:
    def __init__(self, log: EventLog) -> None:
        self.calls = 0
        log.subscribe(self)

    def __call__(self) -> None:
        self.calls += 1


class TestEventLog:
    def test_each_event_has_both_forms(self) -> None:
        log = EventLog()
        log.update(PACKET_SOURCE, 0, _round(1, Tsumo(actor=0, pai="3p")))
        line = log.lines[-1]
        assert line.text == "自家 摸 三筒"
        assert json.loads(line.mjai) == {"type": "tsumo", "actor": 0, "pai": "3p"}

    def test_a_round_is_a_heading(self) -> None:
        log = EventLog()
        log.update(SCREEN, 0, _round(1, Tsumo(actor=1, pai="?")))
        assert [line.heading for line in log.lines] == [True, False]

    def test_a_rewritten_stream_replaces_what_was_shown(self) -> None:
        """畫面那條路的牌會被改:河裡猜的牌,牌面認好之後換掉。"""
        log = EventLog()
        wrong = Dahai(actor=1, pai="2p", tsumogiri=False)
        right = Dahai(actor=1, pai="5p", tsumogiri=False)
        log.update(SCREEN, 0, _round(1, wrong))
        log.update(SCREEN, 0, _round(1, right))
        assert len(log.lines) == 2
        assert log.lines[-1].text == "下家 打 五筒"

    def test_a_shorter_stream_drops_the_tail(self) -> None:
        """暫時接在尾巴的牌,排出來不一樣就收回去。"""
        log = EventLog()
        log.update(SCREEN, 0, _round(1, Tsumo(actor=1, pai="?")))
        log.update(SCREEN, 0, _round(1))
        assert len(log.lines) == 1

    def test_a_new_round_keeps_the_old_one(self) -> None:
        log = EventLog()
        log.update(SCREEN, 0, _round(1, Tsumo(actor=1, pai="?")))
        log.update(SCREEN, 0, _round(2))
        assert [line.heading for line in log.lines] == [True, False, True]

    def test_a_changed_hand_is_still_the_same_round(self) -> None:
        """配牌讀到更準的版本不是換局,留著的話同一局會列兩次。"""
        log = EventLog()
        log.update(SCREEN, 0, (_kyoku(),))
        fixed = replace(_kyoku(), tehais=[["2p"], ["?"], ["?"], ["?"]])
        log.update(SCREEN, 0, (fixed,))
        assert len(log.lines) == 1

    def test_the_start_of_a_game_is_kept_when_the_round_opens(self) -> None:
        log = EventLog()
        log.update(PACKET_SOURCE, 1, (StartGame(id=1),))
        log.update(PACKET_SOURCE, 1, _round())
        assert [line.text[:4] for line in log.lines] == ["對局開始", "東一局 "]

    def test_old_rounds_are_capped(self) -> None:
        log = EventLog()
        body = tuple(Tsumo(actor=1, pai="?") for _ in range(300))
        for kyoku in range(1, 5):
            for honba in range(3):
                head = replace(_kyoku(kyoku), honba=honba)
                log.update(SCREEN, 0, (head, *body))
        assert len(log.lines) <= 2301

    def test_only_a_real_change_notifies(self) -> None:
        log = EventLog()
        seen = Counter(log)
        log.update(SCREEN, 0, _round())
        log.update(SCREEN, 0, _round())
        assert seen.calls == 1
        log.update(SCREEN, 0, _round(), note="等手牌")
        assert seen.calls == 2

    def test_clear_empties_it(self) -> None:
        log = EventLog()
        log.update(SCREEN, 0, _round(1), note="等手牌")
        log.clear()
        assert log.lines == ()
        assert (log.source, log.note) == ("", "")


@pytest.fixture
def state(tmp_path) -> UiState:
    return UiState.load(tmp_path / "ui.json")


@pytest.fixture
def window(qtbot, state):
    log = EventLog()
    widget = EventLogWindow(log, ui_state=state)
    qtbot.addWidget(widget)
    return widget, log


class TestWindow:
    def test_it_starts_hidden(self, window) -> None:
        widget, _ = window
        assert not widget.shown
        assert widget.isHidden()

    def test_it_says_why_it_is_empty(self, window) -> None:
        widget, _ = window
        widget.set_shown(True)
        assert "AI 建議" in widget._body.toPlainText()  # noqa: SLF001

    def test_it_shows_both_forms(self, window) -> None:
        widget, log = window
        widget.set_shown(True)
        log.update(PACKET_SOURCE, 0, _round(1, Tsumo(actor=0, pai="3p")))
        text = widget._body.toPlainText()  # noqa: SLF001
        assert "自家 摸 三筒" in text
        assert '{"type":"tsumo","actor":0,"pai":"3p"}' in text
        assert widget._source.text() == "封包 · 2 則"  # noqa: SLF001

    def test_what_happened_while_hidden_shows_up(self, window) -> None:
        widget, log = window
        log.update(SCREEN, 0, _round())
        widget.set_shown(True)
        assert "東一局" in widget._body.toPlainText()  # noqa: SLF001

    def test_a_note_is_shown_only_when_there_is_one(self, window) -> None:
        widget, log = window
        widget.set_shown(True)
        log.update(SCREEN, 0, _round(), note="余牌對不上")
        assert widget._note.isVisibleTo(widget)  # noqa: SLF001
        assert widget._note.text() == "余牌對不上"  # noqa: SLF001
        log.update(SCREEN, 0, _round())
        assert not widget._note.isVisibleTo(widget)  # noqa: SLF001

    def test_html_in_a_name_is_not_markup(self, window) -> None:
        widget, log = window
        widget.set_shown(True)
        log.update(SCREEN, 0, _round(1, Dahai(actor=1, pai="<b>", tsumogiri=False)))
        assert "<b>" in widget._body.toPlainText()  # noqa: SLF001

    def test_shown_and_locked_are_remembered(self, window, state) -> None:
        widget, _ = window
        widget.set_shown(True)
        widget.set_locked(True)
        back = UiState.load(state.path)
        assert (back.log_visible, back.log_locked) == (True, True)

    def test_it_does_not_touch_the_overlay_settings(self, window, state) -> None:
        widget, _ = window
        widget.set_shown(True)
        widget.set_locked(True)
        widget._remember_position((5, 6))  # noqa: SLF001
        back = UiState.load(state.path)
        assert (back.overlay_visible, back.overlay_locked, back.overlay_pos) == (False, False, None)

    def test_a_locked_log_hides_the_handles(self, window) -> None:
        widget, _ = window
        widget.set_shown(True)
        widget.set_locked(True)
        assert not widget._grip.isVisibleTo(widget)  # noqa: SLF001
        widget.set_locked(False)
        assert widget._grip.isVisibleTo(widget)  # noqa: SLF001

    def test_the_size_comes_back(self, qtbot, tmp_path) -> None:
        state = UiState.load(tmp_path / "ui.json")
        state.log_size = (600, 400)
        widget = EventLogWindow(EventLog(), ui_state=state)
        qtbot.addWidget(widget)
        assert (widget.width(), widget.height()) == (600, 400)


class TestScrolling:
    @staticmethod
    def _fill(log: EventLog, count: int = 120) -> None:
        log.update(SCREEN, 0, _round(1, *(Tsumo(actor=1, pai="?") for _ in range(count))))

    def test_it_follows_the_newest(self, window) -> None:
        widget, log = window
        widget.set_shown(True)
        self._fill(log)
        self._fill(log, 160)
        bar = widget._body.verticalScrollBar()  # noqa: SLF001
        assert bar.value() == bar.maximum() > 0

    def test_scrolling_up_stays_put(self, window) -> None:
        widget, log = window
        widget.set_shown(True)
        self._fill(log)
        bar = widget._body.verticalScrollBar()  # noqa: SLF001
        bar.setValue(30)
        self._fill(log, 160)
        assert bar.value() == 30

    def test_a_locked_log_always_follows(self, window) -> None:
        """鎖定之後捲不動,停在舊的地方就再也看不到新的事件。"""
        widget, log = window
        widget.set_shown(True)
        self._fill(log)
        widget._body.verticalScrollBar().setValue(30)  # noqa: SLF001
        widget.set_locked(True)
        self._fill(log, 160)
        bar = widget._body.verticalScrollBar()  # noqa: SLF001
        assert bar.value() == bar.maximum()
