"""放銃分析分頁。

只驗接線與**話有沒有說對**。後者在這一頁特別重要:這是唯一一個「畫面上寫
安全、使用者就會照著打」的分頁,而實測踩過的坑正是把逐家的資訊壓成一個標籤。
"""

from __future__ import annotations

import pytest

from mia.analysis import DangerLevel, assess
from mia.mjai import Dahai, Pon, Reach, StartGame, StartKyoku
from mia.mjai.table import TableTracker
from mia.ui.viewmodel import ViewModel

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from mia.ui.widgets.danger import AnalysisDangerTab, seat_name
from mia.ui.widgets.tiles import TileIcons


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


@pytest.fixture
def tab(qtbot) -> AnalysisDangerTab:
    widget = AnalysisDangerTab(TileIcons())
    qtbot.addWidget(widget)
    return widget


def shown(widget) -> bool:
    """用 isHidden 而不是 isVisible:沒 show() 過的視窗底下,後者一律是 False,
    測不出我們自己的 setVisible 有沒有下對。與 test_ui_panel 同一個理由。"""
    return not widget.isHidden()


def _apply(tab: AnalysisDangerTab, report, *, enabled: bool = True) -> None:
    model = ViewModel()
    model.update_dangers(report)
    tab.update_from(model.state, enabled=enabled)


class TestSeatNames:
    """「對 3 家危險」要在腦裡換算一次;「對下家危險」直接就能用。"""

    def test_relative_names(self) -> None:
        assert seat_name(0, 1) == "下家"
        assert seat_name(0, 2) == "對家"
        assert seat_name(0, 3) == "上家"

    def test_it_wraps_around(self) -> None:
        assert seat_name(3, 0) == "下家"

    def test_an_unknown_seat_falls_back_to_the_number(self) -> None:
        """還沒看到 start_game 時不知道自己坐哪。寫錯的相對稱呼比數字糟。"""
        assert seat_name(None, 2) == "2 家"


class TestEmptyStates:
    def test_the_feature_being_off_says_so(self, tab) -> None:
        _apply(tab, None, enabled=False)
        assert tab._headline.text() == "未開啟"  # noqa: SLF001

    def test_no_data_is_not_the_same_as_safe(self, tab) -> None:
        """功能開著但還沒有資料。這時候**不能**畫成一片安全。"""
        _apply(tab, None)
        assert "等待" in tab._headline.text()  # noqa: SLF001
        assert not any(shown(row) for row in tab._rows)  # noqa: SLF001

    def test_it_says_it_needs_the_packets(self, tab) -> None:
        """這一頁靠封包,CV 一張別家的牌都看不到 —— 使用者要知道為什麼空著。"""
        _apply(tab, None)
        assert "封包" in tab._source.text()  # noqa: SLF001


class TestWhatItSays:
    def test_a_reach_is_named_relatively(self, tab) -> None:
        table = _table(seat=0)
        table.handle(Reach(actor=1))
        table.handle(Dahai(actor=1, pai="1z", tsumogiri=False))
        _apply(tab, assess(["3m"], table))
        assert "下家" in tab._headline.text()  # noqa: SLF001

    def test_nobody_named_says_that_is_not_safety(self, tab) -> None:
        """**這一條釘的是真實牌譜教的事**:那場三次榮和,兩次是沒立直的人
        和的。畫面上只寫「沒有人立直」會被讀成「現在很安全」。
        """
        _apply(tab, assess(["3m"], _table()))
        assert "沒有人" in tab._headline.text()  # noqa: SLF001
        assert "不代表安全" in tab._source.text()  # noqa: SLF001

    def test_a_melded_seat_is_named_in_the_headline(self, tab) -> None:
        """有人坐在三副露上,標題卻寫「沒有人立直」—— 那是畫面說沒事而
        實際有事,與上次那個坑同一類。"""
        table = _table(seat=0)
        for pai in ("1z", "2z", "3z"):
            table.handle(Pon(actor=1, target=0, pai=pai, consumed=[pai, pai]))
        _apply(tab, assess(["3m"], table))
        assert tab._headline.text() == "下家3副露"  # noqa: SLF001

    def test_a_row_says_the_seat_is_melded(self, tab) -> None:
        """「對下家危險」與「對下家(3副露)危險」是不同份量的話。"""
        table = _table(seat=0)
        for pai in ("1z", "2z", "3z"):
            table.handle(Pon(actor=1, target=0, pai=pai, consumed=[pai, pai]))
        _apply(tab, assess(["3m"], table))
        assert "3副露" in tab._rows[0]._detail.text()  # noqa: SLF001

    def test_honour_tiles_do_not_blow_up(self, tab) -> None:
        """牌名已經是 MJAI 記法了。再轉一次的話數牌剛好過得去、字牌會拋
        TileError —— 只有摸到字牌那一刻才會炸,而那是實測踩到的。"""
        _apply(tab, assess(["S", "1z", "5m"], _table(seat=0)))
        assert sum(shown(row) for row in tab._rows) == 3  # noqa: SLF001

    def test_a_row_says_who_and_why(self, tab) -> None:
        table = _table(seat=0)
        table.handle(Reach(actor=2))
        table.handle(Dahai(actor=2, pai="1z", tsumogiri=False))
        _apply(tab, assess(["3m"], table))
        detail = tab._rows[0]._detail.text()  # noqa: SLF001
        assert "對" in detail
        assert any(word in detail for word in ("両面", "嵌張", "単騎", "雙碰", "現物"))

    def test_a_reached_seat_is_marked_as_such(self, tab) -> None:
        table = _table(seat=0)
        # 另外兩家先打 3m —— 對他們是現物。順序很重要:立直**之後**別人打的
        # 牌對立直者也會變安全,那樣三家就都是現物了。
        table.handle(Dahai(actor=2, pai="3m", tsumogiri=False))
        table.handle(Dahai(actor=3, pai="3m", tsumogiri=False))
        table.handle(Reach(actor=1))
        table.handle(Dahai(actor=1, pai="1z", tsumogiri=False))
        _apply(tab, assess(["3m"], table))
        assert "立直" in tab._rows[0]._detail.text()  # noqa: SLF001

    def test_all_furiten_collapses_into_one_line(self, tab) -> None:
        """三家都是現物是最好的情況,不必把三行一樣的東西攤開。"""
        table = _table(seat=0)
        for seat in (1, 2, 3):
            table.handle(Dahai(actor=seat, pai="3m", tsumogiri=False))
        report = assess(["3m"], table)
        assert report.tiles[0].level is DangerLevel.SAFE
        _apply(tab, report)
        assert tab._rows[0]._detail.text() == "三家都是現物"  # noqa: SLF001


class TestRowPool:
    def test_a_shorter_hand_hides_the_leftover_rows(self, tab) -> None:
        """副露之後手牌變短,不該留著上一手的殘影。"""
        table = _table(seat=0)
        _apply(tab, assess(["1m", "2m", "3m", "4m", "5m"], table))
        _apply(tab, assess(["1m", "2m"], table))
        assert sum(shown(row) for row in tab._rows) == 2  # noqa: SLF001

    def test_switching_the_feature_off_clears_the_rows(self, tab) -> None:
        table = _table(seat=0)
        _apply(tab, assess(["1m", "2m", "3m"], table))
        _apply(tab, None, enabled=False)
        assert not any(shown(row) for row in tab._rows)  # noqa: SLF001
