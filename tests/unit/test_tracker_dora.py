"""寶牌指示牌:新翻開的 → ``Dora`` 事件。"""

from __future__ import annotations

from mia.tracker import DoraWatcher, NewDora
from mia.vision.tiles.classify import Match


def _sure(label: str) -> Match:
    """一個過得了門檻的讀數。實測寶牌的分數 0.929、差距 0.391。"""
    return Match(label=label, score=0.93, runner_up="5z", runner_up_score=0.54)


def _unsure(label: str) -> Match:
    return Match(label=label, score=0.50, runner_up="5z", runner_up_score=0.49)


class TestItReportsNewIndicators:
    def test_the_first_one(self) -> None:
        watcher = DoraWatcher()
        assert watcher.observe([_sure("3s"), None, None, None, None]) == [
            NewDora(index=0, indicator="3s")
        ]

    def test_seeing_the_same_panel_again_is_not_a_new_dora(self) -> None:
        watcher = DoraWatcher()
        slots = [_sure("3s"), None, None, None, None]
        watcher.observe(slots)
        assert watcher.observe(slots) == []

    def test_a_kan_reveals_another(self) -> None:
        watcher = DoraWatcher()
        watcher.observe([_sure("3s"), None, None, None, None])
        assert watcher.observe([_sure("3s"), _sure("7p"), None, None, None]) == [
            NewDora(index=1, indicator="7p")
        ]
        assert watcher.indicators == ["3s", "7p"]

    def test_reset_clears_them(self) -> None:
        watcher = DoraWatcher()
        watcher.observe([_sure("3s"), None, None, None, None])
        watcher.reset()
        assert watcher.indicators == []


class TestItIgnoresTheFlipAnimation:
    """已經翻開的指示牌讀成「沒翻開」是常態,而且**一定**是動畫。

    實測 ``20260801-110639`` 有 11 幀這樣,全部落在指示牌變化後的
    0.2~1.6 秒內,沒有一次散在局中。同一局之內指示牌只會增加 ——
    不會有東西把翻開的蓋回去。
    """

    def test_a_momentary_blank_does_not_remove_it(self) -> None:
        watcher = DoraWatcher()
        watcher.observe([_sure("3s"), None, None, None, None])
        watcher.observe([None, None, None, None, None])
        assert watcher.indicators == ["3s"]

    def test_and_it_is_not_re_emitted_afterwards(self) -> None:
        watcher = DoraWatcher()
        watcher.observe([_sure("3s"), None, None, None, None])
        watcher.observe([None, None, None, None, None])
        assert watcher.observe([_sure("3s"), None, None, None, None]) == []


class TestItWaitsForAConfidentRead:
    def test_an_unsure_read_is_not_taken(self) -> None:
        """反正下一幀它還在那裡,而且不會變。"""
        watcher = DoraWatcher()
        assert watcher.observe([_unsure("3s"), None, None, None, None]) == []
        assert watcher.indicators == []

    def test_it_is_taken_once_the_read_firms_up(self) -> None:
        watcher = DoraWatcher()
        watcher.observe([_unsure("3s"), None, None, None, None])
        assert watcher.observe([_sure("3s"), None, None, None, None]) == [
            NewDora(index=0, indicator="3s")
        ]

    def test_a_later_slot_cannot_jump_the_queue(self) -> None:
        """指示牌是從左往右翻的。

        第 0 槽還沒翻開的時候第 1 槽不可能已經翻開 —— 讀到的話那是雜訊,
        不是槓寶。收了會讓槽的編號整個錯開。
        """
        watcher = DoraWatcher()
        assert watcher.observe([None, _sure("7p"), None, None, None]) == []
        assert watcher.indicators == []
