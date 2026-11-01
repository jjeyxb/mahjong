"""Log 視窗的資料:事件流,加上給人看的中文。

不走 :class:`~mia.ui.viewmodel.ViewModel`:那個的每個訂閱者每次更新都要重畫,而 Log
與 Overlay、側邊視窗互不相干,不該讓任何一邊多畫一次。

事件流可能被改寫
----------------
畫面那條路每一輪重算(見 :mod:`mia.tracker.stream`):牌面認好了、排序補上了,前面的
事件就會變。所以這裡收的是**這一局到目前的完整事件流**,不是一則一則累加,改寫了就照
新的顯示。郵箱每個 slot 只留最新一筆,累加的話也會掉事件。

只有換局才把上一局收起來留著。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from mia.mjai.describe import describe
from mia.mjai.events import MjaiEvent, StartKyoku

__all__ = ["PACKET_SOURCE", "EventLog", "LogLine"]

#: 封包那條路的名字。只有它的摸切與手切是真的。
PACKET_SOURCE = "封包"

#: 最多留幾行。一局約 150 則,夠看完一場半莊。
_MAX_LINES = 2000


@dataclass(frozen=True, slots=True)
class LogLine:
    """一則事件。

    Attributes:
        text: 中文說明。
        mjai: MJAI 格式(一行 JSON)。
        heading: 是不是一局的開頭。
    """

    text: str
    mjai: str
    heading: bool = False


class EventLog:
    """保存事件流,有變動時通知訂閱者。"""

    def __init__(self) -> None:
        self.source = ""
        self.note = ""
        self._past: list[LogLine] = []
        self._events: tuple[MjaiEvent, ...] = ()
        self._current: list[LogLine] = []
        self._listeners: list[Callable[[], None]] = []

    @property
    def lines(self) -> tuple[LogLine, ...]:
        return (*self._past, *self._current)

    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def update(
        self, source: str, seat: int | None, events: tuple[MjaiEvent, ...], note: str = ""
    ) -> None:
        """收一份事件流。內容沒變就不通知。"""
        changed = source != self.source or note != self.note
        self.source, self.note = source, note
        if events and events != self._events:
            changed = True
            self._take(source, seat, events)
        if changed:
            self._notify()

    def clear(self) -> None:
        """全部清掉。AI 建議重開時封包會從頭再讀一遍,留著會重複。"""
        self.source = self.note = ""
        self._past.clear()
        self._events = ()
        self._current.clear()
        self._notify()

    def _take(self, source: str, seat: int | None, events: tuple[MjaiEvent, ...]) -> None:
        """只重算變動的部分:一局尾巴一輪要比兩百則,而前面幾乎都沒變。"""
        if self._events and not _same_round(self._events[0], events[0]):
            self._past.extend(self._current)
            del self._past[: max(0, len(self._past) - _MAX_LINES)]
            self._current = []
            self._events = ()
        kept = 0
        for old, new in zip(self._events, events, strict=False):
            if old != new:
                break
            kept += 1
        exact = source == PACKET_SOURCE
        self._current[kept:] = [_line(event, seat, exact) for event in events[kept:]]
        self._events = events

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()


def _same_round(old: MjaiEvent, new: MjaiEvent) -> bool:
    """開頭的事件還是不是同一局。配牌可能被改寫,只看局數。"""
    if isinstance(old, StartKyoku) and isinstance(new, StartKyoku):
        return (old.bakaze, old.kyoku, old.honba) == (new.bakaze, new.kyoku, new.honba)
    return type(old) is type(new)


def _line(event: MjaiEvent, seat: int | None, exact: bool) -> LogLine:
    text = describe(event, seat=seat, exact_discards=exact)
    mjai = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return LogLine(text, mjai, heading=isinstance(event, StartKyoku))
