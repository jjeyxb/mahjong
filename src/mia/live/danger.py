"""封包 → 桌面狀態 → 放銃危險度。功能 3 的即時版。

跟 :class:`~mia.live.packets.PacketWorker` 走同一份錄影檔、同一套解碼,
但**不開引擎子程序**。這正是把功能 2 拆成兩層的兌現:「看別人打了什麼」與
「問 Mortal 該怎麼打」是兩件代價差很多的事,不該綁在同一個開關上。

為什麼是各自讀一次檔案,而不是共用一個 worker
----------------------------------------------
兩個功能可以各自開關,而 :class:`~mia.live.runtime.Feature` 的模型是「開啟
就造一個新的 worker,關閉就丟掉」。共用的話得做引用計數,還要處理「AI 建議
關掉了但放銃分析還開著」這種狀態 —— 那是這個專案一路避開的那種耦合。

代價是同一份錄影檔被解析兩次。實測整場東風戰(1398 個 frame)解析約 2 秒,
之後是每秒幾個事件,可以忽略。**便宜的重複勝過一個要維護的共用狀態。**
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

from mia.analysis.danger import DangerReport, assess
from mia.groundtruth.stream import MjaiDecoder
from mia.live.bus import Dangers, UpdateBus, WorkerStatus
from mia.live.packets import DumpTail
from mia.mjai.events import MjaiEvent
from mia.mjai.handstate import HandTracker, UnknownSeatError
from mia.mjai.table import TableTracker
from mia.utils.logging import logger

__all__ = ["DangerWorker"]


class DangerWorker(threading.Thread):
    """讀封包錄影,追蹤整桌狀態,把每張牌的危險度投進郵箱。

    Args:
        bus: 結果往哪投。
        dump: 要跟著走的錄影檔。
        from_start: 見 :class:`~mia.live.packets.DumpTail`。從頭讀是必要的
            —— 安全牌是**整局累積**出來的,從中途接上會少掉前面所有的捨牌,
            而少掉的方向是**把安全牌說成不安全**(保守),不是反過來。
    """

    def __init__(
        self,
        bus: UpdateBus,
        *,
        dump: Path | str,
        from_start: bool = True,
    ) -> None:
        super().__init__(name="danger", daemon=True)
        self._bus = bus
        self._tail = DumpTail(dump, from_start=from_start)
        self._decoder = MjaiDecoder()
        self._table = TableTracker()
        self._hand = HandTracker(_name="danger")
        self._stop = threading.Event()
        self.status = WorkerStatus("放銃")

        #: 上一次投出去的報告。一樣就不投 —— 別家摸牌之類的事件佔多數,
        #: 那些不會改變任何一張牌的危險度。
        self._last: DangerReport | None = None
        #: 統計:處理過幾個事件、投了幾次。
        self.events = 0
        self.posts = 0

    # ------------------------------------------------------------------ 生命週期

    def stop(self) -> None:
        self._stop.set()

    @property
    def seat(self) -> int | None:
        return self._table.seat

    def run(self) -> None:
        self.status.alive = True
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - 執行緒裡漏出來的例外只會被吞掉
            logger.exception("放銃分析執行緒異常結束")
            self.status.say(f"放銃分析已停止:{exc}")
        finally:
            self.status.alive = False

    def _run(self) -> None:
        self.status.say(f"等待對局開始({self._tail.path.name})")
        for frame in self._tail.frames(self._stop):
            decoded = self._decoder.feed(frame)
            if decoded is None:
                continue
            for event in decoded.events:
                self._handle(event)

    # ------------------------------------------------------------------ 一個事件

    def _handle(self, event: MjaiEvent) -> None:
        if event.TYPE == "start_game":
            # 新的一場:座位可能變了,兩個追蹤器都要從頭來
            self._table = TableTracker()
            self._hand = HandTracker(_name="danger")
            self._last = None

        self.events += 1
        self._table.handle(event)
        try:
            self._hand.feed(event)
        except UnknownSeatError as exc:
            logger.warning("放銃分析的手牌追蹤:{}", exc)
            return

        self._publish(self._hand.tiles)

    def _publish(self, hand: Sequence[str]) -> None:
        """算一次並投出去 —— 但只在結果真的變了的時候。

        別家摸牌、寶牌翻開之類的事件佔多數,那些不會改變任何一張牌的危險度。
        :class:`DangerReport` 整棵都是 frozen dataclass 與 tuple,所以直接
        用值比較就夠了。
        """
        if not hand:
            return
        report = assess(list(hand), self._table)
        if report == self._last:
            return
        self._last = report
        self.posts += 1
        self._bus.post(Dangers(report))
        self.status.say("")

    def __repr__(self) -> str:
        return (
            f"<DangerWorker alive={self.status.alive} seat={self.seat} "
            f"events={self.events} posts={self.posts}>"
        )
