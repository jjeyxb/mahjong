"""畫面 → 事件流 → 引擎建議。**完全不靠封包。**

與 :class:`~mia.live.packets.PacketWorker` 是同一件事的兩條路:那邊讀封包,這邊
只看畫面,最後都是一串 MJAI 事件餵進同一組引擎。Steam 版沒有封包,這是它唯一的路。

事件流會改寫
------------
封包的事件發生了就是發生了,這條路不是:副露的牌面晚幾秒才認好、立直要點數變了才知道。
:mod:`mia.tracker.stream` 每一輪都重算一整局,而引擎是有狀態的,不能改過去。所以:

* 新的流接得上舊的 → 只送尾巴。
* 接不上 → ``start_game`` 重來,從 ``start_kyoku`` 重播這一局。

Mortal 重播一局只要幾十毫秒到一秒,而且只在有新知識時才發生。這比「等到確定才送」好:
那要等四五秒,輪到自己的時候還沒排到。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from mia.config.models import RoiConfig
from mia.engine.base import Advice, AIEngine, EngineError
from mia.engine.multiplex import EngineGroup, GroupResult
from mia.live.bus import Advices, UpdateBus, WorkerStatus
from mia.live.table import TableLink, TableWorker
from mia.mjai.events import EndKyoku, MjaiEvent, StartGame
from mia.tracker import Stream
from mia.utils.logging import logger
from mia.vision.tiles.classify import DEFAULT_SKIN

__all__ = ["StreamFeeder", "VisionAdviceWorker"]

#: 多久看一次有沒有新的事件流。牌桌那條路一秒更新一次,這裡只要不拖到它。
_POLL = 0.1

#: 畫面這條路的自己永遠是 0:座位是依視角算的,見 :class:`~mia.live.table.TableWorker`。
_OWN_SEAT = 0


class StreamFeeder:
    """把每一輪的事件流接到引擎。

    Attributes:
        rewinds: 重播過幾次。一直增加代表事件流不穩定。
    """

    def __init__(self, group: EngineGroup) -> None:
        self._group = group
        self._kyoku: int | None = None
        self._sent: tuple[MjaiEvent, ...] = ()
        self._started = False
        self.rewinds = 0

    def update(self, kyoku: int, events: Sequence[MjaiEvent]) -> GroupResult | None:
        """送一輪。回傳最後一個新事件的結果,沒有新事件回 ``None``。

        Args:
            kyoku: 第幾局。換了就先收掉上一局。
            events: 這一局完整的事件流,從 ``start_kyoku`` 起。
        """
        stream = tuple(events)
        if kyoku != self._kyoku:
            if self._sent:
                self._group.react(EndKyoku())
            self._kyoku = kyoku
            self._sent = ()
        if not stream or stream == self._sent:
            return None
        sent = len(self._sent)
        if not self._started or stream[:sent] != self._sent:
            self.rewinds += self._started
            self._group.react(StartGame(id=_OWN_SEAT))
            self._started = True
            self._sent = ()
        result = None
        for event in stream[len(self._sent) :]:
            result = self._group.react(event)
        self._sent = stream
        return result


class VisionAdviceWorker(threading.Thread):
    """只看畫面,把引擎的建議投進郵箱。

    自己帶一個 :class:`TableWorker` 而不是共用放銃分析那一個:兩個功能各自開關,
    而共用的話關掉其中一個會把另一個的資料來源一起帶走。

    Args:
        bus: 結果往哪投。
        config: ROI 設定。
        engines: 要用的引擎。
        link: 畫面從哪來。
        skin: 牌面皮膚。
    """

    #: 給人看的「這份建議從哪來」。見 :attr:`~mia.live.table.TableWorker.SOURCE`。
    SOURCE = "圖像辨識"

    def __init__(
        self,
        bus: UpdateBus,
        *,
        config: RoiConfig,
        engines: Sequence[AIEngine],
        link: TableLink,
        skin: str = DEFAULT_SKIN,
    ) -> None:
        super().__init__(name="vision-advice", daemon=True)
        self._bus = bus
        self._group = EngineGroup(engines)
        self._feeder = StreamFeeder(self._group)
        self._link = link
        self._lock = threading.Lock()
        self._latest: tuple[int, Stream] | None = None
        self._stop = threading.Event()
        # 危險度投到自己的郵箱:這一條只要事件流,放銃分析有自己的開關。
        self._table = TableWorker(
            UpdateBus(), config=config, skin=skin, link=link, on_stream=self._offer
        )
        self.status = WorkerStatus("AI 建議")
        self.decisions = 0

    def _offer(self, kyoku: int, stream: Stream) -> None:
        with self._lock:
            self._latest = (kyoku, stream)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.status.alive = True
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - 執行緒裡漏出來的例外只會被吞掉
            logger.exception("畫面建議的執行緒異常結束")
            self.status.say(f"AI 建議已停止:{exc}")
        finally:
            self.status.alive = False
            self._table.stop()
            self._group.close()

    def _run(self) -> None:
        self.status.say("啟動引擎…")
        try:
            self._group.start()
        except EngineError as exc:
            self.status.say(f"引擎啟動失敗:{exc}")
            return
        self._bus.post(Advices(tuple(Advice(engine.name) for engine in self._group)))
        self._link.attach(self._table)
        self._table.start()
        self.status.say("等待對局開始")
        while not self._stop.wait(_POLL):
            with self._lock:
                latest = self._latest
                self._latest = None
            if latest is not None:
                self._feed(*latest)

    def _feed(self, kyoku: int, stream: Stream) -> None:
        if stream.doubt is not None:
            # **要閉嘴。** 拿著對不上的局面給建議比沒有建議糟:使用者會照著打。
            self.status.say(f"事件流對不上,先不給建議:{stream.doubt}")
            self._bus.post(Advices(()))
            return
        self.status.say("等手牌" if stream.stalled else "")
        result = self._feeder.update(kyoku, stream.events)
        if result is None:
            return
        for name, reason in result.failures.items():
            self.status.say(f"{name} 掉隊 — {reason.splitlines()[0]}")
        if result.decisions:
            self.decisions += 1
            self._bus.post(Advices(self._group.resolve_follow_ups(result).advices))

    def __repr__(self) -> str:
        return (
            f"<VisionAdviceWorker alive={self.status.alive} decisions={self.decisions} "
            f"rewinds={self._feeder.rewinds}>"
        )
