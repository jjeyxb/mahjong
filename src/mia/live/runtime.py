"""即時模式的協調層。**這個模組不 import Qt。**

三條執行緒,一個交會點
----------------------
```
擷取執行緒 ──┐
             ├─▶ UpdateBus ──▶ pump()(UI 執行緒)──▶ ViewModel ──▶ 視窗
封包執行緒 ──┘
```

:meth:`LiveRuntime.pump` 是唯一碰 ViewModel 的地方,而它只在 UI 執行緒上跑。
這件事必須成立,原因有兩層:

1. **Qt 的 widget 只能在建立它的執行緒上動。** 從工作執行緒直接改 UI 不是
   「有時候會出問題」,是未定義行為 —— 常見的表現是隨機當掉,而且當掉的位置
   跟真正的原因無關。
2. **ViewModel 自己不是執行緒安全的**(它的 docstring 就這麼寫)。

所以 :class:`LiveRuntime` 不需要鎖:工作執行緒只會碰
:class:`~mia.live.bus.UpdateBus`(它自己有鎖)與自己的
:class:`~mia.live.bus.WorkerStatus`(同上)。

呼叫端只要負責「定期呼叫 pump」。Qt 那邊是一個 ``QTimer``,測試裡就直接呼叫。

為什麼不用 Qt signal
--------------------
Qt 的 signal 跨執行緒是安全的,而且能自動排到主執行緒。但那會讓這一層直接
相依 Qt,Overlay 之外的任何呈現方式(包括測試)都得先起一個事件迴圈。
郵箱 + pump 換來的是「整條即時路徑可以在沒有 Qt 的情況下測」。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mia.live.bus import Advices, CvHand, PacketHand, UpdateBus, WorkerStatus
from mia.live.source import CaptureProcess
from mia.ui.viewmodel import ViewModel
from mia.utils.logging import logger

__all__ = ["LiveRuntime", "Worker"]


class Worker(Protocol):
    """一條工作執行緒。:class:`~threading.Thread` 的子集,加上一句狀態。"""

    status: WorkerStatus

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def is_alive(self) -> bool: ...


class LiveRuntime:
    """把工作執行緒的產出送進 ViewModel。

    Args:
        viewmodel: 要更新的 ViewModel。
        bus: 工作執行緒投遞的郵箱。
        workers: 要一起啟停的工作執行緒。
        capture: 擷取子程序;沒有(例如只跑畫面辨識)時為 ``None``。

    用法::

        runtime = LiveRuntime(model, bus, workers=[vision, packets])
        runtime.start()
        timer = QTimer(); timer.timeout.connect(runtime.pump); timer.start(100)
        ...
        runtime.stop()
    """

    def __init__(
        self,
        viewmodel: ViewModel,
        bus: UpdateBus,
        *,
        workers: Sequence[Worker] = (),
        capture: CaptureProcess | None = None,
    ) -> None:
        self._viewmodel = viewmodel
        self._bus = bus
        self._workers = list(workers)
        self._capture = capture
        self._running = False

    # ------------------------------------------------------------------ 生命週期

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """啟動擷取子程序與所有工作執行緒。

        擷取子程序先開:它要花好幾秒才會建出錄影檔,而封包執行緒會在那裡等。
        """
        if self._running:
            return
        self._running = True
        if self._capture is not None:
            self._capture.start()
        for worker in self._workers:
            worker.start()
        self.pump()

    def stop(self, *, timeout: float = 3.0) -> None:
        """停掉所有東西。可重複呼叫。

        順序:先停封包/擷取(來源),再等執行緒收尾。工作執行緒都是 daemon,
        所以就算 join 超時也不會擋住程式結束 —— 但還是要 join,引擎子程序
        要靠 ``PacketWorker`` 的 finally 才會被關掉。
        """
        if not self._running:
            return
        self._running = False
        if self._capture is not None:
            self._capture.stop()
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout)
            if worker.is_alive():
                logger.warning("{} 執行緒沒有在 {} 秒內結束", worker.status.name, timeout)

    def __enter__(self) -> LiveRuntime:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ 每一輪

    def pump(self) -> None:
        """把郵箱裡的更新套進 ViewModel。**只能在 UI 執行緒呼叫。**

        一輪之內的所有更新包在一次 :meth:`ViewModel.batch` 裡 —— 逐筆通知的話
        UI 一輪重畫三次,而中間那兩次畫的是不完整的組合(手牌已經換成下一巡、
        建議還是上一巡的)。
        """
        if self._capture is not None:
            self._capture.poll()

        with self._viewmodel.batch():
            for update in self._bus.drain():
                match update:
                    case CvHand():
                        self._viewmodel.update_cv_hand(
                            update.tiles, drawn=update.drawn, confident=update.confident
                        )
                    case PacketHand():
                        self._viewmodel.update_packet_hand(update.tiles, drawn=update.drawn)
                    case Advices():
                        self._viewmodel.update_advices(update.advices)
            self._sync_notices()

    # ------------------------------------------------------------------ 狀態列

    def _sync_notices(self) -> None:
        """把各執行緒的狀態彙整成狀態列。

        **狀態列由這裡獨佔。** ViewModel 自己也會在手牌讀不出來時附加一句話,
        但那條路在即時模式下不該用:那些訊息是黏住的(去重後永久留著),
        一幀認錯就會留一句話到程式關掉。所以要顯示的訊息一律由工作執行緒
        寫進自己的 :class:`WorkerStatus`,由這裡整批換掉。

        比對的對象是 **ViewModel 目前的 notices**,不是自己上一輪算出來的值。
        差別在於 ViewModel 會在這中間自己塞東西進去 —— 拿自己的快取來比就
        看不到那個差異,那句黏住的話會一直留在畫面上。
        """
        parts: list[str] = []
        for source in self._sources():
            message = source.read()
            if message:
                parts.append(f"{source.name}:{message}")
        notices = tuple(parts)
        if notices != self._viewmodel.state.notices:
            self._viewmodel.set_notices(notices)

    def _sources(self) -> list[WorkerStatus]:
        statuses = [w.status for w in self._workers]
        if self._capture is not None:
            # 擷取子程序排在最前面:它掛掉的話後面兩個都沒有輸入,
            # 先看到根本原因比先看到症狀有用
            statuses.insert(0, self._capture.status)
        return statuses

    def __repr__(self) -> str:
        names = ", ".join(w.status.name for w in self._workers)
        return f"<LiveRuntime running={self._running} workers=[{names}]>"
