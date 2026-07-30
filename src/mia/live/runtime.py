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

from collections.abc import Callable, Sequence
from typing import Protocol

from mia import features
from mia.features import NAMES
from mia.live.bus import Advices, CvHand, PacketHand, UpdateBus, WorkerStatus
from mia.live.source import CaptureProcess
from mia.ui.viewmodel import ViewModel
from mia.utils.logging import logger

__all__ = ["Feature", "LiveRuntime", "Worker"]


class Worker(Protocol):
    """一條工作執行緒。:class:`~threading.Thread` 的子集,加上一句狀態。"""

    status: WorkerStatus

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def is_alive(self) -> bool: ...


class Feature:
    """一個可以獨立開關的功能。

    Args:
        key: 識別字,見 :mod:`mia.features`。
        factory: **每次開啟都造一個新的 worker。** 不重用舊的:worker 的狀態
            (手牌追蹤、liqi 解析器、引擎子程序)在停掉之後已經沒有意義,
            而重用一個停掉的執行緒在 Python 裡根本不合法(``Thread`` 不能
            重新 ``start()``)。

    Note:
        關掉再開啟會付一次完整的啟動成本 —— AI 建議那條路要重載 130MB 權重
        (約 10 秒)。這是刻意的:「關掉」如果還讓引擎佔著記憶體,那就不叫關掉。
        開啟途中狀態列會說「啟動引擎…」。
    """

    def __init__(self, key: str, factory: Callable[[], Worker]) -> None:
        self.key = key
        self.name = NAMES.get(key, key)
        self._factory = factory
        self.worker: Worker | None = None

    @property
    def enabled(self) -> bool:
        return self.worker is not None

    def enable(self) -> None:
        if self.worker is not None:
            return
        worker = self._factory()
        self.worker = worker
        worker.start()
        logger.info("功能已開啟:{}", self.name)

    def disable(self, *, timeout: float = 3.0) -> None:
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        worker.stop()
        worker.join(timeout)
        if worker.is_alive():
            logger.warning("{} 的執行緒沒有在 {} 秒內結束", self.name, timeout)
        logger.info("功能已關閉:{}", self.name)

    def __repr__(self) -> str:
        return f"<Feature {self.key} enabled={self.enabled}>"


class LiveRuntime:
    """把工作執行緒的產出送進 ViewModel。

    Args:
        viewmodel: 要更新的 ViewModel。
        bus: 工作執行緒投遞的郵箱。
        features: 可以獨立開關的功能。**預設全部關閉。**
        capture: 擷取子程序;沒有(例如只跑畫面辨識)時為 ``None``。

    功能預設**全部關閉**,使用者用側邊視窗上的開關打開。

    擷取子程序**不受開關影響**,跟著 :meth:`start` 一起開、一路開著。綁在開關上
    的話,關掉再打開會殺掉瀏覽器 —— 整場對局就沒了。留著它還有一個好處:
    錄影檔從一開始就在寫,所以中途才打開 AI 建議時
    :class:`~mia.live.packets.PacketWorker` 會從頭讀一遍把局面追上來。

    用法::

        runtime = LiveRuntime(model, bus, features=[vision, advice], capture=capture)
        runtime.start()                      # 只開擷取子程序,功能仍關著
        timer = QTimer(); timer.timeout.connect(runtime.pump); timer.start(100)
        runtime.set_enabled(features.ADVICE, True)
        ...
        runtime.stop()
    """

    def __init__(
        self,
        viewmodel: ViewModel,
        bus: UpdateBus,
        *,
        features: Sequence[Feature] = (),
        capture: CaptureProcess | None = None,
    ) -> None:
        self._viewmodel = viewmodel
        self._bus = bus
        self._features = {f.key: f for f in features}
        self._capture = capture
        self._running = False

    # ------------------------------------------------------------------ 生命週期

    @property
    def running(self) -> bool:
        return self._running

    @property
    def feature_list(self) -> tuple[Feature, ...]:
        return tuple(self._features.values())

    def start(self) -> None:
        """啟動擷取子程序。**功能本身不會被啟動** —— 預設全部關著。

        擷取子程序要花好幾秒才會建出錄影檔,所以越早開越好:等使用者按下
        AI 建議的開關時,錄影檔通常已經在了。
        """
        if self._running:
            return
        self._running = True
        if self._capture is not None:
            self._capture.start()
        self.pump()

    def stop(self, *, timeout: float = 3.0) -> None:
        """停掉所有東西。可重複呼叫。

        順序:先停擷取(來源),再關功能。工作執行緒都是 daemon,所以就算
        join 超時也不會擋住程式結束 —— 但還是要 join,引擎子程序要靠
        ``PacketWorker`` 的 finally 才會被關掉。
        """
        if not self._running:
            return
        self._running = False
        if self._capture is not None:
            self._capture.stop()
        for feature in self._features.values():
            feature.disable(timeout=timeout)

    # ------------------------------------------------------------------ 開關

    def available(self, key: str) -> bool:
        """這個功能有沒有被建起來。

        ``--no-vision`` / ``--no-packets`` 會讓對應的功能整個不存在 —— 那與
        「存在但關著」是兩件事,UI 要分得出來才能把開關畫成停用而不是可按。
        """
        return key in self._features

    def is_enabled(self, key: str) -> bool:
        feature = self._features.get(key)
        return feature is not None and feature.enabled

    def set_enabled(self, key: str, on: bool) -> None:
        """開啟或關閉一個功能。認不得的 key 會被忽略。

        關閉時**要把它在畫面上的產出一起清掉**。留著的話最後一次的手牌或建議
        會停在畫面上,看起來像還在運作 —— 那比空白糟得多,因為使用者會照著
        一個已經不再更新的建議打牌。
        """
        feature = self._features.get(key)
        if feature is None:
            logger.warning("沒有名為 {!r} 的功能,忽略", key)
            return
        if on == feature.enabled:
            return

        if on:
            feature.enable()
        else:
            feature.disable()
            self._clear_output(key)
        self.pump()

    def _clear_output(self, key: str) -> None:
        """把某個功能在 ViewState 上留下的東西清空。"""
        with self._viewmodel.batch():
            if key == features.VISION:
                self._viewmodel.update_cv_hand(())
            elif key == features.ADVICE:
                self._viewmodel.update_packet_hand(())
                self._viewmodel.update_advices([])

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
        # 關著的功能不佔狀態列 —— 那個位置要留給真的出問題的那個。
        # 「它是關著的」由開關本身表達,不需要再寫一句話。
        statuses = [f.worker.status for f in self._features.values() if f.worker is not None]
        if self._capture is not None:
            # 擷取子程序排在最前面:它掛掉的話後面兩個都沒有輸入,
            # 先看到根本原因比先看到症狀有用
            statuses.insert(0, self._capture.status)
        return statuses

    def __repr__(self) -> str:
        state = ", ".join(
            f"{f.key}={'on' if f.enabled else 'off'}" for f in self._features.values()
        )
        return f"<LiveRuntime running={self._running} {state}>"
