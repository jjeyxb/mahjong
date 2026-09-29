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
from mia.calibration.canvas import CanvasChoice
from mia.engine.base import EngineError
from mia.engine.styles import StyleChoice
from mia.features import NAMES
from mia.live.bus import Advices, CvHand, Dangers, PacketHand, UpdateBus, WorkerStatus
from mia.live.source import CaptureLauncher
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
        (實測 0.5~0.6 秒)。這是刻意的:「關掉」如果還讓引擎佔著記憶體,
        那就不叫關掉。
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
        capture: 開遊戲的東西;沒有(``--tail``、``--no-packets``)時為 ``None``。
        style: 打法風格的共用格子;沒有 AI 建議那條路時為 ``None``。
            引擎工廠讀它決定載哪幾份權重,所以改了要重啟才生效 ——
            見 :meth:`set_style`。

    功能預設**全部關閉**,使用者用側邊視窗上的開關打開。

    瀏覽器**不會自己開** —— 要使用者按「開始遊戲」(:meth:`start_game`)。
    開一個瀏覽器並開始寫錄影檔跟「載入 130MB 權重」是同一個層級的事,該是明確
    的動作而不是打開視窗的副作用。

    擷取子程序**不受兩個功能開關影響**,一路開到使用者關掉瀏覽器為止。綁在開關
    上的話,關掉再打開會殺掉瀏覽器 —— 整場對局就沒了。它一路開著還有一個好處:
    錄影檔從按下開始遊戲那一刻就在寫,所以中途才打開 AI 建議時
    :class:`~mia.live.packets.PacketWorker` 會從頭讀一遍把局面追上來。

    用法::

        runtime = LiveRuntime(model, bus, features=[vision, advice], capture=launcher)
        runtime.start()                      # 只是開始 pump,什麼都還沒跑
        timer = QTimer(); timer.timeout.connect(runtime.pump); timer.start(100)
        runtime.start_game()                 # 開瀏覽器,開始錄
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
        capture: CaptureLauncher | None = None,
        canvas: CanvasChoice | None = None,
        style: StyleChoice | None = None,
    ) -> None:
        self._viewmodel = viewmodel
        self._bus = bus
        self._features = {f.key: f for f in features}
        self._capture = capture
        self._canvas = canvas
        self._style = style
        self._running = False
        #: 最近一次「功能起不來」的理由。見 :meth:`_enable`。
        self._failure: str | None = None

    # ------------------------------------------------------------------ 生命週期

    @property
    def running(self) -> bool:
        return self._running

    @property
    def feature_list(self) -> tuple[Feature, ...]:
        return tuple(self._features.values())

    def start(self) -> None:
        """開始運轉。**什麼都不會被啟動** —— 功能預設全關,瀏覽器等使用者按。

        剩下的工作只有讓 :meth:`pump` 有意義:在這之前 pump 出去的東西沒有人
        會收。
        """
        if self._running:
            return
        self._running = True
        self.pump()

    def can_start_game(self) -> bool:
        """MIA 在這個模式下負不負責開遊戲。

        ``--tail`` 是跟著別人正在錄的檔案走、``--no-packets`` 根本不接封包
        —— 兩種情況下遊戲都不是 MIA 開的。
        """
        return self._capture is not None

    def game_running(self) -> bool:
        """瀏覽器 / 擷取子程序現在是不是活著。給按鈕決定要不要可按。"""
        return self._capture is not None and self._capture.running

    def start_game(self) -> None:
        """開一場:開瀏覽器,開始寫一個新的錄影檔。

        換了錄影檔的話,**正在跟著舊檔案走的封包執行緒要重開** —— 它的路徑是
        建構時決定的,不重開的話畫面會停在上一場的最後一手,而且不會有任何
        錯誤訊息。重開要再付一次載權重的成本(實測 0.5~0.6 秒)。

        已經有一場在跑就什麼都不做:按這個按鈕的意思從來不是「把現在這場砍了」。
        """
        if self._capture is None:
            logger.warning("這個模式不由 MIA 開遊戲,忽略")
            return
        before = self._capture.dump
        if not self._capture.launch():
            return
        if self._capture.dump != before:
            self._restart(features.ADVICE)
        self.pump()

    # ------------------------------------------------------------------ 畫布

    def canvas(self) -> str | None:
        """現在選的畫布尺寸,``None`` 表示自動偵測。"""
        return self._canvas.key if self._canvas is not None else None

    def can_pick_canvas(self) -> bool:
        """這個模式下選畫布有沒有意義。錄影重播與 ``--no-vision`` 就沒有。"""
        return self._canvas is not None and self.available(features.VISION)

    def set_canvas(self, key: str | None) -> None:
        """改選畫布尺寸。

        兩件事會發生:**瀏覽器當場被調成新尺寸**(透過控制檔送給擷取子程序),
        以及**畫面辨識重開**(它的校正器是建構時吃設定的)。

        兩者之間有最多一個輪詢週期(250 ms)的空窗,那段時間 CV 會說「畫布
        對不上」—— 那是實話,而且它會自己好。
        """
        if self._canvas is None:
            logger.warning("這個模式不支援選畫布尺寸,忽略")
            return
        if not self._canvas.set(key):
            return
        logger.info("畫布尺寸改為 {}", self._canvas)
        self._push_canvas()
        self._restart(features.VISION)
        self.pump()

    def _push_canvas(self) -> None:
        """把新尺寸推給正在跑的擷取子程序,讓瀏覽器**當場**跟著改。

        只有那個子程序握著瀏覽器,而它的參數是啟動時用命令列傳的 —— 開下去
        之後就只剩控制檔這條路(見 :mod:`mia.live.control`)。

        沒有在跑就不必推:下次「開始遊戲」時新尺寸本來就會走命令列過去。
        """
        if self._capture is None or not self._capture.running:
            return
        assert self._canvas is not None
        self._capture.control.write(canvas=self._canvas.key)

    # ------------------------------------------------------------------ 風格

    def style(self) -> str | None:
        """現在選的風格名稱。"""
        return self._style.name if self._style is not None else None

    def styles(self) -> tuple[str, ...]:
        """可以選的風格。權重缺檔的已經被濾掉,所以列出來的都真的載得起來。"""
        return tuple(p.name for p in self._style.profiles) if self._style is not None else ()

    def can_pick_style(self) -> bool:
        """這個模式下換風格有沒有意義。

        三個條件:有這個格子、裡面至少有一個可用的風格、而且 AI 建議那條路
        真的存在(``--no-packets`` 時它整個不存在,換風格沒有東西可換)。
        """
        return (
            self._style is not None
            and bool(self._style.profiles)
            and self.available(features.ADVICE)
        )

    def set_style(self, name: str) -> None:
        """改選打法風格。

        **會把 AI 建議整個重啟** —— 引擎是建構時吃權重的,換權重沒有別的路。
        代價是重載 130MB 權重(每份實測 0.5~0.6 秒)加上重播一次事件流把局面
        追上來(``from_start=True``),那段時間畫面上沒有建議。

        功能關著的時候只是換掉格子裡的值,不做任何事 —— 下次打開就會用新的。
        """
        if self._style is None:
            logger.warning("這個模式不支援換風格,忽略")
            return
        if not self._style.set(name):
            return
        logger.info("打法風格改為 {}", self._style)
        self._restart(features.ADVICE)
        self.pump()

    def _restart(self, key: str) -> None:
        """把一個開著的功能關掉再開。關著的話什麼都不做。

        ⚠ 重開**可能失敗**:引擎工廠會去檢查權重檔在不在,不在就丟
        :class:`~mia.engine.base.EngineError`。那個例外要是漏出去,呼叫端是
        Qt 的 slot —— 整個程式當場掛掉,而使用者只是換了個選單。所以這裡接住,
        讓功能停在關閉狀態,理由寫進日誌與狀態列。
        """
        feature = self._features.get(key)
        if feature is None or not feature.enabled:
            return
        feature.disable()
        self._clear_output(key)
        self._enable(feature)

    def _enable(self, feature: Feature) -> None:
        """開一個功能,並接住「根本起不來」。

        起不來的原因是使用者改得了的(權重檔不在),所以要說出來而不是只寫
        日誌。訊息存進 :attr:`_failure` 而不是直接塞進 ViewModel ——
        :meth:`_sync_notices` 每一輪都會整批換掉狀態列,直接塞的話下一次
        pump 就被洗掉,使用者只看到它閃一下。
        """
        try:
            feature.enable()
        except EngineError as exc:
            self._failure = f"{feature.name}:起不來 — {exc}"
            logger.error(self._failure)
        else:
            self._failure = None

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

        # 撥到關**一定**要撤掉「起不來」那句話,即使它本來就是關的。
        # 起不來之後功能停在關閉狀態,所以開關畫出來就是關的 —— 使用者再撥
        # 一次是想把那句話關掉,而 `on == feature.enabled` 會讓它什麼都不做。
        if not on:
            self._failure = None
        if on == feature.enabled:
            self.pump()
            return

        if on:
            self._enable(feature)
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
            elif key == features.DANGER:
                # None 而不是空報告:「沒有這份資訊」與「算過了但沒有危險」
                # 是兩件事,而後者會讓人以為現在很安全
                self._viewmodel.update_dangers(None)

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
                        self._settle_capture()
                    case PacketHand():
                        self._viewmodel.update_packet_hand(update.tiles, drawn=update.drawn)
                        self._settle_capture()
                    case Advices():
                        self._viewmodel.update_advices(update.advices)
                    case Dangers():
                        self._viewmodel.update_dangers(update.report)
            self._sync_notices()

    def _settle_capture(self) -> None:
        """讀到手牌就代表對局真的開始了 —— 收掉擷取子程序那句啟動提示。

        兩條路(畫面、封包)任一條都算數:``--no-packets`` 時只有畫面會有東西
        進來,只認封包的話那句提示在那個模式下永遠收不掉。
        """
        if self._capture is not None:
            self._capture.settle()

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
        # 起不來的那個排最前面 —— 它沒有 worker,所以下面那個迴圈看不到它,
        # 而「開關明明是開的卻什麼都沒發生」是最需要解釋的一種狀態。
        if self._failure is not None:
            parts.append(self._failure)
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
            # 還沒按開始遊戲時它的訊息是空的,不會佔位置
            # 擷取子程序排在最前面:它掛掉的話後面兩個都沒有輸入,
            # 先看到根本原因比先看到症狀有用
            statuses.insert(0, self._capture.status)
        return statuses

    def __repr__(self) -> str:
        state = ", ".join(
            f"{f.key}={'on' if f.enabled else 'off'}" for f in self._features.values()
        )
        return f"<LiveRuntime running={self._running} {state}>"
