"""One YOLO inference producer, shared by hand display, advice and danger views."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from mia.analysis.danger import assess
from mia.calibration.canvas import CanvasChoice
from mia.calibration.manual import HandRoiChoice, TableRectChoice
from mia.calibration.table import Calibration
from mia.config.models import AppConfig
from mia.engine.base import EngineError
from mia.hybrid.merge import StreamMerger
from mia.live.bus import CvHand, Dangers, TileChoices, UpdateBus, WorkerStatus
from mia.live.table import TableWorker
from mia.live.vision import VisionWorker
from mia.mjai.events import StartGame
from mia.mjai.handstate import HandTracker
from mia.mjai.table import TableTracker
from mia.mjai.tiles import mjai_to_ms
from mia.platforms import WEB
from mia.tracker import Stream
from mia.utils.logging import logger
from mia.yolo.client import YoloClient
from mia.yolo.stream import LedgerStream

Listener = Callable[[int, Stream], None]

#: YOLO 等開局等這麼久就記一筆它卡在哪,之後原因變了再記(同樣隔這麼久才記一次,原因來回跳
#: 也不會洗版)。換局之間的結算畫面本來就要等幾秒,所以不能太短。
_START_WARN = 8.0

#: 推論子程序中斷(死掉、回應逾時)最多重啟幾次。一場裡偶爾一次是意外,接連幾次就是環境壞了,
#: 再重啟只是反覆載模型。超過之後混合辨識只剩樣板那條,純 YOLO 就停。
_RESTARTS = 3


class YoloHub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: set[Listener] = set()
        self._latest: tuple[int, Stream] | None = None

    def subscribe(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.add(listener)
            latest = self._latest
            if latest is not None:
                listener(*latest)

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def clear(self) -> None:
        with self._lock:
            self._latest = None

    def publish(self, kyoku: int, stream: Stream) -> None:
        with self._lock:
            self._latest = (kyoku, stream)
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(kyoku, stream)


class YoloStreamSource:
    """An advice source with the same lifecycle as a TableWorker, without inference."""

    def __init__(self, hub: YoloHub, listener: Listener, *, name: str = "YOLO") -> None:
        self._hub, self._listener = hub, listener
        self.name = name

    def start(self) -> None:
        self._hub.subscribe(self._listener)

    def stop(self) -> None:
        self._hub.unsubscribe(self._listener)


class YoloVisionWorker(threading.Thread):
    SOURCE = "圖像辨識"

    def __init__(
        self,
        bus: UpdateBus,
        *,
        config: AppConfig,
        hub: YoloHub,
        canvas: CanvasChoice | None = None,
        platform: str = WEB,
        hand_roi: HandRoiChoice | None = None,
        table_rect: TableRectChoice | None = None,
        hybrid: bool = False,
        skin: str = "mjpface_default",
    ) -> None:
        super().__init__(name="yolo-vision", daemon=True)
        self._bus, self._hub = bus, hub
        self.backend = "hybrid" if hybrid else "yolo"
        self.label = "混合辨識" if hybrid else "YOLO"
        #: 推論子程序。重啟過 :data:`_RESTARTS` 次還是死掉就是 ``None``,見 :meth:`_revive`。
        self._client: YoloClient | None = YoloClient()
        self._restarts = 0
        # 混合辨識:同一份畫面另外交給樣板的牌桌那條路,兩條事件流在事件層合併再發佈。
        # 見 mia.hybrid.merge。手牌由擷取那一條用樣板認,樣板的牌桌要它。
        self._merger: StreamMerger | None = None
        self._table: TableWorker | None = None
        if hybrid:
            self._merger = StreamMerger(
                hub.publish, on_choices=lambda _kyoku, choices: bus.post(TileChoices(choices))
            )
            self._table = TableWorker(
                UpdateBus(),
                config=config.roi,
                skin=skin,
                on_stream=lambda kyoku, stream: self._offer(StreamMerger.TEMPLATE, kyoku, stream),
            )
        self._stream = LedgerStream()
        self._lock = threading.Lock()
        self._fresh = threading.Event()
        self._stopping = threading.Event()
        self._latest: tuple[np.ndarray, Calibration, float] | None = None
        self._capture = VisionWorker(
            bus,
            config=config,
            canvas=canvas,
            platform=platform,
            hand_roi=hand_roi,
            table_rect=table_rect,
            sink=self,
            recognize_hand=hybrid,
            skin=skin,
        )
        self.status = WorkerStatus("畫面")
        self.frames = 0
        #: 從什麼時候開始等開局,與已經記過的那個原因。見 _watch_start。
        self._waiting_since: float | None = None
        self._waiting_logged = ""
        self._waiting_logged_at = 0.0
        #: 已經記過「借用樣板點數」的那一局。
        self._borrowed = 0

    def offer_frame(self, image: np.ndarray, calibration: Calibration) -> None:
        with self._lock:
            self._latest = (image, calibration, time.time())
        self._fresh.set()
        if self._table is not None:
            self._table.offer_frame(image, calibration)

    def offer_hand(self, tiles: Sequence[str], drawn: str | None = None) -> None:
        if self._table is not None:
            self._table.offer_hand(tiles, drawn)

    def pick(self, kyoku: int, actor: int, nth: int, pai: str) -> None:
        """玩家看畫面選了兩邊讀成不同的那張是哪張。UI 執行緒呼叫。見 :meth:`StreamMerger.pick`。"""
        if self._merger is not None:
            self._merger.pick(kyoku, actor, nth, pai)

    def _offer(self, source: str, kyoku: int, stream: Stream) -> None:
        """事件流交出去:混合辨識先合併,YOLO 直接發佈。"""
        if self._merger is not None:
            self._merger.offer(source, kyoku, stream)
        else:
            self._hub.publish(kyoku, stream)

    def _watch_start(self, started: bool) -> None:
        """等開局等太久就記下 :attr:`LedgerStream.waiting`。實打才出現過,重播重現不了。"""
        now = time.monotonic()
        if started:
            if self._waiting_logged and self._waiting_since is not None:
                logger.warning(
                    "YOLO 第 {} 局開局資訊到了,等了 {:.1f} 秒",
                    self._stream.kyoku,
                    now - self._waiting_since,
                )
            self._waiting_since, self._waiting_logged = None, ""
            return
        if self._waiting_since is None:
            self._waiting_since = now
        reason = self._stream.waiting
        if (
            now - self._waiting_since > _START_WARN
            and now - self._waiting_logged_at > _START_WARN
            and reason != self._waiting_logged
        ):
            self._waiting_logged, self._waiting_logged_at = reason, now
            logger.warning(
                "YOLO 第 {} 局等開局 {:.0f} 秒:{}",
                self._stream.kyoku,
                now - self._waiting_since,
                reason or "沒有說明",
            )

    def stop(self) -> None:
        """UI 執行緒呼叫,**不等任何東西**。

        以前在這裡 ``close()`` 子程序,等它結束最多四秒,整個視窗跟著卡住。現在只送中斷訊號,
        讓卡在等回應的工作執行緒馬上醒來;真的關掉子程序交給 :meth:`run` 的 finally,
        另外開一條背景執行緒保底:子程序卡死不讀管線時,工作執行緒會卡在寫入,finally 走不到。
        """
        self._stopping.set()
        self._fresh.set()
        self._capture.stop()
        if self._table is not None:
            self._table.stop()
        self._hub.publish(self._stream.kyoku, Stream(doubt=f"{self.label}已停止"))
        client = self._client
        if client is not None:
            client.abort()
            threading.Thread(target=client.close, name="yolo-close", daemon=True).start()

    def _revive(self, exc: EngineError) -> bool:
        """推論子程序中斷了,重開一個。重啟過太多次就放棄,回 ``False``。

        以前一死就讓整條執行緒結束,狀態列停在「已停止」,要手動關掉再開;混合辨識連樣板那條
        也一起收掉。
        """
        reason: EngineError = exc
        while self._restarts < _RESTARTS:
            self._restarts += 1
            logger.warning("YOLO 推論子程序中斷:{},第 {} 次重啟", reason, self._restarts)
            self.status.say(f"{self.label}重啟中…")
            if self._client is not None:
                self._client.close()
            try:
                client = YoloClient()
                client.start()
            except EngineError as again:
                reason = again
                continue
            self._client = client
            return True
        logger.error("YOLO 推論子程序重啟 {} 次都不成,不再重啟:{}", _RESTARTS, reason)
        return False

    def run(self) -> None:
        self.status.alive = True
        self._hub.clear()
        try:
            self.status.say(f"載入{self.label}模型…")
            assert self._client is not None
            self._client.start()
            if self._table is not None:
                self._table.start()
            self._capture.start()
            while not self._stopping.is_set():
                self._fresh.wait(0.25)
                self._fresh.clear()
                with self._lock:
                    latest, self._latest = self._latest, None
                if self._client is None:
                    # YOLO 救不回來了。畫面照樣經 offer_frame 進樣板那條,合併層只剩樣板。
                    self.status.say(f"{self.label}:YOLO 已停止,只用樣板")
                    continue
                if latest is None:
                    self.status.say(self._capture.status.read() or f"{self.label}等待遊戲畫面")
                    continue
                image, calibration, wall = latest
                r = calibration.table_rect
                cropped = image[r.y : r.y + r.height, r.x : r.x + r.width]
                if cropped.size == 0:
                    self.status.say(f"{self.label}找不到牌桌，請重新框選遊戲畫面")
                    continue
                frame = cv2.resize(cropped, (1920, 1080), interpolation=cv2.INTER_AREA)
                try:
                    payload = self._client.feed(frame, wall)
                except EngineError as exc:
                    if self._stopping.is_set():
                        break
                    if self._revive(exc):
                        continue
                    if self._table is None:
                        raise
                    self._client = None
                    self._offer(StreamMerger.YOLO, self._stream.kyoku, Stream(stalled=True))
                    continue
                if self._stopping.is_set():
                    break
                opening = None if self._merger is None else self._merger.template_opening()
                stream = self._stream.update(payload, opening=opening)
                if self._stream.borrowed and self._borrowed != self._stream.kyoku:
                    self._borrowed = self._stream.kyoku
                    logger.info("YOLO 第 {} 局讀不到點數,借用樣板的", self._stream.kyoku)
                self._watch_start(bool(stream.events))
                if not stream.events:
                    stream = Stream(stalled=True, doubt=f"{self.label}等待開局資訊")
                self._offer(StreamMerger.YOLO, self._stream.kyoku, stream)
                hand = payload.get("hand") or {}
                # 混合辨識的手牌由擷取那一條用樣板認,已經投過了。
                if self._table is None and hand.get("confident"):
                    tiles = list(hand["tiles"])
                    drawn = hand.get("drawn")
                    if drawn in tiles:
                        tiles.remove(drawn)
                    self._bus.post(
                        CvHand(
                            tuple(mjai_to_ms(t) for t in tiles),
                            mjai_to_ms(drawn) if drawn else None,
                            True,
                        )
                    )
                self.frames += 1
                self.status.say(
                    stream.doubt
                    or (
                        f"{self.label}等待開局資訊"
                        if not stream.events
                        else f"{self.label} · {self._client.device.upper()}"
                    )
                )
        except Exception as exc:  # noqa: BLE001
            if not self._stopping.is_set():
                logger.exception("YOLO 辨識停止")
                self.status.say(f"{self.label}已停止:{exc}")
                self._hub.publish(self._stream.kyoku, Stream(doubt=f"{self.label}已停止"))
                self._bus.post(CvHand(()))
        finally:
            # 問到一半停掉的話,那幾張不能一直留在畫面上。
            self._bus.post(TileChoices())
            self._capture.stop()
            if self._capture.ident is not None:
                self._capture.join(timeout=2)
            if self._table is not None:
                self._table.stop()
                if self._table.ident is not None:
                    self._table.join(timeout=2)
            if self._client is not None:
                self._client.close()
            self.status.alive = False


class YoloDangerWorker(threading.Thread):
    SOURCE = "圖像辨識"

    def __init__(self, bus: UpdateBus, hub: YoloHub, *, name: str = "YOLO") -> None:
        super().__init__(name="yolo-danger", daemon=True)
        self._bus, self._hub = bus, hub
        self._label = name
        self._lock = threading.Lock()
        self._latest: Stream | None = None
        self._fresh = threading.Event()
        self._stopping = threading.Event()
        self.status = WorkerStatus("放銃分析")

    def _offer(self, _kyoku: int, stream: Stream) -> None:
        with self._lock:
            self._latest = stream
        self._fresh.set()

    def stop(self) -> None:
        self._stopping.set()
        self._fresh.set()
        self._hub.unsubscribe(self._offer)

    def run(self) -> None:
        self.status.alive = True
        self._hub.subscribe(self._offer)
        try:
            while not self._stopping.is_set():
                self._fresh.wait(0.5)
                self._fresh.clear()
                with self._lock:
                    stream, self._latest = self._latest, None
                if stream is None or not stream.events:
                    continue
                if stream.doubt:
                    self._bus.post(Dangers(assess([], TableTracker())))
                    self.status.say(stream.doubt)
                    continue
                table, hand = TableTracker(), HandTracker()
                table.handle(StartGame(id=0))
                hand.feed(StartGame(id=0))
                for event in stream.events:
                    table.handle(event)
                    hand.feed(event)
                if hand.in_sync and hand.tiles:
                    self._bus.post(Dangers(assess(hand.tiles, table), table.visible(hand.tiles)))
                self.status.say(self._label)
        except Exception as exc:  # noqa: BLE001
            logger.exception("YOLO 放銃分析停止")
            self.status.say(f"放銃分析已停止:{exc}")
        finally:
            self._hub.unsubscribe(self._offer)
            self.status.alive = False
