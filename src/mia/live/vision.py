"""擷取執行緒:畫面 → 手牌。功能 1 的即時版。

這條路上的每一步都可能還沒準備好 —— 遊戲沒開、校正還在蒐集、正卡在發牌動畫。
所以整個迴圈的原則是**任何一步不成立就報一句話然後繼續轉**,不是拋例外結束。
使用者要的是「現在為什麼沒東西」,而一個死掉的執行緒什麼都不會說。

跳過沒變的幀
------------
每幀對 14 張牌各做一次 37 個模板的比對,是這條路上唯一貴的地方。但畫面在
一巡之內大部分時間是靜止的(等別人打牌),所以先比一次 ROI 的**位元組是否
完全相同**,相同就整幀跳過。

刻意用完全相等而不是「差異小於門檻」:門檻要靠實測資料訂,而門檻訂高了會漏掉
真的變化 —— 那個錯誤表現成「手牌卡在上一巡」,非常難察覺。完全相等沒有這個
風險,而實測靜止畫面的擷取結果本來就是位元組相同的。
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mia.calibration.canvas import CanvasChoice
from mia.calibration.stable import StableCalibrator
from mia.calibration.table import Calibration
from mia.capture.base import (
    CaptureBackend,
    CaptureError,
    CaptureFailedError,
    WindowInfo,
    WindowNotFoundError,
)
from mia.config.models import AppConfig
from mia.live.bus import CvHand, UpdateBus, WorkerStatus
from mia.utils.logging import logger
from mia.vision.roi import MissingRoiError, RoiSet
from mia.vision.tiles.classify import DEFAULT_SKIN, TemplateSet, classify_hand
from mia.vision.tiles.hand import read_hand

__all__ = ["HAND_ROI", "VisionWorker"]

#: 功能 1 只需要這一個 ROI。沒有它就沒有畫面這條路可走。
HAND_ROI = "own_hand"

#: 找不到視窗時,隔多久再找一次。頻繁列舉視窗在 macOS 上要跨行程查詢,不便宜。
_WINDOW_RETRY = 2.0

#: 連續擷取失敗幾次之後放掉視窗代號重找。視窗可能已經關掉又重開,
#: 舊的 CGWindowID 不會自己變有效。
_REFIND_AFTER = 15


class VisionWorker(threading.Thread):
    """定速擷取遊戲視窗,認出自己的手牌,投進 :class:`~mia.live.bus.UpdateBus`。

    Args:
        bus: 結果往哪投。
        config: 擷取幀率、視窗比對條件、校正參數、ROI 都從這裡來。
        backend: 擷取後端。省略則依平台自動建立。
        window: 直接指定要擷取的視窗;省略則依 ``config.capture.window`` 搜尋。
        skin: 牌面模板的皮膚。
        canvas: 使用者選的畫布尺寸,蓋過 ``config.calibration.canvas``。
    """

    def __init__(
        self,
        bus: UpdateBus,
        *,
        config: AppConfig,
        backend: CaptureBackend | None = None,
        window: WindowInfo | None = None,
        skin: str = DEFAULT_SKIN,
        canvas: CanvasChoice | None = None,
    ) -> None:
        super().__init__(name="vision", daemon=True)
        self._bus = bus
        self._config = config
        self._backend = backend
        self._window = window
        self._skin = skin
        self._stop = threading.Event()
        self.status = WorkerStatus("畫面")

        self._calibrator = StableCalibrator(config.calibration, canvas=canvas)
        self._rois: RoiSet | None = None
        self._templates: TemplateSet | None = None
        self._previous: np.ndarray | None = None
        self._failures = 0
        #: 認過幾手、跳過幾幀。用來回答「CV 到底有沒有在動」。
        self.reads = 0
        self.skipped = 0

    # ------------------------------------------------------------------ 生命週期

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self.status.alive = True
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - 執行緒裡漏出來的例外只會被吞掉
            logger.exception("擷取執行緒異常結束")
            self.status.say(f"畫面辨識已停止:{exc}")
        finally:
            self.status.alive = False
            if self._backend is not None:
                self._backend.close()

    def _run(self) -> None:
        if self._config.roi.own_hand is None:
            self.status.say("設定裡沒有 own_hand ROI,畫面辨識停用(先跑 tools/roi_annotate.py)")
            return

        try:
            self._templates = TemplateSet.load(self._skin)
        except (OSError, ValueError) as exc:
            self.status.say(f"載入牌面模板失敗:{exc}")
            return

        if self._backend is None:
            from mia.capture.factory import create_backend

            self._backend = create_backend(self._config.capture)

        interval = 1.0 / self._config.capture.target_fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                # 用 Event.wait 而不是 sleep:停止訊號才不用等滿一個週期
                self._stop.wait(next_tick - now)
                continue
            next_tick = max(now, next_tick + interval)
            self._tick()

    # ------------------------------------------------------------------ 一幀

    def _tick(self) -> None:
        window = self._ensure_window()
        if window is None:
            return

        assert self._backend is not None
        try:
            frame = self._backend.capture(window)
        except CaptureFailedError as exc:
            self._failures += 1
            self.status.say(f"擷取失敗 {self._failures} 次({exc})")
            if self._failures >= _REFIND_AFTER:
                # 視窗可能關掉又重開了 —— 舊的代號不會自己變有效
                self._window = None
                self._failures = 0
            return
        self._failures = 0

        calibration = self._calibrator.feed(frame)
        if calibration is None:
            # 「鎖不上」與「還在蒐集」是兩件事,不能都顯示成進度 —— 前者停在
            # 「校正中 3/15」不會動,看起來像卡住,而使用者其實有辦法解決。
            failure = self._calibrator.failure
            if failure is not None:
                self.status.say(f"牌桌校正失敗:{failure}")
            else:
                done, need = self._calibrator.progress
                self.status.say(f"牌桌校正中 {done}/{need}")
            return

        roi = self._crop(calibration, frame.image)
        if roi is None:
            return

        # 位元組完全相同就整幀跳過。這裡擋掉的是等別人打牌時的靜止畫面,
        # 佔了絕大多數的幀,而模板比對是這條路上唯一貴的一步。
        if self._previous is not None and np.array_equal(roi, self._previous):
            self.skipped += 1
            return
        self._previous = roi.copy()

        self._read(roi)

    def _ensure_window(self) -> WindowInfo | None:
        if self._window is not None:
            return self._window

        assert self._backend is not None
        match = self._config.capture.window
        try:
            self._window = self._backend.find_window(
                match.title_patterns,
                match.owner_patterns,
                match.host_patterns,
                min_width=match.min_width,
                min_height=match.min_height,
            )
        except WindowNotFoundError:
            self.status.say("找不到遊戲視窗 —— 雀魂開著嗎?")
            self._stop.wait(_WINDOW_RETRY)
            return None
        except CaptureError as exc:
            # 權限被拒是最常見的一種:macOS 首次執行必須手動授權螢幕錄製
            self.status.say(f"擷取後端無法使用:{exc}")
            self._stop.wait(_WINDOW_RETRY)
            return None

        logger.info("擷取目標:{}", self._window)
        self.status.say("")
        return self._window

    def _crop(self, calibration: Calibration, image: np.ndarray) -> np.ndarray | None:
        """切出手牌 ROI。校正換了就重建 :class:`RoiSet`。"""
        if self._rois is None or not self._rois.matches(calibration):
            self._rois = RoiSet(self._config.roi, calibration)
            # 換了校正,上一幀的 ROI 已經不可比 —— 不清掉會拿不同座標的影像
            # 去做「位元組相同」的判斷,結果是永遠判定為變化(不致命,但白做工)
            self._previous = None
            for warning in calibration.warnings:
                logger.warning(warning)
        try:
            return self._rois.crop(image, HAND_ROI)
        except (MissingRoiError, ValueError) as exc:
            self.status.say(f"切不出手牌區域:{exc}")
            return None

    def _read(self, roi: np.ndarray) -> None:
        assert self._templates is not None
        hand = read_hand(roi)
        if not hand.is_plausible:
            # 張數不合法幾乎都是抓在理牌或摸打的動畫中間。不辨識也不清畫面
            # —— 上一巡的手牌仍然是使用者當下最好的資訊。
            self.status.say(f"畫面上讀到 {len(hand)} 張,不是合法手牌(動畫中?)")
            return

        concealed, drawn = classify_hand(roi, hand, self._templates)
        matches = (*concealed, *(m for m in (drawn,) if m is not None))
        confident = all(m.is_confident for m in matches)
        self.reads += 1
        self.status.say("" if confident else "CV 對部分牌沒有把握,建議僅供參考")
        self._bus.post(
            CvHand(
                tiles=tuple(m.label for m in concealed),
                drawn=drawn.label if drawn is not None else None,
                confident=confident,
            )
        )

    def __repr__(self) -> str:
        return (
            f"<VisionWorker alive={self.status.alive} reads={self.reads} "
            f"skipped={self.skipped} locked={self._calibrator.locked}>"
        )
