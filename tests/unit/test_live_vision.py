"""擷取執行緒:畫面 → 手牌。

用真實截圖貼進一張合成的整幀影像,再讓 worker 走完整條路(校正 → ROI →
read_hand → classify)。手牌本身是真的,只有「它在畫面上的位置」是安排的
—— 那正好是這一層要負責的部分。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.capture.base import CaptureBackend, CaptureFailedError, Frame, WindowInfo
from mia.config.models import AppConfig, CalibrationConfig, CaptureConfig, RoiConfig
from mia.live.bus import CvHand, UpdateBus
from mia.live.vision import VisionWorker
from mia.utils.geometry import Rect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 手牌截圖要貼在合成整幀影像的哪裡(像素)。數字本身沒有意義,
#: 重點是設定裡的正規化 ROI 必須指回同一塊。
PASTE = Rect(300, 1100, 0, 0)

#: 整幀影像的尺寸。取真實錄影的解析度(2560x1440),因為手牌 fixture 是從
#: 那個解析度裁出來的 1813x207 —— 塞進比它小的畫布會直接放不下。
FRAME_SIZE = (2560, 1440)  # (寬, 高)


def _hand_image(name: str = "hand_13_full.png") -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"讀不到 fixture {name}"
    return image


def _frame_with_hand(window: WindowInfo, hand: np.ndarray) -> tuple[Frame, RoiConfig]:
    """造一張整幀影像,把手牌貼在 :data:`PASTE`,並回傳指回那裡的 ROI 設定。"""
    width, height = FRAME_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (40, 90, 40)  # 桌布綠,不是純黑 —— 純黑會讓校正把整張剝掉
    hand_h, hand_w = hand.shape[:2]
    assert PASTE.x + hand_w <= width and PASTE.y + hand_h <= height
    canvas[PASTE.y : PASTE.y + hand_h, PASTE.x : PASTE.x + hand_w] = hand

    roi = RoiConfig(
        own_hand=(PASTE.x / width, PASTE.y / height, hand_w / width, hand_h / height)
    )
    return Frame(canvas, window, scale=1.0), roi


def _config(roi: RoiConfig, *, fps: float = 120.0) -> AppConfig:
    """校正走手動指定的逃生門 —— 這裡要測的不是邊框剝除,而且手動指定會立刻鎖定。"""
    return AppConfig(
        capture=CaptureConfig(target_fps=fps),
        calibration=CalibrationConfig(manual_table_rect=(0.0, 0.0, 1.0, 1.0)),
        roi=roi,
    )


class FakeBackend(CaptureBackend):
    """吐預先排好的幀。``None`` 表示這一次擷取失敗。"""

    name = "fake"

    def __init__(self, window: WindowInfo, frames: list[Frame | None]) -> None:
        self._window = window
        self._frames = frames
        self.calls = 0
        self.closed = False

    @staticmethod
    def is_available() -> bool:
        return True

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG002
        return [self._window]

    def capture(self, window: WindowInfo) -> Frame:  # noqa: ARG002
        self.calls += 1
        index = min(self.calls - 1, len(self._frames) - 1)
        frame = self._frames[index]
        if frame is None:
            raise CaptureFailedError("測試刻意讓這一次失敗")
        return frame

    def close(self) -> None:
        self.closed = True


class EmptyBackend(CaptureBackend):
    """一個視窗都沒有。"""

    name = "empty"

    @staticmethod
    def is_available() -> bool:
        return True

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG002
        return []

    def capture(self, window: WindowInfo) -> Frame:  # noqa: ARG002
        raise CaptureFailedError("沒有視窗")


def _run_until(worker: VisionWorker, predicate, *, timeout: float = 5.0) -> None:
    worker.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    worker.stop()
    worker.join(2)


class TestReadingAHand:
    def test_posts_the_thirteen_tiles_it_sees(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
        )
        _run_until(worker, lambda: len(bus) > 0)

        posted = bus.drain()
        assert posted, "什麼都沒投出來"
        hand = posted[0]
        assert isinstance(hand, CvHand)
        assert len(hand.tiles) == 13
        assert hand.drawn is None  # 這張 fixture 是滿手 13 張、不是自己的回合

    def test_a_hand_with_a_drawn_tile_reports_it_separately(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image("hand_14_with_draw.png"))
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: len(bus) > 0)

        hand = bus.drain()[0]
        assert isinstance(hand, CvHand)
        assert len(hand.tiles) == 13
        assert hand.drawn is not None

    def test_tiles_are_in_majsoul_notation(self, window: WindowInfo) -> None:
        """投出去的是 classify 的原始輸出。轉成 MJAI 記法是 ViewModel 的事,
        在這裡先轉會讓兩邊都以為對方負責。
        """
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: len(bus) > 0)

        hand = bus.drain()[0]
        assert isinstance(hand, CvHand)
        for tile in hand.tiles:
            assert tile[-1] in "mpsz", f"{tile} 不是雀魂記法"


class TestSkippingUnchangedFrames:
    def test_an_identical_frame_is_not_classified_twice(self, window: WindowInfo) -> None:
        """這是省下九成模板比對的地方 —— 等別人打牌時畫面是靜止的。"""
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: worker.skipped >= 3)
        assert worker.reads == 1, f"同一張畫面認了 {worker.reads} 次"
        assert worker.skipped >= 3

    def test_a_changed_frame_is_classified_again(self, window: WindowInfo) -> None:
        first, roi = _frame_with_hand(window, _hand_image())
        second, _ = _frame_with_hand(window, _hand_image("hand_14_with_draw.png"))
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            backend=FakeBackend(window, [first, second]),
            window=window,
        )
        _run_until(worker, lambda: worker.reads >= 2)
        assert worker.reads >= 2


class TestDegradedConditions:
    def test_missing_own_hand_roi_stops_with_an_explanation(self, window: WindowInfo) -> None:
        """設定裡沒量測 ROI 時要說清楚,不是靜靜地什麼都不做。"""
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(RoiConfig()),
            backend=FakeBackend(window, []),
            window=window,
        )
        worker.start()
        worker.join(3)
        assert not worker.is_alive()
        assert "own_hand" in worker.status.read()
        assert len(bus) == 0

    def test_no_game_window_says_so_and_keeps_trying(self) -> None:
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(RoiConfig(own_hand=(0.1, 0.7, 0.6, 0.1))), backend=EmptyBackend()
        )
        _run_until(worker, lambda: "找不到遊戲視窗" in worker.status.read(), timeout=3)
        assert "找不到遊戲視窗" in worker.status.read()

    def test_capture_failures_are_reported_but_not_fatal(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        # 前三次失敗,之後成功 —— 要驗的是它撐過去了
        backend = FakeBackend(window, [None, None, None, frame])
        worker = VisionWorker(bus, config=_config(roi), backend=backend, window=window)
        _run_until(worker, lambda: len(bus) > 0, timeout=5)
        assert len(bus) > 0, "擷取失敗之後就再也沒有認出手牌"

    def test_an_animation_frame_is_skipped_without_clearing_the_hand(
        self, window: WindowInfo
    ) -> None:
        """張數不合法幾乎都是抓在動畫中間。這時上一巡的手牌仍是最好的資訊。"""
        good, roi = _frame_with_hand(window, _hand_image())
        # 只留左邊三分之一,其餘整條塗成桌布 —— 剩下的張數不是 13/10/7/4/1
        partial = good.image.copy()
        hand_w = round(roi.own_hand[2] * FRAME_SIZE[0])
        hand_h = round(roi.own_hand[3] * FRAME_SIZE[1])
        cut = PASTE.x + hand_w // 3
        partial[PASTE.y : PASTE.y + hand_h, cut : PASTE.x + hand_w] = (40, 90, 40)
        broken = Frame(partial, window, scale=1.0)

        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [broken]), window=window
        )
        _run_until(worker, lambda: "合法手牌" in worker.status.read(), timeout=3)
        assert worker.reads == 0
        assert len(bus) == 0

    def test_the_backend_is_closed_when_the_thread_ends(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        backend = FakeBackend(window, [frame])
        worker = VisionWorker(UpdateBus(), config=_config(roi), backend=backend, window=window)
        _run_until(worker, lambda: worker.reads > 0)
        assert backend.closed


class TestStopping:
    def test_stop_ends_the_thread_promptly(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        worker = VisionWorker(
            UpdateBus(),
            # 低幀率:停止不該等滿一個週期
            config=_config(roi, fps=1.0),
            backend=FakeBackend(window, [frame]),
            window=window,
        )
        worker.start()
        time.sleep(0.05)
        started = time.monotonic()
        worker.stop()
        worker.join(2)
        assert not worker.is_alive()
        assert time.monotonic() - started < 1.0, "停止等了超過一個擷取週期"


@pytest.mark.parametrize("count", [1, 4])
def test_repr_is_informative(window: WindowInfo, count: int) -> None:
    frame, roi = _frame_with_hand(window, _hand_image())
    worker = VisionWorker(
        UpdateBus(),
        config=_config(roi),
        backend=FakeBackend(window, [frame] * count),
        window=window,
    )
    assert "VisionWorker" in repr(worker)


def test_threading_module_is_used_for_the_stop_event() -> None:
    """釘住「用 Event 而不是 sleep」這個決定 —— 它是停止延遲的來源。"""
    worker = VisionWorker(UpdateBus(), config=_config(RoiConfig()))
    assert isinstance(worker._stop, threading.Event)  # noqa: SLF001
