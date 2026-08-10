"""mss fallback 後端 —— 重點在「視窗邊界會過期」這件事。

呼叫端(:class:`~mia.live.vision.VisionWorker`)找到視窗之後就一直沿用同一個
:class:`WindowInfo`,而視窗會被拖曳、會被畫布尺寸設定當場調整。邊界過期在
mss 這條路上代表**抓錯螢幕位置**,而畫面上仍然會有東西 —— 錯得完全無聲。
"""

from __future__ import annotations

import numpy as np
import pytest

from mia.capture.base import CaptureFailedError, WindowInfo
from mia.capture.mss_fallback import MSSCaptureBackend
from mia.utils.geometry import Rect


class FakeShots:
    """記下每次被要求抓哪一塊,並回傳那個大小的 BGRA。"""

    def __init__(self, scale: float = 1.0) -> None:
        self.regions: list[dict[str, int]] = []
        self.scale = scale

    def grab(self, region: dict[str, int]) -> np.ndarray:
        self.regions.append(region)
        height = round(region["height"] * self.scale)
        width = round(region["width"] * self.scale)
        return np.zeros((height, width, 4), dtype=np.uint8)


def make_window(bounds: Rect) -> WindowInfo:
    return WindowInfo(handle=7, title="雀魂麻將", owner="Chrome", bounds=bounds)


def make_backend(*, listed: list[WindowInfo], scale: float = 1.0):
    backend = MSSCaptureBackend(window_lister=lambda **_: listed)
    shots = FakeShots(scale)
    backend._sct = shots  # noqa: SLF001 - 避開真的去抓螢幕
    return backend, shots


class TestStaleBounds:
    def test_it_grabs_where_the_window_is_now(self) -> None:
        moved = make_window(Rect(400, 300, 1280, 807))
        backend, shots = make_backend(listed=[moved])

        backend.capture(make_window(Rect(0, 0, 900, 600)))  # 找到時的舊位置

        assert shots.regions == [{"left": 400, "top": 300, "width": 1280, "height": 807}]

    def test_the_frame_carries_the_new_bounds(self) -> None:
        """下游會拿 frame.window.bounds 把畫面座標換回可點擊的邏輯座標。"""
        moved = make_window(Rect(400, 300, 1280, 807))
        backend, _ = make_backend(listed=[moved])

        frame = backend.capture(make_window(Rect(0, 0, 900, 600)))

        assert frame.window.bounds == moved.bounds

    def test_scale_follows_the_new_size(self) -> None:
        """這正是實測踩到的:影像是新的、邊界是舊的,scale 就算成 1.0,
        然後畫布校正說「影像寬 2560 與畫布 1280×720 @1x 差了 1280px」。"""
        resized = make_window(Rect(30, 37, 1280, 807))
        backend, _ = make_backend(listed=[resized], scale=2.0)

        frame = backend.capture(make_window(Rect(30, 37, 2560, 870)))

        assert frame.scale == pytest.approx(2.0)

    def test_a_window_that_vanished_falls_back_to_what_we_had(self) -> None:
        """查不到就用舊的 —— 抓不到才是該報的錯,不該在這裡先炸掉。"""
        backend, shots = make_backend(listed=[])

        backend.capture(make_window(Rect(10, 20, 800, 600)))

        assert shots.regions == [{"left": 10, "top": 20, "width": 800, "height": 600}]

    def test_a_shrunk_to_nothing_window_is_reported(self) -> None:
        backend, _ = make_backend(listed=[make_window(Rect(0, 0, 0, 0))])

        with pytest.raises(CaptureFailedError):
            backend.capture(make_window(Rect(0, 0, 800, 600)))
