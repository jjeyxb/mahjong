"""測試共用的 fixture 與合成影像工具。"""

from __future__ import annotations

import os

# UI 測試要在無頭環境跑。這行必須在任何 Qt import **之前** —— Qt 只在第一次
# 建 QApplication 時讀這個變數,之後改就沒用了。設在 conftest 是因為它必然
# 比測試模組早載入。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mia.capture.base import Frame, WindowInfo
from mia.utils.geometry import Rect


@pytest.fixture
def window() -> WindowInfo:
    return WindowInfo(
        handle=42,
        title="雀魂麻將 MahjongSoul",
        owner="Google Chrome",
        bounds=Rect(100, 50, 1280, 720),
        pid=1234,
    )


def make_letterboxed(
    image_size: tuple[int, int],
    content_rect: Rect,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
    seed: int = 0,
) -> np.ndarray:
    """造一張「純色邊框包住一塊雜訊內容區」的合成影像。

    內容區用隨機雜訊而非純色,是為了確保偵測是真的靠「與背景不同」找到邊界,
    而不是碰巧撞對。

    Args:
        image_size: (width, height)。
        content_rect: 內容區在影像中的位置。
        background: 邊框的 BGR 顏色。
    """
    width, height = image_size
    image = np.full((height, width, 3), background, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    # 值域刻意遠離背景色,避免落在 tolerance 之內
    noise = rng.integers(80, 256, (content_rect.height, content_rect.width, 3), dtype=np.uint8)
    image[content_rect.as_slice()] = noise
    return image


def make_frame(image: np.ndarray, window: WindowInfo, scale: float = 1.0) -> Frame:
    return Frame(image=image, window=window, scale=scale)
