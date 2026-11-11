"""測試共用的 fixture 與合成影像工具。"""

from __future__ import annotations

import os

# UI 測試要在無頭環境跑。這行必須在任何 Qt import **之前** —— Qt 只在第一次
# 建 QApplication 時讀這個變數,之後改就沒用了。設在 conftest 是因為它必然
# 比測試模組早載入。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 介面語言固定繁體中文:測試比對的是原文,不能跟著這台機器的介面設定變。見 mia.i18n。
os.environ["MIA_LANGUAGE"] = "zh-TW"

import numpy as np
import pytest

from mia.capture.base import Frame, WindowInfo
from mia.utils.geometry import Rect


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """模態對話框的 ``exec()`` 在測試裡直接回「取消」。

    無頭環境沒有人按得到按鈕,``exec()`` 會永遠等下去 —— 整套測試卡死,而不是某一個失敗。
    2026-11-12 就這樣卡了一整晚:切到混合辨識會跳「YOLO 需要預設皮膚」的提醒,
    而 ``test_hybrid_ui`` 正好切了一次。要驗對話框內容的測試自己建出來檢查,不靠 exec。
    """
    try:
        from PySide6.QtWidgets import QDialog, QMessageBox
    except ImportError:
        return
    monkeypatch.setattr(QDialog, "exec", lambda _self: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(QMessageBox, "exec", lambda _self: QMessageBox.StandardButton.Cancel)


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
