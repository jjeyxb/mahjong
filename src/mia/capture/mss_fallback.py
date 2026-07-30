"""mss 全螢幕 fallback 後端。

.. warning::
   **這是最後手段,不是主要路徑。** 兩個實質缺點:

   1. **會拍到自己的 Overlay。** mss 抓的是螢幕上的合成結果,不是視窗自身的
      緩衝區,所以疊在遊戲上的 HUD 會一起進到辨識輸入。使用此後端時,
      Overlay 必須完全避開辨識區域,或在擷取前暫時隱藏。
   2. **解析度減半。** macOS Retina 上實測 mss 只回傳邏輯解析度
      (1512x982),而 Quartz 可取得完整的 3024x1964。像素少一半直接影響
      模板比對的精度。

   只有在平台原生後端不可用時才該走到這裡。

視窗列舉本身仍交給平台模組處理,mss 只負責「抓指定螢幕區域」這件事。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import cv2
import mss
import numpy as np

from mia.capture.base import (
    CaptureBackend,
    CaptureFailedError,
    Frame,
    WindowInfo,
)
from mia.utils.logging import logger

__all__ = ["MSSCaptureBackend"]

WindowLister = Callable[..., list[WindowInfo]]


def _default_lister() -> WindowLister:
    """依平台挑選視窗列舉函式;無可用者則回傳空清單。"""
    if sys.platform == "darwin":
        from mia.capture import macos

        return macos.list_windows
    if sys.platform == "win32":
        from mia.capture import windows

        return windows.list_windows

    def _none(*, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG001
        logger.warning("此平台沒有視窗列舉實作,mss 後端無法定位目標視窗")
        return []

    return _none


class MSSCaptureBackend(CaptureBackend):
    """以視窗邊界為擷取區域的 mss 後端。"""

    name = "mss"

    def __init__(self, window_lister: WindowLister | None = None) -> None:
        """
        Args:
            window_lister: 視窗列舉函式。預設依平台自動選擇;測試時可注入假的。
        """
        self._list = window_lister or _default_lister()
        # mss 的實例不可跨執行緒共用,故延遲建立並綁在本後端實例上。
        self._sct: mss.base.MSSBase | None = None

    @staticmethod
    def is_available() -> bool:
        return True

    def _session(self) -> mss.base.MSSBase:
        if self._sct is None:
            self._sct = mss.MSS()
        return self._sct

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:
        return self._list(include_all=include_all)

    def capture(self, window: WindowInfo) -> Frame:
        bounds = window.bounds
        if bounds.width <= 0 or bounds.height <= 0:
            raise CaptureFailedError(f"視窗尺寸無效: {window}")

        region = {
            "left": bounds.x,
            "top": bounds.y,
            "width": bounds.width,
            "height": bounds.height,
        }
        try:
            shot = self._session().grab(region)
        except Exception as exc:
            # mss 的例外型別隨平台而異,一律包成 CaptureFailedError 讓上層好處理
            raise CaptureFailedError(f"mss 擷取失敗: {window}") from exc
        captured_at = time.monotonic()

        # mss 回傳 BGRA,前三通道即 BGR。
        # 用 cv2.cvtColor 而非 numpy 切片,原因見 capture/macos.py 的同段註解。
        array = cv2.cvtColor(np.asarray(shot, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
        scale = array.shape[1] / bounds.width if bounds.width else 1.0
        return Frame(image=array, window=window, scale=scale, captured_at=captured_at)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
