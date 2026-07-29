"""macOS 擷取後端 —— Quartz CGWindowListCreateImage。

關於 API 選擇
-------------
``CGWindowListCreateImage`` 自 macOS 14 起被標記為 deprecated,官方建議改用
ScreenCaptureKit。實測在 **macOS 26.5.2 上仍完全正常**:

* 2x Retina 擷取 2458x1498 約 10.6 ms/frame(94 fps)
* 視窗被其他視窗完全遮擋時,仍能取得正確內容(抓的是視窗自身緩衝區)
* 回傳格式為 BGRA / little-endian / premultiplied-first

因此本後端維持使用 Quartz:它同步、無需 async callback、相依只有 pyobjc-Quartz。
若哪天 Apple 真的移除此 API,遷移路徑是 ScreenCaptureKit 的
``SCScreenshotManager.captureImageWithFilter:configuration:completionHandler:``
(macOS 14+),屆時只需新增一個後端類別,上層介面不必改動。
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from majsoul_copilot.capture.base import (
    CaptureBackend,
    CaptureFailedError,
    Frame,
    PermissionDeniedError,
    WindowInfo,
)
from majsoul_copilot.utils.geometry import Rect
from majsoul_copilot.utils.logging import logger

__all__ = ["MacOSCaptureBackend", "list_windows"]

try:  # pragma: no cover - 只有 macOS 會走到 import 成功的分支
    import Quartz

    _QUARTZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    Quartz = None
    _QUARTZ_AVAILABLE = False


# 一般應用程式視窗的 layer 為 0;選單列、Dock、控制中心等系統浮層 layer 不為 0。
_NORMAL_WINDOW_LAYER = 0


def _list_options() -> int:
    return Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements


def list_windows(*, include_all: bool = False) -> list[WindowInfo]:
    """列舉螢幕上的視窗。

    此函式獨立於後端類別之外,是為了讓 :mod:`~majsoul_copilot.capture.mss_fallback`
    能在 macOS 上重用同一套視窗列舉,而不必重複實作。
    """
    if not _QUARTZ_AVAILABLE:
        return []

    raw = Quartz.CGWindowListCopyWindowInfo(_list_options(), Quartz.kCGNullWindowID) or []
    windows: list[WindowInfo] = []
    for entry in raw:
        layer = int(entry.get("kCGWindowLayer", 0))
        if not include_all and layer != _NORMAL_WINDOW_LAYER:
            continue
        bounds = entry.get("kCGWindowBounds") or {}
        windows.append(
            WindowInfo(
                handle=int(entry.get("kCGWindowNumber", 0)),
                # 未授權螢幕錄製時 kCGWindowName 會是 None,標題就取不到。
                title=str(entry.get("kCGWindowName") or ""),
                owner=str(entry.get("kCGWindowOwnerName") or ""),
                bounds=Rect(
                    int(bounds.get("X", 0)),
                    int(bounds.get("Y", 0)),
                    int(bounds.get("Width", 0)),
                    int(bounds.get("Height", 0)),
                ),
                pid=int(entry.get("kCGWindowOwnerPID", 0)),
            )
        )
    return windows


def _cgimage_to_bgr(image: object) -> np.ndarray | None:
    """CGImage → OpenCV 的 BGR ndarray。影像無效時回傳 None。

    CGImage 的每列有 ``bytesPerRow`` 的位元組,通常會為了對齊而大於
    ``width * 4``,所以必須先用 bytesPerRow 重塑再切掉右側的 padding,
    不能直接 reshape 成 (h, w, 4)。
    """
    if image is None:
        return None
    width = int(Quartz.CGImageGetWidth(image))
    height = int(Quartz.CGImageGetHeight(image))
    if width == 0 or height == 0:
        return None

    bytes_per_row = int(Quartz.CGImageGetBytesPerRow(image))
    provider = Quartz.CGImageGetDataProvider(image)
    data = Quartz.CGDataProviderCopyData(provider)
    if data is None:
        return None

    buffer = np.frombuffer(data, dtype=np.uint8)
    expected = bytes_per_row * height
    if buffer.size < expected:  # pragma: no cover - 防禦性,理論上不會發生
        logger.warning("CGImage 緩衝區過小: {} < {}", buffer.size, expected)
        return None

    # (h, bytesPerRow/4, 4) → 切掉列尾 padding → 取 BGR 三通道。
    # macOS 的記憶體順序是 BGRA(little-endian + premultiplied-first),
    # 因此前三個通道正好就是 OpenCV 要的 B, G, R,不需要重排。
    rows = buffer[:expected].reshape((height, bytes_per_row // 4, 4))
    bgra = rows[:, :width, :]
    # 用 cv2.cvtColor 而非 np.ascontiguousarray(bgra[:, :, :3]):兩者輸出完全相同,
    # 但前者實測快約 50 倍(2458x1498 下 0.16 ms vs 7.9 ms)。numpy 的跨步切片複製
    # 在這個尺寸會直接吃掉一半的擷取幀率,而這是整個管線的熱路徑。
    # cv2 能正確處理帶列 padding 的非連續 view,回傳的是連續的新陣列
    # (同時也就脫離了 CFData 持有的緩衝區,不必再額外 copy)。
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


class MacOSCaptureBackend(CaptureBackend):
    """以 CGWindowID 為單位的 macOS 擷取後端。"""

    name = "macos"

    def __init__(self, *, retina: bool = True) -> None:
        """
        Args:
            retina: True 時使用 ``kCGWindowImageBestResolution``,取得完整
                backing-store 像素(Retina 螢幕為邏輯尺寸的 2 倍)。
                False 則使用 ``kCGWindowImageNominalResolution``,與邏輯尺寸一致。
        """
        if not _QUARTZ_AVAILABLE:
            raise RuntimeError("pyobjc-framework-Quartz 未安裝,無法使用 macOS 擷取後端")
        self.retina = retina
        self._resolution_flag = (
            Quartz.kCGWindowImageBestResolution
            if retina
            else Quartz.kCGWindowImageNominalResolution
        )

    @staticmethod
    def is_available() -> bool:
        return sys.platform == "darwin" and _QUARTZ_AVAILABLE

    # --- 權限 ---

    @staticmethod
    def has_permission() -> bool:
        """是否已取得螢幕錄製權限。"""
        if not _QUARTZ_AVAILABLE:
            return False
        return bool(Quartz.CGPreflightScreenCaptureAccess())

    @staticmethod
    def request_permission() -> bool:
        """請求螢幕錄製權限,會跳出系統對話框。

        macOS 的行為是:授權後**必須重啟本程式**權限才會生效,
        因此本函式在首次授權時通常仍會回傳 False。
        """
        if not _QUARTZ_AVAILABLE:
            return False
        Quartz.CGRequestScreenCaptureAccess()
        return bool(Quartz.CGPreflightScreenCaptureAccess())

    def ensure_permission(self) -> None:
        """確認權限,沒有就跳出授權對話框並拋出說明清楚的例外。"""
        if self.has_permission():
            return
        self.request_permission()
        if not self.has_permission():
            raise PermissionDeniedError(
                "缺少螢幕錄製權限。\n"
                "  請到「系統設定 → 隱私權與安全性 → 螢幕與系統錄製」勾選執行本程式的應用程式\n"
                "  (從終端機執行就是終端機 App;從 VS Code 執行就是 VS Code)。\n"
                "  授權後必須完全結束並重新啟動該應用程式,權限才會生效。"
            )

    # --- 擷取 ---

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:
        return list_windows(include_all=include_all)

    def capture(self, window: WindowInfo) -> Frame:
        self.ensure_permission()

        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            window.handle,
            # IgnoreFraming 去掉視窗陰影與外框,只留視窗本體(仍含標題列)。
            Quartz.kCGWindowImageBoundsIgnoreFraming | self._resolution_flag,
        )
        captured_at = time.monotonic()

        array = _cgimage_to_bgr(image)
        if array is None:
            raise CaptureFailedError(
                f"擷取視窗失敗: {window}\n"
                "  常見原因:視窗已關閉、被最小化,或移到其他桌面(Space)。"
            )

        logical_width = window.bounds.width
        scale = array.shape[1] / logical_width if logical_width else 1.0
        return Frame(image=array, window=window, scale=scale, captured_at=captured_at)
