"""Windows 擷取後端 —— PrintWindow + PW_RENDERFULLCONTENT。

.. warning::
   **本模組尚未在真實 Windows 機器上驗證過。** 開發環境為 macOS。
   首次在 Windows 上執行時請先跑 ``python tools/capture_probe.py --list --capture ...``
   逐項確認,特別是下列三點:

   1. ``PW_RENDERFULLCONTENT`` 對硬體加速視窗(Chrome / Edge / Electron)是否
      真的抓得到內容,而不是一片全黑。若全黑,改用 dxcam 或 Windows Graphics
      Capture(見下方「備案」)。
   2. DPI 縮放下 ``DwmGetWindowAttribute`` 回傳的邊界是否與擷取影像尺寸一致。
   3. 遊戲視窗被其他視窗遮擋時是否仍能取得完整內容。

備案
----
若 PrintWindow 對雀魂所在的視窗無效,替代方案依推薦順序:

* **dxcam**(已列在 requirements)—— Desktop Duplication API,速度最快,
  但只能抓螢幕區域,無法排除疊在上面的 Overlay,需讓 Overlay 避開辨識區。
* **Windows Graphics Capture**(``winrt`` / ``windows-capture``)—— 真正的
  單視窗擷取,是 Windows 上與 macOS Quartz 對等的方案,但相依較重。
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

import cv2
import numpy as np

from majsoul_copilot.capture.base import (
    CaptureBackend,
    CaptureFailedError,
    Frame,
    WindowInfo,
)
from majsoul_copilot.utils.geometry import Rect
from majsoul_copilot.utils.logging import logger

__all__ = ["WindowsCaptureBackend", "enable_dpi_awareness", "list_windows"]

try:  # pragma: no cover - 僅 Windows
    import win32api
    import win32con
    import win32gui
    import win32process
    import win32ui

    _WIN32_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WIN32_AVAILABLE = False


# PrintWindow 旗標:渲染完整內容,含 DirectComposition / 硬體加速圖層。
# 沒有這個旗標,Chrome、Edge、Electron 這類視窗會抓到全黑。
PW_RENDERFULLCONTENT = 0x00000002

# DwmGetWindowAttribute
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14

# SetProcessDpiAwarenessContext
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

_dpi_awareness_set = False


def enable_dpi_awareness() -> bool:
    """宣告本行程為 per-monitor DPI aware。

    **必須在建立任何視窗或取得任何座標之前呼叫。** 否則在高 DPI 螢幕上,
    系統會偷偷把座標與視窗尺寸做虛擬化縮放,導致 ``GetWindowRect`` 回報的
    數值與實際像素對不起來,ROI 全部偏移。

    重複呼叫是安全的(第二次會失敗但被忽略)。

    Returns:
        是否成功設定。舊版 Windows 沒有這個 API 時回傳 False。
    """
    global _dpi_awareness_set
    if _dpi_awareness_set:
        return True
    try:
        ok = bool(
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            )
        )
    except (AttributeError, OSError):  # pragma: no cover - Windows 8.1 以前
        try:
            # 退回舊 API:2 = PROCESS_PER_MONITOR_DPI_AWARE
            ok = ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0
        except (AttributeError, OSError):
            logger.warning("無法設定 DPI awareness,高 DPI 螢幕上座標可能失準")
            return False
    _dpi_awareness_set = ok
    return ok


def _extended_frame_bounds(hwnd: int) -> Rect:
    """取得視窗的實際可見邊界。

    ``GetWindowRect`` 在 Windows 10 以後會包含視窗周圍那圈不可見的縮放邊框
    (通常每邊 7~8 px),直接拿來當擷取尺寸會多出黑邊。DWM 的
    ``DWMWA_EXTENDED_FRAME_BOUNDS`` 才是視覺上的真實邊界。
    """
    rect = wintypes.RECT()
    hresult = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hresult != 0:  # 失敗就退回 GetWindowRect
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return Rect.from_bounds(left, top, right, bottom)
    return Rect.from_bounds(rect.left, rect.top, rect.right, rect.bottom)


def _is_cloaked(hwnd: int) -> bool:
    """視窗是否被 DWM 隱藏(其他虛擬桌面、UWP 暫停中的 App)。"""
    cloaked = wintypes.DWORD()
    hresult = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_CLOAKED),
        ctypes.byref(cloaked),
        ctypes.sizeof(cloaked),
    )
    return hresult == 0 and cloaked.value != 0


def _process_name(hwnd: int) -> str:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        try:
            path = win32process.GetModuleFileNameEx(handle, 0)
        finally:
            win32api.CloseHandle(handle)
        return path.rsplit("\\", 1)[-1]
    except Exception:  # noqa: BLE001 - 權限不足等情況很常見,不值得中斷列舉
        return ""


def _window_pid(hwnd: int) -> int:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return int(pid)
    except Exception:  # noqa: BLE001
        return 0


def list_windows(*, include_all: bool = False) -> list[WindowInfo]:
    """列舉頂層視窗。與 macOS 版同名同介面,供 mss fallback 重用。"""
    if not _WIN32_AVAILABLE:
        return []

    collected: list[WindowInfo] = []

    def _callback(hwnd: int, _: object) -> bool:
        if not include_all:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.GetWindow(hwnd, win32con.GW_OWNER):
                return True  # 屬於別人的子視窗(對話框、工具視窗)
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if ex_style & win32con.WS_EX_TOOLWINDOW:
                return True
            if _is_cloaked(hwnd):
                return True

        bounds = _extended_frame_bounds(hwnd)
        if not include_all and (bounds.width <= 0 or bounds.height <= 0):
            return True

        collected.append(
            WindowInfo(
                handle=int(hwnd),
                title=win32gui.GetWindowText(hwnd) or "",
                owner=_process_name(hwnd),
                bounds=bounds,
                pid=_window_pid(hwnd),
            )
        )
        return True

    win32gui.EnumWindows(_callback, None)
    return collected


class WindowsCaptureBackend(CaptureBackend):
    """以 HWND 為單位的 Windows 擷取後端。"""

    name = "windows"

    def __init__(self) -> None:
        if not _WIN32_AVAILABLE:
            raise RuntimeError("pywin32 未安裝,無法使用 Windows 擷取後端")
        enable_dpi_awareness()

    @staticmethod
    def is_available() -> bool:
        return sys.platform == "win32" and _WIN32_AVAILABLE

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:
        return list_windows(include_all=include_all)

    def capture(self, window: WindowInfo) -> Frame:
        hwnd = window.handle
        if not win32gui.IsWindow(hwnd):
            raise CaptureFailedError(f"視窗已不存在: {window}")

        # 用當下的邊界而非快取的 window.bounds —— 使用者可能剛剛縮放了視窗。
        bounds = _extended_frame_bounds(hwnd)
        width, height = bounds.width, bounds.height
        if width <= 0 or height <= 0:
            raise CaptureFailedError(f"視窗尺寸無效({width}x{height}),可能已最小化: {window}")

        window_dc = mfc_dc = mem_dc = bitmap = None
        try:
            window_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(window_dc)
            mem_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            mem_dc.SelectObject(bitmap)

            result = ctypes.windll.user32.PrintWindow(
                wintypes.HWND(hwnd),
                wintypes.HDC(mem_dc.GetSafeHdc()),
                wintypes.UINT(PW_RENDERFULLCONTENT),
            )
            captured_at = time.monotonic()
            if result != 1:
                raise CaptureFailedError(
                    f"PrintWindow 失敗 (回傳 {result}): {window}\n"
                    "  若目標是硬體加速視窗(Chrome / Edge / Electron)而畫面全黑,\n"
                    "  請改用 dxcam 或 Windows Graphics Capture,見本模組 docstring。"
                )

            raw = bitmap.GetBitmapBits(True)
            # GetBitmapBits 給的是 BGRA,前三通道即 OpenCV 的 BGR。
            # 用 cv2.cvtColor 而非 numpy 切片,原因見 capture/macos.py 的同段註解。
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            array = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        finally:
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())
            if mem_dc is not None:
                mem_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            if window_dc is not None:
                win32gui.ReleaseDC(hwnd, window_dc)

        # 已宣告 per-monitor DPI aware,GetWindowRect 回傳的就是實際像素,
        # 因此 scale 恆為 1.0。留著這個計算是為了讓上層邏輯與 macOS 一致。
        scale = array.shape[1] / bounds.width if bounds.width else 1.0
        current = WindowInfo(
            handle=window.handle,
            title=window.title,
            owner=window.owner,
            bounds=bounds,
            pid=window.pid,
        )
        return Frame(image=array, window=current, scale=scale, captured_at=captured_at)
