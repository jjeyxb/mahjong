"""Windows 擷取後端 —— PrintWindow + PW_RENDERFULLCONTENT。

**2026-09-18 首次在真實 Windows 機器上驗證**(RX 9070、主螢幕 2560×1440 @125%、
副螢幕 1920×1080 @100%)。原本列的三個風險,現況:

1. ``PW_RENDERFULLCONTENT`` 對硬體加速視窗是否抓到全黑 —— **不會**。拿一個
   正在播影片的 Chrome 視窗實測,抓到的是正常畫面(mean=111、std=58)。
   下方「備案」那兩條因此都不需要走。
2. DPI 縮放下邊界與影像尺寸對不對得上 —— **原本對不上,已修**。成因不是
   ``DwmGetWindowAttribute`` 不準(它很準),而是**兩種「邏輯像素」被混為
   一談**:視窗座標(宣告 DPI aware 之後等同實體像素)與瀏覽器的 CSS 像素。
   詳見 :meth:`WindowsCaptureBackend.capture` 裡的長註解。
3. 遊戲視窗被遮擋時能不能抓到完整內容 —— **仍未驗證**。自動化 session 搶不到
   前景視窗(被 Windows 的 foreground lock 擋下),排不出真正的遮擋場景。
   要驗需要真人坐在機器前手動切換視窗。

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

from mia.capture.base import (
    CaptureBackend,
    CaptureFailedError,
    Frame,
    WindowInfo,
)
from mia.utils.geometry import Rect
from mia.utils.logging import logger

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

# GetDpiForMonitor / MonitorFromWindow
USER_DEFAULT_SCREEN_DPI = 96
MONITOR_DEFAULTTONEAREST = 2
MDT_EFFECTIVE_DPI = 0

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


def _dpi_scale(hwnd: int) -> float:
    """視窗所在螢幕的 DPI 縮放:1.0 / 1.25 / 1.5 / 2.0 …

    **這就是 `Frame.scale` 要的那個值**,理由見 :meth:`WindowsCaptureBackend.capture`
    裡的說明。

    取的是**視窗所在的那一個螢幕**而不是主螢幕 —— 多螢幕各自縮放不同是很常見的
    設定(這台開發機就是主螢幕 125%、副螢幕 100%),拿主螢幕的值套到另一個螢幕上
    的視窗會整組算錯。

    Note:
        這個值等同於瀏覽器的 ``window.devicePixelRatio``,**前提是瀏覽器縮放為
        100%**。MIA 的瀏覽器是自己開的(見 :mod:`mia.groundtruth.cdp`),不會去
        動縮放,所以這個前提成立。實測 125% 螢幕上兩者都是 1.25,分毫不差。
        使用者手動按 Ctrl+ 放大頁面的話這個假設會破 —— 那時候畫布校正會說
        「對不上」,而不是安靜地給一個偏掉的矩形。
    """
    dpi = 0
    # GetDpiForWindow 是 Windows 10 1607+ 才有的,舊版沒有這個符號
    get_dpi_for_window = getattr(ctypes.windll.user32, "GetDpiForWindow", None)
    if get_dpi_for_window is not None:
        dpi = int(get_dpi_for_window(wintypes.HWND(hwnd)))

    if not dpi:  # 退回:問視窗所在的螢幕
        monitor = ctypes.windll.user32.MonitorFromWindow(
            wintypes.HWND(hwnd), wintypes.DWORD(MONITOR_DEFAULTTONEAREST)
        )
        dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
        hresult = ctypes.windll.shcore.GetDpiForMonitor(
            monitor,
            ctypes.c_int(MDT_EFFECTIVE_DPI),
            ctypes.byref(dpi_x),
            ctypes.byref(dpi_y),
        )
        if hresult == 0:
            dpi = dpi_x.value

    if not dpi:
        logger.warning("問不到視窗 {} 的 DPI,當成 100% 縮放", hwnd)
        return 1.0
    return dpi / USER_DEFAULT_SCREEN_DPI


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

        physical = _extended_frame_bounds(hwnd)
        if not include_all and (physical.width <= 0 or physical.height <= 0):
            return True

        # 與 capture() 一致:對外一律報**邏輯座標**。不一致的話,
        # 「列出來的視窗」與「擷取回來的那一幀」會是兩種單位,而 base.py 的
        # min_width / min_height 過濾又是吃這裡的值 —— 混用了不會報錯,
        # 只會在高 DPI 螢幕上默默篩掉或篩進錯的視窗。
        scale = _dpi_scale(hwnd) if physical.width > 0 else 1.0
        bounds = Rect(
            round(physical.x / scale),
            round(physical.y / scale),
            round(physical.width / scale),
            round(physical.height / scale),
        )

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
        # 這裡拿到的是**實體像素**(本行程宣告了 per-monitor DPI aware),
        # PrintWindow 的點陣圖必須照這個尺寸開,不能用邏輯座標。
        physical = _extended_frame_bounds(hwnd)
        width, height = physical.width, physical.height
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

        # bounds 回報**邏輯座標**、scale 是螢幕的 DPI 縮放 —— 這才符合
        # base.py 與 geometry.py 白紙黑字寫的約定(「bounds 為邏輯座標」、
        # 「scale 即 pixel / logical 的比值」),也才與 macOS 那一份一致。
        #
        # 這裡原本寫的是「已宣告 per-monitor DPI aware,所以 scale 恆為 1.0」。
        # 那句話單看擷取層是對的 —— 以視窗座標為單位,影像確實是 1:1。但它讓
        # **兩種不同的「邏輯像素」被混為一談**:
        #
        #   視窗座標(DPI aware 之後 = 實體像素)  vs  瀏覽器的 CSS 像素
        #
        # 而 Canvas.table_rect 要換算的是後者 —— 畫布 1600×900 是 CSS 像素,
        # 在 125% 螢幕上佔 2000 實體像素。scale 給 1.0 的話它會拿 1600 去對
        # 2004 的影像寬,差 404px 當場判定「畫布對不上」,畫面辨識整個鎖不上。
        # macOS 上碰巧不會出事:Retina 的 backing scale 與 devicePixelRatio
        # 都是 2.0,兩種單位剛好同值,於是混用了也看不出來。
        #
        # 實測(125% 螢幕、畫布 1600×900):改成這樣之後畫布上方剩給瀏覽器介面
        # 的高度從荒謬的 337 變成 89.6 個邏輯像素,而 macOS 實測的 Chrome
        # 工具列是 87 —— 幾何關係一下就對上了,這是這個修法正確的旁證。
        scale = _dpi_scale(hwnd)
        logical = Rect(
            round(physical.x / scale),
            round(physical.y / scale),
            round(physical.width / scale),
            round(physical.height / scale),
        )
        current = WindowInfo(
            handle=window.handle,
            title=window.title,
            owner=window.owner,
            bounds=logical,
            pid=window.pid,
        )
        return Frame(image=array, window=current, scale=scale, captured_at=captured_at)
