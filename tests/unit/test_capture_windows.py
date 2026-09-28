"""Windows 擷取後端 —— 真的只在 Windows 上跑的測試。

先前 ``pyproject.toml`` 宣告了 ``platform_windows`` marker,但**沒有任何一條
測試真的用它**,所以「pytest 全過」從來不代表擷取層在 Windows 上能用。
這個檔案是那個缺口的第一批填補,釘住的都是 2026-09-18 首次真機驗證
(RX 9070、主螢幕 2560×1440 @125%、副螢幕 1920×1080 @100%)抓到的東西。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator

import pytest

from mia.capture.base import CaptureFailedError, WindowInfo
from mia.utils.geometry import Rect

pytestmark = [
    pytest.mark.platform_windows,
    pytest.mark.skipif(sys.platform != "win32", reason="只在 Windows 上有意義"),
]


@pytest.fixture
def backend():
    from mia.capture.windows import WindowsCaptureBackend

    if not WindowsCaptureBackend.is_available():
        pytest.skip("pywin32 未安裝")
    return WindowsCaptureBackend()


def _window(handle: int = 1) -> WindowInfo:
    return WindowInfo(
        handle=handle, title="t", owner="o", bounds=Rect(0, 0, 100, 100), pid=1
    )


@pytest.fixture
def real_window() -> Iterator[WindowInfo]:
    """開一個真的 Win32 視窗給測試用。

    用子行程而不是同行程開:Tk 的視窗必須在建立它的執行緒上跑訊息迴圈,
    塞在 pytest 的主執行緒裡會把後面的測試一起卡住。
    """
    from mia.capture.windows import enable_dpi_awareness, list_windows

    enable_dpi_awareness()
    title = f"mia-test-{uuid.uuid4().hex[:8]}"
    source = textwrap.dedent(f"""
        import ctypes, tkinter as tk
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        root = tk.Tk()
        root.title({title!r})
        root.geometry("400x300+80+80")
        root.after(30_000, root.destroy)   # 保險:測試掛掉也不會留下孤兒視窗
        root.mainloop()
    """)
    process = subprocess.Popen([sys.executable, "-c", source])
    try:
        deadline = time.monotonic() + 15.0
        found = None
        while found is None and time.monotonic() < deadline:
            found = next((w for w in list_windows() if w.title == title), None)
            if found is None:
                time.sleep(0.2)
        if found is None:
            pytest.skip("開不出測試視窗(無桌面 session?)")
        yield found
    finally:
        process.terminate()
        process.wait(timeout=10)


class TestMinimized:
    """視窗被最小化。

    **這個情境原本會安靜地出錯**,是 2026-09-28 跑 ``tools/occlusion_probe.py``
    時順手挖出來的。原本的程式碼以為「最小化」會被 ``width <= 0`` 那道檢查擋下,
    實際上不會:DWM 對最小化的視窗回報的是一個**正數**的小矩形,PrintWindow
    也回傳成功,於是呼叫端拿到「擷取成功」加一張 183x26 的凍結縮圖。

    凍結的畫面是最糟的失敗形式 —— 不報錯、看起來合法,辨識層會照著過期的
    手牌一路算下去。
    """

    def test_dwm_reports_a_positive_size_for_a_minimized_window(self, real_window) -> None:
        """釘住那個反直覺的作業系統行為本身。

        這條測試在講「為什麼不能只靠尺寸檢查」。哪天 Windows 改成回報 0,
        這條會紅 —— 那時候就知道 ``IsIconic`` 那道檢查可以簡化了。
        """
        import win32con
        import win32gui

        from mia.capture.windows import _extended_frame_bounds

        win32gui.ShowWindow(real_window.handle, win32con.SW_MINIMIZE)
        time.sleep(1.0)

        bounds = _extended_frame_bounds(real_window.handle)
        assert win32gui.IsIconic(real_window.handle)
        assert win32gui.IsWindowVisible(real_window.handle), "最小化的視窗仍算「可見」"
        assert bounds.width > 0 and bounds.height > 0, "尺寸檢查攔不住最小化"

    def test_capturing_a_minimized_window_fails_cleanly(self, backend, real_window) -> None:
        import win32con
        import win32gui

        win32gui.ShowWindow(real_window.handle, win32con.SW_MINIMIZE)
        time.sleep(1.0)

        with pytest.raises(CaptureFailedError, match="最小化"):
            backend.capture(real_window)

    def test_capturing_works_again_after_restore(self, backend, real_window) -> None:
        """失敗必須是暫時的 —— VisionWorker 靠重試接回來。"""
        import win32con
        import win32gui

        win32gui.ShowWindow(real_window.handle, win32con.SW_MINIMIZE)
        time.sleep(1.0)
        win32gui.ShowWindow(real_window.handle, win32con.SW_RESTORE)
        time.sleep(1.0)

        frame = backend.capture(real_window)
        assert frame.image.shape[0] > 100
        assert frame.image.shape[1] > 100


class TestDpiScale:
    """``Frame.scale`` 的來源。

    這個值必須等於瀏覽器的 ``devicePixelRatio`` —— 畫布尺寸是 CSS 像素,
    ``Canvas.table_rect`` 拿它換算成影像上的像素。給錯的話畫面辨識整個鎖不上
    (實測 125% 螢幕上會差 404px,狀態列一路刷「畫布對不上」)。
    """

    def test_a_real_window_has_a_plausible_scale(self) -> None:
        from mia.capture.windows import _dpi_scale, enable_dpi_awareness, list_windows

        enable_dpi_awareness()
        windows = [w for w in list_windows() if w.bounds.width > 200]
        if not windows:
            pytest.skip("桌面上沒有夠大的視窗可以問")

        scale = _dpi_scale(windows[0].handle)
        # Windows 的縮放選項是 100% 起跳、25% 一格;放寬到 4.0 是給未來的高解析螢幕
        assert 1.0 <= scale <= 4.0, f"縮放 {scale} 不像真的"
        assert scale * 4 == pytest.approx(round(scale * 4)), f"{scale} 不是 25% 的整數倍"

    def test_an_invalid_handle_still_returns_something_usable(self) -> None:
        """無效的 handle 不可以讓這裡拋例外。

        它被呼叫的時機是**每一幀**,而視窗隨時可能剛好在這一瞬間消失。
        拋出去的話整條擷取執行緒會被收掉(見 :class:`TestGdiFailures`)。

        實測 handle 0 會走到 ``MonitorFromWindow`` 的 ``DEFAULTTONEAREST``
        退路、拿到主螢幕的縮放 —— 那比硬給 1.0 更接近真相,所以這裡只要求
        「是個合理的值」,不釘死成 1.0。
        """
        from mia.capture.windows import _dpi_scale

        assert 1.0 <= _dpi_scale(0) <= 4.0


class TestGdiFailures:
    """視窗在擷取途中被關掉。

    **這是正常操作,不是異常** —— ``docs/live.md`` 明講「要結束就去關瀏覽器」。
    實測(2026-09-18)關掉瀏覽器的那一刻,``finally`` 裡的 ``DeleteDC`` 會拋
    ``win32ui.error`` 竄出去,把整條擷取執行緒打死,畫面上只留一句
    「擷取執行緒異常結束」。

    ``VisionWorker._tick`` 只接 ``CaptureFailedError`` —— 它接到之後走的是
    既有的「失敗幾次就重新找視窗」,而那正是「瀏覽器關掉又重開」該有的行為。
    所以擷取層的義務是:**GDI 壞掉要轉成 CaptureFailedError**。
    """

    def test_a_dead_window_becomes_a_capture_failure(self, backend, monkeypatch) -> None:
        import win32ui

        from mia.capture import windows as mod

        class DyingDC:
            def CreateCompatibleDC(self):
                raise win32ui.error("CreateCompatibleDC failed")

            def DeleteDC(self):  # 清理也失敗 —— 視窗沒了,handle 全失效
                raise win32ui.error("DeleteDC failed")

        monkeypatch.setattr(mod.win32gui, "IsWindow", lambda _h: True)
        monkeypatch.setattr(mod, "_extended_frame_bounds", lambda _h: Rect(0, 0, 100, 100))
        monkeypatch.setattr(mod.win32gui, "GetWindowDC", lambda _h: 1234)
        monkeypatch.setattr(mod.win32gui, "ReleaseDC", lambda _h, _d: None)
        monkeypatch.setattr(mod.win32ui, "CreateDCFromHandle", lambda _h: DyingDC())

        # 重點有二:型別是 CaptureFailedError(不是 win32ui.error),
        # 而且清理階段那個 DeleteDC failed **沒有**把它蓋掉。
        with pytest.raises(CaptureFailedError, match="GDI"):
            backend.capture(_window())

    def test_a_handle_that_is_no_longer_a_window_fails_cleanly(self, backend) -> None:
        with pytest.raises(CaptureFailedError):
            backend.capture(_window(handle=0xDEAD_BEEF))
