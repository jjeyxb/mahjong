"""Windows 擷取後端 —— 真的只在 Windows 上跑的測試。

先前 ``pyproject.toml`` 宣告了 ``platform_windows`` marker,但**沒有任何一條
測試真的用它**,所以「pytest 全過」從來不代表擷取層在 Windows 上能用。
這個檔案是那個缺口的第一批填補,釘住的都是 2026-09-18 首次真機驗證
(RX 9070、主螢幕 2560×1440 @125%、副螢幕 1920×1080 @100%)抓到的東西。
"""

from __future__ import annotations

import sys

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
