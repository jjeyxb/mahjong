"""程式在作業系統眼中的名字。

Dock 上的名字來自 macOS 的 bundle,不是 Qt —— 所以這裡測的是「兩邊都有設到」,
而不是只測 Qt 那一半。只測 Qt 的話,測試全綠而 Dock 上還是寫著 Python。
"""

from __future__ import annotations

import sys

import pytest

from mia.ui.identity import APP_NAME, apply_identity


class _FakeApp:
    """只記下被呼叫了什麼。真的 QApplication 在單元測試裡建不起來。"""

    def __init__(self) -> None:
        self.calls: dict[str, str] = {}

    def setApplicationName(self, name: str) -> None:
        self.calls["name"] = name

    def setApplicationDisplayName(self, name: str) -> None:
        self.calls["display"] = name

    def setDesktopFileName(self, name: str) -> None:
        self.calls["desktop"] = name


class TestApplyIdentity:
    def test_the_qt_side_gets_all_three_names(self) -> None:
        app = _FakeApp()
        apply_identity(app)
        assert app.calls == {"name": APP_NAME, "display": APP_NAME, "desktop": APP_NAME}

    def test_it_works_without_an_app(self) -> None:
        """建 QApplication **之前**要先跑一次,那時候還沒有 app 可以傳。"""
        apply_identity()

    def test_a_missing_pyobjc_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """名字是裝飾,不是功能 —— 它失敗不該讓程式起不來。"""
        monkeypatch.setitem(sys.modules, "Foundation", None)
        apply_identity(_FakeApp())

    @pytest.mark.skipif(sys.platform != "darwin", reason="只有 macOS 有 bundle")
    def test_the_macos_bundle_is_renamed(self) -> None:
        """這才是 Dock 真正看的那個欄位。少了它,圖示旁邊寫的是 Python。"""
        pytest.importorskip("Foundation")
        from Foundation import NSBundle

        apply_identity()
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        assert info["CFBundleName"] == APP_NAME
