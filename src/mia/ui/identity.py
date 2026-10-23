"""程式在作業系統眼中叫什麼名字。

Dock 上那個綠色「中」的圖示滑過去會跳出 **Python** —— 因為 macOS 的
Dock 名稱不是從 Qt 來的,是從**應用程式包**的 ``CFBundleName`` 來的。
我們沒有包,是用 ``python tools/ui.py`` 直接跑的,所以它顯示解譯器的名字。

``QApplication.setApplicationName()`` 改不到這個。那個名字 Qt 自己會用
(設定檔路徑、某些對話框標題),但 macOS 的 Dock、選單列、強制結束視窗
一律看 bundle。

所以這裡做兩件事:Qt 那一份照常設,macOS 另外用 pyobjc 把主 bundle 的
資訊字典**就地改掉**。那個字典是可變的 —— 沒有包的時候 Python 的
``NSBundle.mainBundle()`` 指向解譯器自己的包,改它等於改我們這個行程的
顯示名稱,不會動到磁碟上的任何東西。

⚠ **要在 QApplication 建立之前呼叫。** Qt 在初始化 Cocoa 的時候就把名字
讀走了,之後再改,Dock 上已經是舊的了。

這不是正式的解法 —— 正式的解法是用 ``py2app`` 之類的打包成 ``.app``,
那時候 ``CFBundleName`` 由 plist 決定,這個模組就不需要了。但在那之前,
開發與自用的情境下每天都看得到那個 Python,而修掉它只要這幾行。
"""

from __future__ import annotations

import sys

from mia.utils.logging import logger

__all__ = ["APP_NAME", "apply_identity"]

#: 程式名稱。Dock、選單列、工作列、Qt 的設定檔路徑都用這個。
APP_NAME = "MIA"


def apply_identity(app: object | None = None) -> None:
    """把程式名稱告訴 Qt 與作業系統。

    Args:
        app: :class:`QApplication`。``None`` 表示只做 macOS 那一半 ——
            在建立 ``QApplication`` **之前**呼叫時就是這種用法,而那正是
            macOS 這一半唯一有效的時機。

    失敗一律只記一行 debug:名字是裝飾,不是功能。pyobjc 沒裝、換了作業系統、
    Apple 哪天把這個字典改成唯讀 —— 沒有一個該讓程式起不來。
    """
    if app is not None:
        # Qt 這一份在 Linux/Windows 上就是工作列與視窗管理員看的名字。
        app.setApplicationName(APP_NAME)  # type: ignore[attr-defined]
        app.setApplicationDisplayName(APP_NAME)  # type: ignore[attr-defined]
        # Wayland 下視窗要靠這個對上 .desktop 檔才拿得到圖示。
        app.setDesktopFileName(APP_NAME)  # type: ignore[attr-defined]

    if sys.platform == "darwin":
        _rename_bundle()


def _rename_bundle() -> None:
    """改掉 macOS 主 bundle 的 ``CFBundleName``。見模組說明。"""
    try:
        from Foundation import NSBundle
    except ImportError:  # pragma: no cover - 只有 macOS 裝了 pyobjc 才有
        logger.debug("沒有 pyobjc,Dock 上會顯示 Python")
        return

    bundle = NSBundle.mainBundle()
    if bundle is None:
        return
    info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
    if info is None:
        return
    try:
        info["CFBundleName"] = APP_NAME
    except (TypeError, ValueError) as exc:  # pragma: no cover - 字典理論上可寫
        logger.debug("改不了 CFBundleName:{}", exc)
