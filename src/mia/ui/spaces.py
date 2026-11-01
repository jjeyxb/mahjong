"""macOS 的桌面(Space):讓 HUD 蓋得到全螢幕遊戲。

Mac 的全螢幕 App 自己佔一個桌面,而 Qt 的 Tool 視窗只待在開它的那個桌面。所以
遊戲一全螢幕,Overlay 明明開著卻看不到,也不會有任何錯誤訊息。

實測(2026-11-01,Steam 全螢幕):Overlay 只屬於桌面 1,遊戲在桌面 506。設了
這裡的旗標之後同時屬於兩邊,切到遊戲截圖確認畫在牌桌上面。

Qt 沒有對應的 API,只能直接改底下的 NSWindow。
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

__all__ = ["follow_into_fullscreen"]


def follow_into_fullscreen(widget: QWidget) -> None:
    """讓 Tool 視窗出現在每個桌面,包含別的 App 的全螢幕桌面。

    視窗必須已經 show 過:NSWindow 在那之後才存在,沒有的話這裡什麼都不做。

    NonactivatingPanel 不能省。少了它,點一下 HUD 就會把 MIA 帶到前景,而 MIA 的
    主視窗在另一個桌面,畫面會被拉離遊戲。
    """
    # 測試跑 offscreen 時 winId 不是 NSView,拿去包成物件會當場崩潰。
    if sys.platform != "darwin" or QGuiApplication.platformName() != "cocoa":
        return

    import AppKit
    import objc

    window = objc.objc_object(c_void_p=int(widget.winId())).window()
    if window is None:
        return
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    window.setStyleMask_(window.styleMask() | AppKit.NSWindowStyleMaskNonactivatingPanel)
