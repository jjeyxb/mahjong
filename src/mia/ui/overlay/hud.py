"""浮在遊戲上的視窗共用的部分:無邊框、不搶焦點、拖曳、鎖定、圓角深色底。

鎖定與拖曳是同一件事的兩面,見 :mod:`mia.ui.overlay.window`。這裡只管怎麼浮著,位置
與鎖定存在哪由子類別決定:每個 HUD 各記各的,互不干擾。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen, QShowEvent
from PySide6.QtWidgets import QApplication, QWidget

from mia.ui.spaces import follow_into_fullscreen
from mia.ui.style import ACCENT

__all__ = ["HudWindow"]

_BG = QColor(24, 26, 32, 214)
_RADIUS = 12


class HudWindow(QWidget):
    """HUD 的基底。子類別實作 :attr:`locked` 與位置的存取。"""

    def __init__(self) -> None:
        super().__init__()
        self._drag_offset: QPoint | None = None

        # window flag 必須在第一次 show() **之前**設定。show 之後再改,Qt 會把
        # 視窗隱藏並要求重新 show() —— 症狀是「程式在跑但畫面上沒有東西」。
        #
        # Qt.Tool:不進 Dock、不搶焦點。但 macOS 上 tool window 會在應用程式
        # 失去焦點時自動隱藏 —— 而 Overlay 的正常使用情境正是「焦點在遊戲上」。
        # WA_MacAlwaysShowToolWindow 就是為了關掉那個行為。
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    @property
    def locked(self) -> bool:
        raise NotImplementedError

    def _saved_position(self) -> tuple[int, int] | None:
        raise NotImplementedError

    def _remember_position(self, point: tuple[int, int]) -> None:
        raise NotImplementedError

    def _move_to_default(self) -> None:
        raise NotImplementedError

    def _apply_lock(self, locked: bool) -> None:
        """套用滑鼠穿透。

        改 window flag 會讓已顯示的視窗被 Qt 隱藏,所以要重新 show 一次
        —— 這正是先前「Dock 有圖示但畫面上沒有視窗」那個 bug 的成因。

        **只動 window flag,不要碰 WA_TransparentForMouseEvents。** 那個 attribute
        會讓 Qt 自己把 ``WindowTransparentForInput`` 加進 window flag,而把
        attribute 設回 False **不會**把那個 flag 收回去 —— 結果是鎖過一次之後
        就再也解不開,而且解不開的症狀是「拖不動」,完全不像是 flag 的事。
        真正決定點擊會不會穿到底下那個視窗的本來就是 window flag。
        """
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, locked)
        if was_visible:
            self.show()

    def _restore_position(self) -> None:
        """回到上次拖到的地方。

        存下來的座標可能已經不在任何一個螢幕上(外接螢幕拔掉了、解析度改了)
        —— 那樣視窗會開在看不見的地方,而使用者只會覺得勾了沒反應。
        所以先確認那個點還落在某個螢幕上,不然回預設位置。
        """
        saved = self._saved_position()
        if saved is not None:
            point = QPoint(*saved)
            if QApplication.screenAt(point) is not None:
                self.move(point)
                return
        self._move_to_default()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # 每次 show 都要設:鎖定會改 window flag,Qt 可能重建底下的 NSWindow。
        follow_into_fullscreen(self)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.locked:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """拖完才存,不是拖的過程中每個 pixel 都存一次。"""
        if self._drag_offset is not None:
            self._drag_offset = None
            point = self.frameGeometry().topLeft()
            self._remember_position((point.x(), point.y()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        """圓角深色底。未鎖定時多一圈虛線 —— 那是「現在拖得動」的唯一提示。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        rect = self.rect().adjusted(1, 1, -1, -1)
        path.addRoundedRect(
            float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()),
            _RADIUS, _RADIUS,
        )
        painter.fillPath(path, _BG)
        if not self.locked:
            painter.setPen(QPen(QColor(ACCENT), 1.0, Qt.PenStyle.DashLine))
            painter.drawPath(path)
        super().paintEvent(event)
