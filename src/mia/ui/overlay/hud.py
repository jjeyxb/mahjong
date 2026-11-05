"""浮在遊戲上的視窗共用的部分:無邊框、不搶焦點、拖曳、鎖定、圓角深色底。

鎖定與拖曳是同一件事的兩面,見 :mod:`mia.ui.overlay.window`。這裡只管怎麼浮著,位置
與鎖定存在哪由子類別決定:每個 HUD 各記各的,互不干擾。

右上角那顆鎖(:mod:`mia.ui.overlay.lock`)也在這裡接:它是另一個視窗,要跟著
HUD 顯示、隱藏、移動、關閉,而那四件事每個 HUD 都一樣。子類別只要在狀態讀好之後
呼叫一次 :meth:`HudWindow._attach_lock`。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QHideEvent,
    QMouseEvent,
    QMoveEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import QApplication, QWidget

from mia.ui.overlay.lock import LockButton
from mia.ui.spaces import follow_into_fullscreen
from mia.ui.state import MIN_HUD_OPACITY, UiState
from mia.ui.style import ACCENT

__all__ = ["HudWindow"]

_BG = QColor(24, 26, 32, 214)
_RADIUS = 12


class HudWindow(QWidget):
    """HUD 的基底。子類別實作 :attr:`locked` 與位置的存取。

    子類別都要有 ``_ui_state``:透明度存在那裡,而且**所有 HUD 共用同一個值**
    (見 :attr:`UiState.hud_opacity`)。
    """

    _ui_state: UiState

    def __init__(self) -> None:
        super().__init__()
        self._drag_offset: QPoint | None = None
        self._lock: LockButton | None = None

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

    def set_locked(self, on: bool) -> None:
        """鎖定 / 解鎖,並記下來。右上角那顆鎖按下去就是呼叫這裡。"""
        raise NotImplementedError

    def _attach_lock(self) -> None:
        """建右上角那顆鎖。子類別在 :attr:`locked` 讀得到之後呼叫一次。

        為什麼是另一個視窗,見 :mod:`mia.ui.overlay.lock`:鎖定 = 整個視窗對
        滑鼠透明,畫在裡面的鎖一鎖上就點不到,也就解不開了。
        """
        self._lock = LockButton(self.set_locked, self)
        self._lock.set_locked(self.locked)

    @property
    def opacity(self) -> float:
        """不透明度,:data:`~mia.ui.state.MIN_HUD_OPACITY` ~ 1.0。"""
        return self._ui_state.hud_opacity

    def set_opacity(self, value: float) -> None:
        """調整不透明度並記下來。超出範圍的值夾回來。"""
        value = min(1.0, max(MIN_HUD_OPACITY, value))
        self._ui_state.hud_opacity = value
        self._ui_state.save()
        self._apply_opacity(value)

    def _apply_opacity(self, value: float) -> None:
        """整個視窗(字、牌、底色)一起變淡。

        鎖跟著變淡,但**不低於一半**:它是唯一一個鎖上之後還點得到的東西,
        淡到找不到的話,鎖上的視窗就解不開了。
        """
        self.setWindowOpacity(value)
        if self._lock is not None:
            self._lock.setWindowOpacity(max(0.5, value))

    @property
    def lock_button(self) -> LockButton | None:
        """右上角那顆鎖。給測試與截圖用。"""
        return self._lock

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
        if self._lock is not None:
            self._lock.set_locked(locked)
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
        # 鎖定 / 解鎖會改 window flag,Qt 可能重建底下的 NSWindow,透明度要再套一次。
        self._apply_opacity(self.opacity)
        if self._lock is not None:
            self._lock.follow(self)
            self._lock.show()
            # 都是置頂的浮動視窗;後 show 的在上面。
            self._lock.raise_()

    # 鎖是獨立的視窗,不會自己跟著動。四個時機都要顧:顯示(含鎖定 / 解鎖
    # 造成的重新 show)、隱藏、拖曳、大小改變(內容變多、拖右下角)。

    def hideEvent(self, event: QHideEvent) -> None:
        if self._lock is not None:
            self._lock.hide()
        super().hideEvent(event)

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        if self._lock is not None:
            self._lock.follow(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._lock is not None:
            self._lock.follow(self)

    def closeEvent(self, event: QCloseEvent) -> None:
        """一起收掉 —— 留著的話它會是「最後一個還開著的視窗」,程式結束不了。"""
        if self._lock is not None:
            self._lock.close()
        super().closeEvent(event)

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
