"""浮動視窗(Overlay、事件流 Log)右上角那顆鎖:按一下鎖定 / 解鎖。

為什麼是**另一個視窗**,而不是畫在 Overlay 上的一顆按鈕
--------------------------------------------------------
鎖定 = 整個 Overlay 對滑鼠透明(``WindowTransparentForInput``,見
:meth:`~mia.ui.overlay.hud.HudWindow._apply_lock`)。那是**整個視窗**的屬性,
不能只挖掉一角 —— 畫在 Overlay 裡面的鎖,一鎖上就點不到,也就永遠解不開了。

所以鎖是一個獨立的小浮動視窗,**永遠收得到點擊**,貼在 Overlay 右上角外側,
跟著它移動、顯示、隱藏。貼在外側而不是疊在角落裡:兩個浮動視窗疊在一起時誰在
上面取決於誰最後被 show,而鎖定 / 解鎖本身就會讓 Overlay 重新 show 一次 ——
疊著放,解鎖之後那一角的點擊可能被 Overlay 吃掉,變成拖曳。

樣子
----
* **開著的鎖、綠色**:沒鎖定,可以拖。綠色與 Overlay 那圈虛線同一個意思。
* **鎖上的鎖、灰白**:鎖定中,點擊穿到遊戲。
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QEnterEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QShowEvent,
)
from PySide6.QtWidgets import QWidget

from mia.ui.spaces import follow_into_fullscreen
from mia.ui.style import ACCENT

__all__ = ["GAP", "SIZE", "LockButton"]

#: 鎖的邊長(px)。比 Overlay 的一行字高一點,手指(觸控板)點得準。
SIZE = 26

#: 與 Overlay 右緣的距離。
GAP = 4

_BG = QColor(24, 26, 32, 214)
_BG_HOVER = QColor(44, 47, 56, 230)
_LOCKED = QColor(235, 235, 240)

_TIP_LOCKED = "已鎖定:點擊會穿到遊戲,這個視窗拖不動。\n按這裡解鎖"
_TIP_UNLOCKED = "沒鎖定:可以直接拖曳這個視窗。\n擺好之後按這裡鎖定"


class LockButton(QWidget):
    """一顆浮動的鎖。

    Args:
        on_toggle: 被按下時呼叫,參數是**按下之後**該不該鎖。真正改狀態的是
            呼叫端(:meth:`~mia.ui.overlay.window.OverlayWindow.set_locked`),
            改完再回頭 :meth:`set_locked` 把圖畫對 —— 鎖不自己記狀態,
            免得跟 Overlay 各記一份、日後對不上。
        parent: 擁有者(Overlay)。仍然是獨立的 top-level 視窗,只是跟著
            擁有者一起被回收。
    """

    def __init__(self, on_toggle: Callable[[bool], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 與 HudWindow 同一組 flag,理由見那裡。**唯獨不會加 WindowTransparentForInput**
        # —— 這顆鎖存在的意義就是在 Overlay 穿透時仍然點得到。
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(SIZE, SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_toggle = on_toggle
        self._locked = False
        self._hovered = False
        self.setToolTip(_TIP_UNLOCKED)

    @property
    def locked(self) -> bool:
        return self._locked

    def set_locked(self, on: bool) -> None:
        """照實情重畫。不會呼叫 ``on_toggle``。"""
        self._locked = on
        self.setToolTip(_TIP_LOCKED if on else _TIP_UNLOCKED)
        self.update()

    def follow(self, owner: QWidget) -> None:
        """貼到 ``owner`` 右上角外側;右邊放不下就改貼左上角外側。

        放不下是常態而不是例外:Log 視窗預設就擺在螢幕右上角,離邊緣只有
        24px,鎖貼右邊會有一截跑到螢幕外 —— 看得到一半的按鈕最讓人困惑。
        """
        frame = owner.frameGeometry()
        right = frame.topRight() + QPoint(GAP + 1, 0)
        screen = owner.screen()
        if screen is not None and right.x() + SIZE > screen.availableGeometry().right() + 1:
            self.move(frame.topLeft() - QPoint(GAP + SIZE, 0))
            return
        self.move(right)

    # ------------------------------------------------------------------ 事件

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        follow_into_fullscreen(self)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """放開時才算,不是按下就算 —— 按下去之後滑出去可以反悔。"""
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self._on_toggle(not self._locked)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)  # type: ignore[arg-type]

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        back = QPainterPath()
        back.addRoundedRect(QRectF(0.5, 0.5, SIZE - 1, SIZE - 1), 7, 7)
        painter.fillPath(back, _BG_HOVER if self._hovered else _BG)

        color = _LOCKED if self._locked else QColor(ACCENT)
        # 鎖身:下半部的圓角方塊。
        body = QRectF(SIZE * 0.27, SIZE * 0.46, SIZE * 0.46, SIZE * 0.34)
        body_path = QPainterPath()
        body_path.addRoundedRect(body, 2, 2)
        painter.fillPath(body_path, color)

        # 鎖環:一段 ∩。解鎖時往上抬、右腳離開鎖身 —— 那是所有人都認得的「開著」。
        pen = QPen(color, 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        width = SIZE * 0.30
        left = (SIZE - width) / 2
        lift = SIZE * 0.10 if not self._locked else 0.0
        top = SIZE * 0.20 - lift
        arc = QRectF(left, top, width, width)
        path = QPainterPath()
        bottom = body.top() + 0.5
        path.moveTo(left, bottom)
        path.lineTo(left, top + width / 2)
        path.arcTo(arc, 180, -180)
        right_end = bottom if self._locked else top + width / 2 + SIZE * 0.06
        path.lineTo(left + width, right_end)
        painter.drawPath(path)
