"""Overlay 下緣外側那一條:混合辨識兩邊讀成不同的牌,讓玩家看一眼牌河點對的那張。

為什麼要問
----------
樣板與 YOLO 讀別家捨牌偶爾讀成不同的牌,合併層有規則挑一邊,但規則會挑錯(2026-11-09 三場
CPU 場實打:真正的分歧一場約 1 ~ 2 張,挑錯 2 張)。那張牌就在畫面上的牌河裡,玩家看一眼就知道。
只問兩邊都穩定了之後還不一樣的(:data:`~mia.hybrid.merge.SETTLE`),一邊只是慢一點的不問。

不擋操作
--------
雀魂出牌有時限,所以不跳視窗、不搶焦點:不點也照常給建議(用合併層挑的那張),點了之後
事件流改寫、建議重算。

為什麼是**另一個視窗**
----------------------
跟右上角的鎖同一個理由(:mod:`mia.ui.overlay.lock`):Overlay 鎖定時整個視窗對滑鼠透明,
畫在裡面的按鈕點不到。這一條永遠收得到點擊,貼在 Overlay 下緣外側,跟著它移動、顯示、隱藏。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from mia.hybrid.merge import TileChoice
from mia.ui.spaces import follow_into_fullscreen
from mia.ui.style import ACCENT
from mia.ui.widgets.danger import seat_name
from mia.ui.widgets.tiles import TileIcons, TileLabel

__all__ = ["GAP", "MAX_ROWS", "ChoiceBar"]

#: 與 Overlay 下緣的距離。
GAP = 4

#: 最多同時問幾張。一場真正的分歧約 1 ~ 2 張,同時更多張多半是某一邊整段錯位,問了也選不完。
MAX_ROWS = 2

_TILE_HEIGHT = 34
_BG = QColor(24, 26, 32, 222)
_TEXT = "#F2F2F7"

Pick = Callable[[TileChoice, str], None]


class _TileButton(TileLabel):
    """點得到的牌面。放開時才算,跟鎖一樣:按下去之後滑出去可以反悔。"""

    def __init__(self, icons: TileIcons, on_click: Callable[[], None], parent: QWidget) -> None:
        super().__init__(icons, _TILE_HEIGHT, parent)
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 牌面是固定尺寸,外框(2px)加內距(1px)要另外留,不然牌被框吃掉一圈。
        self.setFixedSize(self.width() + 6, self.height() + 6)
        self.mark(False)

    def mark(self, chosen: bool) -> None:
        """現在事件流用的那張加框。"""
        color = ACCENT if chosen else "transparent"
        self.setStyleSheet(f"border: 2px solid {color}; border-radius: 4px; padding: 1px;")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self._on_click()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _Row(QWidget):
    """一張:「對家第 11 張是?」+ 兩張牌。"""

    def __init__(self, icons: TileIcons, on_pick: Pick, parent: QWidget) -> None:
        super().__init__(parent)
        self._choice: TileChoice | None = None
        self._on_pick = on_pick
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._question = QLabel("", self)
        self._question.setStyleSheet(f"color: {_TEXT}; font-size: 13px;")
        layout.addWidget(self._question)
        self._tiles = [
            _TileButton(icons, lambda: self._pick(self._tiles[0]), self),
            _TileButton(icons, lambda: self._pick(self._tiles[1]), self),
        ]
        for tile in self._tiles:
            layout.addWidget(tile)
        layout.addStretch(1)

    def show_choice(self, choice: TileChoice) -> None:
        self._choice = choice
        self._question.setText(f"{seat_name(0, choice.actor)}第 {choice.nth + 1} 張是?")
        for tile, pai in zip(self._tiles, (choice.template, choice.yolo), strict=True):
            tile.set_tile(pai)
            tile.mark(pai == choice.chosen)

    def _pick(self, tile: TileLabel) -> None:
        if self._choice is not None and tile.tile is not None:
            self._on_pick(self._choice, tile.tile)


class ChoiceBar(QWidget):
    """一條浮動的待選清單。沒有要問的時候藏起來。

    Args:
        icons: 牌面圖,與 Overlay 同一套。
        on_pick: 玩家點了一張:``on_pick(那一題, 選的牌)``。
        parent: 擁有者(Overlay)。仍然是獨立的 top-level 視窗,只是跟著擁有者一起被回收。
    """

    def __init__(self, icons: TileIcons, on_pick: Pick, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 與鎖同一組 flag,同樣**不加 WindowTransparentForInput**:Overlay 鎖定時也要點得到。
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)
        self._rows = [_Row(icons, on_pick, self) for _ in range(MAX_ROWS)]
        for row in self._rows:
            layout.addWidget(row)
        self._choices: tuple[TileChoice, ...] = ()

    @property
    def choices(self) -> tuple[TileChoice, ...]:
        """現在列出來的那幾題。給測試用。"""
        return self._choices

    def show_choices(self, choices: Sequence[TileChoice], owner: QWidget) -> None:
        """換成這幾題。``owner`` 看不見或沒有要問的就藏起來。

        Overlay 一秒呼叫這裡十次,題目沒變就不重畫牌面。
        """
        fresh = tuple(choices[:MAX_ROWS])
        if fresh != self._choices:
            self._choices = fresh
            for row, choice in zip(self._rows, fresh, strict=False):
                row.show_choice(choice)
                row.show()
            for row in self._rows[len(fresh) :]:
                row.hide()
            self.adjustSize()
        if not self._choices or not owner.isVisible():
            self.hide()
            return
        self.follow(owner)
        if self.isHidden():
            self.show()
            self.raise_()

    def follow(self, owner: QWidget) -> None:
        """貼到 ``owner`` 左下角外側;下面放不下就改貼左上角外側。"""
        frame = owner.frameGeometry()
        below = frame.bottomLeft() + QPoint(0, GAP + 1)
        screen = owner.screen()
        if screen is not None and below.y() + self.height() > screen.availableGeometry().bottom():
            self.move(frame.topLeft() - QPoint(0, GAP + self.height()))
            return
        self.move(below)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        follow_into_fullscreen(self)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        back = QPainterPath()
        back.addRoundedRect(self.rect().toRectF().adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        painter.fillPath(back, _BG)
