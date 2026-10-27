"""在一張畫面上拖一個框。手牌範圍與牌桌矩形共用這個。

座標對外一律是**影像像素**,不是預覽座標。預覽會被縮小以塞進螢幕,而縮放比例
是這個元件自己的事 —— 讓呼叫端去換算,等於讓每一個呼叫端都有機會換錯,
而換錯的症狀是「框存下去了,但位置不對」,不會有任何錯誤訊息。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import QWidget

from mia.ui.style import ACCENT

__all__ = ["DragPicker", "to_pixmap"]

#: 預覽圖最大邊長(邏輯像素)。再大就放不進一般的筆電螢幕。
PREVIEW_MAX = 820

#: 拖出來的框至少要這麼大(預覽像素),不然當成誤觸的點擊。
_MIN_DRAG = 12

#: 推導出來的形狀用的第二個顏色。與 :data:`~mia.ui.style.ACCENT` 分開,
#: 是因為畫面上常常同時有兩種:使用者框的那個、以及從它算出來的那些。
_DERIVED = "#34C759"


class DragPicker(QWidget):
    """顯示畫面並讓使用者拖一個框。

    畫三層東西,三層的意思不同:

    * 使用者拖的框 —— **虛線**。它是提示,通常不是最後真正用的那個。
    * :attr:`rects` —— 從那個框推導出來的矩形(對齊過的手牌 ROI 之類)。
    * :attr:`polys` —— 推導出來的多邊形(四家牌河攤平前的四邊形)。

    後兩者是這個設計的安全網:**框歪了要看得見**。自動校正最大的問題正是
    它錯的時候畫面上什麼都不會發生。
    """

    def __init__(self, image: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = (image.shape[1], image.shape[0])
        pixmap = to_pixmap(image)
        scaled = pixmap.size().scaled(PREVIEW_MAX, PREVIEW_MAX, Qt.AspectRatioMode.KeepAspectRatio)
        self._pixmap = pixmap.scaled(
            scaled, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setFixedSize(self._pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start: QPoint | None = None
        self._box = QRect()
        #: 推導出來的矩形(影像座標)。
        self.rects: list[QRect] = []
        #: 推導出來的多邊形(影像座標,每個是四個以上的點)。
        self.polys: list[list[tuple[int, int]]] = []
        self.on_release: object = None

    # -------------------------------------------------------------- 座標換算

    def _to_image(self, rect: QRect) -> QRect:
        """預覽座標 → 影像座標。"""
        kx = self._size[0] / max(1, self._pixmap.width())
        ky = self._size[1] / max(1, self._pixmap.height())
        return QRect(
            round(rect.x() * kx),
            round(rect.y() * ky),
            round(rect.width() * kx),
            round(rect.height() * ky),
        )

    def _to_preview(self, rect: QRect) -> QRect:
        kx = self._pixmap.width() / max(1, self._size[0])
        ky = self._pixmap.height() / max(1, self._size[1])
        return QRect(
            round(rect.x() * kx),
            round(rect.y() * ky),
            round(rect.width() * kx),
            round(rect.height() * ky),
        )

    def _point_to_preview(self, point: tuple[int, int]) -> QPoint:
        kx = self._pixmap.width() / max(1, self._size[0])
        ky = self._pixmap.height() / max(1, self._size[1])
        return QPoint(round(point[0] * kx), round(point[1] * ky))

    @property
    def image_size(self) -> tuple[int, int]:
        """底圖的 ``(寬, 高)``,像素。"""
        return self._size

    @property
    def box(self) -> QRect | None:
        """使用者拖的那個框,影像座標。還沒拖過時 ``None``。"""
        return self._to_image(self._box.normalized()) if not self._box.isNull() else None

    def clear_derived(self) -> None:
        """清掉上一次推導出來的東西。重算之前一定要先清 —— 留著的話
        使用者會看到一組對應**舊框**的輔助線,而那比什麼都不畫更誤導。
        """
        self.rects = []
        self.polys = []

    # -------------------------------------------------------------- 滑鼠

    def _inside(self, event: QMouseEvent) -> QPoint:
        """滑鼠位置,**夾在影像範圍內**。

        不夾的話拖出邊界會得到一個比影像還大的框,而那個框一路傳下去會在
        :class:`~mia.utils.geometry.NormRect` 炸掉(它拒絕超出容器的矩形)。
        實測症狀是「訊息與牌面都畫好了,但確定按不下去」—— 例外從中間竄出去,
        而最後那行啟用按鈕的程式沒跑到。

        夾在這裡而不是只夾結果:使用者看到的框要與真正送下去的一樣,
        不然他會對著一個不存在的框調整。
        """
        point = event.position().toPoint()
        return QPoint(
            max(0, min(point.x(), self.width() - 1)),
            max(0, min(point.y(), self.height() - 1)),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._start = self._inside(event)
        self._box = QRect(self._start, self._start)
        self.clear_derived()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._box = QRect(self._start, self._inside(event))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._start is None:
            return
        self._box = QRect(self._start, self._inside(event)).normalized()
        self._start = None
        if self._box.width() < _MIN_DRAG or self._box.height() < _MIN_DRAG:
            # 點一下沒拖 —— 當成誤觸,不要把一個幾像素的框送下去
            self._box = QRect()
        self.update()
        if callable(self.on_release):
            self.on_release()

    # -------------------------------------------------------------- 畫

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        if not self._box.isNull():
            # 使用者拖的框畫成虛線:它是**提示**,不是最後用的那個
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DashLine))
            painter.drawRect(self._box.normalized().adjusted(0, 0, -1, -1))
        painter.setPen(QPen(QColor(ACCENT), 2))
        for rect in self.rects:
            painter.drawRect(self._to_preview(rect).adjusted(0, 0, -1, -1))
        painter.setPen(QPen(QColor(_DERIVED), 2))
        for poly in self.polys:
            painter.drawPolygon(QPolygon([self._point_to_preview(p) for p in poly]))
        painter.end()


def to_pixmap(image: np.ndarray) -> QPixmap:
    """BGR ndarray → QPixmap。

    ``QImage`` 不複製傳進去的記憶體,而這裡的 ndarray 是臨時的 —— 不 copy()
    的話畫出來是一片雜訊或直接當掉,而且是**有時候**才發生。
    """
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    height, width = rgb.shape[:2]
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
