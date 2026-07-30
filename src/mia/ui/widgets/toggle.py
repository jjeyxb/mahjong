"""iOS 風格的開關。

Qt 內建的 ``QCheckBox`` 表達的是「這一項有沒有被勾選」,而這裡要表達的是
**「這個功能正在跑 / 沒在跑」** —— 那是一個狀態,不是一個選項。滑軌開關讓
「現在是開的」在一公尺外就看得出來,勾選框不行。

顏色用 Apple 的 system green(``#34C759``),不是自己調的綠。跟著系統色的
好處是它在淺色與深色模式下都已經被驗過對比度。

刻意不做的事
------------
**不做三態、不做載入中的轉圈。** 開關只回答「要不要跑」。開啟後真正的進度
(引擎正在載權重、校正還在蒐集)走狀態列 —— 那裡有空間寫完整的句子,
而開關上只有 44 像素。
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QAbstractButton, QWidget

__all__ = ["ToggleSwitch"]

#: 軌道尺寸。iOS 是 51x31;這裡縮到能塞進設定列的一行,比例維持不變。
TRACK_WIDTH = 44
TRACK_HEIGHT = 26

#: 滑塊與軌道邊緣的間距。
KNOB_MARGIN = 2

#: Apple 的 system green。
ON_COLOR = QColor(52, 199, 89)

#: 動畫長度(毫秒)。140 ms 是「看得出它在動」與「不覺得慢」之間;
#: 測試會傳 0 讓狀態立刻到位,免得要等事件迴圈。
DURATION_MS = 140


class ToggleSwitch(QAbstractButton):
    """可勾選的滑軌開關。

    Args:
        parent: 父 widget。
        duration: 動畫長度(毫秒)。0 表示不動畫,狀態立刻到位。

    Note:
        ``isChecked()`` 才是權威狀態。:attr:`position` 只是畫面上滑塊的位置,
        動畫途中它會落在 0 與 1 之間 —— 不要拿它判斷開關開著沒有。
    """

    def __init__(self, parent: QWidget | None = None, *, duration: int = DURATION_MS) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._position = 0.0
        self._animation = QPropertyAnimation(self, b"position", self)
        self._animation.setDuration(duration)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.toggled.connect(self._on_toggled)

    # ------------------------------------------------------------------ 幾何

    def sizeHint(self) -> QSize:
        return QSize(TRACK_WIDTH, TRACK_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # ------------------------------------------------------------------ 動畫

    def _get_position(self) -> float:
        return self._position

    def _set_position(self, value: float) -> None:
        self._position = value
        self.update()

    #: 滑塊位置,0 在左(關)、1 在右(開)。動畫用。
    position = Property(float, _get_position, _set_position)

    def _on_toggled(self, checked: bool) -> None:
        target = 1.0 if checked else 0.0
        if self._animation.duration() <= 0:
            # 沒有動畫就直接到位。事件迴圈還沒起來時(建構期設初值、測試)
            # 靠動畫是等不到的,畫面會停在錯的一邊。
            self._set_position(target)
            return
        self._animation.stop()
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(target)
        self._animation.start()

    def setChecked(self, checked: bool) -> None:
        """設定狀態。與使用者點擊等價,包含動畫。"""
        super().setChecked(checked)
        # 值沒變時 toggled 不會發,但初始化時 position 可能還沒對上
        if (self._position >= 1.0) != checked:
            self._on_toggled(checked)

    def set_initial(self, checked: bool) -> None:
        """設定**起始**狀態:滑塊立刻到位,不動畫,也不發 ``toggled``。

        兩個理由,都是實際踩到的:

        1. 開關的起始狀態不該表現成一段動畫。開視窗的瞬間看到滑塊自己滑過去,
           像是有人剛剛按了它。
        2. 建構期事件迴圈還沒起來,而 ``QPropertyAnimation`` 要靠迴圈推進 ——
           那時候啟動的動畫一格都不會走,滑塊會停在錯的一邊。表現出來是
           「開關看起來是關的,但功能其實在跑」,而那正是這個開關要避免的誤解。

        不發 ``toggled`` 是為了不讓「反映現況」被誤當成「使用者要求改變」。
        """
        self.blockSignals(True)
        super().setChecked(checked)
        self.blockSignals(False)
        self._animation.stop()
        self._set_position(1.0 if checked else 0.0)

    # ------------------------------------------------------------------ 繪製

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track = QRectF(0, 0, self.width(), self.height())
        radius = track.height() / 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._track_color())
        painter.drawRoundedRect(track, radius, radius)

        diameter = track.height() - 2 * KNOB_MARGIN
        travel = track.width() - diameter - 2 * KNOB_MARGIN
        knob = QRectF(
            KNOB_MARGIN + travel * self._position, KNOB_MARGIN, diameter, diameter
        )
        painter.setBrush(self._knob_color())
        painter.drawEllipse(knob)

        if self.hasFocus():
            # 鍵盤操作看得出焦點在哪。畫在軌道外側,不佔滑塊的空間。
            pen = QPen(self.palette().color(QPalette.ColorRole.Highlight))
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track.adjusted(0.75, 0.75, -0.75, -0.75), radius, radius)

    def _track_color(self) -> QColor:
        """關 → 開之間做線性混色。

        關的顏色取自 palette 而不是寫死的灰:深色模式下寫死的淺灰會亮得
        像是「開著」。開的顏色固定用 Apple 的綠 —— 那是這個開關的識別。
        """
        off = self.palette().color(QPalette.ColorRole.Mid)
        colour = _blend(off, ON_COLOR, self._position)
        if not self.isEnabled():
            colour.setAlphaF(0.4)
        return colour

    def _knob_color(self) -> QColor:
        """滑塊固定是白的 —— iOS 在淺色與深色模式下都是白的。"""
        colour = QColor(Qt.GlobalColor.white)
        if not self.isEnabled():
            colour.setAlphaF(0.6)
        return colour

    # ------------------------------------------------------------------ 其他

    def __repr__(self) -> str:
        return (
            f"<ToggleSwitch checked={self.isChecked()} "
            f"position={self._position:.2f} enabled={self.isEnabled()}>"
        )


def _blend(start: QColor, end: QColor, ratio: float) -> QColor:
    ratio = max(0.0, min(1.0, ratio))
    return QColor(
        round(start.red() + (end.red() - start.red()) * ratio),
        round(start.green() + (end.green() - start.green()) * ratio),
        round(start.blue() + (end.blue() - start.blue()) * ratio),
    )
