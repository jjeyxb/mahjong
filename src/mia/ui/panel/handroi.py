"""「框選手牌範圍」對話框。

為什麼需要手動框
----------------
Steam 版的畫布不是 16:9(實機量到 1512x844 = 1.7915),選單裡的畫布 preset
沒有一個對得上;而另一條路 —— 邊框剝除 —— 實測會**安靜地**鎖進偏掉的矩形
(手牌辨識 96% → 29%,全程無錯誤訊息)。框一次最直接。

**使用者拖出來的框不直接拿來用。**
:class:`~mia.vision.tiles.hand.HandLayout` 的槽位是 ROI 寬高的固定比例,
框歪 30px 就整手變成 ``5z 5z 5z …`` —— 13 張、合法、全錯。所以拖完之後交給
:func:`~mia.vision.tiles.locate.find_hand_roi` 在框內還原真正的 ROI,
使用者只要框得**大致**對就好。

為什麼一定要當場認一次
----------------------
這整個設計的安全網就是它。偵測可能失敗(最常見的是框選時手上有副露,
見 ``find_hand_roi`` 的 warning),而失敗的樣子正是這個專案最怕的那種:
一手看起來合法、實際上全錯的牌。認一次畫出來,錯就錯得看得見,
而「確定」在認不出手牌時是停用的 —— 存不下一個已知壞掉的框。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mia.capture.base import Frame
from mia.ui.style import ACCENT, CAPTION, MUTED
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons, TileLabel
from mia.utils.geometry import NormRect
from mia.vision.tiles.classify import TemplateSet, classify_hand
from mia.vision.tiles.hand import MAX_CONCEALED, read_hand
from mia.vision.tiles.locate import find_hand_roi

__all__ = ["HandRoiDialog"]

#: 預覽圖最大邊長(邏輯像素)。再大就放不進一般的筆電螢幕。
_PREVIEW_MAX = 820

#: 預覽上畫牌的高度。
_RESULT_TILE = 34

#: 拖出來的框至少要這麼大(預覽像素),不然當成誤觸的點擊。
_MIN_DRAG = 12


class _Picker(QWidget):
    """顯示畫面並讓使用者拖一個框。座標對外一律是**影像像素**。"""

    def __init__(self, image: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = (image.shape[1], image.shape[0])
        self._pixmap = _to_pixmap(image)
        scaled = self._pixmap.size().scaled(
            _PREVIEW_MAX, _PREVIEW_MAX, Qt.AspectRatioMode.KeepAspectRatio
        )
        self._pixmap = self._pixmap.scaled(
            scaled, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setFixedSize(self._pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start: QPoint | None = None
        self._box = QRect()
        #: 還原出來的 ROI(影像座標),畫成第二個框讓使用者看到差別。
        self.refined: QRect | None = None
        self.on_release: object = None

    # -------------------------------------------------------------- 座標換算

    def _to_image(self, rect: QRect) -> QRect:
        """預覽座標 → 影像座標。"""
        kx = self._size[0] / max(1, self._pixmap.width())
        ky = self._size[1] / max(1, self._pixmap.height())
        return QRect(
            round(rect.x() * kx), round(rect.y() * ky), round(rect.width() * kx),
            round(rect.height() * ky),
        )

    def _to_preview(self, rect: QRect) -> QRect:
        kx = self._pixmap.width() / max(1, self._size[0])
        ky = self._pixmap.height() / max(1, self._size[1])
        return QRect(
            round(rect.x() * kx), round(rect.y() * ky), round(rect.width() * kx),
            round(rect.height() * ky),
        )

    @property
    def box(self) -> QRect | None:
        """使用者拖的那個框,影像座標。還沒拖過時 ``None``。"""
        return self._to_image(self._box.normalized()) if not self._box.isNull() else None

    # -------------------------------------------------------------- 滑鼠

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._start = event.position().toPoint()
        self._box = QRect(self._start, self._start)
        self.refined = None
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._box = QRect(self._start, event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._start is None:
            return
        self._box = QRect(self._start, event.position().toPoint()).normalized()
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
        if self.refined is not None:
            painter.setPen(QPen(QColor(ACCENT), 2))
            painter.drawRect(self._to_preview(self.refined).adjusted(0, 0, -1, -1))
        painter.end()


class HandRoiDialog(QDialog):
    """框一次手牌範圍。

    Args:
        frame: 剛抓到的遊戲畫面。
        skin: 牌面素材,用來把認出來的牌畫出來。

    Note:
        :attr:`result_rect` 在按下「確定」之後才有值,而那顆按鈕只有在**真的
        認出一手合法的牌**時才能按 —— 存不下一個已知壞掉的框。
    """

    def __init__(
        self, frame: Frame, *, skin: str = DEFAULT_SKIN, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("框選手牌範圍")
        self._image = frame.image
        self._templates: TemplateSet | None = None
        #: 還原並驗過的 ROI,正規化成影像的比例。``None`` 代表還沒有結果。
        self.result_rect: NormRect | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QLabel(
            "在自己的手牌上拖一個框,大致框住就好 —— 程式會自己對齊到牌上。\n"
            "框選時手牌要是完整的暗手牌(不要有副露),拖完下面會當場認一次給你看。",
            self,
        )
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)

        self._picker = _Picker(self._image, self)
        self._picker.on_release = self._recognise
        layout.addWidget(self._picker, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._message = QLabel("還沒框 —— 在上面的畫面裡拖一個框", self)
        self._message.setStyleSheet(MUTED)
        layout.addWidget(self._message)

        self._tiles_row = QWidget(self)
        tiles_layout = QHBoxLayout(self._tiles_row)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        tiles_layout.setSpacing(2)
        icons = TileIcons(skin)
        # 多開一個位子給摸的那張 —— 只開 MAX_CONCEALED 的話,
        # 使用者在自己摸牌那一巡框選時,畫出來的牌會剛好少掉最關鍵的那一張。
        self._tiles = [
            TileLabel(icons, _RESULT_TILE, self._tiles_row)
            for _ in range(MAX_CONCEALED + 1)
        ]
        for label in self._tiles:
            label.setVisible(False)
            tiles_layout.addWidget(label)
        tiles_layout.addStretch(1)
        layout.addWidget(self._tiles_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        for standard, text in (
            (QDialogButtonBox.StandardButton.Ok, "確定"),
            (QDialogButtonBox.StandardButton.Cancel, "取消"),
        ):
            button = self._buttons.button(standard)
            if button is not None:
                button.setText(text)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._set_ok(False)

    def _set_ok(self, enabled: bool) -> None:
        """「確定」只有在真的認出一手牌時才能按 —— 存不下一個已知壞掉的框。"""
        button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(enabled)

    # ------------------------------------------------------------------ 辨識

    def _recognise(self) -> None:
        """拖完之後:還原 ROI → 認一次 → 把結果畫出來。

        每一步失敗都說**那一步**的話。「框選失敗」這種句子幫不上忙 ——
        使用者要知道的是該重框、還是該先把副露打掉、還是根本沒框到牌。
        """
        self._show_tiles(())
        self.result_rect = None
        self._picker.refined = None
        self._set_ok(False)

        box = self._picker.box
        if box is None:
            self._message.setText("還沒框 —— 在上面的畫面裡拖一個框")
            self._picker.update()
            return

        crop = self._image[box.y() : box.y() + box.height(), box.x() : box.x() + box.width()]
        roi = find_hand_roi(crop) if crop.size else None
        if roi is None:
            self._message.setText("框裡看不出一排牌 —— 框大一點,把整排手牌包進去")
            self._picker.update()
            return

        x, y = box.x() + roi.x, box.y() + roi.y
        self._picker.refined = QRect(x, y, roi.width, roi.height)
        tall = self._image[y : y + roi.height, x : x + roi.width]
        hand = read_hand(tall)
        if not hand.is_plausible:
            self._message.setText(
                f"對齊到的範圍裡讀到 {len(hand)} 張,不是合法手牌 —— "
                "手上有副露的話先打到沒有再框,或是框歪了,重拖一次"
            )
            self._picker.update()
            return

        if self._templates is None:
            self._templates = TemplateSet.load()
        try:
            concealed, drawn = classify_hand(tall, hand, self._templates, headroom=0)
        except ValueError as exc:
            self._message.setText(f"牌太小認不出來:{exc}")
            self._picker.update()
            return

        matches = [*concealed, *([drawn] if drawn else [])]
        score = float(np.mean([m.score for m in matches]))
        self._show_tiles([m.label for m in matches])
        self._message.setText(
            f"認出 {len(concealed)} 張暗手牌"
            + (f" + 摸的 {drawn.label}" if drawn else "")
            + f",平均把握 {score:.2f} —— 對的話按「確定」"
        )
        height, width = self._image.shape[:2]
        self.result_rect = NormRect(x / width, y / height, roi.width / width, roi.height / height)
        self._set_ok(True)
        self._picker.update()

    def _show_tiles(self, labels: list[str] | tuple[str, ...]) -> None:
        for label, tile in zip(self._tiles, labels, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._tiles[len(labels) :]:
            label.setVisible(False)


def _to_pixmap(image: np.ndarray) -> QPixmap:
    """BGR ndarray → QPixmap。

    ``QImage`` 不複製傳進去的記憶體,而這裡的 ndarray 是臨時的 —— 不 copy()
    的話畫出來是一片雜訊或直接當掉,而且是**有時候**才發生。
    """
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    height, width = rgb.shape[:2]
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
