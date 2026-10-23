"""「框選牌桌範圍」對話框。

為什麼需要手動框
----------------
自動校正是從四邊往中間剝除整條顏色一致的帶狀區域。它是啟發式,而且
**錯的時候很安靜**:

* 實測 ``20260801-112437``(1800x1186):剝除停在瀏覽器工具列裡,剝出來的
  長寬比 1.56,整份錄影沒有一幀校得準。
* 同一份錄影開頭遊戲正在重連,畫面全黑。黑畫面剝得特別順,給出一個長寬比
  1.7645(離 16:9 只差 0.7%)的矩形,既有的兩道防線一道都沒響,於是被鎖進
  整場。手牌辨識 97% → 26%,全程無錯誤訊息。

與「框選手牌範圍」的差別
------------------------
那個是**取代**校正 —— 框了之後那條路上根本不算牌桌矩形。
這個**就是**校正的結果 —— 框了之後 ROI 與四家牌河的角點仍然照常相對它推導,
只是那個矩形改成人給的。所以這個框影響的東西多得多。

為什麼拖完要當場驗一次
----------------------
與 :mod:`~mia.ui.panel.handroi` 同一個理由,而且更要緊:牌桌矩形是所有
正規化座標的共同原點,框歪 3% 手牌的槽位就會一路累加偏移,到第 13 張差了
半張牌 —— 十三張、合法、全錯。

所以拖完會把從這個框推導出來的東西**畫回畫面上**:自己的手牌 ROI,以及
四家牌河的四邊形。框歪了,那些線就不會落在牌上 —— 看得見。再加上當場認一次
手牌,認不出來「確定」就不能按:存不下一個已知壞掉的框。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mia.calibration.table import Calibration
from mia.capture.base import Frame
from mia.config.models import RoiConfig
from mia.ui.panel.picker import DragPicker
from mia.ui.style import CAPTION, MUTED
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons, TileLabel
from mia.utils.geometry import NormRect, Rect, Size
from mia.vision.roi import RoiSet
from mia.vision.tiles.classify import TemplateSet, classify_hand
from mia.vision.tiles.hand import MAX_CONCEALED, read_hand
from mia.vision.tiles.river import DEFAULT_CORNERS, SEATS

__all__ = ["TableRectDialog"]

#: 預覽上畫牌的高度。
_RESULT_TILE = 34

#: 16:9。框出來的比例離它太遠通常是把瀏覽器工具列也框進去了。
_EXPECTED_ASPECT = 16 / 9

#: 長寬比差多少才提醒。不擋 —— 有些環境本來就不是 16:9
#: (實測 Steam 版客戶區 1512x844 = 1.7915)。
_ASPECT_HINT = 0.03


class TableRectDialog(QDialog):
    """框一次牌桌範圍。

    Args:
        frame: 剛抓到的遊戲畫面。
        roi: ROI 設定,用來從框推導出自己的手牌範圍並驗一次。
        skin: 牌面素材,用來把認出來的牌畫出來。

    Note:
        :attr:`result_rect` 在按下「確定」之後才有值,而那顆按鈕只有在**真的
        認出一手合法的牌**時才能按。
    """

    def __init__(
        self,
        frame: Frame,
        *,
        roi: RoiConfig,
        skin: str = DEFAULT_SKIN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("框選牌桌範圍")
        self._image = frame.image
        self._roi = roi
        self._templates: TemplateSet | None = None
        #: 驗過的牌桌矩形,正規化成影像的比例。``None`` 代表還沒有結果。
        self.result_rect: NormRect | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QLabel(
            "框出遊戲畫布 —— 藍色牌桌連同左右兩側的 UI,"
            "**不要**含瀏覽器的分頁列與網址列。\n"
            "拖完會把從這個框算出來的手牌範圍與四家牌河畫回畫面上,"
            "線沒落在牌上就是框歪了。",
            self,
        )
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)

        self._picker = DragPicker(self._image, self)
        self._picker.on_release = self._check
        layout.addWidget(self._picker, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._message = QLabel("還沒框 —— 在上面的畫面裡拖一個框", self)
        self._message.setStyleSheet(MUTED)
        layout.addWidget(self._message)

        self._tiles_row = QWidget(self)
        tiles_layout = QHBoxLayout(self._tiles_row)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        tiles_layout.setSpacing(2)
        icons = TileIcons(skin)
        # 多開一個位子給摸的那張 —— 使用者在自己摸牌那一巡框選時,
        # 少開的話畫出來會剛好少掉最關鍵的那一張。
        self._tiles = [
            TileLabel(icons, _RESULT_TILE, self._tiles_row) for _ in range(MAX_CONCEALED + 1)
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

    # ------------------------------------------------------------------ 驗證

    def _check(self) -> None:
        """拖完之後:推導 ROI 與牌河 → 畫出來 → 認一次手牌。

        每一步失敗都說**那一步**的話。「框選失敗」這種句子幫不上忙 ——
        使用者要知道的是框大了、框小了,還是只是這一巡手上有副露。
        """
        self._show_tiles(())
        self.result_rect = None
        self._picker.clear_derived()
        self._set_ok(False)

        box = self._picker.box
        if box is None:
            self._message.setText("還沒框 —— 在上面的畫面裡拖一個框")
            self._picker.update()
            return

        table = Rect(box.x(), box.y(), box.width(), box.height())
        width, height = self._picker.image_size
        self._picker.polys = _river_quads(table)

        hand_roi = _hand_roi(self._roi, table, Size(width, height))
        if hand_roi is None:
            self._message.setText(
                "設定檔裡沒有 own_hand 這個 ROI —— 框還是存得下去,"
                "但這裡沒辦法當場驗給你看(見 config/default.yaml)"
            )
            self._picker.update()
            return
        self._picker.rects = [QRect(hand_roi.x, hand_roi.y, hand_roi.width, hand_roi.height)]

        aspect_note = _aspect_note(table)
        crop = self._image[hand_roi.as_slice()]
        hand = read_hand(crop) if crop.size else None
        if hand is None or not hand.is_plausible:
            found = 0 if hand is None else len(hand)
            self._message.setText(
                f"算出來的手牌範圍裡讀到 {found} 張,不是合法手牌 —— "
                f"框歪了的話重拖一次;手上有副露的話先打到沒有再框。{aspect_note}"
            )
            self._picker.update()
            return

        if self._templates is None:
            self._templates = TemplateSet.load()
        try:
            concealed, drawn = classify_hand(crop, hand, self._templates, headroom=0)
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
            + f",平均把握 {score:.2f}。綠線是四家牌河 —— 都落在牌上就按「確定」。"
            + aspect_note
        )
        self.result_rect = NormRect(
            table.x / width, table.y / height, table.width / width, table.height / height
        )
        self._set_ok(True)
        self._picker.update()

    def _show_tiles(self, labels: list[str] | tuple[str, ...]) -> None:
        for label, tile in zip(self._tiles, labels, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._tiles[len(labels) :]:
            label.setVisible(False)


def _aspect_note(table: Rect) -> str:
    """長寬比離 16:9 太遠時的提醒。沒問題就回空字串。

    不擋,只提醒:有些環境本來就不是 16:9(實測 Steam 版客戶區 1.7915)。
    但差太多最常見的原因是把瀏覽器工具列也框進去了,那值得當場講一聲。
    """
    deviation = abs(table.aspect - _EXPECTED_ASPECT) / _EXPECTED_ASPECT
    if deviation <= _ASPECT_HINT:
        return ""
    return (
        f"\n⚠ 長寬比 {table.aspect:.3f},離 16:9 差 {deviation:.0%} ——"
        "最常見的原因是把瀏覽器工具列也框進去了。"
    )


def _hand_roi(roi: RoiConfig, table: Rect, image: Size) -> Rect | None:
    """從框出來的牌桌矩形推導自己的手牌範圍。``own_hand`` 沒設定時回 ``None``。"""
    rois = RoiSet(roi, Calibration(table_rect=table, image_size=image, source="manual"))
    return rois["own_hand"].rect if "own_hand" in rois else None


def _river_quads(table: Rect) -> list[list[tuple[int, int]]]:
    """四家牌河在畫面上的四邊形。

    畫出來而不只是算一算:牌河的角點是相對牌桌矩形的比例,所以它們落得準不準
    **就是**這個框對不對的證據 —— 而且是比手牌更嚴的證據,因為它們在畫面中央,
    離框的邊緣最遠,框的誤差在那裡放得最大。
    """
    quads = []
    for seat in SEATS:
        quad = DEFAULT_CORNERS[seat].quad(table)
        quads.append([(int(x), int(y)) for x, y in quad])
    return quads
