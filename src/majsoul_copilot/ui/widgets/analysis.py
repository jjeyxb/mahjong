"""「向聽分析」分頁的內容 —— 功能 1 的輸出。

這一頁不需要連線。手牌從畫面認出來、向聽與進張用 ``mahjong`` 套件算,
整條路都不碰封包。所以就算引擎沒接上、或使用者只想要一個輕量的輔助,
這一頁仍然有東西可看。

進張枚數標成估計值
------------------
算剩餘張數時只扣掉**自己手上**看得到的。牌河、副露、寶牌指示牌都不在功能 1
的辨識範圍內,所以一張已經被打掉三張的牌,這裡仍然當作剩 4 張。UI 上一定要
講清楚,不然使用者會拿它當精確數字用。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from majsoul_copilot.analysis import AGARI, TENPAI
from majsoul_copilot.mjai.tiles import ms_to_mjai, sort_key
from majsoul_copilot.ui.viewmodel import ViewState
from majsoul_copilot.ui.widgets.tiles import HandStrip, TileIcons, TileLabel

__all__ = ["AnalysisTab"]

#: 進張與打牌選項各列幾個。
MAX_UKEIRE = 10
MAX_DISCARDS = 5


class _DiscardRow(QWidget):
    """一列打牌選項:牌面圖 + 切完之後的向聽與進張。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._tile = TileLabel(icons, 30, self)
        self._text = QLabel("", self)
        layout.addWidget(self._tile)
        layout.addWidget(self._text)
        layout.addStretch(1)

    def update_from(self, tile: str, text: str, *, best: bool) -> None:
        self._tile.set_tile(tile)
        self._text.setText(text)
        self._text.setStyleSheet("font-weight: 600;" if best else "")


class AnalysisTab(QWidget):
    """向聽分析分頁。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icons = icons
        self._discard_rows: list[_DiscardRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._shanten = QLabel("等待手牌…", self)
        self._shanten.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(self._shanten)

        self._source = QLabel("", self)
        self._source.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self._source)

        layout.addWidget(_rule(self))
        layout.addWidget(_caption("手牌", self))
        self._hand = HandStrip(icons, height=44, parent=self)
        layout.addWidget(self._hand)

        layout.addWidget(_rule(self))
        self._ukeire_caption = _caption("進張(估計值,未扣牌河與副露)", self)
        layout.addWidget(self._ukeire_caption)
        self._ukeire = HandStrip(icons, height=32, parent=self)
        layout.addWidget(self._ukeire)
        self._ukeire_text = QLabel("", self)
        self._ukeire_text.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._ukeire_text.setWordWrap(True)
        layout.addWidget(self._ukeire_text)

        layout.addWidget(_rule(self))
        self._discard_caption = _caption("打牌選項(只看聽牌速度)", self)
        layout.addWidget(self._discard_caption)
        self._discards = QWidget(self)
        self._discards_layout = QVBoxLayout(self._discards)
        self._discards_layout.setContentsMargins(0, 0, 0, 0)
        self._discards_layout.setSpacing(3)
        layout.addWidget(self._discards)

        layout.addStretch(1)

    def update_from(self, state: ViewState) -> None:
        self._update_headline(state)
        self._hand.set_tiles(_concealed(state), drawn=state.drawn)
        self._update_ukeire(state)
        self._update_discards(state)

    def _update_headline(self, state: ViewState) -> None:
        analysis = state.analysis
        if analysis is None:
            self._shanten.setText("手牌讀不出來" if state.hand else "等待手牌…")
            self._source.setText("")
            return

        if analysis.shanten == AGARI:
            self._shanten.setText("和了")
        elif analysis.shanten == TENPAI:
            self._shanten.setText("聽牌")
        else:
            self._shanten.setText(f"{analysis.shanten} 向聽")

        parts = [_SOURCE_NAMES.get(state.hand_source, state.hand_source)]
        if analysis.melds:
            parts.append(f"副露 {analysis.melds} 組")
        if state.hand_conflict:
            parts.append("⚠ CV 與封包不一致")
        self._source.setText("   ".join(p for p in parts if p))

    def _update_ukeire(self, state: ViewState) -> None:
        analysis = state.analysis
        ukeire = analysis.ukeire[:MAX_UKEIRE] if analysis else ()
        show = bool(ukeire)
        self._ukeire_caption.setVisible(show)
        self._ukeire.setVisible(show)
        self._ukeire_text.setVisible(show)
        if not show:
            return
        self._ukeire.set_tiles(tuple(_to_mjai(u.tile) for u in ukeire))
        total = analysis.total_ukeire if analysis else 0
        detail = "  ".join(f"{u.tile}×{u.count}" for u in ukeire)
        self._ukeire_text.setText(f"共 {total} 枚　{detail}")

    def _update_discards(self, state: ViewState) -> None:
        options = state.discards[:MAX_DISCARDS]
        self._discard_caption.setVisible(bool(options))

        while len(self._discard_rows) < len(options):
            row = _DiscardRow(self._icons, self._discards)
            self._discards_layout.addWidget(row)
            self._discard_rows.append(row)

        for index, (row, option) in enumerate(zip(self._discard_rows, options, strict=False)):
            state_text = "聽牌" if option.shanten == TENPAI else f"{option.shanten} 向聽"
            row.update_from(
                _to_mjai(option.tile),
                f"{state_text}　進張 {option.total_ukeire} 枚",
                best=index == 0,
            )
            row.setVisible(True)
        for row in self._discard_rows[len(options) :]:
            row.setVisible(False)


_SOURCE_NAMES = {"cv": "來源:畫面辨識", "packet": "來源:封包", "both": "來源:畫面 + 封包"}


def _to_mjai(tile: str) -> str:
    """向聽模組回的是雀魂記法,牌面圖要 MJAI 記法。"""
    try:
        return ms_to_mjai(tile)
    except ValueError:
        return tile


def _concealed(state: ViewState) -> tuple[str, ...]:
    """暗手牌部分,排序後顯示。

    ``state.hand`` 含摸的那張,要把它拿掉一份 —— 用 remove 而不是切掉最後一個,
    因為排序過之後那張不一定在尾巴。手上有同款的另一張時任意移除一份即可,
    兩張長得一樣。

    排序用 :func:`sort_key` 而不是預設的字串比較:後者會把筒子插進萬子中間
    (``1m`` < ``1p`` < ``2m``),13 張手牌看起來像壞掉。
    """
    tiles = sorted(state.hand, key=sort_key)
    if state.drawn and state.drawn in tiles:
        tiles.remove(state.drawn)
    return tuple(tiles)


def _rule(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _caption(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setStyleSheet("color: palette(mid); font-size: 11px;")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    return label
