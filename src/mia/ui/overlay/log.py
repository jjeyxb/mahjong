"""Log 視窗:AI 建議吃到的事件流,MJAI 格式加中文說明。

與 :class:`~mia.ui.overlay.window.OverlayWindow` 互不相干:資料走自己的
:class:`~mia.ui.eventlog.EventLog`,開關、位置、大小、鎖定都各記各的。

顏色寫死的理由與 Overlay 相同,它一樣貼在遊戲畫面上。

鎖定用右上角那顆鎖(與 Overlay 同一個元件,見 :mod:`mia.ui.overlay.lock`)。
鎖定之後滑鼠穿透,捲不動,所以永遠跟著最新的事件。沒鎖定時往上捲就停在原地,捲回最
底下才繼續跟。

牌畫成牌面圖,用的是手牌那一欄同一套 :class:`~mia.ui.widgets.tiles.TileIcons`。素材不在時
退回中文牌名。
"""

from __future__ import annotations

import re
from html import escape

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QFontDatabase, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizeGrip,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from mia.mjai.describe import tile_text
from mia.ui.eventlog import TILE_MARK, EventLog, LogLine
from mia.ui.overlay.hud import HudWindow
from mia.ui.state import UiState
from mia.ui.style import ACCENT
from mia.ui.widgets.tiles import TILE_ASPECT, TileIcons

__all__ = ["EventLogWindow"]

_DEFAULT_SIZE = QSize(480, 360)
_MIN_SIZE = QSize(320, 180)

#: 拖完多久才把大小存起來。拖的過程中每個 pixel 都會觸發一次。
_SAVE_DELAY_MS = 400

#: 離底部幾像素以內算「跟著最新的」。
_FOLLOW_SLACK = 8

_TEXT = "#F2F2F7"
_MUTED = "#9CA0AA"
_WARN = "#FFB340"

#: 系統的捲軸是淺色的,在深色底上比內容還搶眼。
_BODY_STYLE = (
    f"QTextBrowser {{ background: transparent; border: none; color: {_TEXT}; }}"
    "QScrollBar:vertical { background: transparent; width: 6px; }"
    "QScrollBar::handle:vertical {"
    " background: rgba(255,255,255,0.25); border-radius: 3px; min-height: 24px; }"
    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
    "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }"
)

_EMPTY = "還沒有事件。AI 建議要開著,對局開始後才有事件流。"

#: 牌面圖的高度,邏輯像素。說明文字是 13px,圖比字高一點才認得出牌。
_TILE_HEIGHT = 26

#: 圖片網址的 scheme:``tile:5mr``。
_SCHEME = "tile"

_MARKED = re.compile(f"{TILE_MARK}([^{TILE_MARK}]+){TILE_MARK}")


class _TileBrowser(QTextBrowser):
    """``tile:<牌>`` 的圖片向 :class:`TileIcons` 要,依螢幕的裝置像素比畫。"""

    def __init__(self, parent: QWidget, icons: TileIcons) -> None:
        super().__init__(parent)
        self._icons = icons

    def loadResource(self, kind: int, name: QUrl | str) -> object:
        url = QUrl(name) if isinstance(name, str) else name
        if url.scheme() == _SCHEME:
            return self._icons.pixmap(url.path(), _TILE_HEIGHT, self.devicePixelRatioF())
        return super().loadResource(kind, name)


class EventLogWindow(HudWindow):
    """Log 視窗。

    Args:
        log: 資料來源。
        ui_state: 跨次啟動記住的位置與勾選。省略時讀預設路徑那一份。
    """

    def __init__(self, log: EventLog, *, ui_state: UiState | None = None) -> None:
        super().__init__()
        self._log = log
        self._ui_state = ui_state if ui_state is not None else UiState.load()
        self._attach_lock()
        self._apply_opacity(self.opacity)
        self._mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        self._following = True
        self._updating_scroll = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        layout.addLayout(self._build_head())

        self._note = QLabel("", self)
        self._note.setStyleSheet(f"color: {_WARN}; font-size: 11px;")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        try:
            self._icons: TileIcons | None = TileIcons()
        except FileNotFoundError:
            self._icons = None
        self._body = (
            _TileBrowser(self, self._icons) if self._icons is not None else QTextBrowser(self)
        )
        self._body.setStyleSheet(_BODY_STYLE)
        self._body.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._body.setOpenLinks(False)
        bar = self._body.verticalScrollBar()
        bar.valueChanged.connect(self._track_scroll)
        bar.rangeChanged.connect(self._follow_range)
        layout.addWidget(self._body, 1)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        self._hint = QLabel("拖曳上緣移動,右下角改大小 —— 擺好之後按右上角的鎖", self)
        self._hint.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        self._grip = QSizeGrip(self)
        foot.addWidget(self._hint, 1)
        foot.addWidget(self._grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        layout.addLayout(foot)

        self._save_size = QTimer(self)
        self._save_size.setSingleShot(True)
        self._save_size.setInterval(_SAVE_DELAY_MS)
        self._save_size.timeout.connect(self._remember_size)

        self.setMinimumSize(_MIN_SIZE)
        saved = self._ui_state.log_size
        self.resize(QSize(*saved) if saved is not None else _DEFAULT_SIZE)
        self._apply_lock(self.locked)
        self._show_lock(self.locked)
        self._render()
        self._restore_position()
        log.subscribe(self.apply)

    def _build_head(self) -> QHBoxLayout:
        """標題列。label 不收滑鼠事件,所以這一條就是拖曳的把手。"""
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel("事件流 Log", self)
        title.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        self._source = QLabel("", self)
        self._source.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self._source)
        return head

    # ------------------------------------------------------------------ 開關

    @property
    def shown(self) -> bool:
        """使用者要不要看到它。見 :attr:`OverlayWindow.shown`。"""
        return self._ui_state.log_visible

    @property
    def locked(self) -> bool:
        return self._ui_state.log_locked

    def set_shown(self, on: bool) -> None:
        """顯示或收起來。狀態會記住,下次啟動照舊。"""
        self._ui_state.log_visible = on
        self._ui_state.save()
        if on:
            # 隱藏期間不重畫(見 :meth:`apply`),所以顯示之前要先補上最新的狀態
            self._render()
            self.show()
            self.raise_()
            self._scroll_to_end()
        else:
            self.hide()

    def set_locked(self, on: bool) -> None:
        """鎖定 = 對滑鼠透明。解鎖 = 可以拖、可以捲。右上角那顆鎖按下去就是呼叫這裡。"""
        self._ui_state.log_locked = on
        self._ui_state.save()
        self._apply_lock(on)
        self._show_lock(on)
        self.update()

    def _show_lock(self, locked: bool) -> None:
        self._hint.setVisible(not locked)
        self._grip.setVisible(not locked)

    # ------------------------------------------------------------------ 位置

    def _saved_position(self) -> tuple[int, int] | None:
        return self._ui_state.log_pos

    def _remember_position(self, point: tuple[int, int]) -> None:
        self._ui_state.log_pos = point
        self._ui_state.save()

    def _move_to_default(self) -> None:
        """預設擺在螢幕右上。Overlay 在上緣置中,手牌在下緣,右邊最不擋東西。"""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.right() - self.width() - 24, area.top() + 32)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._ui_state.log_size != (self.width(), self.height()) and self.isVisible():
            self._save_size.start()

    def _remember_size(self) -> None:
        self._ui_state.log_size = (self.width(), self.height())
        self._ui_state.save()

    # ------------------------------------------------------------------ 資料

    def apply(self) -> None:
        """有新的事件。隱藏著就不畫,:meth:`set_shown` 會在顯示前補上。"""
        if not self.isHidden():
            self._render()

    def _render(self) -> None:
        bar = self._body.verticalScrollBar()
        following = self.locked or self._following
        position = bar.value()
        lines = self._log.lines
        self._source.setText(f"{self._log.source} · {len(lines)} 則" if lines else "")
        self._note.setText(self._log.note)
        self._note.setVisible(bool(self._log.note))
        pictures = self._icons is not None
        self._updating_scroll = True
        try:
            self._body.setHtml(
                "".join(_html(line, self._mono, pictures=pictures) for line in lines)
                or f'<p style="color:{_MUTED}">{_EMPTY}</p>'
            )
            if following:
                self._scroll_to_end()
            else:
                bar.setValue(position)
        finally:
            self._updating_scroll = False

    def _track_scroll(self) -> None:
        if not self._updating_scroll:
            bar = self._body.verticalScrollBar()
            self._following = bar.value() >= bar.maximum() - _FOLLOW_SLACK

    def _follow_range(self) -> None:
        # 長文件會延後排版,顯示、換行或改視窗大小後終點還會變。
        # 此時保留原本的跟隨狀態,不要把新增的高度誤認成使用者往上捲。
        if self.locked or self._following:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self._following = True
        was_updating = self._updating_scroll
        self._updating_scroll = True
        try:
            bar = self._body.verticalScrollBar()
            bar.setValue(bar.maximum())
        finally:
            self._updating_scroll = was_updating


def _html(line: LogLine, mono: str, *, pictures: bool = True) -> str:
    colour = ACCENT if line.heading else _TEXT
    weight = 600 if line.heading else 400
    text = _with_tiles(line.marked, pictures=pictures) if line.marked else escape(line.text)
    return (
        '<p style="margin:0 0 6px 0">'
        f'<span style="color:{colour};font-size:13px;font-weight:{weight}">'
        f"{text}</span><br>"
        f'<span style="color:{_MUTED};font-size:10px;font-family:{escape(mono)}">'
        f"{escape(line.mjai)}</span></p>"
    )


def _with_tiles(marked: str, *, pictures: bool) -> str:
    """把夾起來的牌換成牌面圖;沒有素材就換回中文牌名。其餘文字照常跳脫。"""
    width = round(_TILE_HEIGHT * TILE_ASPECT)
    parts = _MARKED.split(marked)
    out = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            out.append(escape(part))
        elif pictures:
            out.append(
                f'<img src="{_SCHEME}:{escape(part)}" width="{width}" height="{_TILE_HEIGHT}"'
                f' align="middle" title="{escape(tile_text(part))}">'
            )
        else:
            out.append(escape(tile_text(part)))
    return "".join(out)
