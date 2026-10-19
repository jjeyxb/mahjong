"""Overlay:直接疊在遊戲畫面上的 HUD。

側邊視窗與這裡是**兩種呈現方式、同一份狀態** —— 兩者都訂閱同一個
:class:`~mia.ui.viewmodel.ViewModel`(理由見該模組的 docstring)。所以這個
檔案裡沒有任何「算什麼」的邏輯,只有「畫成什麼樣」;要顯示哪個動詞、哪幾張牌
一律問 :class:`~mia.ui.viewmodel.EngineView` 的屬性,不在這裡重拆一次
``action`` 字串。

為什麼疊上去不會干擾辨識
------------------------
主路徑是**單視窗擷取**(macOS ``CGWindowListCreateImage``、Windows
``PrintWindow``),抓的是視窗自身的緩衝區而不是螢幕上的合成結果,所以蓋在
遊戲上面的東西根本不會出現在辨識輸入裡 —— macOS 上實測連視窗被完全遮住都
照樣抓得到。**mss fallback 是例外**:它抓螢幕,Overlay 會被拍進去,所以那條
路上要讓 HUD 避開手牌區(:mod:`mia.capture.factory` 在退回 mss 時就會警告)。

鎖定與拖曳是同一件事的兩面
--------------------------
對滑鼠透明的視窗收不到滑鼠事件 —— 「可以拖」與「點得到底下的遊戲」不可能
同時成立。所以做成一個鎖:

* **未鎖定**:畫一圈虛線邊框,可以拖,滑鼠事件停在 Overlay 上。
* **鎖定**:邊框收起來,整個視窗對滑鼠透明,點擊直接穿到遊戲。

預設未鎖定,因為第一次開起來一定得先把它拖到想要的位置。

顏色為什麼寫死
--------------
側邊視窗貼在系統的視窗裝飾旁邊,跟著 ``palette()`` 走才不突兀;Overlay 貼的是
**遊戲畫面** —— 牌桌是綠的、立繪什麼顏色都有。跟著系統的淺色主題走的話,
淺色的字會直接消失在牌桌上。所以這裡是寫死的深底淺字,與系統主題無關。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mia import features
from mia.analysis import DangerReport, TileDanger, Ukeire
from mia.engine.actions import Candidate
from mia.mjai.tiles import tile_name
from mia.ui.state import UiState
from mia.ui.switchboard import Switchboard, is_on
from mia.ui.viewmodel import EngineView, ViewModel, ViewState, q_fraction, shanten_text
from mia.ui.widgets.danger import LEVEL_COLORS, danger_note, seat_name
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons, TileLabel

__all__ = ["OverlayWindow"]

#: 展開時最多列幾個候選。比側邊視窗(8)少 —— Overlay 蓋在遊戲上,長度是代價。
MAX_CANDIDATES = 5

#: 「自己要拿出來的牌」最多幾張。吃碰是 2、大明槓是 3。與側邊視窗同一個理由。
_MAX_OWN_TILES = 3

#: 牌面圖的高度。比側邊視窗(64)小:HUD 要看得清楚,但不能占掉半個牌桌。
_TILE_HEIGHT = 40
_CANDIDATE_TILE_HEIGHT = 22

#: 「其他引擎怎麼說」最多列幾行。
#:
#: Overlay 不是完整的比較表 —— 那在側邊視窗的 AI 建議頁。這裡只回答
#: 「另一個風格會不會切不一樣的牌」,而那是一行就講得完的事。
_MAX_OTHER_ENGINES = 2

#: 放銃危險度最多列幾張 —— 一手牌就是這麼多,所以這不是「只列前幾名」,
#: 而是一個防呆上限。使用者要的就是**每一張都列出來**:挑要切哪一張的時候,
#: 被截掉的那幾張正好是最危險的那幾張,而那是最需要看到的。
_MAX_DANGER_ROWS = 14

#: 危險度那一列的牌面高度。比進張再小一階 —— 它一次要疊十四列。
_DANGER_TILE_HEIGHT = 22

_BG = QColor(24, 26, 32, 214)
_TEXT = "#F2F2F7"
_MUTED = "#9CA0AA"
#: Apple 的 system green,與 :class:`~mia.ui.widgets.toggle.ToggleSwitch` 同一個。
_ACCENT = "#34C759"
_RADIUS = 12

_BAR_RANGE = 1000
_VERB_STYLE = f"font-size: 22px; font-weight: 600; color: {_TEXT};"
_VERB_MUTED = f"font-size: 22px; font-weight: 600; color: {_MUTED};"


class _CandidateRow(QWidget):
    """展開後的一列:牌面圖(或動作名)/ Q 值 / 長條。

    刻意比側邊視窗那一列窄:這裡只回答「差多少」,要看完整清單就去側邊視窗。
    """

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tile = TileLabel(icons, _CANDIDATE_TILE_HEIGHT, self)
        self._name = QLabel("", self)
        self._name.setMinimumWidth(34)
        self._name.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
        self._value = QLabel("", self)
        self._value.setMinimumWidth(46)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, _BAR_RANGE)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setMinimumWidth(90)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self._tile)
        layout.addWidget(self._name)
        layout.addWidget(self._value)
        layout.addWidget(self._bar, 1)

    def update_from(self, candidate: Candidate, *, low: float, high: float) -> None:
        if candidate.is_tile:
            self._tile.set_tile(candidate.label)
            self._tile.setVisible(True)
            self._name.setText("")
            self._name.setVisible(False)
        else:
            self._tile.setVisible(False)
            self._name.setText(candidate.display)
            self._name.setVisible(True)

        colour = _ACCENT if candidate.chosen else _MUTED
        weight = "600" if candidate.chosen else "400"
        self._value.setText(f"{candidate.q:+.2f}")
        self._value.setStyleSheet(f"color: {colour}; font-weight: {weight}; font-size: 12px;")
        self._bar.setValue(round(q_fraction(candidate.q, low=low, high=high) * _BAR_RANGE))
        self._bar.setStyleSheet(
            "QProgressBar { background: rgba(255,255,255,0.10);"
            " border: none; border-radius: 3px; }"
            f"QProgressBar::chunk {{ background: {colour}; border-radius: 3px; }}"
        )


class _DangerLine(QWidget):
    """危險度那一列:牌面圖 / 等級 / 對誰。

    比側邊視窗那一列少了待牌型的清單 —— 那一串在 HUD 上會橫跨半個牌桌
    (理由見 :func:`~mia.ui.widgets.danger.danger_note`)。
    """

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tile = TileLabel(icons, _DANGER_TILE_HEIGHT, self)
        self._level = QLabel("", self)
        self._level.setMinimumWidth(56)
        self._note = QLabel("", self)
        self._note.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")

        layout.addWidget(self._tile)
        layout.addWidget(self._level)
        layout.addWidget(self._note, 1)

    def update_from(self, danger: TileDanger, seat: int | None) -> None:
        # 牌名已經是 MJAI 記法(這條路的資料來自 MJAI 事件流),不要再轉一次
        # —— 數牌轉得過去、字牌會拋 TileError,所以那種錯只有摸到字牌才會爆。
        self._tile.set_tile(danger.tile)
        self._level.setText(danger.level.label)
        self._level.setStyleSheet(
            f"color: {LEVEL_COLORS[danger.level]}; font-size: 12px; font-weight: 600;"
        )
        self._note.setText(danger_note(danger, seat))


class OverlayWindow(QWidget):
    """疊在遊戲上的精簡 HUD。

    Args:
        viewmodel: 狀態來源。建構時先畫一次目前的狀態;之後由呼叫端
            ``viewmodel.subscribe(overlay.apply)`` 推過來。
        skin: 牌面素材,與側邊視窗同一套。
        switchboard: 三個功能的開關接到誰。``None``(重播、示範)時一律視為
            開著 —— 那些模式是命令列決定跑哪一條路,而它確實在跑。
        ui_state: 跨次啟動記住的位置與勾選。省略時讀預設路徑那一份。

    Note:
        自己**不決定要不要顯示** —— 那是側邊視窗設定頁上的勾選。這裡只提供
        :meth:`set_shown` / :meth:`set_expanded` / :meth:`set_danger_shown` /
        :meth:`set_locked` 四個動作,每一個都會順手把狀態寫回磁碟。
    """

    def __init__(
        self,
        viewmodel: ViewModel,
        *,
        skin: str = DEFAULT_SKIN,
        switchboard: Switchboard | None = None,
        ui_state: UiState | None = None,
    ) -> None:
        super().__init__()
        self._switchboard = switchboard
        self._ui_state = ui_state if ui_state is not None else UiState.load()
        self._state = viewmodel.state
        self._icons = TileIcons(skin)
        self._rows: list[_CandidateRow] = []
        self._danger_rows: list[_DangerLine] = []
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        # 內容一變(展開、候選變多、鳴牌多兩張牌)視窗就跟著長 —— HUD 不該有
        # 固定尺寸留一塊空白蓋在牌桌上,也不該被使用者拖成奇怪的形狀。
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        layout.addWidget(self._build_main())
        layout.addWidget(self._build_others())
        self._candidates = QWidget(self)
        self._candidates_layout = QVBoxLayout(self._candidates)
        self._candidates_layout.setContentsMargins(0, 2, 0, 0)
        self._candidates_layout.setSpacing(3)
        layout.addWidget(self._candidates)
        layout.addWidget(self._build_dangers())

        self._hint = QLabel("拖曳移動 —— 擺好之後到設定頁鎖定", self)
        self._hint.setStyleSheet(f"color: {_ACCENT}; font-size: 11px;")
        layout.addWidget(self._hint)

        self._apply_lock(self._ui_state.overlay_locked)
        self._hint.setVisible(not self._ui_state.overlay_locked)
        self._render()
        self._restore_position()

    # ------------------------------------------------------------------ 建構

    def _build_main(self) -> QWidget:
        """精簡那一行:動詞 + 牌 + 向聽/進張。"""
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._verb = QLabel("—", row)
        self._verb.setStyleSheet(_VERB_STYLE)
        self._subject = TileLabel(self._icons, _TILE_HEIGHT, row)
        self._joiner = QLabel("", row)
        self._joiner.setStyleSheet(f"color: {_MUTED}; font-size: 13px;")
        self._own = [TileLabel(self._icons, _TILE_HEIGHT, row) for _ in range(_MAX_OWN_TILES)]
        self._mark = QLabel("", row)
        self._mark.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        self._separator = QLabel("│", row)
        self._separator.setStyleSheet("color: rgba(255,255,255,0.22); font-size: 16px;")
        self._shanten = QLabel("", row)
        self._shanten.setStyleSheet(f"color: {_TEXT}; font-size: 15px; font-weight: 600;")

        layout.addWidget(self._verb)
        layout.addWidget(self._subject)
        layout.addWidget(self._joiner)
        for label in self._own:
            layout.addWidget(label)
        layout.addWidget(self._mark)
        layout.addStretch(1)
        layout.addWidget(self._separator)
        layout.addWidget(self._shanten)
        return row

    def _build_others(self) -> QWidget:
        """比較對象那幾行:引擎名 + 它會怎麼打。

        **為什麼 Overlay 也要有這個。** 挑了比較對象卻只看得到主要引擎,那個
        選單等於沒作用 —— 而兩份權重真正的差別只在**分歧的那幾手**顯現,
        看不到另一邊就等於看不到差別。

        寫成文字(``切 東``、``吃 3m ← 1m 2m``)而不是牌面圖:這一段是
        「另一個人會怎麼做」,不是要你現在點下去的動作。畫成同樣大小的牌面,
        畫面上就有兩個平起平坐的建議而沒有說明哪個是哪個 —— 與向聽那一句
        不寫成「切 + 牌面」是同一個理由。
        """
        block = QWidget(self)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(2)
        self._other_rows: list[tuple[QWidget, QLabel, QLabel]] = []
        self._others_layout = layout
        self._others = block
        return block

    def _build_dangers(self) -> QWidget:
        """危險度那一段:一行標題 + 每張牌一列。

        **為什麼是往下長一段,而不是擠進上面那一行。** 上面那一行回答的是
        「現在做什麼」(切哪張、還差幾向聽);這一段回答的是「哪些不能切」,
        而那是一份要逐張掃過去的清單,壓成一行就沒有用了。

        全部列出來、安全的在最上面 —— 決定要切哪一張的時候,眼睛從上往下
        找到第一張不影響進張的牌就停,不需要再去對照另一個視窗。
        """
        block = QWidget(self)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(3)

        self._danger_head = QLabel("", block)
        self._danger_head.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        layout.addWidget(self._danger_head)

        self._danger_list = QWidget(block)
        self._danger_layout = QVBoxLayout(self._danger_list)
        self._danger_layout.setContentsMargins(0, 0, 0, 0)
        self._danger_layout.setSpacing(2)
        layout.addWidget(self._danger_list)

        self._dangers = block
        return block

    # ------------------------------------------------------------------ 開關

    @property
    def shown(self) -> bool:
        """使用者要不要看到它。與 ``isVisible()`` 不同 —— 那個在沒 show 過的
        情況下永遠是 False,而這裡問的是「勾了沒」。"""
        return self._ui_state.overlay_visible

    @property
    def expanded(self) -> bool:
        return self._ui_state.overlay_expanded

    @property
    def danger_shown(self) -> bool:
        """放銃危險度要不要一起列在 HUD 上。"""
        return self._ui_state.overlay_danger

    @property
    def locked(self) -> bool:
        return self._ui_state.overlay_locked

    def set_shown(self, on: bool) -> None:
        """顯示或收起來。狀態會記住,下次啟動照舊。"""
        self._ui_state.overlay_visible = on
        self._ui_state.save()
        if on:
            # 隱藏期間不重畫(見 :meth:`apply`),所以顯示之前要先補上最新的狀態
            self._render()
            self.show()
            self.raise_()
        else:
            self.hide()

    def set_expanded(self, on: bool) -> None:
        """展開 / 收起候選 Q 值。開關在側邊視窗的設定頁,不在 Overlay 上
        —— Overlay 鎖定之後點不到,那個按鈕會變成死的。"""
        self._ui_state.overlay_expanded = on
        self._ui_state.save()
        self._render()

    def set_danger_shown(self, on: bool) -> None:
        """把放銃危險度加進 HUD(往下長一段)或收起來。

        與 ``features.DANGER`` 是兩個不同的問題:那個決定**有沒有在算**,
        這個決定**要不要占用畫面**。功能關著的時候勾這個不會有東西出現
        —— 那是對的,沒有資料可畫。
        """
        self._ui_state.overlay_danger = on
        self._ui_state.save()
        self._render()

    def set_locked(self, on: bool) -> None:
        """鎖定 = 對滑鼠透明。解鎖 = 可以拖。"""
        self._ui_state.overlay_locked = on
        self._ui_state.save()
        self._apply_lock(on)
        self._hint.setVisible(not on)
        self.update()

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

    # ------------------------------------------------------------------ 位置

    def _restore_position(self) -> None:
        """回到上次拖到的地方。

        存下來的座標可能已經不在任何一個螢幕上(外接螢幕拔掉了、解析度改了)
        —— 那樣 Overlay 會開在看不見的地方,而使用者只會覺得勾了沒反應。
        所以先確認那個點還落在某個螢幕上,不然回預設位置。
        """
        saved = self._ui_state.overlay_pos
        if saved is not None:
            point = QPoint(*saved)
            if QApplication.screenAt(point) is not None:
                self.move(point)
                return
        self._move_to_default()

    def _move_to_default(self) -> None:
        """預設擺在螢幕上緣置中。

        上緣是因為自己的手牌在畫面**下緣** —— 蓋住手牌的 HUD 會逼使用者
        為了看牌把它拖走,那第一印象就壞了。
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        size = self.sizeHint()
        self.move(
            area.left() + max(0, (area.width() - size.width()) // 2),
            area.top() + 32,
        )

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
            self._ui_state.overlay_pos = (point.x(), point.y())
            self._ui_state.save()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ 外觀

    def paintEvent(self, event: QPaintEvent) -> None:
        """圓角深色底。未鎖定時多一圈虛線 —— 那是「現在拖得動」的唯一提示。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        rect = self.rect().adjusted(1, 1, -1, -1)
        path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()),
                            float(rect.height()), _RADIUS, _RADIUS)
        painter.fillPath(path, _BG)
        if not self.locked:
            pen = QPen(QColor(_ACCENT), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        super().paintEvent(event)

    # ------------------------------------------------------------------ 資料

    def apply(self, state: ViewState) -> None:
        """把狀態畫上去。:class:`~mia.ui.viewmodel.ViewModel` 的訂閱入口。

        隱藏著就只記下來不重畫 —— 即時模式下這個函式一秒被呼叫 10 次,而
        沒在看的時候縮放十幾張牌面圖是白花的。:meth:`set_shown` 會在顯示前補畫。
        """
        self._state = state
        if self.isHidden():
            return
        self._render()

    def _render(self) -> None:
        state = self._state
        advice_on = is_on(self._switchboard, features.ADVICE)
        vision_on = is_on(self._switchboard, features.VISION)
        # 兩個條件都要:功能開著代表有在算,勾選代表使用者要看。
        danger_running = is_on(self._switchboard, features.DANGER)
        danger_on = self.danger_shown and danger_running
        self._show_dangers(state.dangers if danger_on else None)

        if not advice_on and not vision_on:
            self._show_shanten("")
            self._show_others(None)
            self._show_candidates(None)
            if danger_on:
                # 危險度自己就是內容了。這時候再喊一次「未開啟」會變成
                # 一個蓋在清單上面、說著相反的話的標題。
                self._show_advice(None, verb="")
            elif danger_running:
                # 放銃分析**正在跑**,只是還沒被請上 HUD。說「未開啟」是錯的
                # —— 使用者會去撥一個已經開著的開關,然後以為程式壞了。
                # 實機第一次開起來就是踩到這個。
                self._show_advice(
                    None, muted=True, verb="放銃分析中",
                    mark="到設定頁勾「Overlay 顯示放銃危險度」",
                )
            else:
                # 全部關著時要**主動說**,不然畫面上是一個空的黑框蓋在牌桌上,
                # 看起來像程式壞了。
                self._show_advice(
                    None, muted=True, verb="未開啟", mark="用側邊視窗的開關打開"
                )
            return

        engine = state.primary if advice_on else None
        if advice_on:
            self._show_advice(engine, stale=state.advice_is_stale)
        elif state.analysis is None:
            self._show_advice(None, muted=True, verb="等待手牌…")
        else:
            # 只有畫面辨識在跑:整段建議收起來,把位置讓給向聽
            self._show_advice(None, verb="")

        # 向聽**只在畫面辨識開著時顯示**。`analysis` 是從手牌算的,而手牌是
        # 封包優先 —— 所以只開 AI 建議時它照樣算得出來,於是 HUD 上會出現一個
        # 使用者沒有打開任何東西去要的數字。那一頁的開關關著,它的輸出就不該
        # 出現在另一個畫面上。
        self._show_shanten(_shanten_line(state) if vision_on else "")
        self._show_others(state if advice_on else None)
        self._show_candidates(engine if self.expanded else None)

    def _show_advice(
        self,
        engine: EngineView | None,
        *,
        stale: bool = False,
        muted: bool = False,
        verb: str | None = None,
        mark: str = "",
    ) -> None:
        if engine is None:
            text = "等待引擎…" if verb is None else verb
            self._verb.setText(text)
            self._verb.setVisible(bool(text))
            self._verb.setStyleSheet(_VERB_MUTED if muted or not text else _VERB_STYLE)
            self._show_tiles()
            self._mark.setText(mark)
            self._mark.setVisible(bool(mark))
            return

        self._verb.setText(engine.verb)
        self._verb.setVisible(True)
        # 「不需要動作」佔了九成的事件 —— 灰字讓它退到背景去,不然 HUD 會
        # 一直用同樣的份量喊一句沒有內容的話。建議已經打掉了(stale)也一樣。
        quiet = stale or (engine.action is None and not engine.declined)
        self._verb.setStyleSheet(_VERB_MUTED if quiet else _VERB_STYLE)
        self._show_tiles(engine.subject, joiner=engine.joiner, own=engine.own_tiles)
        self._mark.setText("·已打出" if stale else "")
        self._mark.setVisible(stale)

    def _show_tiles(
        self, subject: str | None = None, *, joiner: str = "", own: tuple[str, ...] = ()
    ) -> None:
        """畫這一手的牌。沒用到的位子要藏起來,不然會留上一手的殘影。"""
        if subject:
            self._subject.set_tile(subject)
        self._subject.setVisible(bool(subject))
        self._joiner.setText(joiner)
        self._joiner.setVisible(bool(joiner))
        own = own[:_MAX_OWN_TILES]
        for label, tile in zip(self._own, own, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._own[len(own) :]:
            label.setVisible(False)

    def _show_dangers(self, report: DangerReport | None) -> None:
        """畫危險度清單。``None``(沒開、或還沒有資料)時整段收起來。

        整段收起來而不是留一句「等待對局…」:HUD 的高度是直接從牌桌上拿走的,
        沒有內容的時候就該把那塊還回去。要知道為什麼沒東西,側邊視窗那一頁
        寫得清楚。
        """
        tiles = report.tiles[:_MAX_DANGER_ROWS] if report else ()
        if not tiles:
            self._dangers.setVisible(False)
            return

        assert report is not None
        while len(self._danger_rows) < len(tiles):
            row = _DangerLine(self._icons, self._danger_list)
            self._danger_layout.addWidget(row)
            self._danger_rows.append(row)

        for row, danger in zip(self._danger_rows, tiles, strict=False):
            row.update_from(danger, report.seat)
            row.setVisible(True)
        for row in self._danger_rows[len(tiles) :]:
            row.setVisible(False)

        self._danger_head.setText(_danger_headline(report))
        self._dangers.setVisible(True)

    def _show_others(self, state: ViewState | None) -> None:
        """畫比較對象。``None``(AI 建議沒開)時整段收起來。

        **分歧要標出來。** 兩份風格權重九成以上的手會給同一個答案,而那些手
        看了也沒用;有價值的是剩下那幾手。不標的話使用者得自己逐行比對兩個
        字串,那正是這塊面積該省掉的工。
        """
        engines = () if state is None else state.engines
        primary = None if state is None else state.primary
        others = [e for e in engines if e is not primary][:_MAX_OTHER_ENGINES]

        while len(self._other_rows) < len(others):
            name = QLabel("", self._others)
            name.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
            name.setMinimumWidth(96)
            value = QLabel("", self._others)
            row = QWidget(self._others)
            inner = QHBoxLayout(row)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(6)
            inner.addWidget(name)
            inner.addWidget(value, 1)
            self._others_layout.addWidget(row)
            self._other_rows.append((row, name, value))

        for (row, name, value), engine in zip(self._other_rows, others, strict=False):
            name.setText(engine.name)
            # 跳過的引擎也要標。這一手的分歧可能正是「它鳴、我不鳴」,而不標的話
            # 畫面上只有鳴的那一邊被標成分歧,看起來像它自己跟自己不一致。
            answered = engine.action is not None or engine.declined
            split = state is not None and not state.is_unanimous and answered
            value.setText(engine.headline + ("　⚠ 分歧" if split else ""))
            value.setStyleSheet(
                f"color: {_ACCENT if split else _TEXT}; font-size: 12px;"
                f" font-weight: {'600' if split else '400'};"
            )
            row.setVisible(True)
        for row, _name, _value in self._other_rows[len(others) :]:
            row.setVisible(False)
        self._others.setVisible(bool(others))

    def _show_shanten(self, text: str) -> None:
        self._shanten.setText(text)
        self._shanten.setVisible(bool(text))
        # 分隔線只有在兩邊都有東西時才有意義
        self._separator.setVisible(bool(text) and self._verb.isVisibleTo(self))

    def _show_candidates(self, engine: EngineView | None) -> None:
        candidates = engine.candidates[:MAX_CANDIDATES] if engine else ()
        while len(self._rows) < len(candidates):
            row = _CandidateRow(self._icons, self._candidates)
            self._candidates_layout.addWidget(row)
            self._rows.append(row)

        if candidates and engine is not None:
            values = [c.q for c in engine.candidates]
            low, high = min(values), max(values)
            for row, candidate in zip(self._rows, candidates, strict=False):
                row.update_from(candidate, low=low, high=high)
        for row in self._rows[: len(candidates)]:
            row.setVisible(True)
        for row in self._rows[len(candidates) :]:
            row.setVisible(False)
        self._candidates.setVisible(bool(candidates))


def _danger_headline(report: DangerReport) -> str:
    """清單上面那一行小字。

    誰值得指名要寫出來,而且**立直與三副露同一階** —— 有人坐在三副露上卻
    只說「沒人立直」,那是畫面說沒事而實際有事(實測那場最重的一次放銃,
    和牌的就是一個三副露、沒立直的人)。

    都沒有的時候也**不能只寫「無人立直」** —— 那會被讀成可以隨便打。
    HUD 塞不下側邊視窗那整段說明,所以濃縮成一句提醒。
    """
    if report.threats:
        who = "、".join(f"{seat_name(report.seat, t.seat)}{t.label}" for t in report.threats)
        return f"放銃危險度 —— {who},安全的在上"
    return "放銃危險度 —— 沒人立直也不代表安全,安全的在上"


def _effective_ukeire(state: ViewState) -> tuple[Ukeire, ...]:
    """這一刻該畫哪些進張。

    **摸完牌的那一瞬間 ``analysis.ukeire`` 依定義是空的** —— 手上 14 張,
    「再摸一張會怎樣」沒有意義,要先決定切哪張(見
    :attr:`~mia.analysis.shanten.HandAnalysis.needs_discard`)。

    而那正是最需要資訊的時候。原本的寫法在那一刻把整排牌面藏起來,只剩一個
    向聽數 —— 使用者回報的就是這個:「摸到牌的時候 overlay 顯示信息會消失」。

    所以這種時候改畫**切掉最優那張之後**的進張。是切哪一張寫在
    :func:`_shanten_line` 的括號裡,不然「進張 6」會是一個沒有前提的數字。
    """
    analysis = state.analysis
    if analysis is None or analysis.is_agari:
        # 和了的手牌 ukeire 也是空的,但那時候該做的事是**和牌**,不是切一張。
        # 少了這個判斷,一手 123m456m789m1p1p234p 會顯示「和了(切1m)進張 10」。
        return ()
    if analysis.ukeire:
        return analysis.ukeire
    best = state.best_discard
    return best.ukeire if best else ()


def _shanten_line(state: ViewState) -> str:
    """右半邊那一句:向聽 +(切哪張)。算不出來就是空的。

    **不寫進張。** 原本這裡會畫出進張的牌面與枚數,拿掉的理由是那塊面積在
    HUD 上太貴:進張種類多的時候整排牌會橫跨半個牌桌,而真正在決定下一手的
    是左半邊那個建議。要看進張就去側邊視窗的向聽分析頁,那裡有完整的枚數表
    (而且會說清楚那個數字只扣掉自己手上看得到的)。

    要切哪張是寫成**括號裡的小字**而不是像左半邊那樣的「切 + 牌面」:左邊那個
    是 Mortal 的建議,這邊是純速度算出來的,兩者常常不一樣。做成同樣的份量會
    變成畫面上有兩個平起平坐的「切某張」而沒有任何說明哪個是哪個。
    """
    analysis = state.analysis
    if analysis is None:
        return ""
    text = shanten_text(analysis.shanten)
    best = state.best_discard
    # 摸完牌那一刻 ukeire 依定義是空的(手上 14 張,要先決定切哪張)——
    # 那時候向聽數講的是「切掉某張之後」,不寫是哪張的話那個數字沒有前提。
    #
    # 仍然要走 _effective_ukeire 而不是直接看 best:**和了的手牌 ukeire 也是
    # 空的**,少了那道判斷會顯示「和了(切1m)」。進張的數字不畫了,這個判斷
    # 還是得留著。
    if _effective_ukeire(state) and not analysis.ukeire and best is not None:
        text += f"(切{tile_name(best.tile)})"
    if state.hand_conflict:
        # 兩個來源對不上本身就是使用者該知道的事,而 Overlay 沒有狀態列
        text += "　⚠"
    return text
