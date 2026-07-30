"""側邊視窗:左側功能列 + 每頁自己的設定列 + 狀態列。

```
┌──────┬──────────────────────────┐
│ AI   │ 主要引擎 [mortal ▾] 候選 8│  ← 該頁的設定
│ 建議 ├──────────────────────────┤
│      │                          │
│ 向聽 │  內容                     │
│ 分析 │                          │
│      │                          │
│ 設定 │                          │
└──────┴──────────────────────────┘
│ 狀態列                           │
```

**左側直排而不是頂端分頁。** 功能只會越加越多(Overlay、風格比較、錄製),
頂端分頁一多就開始擠成兩排或出現左右箭頭;直排往下長不會擠到內容,而且每頁
的頂端就空出來放那一頁自己的設定。這個排法參考 MAA。

分頁而不是把所有內容塞一頁的理由沒有變:功能 1(向聽/進張)與功能 2(AI 建議)
是兩條獨立的路 —— 前者不碰連線,後者不碰畫面。擠在一起會讓「只有一邊在運作」
很難表達,而那是實際使用時的常態。
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mia import APP_TITLE, features
from mia.ui.viewmodel import ViewModel, ViewState
from mia.ui.widgets.advice import AdviceTab
from mia.ui.widgets.analysis import AnalysisTab
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons
from mia.ui.widgets.toggle import ToggleSwitch

__all__ = ["PanelWindow", "Switchboard", "present"]

#: 放得下 14 張 44px 高的牌、加上左側功能列。
MIN_WIDTH = 500
MIN_HEIGHT = 540

#: 左側功能列的寬度。夠放四個全形字。
NAV_WIDTH = 96


class Switchboard(Protocol):
    """能開關功能的東西。

    只宣告視窗真正用到的兩個方法,而不是直接吃一個
    :class:`~mia.live.runtime.LiveRuntime` —— UI 這一層不該相依 live
    (它也要服務錄影重播,那時候根本沒有工作執行緒可開關)。
    """

    def available(self, key: str) -> bool: ...
    def is_enabled(self, key: str) -> bool: ...
    def set_enabled(self, key: str, on: bool) -> None: ...


class _Page(QWidget):
    """一個功能頁:頂端一條設定列,底下是內容。

    設定列**沒有控制項時整條隱藏**,不留一條空白的橫線 —— 那會讓人以為
    有東西沒載出來。
    """

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._settings = QWidget(self)
        self._settings_layout = QHBoxLayout(self._settings)
        self._settings_layout.setContentsMargins(8, 6, 8, 6)
        self._settings_layout.setSpacing(8)
        self._settings_layout.addStretch(1)
        self._settings.setVisible(False)

        self._rule = _rule(self)
        self._rule.setVisible(False)

        layout.addWidget(self._settings)
        layout.addWidget(self._rule)
        layout.addWidget(content, 1)

    def add_setting(self, label: str, widget: QWidget) -> None:
        """在設定列加一個「說明 + 控制項」。加在伸縮元件之前,所以會靠左排。"""
        index = self._settings_layout.count() - 1
        caption = QLabel(label, self._settings)
        caption.setStyleSheet("color: palette(mid); font-size: 11px;")
        self._settings_layout.insertWidget(index, caption)
        self._settings_layout.insertWidget(index + 1, widget)
        self._reveal()

    def add_trailing(self, widget: QWidget) -> None:
        """加在設定列**最右邊**(伸縮元件之後)。

        開關放右邊而不是跟其他設定排在一起:它決定「這一頁有沒有在運作」,
        與「候選列幾個」不是同一個層級的東西。混在一排裡會讓人以為它也只是
        一個顯示選項。
        """
        self._settings_layout.addWidget(widget)
        self._reveal()

    def _reveal(self) -> None:
        self._settings.setVisible(True)
        self._rule.setVisible(True)


class PanelWindow(QMainWindow):
    """輔助資訊的側邊視窗。

    Args:
        viewmodel: 狀態來源。視窗自己訂閱它,不主動去拉。
        skin: 牌面素材。
        always_on_top: 是否置頂。
        switchboard: 兩個功能的開關要接到誰。``None``(錄影重播、示範資料)時
            開關仍然畫出來,但是**停用**並顯示為開啟 —— 那些模式裡是命令列
            決定跑哪一條路,把開關做成可按的會讓人以為按了有效。

    Note:
        :meth:`apply` 是唯一的資料入口。ViewModel 的通知可能來自別的執行緒,
        呼叫端要負責把它切回主執行緒 —— Qt 的 widget 只能在建立它的執行緒上動。
        ``tools/ui.py`` 用 ``QTimer`` 做這件事。
    """

    def __init__(
        self,
        viewmodel: ViewModel,
        *,
        skin: str = DEFAULT_SKIN,
        always_on_top: bool = True,
        switchboard: Switchboard | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        # window flag 必須在 show() **之前**設定。show 之後再改,Qt 會把視窗
        # 隱藏並要求重新 show() —— 表現出來是「Dock 有圖示但畫面上沒有視窗」。
        if always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._viewmodel = viewmodel
        self._switchboard = switchboard
        self._switches: dict[str, ToggleSwitch] = {}
        icons = TileIcons(skin)
        self._advice = AdviceTab(icons, self)
        self._analysis = AnalysisTab(icons, self)

        self._nav = QListWidget(self)
        self._nav.setFixedWidth(NAV_WIDTH)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setStyleSheet(
            "QListWidget { background: palette(window); }"
            "QListWidget::item { padding: 10px 6px; }"
            "QListWidget::item:selected { background: palette(highlight);"
            " color: palette(highlighted-text); }"
        )
        self._stack = QStackedWidget(self)

        self._advice_page = self._add_page("AI 建議", self._advice)
        self._analysis_page = self._add_page("向聽分析", self._analysis)
        self._settings_page = self._add_page("設定", self._build_settings())
        self._build_advice_settings()
        # 開關最後加,才會排在設定列最右邊
        self._add_switch(self._advice_page, features.ADVICE)
        self._add_switch(self._analysis_page, features.VISION)

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav)
        layout.addWidget(_rule(central, vertical=True))
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._notice = QLabel("", self)
        self._notice.setStyleSheet("color: palette(mid);")
        status = QStatusBar(self)
        status.addWidget(self._notice, 1)
        self.setStatusBar(status)

        self.apply(viewmodel.state)

    # ------------------------------------------------------------------ 建構

    def _add_page(self, name: str, content: QWidget) -> _Page:
        page = _Page(content, self._stack)
        self._stack.addWidget(page)
        self._nav.addItem(QListWidgetItem(name))
        return page

    def _add_switch(self, page: _Page, key: str) -> None:
        """在一頁的設定列右邊放一個開關。

        兩個功能**預設都關著**,要使用者自己打開。理由不只是省資源:打開
        AI 建議會開一個載著 130MB 權重的子程序、打開畫面辨識會開始持續擷取
        螢幕。這兩件事都該是明確的動作,不是打開視窗的副作用。
        """
        switch = ToggleSwitch(page)
        switch.setToolTip(f"{features.NAMES[key]} —— 開啟後才會開始運作")
        if self._switchboard is None:
            # 錄影重播沒有可開關的東西:命令列已經決定了跑哪一條路。
            # 顯示為開啟(那條路真的在跑)但停用,免得使用者以為按了有效。
            switch.set_initial(True)
            switch.setEnabled(False)
            switch.setToolTip("重播與示範模式由命令列決定,開關不適用")
        elif not self._switchboard.available(key):
            # --no-vision / --no-packets:這條路根本沒有建起來。做成可按的話
            # 使用者撥了它會自己彈回去,那看起來像壞掉,而實際上是他自己關的。
            switch.setEnabled(False)
            switch.setToolTip(f"{features.NAMES[key]} 已在啟動參數中停用")
        else:
            switch.set_initial(self._switchboard.is_enabled(key))
            switch.toggled.connect(lambda on, k=key: self._on_switch(k, on))

        self._switches[key] = switch
        page.add_trailing(switch)

    def _build_advice_settings(self) -> None:
        """AI 建議頁的設定列。

        「主要引擎」不只是方便:預設取第一個有動作的引擎,而規則式 baseline
        若排在模型前面,headline 就會顯示 baseline、Q 值長條整段消失。
        下拉選單讓使用者直接指定,不必去改啟動參數。
        """
        self._engine_picker = QComboBox(self._advice_page)
        self._engine_picker.setMinimumWidth(140)
        self._engine_picker.addItem("(自動)", None)
        self._engine_picker.currentIndexChanged.connect(self._on_engine_picked)
        self._advice_page.add_setting("主要引擎", self._engine_picker)

        self._candidate_count = QSpinBox(self._advice_page)
        self._candidate_count.setRange(1, 20)
        self._candidate_count.setValue(self._advice.max_candidates)
        self._candidate_count.valueChanged.connect(self._on_candidate_count)
        self._advice_page.add_setting("候選數", self._candidate_count)

    def _build_settings(self) -> QWidget:
        """全域設定頁。

        刻意只放**真的有作用**的東西。擺一排還沒接線的控制項會讓人以為調了有效,
        那比沒有設定頁更糟。
        """
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._on_top = QCheckBox("視窗置頂", page)
        self._on_top.setChecked(bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        self._on_top.toggled.connect(self._on_always_on_top)
        layout.addWidget(self._on_top)

        hint = QLabel(
            "置頂關掉之後視窗會退到遊戲後面。macOS 上遊戲視窗被遮住仍然擷取得到,"
            "所以擋住畫面不影響辨識。",
            page,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(hint)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ 事件

    def _on_switch(self, key: str, on: bool) -> None:
        """使用者撥了開關。

        撥回實際狀態而不是使用者要的狀態:開啟可能失敗(權重不見了、
        找不到遊戲視窗),那時開關要自己彈回去,不能停在「開」而底下沒東西在跑。
        """
        assert self._switchboard is not None
        self._switchboard.set_enabled(key, on)
        actual = self._switchboard.is_enabled(key)
        if actual != on:
            self._switches[key].set_initial(actual)
        self.apply(self._viewmodel.state)

    def _on_engine_picked(self, index: int) -> None:
        self._viewmodel.set_preferred_engine(self._engine_picker.itemData(index))

    def _on_candidate_count(self, value: int) -> None:
        self._advice.max_candidates = value
        self._advice.update_from(self._viewmodel.state)

    def _on_always_on_top(self, enabled: bool) -> None:
        """切換置頂。

        改 window flag 會讓已顯示的視窗被隱藏,所以要重新 show 一次 ——
        這正是先前「Dock 有圖示但沒有視窗」那個 bug 的成因。
        """
        was_visible = not self.isHidden()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()

    def show_page(self, index: int) -> None:
        """切到第幾個功能頁。給測試與截圖用。"""
        self._nav.setCurrentRow(index)

    # ------------------------------------------------------------------ 資料

    def apply(self, state: ViewState) -> None:
        """把狀態畫上去。

        每一頁都更新,不只是當前可見的那個 —— 切頁時才更新會讓使用者看到
        一瞬間的舊資料。更新一次的成本是十幾個 label,遠低於那個閃動的代價。
        """
        self._sync_engine_picker(state)
        self._advice.update_from(state, enabled=self._is_on(features.ADVICE))
        self._analysis.update_from(state, enabled=self._is_on(features.VISION))
        self._notice.setText(self._status_text(state))

    def _is_on(self, key: str) -> bool:
        """這個功能現在該不該顯示內容。

        沒有 switchboard(重播、示範)時一律視為開著 —— 那些模式裡是命令列
        決定跑哪一條路,而它確實在跑。
        """
        if self._switchboard is None:
            return True
        return self._switchboard.is_enabled(key)

    def _status_text(self, state: ViewState) -> str:
        """狀態列。兩個功能都關著時要**主動說**,不然畫面上只有一片空白。"""
        if self._switchboard is not None and not any(
            self._switchboard.is_enabled(k) for k in (features.ADVICE, features.VISION)
        ):
            return "兩個功能都關著 —— 用各頁右上角的開關打開"
        return _status_text(state)

    def _sync_engine_picker(self, state: ViewState) -> None:
        """引擎清單是跑起來才知道的,所以下拉選單要跟著狀態長出來。"""
        names = [e.name for e in state.engines]
        existing = [self._engine_picker.itemData(i) for i in range(1, self._engine_picker.count())]
        if names == existing:
            return

        # 重建期間要擋掉 currentIndexChanged,否則清空的瞬間會把偏好設成 None
        self._engine_picker.blockSignals(True)
        while self._engine_picker.count() > 1:
            self._engine_picker.removeItem(1)
        for name in names:
            self._engine_picker.addItem(name, name)
        wanted = state.preferred_engine
        self._engine_picker.setCurrentIndex(
            names.index(wanted) + 1 if wanted in names else 0
        )
        self._engine_picker.blockSignals(False)


def _status_text(state: ViewState) -> str:
    parts: list[str] = []
    if state.hand_conflict:
        parts.append("⚠ 畫面與封包的手牌不一致")
    parts.extend(state.notices)
    if not parts:
        engines = "、".join(e.name for e in state.engines)
        return f"引擎:{engines}" if engines else "等待資料…"
    return "　|　".join(parts)


def _rule(parent: QWidget, *, vertical: bool = False) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def present(window: QMainWindow) -> None:
    """顯示視窗並放到螢幕右側 —— 遊戲通常在中間或左邊。

    順序有意義:**先移動、再 show**。反過來的話,任何在 show 之後動到 window
    flag 的操作都會讓視窗被 Qt 隱藏,而 Dock 上仍然看得到圖示 —— 那個症狀
    完全不像「視窗被藏起來了」,很難聯想到是 flag 的關係。

    :class:`PanelWindow` 已經在建構時設好置頂,這裡只負責位置與顯示。
    """
    screen = window.screen()
    if screen is not None:
        available = screen.availableGeometry()
        size = window.sizeHint().expandedTo(window.minimumSize())
        window.move(
            available.right() - size.width() - 24,
            available.top() + max(0, (available.height() - size.height()) // 2),
        )
    window.show()
    window.raise_()
    window.activateWindow()

