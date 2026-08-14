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

from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mia import APP_TITLE, features
from mia.calibration.canvas import PRESETS as CANVAS_PRESETS
from mia.calibration.canvas import Canvas
from mia.ui.style import CAPTION, MUTED, MUTED_COLOR
from mia.ui.switchboard import CanvasPicker, GameLauncher, Switchboard, is_on
from mia.ui.viewmodel import ViewModel, ViewState
from mia.ui.widgets.advice import AdviceTab
from mia.ui.widgets.analysis import AnalysisTab
from mia.ui.widgets.danger import AnalysisDangerTab
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons
from mia.ui.widgets.toggle import ToggleSwitch

if TYPE_CHECKING:
    from mia.ui.overlay.window import OverlayWindow

#: ``Switchboard`` 原本定義在這裡,現在搬到 :mod:`mia.ui.switchboard`
#: —— Overlay 也要用它,而那兩個是平行的呈現方式,誰都不該相依誰。
__all__ = ["PanelWindow", "Switchboard", "present"]

#: 放得下 14 張 44px 高的牌、加上左側功能列。
MIN_WIDTH = 500
MIN_HEIGHT = 540

#: 左側功能列的寬度。夠放四個全形字。
NAV_WIDTH = 96

#: 「開始遊戲」按鈕的字。兩處(按鈕與狀態列提示)都取這裡,不會寫得不一樣。
START_GAME = "開始遊戲"


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
        caption.setStyleSheet(CAPTION)
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
        launcher: 「開始遊戲」要接到誰。``None``(重播、``--tail``、
            ``--no-packets``)時按鈕畫成停用 —— 那些模式裡遊戲不是 MIA 開的。
        canvas: 畫布尺寸選單要接到誰。``None`` 時選單畫成停用。
        overlay: 要控制的 Overlay。設定頁上的三個勾選都作用在它身上;
            ``None`` 時那三個畫成停用。

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
        launcher: GameLauncher | None = None,
        canvas: CanvasPicker | None = None,
        overlay: OverlayWindow | None = None,
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
        self._launcher = launcher
        self._canvas = canvas
        self._overlay = overlay
        self._switches: dict[str, ToggleSwitch] = {}
        icons = TileIcons(skin)
        self._advice = AdviceTab(icons, self)
        self._analysis = AnalysisTab(icons, self)
        self._danger = AnalysisDangerTab(icons, self)

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
        self._danger_page = self._add_page("放銃分析", self._danger)
        self._settings_page = self._add_page("設定", self._build_settings())
        self._build_advice_settings()
        # 開關最後加,才會排在設定列最右邊
        self._add_switch(self._advice_page, features.ADVICE)
        self._add_switch(self._analysis_page, features.VISION)
        self._add_switch(self._danger_page, features.DANGER)

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_nav_column(central))
        layout.addWidget(_rule(central, vertical=True))
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._notice = QLabel("", self)
        self._notice.setStyleSheet(MUTED)
        status = QStatusBar(self)
        status.addWidget(self._notice, 1)
        self.setStatusBar(status)

        self.apply(viewmodel.state)

    # ------------------------------------------------------------------ 建構

    def _build_nav_column(self, parent: QWidget) -> QWidget:
        """左側功能列 + 底下的「開始遊戲」。

        按鈕放在**功能列最下面**而不是某一頁裡面:它與「現在看哪一頁」無關,
        而且是使用者每次坐下來要做的第一件事,不該藏在某一頁後面。這個位置
        與 MAA 的「開始」一致。
        """
        column = QWidget(parent)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav, 1)
        layout.addWidget(self._build_start_button(column))
        column.setFixedWidth(NAV_WIDTH)
        return column

    def _build_start_button(self, parent: QWidget) -> QWidget:
        """開瀏覽器並開始錄封包。

        瀏覽器**不會自己開**,理由與兩個功能預設關閉相同:開一個瀏覽器並開始
        往磁碟寫錄影檔該是明確的動作。實際做事的是
        :meth:`~mia.live.runtime.LiveRuntime.start_game`。
        """
        button = QPushButton(START_GAME, parent)
        button.setStyleSheet(
            "QPushButton { background: #34C759; color: white; border: none;"
            " border-radius: 6px; padding: 9px 4px; font-weight: 600; margin: 6px; }"
            "QPushButton:hover:enabled { background: #2FB350; }"
            "QPushButton:disabled { background: palette(midlight);"
            f" color: {MUTED_COLOR}; }}"
        )
        button.clicked.connect(self._on_start_game)
        self._start_button = button
        return button

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

    def _build_canvas_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """遊戲視窗尺寸。

        **放在全域設定頁而不是向聽分析頁。** 它同時做兩件事 —— 調整瀏覽器
        視窗、決定牌桌校正的基準 —— 而使用者心裡它就是「遊戲視窗要開多大」,
        那是一個應用程式層級的設定,不是某一頁的顯示選項。

        為什麼需要它,見 :mod:`mia.calibration.canvas`:自動偵測是啟發式,
        實測在某些視窗尺寸下會安靜地給出偏掉 0.7 張牌寬的矩形,手牌辨識從
        96% 掉到 29%。指定一個尺寸之後 MIA 直接把瀏覽器調成那樣,牌桌矩形
        就用算的,不必猜。
        """
        row = QWidget(page)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        # 可編輯:常用的尺寸放下拉,其餘直接打。**沒有「自動偵測」這個選項** ——
        # 那條路是猜的,實測會安靜地鎖進偏掉 0.7 張牌寬的矩形。它仍然存在
        # (螢幕小到放不下任何尺寸時的退路、--connect 模式),但不該擺在選單上
        # 邀請使用者去選一個已知比較差的做法。
        picker = QComboBox(row)
        picker.setMinimumWidth(150)
        picker.setEditable(True)
        picker.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for preset in CANVAS_PRESETS:
            picker.addItem(preset.label, preset.key)

        self._canvas_picker = picker
        if self._canvas is None or not self._canvas.can_pick_canvas():
            picker.setEnabled(False)
            picker.setToolTip(
                "這個模式下視窗尺寸不是 MIA 決定的(重播、--no-vision、或連到別人的瀏覽器)"
            )
        else:
            self._show_canvas(self._canvas.canvas())
            picker.setToolTip("可以直接輸入,格式是 1440x810")
            picker.currentIndexChanged.connect(self._on_canvas_picked)
            # 打字要等**輸入結束**才套用,不能用 currentTextChanged ——
            # 那個每按一個鍵就觸發一次,「1440x810」打到一半的「14」會先
            # 被當成一個尺寸送出去,瀏覽器就跟著閃。
            edit = picker.lineEdit()
            if edit is not None:
                edit.editingFinished.connect(self._on_canvas_typed)
        row_layout.addWidget(picker)
        row_layout.addStretch(1)
        layout.addWidget(row)

        hint = QLabel(
            "這是「遊戲畫布」的大小,瀏覽器視窗會再高一點(多出工具列那一截)。"
            "選了之後瀏覽器會立刻跟著調整,向聽分析的牌桌校正也直接用這個尺寸算,"
            "不再靠猜邊界 —— 猜錯過一次就是整手牌都認錯。"
            "第一次啟動時程式會自己挑螢幕放得下的最大值,你動過之後就照你選的。",
            page,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)

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
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)

        layout.addWidget(_rule(page))
        layout.addWidget(_caption("遊戲視窗尺寸", page))
        self._build_canvas_settings(page, layout)

        layout.addWidget(_rule(page))
        layout.addWidget(_caption("Overlay", page))
        self._build_overlay_settings(page, layout)

        layout.addStretch(1)
        return page

    def _build_overlay_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """Overlay 的三個勾選。

        **為什麼展開與鎖定的開關在這裡,而不在 Overlay 上。** Overlay 鎖定之後
        對滑鼠是透明的 —— 擺在它自己身上的按鈕會變成死的,而使用者看得到卻按不到
        的按鈕比沒有按鈕糟。所以控制項一律留在側邊視窗這一側。

        Overlay 也**不進 features.py 的開關體系**:那兩個開關管的是「要不要花
        130MB 載權重 / 要不要持續擷取螢幕」,而這裡只是換一種畫法,沒有任何
        執行成本。混在一起會讓「開了但沒東西跑」變得難解釋。
        """
        overlay = self._overlay
        self._overlay_shown = QCheckBox("在遊戲上顯示 Overlay", page)
        self._overlay_expanded = QCheckBox("Overlay 顯示候選 Q 值", page)
        self._overlay_locked = QCheckBox("鎖定位置(滑鼠可穿透)", page)
        boxes = (self._overlay_shown, self._overlay_expanded, self._overlay_locked)

        if overlay is None:
            for box in boxes:
                box.setEnabled(False)
                box.setToolTip("這個模式沒有 Overlay")
                layout.addWidget(box)
            return

        # 先把初始值設好再接訊號 —— 反過來的話 setChecked 會立刻觸發一次
        # handler,而那一次會把「上次記住的狀態」當成使用者剛剛的操作存回去
        self._overlay_shown.setChecked(overlay.shown)
        self._overlay_expanded.setChecked(overlay.expanded)
        self._overlay_locked.setChecked(overlay.locked)
        self._overlay_shown.toggled.connect(self._on_overlay_shown)
        self._overlay_expanded.toggled.connect(overlay.set_expanded)
        self._overlay_locked.toggled.connect(overlay.set_locked)
        for box in boxes:
            layout.addWidget(box)

        hint = QLabel(
            "沒鎖定時 Overlay 可以直接拖曳,鎖定之後點擊會穿過去給遊戲。"
            "位置會記住,下次啟動回到同一個地方。",
            page,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)
        self._sync_overlay_boxes(overlay.shown)

    def _on_overlay_shown(self, on: bool) -> None:
        assert self._overlay is not None
        self._overlay.set_shown(on)
        self._sync_overlay_boxes(on)

    def _sync_overlay_boxes(self, shown: bool) -> None:
        """收起來的時候「展開」與「鎖定」沒有意義,停用但保留勾選狀態。

        不清掉勾選:那是使用者的偏好,下次顯示時該照舊。
        """
        self._overlay_expanded.setEnabled(shown)
        self._overlay_locked.setEnabled(shown)

    def closeEvent(self, event: QCloseEvent) -> None:
        """關掉側邊視窗時把 Overlay 一起收掉。

        少了這一段,顯示中的 Overlay 會是「最後一個還開著的視窗」,Qt 於是
        不結束程式 —— 畫面上只剩一個關不掉的浮動 HUD,而終端機看起來一切正常。
        """
        if self._overlay is not None:
            self._overlay.close()
        super().closeEvent(event)

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

    def _on_start_game(self) -> None:
        """按下「開始遊戲」。

        按完立刻重畫一次,不等 ViewModel 的下一次通知 —— 開瀏覽器要好幾秒,
        中間按鈕若還停在可按的樣子,使用者會再按一次。狀態列上那句
        「還沒開始」也要跟著換掉。
        """
        assert self._launcher is not None
        self._launcher.start_game()
        self.apply(self._viewmodel.state)

    def _sync_start_button(self) -> None:
        """按鈕要說現在的實情。

        三種狀態:沒有可開的遊戲(重播 / ``--tail`` / ``--no-packets``)、
        可以開、已經在跑。**已經在跑時停用而不是換成「結束遊戲」** —— 關掉
        瀏覽器等於把那一場丟掉,不該是一個手滑就會按到的按鈕。
        """
        launcher = self._launcher
        if launcher is None or not launcher.can_start_game():
            self._start_button.setText(START_GAME)
            self._start_button.setEnabled(False)
            self._start_button.setToolTip(
                "重播與示範模式不需要開遊戲"
                if launcher is None
                else "啟動參數裡沒有要 MIA 開遊戲(--tail / --no-packets)"
            )
            return
        running = launcher.game_running()
        self._start_button.setText("遊戲進行中" if running else START_GAME)
        self._start_button.setEnabled(not running)
        self._start_button.setToolTip(
            "瀏覽器已經開著 —— 關掉它就會結束這一場"
            if running
            else "開一個受控的瀏覽器並開始錄封包"
        )

    def _show_canvas(self, key: str | None) -> None:
        """把選單顯示成這個尺寸,**不觸發任何 handler**。

        選單是可編輯的,所以「顯示什麼」是文字而不是索引 —— 使用者手打的
        尺寸根本不在清單裡。擋掉訊號是必要的:setCurrentText 會讓
        currentIndexChanged 觸發,而那個 handler 又會回頭呼叫這裡。
        """
        canvas = Canvas.parse(key)
        blocker = QSignalBlocker(self._canvas_picker)
        self._canvas_picker.setCurrentText(canvas.label if canvas else "")
        del blocker

    def _on_canvas_typed(self) -> None:
        """使用者自己打了一個尺寸。看不懂就退回原本的值。

        看不懂時**不能留著那串字** —— 選單上顯示 1440x81o 而實際跑的是
        1280x720,那是最糟的一種:畫面說一套、程式做另一套。
        """
        if self._canvas is None:
            return
        text = self._canvas_picker.currentText()
        canvas = Canvas.parse(text)
        if canvas is None:
            self._show_canvas(self._canvas.canvas())
            self._notice.setText(f"看不懂的尺寸「{text}」—— 格式是 1440x810")
            return
        self._canvas.set_canvas(canvas.key)
        self._show_canvas(canvas.key)  # 正規化成 1440×810
        self.apply(self._viewmodel.state)

    def _on_canvas_picked(self, index: int) -> None:
        """改選遊戲視窗尺寸。

        兩件事同時發生:**瀏覽器當場被調成新尺寸**(透過控制檔送給擷取子程序,
        見 :mod:`mia.live.control`),以及**畫面辨識重開**去用新的校正基準。
        中間有最多一個輪詢週期(250 ms)的空窗,那段時間狀態列會說「畫布
        對不上」—— 那是實話,而且它會自己好。
        """
        if self._canvas is None:
            return
        key = self._canvas_picker.itemData(index)
        self._canvas.set_canvas(key)
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
        self._sync_start_button()
        self._advice.update_from(state, enabled=self._is_on(features.ADVICE))
        self._analysis.update_from(state, enabled=self._is_on(features.VISION))
        self._danger.update_from(state, enabled=self._is_on(features.DANGER))
        self._notice.setText(self._status_text(state))

    def _is_on(self, key: str) -> bool:
        return is_on(self._switchboard, key)

    def _status_text(self, state: ViewState) -> str:
        """狀態列。什麼都還沒開始時要**主動說**,不然畫面上只有一片空白。

        順序是刻意的:遊戲還沒開的時候,先講那個。打開了功能卻沒有遊戲可看,
        使用者會盯著一片空白等 —— 而該做的事其實是按左下角那顆按鈕。
        """
        if (
            self._launcher is not None
            and self._launcher.can_start_game()
            and not self._launcher.game_running()
        ):
            return f"還沒開始 —— 按左下角的「{START_GAME}」開啟遊戲"
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


def _select_data(picker: QComboBox, data: object) -> None:
    """把選單切到 ``itemData`` 等於 ``data`` 的那一項。找不到就留在原地。

    用 ``findData`` 而不是記索引:選項順序是 :data:`CANVAS_PRESETS` 決定的,
    日後加一個尺寸就會讓寫死的索引指到別的地方。
    """
    index = picker.findData(data)
    if index >= 0:
        picker.setCurrentIndex(index)


def _caption(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setStyleSheet(CAPTION)
    return label


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

