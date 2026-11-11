"""側邊視窗:左側功能列 + 每頁「主頁 | 設定」兩個子頁 + 狀態列。

```
┌──────┬──────────────────────────┐
│ MIA  │ [主頁] 設定          ( ●) │  ← 子頁切換 + 這一頁的開關
│ AI   ├──────────────────────────┤
│ 建議 │                          │
│ 向聽 │  主頁:內容               │
│ 分析 │  設定:只屬於這個功能的選項│
│ 放銃 │                          │
│ 分析 │                          │
│ 一般 │                          │
│ 設定 │                          │
└──────┴──────────────────────────┘
│ 狀態列                           │
```

**左側直排而不是頂端分頁。** 功能只會越加越多(Overlay、風格比較、錄製),
頂端分頁一多就開始擠成兩排或出現左右箭頭;直排往下長不會擠到內容,而且每頁
的頂端就空出來放那一頁自己的子頁切換。這個排法參考 MAA。

**設定跟著功能走。** 每個功能頁有自己的「設定」子頁,一般設定頁只放不屬於
任何一個功能的東西。見 :class:`_Page`。

分頁而不是把所有內容塞一頁的理由沒有變:功能 1(向聽/進張)與功能 2(AI 建議)
是兩條獨立的路 —— 前者不碰連線,後者不碰畫面。擠在一起會讓「只有一邊在運作」
很難表達,而那是實際使用時的常態。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QPoint, QRectF, QSignalBlocker, Qt
from PySide6.QtGui import QCloseEvent, QEnterEvent, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from mia import APP_ABBR, APP_TITLE, features, platforms
from mia.calibration.canvas import PRESETS as CANVAS_PRESETS
from mia.calibration.canvas import Canvas
from mia.i18n import LANGUAGES, tr, translate
from mia.ui.icon import app_icon
from mia.ui.state import MIN_HUD_OPACITY
from mia.ui.style import (
    CAPTION,
    MUTED,
    Hairline,
    mix,
)
from mia.ui.switchboard import (
    AdviceSourcePicker,
    AppPreferences,
    CanvasPicker,
    DangerSourcePicker,
    GameLauncher,
    HandRoiPicker,
    NoticeMemory,
    PlatformPicker,
    RecognitionPicker,
    StylePicker,
    Switchboard,
    TableRectPicker,
    is_on,
)
from mia.ui.theme import (
    COMBO_WIDTH,
    AlignedComboBox,
    ChoiceButtons,
    controls_stylesheet,
    nav_stylesheet,
    polish_combo,
    segmented_stylesheet,
    start_button_stylesheet,
)
from mia.ui.viewmodel import ViewModel, ViewState
from mia.ui.widgets.advice import AdviceTab
from mia.ui.widgets.analysis import AnalysisTab
from mia.ui.widgets.danger import AnalysisDangerTab
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons
from mia.ui.widgets.toggle import ToggleSwitch

if TYPE_CHECKING:
    from mia.ui.overlay.log import EventLogWindow
from mia.ui.overlay.window import OverlayWindow

#: ``Switchboard`` 原本定義在這裡,現在搬到 :mod:`mia.ui.switchboard`
#: —— Overlay 也要用它,而那兩個是平行的呈現方式,誰都不該相依誰。
__all__ = ["PanelWindow", "Switchboard", "present"]

#: 「比較對象」選單上代表「不比較」的那一列。
#:
#: 寫成「不比較」而不是留白或「(無)」:這個選單控制的是**要不要多載一份
#: 130MB 的權重**,而留白看起來像還沒載入完。
NO_COMPARE = tr("不比較")

#: 放得下 14 張 44px 高的牌、加上左側功能列。
MIN_WIDTH = 500
MIN_HEIGHT = 540

#: 左側功能列的寬度。夠放四個全形字。
NAV_WIDTH = 96

#: 「開始遊戲」按鈕的字。兩處(按鈕與狀態列提示)都取這裡,不會寫得不一樣。
START_GAME = tr("開始遊戲")


def _scrollable(content: QWidget) -> QScrollArea:
    """內容比視窗高時捲動,不要把整個視窗的最小高度撐大:那會連其他分頁一起變高。"""
    area = QScrollArea()
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    return area


#: 功能頁頂端的兩個子頁。
SUBPAGES = (tr("主頁"), tr("設定"))


#: 會用到 YOLO 的辨識方式。混合辨識也算:它的摸牌一樣靠 YOLO 那一半。
_YOLO_BACKENDS = ("yolo", "hybrid")

#: 「YOLO 要用預設皮膚」那個提醒在 :class:`~mia.ui.switchboard.NoticeMemory` 裡的鍵。
YOLO_SKIN_NOTICE = "yolo_skin"


def yolo_skin_box(parent: QWidget | None) -> QMessageBox:
    """「YOLO 要用預設皮膚」的提醒框,附「不再顯示」勾選。

    分出來是為了測試:測試要檢查框裡有什麼,但不能真的 ``exec()``(會卡住)。
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("YOLO 需要預設皮膚"))
    box.setText(tr("使用 YOLO 前，請把雀魂的牌桌和牌背都換回預設皮膚。"))
    box.setInformativeText(
        tr("YOLO 的模型是用預設皮膚的畫面訓練的。換了別的牌桌或牌背，"
        "它會認不出你摸到的牌，事件流卡住，AI 建議也出不來。\n"
        "網頁版和 Steam 版都一樣。")
    )
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setCheckBox(QCheckBox(tr("不再顯示"), box))
    return box


def _ask_yolo_skin(parent: QWidget) -> bool:
    """跳出提醒框並等使用者按掉。回傳有沒有勾「不再顯示」。"""
    box = yolo_skin_box(parent)
    box.exec()
    check = box.checkBox()
    return check is not None and check.isChecked()


class _Segmented(QWidget):
    """「主頁 | 設定」那一排切換。

    選中的那一顆與左側功能列的藥丸同一個顏色 —— 兩者的意思都是「你現在在
    這裡」。樣式由 :meth:`PanelWindow._apply_theme` 套在這個容器上
    (:func:`mia.ui.theme.segmented_stylesheet`),顏色要從 palette 算。
    """

    def __init__(self, labels: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for index, label in enumerate(labels):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(button, index)
            layout.addWidget(button)
        self.group.button(0).setChecked(True)

    def select(self, index: int) -> None:
        button = self.group.button(index)
        if button is not None:
            button.click()


class _Page(QWidget):
    """一個功能頁:頂端「主頁 | 設定」+ 開關,底下是兩個子頁。

    ```
    ┌──────────────────────────────┐
    │ [主頁] 設定             ( ●) │  ← 開關:這一頁有沒有在運作
    ├──────────────────────────────┤
    │  主頁:這個功能的內容          │
    │  設定:只屬於這個功能的選項    │
    └──────────────────────────────┘
    ```

    **設定跟著功能走,不集中到一頁。** 以前「使用圖像辨識」放在全域設定頁、
    「風格」擠在 AI 建議頁頂端的一條橫列上 —— 想調 AI 建議得記得兩個地方。
    現在每個功能的東西都在自己那一頁的「設定」裡,一般設定頁只剩不屬於
    任何一個功能的東西(玩哪一版、視窗多大、Overlay、Log)。

    開關放在頂端而不是「主頁」裡:兩個子頁都看得到它。在設定子頁改完選項
    想立刻打開,不必再切回去。

    沒有任何設定時「設定」那一顆按鈕隱藏 —— 點進去一片空白會讓人以為
    有東西沒載出來。
    """

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QWidget(self)
        header = QHBoxLayout(self._header)
        # 左緣要與底下內容的 12px 對齊 —— 差幾個像素不會有人說得出哪裡怪,
        # 但整頁會看起來像沒對好。按鈕自己有 14px 的內距,所以這裡退 6。
        header.setContentsMargins(6, 8, 12, 8)
        header.setSpacing(8)
        self._tabs = _Segmented(SUBPAGES, self._header)
        self._tabs.group.button(1).setVisible(False)
        header.addWidget(self._tabs)
        header.addStretch(1)
        self._header_layout = header

        self._body = QStackedWidget(self)
        self._body.addWidget(content)
        self.settings = QWidget(self)
        self._form = QVBoxLayout(self.settings)
        self._form.setContentsMargins(12, 12, 12, 12)
        self._form.setSpacing(8)
        self._form.addStretch(1)
        self.settings_area = _scrollable(self.settings)
        self._body.addWidget(self.settings_area)
        self._tabs.group.idClicked.connect(self._body.setCurrentIndex)
        self._groups = 0

        layout.addWidget(self._header)
        layout.addWidget(_rule(self))
        layout.addWidget(self._body, 1)

    def add_group(self, title: str, *widgets: QWidget, info: str = "") -> None:
        """在設定子頁加一段:標題(旁邊的 ⓘ 帶說明)、控制項。段與段之間一條細線。

        說明**不直接印在控制項底下**:每段都多一行灰字,整頁一半是說明,
        要找的控制項反而被淹掉。收進 ⓘ 之後,想知道的人移過去就看得到。
        """
        index = self._form.count() - 1  # 伸縮元件之前
        items: list[QWidget] = []
        if self._groups:
            items.append(_rule(self.settings))
        items.append(_section(title, self.settings, info=info))
        items.extend(widgets)
        for offset, item in enumerate(items):
            self._form.insertWidget(index + offset, item)
        self._groups += 1
        self._tabs.group.button(1).setVisible(True)

    def add_trailing(self, widget: QWidget) -> None:
        """加在頂端**最右邊**。

        開關放右邊而不是跟子頁切換排在一起:它決定「這一頁有沒有在運作」,
        與「現在看哪一個子頁」不是同一個層級的東西。
        """
        self._header_layout.addWidget(widget)

    def has_settings(self) -> bool:
        return self._groups > 0

    def show_subpage(self, index: int) -> None:
        self._tabs.select(index)

    def subpage(self) -> int:
        return self._body.currentIndex()


#: 左邊導覽列的功能頁,依序。**頁數不等於功能數** —— 放銃分析有兩條路
#: (封包與純畫面)但只有一頁:它們算的是同一件事、投到同一個格子,
#: 各開一頁會變成兩頁長得一模一樣。頁是依「看到什麼」分的,開關依「哪條路」。
#:
#: 「一般設定」不在這裡 —— 它不是功能頁(沒有開關,也沒有主頁)。
PAGES = (tr("AI 建議"), tr("向聽分析"), tr("放銃分析"))

#: 功能列最後一項:不屬於任何一個功能的設定。叫「一般設定」而不是「設定」,
#: 免得跟每一頁頂端那顆「設定」搞混。
GENERAL = tr("一般設定")

#: 設定頁 AI 建議「使用圖像辨識」的說明。
_ADVICE_SOURCE_TIP = (
    tr("不勾:讀封包,最準。\n"
    "勾起來:只看畫面,不讀封包,但比較慢,偶爾會認錯。\n"
    "會重啟 AI 建議,並一併打開畫面辨識。\n"
    "遊戲還是 MIA 開的瀏覽器,封包照樣會錄,只是建議不看。")
)


class PanelWindow(QMainWindow):
    """輔助資訊的側邊視窗。

    Args:
        viewmodel: 狀態來源。視窗自己訂閱它,不主動去拉。
        skin: 牌面素材。
        always_on_top: 是否置頂。
        switchboard: 三個功能的開關要接到誰。``None``(錄影重播、示範資料)時
            開關仍然畫出來,但是**停用**並顯示為開啟 —— 那些模式裡是命令列
            決定跑哪一條路,把開關做成可按的會讓人以為按了有效。
        launcher: 「開始遊戲」要接到誰。``None``(重播、``--tail``、
            ``--no-packets``)時按鈕畫成停用 —— 那些模式裡遊戲不是 MIA 開的。
        canvas: 畫布尺寸選單要接到誰。``None`` 時選單畫成停用。
        style: 打法風格選單要接到誰。``None``(重播、``--no-packets``)時
            選單畫成停用 —— 那些模式裡引擎是命令列決定的。
        platform: 遊戲版本選單要接到誰。``None``(重播、示範)時畫成停用
            —— 那些模式裡「現在玩的是哪一版」是錄的時候就定了的。
        hand_roi: 「框選手牌範圍」要接到誰。``None`` 時按鈕畫成停用。
        table_rect: 「框選整個遊戲畫面」要接到誰。``None`` 時按鈕畫成停用。
        danger_source: 「強制使用圖像辨識」要接到誰。``None`` 時勾選框畫成停用。
        advice_source: 設定頁 AI 建議的「使用圖像辨識」要接到誰。``None`` 時畫成停用。
        event_log: Log 視窗。``None`` 時勾選框畫成停用。
        overlay: 要控制的 Overlay。設定頁上的四個勾選都作用在它身上;
            ``None`` 時那四個畫成停用。
        preferences: 「語言與更新」那一區要接到誰。``None`` 時整區畫成停用。
        notices: 記得哪些提醒勾了「不再顯示」。``None`` 時每次都跳(記不住)。

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
        style: StylePicker | None = None,
        platform: PlatformPicker | None = None,
        hand_roi: HandRoiPicker | None = None,
        table_rect: TableRectPicker | None = None,
        danger_source: DangerSourcePicker | None = None,
        advice_source: AdviceSourcePicker | None = None,
        recognition: RecognitionPicker | None = None,
        notices: NoticeMemory | None = None,
        event_log: EventLogWindow | None = None,
        overlay: OverlayWindow | None = None,
        preferences: AppPreferences | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        # Overlay 是獨立的 top-level 視窗,不會從這裡繼承圖示 —— 它自己也設。
        self.setWindowIcon(app_icon())
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        # window flag 必須在 show() **之前**設定。show 之後再改,Qt 會把視窗
        # 隱藏並要求重新 show() —— 表現出來是「Dock 有圖示但畫面上沒有視窗」。
        if always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._viewmodel = viewmodel
        self._switchboard = switchboard
        self._launcher = launcher
        self._canvas = canvas
        self._style = style
        self._platform = platform
        self._hand_roi = hand_roi
        self._table_rect = table_rect
        self._danger_source = danger_source
        self._advice_source = advice_source
        self._recognition = recognition
        self._notices = notices
        self._event_log = event_log
        self._overlay = overlay
        self._preferences = preferences
        self._switches: dict[str, ToggleSwitch] = {}
        self._skin = skin
        icons = TileIcons(skin)
        self._advice = AdviceTab(icons, self)
        self._analysis = AnalysisTab(icons, self)
        self._danger = AnalysisDangerTab(icons, self)

        self._nav = QListWidget(self)
        self._nav_width = self._nav_width_for((*PAGES, GENERAL, START_GAME))
        self._nav.setFixedWidth(self._nav_width)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        # 選中的那一項是一顆**圓角藥丸**,不是整條貫穿的色塊。色塊的兩端
        # 正好貼著視窗左緣與分隔線,看起來像那一列破了一個洞;留白 + 圓角
        # 才看得出「這是一個被選起來的項目」。這個畫法與 Finder 側邊欄一致。
        #
        # 顏色是**中性的灰**,不是 MIA 的綠也不是系統藍:綠在這個程式裡的意思
        # 是「這個東西正在運作」,藍是「這個勾選打開了」。「現在看哪一頁」
        # 不該借用任何一個 —— 樣式在 _apply_theme 設(見 mia.ui.theme)。
        self._stack = QStackedWidget(self)

        self._advice_page = self._add_page(PAGES[0], self._advice)
        self._analysis_page = self._add_page(PAGES[1], self._analysis)
        self._danger_page = self._add_page(PAGES[2], self._danger)
        # 一般設定**不是** _Page:它整頁就是設定,再掛一個「主頁 | 設定」
        # 會變成一個只有一邊有東西的切換。
        self._settings_page = _scrollable(self._build_settings())
        self._stack.addWidget(self._settings_page)
        self._nav.addItem(QListWidgetItem(GENERAL))
        self._build_advice_settings()
        self._build_analysis_settings()
        self._build_danger_settings()
        self._recognition_pickers: list[QComboBox] = []
        for page in (self._advice_page, self._analysis_page, self._danger_page):
            self._add_recognition_picker(page)
        #: 給測試用的那一個;三個永遠同值。
        self._recognition_picker = self._recognition_pickers[0]
        # 開關最後加,才會排在設定列最右邊
        self._add_switch(self._advice_page, features.ADVICE)
        self._add_switch(self._analysis_page, features.VISION)
        # 放銃分析有兩條路(封包與純畫面),但**只有一個開關** —— 走哪條
        # 由程式自己選。見 mia.features.DangerSourceChoice:使用者要的是
        # 「告訴我哪張牌危險」,不是「選一個資料來源」。
        self._add_switch(self._danger_page, features.DANGER)

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav_column(central))
        body.addWidget(_rule(central, vertical=True))
        body.addWidget(self._stack, 1)
        outer.addLayout(body, 1)
        outer.addWidget(_rule(central))

        # 狀態訊息放在視窗底部、**會換行**,一個來源一行。原本是 QStatusBar 上的一行字,
        # 幾個來源用「|」串起來 —— 使用者回報(2026-11-11)真正要看的那句(「牌桌校正失敗:
        # 畫布尺寸對不上…」)排在最後,被視窗寬度切掉,根本看不到。狀態訊息講的正是「為什麼
        # 沒有東西」,被切掉就等於沒講。
        self._notice = QLabel("", central)
        self._notice.setStyleSheet(MUTED)
        self._notice.setWordWrap(True)
        self._notice.setTextFormat(Qt.TextFormat.PlainText)
        self._notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._notice.setContentsMargins(10, 6, 10, 6)
        outer.addWidget(self._notice)
        self.setCentralWidget(central)

        self._apply_theme()
        # 一打開焦點會落在第一個收得到鍵盤的控制項 —— AI 建議頁的候選數框,
        # 於是那個框一直亮著藍色外框加游標,像是在等人輸入。放在功能列上,
        # 那裡沒有焦點框,方向鍵還能換頁。
        self._nav.setFocus()
        self.apply(viewmodel.state)

    # ------------------------------------------------------------------ 外觀

    def _themed(self) -> list[QWidget]:
        """套控制項樣式的容器。

        **只套在放控制項的地方**(各頁的設定子頁、一般設定、AI 建議的候選數),
        不套在整個視窗上:頂端的「主頁 | 設定」與左邊的功能列各有自己的樣式,
        上層的 QPushButton 規則裡它們沒寫到的屬性(例如邊框)會被一起繼承下去。
        """
        widgets: list[QWidget] = [self._settings_page, self._advice]
        for page in (self._advice_page, self._analysis_page, self._danger_page):
            widgets.append(page.settings_area)
        return widgets

    def _apply_theme(self) -> None:
        """照**現在的** palette 產生樣式並套上。系統換淺／深色時再呼叫一次。"""
        palette = QApplication.palette()
        sheet = controls_stylesheet(palette)
        for widget in self._themed():
            if widget.styleSheet() != sheet:
                widget.setStyleSheet(sheet)
        for combo in self.findChildren(QComboBox):
            polish_combo(combo)
        nav = nav_stylesheet(palette)
        if self._nav.styleSheet() != nav:
            self._nav.setStyleSheet(nav)
        tabs = segmented_stylesheet(palette)
        for segmented in self.findChildren(_Segmented):
            if segmented.styleSheet() != tabs:
                segmented.setStyleSheet(tabs)
        start = start_button_stylesheet(palette)
        if self._start_button.styleSheet() != start:
            self._start_button.setStyleSheet(start)

    def changeEvent(self, event: QEvent) -> None:
        """系統切換淺／深色 → 顏色重算。

        比對字串、一樣就不重套:setStyleSheet 本身會觸發 polish,
        而 polish 又可能送來 palette 變更,不擋的話會一直重套下去。
        """
        if event.type() in (QEvent.Type.ThemeChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_theme()
        super().changeEvent(event)

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
        layout.addWidget(_wordmark(column))
        layout.addWidget(self._nav, 1)
        layout.addWidget(self._build_start_button(column))
        column.setFixedWidth(self._nav_width)
        return column

    def _build_start_button(self, parent: QWidget) -> QWidget:
        """開瀏覽器並開始錄封包。

        瀏覽器**不會自己開**,理由與三個功能預設關閉相同:開一個瀏覽器並開始
        往磁碟寫錄影檔該是明確的動作。實際做事的是
        :meth:`~mia.live.runtime.LiveRuntime.start_game`。
        """
        button = QPushButton(START_GAME, parent)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        # 樣式在 _apply_theme 設:停用時的底色要從 palette 算。
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

        三個功能**預設都關著**,要使用者自己打開。理由不只是省資源:打開
        AI 建議會開一個載著 130MB 權重的子程序、打開畫面辨識會開始持續擷取
        螢幕。這兩件事都該是明確的動作,不是打開視窗的副作用。

        放銃分析幾乎沒有執行成本,預設仍然關著 —— 三個開關的行為一致比
        「只有這個不一樣」好解釋。
        """
        switch = ToggleSwitch(page)
        switch.setToolTip(tr("{value} —— 開啟後才會開始運作", value=features.NAMES[key]))
        if self._switchboard is None:
            # 錄影重播沒有可開關的東西:命令列已經決定了跑哪一條路。
            # 顯示為開啟(那條路真的在跑)但停用,免得使用者以為按了有效。
            switch.set_initial(True)
            switch.setEnabled(False)
            switch.setToolTip(tr("重播與示範模式由命令列決定,開關不適用"))
        elif not self._switchboard.available(key):
            # 兩種原因:這條路沒有建起來(--no-vision / --no-packets),或者
            # 這一版沒有資料來源(Steam 版沒有封包)。做成可按的話使用者撥了
            # 會自己彈回去,那看起來像壞掉,而實際上是有理由的 —— 所以理由
            # 一定要寫出來,而且兩種要說得不一樣:後者換個版本就有了。
            switch.setEnabled(False)
            switch.setToolTip(self._unavailable_reason(key))
        else:
            switch.set_initial(self._switchboard.is_enabled(key))
            switch.toggled.connect(lambda on, k=key: self._on_switch(k, on))

        self._switches[key] = switch
        page.add_trailing(switch)

    def _unavailable_reason(self, key: str) -> str:
        """開關為什麼撥不動。

        先問 switchboard —— :class:`~mia.live.runtime.LiveRuntime` 分得出
        「沒建起來」與「這一版沒有來源」。問不到(重播用的假物件)才退回
        自己從版本推一句,再不然就講啟動參數。
        """
        ask = getattr(self._switchboard, "unavailable_reason", None)
        if ask is not None:
            reason = ask(key)
            if reason:
                return str(reason)
        if self._platform is not None:
            why = platforms.why_unsupported(self._platform.platform(), key)
            if why:
                return why
        return tr("{value} 已在啟動參數中停用", value=features.NAMES[key])

    def _build_advice_settings(self) -> None:
        """AI 建議的設定子頁:風格、比較對象、資料來源。

        **曾經還有一個下拉「主要引擎」**,用來指定已經在跑的那幾個裡誰當
        headline。拿掉了:當初是因為規則式 baseline 若排在模型前面,headline
        就會顯示 baseline、Q 值長條整段消失 —— 而 ``tools/ui.py`` 的
        ``build_engines`` 現在一律把 baseline 接在最後,順序從源頭就是對的。
        留著一個下拉讓人去修一個不會發生的問題,等於把實作細節搬到使用者面前。

        順序:風格在前。它們是兩個層級 —— 風格決定**載哪幾份權重**
        (換了要重啟引擎),比較對象只是在那之上多載一份。
        """
        page = self._advice_page
        self._style_picker = AlignedComboBox(page.settings)
        self._style_picker.setFixedWidth(COMBO_WIDTH)
        page.add_group(
            tr("打法風格"),
            _row(page.settings, self._style_picker),
            info=tr("換風格會重啟引擎、重載權重,中間約半秒沒有建議。"),
        )
        self._compare_picker = AlignedComboBox(page.settings)
        self._compare_picker.setFixedWidth(COMBO_WIDTH)
        page.add_group(
            tr("比較對象"),
            _row(page.settings, self._compare_picker),
            info=tr("多載一份風格並排比較,推論成本與記憶體都是兩倍。"),
        )
        self._build_advice_source_settings(page)
        if self._style is None or not self._style.can_pick_style():
            self._style_picker.addItem(tr("(命令列指定)"))
            self._style_picker.setEnabled(False)
            self._compare_picker.addItem(NO_COMPARE)
            self._compare_picker.setEnabled(False)
        else:
            for name in self._style.styles():
                self._style_picker.addItem(translate(name), name)
            _select_data(self._style_picker, self._style.style())
            self._style_picker.currentIndexChanged.connect(self._on_style_picked)
            self._sync_compare_picker()
            self._compare_picker.currentIndexChanged.connect(self._on_compare_picked)

        # 「候選數」不在設定子頁,而在它控制的那段清單旁邊
        # (見 :class:`~mia.ui.widgets.advice.AdviceTab`)—— 它只是少畫幾列,
        # 邊看邊調才有意義。
        self._advice.candidate_count_changed.connect(self._on_candidate_count)

    def _build_advice_source_settings(self, page: _Page) -> None:
        """AI 建議要不要走圖像辨識。"""
        self._advice_vision_only = QCheckBox(tr("使用圖像辨識"), page.settings)
        self._sync_advice_source()
        self._advice_vision_only.toggled.connect(self._on_advice_source)
        page.add_group(
            tr("資料來源"),
            self._advice_vision_only,
            info=tr("不勾是讀封包(最準)。Steam 版沒有封包,一律走圖像辨識。"),
        )

    def _sync_advice_source(self) -> None:
        """勾選框要說現在的實情。換到 Steam 版時它變成打勾、不能改。

        擋掉訊號:這裡是把狀態畫出來,不是使用者按的,接著的話會白白重啟一次 AI 建議。
        """
        box = self._advice_vision_only
        picker = self._advice_source
        can_pick = picker is not None and picker.can_pick_advice_source()
        on_screen = picker is not None and picker.advice_uses_screen()
        box.blockSignals(True)
        box.setChecked(on_screen)
        box.blockSignals(False)
        box.setEnabled(can_pick)
        if can_pick:
            box.setToolTip(_ADVICE_SOURCE_TIP)
        elif on_screen:
            box.setToolTip(tr("Steam 版沒有封包,AI 建議只能靠圖像辨識"))
        else:
            box.setToolTip(tr("這個模式只有一條路,沒有得選"))

    def _on_advice_source(self, on: bool) -> None:
        """換來源。走畫面時畫面辨識會一併打開,所以開關要跟著重畫。"""
        assert self._advice_source is not None
        self._advice_source.set_advice_vision_only(on)
        self._sync_switches()
        self.apply(self._viewmodel.state)

    def _build_analysis_settings(self) -> None:
        """向聽分析的設定子頁:手動框選手牌範圍。

        **放在這一頁而不是一般設定。** 它只影響畫面辨識那條路,而且是對著
        這一頁的內容在調 —— 框完之後切回主頁當場就會變。一般設定放的是
        「整個程式的前提」(玩哪一版、遊戲視窗多大),不是某一條路的校正。

        「框選整個遊戲畫面」在**放銃分析**那一頁 —— 同一個理由:
        靠它最深的是牌河與寶牌,見 :meth:`_build_danger_settings`。
        """
        page = self._analysis_page
        self._hand_roi_button = QPushButton(tr("框選手牌範圍"), page.settings)
        self._hand_roi_button.clicked.connect(self._on_pick_hand_roi)
        self._hand_roi_clear = QPushButton(tr("改回自動"), page.settings)
        self._hand_roi_clear.clicked.connect(self._on_clear_hand_roi)
        if self._hand_roi is None or not self._hand_roi.can_pick_hand_roi():
            for button in (self._hand_roi_button, self._hand_roi_clear):
                button.setEnabled(False)
                button.setToolTip(tr("這個模式沒有畫面可以框(重播、示範、--no-vision)"))
        else:
            self._hand_roi_button.setToolTip(
                tr("在遊戲畫面上框出自己的手牌。Steam 版的畫布不是 16:9,\n"
                "選單裡的尺寸沒有一個對得上,框一次最直接。")
            )
            self._hand_roi_clear.setToolTip(tr("不用框的,回去用牌桌自動校正"))
        page.add_group(
            tr("手牌範圍"),
            _row(page.settings, self._hand_roi_button, self._hand_roi_clear),
            info=tr("手牌一直認錯、或 Steam 版畫布不是 16:9 時,直接框一次最穩。"),
        )
        self._sync_hand_roi_buttons()

    def _build_danger_settings(self) -> None:
        """放銃分析的設定子頁:資料來源(與共用的畫面辨識方式)。"""
        self._build_danger_source_settings(self._danger_page)

    def _build_table_rect_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """「框選整個遊戲畫面」。

        **放在一般設定,不放某一個功能頁。** 這個框是所有正規化座標的共同
        原點:四家牌河的四個角、寶牌面板的位置全部相對它推導(見
        mia.vision.tiles.river.DEFAULT_CORNERS),手牌沒自己框的時候也靠它。
        三個功能走畫面時都吃它,所以它跟「遊戲視窗尺寸」放在一起 ——
        兩個都是在回答「遊戲畫面在哪、多大」。

        曾經放在放銃分析頁(靠它最深的是牌河),但使用者去那裡找不到它。
        """
        self._table_rect_button = QPushButton(tr("框選整個遊戲畫面"), page)
        self._table_rect_button.clicked.connect(self._on_pick_table_rect)
        self._table_rect_clear = QPushButton(tr("改回自動"), page)
        self._table_rect_clear.clicked.connect(self._on_clear_table_rect)
        if self._table_rect is None or not self._table_rect.can_pick_table_rect():
            for button in (self._table_rect_button, self._table_rect_clear):
                button.setEnabled(False)
                button.setToolTip(tr("這個模式沒有畫面可以框(重播、示範、--no-vision)"))
        else:
            self._table_rect_button.setToolTip(
                tr("框出整個遊戲畫面 —— 藍色牌桌連同左右兩側的 UI。\n"
                "四家牌河與寶牌面板的位置都是相對它算的,所以純畫面的\n"
                "放銃分析最吃這個框。\n"
                "自動校正是從四邊剝除邊框,而它錯的時候很安靜 —— 實測鎖在\n"
                "載入畫面上之後,整場的手牌辨識從 97% 掉到 26%,沒有任何錯誤訊息。")
            )
            self._table_rect_clear.setToolTip(tr("不用框的,回去讓程式自己剝邊框"))
        layout.addWidget(
            _section(
                tr("遊戲畫面範圍"),
                page,
                info=tr("四家牌河、寶牌與手牌的位置都相對這個框推導。\n"
                "牌河或手牌一直讀不到時,框一次最直接。"),
            )
        )
        layout.addWidget(_row(page, self._table_rect_button, self._table_rect_clear))
        self._sync_table_rect_buttons()

    def _sync_hand_roi_buttons(self) -> None:
        """「改回自動」只有在真的框過的時候才有意義。"""
        picker = self._hand_roi
        if picker is None or not picker.can_pick_hand_roi():
            return
        framed = picker.hand_roi() is not None
        self._hand_roi_clear.setEnabled(framed)
        self._hand_roi_button.setText(tr("重新框選") if framed else tr("框選手牌範圍"))

    def _on_pick_hand_roi(self) -> None:
        """抓一張畫面,開框選對話框。

        抓畫面會失敗,而兩種失敗都是使用者救得回來的 —— 遊戲沒開、或 macOS
        沒給螢幕錄製權限。所以要接住並**說那一種**,不是讓例外竄上去把程式
        帶走:這是一個 Qt 的 slot,漏出去就是當場關窗。
        """
        from mia.capture.base import CaptureError, WindowNotFoundError

        from .handroi import HandRoiDialog

        picker = self._hand_roi
        if picker is None:
            return
        try:
            frame = picker.grab_frame()
        except WindowNotFoundError:
            self._notice.setText(tr("找不到遊戲視窗 —— 雀魂開著嗎?版本選對了嗎?"))
            return
        except CaptureError as exc:
            self._notice.setText(tr("抓不到畫面:{exc}", exc=exc))
            return

        dialog = HandRoiDialog(frame, skin=self._skin, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_rect is not None:
            picker.set_hand_roi(dialog.result_rect)
            self._sync_hand_roi_buttons()
            self._notice.setText(tr("手牌範圍已更新 —— 下一幀就會用新的框"))

    def _on_clear_hand_roi(self) -> None:
        if self._hand_roi is None:
            return
        self._hand_roi.set_hand_roi(None)
        self._sync_hand_roi_buttons()
        self._notice.setText(tr("手牌範圍改回自動校正"))

    def _sync_table_rect_buttons(self) -> None:
        """「改回自動」只有在真的框過的時候才有意義。"""
        picker = self._table_rect
        if picker is None or not picker.can_pick_table_rect():
            return
        framed = picker.table_rect() is not None
        self._table_rect_clear.setEnabled(framed)
        self._table_rect_button.setText(tr("重新框選") if framed else tr("框選整個遊戲畫面"))

    def _on_pick_table_rect(self) -> None:
        """抓一張畫面,開框選對話框。

        與 :meth:`_on_pick_hand_roi` 同一套錯誤處理,理由也一樣:這是 Qt 的
        slot,例外漏出去就是當場關窗,而兩種失敗都是使用者救得回來的。
        """
        from mia.capture.base import CaptureError, WindowNotFoundError
        from mia.config.loader import load_config

        from .tablerect import TableRectDialog

        picker = self._table_rect
        if picker is None:
            return
        try:
            frame = picker.grab_frame()
        except WindowNotFoundError:
            self._notice.setText(tr("找不到遊戲視窗 —— 雀魂開著嗎?版本選對了嗎?"))
            return
        except CaptureError as exc:
            self._notice.setText(tr("抓不到畫面:{exc}", exc=exc))
            return

        dialog = TableRectDialog(frame, roi=load_config().roi, skin=self._skin, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_rect is not None:
            picker.set_table_rect(dialog.result_rect)
            self._sync_table_rect_buttons()
            self._notice.setText(tr("遊戲畫面範圍已更新 —— 下一幀就會重新校正"))

    def _on_clear_table_rect(self) -> None:
        if self._table_rect is None:
            return
        self._table_rect.set_table_rect(None)
        self._sync_table_rect_buttons()
        self._notice.setText(tr("遊戲畫面範圍改回自動剝除邊框"))

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
        picker = AlignedComboBox(row)
        picker.setFixedWidth(COMBO_WIDTH)
        picker.setEditable(True)
        picker.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for preset in CANVAS_PRESETS:
            picker.addItem(preset.label, preset.key)

        self._canvas_picker = picker
        if self._canvas is None or not self._canvas.can_pick_canvas():
            picker.setEnabled(False)
            picker.setToolTip(
                tr("這個模式下視窗尺寸不是 MIA 決定的(重播、--no-vision、或連到別人的瀏覽器)")
            )
        else:
            self._show_canvas(self._canvas.canvas())
            picker.setToolTip(
                tr("這是「遊戲畫布」的大小,瀏覽器視窗會再高一點(多出工具列那一截)。\n"
                "選了之後瀏覽器立刻跟著調整,牌桌校正也直接用這個尺寸算,不再猜邊界\n"
                "—— 猜錯過一次就是整手牌都認錯。\n"
                "第一次啟動時程式會自己挑螢幕放得下的最大值,你動過之後就照你選的。\n"
                "可以直接輸入,格式是 1440x810。")
            )
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

    def _build_settings(self) -> QWidget:
        """一般設定頁:**不屬於任何一個功能**的東西。

        只屬於某個功能的設定(風格、框選範圍、資料來源)在那一頁自己的
        「設定」子頁裡,不放這裡 —— 見 :class:`_Page`。

        刻意只放**真的有作用**的東西。擺一排還沒接線的控制項會讓人以為調了有效,
        那比沒有設定頁更糟。
        """
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(_section(tr("遊戲版本"), page))
        self._build_platform_settings(page, layout)
        layout.addWidget(_rule(page))

        layout.addWidget(_section(tr("遊戲視窗尺寸"), page))
        self._build_canvas_settings(page, layout)

        layout.addWidget(_rule(page))
        self._build_table_rect_settings(page, layout)

        layout.addWidget(_rule(page))
        layout.addWidget(_section(tr("MIA 視窗"), page))
        self._on_top = QCheckBox(tr("視窗置頂"), page)
        self._on_top.setChecked(bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        self._on_top.setToolTip(
            tr("關掉之後視窗會退到遊戲後面。macOS 上遊戲視窗被遮住仍然擷取得到,\n"
            "所以擋住畫面不影響辨識。")
        )
        self._on_top.toggled.connect(self._on_always_on_top)
        layout.addWidget(self._on_top)

        layout.addWidget(_rule(page))
        layout.addWidget(_section("Overlay", page))
        self._build_overlay_settings(page, layout)
        self._build_opacity_settings(page, layout)

        layout.addWidget(_rule(page))
        layout.addWidget(_section(tr("事件流 Log"), page))
        self._build_log_settings(page, layout)

        layout.addWidget(_rule(page))
        layout.addWidget(_section(tr("語言與更新"), page))
        self._build_preferences(page, layout)

        layout.addStretch(1)
        return page

    def _nav_width_for(self, labels: tuple[str, ...]) -> int:
        """功能列要多寬。中文四個全形字放得下 :data:`NAV_WIDTH`;英文的頁名長得多
        (「Shanten analysis」),固定寬度會被截成「Shanten ...」,所以照字寬加大。"""
        metrics = self.fontMetrics()
        widest = max(metrics.horizontalAdvance(label) for label in labels)
        # 左右各留藥丸的內距與外距,與中文版的比例相同(96 放四個約 14px 的字)。
        return max(NAV_WIDTH, widest + 40)

    def _build_preferences(self, page: QWidget, layout: QVBoxLayout) -> None:
        """介面語言、目前版本、檢查更新。

        語言**重新啟動才生效**(見 :mod:`mia.i18n`),所以選了之後在下面說一句,並給一顆
        「重新啟動」:不給的話,使用者會以為選單壞了。選單裡每個語言用它自己的名字寫,
        選錯語言看不懂的時候還找得回來。
        """
        prefs = self._preferences
        picker = AlignedComboBox(page)
        picker.setFixedWidth(COMBO_WIDTH)
        for code, name in LANGUAGES.items():
            picker.addItem(name, code)
        self._language_picker = picker
        self._language_note = QLabel("", page)
        self._language_note.setStyleSheet(CAPTION)
        self._language_note.setWordWrap(True)
        self._language_note.hide()
        self._restart_button = QPushButton(tr("重新啟動 MIA"), page)
        self._version_label = QLabel("", page)
        self._version_label.setStyleSheet(CAPTION)
        self._update_button = QPushButton(tr("檢查更新…"), page)
        if prefs is None:
            for widget in (picker, self._update_button):
                widget.setEnabled(False)
            picker.setToolTip(tr("這個模式沒有可以儲存的設定"))
        else:
            self._chosen_language = prefs.language()
            _select_data(picker, self._chosen_language)
            picker.currentIndexChanged.connect(self._on_language_picked)
            self._version_label.setText(tr("目前版本 {value}", value=prefs.version()))
            self._update_button.clicked.connect(prefs.check_updates)
            self._restart_button.clicked.connect(self._on_restart)
        layout.addWidget(_row(page, _caption(tr("語言"), page), picker))
        layout.addWidget(self._language_note)
        # 整列一起藏:只藏按鈕的話,空的那一列仍佔著版面的間距。
        self._restart_row = _row(page, self._restart_button)
        self._restart_row.hide()
        layout.addWidget(self._restart_row)
        layout.addWidget(_row(page, self._update_button, self._version_label))

    def _on_language_picked(self, index: int) -> None:
        if self._preferences is None:
            return
        code = str(self._language_picker.itemData(index))
        self._preferences.set_language(code)
        changed = code != self._chosen_language
        self._language_note.setText(tr("重新啟動 MIA 之後換成新的語言。") if changed else "")
        self._language_note.setVisible(changed)
        self._restart_row.setVisible(changed)

    def _on_restart(self) -> None:
        """重新啟動。遊戲或辨識還在跑的話先問:瀏覽器會跟著關,這一場就結束了。"""
        prefs = self._preferences
        if prefs is None:
            return
        if prefs.busy():
            answer = QMessageBox.question(
                self,
                tr("重新啟動 MIA"),
                tr("遊戲或辨識功能還在執行，重新啟動會一起關閉。繼續嗎？"),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        prefs.restart()

    def _add_recognition_picker(self, page: _Page) -> None:
        """在一個功能頁的設定子頁放「畫面辨識方式」。

        **三頁各一個選單,背後是同一個值。** 辨識方式是整條畫面辨識那條路的
        設定,三個功能都吃它 —— 但使用者是在調某一個功能時想到它的,所以
        每一頁都要找得到。改了任何一頁,另外兩頁當場跟著變
        (見 :meth:`_on_recognition_picked`),不會有兩頁寫著不一樣的值。
        """
        picker = AlignedComboBox(page.settings)
        picker.setFixedWidth(COMBO_WIDTH)
        picker.addItem(tr("樣板比對"), "template")
        picker.addItem("YOLO", "yolo")
        picker.addItem(tr("混合辨識（樣板＋YOLO）"), "hybrid")
        picker.setToolTip(
            tr("用於畫面辨識、圖像 AI 建議與圖像放銃分析。\n"
            "切換會重新建立畫面追蹤，建議在新一局開始前切換。\n"
            "網頁版 AI 建議要走畫面,另外到「AI 建議 → 設定」勾「使用圖像辨識」。")
        )
        if self._recognition is None or not self._recognition.can_pick_recognition():
            picker.setEnabled(False)
        else:
            _select_data(picker, self._recognition.recognition_backend())
            picker.currentIndexChanged.connect(
                lambda index, p=picker: self._on_recognition_picked(p, index)
            )
        self._recognition_pickers.append(picker)
        page.add_group(
            tr("畫面辨識方式"),
            _row(page.settings, picker),
            info=tr("三個功能共用這一個設定,改了這裡,另外兩頁跟著變。"),
        )

    def _on_recognition_picked(self, source: QComboBox, index: int) -> None:
        """換辨識方式,並把另外兩頁的選單同步過去。

        同步時擋訊號:那是把狀態畫出來,不是使用者按的 —— 接著的話
        會再呼叫一次 ``set_recognition_backend``,白白多重建一次畫面追蹤。
        """
        if self._recognition is None:
            return
        value = source.itemData(index)
        self._recognition.set_recognition_backend(value)
        for picker in self._recognition_pickers:
            if picker is not source:
                with QSignalBlocker(picker):
                    _select_data(picker, value)
        if value in _YOLO_BACKENDS:
            self._warn_yolo_skin()

    def _warn_yolo_skin(self) -> None:
        """切到會用 YOLO 的辨識方式時,提醒把牌桌與牌背換回預設皮膚。

        YOLO 的模型是用預設皮膚的畫面訓練的。2026-11-11 Steam 版實測:
        換了牌桌與牌背,區域模型完全框不到「摸到的牌」,換回預設就正常 ——
        而且失敗時沒有任何錯誤訊息,只是事件流卡住、AI 建議出不來,
        使用者不會想到是皮膚。所以在切換的當下講,不是寫在說明文件裡等人找。

        跳在切換之後而不是之前:換不換是使用者已經決定的事,這只是提醒,
        不該變成一個要按「確定」才換得成的關卡。
        """
        notices = self._notices
        if notices is not None and not notices.wants(YOLO_SKIN_NOTICE):
            return
        if _ask_yolo_skin(self) and notices is not None:
            notices.dismiss(YOLO_SKIN_NOTICE)

    def _build_platform_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """玩的是網頁版還是 Steam 版。

        **放在設定頁的最上面,而且是全域設定而不是某一頁的。** 它決定的是
        「資料從哪裡來」—— 網頁版三個功能都有料,Steam 版只有畫面那條路
        (見 :mod:`mia.platforms`)。那不是某一頁的顯示選項,是整個程式的前提,
        所以排在所有東西之前。

        換了之後會關掉新版本撐不起的功能,所以三個開關與「開始遊戲」都要
        重新同步一次 —— 不然畫面上會留著一個已經被關掉、卻還撥在「開」的開關。
        """
        row = QWidget(page)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        # 只有兩個選項,做成兩顆並排的按鈕而不是下拉選單:兩個都一直看得到,
        # 按一下就換,不必先點開才知道另一個是什麼。
        picker = ChoiceButtons([(platforms.NAMES[key], key) for key in platforms.PLATFORMS], row)
        self._platform_picker = picker

        if self._platform is None or not self._platform.can_pick_platform():
            picker.setEnabled(False)
            picker.setToolTip(tr("這個模式下版本不是現在決定的(重播、示範、--no-vision)"))
        else:
            picker.select(self._platform.platform())
            picker.setToolTip(
                tr("網頁版:MIA 自己開瀏覽器,資料來自封包。\n"
                "Steam 版:遊戲自己開,MIA 找到視窗,全部靠畫面辨識;\n"
                "看不到的牌只能猜,AI 建議的準確度比網頁版低。")
            )
            picker.chosen.connect(self._on_platform_picked)

        row_layout.addWidget(picker)
        row_layout.addStretch(1)
        layout.addWidget(row)

    def _on_platform_picked(self, value: object) -> None:
        """換版本 —— **會關掉新版本撐不起的功能,並重啟畫面辨識**。

        不在這裡擋重複選擇::meth:`~mia.platforms.PlatformChoice.set` 自己
        會比對。但換完之後要把整個畫面重新同步一次:開關的可按狀態、它們的
        tooltip、「開始遊戲」能不能按,三件事都跟著版本變。
        """
        if self._platform is None:
            return
        self._platform.set_platform(str(value))
        self._sync_advice_source()
        self._sync_switches()
        self.apply(self._viewmodel.state)

    def _sync_switches(self) -> None:
        """把每個開關的可按狀態、理由與**開關位置**重新算一次。

        兩個時機都必須做:

        * 換版本之後 —— 新的版本若沒有某個功能的來源,runtime 已經把它關掉了。
          開關若還撥在「開」,畫面就在說謊。
        * 撥任何一個開關之後 —— 一個功能可能帶動另一個(見 :meth:`_on_switch`)。
        """
        if self._switchboard is None:
            return
        for key, switch in self._switches.items():
            usable = self._switchboard.available(key)
            switch.setEnabled(usable)
            switch.setToolTip(
                tr("{value} —— 開啟後才會開始運作", value=features.NAMES[key])
                if usable
                else self._unavailable_reason(key)
            )
            if not usable:
                switch.set_initial(False)
            else:
                switch.set_initial(self._switchboard.is_enabled(key))

    def _build_danger_source_settings(self, page: _Page) -> None:
        """「強制使用圖像辨識」。

        放銃分析有兩條路(封包與純畫面),而**平常不該讓使用者選** ——
        他要的是「告訴我哪張牌危險」,不是「選一個資料來源」。有封包就用
        封包,沒有就用畫面,程式自己知道。所以這裡只有一個勾選框,而且
        預設不勾。

        留著它是為了**比較**:同一個局面兩條路的答案擺在一起,差別一眼就
        看得到 —— 寫論文與抓問題都用得上。
        """
        self._danger_vision_only = QCheckBox(tr("強制使用圖像辨識"), page.settings)
        self._danger_vision_only.setToolTip(
            tr("平常不必勾 —— 有封包就自動用封包(準),沒有才用圖像辨識。\n"
            "Steam 版本來就只有圖像辨識那條。\n"
            "勾起來會重啟放銃分析,因為走哪條路是建構時決定的。")
        )
        picker = self._danger_source
        if picker is None or not picker.can_pick_danger_source():
            self._danger_vision_only.setEnabled(False)
            self._danger_vision_only.setToolTip(tr("這個模式只有一條路,沒有得選"))
        else:
            # 先設初始值再接訊號 —— 反過來會把「上次記住的狀態」當成使用者
            # 剛剛的操作,而那會白白重啟一次放銃分析。
            self._danger_vision_only.setChecked(picker.danger_vision_only())
            self._danger_vision_only.toggled.connect(picker.set_danger_vision_only)
        page.add_group(
            tr("資料來源"),
            self._danger_vision_only,
            info=tr("平常不必勾:有封包就用封包,沒有才用畫面。勾了是給比較兩條路用的。"),
        )

    def _build_overlay_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """Overlay 的三個勾選。

        **鎖定不在這裡**,在 Overlay 右上角那顆鎖(:mod:`mia.ui.overlay.lock`)
        —— 擺好位置跟鎖上是連著做的,眼睛已經在 Overlay 上,不該再回側邊視窗找
        一個勾選。那顆鎖是獨立的小視窗,Overlay 穿透時照樣點得到。

        **展開的開關仍在這裡,不在 Overlay 上。** Overlay 鎖定之後對滑鼠是透明的
        —— 擺在它自己身上的按鈕會變成死的,而使用者看得到卻按不到的按鈕比沒有
        按鈕糟。

        Overlay 也**不進 features.py 的開關體系**:那些開關管的是「要不要花
        130MB 載權重 / 要不要持續擷取螢幕 / 要不要跑排除法」,而這裡只是換一種
        畫法。混在一起會讓「開了但沒東西跑」變得難解釋。
        """
        overlay = self._overlay
        self._overlay_shown = QCheckBox(tr("在遊戲上顯示 Overlay"), page)
        self._overlay_expanded = QCheckBox(tr("Overlay 顯示候選 Q 值"), page)
        # 放銃分析**要不要跟著上 HUD 是分開問的**:那是一手 14 列,會把
        # Overlay 往下拉長一大段。想一直看著的人與只想要一行建議的人是
        # 兩種用法,而後者不該被迫接受前者的高度。
        self._overlay_danger = QCheckBox(tr("Overlay 顯示放銃危險度(會往下變長)"), page)
        self._overlay_danger.setToolTip(tr("還要「放銃分析」那一頁的開關也打開才會有資料"))
        self._overlay_shown.setToolTip(
            tr("位置會記住,下次啟動回到同一個地方。\n"
            "要鎖定位置(讓點擊穿到遊戲),按 Overlay 右上角的鎖。")
        )
        boxes = (
            self._overlay_shown,
            self._overlay_expanded,
            self._overlay_danger,
        )

        if overlay is None:
            for box in boxes:
                box.setEnabled(False)
                box.setToolTip(tr("這個模式沒有 Overlay"))
                layout.addWidget(box)
            return

        # 先把初始值設好再接訊號 —— 反過來的話 setChecked 會立刻觸發一次
        # handler,而那一次會把「上次記住的狀態」當成使用者剛剛的操作存回去
        self._overlay_shown.setChecked(overlay.shown)
        self._overlay_expanded.setChecked(overlay.expanded)
        self._overlay_danger.setChecked(overlay.danger_shown)
        self._overlay_shown.toggled.connect(self._on_overlay_shown)
        self._overlay_expanded.toggled.connect(overlay.set_expanded)
        self._overlay_danger.toggled.connect(overlay.set_danger_shown)
        for box in boxes:
            layout.addWidget(box)

        self._sync_overlay_boxes(overlay.shown)

    def _build_opacity_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """「透明度」滑軌:**同時**作用在 Overlay 與事件流 Log。

        只有一條,不是兩個視窗各一條:它們疊在同一個遊戲畫面上,一個淡一個濃
        看起來像其中一個壞了。值存在 :attr:`~mia.ui.state.UiState.hud_opacity`。

        滑軌顯示的是**透明度**(往右越透明),不是不透明度 —— 使用者說的是
        「調透明度」,0% 就是原本的樣子。最多到 80%,理由見
        :data:`~mia.ui.state.MIN_HUD_OPACITY`。
        """
        targets = [w for w in (self._overlay, self._event_log) if w is not None]
        self._hud_windows = targets
        slider = QSlider(Qt.Orientation.Horizontal, page)
        slider.setRange(0, round((1 - MIN_HUD_OPACITY) * 100))
        slider.setSingleStep(5)
        slider.setPageStep(10)
        slider.setMinimumWidth(160)
        self._opacity_slider = slider
        self._opacity_value = QLabel("", page)
        self._opacity_value.setStyleSheet(CAPTION)
        self._opacity_value.setMinimumWidth(34)

        if not targets:
            slider.setEnabled(False)
            slider.setToolTip(tr("這個模式沒有懸浮視窗"))
        else:
            slider.setValue(round((1 - targets[0].opacity) * 100))
            slider.valueChanged.connect(self._on_opacity)
        self._show_opacity(slider.value())

        caption = _caption(tr("透明度"), page)
        info = InfoDot(
            tr("同時套用到所有懸浮視窗:Overlay 與事件流 Log。\n0% 是原本的樣子,往右越透明。"), page
        )
        layout.addWidget(_row(page, caption, info, slider, self._opacity_value))

    def _on_opacity(self, transparency: int) -> None:
        opacity = 1 - transparency / 100
        for window in self._hud_windows:
            window.set_opacity(opacity)
        self._show_opacity(transparency)

    def _show_opacity(self, transparency: int) -> None:
        self._opacity_value.setText(f"{transparency}%")

    def _build_log_settings(self, page: QWidget, layout: QVBoxLayout) -> None:
        """Log 視窗的勾選。與 Overlay 無關,也不互相連動。

        **鎖定不在這裡**,在 Log 視窗右上角那顆鎖 —— 理由與 Overlay 相同,
        見 :meth:`_build_overlay_settings`。
        """
        log = self._event_log
        self._log_shown = QCheckBox(tr("開啟 Log 視窗"), page)
        self._log_shown.setToolTip(
            tr("獨立的浮動視窗,顯示 AI 建議吃到的事件流:中文說明加 MJAI 格式。\n"
            "要開著 AI 建議才有內容。位置與大小會記住。\n"
            "要鎖定位置(讓點擊穿到遊戲),按 Log 視窗右上角的鎖。")
        )
        if log is None:
            self._log_shown.setEnabled(False)
            self._log_shown.setToolTip(tr("這個模式沒有 Log 視窗"))
        else:
            # 先設初始值再接訊號,理由見 _build_overlay_settings。
            self._log_shown.setChecked(log.shown)
            self._log_shown.toggled.connect(log.set_shown)
        layout.addWidget(self._log_shown)

    def _on_overlay_shown(self, on: bool) -> None:
        assert self._overlay is not None
        self._overlay.set_shown(on)
        self._sync_overlay_boxes(on)

    def _sync_overlay_boxes(self, shown: bool) -> None:
        """收起來的時候另外兩個勾選沒有意義,停用但保留勾選狀態。

        不清掉勾選:那是使用者的偏好,下次顯示時該照舊。
        """
        self._overlay_expanded.setEnabled(shown)
        self._overlay_danger.setEnabled(shown)

    def closeEvent(self, event: QCloseEvent) -> None:
        """關掉側邊視窗時把 Overlay 一起收掉。

        少了這一段,顯示中的 Overlay 會是「最後一個還開著的視窗」,Qt 於是
        不結束程式 —— 畫面上只剩一個關不掉的浮動 HUD,而終端機看起來一切正常。
        """
        if self._overlay is not None:
            self._overlay.close()
        if self._event_log is not None:
            self._event_log.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------ 事件

    def _on_switch(self, key: str, on: bool) -> None:
        """使用者撥了開關。

        撥完**每一個**開關都重算一次,不是只算被撥的那個。兩個理由:

        * 開啟可能失敗(權重不見了、找不到遊戲視窗),那時開關要自己彈回去,
          不能停在「開」而底下沒東西在跑。
        * **一個開關可能帶動另一個。** 放銃分析走圖像辨識那條時要靠畫面辨識
          (牌河的畫面與自己的手牌都來自它),runtime 會一併打開 —— 而那件事
          應該**看得見**:畫面辨識的開關自己亮起來就說明了一切,不必再寫一句話
          去解釋。只同步被撥的那一個的話,它會停在暗的,使用者只會覺得哪裡怪怪的。
        """
        assert self._switchboard is not None
        self._switchboard.set_enabled(key, on)
        self._sync_switches()
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

    def _cannot_start_reason(self, launcher: GameLauncher | None) -> str:
        """「開始遊戲」為什麼是灰的。

        先問 launcher —— :class:`~mia.live.runtime.LiveRuntime` 分得出是
        「這一版不由我開」還是「啟動參數沒要我開」。問不到(重播用的假物件)
        才退回自己講。
        """
        if launcher is None:
            return tr("重播與示範模式不需要開遊戲")
        ask = getattr(launcher, "why_cannot_start_game", None)
        if ask is not None:
            reason = ask()
            if reason:
                return str(reason)
        return tr("啟動參數裡沒有要 MIA 開遊戲(--tail / --no-packets)")

    def _sync_start_button(self) -> None:
        """按鈕要說現在的實情。

        三種狀態:沒有可開的遊戲(重播 / ``--tail`` / ``--no-packets`` /
        **Steam 版**)、可以開、已經在跑。**已經在跑時停用而不是換成
        「結束遊戲」** —— 關掉瀏覽器等於把那一場丟掉,不該是一個手滑就會
        按到的按鈕。

        停用時的那句話要說得出**為什麼**。Steam 版尤其重要:使用者看到按鈕
        是灰的,要知道「因為這一版的遊戲由你自己從 Steam 開」,而不是以為
        程式壞了或還沒載完。
        """
        launcher = self._launcher
        if launcher is None or not launcher.can_start_game():
            self._start_button.setText(START_GAME)
            self._start_button.setEnabled(False)
            self._start_button.setToolTip(self._cannot_start_reason(launcher))
            return
        running = launcher.game_running()
        self._start_button.setText(tr("遊戲進行中") if running else START_GAME)
        self._start_button.setEnabled(not running)
        self._start_button.setToolTip(
            tr("瀏覽器已經開著 —— 關掉它就會結束這一場")
            if running
            else tr("開一個受控的瀏覽器並開始錄封包")
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
            self._notice.setText(tr("看不懂的尺寸「{text}」—— 格式是 1440x810", text=text))
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

    def _on_style_picked(self, index: int) -> None:
        """換打法風格 —— **引擎會整個重啟**。

        重啟要重載權重(每份 130MB,實測 0.5~0.6 秒)並從頭重播事件流把局面
        追上來,那段時間畫面上沒有建議、狀態列會說「啟動引擎…」。

        不在這裡擋重複選擇::meth:`StylePicker.set_style` 自己會比對,
        選到同一項時什麼都不做。擋兩次只是把同一個判斷寫兩份。
        """
        if self._style is None:
            return
        name = self._style_picker.itemData(index)
        if name is not None:
            self._style.set_style(name)
        # 換主要風格可能把比較對象撞掉(不能跟自己比),而那個選單上的字
        # 不會自己改 —— 留著就是畫面寫著一個已經不生效的選擇。
        self._sync_compare_picker()

    def _on_compare_picked(self, index: int) -> None:
        """換並排比較的對象 —— 與換風格同樣會重啟引擎。

        ``None``(「不比較」那一列)是合法的值,所以這裡不像
        :meth:`_on_style_picked` 那樣擋掉它。
        """
        if self._style is None:
            return
        self._style.set_compare(self._compare_picker.itemData(index))

    def _sync_compare_picker(self) -> None:
        """重建「比較對象」的清單:不比較,加上**除了主要風格以外**的每一個。

        把主要風格排除掉而不是選了再擋:選單上出現一個按下去沒反應的項目,
        比它根本不在那裡糟 —— 使用者會以為程式壞了。

        重建期間一定要擋訊號。``clear()`` 與 ``addItem()`` 都會發
        ``currentIndexChanged``,而接在上面的 slot 會去 ``set_compare`` ——
        結果是**重建一次選單就重啟一次引擎**,每份權重 130MB。
        """
        if self._style is None or not self._style.can_pick_style():
            return
        current = self._style.compare()
        primary = self._style.style()
        with QSignalBlocker(self._compare_picker):
            self._compare_picker.clear()
            self._compare_picker.addItem(NO_COMPARE, None)
            for name in self._style.styles():
                if name != primary:
                    self._compare_picker.addItem(translate(name), name)
            _select_data(self._compare_picker, current)

    def _on_candidate_count(self, value: int) -> None:
        """候選清單要列幾個。重畫一次,不必等下一個事件。"""
        self._advice.max_candidates = value
        self._advice.update_from(self._viewmodel.state, enabled=self._is_on(features.ADVICE))

    def _on_always_on_top(self, enabled: bool) -> None:
        """切換置頂。

        改 window flag 會讓已顯示的視窗被隱藏,所以要重新 show 一次 ——
        這正是先前「Dock 有圖示但沒有視窗」那個 bug 的成因。
        """
        was_visible = not self.isHidden()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()

    def show_page(self, index: int, *, settings: bool = False) -> None:
        """切到第幾個功能頁;``settings`` 時順便切到那一頁的「設定」子頁。

        給測試與截圖用。一般設定頁沒有子頁,``settings`` 對它沒有作用。
        """
        self._nav.setCurrentRow(index)
        page = self._stack.widget(index)
        if isinstance(page, _Page):
            page.show_subpage(1 if settings else 0)

    # ------------------------------------------------------------------ 資料

    def apply(self, state: ViewState) -> None:
        """把狀態畫上去。

        每一頁都更新,不只是當前可見的那個 —— 切頁時才更新會讓使用者看到
        一瞬間的舊資料。更新一次的成本是十幾個 label,遠低於那個閃動的代價。
        """
        self._sync_start_button()
        self._advice.update_from(state, enabled=self._is_on(features.ADVICE))
        self._analysis.update_from(state, enabled=self._is_on(features.VISION))
        self._danger.update_from(
            state,
            enabled=self._is_on(features.DANGER),
            # 問**實際在跑的那個**,不是問設定 —— 沒有封包來源時本來就走
            # 圖像辨識,使用者沒勾也一樣。
            source=(
                self._danger_source.danger_source_name()
                if self._danger_source is not None
                else None
            ),
        )
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
            return tr("還沒開始 —— 按左下角的「{START_GAME}」開啟遊戲", START_GAME=START_GAME)
        # 問**每一個**功能,不是寫死其中兩個。加了放銃分析之後,原本那句
        # 會在它開著的時候照樣說「兩個功能都關著」—— 畫面上明明有東西在跑。
        if self._switchboard is not None and not any(
            self._switchboard.is_enabled(k) for k in features.NAMES
        ):
            return tr("功能都關著 —— 用各頁右上角的開關打開")
        return _status_text(state)


def _status_text(state: ViewState) -> str:
    parts: list[str] = []
    if state.hand_conflict:
        parts.append(tr("⚠ 畫面與封包的手牌不一致"))
    parts.extend(state.notices)
    if not parts:
        engines = tr("、").join(translate(e.name) for e in state.engines)
        return tr("引擎:{engines}", engines=engines) if engines else tr("等待資料…")
    # 一個來源一行。串成一行的話後面那幾句會被視窗寬度切掉,見 __init__ 的 _notice。
    return "\n".join(parts)


def _select_data(picker: QComboBox, data: object) -> None:
    """把選單切到 ``itemData`` 等於 ``data`` 的那一項。找不到就留在原地。

    用 ``findData`` 而不是記索引:選項順序是 :data:`CANVAS_PRESETS` 決定的,
    日後加一個尺寸就會讓寫死的索引指到別的地方。
    """
    index = picker.findData(data)
    if index >= 0:
        picker.setCurrentIndex(index)


def _wordmark(parent: QWidget) -> QWidget:
    """功能列最上面的圖示 + 名字。

    純粹是**讓它看起來像一個程式**。視窗標題列在 macOS 上很細、在全螢幕的
    遊戲旁邊幾乎看不到,少了這一塊,側邊視窗就是一片沒有身分的灰。

    只寫縮寫不寫全名:這一欄寬 96px,放得下四個全形字,全名會被截成
    「Mahjong Intell…」。
    """
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(12, 12, 10, 8)
    layout.setSpacing(7)

    mark = QLabel(row)
    mark.setPixmap(app_icon().pixmap(20, 20))
    name = QLabel(APP_ABBR, row)
    name.setStyleSheet("font-size: 15px; font-weight: 700;")
    name.setToolTip(APP_TITLE)

    layout.addWidget(mark)
    layout.addWidget(name)
    layout.addStretch(1)
    return row


def _caption(text: str, parent: QWidget) -> QLabel:
    """欄位說明。小字、次要顏色。"""
    label = QLabel(text, parent)
    label.setStyleSheet(CAPTION)
    return label


class InfoDot(QWidget):
    """標題旁邊的 ⓘ。滑鼠移上去**立刻**顯示說明,移開就收。

    不用 Qt 內建的 tooltip 延遲(約 0.7 秒):那個延遲是為了「不小心滑過去
    不要跳東西」,但這個圖示存在的唯一目的就是被滑過去 —— 等它等於讓人以為
    沒有東西。仍然 ``setToolTip`` 一份,鍵盤與輔助工具拿得到同一段字。

    自己畫而不是用「ⓘ」字元:那個字元在不同字型裡大小、粗細差很多,
    PingFang 裡它比旁邊的中文標題還大一圈。顏色用 :func:`~mia.ui.style.mix`
    算,理由與分隔線相同 —— 要在淺／深色模式都落在「看得到但不搶眼」。
    """

    SIZE = 14

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = text
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setToolTip(text)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self._hovered = False

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        QToolTip.showText(self.mapToGlobal(QPoint(0, self.height() + 4)), self.text, self)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self.update()
        QToolTip.hideText()
        super().leaveEvent(event)

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = mix(self.palette(), 0.85 if self._hovered else 0.5)
        pen = QPen(color)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(0.6, 0.6, self.SIZE - 1.2, self.SIZE - 1.2))
        font = QFont(self.font())
        font.setPixelSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, self.SIZE, self.SIZE), Qt.AlignmentFlag.AlignCenter, "i")


def _section(text: str, parent: QWidget, *, info: str = "") -> QWidget:
    """一段設定的標題;有 ``info`` 時後面跟一個 :class:`InfoDot`。

    與 :func:`_caption` 同樣大小、同樣顏色,只是**粗體**。設定頁上兩種東西
    都在:「遊戲視窗尺寸」是一段的名字,而「風格」是一個欄位的說明。長得
    一模一樣的話,整頁讀起來是一串沒有層次的灰字。
    """
    label = QLabel(text, parent)
    label.setStyleSheet(CAPTION + "font-weight: 700;")
    if not info:
        return label
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    label.setParent(row)
    layout.addWidget(label)
    layout.addWidget(InfoDot(info, row), 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addStretch(1)
    return row


def _row(parent: QWidget, *widgets: QWidget) -> QWidget:
    """一列靠左排的控制項。

    包一層而不是直接放進直排版面:直排裡的下拉選單會被拉到整頁寬,
    一個只有「標準」兩個字的選單橫跨四百像素,看起來像一個輸入框。
    """
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return row


def _rule(parent: QWidget, *, vertical: bool = False) -> Hairline:
    """一條分隔線。理由與顏色見 :class:`~mia.ui.style.Hairline`。"""
    return Hairline(parent, vertical=vertical)


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
