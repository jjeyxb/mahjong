"""側邊視窗的控制項外觀:下拉選單、按鈕、勾選框、滑軌、數字框。

為什麼不用系統原生的樣子
------------------------
macOS 原生控制項放在一起時彼此不一致:可編輯的下拉選單(遊戲視窗尺寸)畫成
黑底輸入框加一顆小方塊箭頭,不可編輯的畫成上下雙箭頭的按鈕,兩個並排像是
兩個程式的東西。深色模式下沒勾的勾選框是一塊沒有邊框的深灰方塊,幾乎看不見。
停用的「開始遊戲」只剩一行灰字,看不出是一顆按鈕。

所以設定頁的控制項一律用這裡的 stylesheet:同一組底色、邊框、圓角、高度。

顏色從 palette 算,不寫死
-------------------------
與 :mod:`mia.ui.style` 同一套做法:用 :func:`~mia.ui.style.mix` 把文字色按比例
壓在視窗底色上,淺色與深色模式各自得到「比背景深一階 / 亮一階」的面。
stylesheet 裡寫不出合成色(``palette(...)`` 只取得到角色),所以這裡每次都
**從 palette 重新產生整份字串**,系統切換淺／深色時呼叫端再套一次。

勾選框的勾、下拉的箭頭要用圖檔(stylesheet 只吃 ``url()``),所以在暫存目錄
寫幾個小 SVG,檔名帶顏色 —— 換模式就是換一組檔名,不會讀到舊顏色的快取。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout, QPushButton, QWidget

from mia.distribution.layout import LAYOUT
from mia.ui.style import ACCENT, ACCENT_HOVER, ACCENT_PRESSED, mix

__all__ = [
    "COMBO_WIDTH",
    "CONTROL_HEIGHT",
    "POPUP_GAP",
    "AlignedComboBox",
    "ChoiceButtons",
    "controls_stylesheet",
    "nav_stylesheet",
    "polish_combo",
    "segmented_stylesheet",
    "start_button_stylesheet",
]

#: 選單框與按鈕組的高度(px)。下拉選單的高度是 stylesheet 撐出來的
#: (1 + 5 + 18 + 5 + 1),按鈕組直接固定成同一個數字 —— 兩個上下排在一起,
#: 差兩個像素就看得出來。
CONTROL_HEIGHT = 30

#: 按鈕組外框到裡面那幾顆按鈕的距離(px)。
_CHOICE_INSET = 3

#: 展開的清單與選單框之間的距離(px)。
POPUP_GAP = 4

#: 設定頁下拉選單的寬度。**一律同寬**:寬度跟著內容走的話,「標準」與
#: 「樣板比對」兩個選單上下排在一起,右緣差一截,看起來像沒對齊。
COMBO_WIDTH = 200

_ICON_DIR = (LAYOUT.cache if LAYOUT.packaged else Path(tempfile.gettempdir())) / "mia-ui-icons"

_CHEVRON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
    '<path d="M2.5 4.5 L6 8 L9.5 4.5" fill="none" stroke="{color}" stroke-width="1.6"'
    ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_CHEVRON_UP = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
    '<path d="M2.5 7.5 L6 4 L9.5 7.5" fill="none" stroke="{color}" stroke-width="1.6"'
    ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_CHECK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
    '<path d="M2.6 6.3 L5 8.6 L9.6 3.6" fill="none" stroke="{color}" stroke-width="1.8"'
    ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _icon(template: str, name: str, color: QColor) -> str:
    """寫一個 SVG,回傳 stylesheet 用的路徑。寫不了就回空字串(沒有圖示,不是當掉)。"""
    hex_color = color.name()
    path = _ICON_DIR / f"{name}-{hex_color[1:]}.svg"
    try:
        if not path.exists():
            _ICON_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(template.format(color=hex_color), encoding="utf-8")
    except OSError:
        return ""
    return path.as_posix()


def _image(path: str) -> str:
    return f"image: url({path});" if path else ""


class _Colors:
    """一份 palette 算出來的所有顏色。"""

    def __init__(self, palette: QPalette) -> None:
        role = QPalette.ColorRole
        self.text = palette.color(role.WindowText).name()
        self.window = palette.color(role.Window).name()
        # 系統強調色,**取 Active 那一組**:側邊視窗大部分時間沒有焦點(焦點在
        # 遊戲上),而 Inactive 的 Highlight 在 macOS 淺色模式下退成淡藍、
        # 勾勾變黑字 —— 整頁看起來像停用了。Accent 角色在兩組裡都是同一個實色。
        self.accent = palette.color(QPalette.ColorGroup.Active, role.Accent).name()
        self.on_accent = "#FFFFFF"
        dark = palette.color(role.Window).lightness() < 128
        # 深色模式的面要比背景「亮」,淺色模式要比背景「暗」——mix 自動做到,
        # 但淺色模式下控制項用白底更像 macOS,所以那邊直接取 Base。
        self.field = mix(palette, 0.08).name() if dark else palette.color(role.Base).name()
        self.field_hover = mix(palette, 0.13 if dark else 0.04).name()
        self.field_pressed = mix(palette, 0.18 if dark else 0.09).name()
        self.field_off = mix(palette, 0.04).name()
        self.border = mix(palette, 0.16 if dark else 0.20).name()
        self.border_hover = mix(palette, 0.30).name()
        self.border_off = mix(palette, 0.09).name()
        self.box_border = mix(palette, 0.38 if dark else 0.34).name()
        self.muted = mix(palette, 0.42).name()
        self.popup = mix(palette, 0.06).name() if dark else palette.color(role.Base).name()
        self.check_off = mix(palette, 0.30).name()
        self.chevron = mix(palette, 0.60).name()
        self.chevron_off = mix(palette, 0.28).name()
        #: 「現在在這一頁」的底色。中性灰,理由見 :func:`nav_stylesheet`。
        self.selected = mix(palette, 0.13 if dark else 0.09).name()
        self.selected_hover = mix(palette, 0.07 if dark else 0.05).name()


def controls_stylesheet(palette: QPalette) -> str:
    """設定頁(與其他放控制項的地方)整份 stylesheet。

    **只寫控制項的型別**(QComboBox、QPushButton、QCheckBox、QSlider、
    QSpinBox、QScrollBar),不碰 QLabel 與 QWidget —— 套在容器上時,那兩個
    會一路傳到每一個標題與說明,把它們各自設好的字級與顏色蓋掉。
    """
    c = _Colors(palette)
    chevron = _image(_icon(_CHEVRON, "chevron", QColor(c.chevron)))
    chevron_off = _image(_icon(_CHEVRON, "chevron", QColor(c.chevron_off)))
    chevron_up = _image(_icon(_CHEVRON_UP, "chevron-up", QColor(c.chevron)))
    check = _image(_icon(_CHECK, "check", QColor(c.on_accent)))
    check_off = _image(_icon(_CHECK, "check", QColor(c.check_off)))
    return f"""
QComboBox, QSpinBox {{
    background: {c.field}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 7px;
    padding: 5px 28px 5px 10px; min-height: 18px;
    combobox-popup: 0;
}}
/* 右邊內距不必替上下箭頭留位置:箭頭(下面的 ::up-button,寬 18)貼著邊框擺,
   Qt 算數字欄位時已經扣掉它。原本這裡也留 20px,等於扣兩次 —— 58px 寬的框只剩
   10px 給數字,Windows 125% 縮放下連一位數都被切掉(2026-11-12 實測)。 */
QSpinBox {{ padding: 4px 4px 4px 8px; }}
QComboBox:hover, QSpinBox:hover {{ background: {c.field_hover}; border-color: {c.border_hover}; }}
QComboBox:on, QComboBox:focus, QSpinBox:focus {{ border-color: {c.accent}; }}
QComboBox:disabled, QSpinBox:disabled {{
    background: {c.field_off}; color: {c.muted}; border-color: {c.border_off};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 26px; border: none; background: transparent;
}}
QComboBox::down-arrow {{ {chevron} width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ {chevron_off} }}
QComboBox QLineEdit {{
    background: transparent; border: none; padding: 0; margin: 0;
    color: {c.text}; selection-background-color: {c.accent};
}}
QComboBox QAbstractItemView {{
    background: {c.popup}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 8px;
    padding: 4px; outline: none;
}}
/* 清單裡的字要與框裡的字左緣對齊:框是 1 邊框 + 10 內距 = 11px;
   清單是 1 邊框 + 4 內距 + 項目自帶的文字邊距(約 5px)+ 這裡的 1px。實測對齊。 */
QComboBox QAbstractItemView::item {{
    min-height: 26px; padding: 0 6px 0 1px; border-radius: 5px;
}}
QComboBox QAbstractItemView::item:selected, QComboBox QAbstractItemView::item:hover {{
    background: {c.accent}; color: {c.on_accent};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border; width: 18px; border: none; background: transparent;
}}
QSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 3px; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {c.field_pressed}; border-radius: 4px;
}}
QSpinBox::up-arrow {{ {chevron_up} width: 9px; height: 9px; }}
QSpinBox::down-arrow {{ {chevron} width: 9px; height: 9px; }}

QPushButton {{
    background: {c.field}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 7px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: {c.field_hover}; border-color: {c.border_hover}; }}
QPushButton:pressed {{ background: {c.field_pressed}; }}
QPushButton:disabled {{
    background: {c.field_off}; color: {c.muted}; border-color: {c.border_off};
}}

QCheckBox {{ spacing: 9px; color: {c.text}; padding: 2px 0; }}
QCheckBox:disabled {{ color: {c.muted}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 5px;
    border: 1px solid {c.box_border}; background: {c.field};
}}
QCheckBox::indicator:hover {{ border-color: {c.accent}; }}
QCheckBox::indicator:checked {{ background: {c.accent}; border-color: {c.accent}; {check} }}
QCheckBox::indicator:disabled {{ background: {c.field_off}; border-color: {c.border_off}; }}
QCheckBox::indicator:checked:disabled {{
    background: {c.field_pressed}; border-color: {c.border}; {check_off}
}}

QSlider {{ min-height: 20px; }}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {c.border}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {c.accent}; }}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -7px 0; border-radius: 9px;
    background: #FFFFFF; border: 1px solid {c.border_hover};
}}
QSlider::handle:horizontal:hover {{ border-color: {c.accent}; }}
QSlider::sub-page:horizontal:disabled {{ background: {c.border_hover}; }}

/* ChoiceButtons:一條外框裡並排幾顆,選中的亮、其他的暗。 */
QWidget#choiceGroup {{
    background: {c.field}; border: 1px solid {c.border}; border-radius: 8px;
}}
QWidget#choiceGroup:disabled {{ background: {c.field_off}; border-color: {c.border_off}; }}
QPushButton#choice {{
    background: transparent; color: {c.muted}; border: none; border-radius: 6px;
    padding: 0;
}}
QPushButton#choice:hover:!checked {{ color: {c.text}; background: {c.field_hover}; }}
QPushButton#choice:checked {{ background: {c.accent}; color: {c.on_accent}; font-weight: 600; }}
QPushButton#choice:disabled {{ color: {c.border_hover}; background: transparent; }}
QPushButton#choice:checked:disabled {{ background: {c.field_pressed}; color: {c.muted}; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {c.border_hover}; border-radius: 2px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {c.muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


def nav_stylesheet(palette: QPalette) -> str:
    """左側功能列。

    選中的那一項是中性灰的圓角藥丸、粗體字。**不用系統藍**:側邊視窗大部分
    時間沒有焦點,macOS 的 Inactive 選取色在深色模式下是一塊發灰的暗藍,
    看起來髒;也不用 MIA 的綠(那是「正在運作」)。
    """
    c = _Colors(palette)
    return (
        "QListWidget { background: transparent; border: none; padding: 6px 6px; outline: none; }"
        f"QListWidget::item {{ padding: 10px 8px; border-radius: 8px; margin-bottom: 2px;"
        f" color: {c.text}; }}"
        f"QListWidget::item:hover:!selected {{ background: {c.selected_hover}; }}"
        f"QListWidget::item:selected {{ background: {c.selected}; color: {c.text};"
        " font-weight: 600; }"
    )


def segmented_stylesheet(palette: QPalette) -> str:
    """「主頁 | 設定」。與功能列同一個「選中」的顏色。"""
    c = _Colors(palette)
    return (
        "QPushButton { border: none; border-radius: 7px; padding: 5px 14px;"
        f" background: transparent; color: {c.muted}; }}"
        f"QPushButton:hover:!checked {{ color: {c.text}; background: {c.selected_hover}; }}"
        f"QPushButton:checked {{ background: {c.selected}; color: {c.text}; font-weight: 600; }}"
    )


class AlignedComboBox(QComboBox):
    """展開的清單**與選單框同寬、左緣對齊、緊貼在正下方**的下拉選單。

    Qt 自己算的清單寬度跟著內容走(「網頁版 / Steam 版」比框窄一截),位置也
    依平台各有偏移 —— 框跟清單錯開幾個像素,一眼就看得出來。這裡在 Qt 算完
    之後把清單放回框的正下方。

    下方放不下(選單在螢幕底部)時改放在框的正上方,一樣同寬、對齊。
    清單裡的字與框裡的字左緣對齊,見 :func:`controls_stylesheet` 的內距。
    """

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.view().window()
        width = self.width()
        height = popup.height()
        below = self.mapToGlobal(QPoint(0, self.height() + POPUP_GAP))
        rect = QRect(below.x(), below.y(), width, height)
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            if rect.bottom() > area.bottom():
                above = self.mapToGlobal(QPoint(0, -POPUP_GAP - height))
                if above.y() >= area.top():
                    rect.moveTop(above.y())
        popup.setMinimumWidth(width)
        popup.setMaximumWidth(width)
        popup.setGeometry(rect)


class ChoiceButtons(QWidget):
    """幾個選項並排成一條,按下一個,其他的變暗。給選項少到不值得一個下拉選單的地方。

    與 :class:`AlignedComboBox` 同寬(:data:`COMBO_WIDTH`),兩者上下排在一起時
    左右緣對齊。樣式在 :func:`controls_stylesheet` 的 ``#choiceGroup`` / ``#choice``。

    Signals:
        chosen: **使用者**按了另一個選項,參數是那個選項的值。:meth:`select`
            (程式把狀態畫出來)不會發這個訊號 —— 不然畫一次就會被當成使用者
            又選了一次,換版本這種事會白白重啟一次畫面辨識。按已經選中的那一顆
            也不發。
    """

    chosen = Signal(object)

    def __init__(self, options: list[tuple[str, object]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("choiceGroup")
        # 純 QWidget 預設不畫 stylesheet 的背景與邊框。
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(COMBO_WIDTH, CONTROL_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(*(_CHOICE_INSET,) * 4)
        layout.setSpacing(_CHOICE_INSET)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._values: list[object] = []
        self._buttons: list[QPushButton] = []
        for label, value in options:
            button = QPushButton(label, self)
            button.setObjectName("choice")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            # 直接定高度:macOS 的按鈕有自己的固定高度,只靠版面會被壓成一條扁的。
            button.setFixedHeight(CONTROL_HEIGHT - 2 * _CHOICE_INSET)
            self._group.addButton(button, len(self._values))
            layout.addWidget(button, 1)
            self._values.append(value)
            self._buttons.append(button)
        self._current: object = None
        self._group.idClicked.connect(self._on_clicked)

    def values(self) -> list[object]:
        return list(self._values)

    def current(self) -> object:
        return self._current

    def button(self, value: object) -> QPushButton:
        return self._buttons[self._values.index(value)]

    def select(self, value: object) -> None:
        """把 ``value`` 畫成選中。不發 :attr:`chosen`。認不得的值就不動。"""
        if value not in self._values:
            return
        self._current = value
        self.button(value).setChecked(True)

    def _on_clicked(self, index: int) -> None:
        value = self._values[index]
        if value == self._current:
            return
        self._current = value
        self.chosen.emit(value)


def polish_combo(combo: QComboBox) -> None:
    """讓下拉清單的圓角是真的圓角。

    清單是另一個 top-level 視窗,本身是方的;stylesheet 只把**裡面**畫成圓角,
    四個角落就露出視窗的黑底。把那個視窗設成透明底,角落才會透出底下的東西。
    """
    popup = combo.view().window()
    popup.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    popup.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
    popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


def start_button_stylesheet(palette: QPalette) -> str:
    """「開始遊戲」。

    綠色是 MIA 的強調色(「正在運作」),所以可按時是實心綠。**停用時仍然畫成
    一顆按鈕**(淡底 + 灰字),不是只剩一行字 —— 那看起來像是標籤,使用者不會
    想到要把滑鼠移上去看 tooltip 說的「為什麼不能按」。
    """
    c = _Colors(palette)
    return (
        f"QPushButton {{ background: {ACCENT}; color: white; border: none;"
        " border-radius: 9px; padding: 10px 4px; font-weight: 600; margin: 8px 8px 10px; }"
        f"QPushButton:hover:enabled {{ background: {ACCENT_HOVER}; }}"
        f"QPushButton:pressed:enabled {{ background: {ACCENT_PRESSED}; }}"
        f"QPushButton:disabled {{ background: {c.field_off}; color: {c.muted};"
        f" border: 1px solid {c.border_off}; }}"
    )
