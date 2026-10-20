"""側邊視窗與 Overlay 共用的顏色、線條與強調色。集中在這裡,因為深色模式踩過一次坑。

為什麼不是 ``palette(mid)``
---------------------------
次要文字(說明、提示、來源標註)原本一律寫 ``color: palette(mid)``。看起來
很對 —— 沒有寫死任何顏色,一切交給系統 palette。但實測 macOS 深色模式
(PySide6 6.11.1)::

    Window            30, 30, 30
    Mid               21, 21, 21   ← 比背景還暗
    WindowText       255,255,255
    PlaceholderText  255,255,255 @ alpha 140

``Mid`` 是給 3D 邊框陰影用的角色,不是給文字的。淺色模式下它恰好比背景暗
(184 對 239)所以讀得到,深色模式下整片沉進背景 —— 實機上就是「字幾乎看不
見」。**用了 palette 不等於跟著模式走**,要看用的是哪一個角色。

``PlaceholderText`` 是由**文字色**推出來的(同色系加透明度),所以兩種模式
都自動成立。而且它是在 stylesheet 裡解析的,系統切換淺／深色時 Qt 會重新
polish,顏色跟著變 —— 不需要監聽外觀變化再重套一次樣式。

需要「比背景亮一階」的面而不是文字時
------------------------------------
分隔線、進度條的軌道、關著的開關 —— 這些都不是文字,沒有一個 palette 角色
在兩種模式下都落在對的位置(``Mid`` 的毛病同上,``AlternateBase`` 在淺色模式
下幾乎與背景同色)。:func:`mix` 直接算:把文字色按比例壓在視窗底色上,
**兩種模式各自算各自的**,所以深色模式得到比背景亮的灰、淺色模式得到比背景
暗的灰,而比例一樣。

它回傳**已經合成好的實色**而不是帶 alpha 的顏色:軌道還要與強調色做線性
混色,而混色是逐分量算的,帶著 alpha 進去混出來的中間色不對。
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = [
    "ACCENT",
    "ACCENT_HOVER",
    "ACCENT_PRESSED",
    "CAPTION",
    "HAIRLINE_MIX",
    "MUTED",
    "MUTED_COLOR",
    "SMALL",
    "TRACK_MIX",
    "Hairline",
    "hairline",
    "mix",
    "off_track",
]

#: 次要文字的顏色。理由見模組說明 —— 不要換回 ``palette(mid)``。
MUTED_COLOR = "palette(placeholder-text)"

#: 次要文字。
MUTED = f"color: {MUTED_COLOR};"

#: 小字。標題、提示、單位這類東西。
SMALL = "font-size: 11px;"

#: 區塊標題與提示:小字 + 次要顏色。
CAPTION = MUTED + SMALL

#: 強調色:Apple 的 system green。
#:
#: **全程式只有這一個值。** 它同時是開關打開的顏色、「開始遊戲」的底色、
#: Overlay 上「這是建議的那一張」的顏色、程式圖示的底色 —— 原本四個地方各自
#: 寫死 ``#34C759``,改一次要全域搜尋,而漏掉一處的症狀是「有個地方的綠不太
#: 一樣」,不會有人回報。
ACCENT = "#34C759"

#: 滑鼠移上去。比 :data:`ACCENT` 暗一階。
ACCENT_HOVER = "#2FB350"

#: 按下去。再暗一階 —— 少了這個,按鈕按下去畫面毫無反應,
#: 而「開始遊戲」要等瀏覽器開起來才看得到結果,那段空窗最需要回饋。
ACCENT_PRESSED = "#279C45"

#: 開關關閉時,軌道要用多少比例的文字色蓋在視窗底色上。
#:
#: 這個比例讓兩種模式都落在「比背景明顯一階、但不會被誤認成打開」的位置 ——
#: 深色 (83,83,83) 對背景 30,淺色 (180,180,180) 對背景 239。
TRACK_MIX = 0.235

#: 分隔線。比軌道淡得多 —— 線要能被看見,但它分隔的是內容,
#: 本身不該變成畫面上第一個被注意到的東西。
HAIRLINE_MIX = 0.16


def mix(palette: QPalette, ratio: float) -> QColor:
    """把文字色按 ``ratio`` 壓在視窗底色上,回傳合成好的實色。

    ``ratio`` 0 是純背景、1 是純文字色。理由見模組說明。
    """
    text = palette.color(QPalette.ColorRole.WindowText)
    base = palette.color(QPalette.ColorRole.Window)
    channels = ((text.red(), base.red()), (text.green(), base.green()), (text.blue(), base.blue()))
    return QColor(*(round(b + (t - b) * ratio) for t, b in channels))


def off_track(palette: QPalette) -> QColor:
    """開關關閉時的軌道顏色。"""
    return mix(palette, TRACK_MIX)


def hairline(palette: QPalette) -> QColor:
    """分隔線的顏色。"""
    return mix(palette, HAIRLINE_MIX)


class Hairline(QWidget):
    """一條分隔線。

    取代 ``QFrame(HLine, Sunken)``。後者畫的是 1990 年代的「凹槽」 ——
    兩條線(暗 + 亮)做出立體感,在平面的介面上看起來像有人忘了更新樣式。
    這裡只畫一條 1px 的淡線。

    **自己畫而不是用 stylesheet。** 需要的是「比背景淡一階」,而那個顏色
    算得出來(:func:`hairline`)卻寫不進 stylesheet —— ``palette(...)`` 只
    取得到角色,取不到合成色。自己畫還順便解決了跟隨系統淺／深色切換的問題:
    palette 一變 Qt 就重畫,這裡每次都重新算。
    """

    def __init__(self, parent: QWidget | None = None, *, vertical: bool = False) -> None:
        super().__init__(parent)
        self._vertical = vertical
        if vertical:
            self.setFixedWidth(1)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(1)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), hairline(self.palette()))
