"""側邊視窗的顏色。集中在這裡,因為深色模式踩過一次坑。

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
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

__all__ = ["CAPTION", "MUTED", "MUTED_COLOR", "SMALL", "off_track"]

#: 次要文字的顏色。理由見模組說明 —— 不要換回 ``palette(mid)``。
MUTED_COLOR = "palette(placeholder-text)"

#: 次要文字。
MUTED = f"color: {MUTED_COLOR};"

#: 小字。標題、提示、單位這類東西。
SMALL = "font-size: 11px;"

#: 區塊標題與提示:小字 + 次要顏色。
CAPTION = MUTED + SMALL

#: 開關關閉時,軌道要用多少比例的文字色蓋在視窗底色上。
#:
#: 同樣不能用 ``Mid``:深色模式下那個值比背景還暗,關著的開關會整個消失,
#: 看不出那裡有一個可以點的東西。改成文字色壓在底色上,兩種模式都會落在
#: 「比背景明顯一階、但不會被誤認成打開」的位置 ——
#: 深色 (83,83,83) 對背景 30,淺色 (180,180,180) 對背景 239。
_OFF_TRACK_MIX = 0.235


def off_track(palette: QPalette) -> QColor:
    """開關關閉時的軌道顏色。

    回傳的是**已經合成好的實色**,不是帶 alpha 的顏色:軌道還要與「開」的綠
    做線性混色,而混色是逐分量算的,帶著 alpha 進去混出來的中間色不對。
    """
    text = palette.color(QPalette.ColorRole.WindowText)
    base = palette.color(QPalette.ColorRole.Window)
    channels = ((text.red(), base.red()), (text.green(), base.green()), (text.blue(), base.blue()))
    return QColor(*(round(b + (t - b) * _OFF_TRACK_MIX) for t, b in channels))
