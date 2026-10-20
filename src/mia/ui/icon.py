"""程式圖示。

畫的是一張「中」牌。紅中是麻將裡辨識度最高的單一字形,不需要任何說明就看得
出這是麻將相關的東西 —— 而圖示能講的話只有這麼多。底色是
:data:`~mia.ui.style.ACCENT`,與開關、「開始遊戲」、Overlay 的強調同一個綠。

為什麼是一個模組而不是在 ``tools/ui.py`` 直接 ``QIcon(path)``
-------------------------------------------------------------
用它的有三處:``QApplication`` 的預設圖示、側邊視窗自己的 ``setWindowIcon``、
以及功能列最上面那個小標誌。**Overlay 不在裡面** —— 它是 frameless 的
``Qt.Tool``,沒有標題列也不進工作列,設了也沒有地方畫。

檔案可能不在
------------
``assets/icon.svg`` 是版控裡的檔案,照理一定在。但這個程式會被 clone 到別的
機器、會從別的工作目錄啟動,而**圖示找不到不該讓程式起不來** —— 它是裝飾,
不是功能。所以 :func:`app_icon` 在檔案不在時回傳空的 :class:`QIcon`,
Qt 看到空圖示就用系統預設,畫面照常。
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtGui import QIcon

from mia.utils.paths import ASSETS_DIR

__all__ = ["ICON_PATH", "app_icon"]

#: 圖示檔。SVG 而不是 PNG:視窗圖示在不同平台、不同 DPI 下被要求的尺寸從
#: 16px 到 1024px 都有,一份向量省掉維護一整組 PNG。
ICON_PATH = ASSETS_DIR / "icon.svg"


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """程式圖示。找不到檔案時回傳空的 :class:`QIcon`(Qt 會退回系統預設)。

    快取一份:每個視窗都要用,而 ``QIcon`` 自己會按需要 render 各種尺寸並
    快取,重複建立等於把那份快取丟掉。
    """
    return QIcon(str(ICON_PATH)) if ICON_PATH.is_file() else QIcon()
