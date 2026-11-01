"""Steam 客戶端(Unity)全螢幕的版面。

Mac 全螢幕的視窗比例約 1.59,不是 16:9。但遊戲不是換了排版,而是**同一個 16:9 的場景
再多露出上下的桌面**:拿牌河的認牌分數搜牌桌矩形,放開寬高比之後收斂到 1.78。所以找到
那個 16:9 的矩形,牌河、副露、手牌、中央 HUD 的座標都不用重量。矩形比視窗寬一點,
左右會超出畫面,這是正常的。

左上角的寶牌與立直棒面板例外:它貼著視窗角落,不跟著場景,所以另外記位置。

這些比例量自一場全螢幕錄影(3024x1898),``docs/cv_mjai.md`` §4.4。其他比例沒有量過:
一般視窗實機量過 1.79,版面還不知道。比例對不上就回 ``None``,不要硬套。
"""

from __future__ import annotations

from mia.utils.geometry import Rect, Size

__all__ = ["unity_layout"]

#: 量測時的影像。
_MEASURED = Size(3024, 1898)

#: 比例差多少以內算同一種版面。
_ASPECT_TOLERANCE = 0.02

#: 16:9 的場景矩形,像素是量測時的影像。
_TABLE = Rect(-8, 102, 3040, 1710)

#: 左上角面板。
_PANEL = Rect(30, 65, 470, 215)


def unity_layout(size: Size) -> tuple[Rect, Rect] | None:
    """這個影像大小的牌桌矩形與左上角面板。比例沒量過回 ``None``。"""
    measured = _MEASURED.width / _MEASURED.height
    if abs(size.width / size.height - measured) / measured > _ASPECT_TOLERANCE:
        return None
    return _scaled(_TABLE, size), _scaled(_PANEL, size)


def _scaled(rect: Rect, size: Size) -> Rect:
    across = size.width / _MEASURED.width
    down = size.height / _MEASURED.height
    return Rect(
        round(rect.x * across),
        round(rect.y * down),
        round(rect.width * across),
        round(rect.height * down),
    )
