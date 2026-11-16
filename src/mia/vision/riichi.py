"""上家宣告立直時的「立直」特效字。

為什麼要看特效字
----------------
立直本來靠兩條路認:橫放的宣言牌(要靠後面的牌驗,宣言牌是當列最後一張時驗不了),與點數少
1000(:class:`~mia.tracker.round.ReachWatcher`)。點數要等宣言牌沒人榮和、**下一家摸牌**時才扣,
而上家的下一家就是自己:29 場 52 次上家立直,點數那條路排進事件流時平均已經比自己摸牌晚
0.7 秒。這時引擎看到的是「上家沒有立直」,15 次因此給了不同的答案(該打現物時照常進攻)。

特效字在宣言牌落下之前就出現了(宣言後 0.13 ~ 0.3 秒,宣言牌排進事件流前至少 0.26 秒),
停留約 0.6 秒。只看上家:另外三家輪到自己之前至少隔著一家,點數那條路來得及。

量出來的
--------
字是固定的(奶油色、深色描邊),旁邊的立繪隨角色不同,所以樣板只切字。牌桌寬縮成
:data:`_WIDTH`、高等比例,在 :data:`_WINDOW` 裡找最像的位置(``TM_CCOEFF_NORMED``)。

* 29 場 52 次上家立直,每一次都過門檻,最高分最低 0.993。
* 其他 47.7 萬幀(別家的立直、碰吃槓與和牌的特效字、角色的手、結算、大廳)最高 0.435,
  第一局到最後一局之間最高 0.413。
* Mac 的 Steam 版(牌桌比例 1.59,不是 16:9)照寬度縮放之後同一個位置、同樣大小,0.99。所以只照
  寬度縮放。Windows 的 Steam 版全螢幕是 16:9,跟瀏覽器一樣。

自己立直的特效字在下面(y 約 550),不在 :data:`_WINDOW` 裡。
"""

from __future__ import annotations

from functools import cache

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.utils.paths import ASSETS_DIR

__all__ = ["kamicha_riichi", "riichi_score"]

#: 牌桌寬縮成這麼多再比,高照比例。
_WIDTH = 1280

#: 搜尋範圍 ``(x0, y0, x1, y1)``,用縮過的牌桌座標。字在 (270, 345) 左右,四周各留 40 以上。
_WINDOW = (215, 300, 440, 470)

#: 實測真的特效字 0.993 以上,其他畫面 0.435 以下。
_SCORE = 0.7


@cache
def _template() -> np.ndarray:
    path = ASSETS_DIR / "banner" / "riichi.png"
    face = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if face is None:
        raise FileNotFoundError(f"讀不到模板 {path}")
    return face


def riichi_score(image: np.ndarray, table: Rect) -> float:
    """上家那一塊有多像「立直」兩個字,-1 ~ 1。切不到東西回 0。"""
    scale = table.width / _WIDTH
    x0, y0, x1, y1 = _WINDOW
    patch = image[
        table.y + round(y0 * scale) : table.y + round(y1 * scale),
        table.x + round(x0 * scale) : table.x + round(x1 * scale),
    ]
    template = _template()
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gray = cv2.resize(gray, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())


def kamicha_riichi(image: np.ndarray, table: Rect) -> bool:
    """上家正在宣告立直。一次約 0.2 毫秒。"""
    return riichi_score(image, table) >= _SCORE
