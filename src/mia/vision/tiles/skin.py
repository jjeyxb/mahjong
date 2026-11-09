"""這一格是角色的手,不是牌。

打牌時角色的手伸過自己的河與手牌右邊摸牌那一格。手掌平滑、又亮,空白的白模板對上它,讀成
**有信心的白**。2026-11-08 真人場:

* 東四局:手掌底下自己的河一次多出兩格,第二格讀成白,收進去就佔掉一個捨牌序號;手收回去之後
  河短了一張,但讀數留著,之後自己打的 6m 被當成白,摸牌與捨牌都錯到局尾。
* 東二局:摸切 8s 時手掌蓋在摸牌那一格,手牌記成「打 8s、摸白」,手收回去又記成「摸切白」,
  憑空多一手,之後每一手都錯開,直到 30 秒後整局重寫。

真的牌只有白是平滑的,而白的牌面是中性的灰白,手掌偏粉紅。八份錄影、四家牌河每一格平滑的
格子:真的白中位的紅減藍 0 ~ 4(最多約 10,手指壓在白上),手掌 18 ~ 28。

所以平滑又偏紅的不收,等手收回去再讀。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SKIN_FLAT", "SKIN_WARM", "is_skin"]

#: 像素標準差低於這個才算平滑。有字的牌 50 以上,白 1 ~ 2,手掌約 10。
SKIN_FLAT = 25.0

#: 中位的紅減藍高於這個算偏紅。
SKIN_WARM = 12.0


def is_skin(patch: np.ndarray) -> bool:
    """``patch`` 是 BGR 的牌面。平滑又偏紅就是手,不是牌。"""
    if patch.size == 0:
        return False
    pixels = patch.reshape(-1, 3).astype(np.float32)
    if float(pixels.std()) >= SKIN_FLAT:
        return False
    blue, _, red = np.median(pixels, axis=0)
    return float(red - blue) > SKIN_WARM
