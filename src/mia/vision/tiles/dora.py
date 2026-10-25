"""左上角那塊面板:寶牌指示牌。

為什麼它比牌河容易
------------------
指示牌是**正面朝鏡頭**的 —— 沒有透視壓縮、沒有旋轉、位置固定、不會被別的牌
擋住。實測一張 ``3s`` 的分數是 **0.929**、差距 **0.391**,而牌河的牌典型是
0.6~0.7。所以這裡不需要牌河那一整套(四邊形攤平、連續性檢查、信心門檻的
取捨),直接切槽、直接比對。

它在事件流裡的角色
------------------
``Dora`` 事件要的不是「哪幾張是寶牌」而是「**指示牌**是什麼」—— 換算成寶牌
是規則的事,不是畫面的事。而槓之後會多翻一張,所以這裡問的是「**現在**翻開了
幾張」,新增的那張就是新的 ``Dora``。

面板的構造(實測 2536x1430 牌桌,面板 374x189)
----------------------------------------------
上面約六成高度是 5 個槽排成一列(1 張起始 + 最多 4 張槓寶),下面是
「立直棒 xN」「本場棒 xN」兩個計數 —— 那兩個不在這裡讀,它們是數字 OCR。
"""

from __future__ import annotations

import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import Match, TemplateSet, classify
from mia.vision.tiles.river import is_face_up

__all__ = ["DORA_FACE", "SLOTS", "SLOT_HEIGHT", "read_dora", "slot_rect"]

#: 槽的數量。1 張起始指示牌 + 最多 4 張槓寶。
SLOTS = 5

#: 槽佔面板高度的比例。下面那塊是立直棒與本場的計數,不是牌。
#:
#: 實測:面板高 189,牌的區域到 y≈112(亮度從 213 掉到 68 的那一列)。
SLOT_HEIGHT = 0.59

#: 牌面佔一個槽的比例 ``(寬, 高)``。
#:
#: **量出來的,不是猜的。** 掃描一張已知是 ``3s`` 的指示牌(槽 75x112):
#:
#: ============  ========  ========
#: 模板尺寸      答案      分數
#: ============  ========  ========
#: **58x92**     **3s**    **0.929**
#: 58x86         3s        0.869
#: 54x92         3s        0.857
#: 66x(任何)     5z        ~0.57  ← 崩掉
#: ============  ========  ========
#:
#: 尖峰很銳利 —— 寬度差 8 px 答案就從 ``3s`` 變成 ``5z``。這裡踩過同樣的坑:
#: 立直橫放牌的模板尺寸原本是「假設寬高交換」,一量就差很多
#: (見 :data:`~mia.vision.tiles.river.SIDEWAYS_FACE`)。
DORA_FACE = (0.77, 0.82)


def slot_rect(panel: tuple[int, int], index: int) -> Rect:
    """第 ``index`` 個槽在面板上的位置。

    Args:
        panel: 面板的 ``(寬, 高)``。
        index: 0 起算,0 是起始指示牌。
    """
    width, height = panel
    slot_width = width / SLOTS
    return Rect(
        round(index * slot_width),
        0,
        round(slot_width),
        round(height * SLOT_HEIGHT),
    )


def read_dora(panel: np.ndarray, templates: TemplateSet) -> tuple[Match | None, ...]:
    """讀整塊面板,回傳每一槽的結果。

    Args:
        panel: ``dora_indicators`` ROI 的 BGR 影像(**整塊**,含下面的棒子
            計數 —— 切槽是這支的工作)。
        templates: 牌面模板集。

    Returns:
        長度 :data:`SLOTS` 的 tuple。沒翻開的槽是 ``None`` ——
        與「翻開了但認不出來」是兩件事,後者會回一個分數低的 ``Match``。

    Note:
        **不判斷哪幾張是寶牌。** 指示牌換算成寶牌是規則的事
        (``9m`` 的下一張是 ``1m``,不是 ``10m``),屬於 ``mia.mjai``。
    """
    if panel.ndim != 3 or panel.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={panel.shape}")
    height, width = panel.shape[:2]
    out: list[Match | None] = []
    for index in range(SLOTS):
        rect = slot_rect((width, height), index)
        slot = panel[rect.as_slice()]
        if slot.size == 0 or not is_face_up(slot):
            out.append(None)
            continue
        size = (round(rect.width * DORA_FACE[0]), round(rect.height * DORA_FACE[1]))
        if size[0] >= rect.width or size[1] >= rect.height:
            out.append(None)
            continue
        out.append(classify(slot, templates, size=size, merge_red=True))
    return tuple(out)
