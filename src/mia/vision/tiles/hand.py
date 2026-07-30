"""讀出自家手牌:每一張在哪、有幾張、哪張是剛摸的。

輸入是 ``own_hand`` ROI 裁出來的影像,輸出是每張牌的矩形。這一步不判斷牌面
是什麼 —— 那是模板比對的工作。

手牌的位置是**固定的**,不需要偵測
------------------------------------
雀魂的暗手牌永遠靠左對齊在同一個位置,張數與副露數都不影響起點::

    x_k = origin + k * pitch

實測三張素材(13 張、14 張含摸牌、4 張含三組副露),殘差都 **< 1 px**。
**副露不會把手牌推走** —— 副露是從畫面右緣往左長的,兩者各據一端。

所以「第 k 張牌在哪」是查表,不是偵測。剩下要回答的只有兩件事:

1. **有幾張** —— 從槽位 0 往右掃,第一個空槽就是結尾。
2. **最後那張是不是剛摸的** —— 摸牌會與整理好的手牌空一格,實測穩定偏離
   槽位 ``draw_gap``。

為什麼用紋理而不是顏色判斷槽位有沒有牌
--------------------------------------
最初的做法是找牌的橘色斜面(H≈16)。它切得很準,但那個橘色**就是可以被玩家
自訂的牌背材質** —— 實測色相只要偏 20 度就一張都切不出來,而且是靜默失敗。

改用標準差就沒有這個問題:牌一定有邊緣,桌布是平的。實測(整張牌高)::

    一般牌(暗手牌或副露)  標準差 55.3 ~ 94.9
    白板(牌面整片空白)    標準差 35.6   ← 下限
    空桌面                  標準差 21.1 ~ 23.0

**取樣一定要含整張牌高**,不能只取牌面中段。第一版只取中間 35%~85%,遇到
白板時該區域是一整片均勻的白,標準差量到 **0.0**,直接把摸進來的白板判成空槽。
含上下緣之後,白板靠的是牌緣與斜面的邊界 —— 那是每張牌都有的結構,與牌面
圖案無關。

白板是這個判據的下限(35.6),空桌面是上限(23.0),門檻取在中間。餘裕不算大,
所以若日後發現漏讀,先查是不是取樣範圍又被縮小了。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect

__all__ = ["DEFAULT_LAYOUT", "MAX_CONCEALED", "Hand", "HandLayout", "read_hand"]

#: 暗手牌最多 13 張(摸進來的那張另外算)。四組副露時只剩 1 張。
MAX_CONCEALED = 13

#: 判定槽位有牌的標準差門檻。實測空桌面上限 23.0、白板下限 35.6,取在中間。
_OCCUPIED_STD = 30.0

#: 水平方向從牌的兩側各內縮這個比例,避開牌與牌交界的陰影。
#: 相鄰槽位間距比牌寬多約 4.5 px,內縮 10% 後不會取到隔壁那張。
#: **垂直方向不內縮** —— 白板的牌面中段是一整片均勻的白,不含上下緣就量不到
#: 任何變化(實測標準差 0.0)。
_SAMPLE_INSET = 0.10


@dataclass(frozen=True, slots=True)
class HandLayout:
    """手牌在 ROI 內的固定幾何,全部是**相對 ROI 寬度**的比例。

    用比例而不是像素,是為了跟著視窗大小縮放 —— 與 ROI 本身用正規化座標
    儲存是同一個理由。

    量測條件:``own_hand`` ROI 於 2560x1440 畫布下裁出 1813x207,
    以 13 張滿手的素材線性擬合(見 ``docs/decisions.md``)。
    """

    #: 第 0 張牌的左緣。實測 4.23 px / 1813。
    origin: float = 0.002333
    #: 相鄰兩張牌左緣的間距。實測 126.50 px / 1813。
    pitch: float = 0.069774
    #: 一張牌的寬度。實測 122 px / 1813,略小於 pitch(牌與牌之間有縫)。
    width: float = 0.067291
    #: 摸牌那張額外偏離槽位的距離。實測 39 px / 1813。
    draw_gap: float = 0.021512

    def slot(self, roi_width: int, roi_height: int, index: int) -> Rect:
        """第 ``index`` 個槽位的矩形。高度就是整個 ROI —— ROI 是貼著牌量的。"""
        x = round((self.origin + index * self.pitch) * roi_width)
        return Rect(x, 0, round(self.width * roi_width), roi_height)

    def drawn_slot(self, roi_width: int, roi_height: int, concealed: int) -> Rect:
        """有 ``concealed`` 張暗手牌時,剛摸進來那張會在哪。

        它接在暗手牌右側再空一格,所以位置隨暗手牌長度移動 —— 無副露時在
        槽位 13、四組副露時在槽位 1,沒有固定座標可以直接框。
        """
        x = round(
            (self.origin + concealed * self.pitch + self.draw_gap) * roi_width
        )
        return Rect(x, 0, round(self.width * roi_width), roi_height)


DEFAULT_LAYOUT = HandLayout()


@dataclass(frozen=True, slots=True)
class Hand:
    """一幀裡讀到的自家手牌。

    Attributes:
        concealed: 整理好的暗手牌,由左而右。座標相對傳進來的 ROI 影像。
        drawn: 剛摸進來那張;不是自己的回合(或還沒摸)時為 ``None``。
            **刻意與 concealed 分開** —— 對應 MJAI 的 ``tsumo``,
            「手上有這張牌」與「這一巡摸到這張牌」是兩件事。
    """

    concealed: tuple[Rect, ...]
    drawn: Rect | None = None

    def __len__(self) -> int:
        """暗手牌張數,**不含**摸牌那張。"""
        return len(self.concealed)

    @property
    def total(self) -> int:
        """含摸牌那張的總張數。"""
        return len(self.concealed) + (1 if self.drawn else 0)

    @property
    def melds(self) -> int:
        """副露組數。

        由暗手牌張數反推 —— 副露的**內容**不需要辨識,向聽數只要知道組數。
        張數不是 13/10/7/4/1 時回傳 -1,代表這一幀讀到的手牌不合法
        (通常是抓在理牌或摸打的動畫中間)。
        """
        remainder = MAX_CONCEALED - len(self.concealed)
        return remainder // 3 if remainder % 3 == 0 else -1

    @property
    def is_plausible(self) -> bool:
        """張數是否構成一副合法的手牌。不合法就該丟掉這一幀重讀。"""
        return 0 < len(self.concealed) <= MAX_CONCEALED and self.melds >= 0

    def __str__(self) -> str:
        tail = " +摸牌" if self.drawn else ""
        return f"Hand({len(self.concealed)} 張{tail}, 副露 {self.melds} 組)"


def _has_tile(roi: np.ndarray, box: Rect) -> bool:
    """這個槽位上有沒有牌。

    判據是**整張牌高**的灰階標準差 —— 牌一定有邊緣,桌布是平的。刻意不看顏色:
    牌背與牌面都能被玩家換皮膚,邊緣則是每張牌都有的結構。
    """
    width = roi.shape[1]
    inset = round(box.width * _SAMPLE_INSET)
    x0 = max(0, box.x + inset)
    x1 = min(width, box.x + box.width - inset)
    if x1 - x0 < 2:
        return False

    grey = cv2.cvtColor(roi[:, x0:x1], cv2.COLOR_BGR2GRAY)
    return bool(grey.std() > _OCCUPIED_STD)


def read_hand(roi: np.ndarray, *, layout: HandLayout = DEFAULT_LAYOUT) -> Hand:
    """從 ``own_hand`` ROI 影像讀出手牌。

    Args:
        roi: 從 :class:`~mia.vision.roi.RoiSet` 切出來的 BGR 影像。
        layout: 槽位幾何。預設值是實測的,只有換皮膚改變牌的尺寸時才需要動。

    Returns:
        讀到的手牌。ROI 裡沒有牌(對局還沒開始、正在發牌動畫)時
        ``concealed`` 為空,不是錯誤。

    Note:
        掃到**第一個空槽就停**。理牌動畫進行到一半時中間可能出現空隙,這時會
        少報幾張 —— 這是刻意的:少報會被 :attr:`Hand.is_plausible` 擋下來
        重讀,硬跨過空隙補齊則會產生一副看起來合法、實際上錯的手牌。
    """
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={roi.shape}")

    height, width = roi.shape[:2]
    concealed: list[Rect] = []
    for index in range(MAX_CONCEALED):
        box = layout.slot(width, height, index)
        if box.x + box.width > width or not _has_tile(roi, box):
            break
        concealed.append(box)

    drawn: Rect | None = None
    if concealed:
        candidate = layout.drawn_slot(width, height, len(concealed))
        if candidate.x + candidate.width <= width and _has_tile(roi, candidate):
            drawn = candidate

    return Hand(tuple(concealed), drawn)
