"""把一條牌切成一張一張。

輸入是某個 ROI 裁出來的影像(手牌、副露),輸出是每張牌在該 ROI 內的矩形。
這一步不判斷牌面是什麼 —— 那是模板比對的工作。

用「橘色斜面」而不是亮度或邊緣
------------------------------
最直覺的做法是找牌與牌之間的暗縫,但實測完全不可行:索子的竹紋、萬子的筆畫
在逐行亮度剖面上製造出的暗帶比真正的縫隙還多還深(實測一條 14 張的手牌切出
49 條暗帶)。

牌的**上緣斜面**則完全沒有圖案:實測色相 H≈15、飽和度 S≈114~181 的橘色,
而牌面是 S≈3 的白、桌面是 H≈110 的藍 —— 三者在 HSV 空間離得很開。只取斜面
那幾列做剖面,14 張牌切出來的寬度是 116~117 px(誤差 ±1 px),間距 126~127
完全均勻。

斜面在哪一邊
------------
**暗手牌的橘色斜面在上緣,副露的在下緣** —— 副露是朝另一個方向躺平的,看到的
是牌的底面。這不只是切牌時要注意的細節,它本身就是區分「這是手牌還是副露」
的天然訊號。自家手牌的 ROI 與副露 ROI 在設計上就會重疊(手牌靠左長、副露靠右
長,中間那段共用),所以切牌時**必須**指定要找哪一邊的斜面,否則兩者混在一起
會導致一張都切不出來。

適用範圍
--------
**只適用於自家手牌、自家副露、對面副露**。實測驗證:自家副露在三組時切出 9 張、
四組含槓時切出 13 張,與 GT 完全吻合。另外兩類區域各自因為不同原因不能用:

* **牌河** —— 6xN 的網格,前排會遮掉後排的斜面。實測自家牌河三列的斜面覆蓋率
  分別只有 0.05~0.11、0.03~0.07,只有最前排是 0.77。而且牌面彼此相連,用亮度
  或連通元件也切不開(整片牌河會連成單一元件)。
* **上家/下家的副露** —— 兩個問題。一是對手的暗手牌顯示為**牌背,整片都是橘色**,
  與斜面同色,會把遮罩灌滿;二是側面兩家的副露在螢幕上是**斜向排列**的,不是
  水平也不是垂直,轉置也對不齊。這兩家的副露要改用「白色牌面 vs 橘色牌背」
  來分,幾何則需要透視校正。

這兩類區域都需要固定的網格幾何(四角 + 單應變換),不是逐幀偵測。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from majsoul_copilot.utils.geometry import Rect

__all__ = ["Bevel", "TileStrip", "find_drawn_index", "segment_tiles"]

Bevel = Literal["top", "bottom"]

#: 斜面的橘色在 HSV 的範圍。色相跨越 0 度,所以是兩段。
_HUE_MAX = 32
_HUE_MIN_WRAPPED = 170
_MIN_SATURATION = 90

#: 斜面帶只在 ROI 的這個比例範圍內找。手牌的斜面貼著 ROI 上緣、副露的貼著下緣,
#: 限制搜尋範圍才不會被另一種抓走 —— 兩種 ROI 本來就重疊。
_BAND_SEARCH_RATIO = 0.45

#: 斜面帶的厚度(相對 ROI 高度)。實測 207 px 高的手牌 ROI 斜面約 7 px。
_BAND_THICKNESS_RATIO = 0.05

#: 一張牌至少要有 ROI 高度這個比例的寬度。實測正面手牌約 0.56,
#: 副露因為有透視角度會窄一些(實測 0.46),抓 0.3 當下限有足夠餘裕。
_MIN_TILE_WIDTH_RATIO = 0.30

#: 某一行要有這個比例以上的像素是斜面色,才算落在牌上。
_COLUMN_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class TileStrip:
    """一條牌的切割結果。

    Attributes:
        boxes: 每張牌的矩形,座標**相對傳進來的 ROI 影像**,由左而右。
        bevel_band: 用來切割的斜面帶所在的列範圍 (top, bottom)。診斷用 ——
            切出來的張數不對時,先看這個帶抓對了沒。
    """

    boxes: tuple[Rect, ...]
    bevel_band: tuple[int, int]

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.boxes)

    @property
    def pitches(self) -> tuple[int, ...]:
        """相鄰兩張牌左緣的間距。均勻代表切得乾淨。"""
        return tuple(b.x - a.x for a, b in zip(self.boxes, self.boxes[1:], strict=False))

    def crop(self, image: np.ndarray, index: int) -> np.ndarray:
        """切出第 index 張牌的影像(回傳 view,非複本)。"""
        return image[self.boxes[index].as_slice()]

    def __str__(self) -> str:
        return f"TileStrip({len(self.boxes)} 張, 斜面帶 {self.bevel_band})"


def _bevel_mask(roi: np.ndarray) -> np.ndarray:
    """哪些像素是牌的橘色斜面。"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[..., 0], hsv[..., 1]
    is_orange = (hue <= _HUE_MAX) | (hue >= _HUE_MIN_WRAPPED)
    return is_orange & (saturation > _MIN_SATURATION)


def _find_band(mask: np.ndarray, bevel: Bevel) -> tuple[int, int] | None:
    """找出斜面所在的那幾列。

    只在 ROI 的上緣(或下緣)區段裡找覆蓋率最高的一列,再往兩側取固定厚度。
    不用「覆蓋率超過最大值一半的所有列」那種做法 —— 手牌與副露的斜面同時存在
    時,那樣會把兩條相距一百多列的帶子併成一大塊,結果沒有任何一行的覆蓋率
    過得了門檻,一張都切不出來。
    """
    height = mask.shape[0]
    limit = max(1, int(height * _BAND_SEARCH_RATIO))
    search = slice(0, limit) if bevel == "top" else slice(height - limit, height)

    coverage = mask[search].mean(axis=1)
    if coverage.size == 0 or coverage.max() <= 0.0:
        return None

    peak = int(np.argmax(coverage)) + (0 if bevel == "top" else height - limit)
    half = max(1, int(height * _BAND_THICKNESS_RATIO) // 2)
    return max(0, peak - half), min(height, peak + half + 1)


def _runs(flags: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """連續為 True 的區段 [start, end),過短的丟掉。"""
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    pairs = zip(edges[::2], edges[1::2], strict=True)
    return [(int(a), int(b)) for a, b in pairs if b - a >= min_length]


def segment_tiles(roi: np.ndarray, *, bevel: Bevel = "top") -> TileStrip:
    """把 ROI 影像切成一張一張牌。

    Args:
        roi: 從 :class:`~majsoul_copilot.vision.roi.RoiSet` 切出來的 BGR 影像。
        bevel: 斜面在哪一邊。自家暗手牌 ``top``,自家與對面的副露 ``bottom``
            —— 這兩種 ROI 會重疊,選錯邊通常是一張都切不出來。

    Returns:
        切割結果,由左而右。ROI 裡沒有牌(例如該家還沒副露)時 boxes 為空,
        不是錯誤。
    """
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={roi.shape}")

    height, width = roi.shape[:2]
    mask = _bevel_mask(roi)
    band = _find_band(mask, bevel)
    if band is None:
        return TileStrip((), (0, 0))

    profile = mask[band[0] : band[1]].mean(axis=0) > _COLUMN_THRESHOLD
    min_width = max(2, int(height * _MIN_TILE_WIDTH_RATIO))
    boxes = tuple(
        Rect(start, 0, end - start, height) for start, end in _runs(profile, min_width)
    )
    if not boxes:
        return TileStrip((), band)

    # 牌的垂直範圍就是整個 ROI 高度:ROI 是量測時貼著牌框出來的,再往內縮反而
    # 會把牌的上下緣切掉,對後續的模板比對不利。
    del width
    return TileStrip(boxes, band)


#: 摸進來那張與整理好的手牌之間會空一格。間距超過中位數這個倍率就算「有空隙」。
_DRAW_GAP_RATIO = 1.25


def find_drawn_index(strip: TileStrip) -> int | None:
    """哪一張是剛摸進來的。沒有明顯空隙就回傳 ``None``。

    雀魂會把摸到的牌與整理好的暗手牌隔開一段距離顯示。那張牌的**位置不固定**
    —— 它緊接在暗手牌右側,而暗手牌長度隨副露數改變(無副露時在畫面 x≈0.78,
    四組副露時只剩 1 張暗牌,它就跑到 x≈0.19),所以只能靠這個空隙認,不能靠
    座標。

    對應到 MJAI 就是 ``tsumo`` 那張:知道哪張是新摸的,才分得出「手上有這張牌」
    與「這一巡摸到這張牌」。
    """
    pitches = strip.pitches
    if len(pitches) < 2:
        return None

    median = float(np.median(pitches))
    if median <= 0:
        return None

    # 只認最後一段的空隙。中間出現大間距代表切歪了,不是摸牌。
    if pitches[-1] > median * _DRAW_GAP_RATIO:
        return len(strip.boxes) - 1
    return None
