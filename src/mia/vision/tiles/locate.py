"""在一塊粗略的範圍裡,找出手牌那一列真正的邊界。

為什麼需要它
------------
:class:`~mia.vision.tiles.hand.HandLayout` 的槽位是**ROI 寬高的固定比例**
(「ROI 是貼著牌量的」),所以 ROI 歪一點,每一個槽位都跟著歪。實測:框得比
真值鬆 10px 還認得對,鬆 30px 就整手變成 ``5z 5z 5z …``
—— **13 張、合法、而且全錯**,正是最危險的那種失敗。

而使用者是用滑鼠在一張縮小的預覽圖上拖框的:預覽縮到 800px 寬時,
一個預覽像素是將近 4 個影像像素,手抖幾像素就進了危險區。所以不能直接用
使用者拖出來的框。

為什麼這次可以用偵測
--------------------
:mod:`mia.calibration.letterbox` 的剝除失敗過(手牌辨識 96% → 29%,
全程無錯誤訊息),所以這個專案對「自動偵測」是有戒心的。差別在**搜尋範圍**:
那條路要在整張畫面裡找牌桌邊界,對手是瀏覽器工具列與深色牌桌之間根本不存在
的對比;這裡只在使用者框的那一小塊裡找,對手是米白的牌面與深色的桌面,
而且使用者已經告訴我們牌在哪了。

實測(2026-10-21,Steam 客戶端 3024x1744):粗框從真值往外鬆 30 / 60 / 100px
三種,**收斂到完全相同的絕對座標**。

還有一層保險:框完之後當場認一次給使用者看。偵測歪掉是**看得見**的,
而看得見的錯誤遠好過看不見的。
"""

from __future__ import annotations

import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.hand import DEFAULT_LAYOUT, HandLayout

__all__ = ["BAND_MARGIN", "PITCH_FROM_HEIGHT", "find_hand_roi"]

#: 找到亮帶之後往外擴多少,佔亮帶高度的比例。
#:
#: 亮帶的邊界是「亮度掉到一半」的地方,而牌的最外圈像素本來就與背景混色,
#: 所以貼著亮帶切會切掉一點點牌。實測(同上那張 3024x1744 的畫面,
#: 亮帶高 227px):往外擴 6px 時平均比對分數 0.934 最高,原樣是 0.787、
#: 擴 16px 掉到 0.732。6/227 ≈ 0.026。
#:
#: 四個邊都用**高度**的比例:水平方向擴同樣的像素數就好,用寬度的比例算
#: 會變成擴 50 多個像素,那就過頭了。
BAND_MARGIN = 0.026

#: 算每一列的亮度時取第幾分位,而不是平均。
#:
#: 平均會被桌面蓋掉:四組副露時手牌只剩一張暗牌,那一列有九成是深色桌面,
#: 平均亮度幾乎與上下沒有牌的列一樣,門檻就切在雜訊上(實測
#: ``hand_4_with_melds`` 量出 181px 高,真值 202)。取高分位數問的是
#: 「這一列**有沒有**夠亮的東西」,那與牌佔多寬無關。
#:
#: 0.9 而不是最大值:最大值對單一個亮點(滑鼠游標、特效的火花)沒有抵抗力。
_ROW_QUANTILE = 0.9

#: 亮帶至少要有這麼高、這麼寬(像素),不然當成沒找到。
#:
#: 擋的是「使用者框到一塊空桌面」:那時亮度幾乎是均勻的,門檻會切在雜訊上,
#: 找出來的「亮帶」只有幾個像素高。回 ``None`` 讓呼叫端說「這裡沒有手牌」,
#: 比回一個幾像素高的框好 —— 後者會一路傳到辨識,然後整手認成白板。
MIN_BAND = 8


def _bright_band(image: np.ndarray) -> Rect | None:
    """在 ``image`` 裡找出手牌那一列。座標相對 ``image`` 自己。

    Args:
        image: 使用者框的那一塊(BGR 或 RGB 都可以 —— 只取亮度)。

    Returns:
        **緊貼著看得到的牌**的矩形。注意這不是 ROI —— 它隨手牌張數變,
        見 :func:`find_hand_roi`。找不到時 ``None``。

    作法是兩次一維剖面,沒有任何形狀假設:

    1. **每一列的平均亮度。** 牌是米白的、桌面是深色的,所以手牌那一列是一條
       明顯的亮帶。取**最長的連續亮段** —— 不是「第一個到最後一個亮的列」,
       因為使用者框大了的話上面可能帶到玩家名牌或分數字樣,那些也是亮的。
    2. **亮帶之內,每一欄的平均亮度。** 這次取第一個到最後一個 —— 牌與牌之間
       有縫,中間本來就會掉下門檻,取最長連續段只會找到其中一張牌。
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return None
    lum = image.astype(np.float32).mean(axis=2)
    height, width = lum.shape

    top, bottom = _longest_run(np.quantile(lum, _ROW_QUANTILE, axis=1))
    if top is None or bottom is None or bottom - top + 1 < MIN_BAND:
        return None

    band = lum[top : bottom + 1]
    left, right = _span(band.mean(axis=0))
    if left is None or right is None or right - left + 1 < MIN_BAND:
        return None

    del width, height
    return Rect(left, top, right - left + 1, bottom - top + 1)


def _threshold(profile: np.ndarray) -> float | None:
    """亮與暗的分界。整段幾乎一樣亮時回 ``None`` —— 那裡沒有東西可分。"""
    low, high = float(profile.min()), float(profile.max())
    # 10 是亮度刻度(0~255)上的「幾乎沒有差別」。擷取本身帶雜訊,
    # 不留這個下限的話,一塊純桌面也會被切成一半亮一半暗。
    return None if high - low < 10 else (low + high) / 2


def _longest_run(profile: np.ndarray) -> tuple[int | None, int | None]:
    """最長的一段「高於門檻」。回傳閉區間 ``(起, 迄)``。"""
    cut = _threshold(profile)
    if cut is None:
        return None, None
    best = best_start = None
    length = 0
    start = 0
    for i, bright in enumerate(profile > cut):
        if bright:
            if length == 0:
                start = i
            length += 1
            if best is None or length > best:
                best, best_start = length, start
        else:
            length = 0
    if best is None or best_start is None:
        return None, None
    return best_start, best_start + best - 1


def _span(profile: np.ndarray) -> tuple[int | None, int | None]:
    """第一個到最後一個「高於門檻」。回傳閉區間。"""
    cut = _threshold(profile)
    if cut is None:
        return None, None
    idx = np.flatnonzero(profile > cut)
    return (int(idx[0]), int(idx[-1])) if idx.size else (None, None)


#: 牌距約是牌高的幾倍。
#:
#: 兩個常數相乘得到:牌面素材是 64x105(切掉上緣之後),所以牌寬 / 牌高 =
#: 0.6095;而 :class:`~mia.vision.tiles.hand.HandLayout` 的 pitch / width =
#: 0.069774 / 0.067291 = 1.0369(牌與牌之間有縫)。0.6095 x 1.0369 = 0.632。
#:
#: 這是一個**物理估計**,不是量測值:牌是長方形的,長寬比不隨視窗大小變。
#: 用它來框住自相關的搜尋範圍 —— 單靠它誤差約 1~2%(ROI 寬會差 40px,
#: 已經足以讓右邊幾張牌認錯),單靠自相關則會鎖到半週期(實測
#: ``hand_4_with_melds`` 只有四張牌時鎖在 65 而不是 126)。
PITCH_FROM_HEIGHT = 0.632

#: 自相關在估計值的多少倍範圍內找峰值。
#:
#: 0.8~1.25 夠寬到容納估計本身的誤差,又窄到排除半週期(0.5)與倍週期(2.0)。
_PITCH_SEARCH = (0.8, 1.25)


def find_hand_roi(image: np.ndarray, layout: HandLayout = DEFAULT_LAYOUT) -> Rect | None:
    """在 ``image`` 裡找出可以直接餵給 :func:`~mia.vision.tiles.hand.read_hand`
    的手牌 ROI。座標相對 ``image`` 自己。

    **亮帶不等於 ROI。** 亮帶是「現在看得到的牌」的範圍,而它隨手牌張數變:
    ``hand_13_full``(13 張、沒摸牌)量到 1651px,但那張素材的 ROI 是 1813px
    —— 右邊空著的那一格也算 ROI 的一部分,因為
    :class:`~mia.vision.tiles.hand.HandLayout` 的槽位是 ROI 寬度的比例。
    拿亮帶當 ROI 的話,四組副露只剩一張暗牌時會算出一個窄到離譜的 ROI。

    所以**從牌距還原**:牌距與張數無關。先用牌的長寬比估一個
    (:data:`PITCH_FROM_HEIGHT`),再用欄亮度的自相關在附近找真正的峰值。

    .. warning::
       **框選時手牌要是完整的暗手牌,不要有副露。** 副露那幾組在畫面上貼得
       比暗手牌緊,而且與暗手牌之間隔著一大段空桌面 —— 兩群不同牌距的東西
       讓自相關找不到單一的週期(實測 ``hand_4_with_melds``:四張暗牌加七張
       副露,量出來的牌距是真值的 0.74 倍)。

       這只影響**框選那一刻**。框好之後 ROI 是固定的,之後副露照樣認得出來
       —— 那本來就是 :class:`~mia.vision.tiles.hand.HandLayout` 在處理的事。

       所以這個函式不保證回傳的框是對的,呼叫端**一定要當場認一次給使用者看**。
       認出 ``5z 5z 5z …`` 或「不是合法手牌」就是框歪了,而那是看得見的。
    """
    band = _bright_band(image)
    if band is None:
        return None

    crop = image[band.y : band.y + band.height, band.x : band.x + band.width]
    pitch = _pitch(crop.astype(np.float32).mean(axis=2), band.height)
    if pitch is None:
        return None

    width = round(pitch / layout.pitch)
    # 亮帶的左緣是第一張牌的左緣,而 ROI 左緣在它左邊 origin x 寬度處。
    x = max(0, band.x - round(layout.origin * width))
    width = min(width, image.shape[1] - x)
    margin = round(band.height * BAND_MARGIN)
    y = max(0, band.y - margin)
    height = min(band.height + 2 * margin, image.shape[0] - y)
    return Rect(x, y, width, height)


def _pitch(band: np.ndarray, height: int) -> float | None:
    """牌距(像素)。``band`` 是亮帶的亮度圖。"""
    guess = height * PITCH_FROM_HEIGHT
    lo = max(2, round(guess * _PITCH_SEARCH[0]))
    hi = round(guess * _PITCH_SEARCH[1])
    cols = band.mean(axis=0)
    centred = cols - cols.mean()
    acf = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    if acf[0] <= 0 or hi <= lo or hi >= len(acf):
        # 框得比一個牌距還窄 —— 那不是手牌,是使用者框到別的東西了
        return None
    return float(lo + int(np.argmax(acf[lo:hi])))
