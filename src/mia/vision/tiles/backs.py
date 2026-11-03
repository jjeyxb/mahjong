"""別家手牌的牌背:這一手是摸切還是手切。

牌河上看不出來
--------------
摸切與手切的牌在牌河裡畫得一模一樣。唯一看得出差別的是**打牌那一刻手牌怎麼動**:別家的
手牌是一排橘色牌背,摸到的那張隔一點空放在最尾端。

* 摸切:尾端那張直接不見,其餘不動。
* 手切:中間空出一格(或一端短一截),尾端那張還在,過一陣子才滑進去補。

手切的這個樣子從打牌開始維持約一秒,0.15 秒看一次牌河的迴圈看得到。

各家量法不同
------------
牌背的橘色跟桌布的藍差很多,一個顏色範圍就切得出來。但打牌時角色的手會伸過手牌,把它切成
好幾段,而三家的角度不一樣:

* 對家、上家:沿手牌畫一條線,量線上哪裡有牌背。手只會蓋住細細一條,形狀分得開。
* 下家:手牌在畫面右側,遠端被透視壓扁,沿線量的話空出來的那格只剩幾個像素,而且手會整條
  橫過手牌。改成數橘色區塊:同一段的牌膨脹一點就連成一塊,手切時分成兩塊、總面積幾乎不少,
  被手蓋住時總面積少一大截。

量測一律換算成「牌桌寬 2560 像素」的單位,門檻才不跟著視窗大小變。座標與門檻見
``docs/cv_mjai.md``「摸切與手切」。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect

__all__ = ["BackShape", "read_backs"]

#: 量測的參考牌桌寬。座標寫成牌桌裡的比例,長度與面積照這個寬換算。
_REFERENCE_WIDTH = 2560

#: 每家手牌那條線,牌桌裡的比例 ``(起點, 終點)``,終點是摸到的那張那一端。
_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "shimocha": ((2368 / 2560, 1086 / 1440), (2085 / 2560, 156 / 1440)),
    "toimen": ((1820 / 2560, 51 / 1440), (850 / 2560, 51 / 1440)),
    "kamicha": ((480 / 2560, 106 / 1440), (225 / 2560, 886 / 1440)),
}

#: 沒有副露時,手牌 13 張在線上的起點與終點(線長的比例)。
#:
#: 副露從線的另一端往回排,牌少了手牌就短了。副露的牌面朝上,沿線量不到,但暗槓的兩張牌背
#: 是橘的,會被當成摸到的那張或手牌的一段;下家那邊每張副露的橘色側面也會連成區塊。真人比
#: 電腦常鳴牌:真人一場別家有兩組副露時摸切只對 68%(沒有副露時 94%)。所以有副露時只看手牌
#: 該在的那一段。
_HAND = {"shimocha": (0.07, 0.85), "toimen": (0.04, 0.875), "kamicha": (0.02, 0.85)}

#: 手牌該結束的地方往後再看多少:摸到的那張,加上透視讓近端的牌比平均寬。實測手牌與副露之間
#: 隔約四張牌,三家量到的副露起點都在這之外。
_ROOM = 0.2

#: 沿線取幾個點。與牌桌大小無關,所以下面以點數為單位的門檻也無關。
_SAMPLES = 400

#: 每個點往線的兩側各看多遠(參考寬下的像素),看幾個位置。
_BAND = 14
_ACROSS = 15

#: 牌背的顏色(OpenCV 的 HSV)。
_ORANGE = ((8, 120, 120), (25, 255, 255))

#: 沿線量的兩家:一個點兩側至少幾個位置是牌背才算有牌、多窄的縫當成牌與牌的接縫、
#: 摸到的那張與手牌之間至少空幾個點。對家的接縫比較深,上家摸到的那張隔得比較近。
_LINE_RULES = {"toimen": (3, 3, 2), "kamicha": (5, 0, 1)}

#: 摸到的那張沿線的長度範圍,點數。
_DRAWN = (12, 45)

#: 下家的區塊:框住手牌的寬度、往兩端多框多少(參考寬下的像素)。
_HALF = 45
_BEYOND = 30

#: 下家的區塊:小於這個面積的不算(參考寬下的平方像素),一張牌約 3400。
_SPECK = 300
_PIECE = 2000


@dataclass(frozen=True, slots=True)
class BackShape:
    """一家手牌這一刻的形狀。

    Attributes:
        body: 手牌本體有多少牌背。沿線量的是點數(不含摸到的那張),數區塊的是面積,單位不同,
            只跟同一家自己比。
        drawn: 摸到的那張隔開放著(只有沿線量的看得出來)。
        pieces: 本體分成幾段。
        clean: 沒被手切成好幾段。沿線量的兩家用;數區塊的一律是 ``True``,遮擋改看總面積。
    """

    body: float
    drawn: bool
    pieces: int
    clean: bool


def read_backs(image: np.ndarray, table: Rect, seat: str, melds: int = 0) -> BackShape:
    """量一家的手牌。

    ``seat`` 是 ``shimocha`` / ``toimen`` / ``kamicha``,``melds`` 是他有幾組副露。
    """
    if seat == "shimocha":
        return _blobs(image, table, seat, melds)
    return _line(image, table, seat, melds)


def _ends(table: Rect, seat: str, melds: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """手牌那條線的兩端。有副露時終點拉回手牌該結束的地方,見 :data:`_HAND`。"""
    (fx0, fy0), (fx1, fy1) = _LINES[seat]
    start = np.array([table.x + fx0 * table.width, table.y + fy0 * table.height])
    end = np.array([table.x + fx1 * table.width, table.y + fy1 * table.height])
    return start, start + (end - start) * _reach(seat, melds)


def _reach(seat: str, melds: int) -> float:
    """線要看到多遠(線長的比例)。沒有副露時是整條。"""
    first, full = _HAND[seat]
    tiles = max(13 - 3 * melds, 1)
    return min(1.0, first + (full - first) * tiles / 13 + _ROOM)


def _orange(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, *_ORANGE)


def _crop(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """框住這些點的那一塊,與它的左上角。只轉這一塊的顏色,整張畫面要十毫秒。"""
    height, width = image.shape[:2]
    left, top = np.floor(points.min(axis=0)).astype(int).clip(0, (width, height))
    right, bottom = np.ceil(points.max(axis=0)).astype(int).clip(0, (width, height))
    return image[top : bottom + 1, left : right + 1], np.array([left, top])


def _line(image: np.ndarray, table: Rect, seat: str, melds: int) -> BackShape:
    scale = table.width / _REFERENCE_WIDTH
    start, end = _ends(table, seat)
    along = end - start
    # 點數照整條線算,門檻才不跟著副露變。看不到的那一段當成沒有牌。
    seen = round(_SAMPLES * _reach(seat, melds))
    normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
    steps = np.linspace(0, 1, _SAMPLES)[:, None, None]
    offsets = np.linspace(-_BAND * scale, _BAND * scale, _ACROSS)[None, :, None]
    points = start + along * steps + normal * offsets
    patch, origin = _crop(image, points.reshape(-1, 2))
    if patch.size == 0:
        return BackShape(0.0, False, 0, False)
    mask = _orange(patch)
    xy = (points - origin).round().astype(int)
    xs = xy[..., 0].clip(0, mask.shape[1] - 1)
    ys = xy[..., 1].clip(0, mask.shape[0] - 1)
    profile = (mask[ys, xs] > 0).sum(axis=1)
    profile[seen:] = 0
    on, seam, gap = _LINE_RULES[seat]
    runs = _runs(profile >= on, seam)
    drawn = (
        len(runs) >= 2
        and _DRAWN[0] <= runs[-1][1] <= _DRAWN[1]
        and runs[-1][0] - sum(runs[-2]) >= gap
    )
    body = runs[:-1] if drawn else runs
    return BackShape(float(sum(n for _, n in body)), drawn, len(body), len(body) <= 2)


def _runs(on: np.ndarray, seam: int) -> list[tuple[int, int]]:
    """連續有牌的幾段 ``(起點, 長度)``。``seam`` 個點以內的縫當成接縫。"""
    index = np.flatnonzero(on)
    if not len(index):
        return []
    runs = [[int(index[0]), int(index[0])]]
    for i in index[1:]:
        if i - runs[-1][1] <= seam + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return [(a, b - a + 1) for a, b in runs]


def _blobs(image: np.ndarray, table: Rect, seat: str, melds: int) -> BackShape:
    scale = table.width / _REFERENCE_WIDTH
    start, end = _ends(table, seat, melds)
    unit = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-unit[1], unit[0]])
    half, beyond = _HALF * scale, _BEYOND * scale
    corners = np.array(
        [
            start - normal * half - unit * beyond,
            start + normal * half - unit * beyond,
            end + normal * half + unit * beyond,
            end - normal * half + unit * beyond,
        ]
    )
    patch, origin = _crop(image, corners)
    if patch.size == 0:
        return BackShape(0.0, False, 0, True)
    region = np.zeros(patch.shape[:2], np.uint8)
    cv2.fillPoly(region, [(corners - origin).round().astype(np.int32)], 255)
    size = max(3, round(5 * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.bitwise_and(cv2.dilate(_orange(patch), kernel), region)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = [stats[i, cv2.CC_STAT_AREA] / scale**2 for i in range(1, count)]
    areas = [area for area in areas if area >= _SPECK]
    pieces = sum(1 for area in areas if area >= _PIECE)
    return BackShape(float(sum(areas)), False, pieces, True)
