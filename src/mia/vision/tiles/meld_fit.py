"""認副露的另一條路:先列出這組可能是什麼,再到畫面上比哪一個最像。

:func:`~mia.vision.tiles.melds.read_meld_faces` 是**從畫面認牌**:掃整段帶子,每一框問
「這是 37 種牌的哪一種」。副露小、斜、橫放那張又躺著,認不全是常態:十四份錄影別家 84 組
要選的吃碰(從上家鳴的數牌),它只認得全 43 組,其餘只能靠張數猜。

但看副露之前,大部分答案已經知道了:

* **被鳴的是哪張**:牌河剛少掉的那張。河裡的牌是直的、大的,有把握的實測 99.7% 對。
* **從誰那裡鳴的**:排序器知道。它決定橫放那張在哪一格:上家的放最左、對家的放中間、
  下家的放最右,吃一定是上家的,排成 [被吃的, 小, 大]。
* **吃、碰還是槓**:長度。

剩下的只有幾種可能:碰一種,吃最多三種(被鳴的那張在順子的低、中、高)。所以這裡
不問「這是什麼牌」,只問「這組是 A 還是 B」:照每個候選的排法,把該在每一格的那張牌的
模板拿去比,整組一起挪、一起縮放,分數加起來最高的就是它。

為什麼這樣比較準
----------------
* **選項少。** 三選一不必每張都認清楚,差的那一兩張分數拉得開就夠了。
* **整組一起對齊。** 逐張切框的雛型,框偏一點就認不出來,只認到 58%。這裡一個位移、
  一個牌寬套在整組上,所有牌一起投票,連橫放那張也算進去。
* **不會湊出不存在的組合。** 每個候選都是一組合法的副露,錯也只會錯成另一個候選,而
  差距不夠的時候 :func:`pick` 棄權,上層照舊用 :func:`read_meld_faces` 的推法。

十四份錄影(真人一場、CPU 十二場、網頁版一場)別家 84 組要選的吃碰:**84 組全對**,
差距最小 0.041,門檻 :data:`_MARGIN` 取 0.05,棄權 2 組、錯 0 組。

座標不用副露帶子的格子
----------------------
帶子的角點是為了量長度搜的(:data:`~mia.vision.tiles.melds.DEFAULT_VIEWS`),一格不等於
一張牌,而且**上一組的結尾**常常量偏:真人那場下家的第二組,上一組量到的終點落在 4 萬
正中間。量得準的只有**最新那端的邊**:從那裡沿著帶子往內,在牌河的攤平平面上排牌。
牌寬、起點、上下的位移都讓搜尋決定(:data:`_PITCH`、:data:`_ALONG`、:data:`_ACROSS`):
對家的帶子中線壓在牌的頂面上,不往下挪 45 像素,分數只有 0.2。

赤五
----
排好之後每一張牌在哪裡就知道了,在那一格把模板對到的牌面切下來,問墨水有多少是紅的。
舊的做法掃整組找「最紅的那一框」,框歪的時候會框到萬字的下半部(一到九萬的「萬」都是
紅的),普通五萬到 0.75,赤五萬只有 0.74。對到牌面再量就分得開,見 :data:`RED`。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet, ink_masks
from mia.vision.tiles.melds import _SIZES, _WINDOW, CELL, DEFAULT_VIEWS, EXTENT
from mia.vision.tiles.river import CELL as RIVER_CELL
from mia.vision.tiles.river import COLUMNS as RIVER_COLUMNS
from mia.vision.tiles.river import DEFAULT_CORNERS, ROWS

__all__ = ["RED", "SIDEWAYS", "Fit", "MeldPicture", "Pick", "pick"]

#: 橫放的牌比直的寬幾倍。八組副露量的。
SIDEWAYS = 1.4

#: 直放的牌在牌河平面上寬幾像素,依座位。十四份錄影搜到的落在 60 ~ 105。
_PITCH: dict[str, tuple[int, int]] = {
    "shimocha": (75, 125),
    "toimen": (65, 115),
    "kamicha": (50, 90),
}

#: 最新那端的邊往內外挪多少像素。
_ALONG = 48

#: 垂直帶子的方向挪多少像素。對家要 45 左右,見模組說明。
_ACROSS = 60

#: 粗搜的步距(沿帶子、牌寬、垂直帶子),之後在最好的附近用一半、四分之一的步距細搜。
_COARSE = (12, 10, 15)
_FINE = (6, 3)

#: 最像的候選至少要贏第二名這麼多(每張分數的總和)。十四份錄影對的差距最小 0.041,一組錯的都
#: 沒有;留一點餘裕,84 組裡棄權 2 組。
_MARGIN = 0.05

#: 贏過其他候選還不夠,整組平均分數也要過這一關。真人場角色的手蓋住副露時,
#: 錯的候選以 0.33 分贏 0.08,反而蓋掉原本正確的吃牌。十四份錄影 156 組的最佳
#: 平均分數最低 0.48,第二清楚的那張也至少 0.48。取 0.45,並要求至少兩張過關,
#: 不讓唯一一張清楚的牌撐起整組分數;看不清楚就等下一張畫面。
_MIN_SCORE = 0.45

#: 牌面的墨水裡紅色佔這麼多以上就是赤五。十四份錄影 31 組有五的副露:
#:
#: ====  ==========  ==========
#: 花色  普通五最高  赤五最低
#: ====  ==========  ==========
#: 萬    0.46        0.81
#: 筒    0.14        0.81
#: 索    0.33        0.69
#: ====  ==========  ==========
RED = 0.6

#: 量紅色時牌面四邊各切掉多少,擋掉牌邊與桌布。
_RED_INSET = 0.1

#: 一組最長有幾張直牌寬。大明槓三直一橫。
_LONGEST = 3 + SIDEWAYS


@dataclass(frozen=True, slots=True)
class Fit:
    """一個候選對在畫面上的結果。

    Attributes:
        score: 每張分數的總和。
        scores: 每一張的分數,由左而右。
        centres: 每一張的中心,在 :class:`MeldPicture` 的影像座標上。
    """

    score: float
    scores: tuple[float, ...]
    centres: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class Pick:
    """:func:`pick` 選出來的候選。

    Attributes:
        index: 第幾個候選。
        reds: 每一張是不是赤五,由左而右。不是五的一律 ``False``。
        margin: 贏第二名多少。只有一個候選時是無限大。
        quality: 整組平均模板分數,重拍時用來保留較清楚的答案。
    """

    index: int
    reds: tuple[bool, ...]
    margin: float
    quality: float = 1.0


class MeldPicture:
    """一組副露在牌河攤平平面上的影像,只留這一組可能在的範圍。

    ``end`` 是這一組最新那端的邊,:func:`~mia.vision.tiles.melds.occupied_span` 的單位。
    建構只做一次透視變換,便宜;比對(:meth:`fit`)才花時間,可以丟到別的執行緒。
    """

    def __init__(self, image: np.ndarray, table: Rect, seat: str, *, end: float) -> None:
        self.seat = seat
        river_width, river_height = RIVER_CELL
        target = np.array(
            [
                [0, 0],
                [RIVER_COLUMNS * river_width, 0],
                [RIVER_COLUMNS * river_width, ROWS * river_height],
                [0, ROWS * river_height],
            ],
            dtype=np.float32,
        )
        to_plane = cv2.getPerspectiveTransform(DEFAULT_CORNERS[seat].quad(table), target)
        from_strip = np.linalg.inv(DEFAULT_VIEWS[seat].matrix(table, columns=EXTENT))
        width, height = CELL
        edge = np.array([[(end * width, height / 2), ((end - 1) * width, height / 2)]])
        start, inward = cv2.perspectiveTransform(edge.astype(np.float32), to_plane @ from_strip)[0]
        unit = (inward - start) / np.linalg.norm(inward - start)
        normal = np.array([-unit[1], unit[0]])
        window_width, window_height = _WINDOW
        widest = _PITCH[seat][1]
        corners = np.array(
            [
                start + unit * along + normal * across
                for along in (-_ALONG - window_width, _ALONG + _LONGEST * widest + window_width)
                for across in (-_ACROSS - window_height, _ACROSS + window_height)
            ]
        )
        left, top = np.floor(corners.min(axis=0))
        right, bottom = np.ceil(corners.max(axis=0))
        shift = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
        size = (int(right - left), int(bottom - top))
        self.image = cv2.warpPerspective(image, shift @ to_plane, size)
        self._start = start - (left, top)
        self._unit = unit
        self._normal = normal
        self._maps: dict[tuple[str, bool, int], list[tuple[np.ndarray, int, int]]] = {}
        self._ranges: dict[tuple[str, bool], tuple[int, int, int, int]] = {}

    def fit(self, tiles: Sequence[str], sideways: int, templates: TemplateSet) -> Fit:
        """``tiles``(雀魂記法,由左而右)照這個排法最像的對齊。

        先粗搜,再在最好的那組參數附近細搜兩次。分數是每張牌在它那一格的模板分數,
        取整組的總和。
        """
        self._prepare(tiles, sideways)
        low, high = _PITCH[self.seat]
        step_along, step_pitch, step_across = _COARSE
        grid: list[tuple[float, float, float]] = [
            (along, pitch, across)
            for along in range(-_ALONG, _ALONG + 1, step_along)
            for pitch in range(low, high + 1, step_pitch)
            for across in range(-_ACROSS, _ACROSS + 1, step_across)
        ]
        best, params = max(
            ((self._try(tiles, sideways, templates, p), p) for p in grid), key=lambda t: t[0].score
        )
        for step in _FINE:
            along, pitch, across = params
            near = [
                (along + i * step, pitch + j * step / 2, across + k * step)
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                for k in (-1, 0, 1)
            ]
            best, params = max(
                [(best, params)] + [(self._try(tiles, sideways, templates, p), p) for p in near],
                key=lambda t: t[0].score,
            )
        return best

    def red(
        self, fit: Fit, index: int, label: str, templates: TemplateSet, *, turned: bool
    ) -> float:
        """第 ``index`` 張(``label``)的牌面裡,墨水有多少是紅的。"""
        face = self._face(fit.centres[index], label, templates, turned=turned)
        if face is None:
            return 0.0
        height, width = face.shape[:2]
        dy, dx = int(height * _RED_INSET), int(width * _RED_INSET)
        ink, red = ink_masks(face[dy : height - dy, dx : width - dx])
        total = int(np.count_nonzero(ink))
        return np.count_nonzero(red) / total if total else 0.0

    def layout(
        self, count: int, sideways: int, params: tuple[float, float, float]
    ) -> tuple[tuple[float, float], ...]:
        """``count`` 張牌(第 ``sideways`` 張橫放)的中心。

        ``params`` 是 ``(沿帶子往內挪, 直牌的寬, 垂直帶子挪)``,單位是 :attr:`image` 的像素。
        """
        along, pitch, across = params
        origin = self._start + self._normal * across
        out = []
        for index in range(count):
            width = SIDEWAYS * pitch if index == sideways else pitch
            x, y = origin + self._unit * (along + width / 2)
            out.append((float(x), float(y)))
            along += width
        return tuple(out)

    # ------------------------------------------------------------------ 內部

    def _prepare(self, tiles: Sequence[str], sideways: int) -> None:
        """每張牌只搜尋排法允許的範圍,包含粗搜、兩輪細搜與框內平移的全部餘裕。

        橫放牌只會在一組裡的某一格,卻原本把整組帶子的每個像素都比一遍。
        先列出所有候選的範圍再做比對,分數與搜尋自由度不變,只省掉不會查到的位置。
        """
        low, high = _PITCH[self.seat]
        fine = sum(_FINE)
        for index, label in enumerate(tiles):
            turned = index == sideways
            distance = index + (SIDEWAYS / 2 if turned else 0.5)
            if sideways < index:
                distance += SIDEWAYS - 1
            centres = np.array(
                [
                    self._start + self._unit * (along + distance * pitch) + self._normal * across
                    for along in (-_ALONG - fine, _ALONG + fine)
                    for pitch in (low - fine / 2, high + fine / 2)
                    for across in (-_ACROSS - fine, _ACROSS + fine)
                ]
            )
            window = np.array(_WINDOW[::-1] if turned else _WINDOW) / 2 + 2
            left, top = np.maximum(np.floor(centres.min(axis=0) - window), 0).astype(int)
            right, bottom = np.minimum(
                np.ceil(centres.max(axis=0) + window), self.image.shape[1::-1]
            ).astype(int)
            key = (label, turned)
            before = self._ranges.get(key)
            bounds = (int(left), int(top), int(right), int(bottom))
            if before is not None:
                bounds = (
                    min(before[0], bounds[0]), min(before[1], bounds[1]),
                    max(before[2], bounds[2]), max(before[3], bounds[3]),
                )
            if bounds != before:
                self._ranges[key] = bounds
                for cached in list(self._maps):
                    if cached[:2] == key:
                        del self._maps[cached]

    def _try(
        self,
        tiles: Sequence[str],
        sideways: int,
        templates: TemplateSet,
        params: tuple[float, float, float],
    ) -> Fit:
        centres = self.layout(len(tiles), sideways, params)
        scores = tuple(
            self._score(centre, label, templates, turned=index == sideways)
            for index, (centre, label) in enumerate(zip(centres, tiles, strict=True))
        )
        return Fit(sum(scores), scores, centres)

    def _score(
        self, centre: tuple[float, float], label: str, templates: TemplateSet, *, turned: bool
    ) -> float:
        best = -1.0
        for response, offset_x, offset_y in self._response(label, templates, turned=turned):
            x = round(centre[0]) - offset_x
            y = round(centre[1]) - offset_y
            if 0 <= y < response.shape[0] and 0 <= x < response.shape[1]:
                best = max(best, float(response[y, x]))
        return best

    def _response(
        self, label: str, templates: TemplateSet, *, turned: bool
    ) -> list[tuple[np.ndarray, int, int]]:
        """這種牌的模板在整張影像上的分數圖,已經對 :data:`_WINDOW` 的餘裕取過最大值。

        一個候選要試上千種排法,每一種逐框比模板要好幾秒;同一種牌在可查到的範圍內比一次,
        之後每一框的分數只是查表。取最大值是為了跟逐框比的結果一樣:一框裡模板可以
        在餘裕內任意對齊。
        """
        key = (label, turned, id(templates))
        cached = self._maps.get(key)
        if cached is not None:
            return cached
        left, top, right, bottom = self._ranges[(label, turned)]
        image = self.image[top:bottom, left:right]
        window_width, window_height = _WINDOW
        out = []
        for size in _SIZES[self.seat]:
            width, height = size[::-1] if turned else size
            slack_x, slack_y = (
                ((window_height - width) // 2, (window_width - height) // 2)
                if turned
                else ((window_width - width) // 2, (window_height - height) // 2)
            )
            kernel = np.ones((2 * slack_y + 1, 2 * slack_x + 1), np.uint8)
            for face in _faces(label, templates, size, turned=turned):
                response = cv2.matchTemplate(image, face, cv2.TM_CCOEFF_NORMED)
                out.append((cv2.dilate(response, kernel), left + width // 2, top + height // 2))
        self._maps[key] = out
        return out

    def _face(
        self, centre: tuple[float, float], label: str, templates: TemplateSet, *, turned: bool
    ) -> np.ndarray | None:
        """``centre`` 那一框裡模板對到的那塊牌面。"""
        window_width, window_height = _WINDOW
        if turned:
            window_width, window_height = window_height, window_width
        x = max(int(centre[0]) - window_width // 2, 0)
        y = max(int(centre[1]) - window_height // 2, 0)
        window = self.image[y : y + window_height, x : x + window_width]
        best: tuple[float, np.ndarray | None] = (-1.0, None)
        for size in _SIZES[self.seat]:
            for face in _faces(label, templates, size, turned=turned):
                height, width = face.shape[:2]
                if window.shape[0] < height or window.shape[1] < width:
                    continue
                _, score, _, at = cv2.minMaxLoc(
                    cv2.matchTemplate(window, face, cv2.TM_CCOEFF_NORMED)
                )
                if score > best[0]:
                    best = (score, window[at[1] : at[1] + height, at[0] : at[0] + width])
        return best[1]


def pick(
    picture: MeldPicture,
    shapes: Sequence[tuple[Sequence[str], int]],
    templates: TemplateSet,
) -> Pick | None:
    """``shapes`` 裡哪一個最像畫面。畫面不夠像、差距不足時棄權。

    Args:
        picture: 這一組的影像。
        shapes: 候選,``(由左而右的牌, 橫放的是第幾張)``,雀魂記法、不分赤。
        templates: 牌河那份模板。

    Returns:
        選出來的候選與它的赤五。只有一個候選時仍要看清楚牌面,才能量赤五。
    """
    if not shapes:
        return None
    for tiles, sideways in shapes:
        picture._prepare(tiles, sideways)  # noqa: SLF001 - 先合併同一張圖的所有候選範圍
    fits = [picture.fit(tiles, sideways, templates) for tiles, sideways in shapes]
    ranked = sorted(range(len(fits)), key=lambda i: fits[i].score, reverse=True)
    margin = fits[ranked[0]].score - fits[ranked[1]].score if len(fits) > 1 else float("inf")
    index = ranked[0]
    if (
        fits[index].score < _MIN_SCORE * len(shapes[index][0])
        or sum(score >= _MIN_SCORE for score in fits[index].scores) < 2
        or margin < _MARGIN
    ):
        return None
    tiles, sideways = shapes[index]
    reds = tuple(
        tile[0] == "5"
        and bool(picture.red(fits[index], i, tile, templates, turned=i == sideways) >= RED)
        for i, tile in enumerate(tiles)
    )
    return Pick(index, reds, margin, fits[index].score / len(tiles))


def _faces(
    label: str, templates: TemplateSet, size: tuple[int, int], *, turned: bool
) -> list[np.ndarray]:
    """``label`` 的模板(含赤五與牌河那份),橫放的轉兩個方向。"""
    wanted = {label, f"0{label[1]}"} if label[0] == "5" else {label}
    faces = [face for name, face in templates.scaled(size) if name in wanted]
    if not turned:
        return faces
    return [
        cv2.rotate(face, turn)
        for face in faces
        for turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
