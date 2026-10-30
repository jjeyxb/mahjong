"""副露區的佔用長度 → 「有人鳴了一組」。

只數組數,不認牌面(第一階段,見 ``docs/cv_mjai.md`` §3.1)。換到的是:

* **副露組數**,放銃分析的「副露威脅」要用(現在畫面那條路是 0,低估威脅)。
* **鳴牌的直接證據** —— ``RiverWatcher`` 現在是靠河變短猜的。

為什麼看「長了多少」而不是「現在多長」
--------------------------------------
一組的寬依種類不同:吃 / 碰 / 加槓約 1 個組長、暗槓 1.2、大明槓 1.3
(四槓子實測 1.13~1.21,理論 4 / 3.33)。從總長反推組數會撞上歧義:四組碰
(4.0)與三組大明槓(3.9)幾乎一樣長。

但鳴牌是**一組一組發生的**,兩次讀取(一秒)之間不會有兩個人各鳴一組到
同一家。所以只看每次長了多少:長了 0.7 組長以上就是一組,四捨五入。種類
差異只影響那一次的增量,不會累積。

加槓看不出來(疊在橫放那張上面,不變長),也不需要:它不增加組數。

兩條和寶牌同一套的規則
----------------------
* **同一局只增不減。** 變短一定是畫面的問題(立繪、結算面板蓋住),不是
  牌桌的事件。直接忽略。
* **連續兩次讀到同樣的長度才採信。** 鳴牌的動畫中間,牌在飛,長度是過渡值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MeldWatcher"]

#: 長了多少個組長才算一組。一組最短是 1.0(吃 / 碰),量測誤差約 ±0.1。
_MIN_GROUP = 0.7

#: 換算組數用的平均組寬。一組在 1.0 ~ 1.3 之間,取中間,兩組(2.0 ~ 2.6)
#: 四捨五入也不會變成一組或三組。
_MEAN_GROUP = 1.15

#: 兩次讀數差多少以內算「同樣的長度」,以組長為單位。
_STABLE = 0.15


@dataclass
class MeldWatcher:
    """盯著每一家的副露長度,回報新長出來的組數。

    Args:
        units: 每一家的組長(:func:`~mia.vision.tiles.melds.occupied_length`
            的單位),依位置。各視角的角點框的比例不同,所以各用各的。
    """

    units: tuple[float, ...]
    groups: list[int] = field(default_factory=list)
    _base: list[float] = field(default_factory=list, repr=False)
    _pending: list[float | None] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """新的一局。"""
        self.groups = [0 for _ in self.units]
        self._base = [0.0 for _ in self.units]
        self._pending = [None for _ in self.units]

    def observe(self, position: int, length: float | None) -> int:
        """收一家的一次讀數,回傳**新**長出來幾組(通常是 0)。

        Args:
            length: 佔用長度。``None`` 表示這一幀這家讀不到(被蓋住)。
        """
        unit = self.units[position]
        if length is None or length < self._base[position] + _MIN_GROUP * unit:
            # 沒長、變短(被蓋住),或還在動畫中間。只增不減:不動基準。
            self._pending[position] = None
            return 0
        pending = self._pending[position]
        self._pending[position] = length
        if pending is None or abs(length - pending) > _STABLE * unit:
            return 0
        fresh = max(1, round((length - self._base[position]) / (_MEAN_GROUP * unit)))
        self._base[position] = length
        self._pending[position] = None
        self.groups[position] += fresh
        return fresh
