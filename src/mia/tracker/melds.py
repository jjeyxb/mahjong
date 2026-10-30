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

from collections.abc import Iterable
from dataclasses import dataclass, field

from mia.mjai.tiles import HONOR_ORDER, UNKNOWN

__all__ = ["MeldWatcher", "guess_meld"]

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
    #: 連續幾次讀到同樣的長度才採信。
    confirm: int = 2
    groups: list[int] = field(default_factory=list)
    _base: list[float] = field(default_factory=list, repr=False)
    _pending: list[list[float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """新的一局。"""
        self.groups = [0 for _ in self.units]
        self._base = [0.0 for _ in self.units]
        self._pending = [[] for _ in self.units]

    def length(self, position: int) -> float:
        """這一家目前採信的長度。新長出來的組就在這個長度之後。"""
        return self._base[position]

    def observe(self, position: int, length: float | None) -> int:
        """收一家的一次讀數,回傳**新**長出來幾組(通常是 0)。

        Args:
            length: 佔用長度。``None`` 表示這一幀這家讀不到(被蓋住)。
        """
        unit = self.units[position]
        if length is None or length < self._base[position] + _MIN_GROUP * unit:
            # 沒長、變短(被蓋住),或還在動畫中間。只增不減:不動基準。
            self._pending[position] = []
            return 0
        pending = self._pending[position]
        if pending and abs(length - pending[0]) > _STABLE * unit:
            pending.clear()
        pending.append(length)
        if len(pending) < self.confirm:
            return 0
        fresh = max(1, round((length - self._base[position]) / (_MEAN_GROUP * unit)))
        self._base[position] = length
        self._pending[position] = []
        self.groups[position] += fresh
        return fresh


def guess_meld(faces: Iterable[str], *, repeated: Iterable[str] = ()) -> tuple[str, ...]:
    """一組副露認出來的牌面(MJAI 記法)→ 記進 ``Player.melds`` 的那一組。

    認牌面只保證「認出來的是對的」,不保證認得全(橫放、遠的那家常常漏)。
    所以這裡只做**一定成立**的推論,其餘位置填未知:

    * 字牌只能碰、槓 —— 認出一張字牌,整組就是它。
    * 數牌只認出一種:在兩個位置都認到(``repeated``)就是碰,否則可能是碰
      也可能是吃,只記那一張。
    * 數牌兩三種:必須同一色、湊得成一個順子。兩張差 2 就知道中間那張;
      差 1 而碰到 1 或 9 也只有一種順子(1 2 → 1 2 3、8 9 → 7 8 9)。

    湊不成一組(混了字牌與數牌、兩種花色、相差太遠、順子裡有一張出現兩次)
    就代表有一張認錯了,或是框到了別組的牌。不知道是哪一張錯,**整組當作
    沒認出來**。

    下游只看第一張(役牌、染手),所以認得出的牌排在前面。

    用未知佔位而不是少記一組:``Player.melded`` 要數得到它(副露威脅),而
    未知不會被當成役牌、不會被當成某個花色。
    """
    tiles = sorted(set(faces))
    twice = set(repeated)
    unread = (UNKNOWN, UNKNOWN, UNKNOWN)
    if not tiles or len(tiles) > 3:
        return unread
    if any(tile in HONOR_ORDER for tile in tiles):
        return (tiles[0],) * 3 if len(tiles) == 1 else unread
    suits = {tile[1] for tile in tiles}
    numbers = [int(tile[0]) for tile in tiles]
    low, high = min(numbers), max(numbers)
    if len(suits) != 1 or high - low > 2:
        return unread
    suit = tiles[0][1]
    if len(tiles) == 1:
        return (tiles[0],) * 3 if tiles[0] in twice else (tiles[0], UNKNOWN, UNKNOWN)
    if twice:
        # 順子不會有兩張一樣的牌。
        return unread
    if high - low == 2 or low == 1 or high == 9:
        first = low if high - low == 2 or low == 1 else high - 2
        return tuple(f"{n}{suit}" for n in range(first, first + 3))
    return (*tiles, UNKNOWN)

