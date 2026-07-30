"""向聽數、進張、與打牌建議。

輸入是 :func:`~mia.vision.tiles.classify.classify_hand` 認出來的
牌名(雀魂記法),輸出是「還差幾張聽牌」「摸什麼會變好」「該切哪一張」。
這是功能 1 的最後一段,**不需要封包、不碰遊戲連線**。

向聽計算不自己寫
----------------
用 ``mahjong`` 套件。向聽演算法要同時處理一般型、七對子、國士,還有各種
面子/搭子拆法的組合爆炸 —— 自己重寫只會多一個難以察覺的錯誤來源,
而且它不是本專題的貢獻點。

副露怎麼表達
------------
**不需要表達。** ``mahjong`` 套件用「張數變少」隱含副露:它只接受
1/2/4/5/7/8/10/11/13/14 張,而那正好就是暗手牌的合法張數,少掉的每 3 張
就是一組已完成的面子。套件的 docstring 也明講 "remaining melds are implied open"。

這與 :mod:`~mia.vision.tiles.hand` 那邊的推論是同一件事:
向聽數只需要副露的**組數**,不需要知道副露了什麼。

七對子與國士也不必特別處理 —— 套件只在 13 張以上才評估這兩種牌型,
而有副露時張數必定 ≤ 11,結構上就不可能誤判。

進張枚數會略微高估
------------------
算剩餘張數時只扣掉**自己手上**看得到的。牌河、副露、寶牌指示牌都不在功能 1
的辨識範圍內,所以一張已經被別人打掉三張的牌,這裡仍然當作剩 4 張。

這是刻意的取捨:功能 1 的定位是「不碰連線的輕量模式」,要精確的剩餘枚數就
得走封包那條路(功能 2)。UI 上應該把枚數標示為估計值。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mahjong.shanten import Shanten

__all__ = [
    "AGARI",
    "TENPAI",
    "DiscardOption",
    "HandAnalysis",
    "HandError",
    "Ukeire",
    "analyse",
    "suggest_discards",
]

#: 和了。``mahjong`` 套件用 -1 表示。
AGARI = -1
#: 聽牌。
TENPAI = 0

#: 一種牌最多 4 張。
COPIES = 4

#: 34 格陣列的排列:1m~9m, 1p~9p, 1s~9s, 東南西北白發中。
_SUIT_BASE = {"m": 0, "p": 9, "s": 18}
_HONOR_BASE = 27

#: 暗手牌的合法張數。逢 3 的倍數(3/6/9/12)不合法 —— 那代表切到一半、
#: 或是辨識抓在動畫中間。分成「等待中」與「剛摸完」兩組,因為只有後者
#: 該去算打牌建議。
WAITING_COUNTS = frozenset({1, 4, 7, 10, 13})
DRAWN_COUNTS = frozenset({2, 5, 8, 11, 14})

_shanten = Shanten()


class HandError(ValueError):
    """手牌不合法 —— 張數不對,或有認不得的牌。"""


def _index(tile: str) -> int:
    """雀魂記法 → 34 格陣列的索引。赤五併入普通五。"""
    if len(tile) != 2:
        raise HandError(f"認不得的牌: {tile!r}")
    rank, suit = tile[0], tile[1]
    if not rank.isdigit():
        raise HandError(f"認不得的牌: {tile!r}")
    number = int(rank)
    if suit == "z":
        if not 1 <= number <= 7:
            raise HandError(f"字牌只有 1z~7z: {tile!r}")
        return _HONOR_BASE + number - 1
    if suit not in _SUIT_BASE:
        raise HandError(f"認不得的花色: {tile!r}")
    # 0 是赤五,在向聽計算上與普通五完全等價
    return _SUIT_BASE[suit] + (4 if number == 0 else number - 1)


def _label(index: int) -> str:
    """34 格索引 → 雀魂記法(一律回普通五,不回赤五)。"""
    if index >= _HONOR_BASE:
        return f"{index - _HONOR_BASE + 1}z"
    suit = "mps"[index // 9]
    return f"{index % 9 + 1}{suit}"


def _counts(tiles: Sequence[str]) -> list[int]:
    array = [0] * 34
    for tile in tiles:
        array[_index(tile)] += 1
    for index, count in enumerate(array):
        if count > COPIES:
            raise HandError(f"{_label(index)} 出現 {count} 張,一種牌最多 {COPIES} 張")
    return array


@dataclass(frozen=True, slots=True)
class Ukeire:
    """一種能讓向聽數前進的牌。

    Attributes:
        tile: 牌名(雀魂記法)。
        count: 估計還剩幾張 —— **只扣掉自己手上的**,見模組說明。
    """

    tile: str
    count: int

    def __str__(self) -> str:
        return f"{self.tile}×{self.count}"


@dataclass(frozen=True, slots=True)
class HandAnalysis:
    """一手牌的分析結果。

    Attributes:
        shanten: 向聽數。``-1`` 和了、``0`` 聽牌、正數為還差幾步。
        ukeire: 能推進向聽的牌,依剩餘枚數由多到少。**剛摸完的手牌一律為空**
            —— 見 :attr:`needs_discard`。
        concealed: 暗手牌張數。
        melds: 副露組數,由張數反推。
    """

    shanten: int
    ukeire: tuple[Ukeire, ...]
    concealed: int
    melds: int

    @property
    def total_ukeire(self) -> int:
        """進張總枚數。比較不同打法時看這個,不是看有幾種。"""
        return sum(u.count for u in self.ukeire)

    @property
    def is_agari(self) -> bool:
        return self.shanten == AGARI

    @property
    def is_tenpai(self) -> bool:
        return self.shanten == TENPAI

    @property
    def needs_discard(self) -> bool:
        """手上多一張,該切牌了。這種狀態下 :attr:`ukeire` 是空的。

        「再摸一張會怎樣」對 14 張的手牌沒有意義 —— 麻將不會連摸兩張。
        要問的是「切哪一張」,那是 :func:`suggest_discards`。
        """
        return self.concealed in DRAWN_COUNTS

    def __str__(self) -> str:
        if self.is_agari:
            return "和了"
        state = "聽牌" if self.is_tenpai else f"{self.shanten} 向聽"
        if self.needs_discard:
            # 進張是空的,印「進張 0 枚」會被誤讀成「這手沒救了」
            return f"{state},該切牌"
        tiles = " ".join(str(u) for u in self.ukeire[:8])
        return f"{state} 進張 {self.total_ukeire} 枚({tiles})"


@dataclass(frozen=True, slots=True)
class DiscardOption:
    """切掉某一張之後的結果。

    Attributes:
        tile: 要切的牌。
        shanten: 切完之後的向聽數。
        ukeire: 切完之後的進張。
    """

    tile: str
    shanten: int
    ukeire: tuple[Ukeire, ...]

    @property
    def total_ukeire(self) -> int:
        return sum(u.count for u in self.ukeire)

    def __str__(self) -> str:
        state = "聽牌" if self.shanten == TENPAI else f"{self.shanten} 向聽"
        return f"切 {self.tile} → {state},進張 {self.total_ukeire} 枚"


def _shanten_of(array: Sequence[int]) -> int:
    return int(_shanten.calculate_shanten(array))


def _ukeire_of(array: list[int], current: int) -> tuple[Ukeire, ...]:
    """哪些牌摸到之後向聽數會前進。

    做法是逐一試摸:對 34 種牌各加一張、重算向聽,有變好就算進張。
    這比自己分析搭子形狀可靠得多 —— 邊張、坎張、多面聽、七對子與一般型
    的交叉情況全都自動涵蓋。
    """
    found = []
    for index in range(34):
        if array[index] >= COPIES:
            continue  # 四張都在自己手上,摸不到了
        array[index] += 1
        improved = _shanten_of(array) < current
        array[index] -= 1
        if improved:
            found.append(Ukeire(_label(index), COPIES - array[index]))
    found.sort(key=lambda u: (-u.count, u.tile))
    return tuple(found)


def analyse(tiles: Sequence[str]) -> HandAnalysis:
    """算一手牌的向聽數與進張。

    Args:
        tiles: 暗手牌(雀魂記法)。含剛摸進來那張也可以 —— 副露組數由張數
            反推,不需要另外傳。

    Returns:
        向聽數與進張。

    Raises:
        HandError: 張數不合法,或有認不得 / 超過 4 張的牌。

    Note:
        張數是 14/11/8/5/2(剛摸完)時 :attr:`HandAnalysis.ukeire` **一律為空**。
        麻將不會連摸兩張,「再摸一張會怎樣」沒有意義;而且真的去算的話會湊出
        15 張,``mahjong`` 套件會直接拋錯。這種狀態要問的是「切哪一張」,
        用 :func:`suggest_discards`。
    """
    count = len(tiles)
    if count not in WAITING_COUNTS and count not in DRAWN_COUNTS:
        raise HandError(
            f"{count} 張不是合法的暗手牌張數。合法的是 "
            f"{sorted(WAITING_COUNTS | DRAWN_COUNTS)} —— 逢 3 的倍數通常代表"
            "辨識抓在理牌或摸打的動畫中間,該丟掉這一幀重讀。"
        )

    array = _counts(tiles)
    shanten = _shanten_of(array)
    drawn = count in DRAWN_COUNTS
    # 已經和了就沒有「還要摸什麼」;剛摸完則是連摸兩張,牌數會爆掉。
    ukeire = () if drawn or shanten == AGARI else _ukeire_of(array, shanten)
    melds = ((14 if drawn else 13) - count) // 3
    return HandAnalysis(shanten, ukeire, count, melds)


def suggest_discards(tiles: Sequence[str]) -> tuple[DiscardOption, ...]:
    """摸完之後該切哪一張。

    對每一種**不同的**牌試切一次,回傳依「向聽數由小到大、同向聽則進張由多
    到少」排序的結果 —— 第一個就是建議。

    Args:
        tiles: 剛摸完的暗手牌,張數須為 14/11/8/5/2。

    Returns:
        每種可切的牌各一項。手上有兩張一樣的牌時只會出現一次 ——
        切哪一張都一樣。

    Raises:
        HandError: 張數不是「剛摸完」的張數。

    Note:
        排序只看向聽與進張,**不看打點、安全度、場況**。那些是功能 2
        (Mortal)的工作;這裡只回答「純粹以聽牌速度而言該切什麼」。
    """
    count = len(tiles)
    if count not in DRAWN_COUNTS:
        raise HandError(
            f"{count} 張不能算打牌建議 —— 要先摸牌。合法的是 {sorted(DRAWN_COUNTS)}。"
        )

    array = _counts(tiles)
    options = []
    for index in range(34):
        if array[index] == 0:
            continue
        array[index] -= 1
        shanten = _shanten_of(array)
        # 切完只剩 13/10/7/4/1 張,那些張數構不成和了型(和了要 14/11/8/5/2),
        # 所以這裡不需要像 analyse() 那樣先擋掉 AGARI 再算進張。
        ukeire = _ukeire_of(array, shanten)
        array[index] += 1
        options.append(DiscardOption(_label(index), shanten, ukeire))

    options.sort(key=lambda o: (o.shanten, -o.total_ukeire, o.tile))
    return tuple(options)
