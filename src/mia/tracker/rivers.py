"""牌河的逐幀觀察 → 捨牌事件。

為什麼不是單純的 diff
---------------------
牌河看起來只會在尾端長,但有三件事讓天真的 diff 失效:

1. **鳴牌會讓它變短。** 被鳴走的那張從河裡消失,後面的牌整排往前遞補
   (雀魂不留空位,見 :attr:`~mia.mjai.table.Player.visible_river`)。
   天真的 diff 會把「整排往前移」讀成一連串新捨牌 —— 一次鳴牌製造五六個
   假事件,而且之後每一張都對不上。
2. **讀不到的格子是常態。** 牌河的信心門檻刻意訂得嚴
   (:data:`~mia.vision.tiles.river.CONFIDENT`),約四分之一的牌會棄權。
   棄權**不是**「那一格沒牌」,而是「那張牌我還不知道是什麼」——
   牌還在桌上,下一幀可能就讀到了。所以 :data:`UNKNOWN` 要當萬用字元用,
   而且事件發出去之後還能被後來的觀察補上。
3. **一幀之間可能發生不只一件事。** 掉幀、動畫遮擋,回來的時候河已經多了
   兩張。順序是確定的(河是有序的),所以照序補發就好。

什麼時候**不要**發事件
----------------------
看不懂的觀察一律丟掉,不要猜。實測牌河有 15.8% 的幀被立直/和了的立繪蓋住,
那些幀讀出來的「牌」是角色的臉 —— 連續性檢查擋得下大部分,但擋不下全部。
一個猜錯的捨牌事件會污染整條事件流,而少發一個只是延後(牌還在桌上)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["UNKNOWN", "Discard", "RiverWatcher", "WatchStats"]

#: 「這一格有牌,但還不知道是什麼」。
#:
#: 與「沒有牌」是**完全不同**的兩件事。棄權的格子在張數上算數 ——
#: 它參與位置的對齊,只是內容待定。
UNKNOWN = "?"


@dataclass(frozen=True, slots=True)
class Discard:
    """一次捨牌。

    Attributes:
        position: **依視角**的座位 0~3,與
            :data:`~mia.vision.tiles.river.SEATS` 同序(自己、下家、對家、上家)。
            不是絕對座位 —— 這一層只看得見畫面,換算成絕對座位是上層的事。
        index: 這是他的第幾張捨牌(從 0 起算)。
        pai: 牌(雀魂記法),:data:`UNKNOWN` 表示還不知道。
    """

    position: int
    index: int
    pai: str


@dataclass
class WatchStats:
    """看得懂多少、猜了多少 —— 診斷用。

    Attributes:
        observations: 收到幾次觀察。
        emitted: 發出幾個捨牌事件。
        resolved: 有幾個原本 :data:`UNKNOWN` 的事後被補上。
        calls: 判定為「被鳴走一張」的次數。
        confused: 看不懂而整筆丟掉的次數。**這一欄要盯著** ——
            它變大代表辨識或對齊出了問題,而症狀不會出現在別的地方。
    """

    observations: int = 0
    emitted: int = 0
    resolved: int = 0
    calls: int = 0
    confused: int = 0


def _fits(known: str, seen: str) -> bool:
    """兩個讀數可以是同一張牌嗎。任一邊未知就算相容。"""
    return known in (UNKNOWN, seen) or seen == UNKNOWN


def _is_prefix(known: Sequence[str], seen: Sequence[str]) -> bool:
    """``known`` 是 ``seen`` 的前綴嗎(未知當萬用字元)。"""
    if len(known) > len(seen):
        return False
    return all(_fits(a, b) for a, b in zip(known, seen, strict=False))


def _drop_one(known: Sequence[str], seen: Sequence[str]) -> int | None:
    """``seen`` 是 ``known`` 拿掉**一張**之後的樣子嗎,拿掉的是第幾張。

    鳴牌就長這樣:中間少一張,後面整排往前遞補。回 ``None`` 代表不是。
    """
    if len(seen) + 1 != len(known):
        return None
    for cut in range(len(known)):
        rest = [*known[:cut], *known[cut + 1 :]]
        if all(_fits(a, b) for a, b in zip(rest, seen, strict=True)):
            return cut
    return None


@dataclass
class RiverWatcher:
    """盯著四家的牌河,把變化講成捨牌事件。

    用法::

        watcher = RiverWatcher()
        for frame in frames:
            for position, tiles in enumerate(frame.rivers):
                for discard in watcher.observe(position, tiles):
                    ...

    每一家各自獨立 —— 這一層**不處理四家之間的先後順序**。那需要輪替規則,
    是上層的事(見 ``docs/cv_mjai.md`` 的 tracker 清單)。
    """

    rivers: list[list[str]] = field(default_factory=lambda: [[] for _ in range(4)])
    stats: WatchStats = field(default_factory=WatchStats)

    def reset(self) -> None:
        """新的一局。四家的河都清空。"""
        self.rivers = [[] for _ in range(4)]

    def observe(self, position: int, tiles: Sequence[str]) -> list[Discard]:
        """收一次觀察,回傳**新**的捨牌事件(可能是空的)。

        Args:
            position: 依視角的座位 0~3。
            tiles: 這一家畫面上看得見的牌,依序。讀不到的那幾格放
                :data:`UNKNOWN` —— **不要省略**,省略會讓後面的位置全部錯開。

        Returns:
            這次觀察帶來的新捨牌。看不懂的觀察回空串列並記進
            :attr:`WatchStats.confused`。
        """
        self.stats.observations += 1
        known = self.rivers[position]
        seen = list(tiles)

        if _is_prefix(known, seen):
            self._fill(position, seen[: len(known)])
            fresh = [
                Discard(position=position, index=len(known) + offset, pai=pai)
                for offset, pai in enumerate(seen[len(known) :])
            ]
            self.rivers[position] = [*self.rivers[position], *seen[len(known) :]]
            self.stats.emitted += len(fresh)
            return fresh

        cut = _drop_one(known, seen)
        if cut is not None:
            # 被鳴走一張。**不發事件** —— 那張牌的 dahai 早就發過了,
            # 它現在只是從畫面上移到別人的副露去。
            self.stats.calls += 1
            self.rivers[position] = [*known[:cut], *known[cut + 1 :]]
            self._fill(position, seen)
            return []

        self.stats.confused += 1
        return []

    def _fill(self, position: int, seen: Sequence[str]) -> None:
        """拿新的讀數補上原本未知的格子。

        這就是「棄權是延後不是放棄」的實作:牌還在桌上,下一幀讀到了就補。
        **只補不改** —— 已經確定的格子不會被後來的讀數推翻,因為那通常
        代表對齊出了問題,而不是前一次讀錯。
        """
        river = self.rivers[position]
        for index, pai in enumerate(seen):
            if index < len(river) and river[index] == UNKNOWN and pai != UNKNOWN:
                river[index] = pai
                self.stats.resolved += 1
