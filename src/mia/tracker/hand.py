"""畫面上自己的手牌 → 自己摸了什麼、打了什麼、鳴了什麼。

這是事件流裡**唯一看得到牌面**的一家,而且比牌河可靠得多:別家打的牌要靠河裡
那一格認得出來,自己打的牌只要比前後兩副手牌。

比的是張數,不是位置
--------------------
手牌會理牌,位置沒有意義。``張數 % 3`` 就說明了狀態:餘 1 是等別人,餘 2 是
輪到自己。兩副手牌的差就是動作:

=========  ================  ==================
張數變化   少了 / 多了       動作
=========  ================  ==================
+1         多 1              摸牌
-1         少 1              打牌
-2         少 2              吃、碰
-2         少 3 多 1         大明槓 + 嶺上
-3         少 4 多 1         暗槓 + 嶺上
0          少 1 多 1         河多一張:摸打都沒看到。沒有:加槓 + 嶺上
=========  ================  ==================

一張認錯不能變成一個動作
------------------------
手牌每次畫面停下來才認一次,認錯一張的話,那一張的差會被當成動作。所以:

* 張數對得上但差太多的讀數不收,等下一次;收下的是**推出來的**手牌,不是這次讀到的。
* 同一種牌超過四張的讀數不收。開局前的介面實測被讀成 14 張、裡面有 5 張白。
* 摸到的牌讀錯了,下一次打牌時會多差一張。這時候改回去,不然那張假牌留在手上,
  之後每一手都解釋不了,整局就斷了。
* 13 張換了一張,可能是兩次讀取之間摸打完了,也可能只是認錯。自己的河多一張才算
  (:data:`_STUBBORN` 次讀到一樣的也算:打出去馬上被鳴走的牌進不了河)。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from mia.mjai.tiles import normalize_red

__all__ = ["HandWatcher", "Move", "OwnMove"]

#: 同一種牌最多幾張。
_COPIES = 4

#: 解釋不了的同一副手牌連續讀到幾次,就相信它。
_STUBBORN = 3


class Move(Enum):
    """自己的一個動作。"""

    DRAW = "draw"
    DISCARD = "discard"
    CALL = "call"


class _Swap(Enum):
    """張數一樣、換了一張的時候,有多少證據說這是真的。"""

    NO = "no"
    THROWN = "thrown"
    STUBBORN = "stubborn"


@dataclass(frozen=True, slots=True)
class OwnMove:
    """一個動作。

    Attributes:
        move: 做了什麼。
        tiles: 摸到或打出的那張;鳴牌是手上拿出去的那幾張。
        tsumogiri: 打的是不是剛摸到的那張。
    """

    move: Move
    tiles: tuple[str, ...]
    tsumogiri: bool = False


@dataclass
class HandWatcher:
    """一局的手牌。每一局 :meth:`reset`。

    Attributes:
        tehai: 配牌,13 張。還沒讀到是空的。
        moves: 到目前為止的動作,依序。
        hand: 推出來的手牌,含剛摸的那張。
        rejected: 不收的讀數有幾次。
        stuck: 連續幾副**不同的**手牌解釋不了。同一副重讀幾次都只算一次:那只是畫面
            沒變。一直累積代表手牌已經對不上了,見
            :attr:`~mia.tracker.stream.Stream.doubt`。
    """

    tehai: tuple[str, ...] = ()
    moves: list[OwnMove] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    rejected: int = 0
    stuck: int = 0
    _drawn: str | None = None
    _doubted: tuple[str, ...] = ()
    _doubts: int = 0
    _unexplained: tuple[str, ...] = ()

    def reset(self) -> None:
        self.tehai = ()
        self.moves = []
        self.hand = []
        self.rejected = 0
        self.stuck = 0
        self._drawn = None
        self._doubted = ()
        self._doubts = 0
        self._unexplained = ()

    def observe(
        self, tiles: Sequence[str], drawn: str | None = None, *, thrown: int | None = None
    ) -> list[OwnMove]:
        """收一副手牌(MJAI 記法,含 ``drawn``),回傳新的動作。

        Args:
            tiles: 手上全部的牌。
            drawn: 其中剛摸到、擺在旁邊的那一張。
            thrown: 河裡看到自己打過幾張。``None`` 表示不知道。
        """
        if max(Counter(normalize_red(tile) for tile in tiles).values(), default=0) > _COPIES:
            self.rejected += 1
            return []
        if not self.tehai:
            return self._deal(tiles, drawn)
        count = Counter(tiles)
        have = Counter(self.hand)
        lost = list((have - count).elements())
        gained = list((count - have).elements())
        change = len(tiles) - len(self.hand)
        if self._only_dealt() and change == 0 and len(lost) > 1:
            # 還沒動過手:上一局留在畫面上的手牌,或發牌動畫中途的牌面。
            self.reset()
            return self._deal(tiles, drawn)
        fresh = self._explain(change, lost, gained, drawn, self._swap(tiles, thrown))
        if fresh is None:
            if lost or gained:
                self.rejected += 1
                current = tuple(sorted(tiles))
                self.stuck += current != self._unexplained
                self._unexplained = current
            else:
                self.stuck = 0
            return []
        self._doubts = 0
        self.stuck = 0
        for move in fresh:
            self._apply(move)
        return fresh

    def _deal(self, tiles: Sequence[str], drawn: str | None) -> list[OwnMove]:
        """第一副手牌。莊家一開始就是 14 張。"""
        if len(tiles) == 13:
            self.tehai = tuple(tiles)
            self.hand = list(tiles)
            return []
        if len(tiles) != 14 or drawn is None:
            return []
        rest = list(tiles)
        rest.remove(drawn)
        self.tehai = tuple(rest)
        self.hand = rest
        move = OwnMove(Move.DRAW, (drawn,))
        self._apply(move)
        return [move]

    def _only_dealt(self) -> bool:
        return all(move.move is Move.DRAW for move in self.moves) and len(self.moves) <= 1

    def _swap(self, tiles: Sequence[str], thrown: int | None) -> _Swap:
        """張數一樣、換了一張,憑什麼相信它。"""
        count = sum(move.move is Move.DISCARD for move in self.moves)
        if thrown is not None and thrown > count:
            return _Swap.THROWN
        current = tuple(sorted(tiles))
        self._doubts = self._doubts + 1 if current == self._doubted else 1
        self._doubted = current
        return _Swap.STUBBORN if self._doubts >= _STUBBORN else _Swap.NO

    def _explain(
        self, change: int, lost: list[str], gained: list[str], drawn: str | None, swap: _Swap
    ) -> list[OwnMove] | None:
        """這兩副手牌的差是哪個動作。解釋不了回 ``None``。"""
        waiting = len(self.hand) % 3 == 1
        if change == 1 and waiting:
            if len(gained) == 1 and not lost:
                return [OwnMove(Move.DRAW, (gained[0],))]
            # 舊的有一張這次認錯了:摸到的是擺在旁邊的那張。
            if len(gained) == 2 and len(lost) == 1 and drawn in gained:
                return [OwnMove(Move.DRAW, (drawn,))]
            return None
        if change == -1 and not waiting:
            if len(lost) == 1 and not gained:
                return [self._discard(lost[0])]
            if len(lost) == 2 and len(gained) == 1 and self._repair(lost, gained[0]):
                return [self._discard(lost[0])]
            return None
        if change == -2 and waiting:
            if len(lost) == 2 and not gained:
                return [OwnMove(Move.CALL, tuple(lost))]
            if len(lost) == 3 and len(gained) == 1:
                return [OwnMove(Move.CALL, tuple(lost)), OwnMove(Move.DRAW, (gained[0],))]
            return None
        if change == -3 and not waiting and len(lost) == 4 and len(gained) == 1:
            return [OwnMove(Move.CALL, tuple(lost)), OwnMove(Move.DRAW, (gained[0],))]
        if change == 0 and len(lost) == 1 and len(gained) == 1:
            if swap is _Swap.NO:
                return None
            if waiting:
                # 摸打都發生在兩次讀取之間。
                return [OwnMove(Move.DRAW, (gained[0],)), self._discard(lost[0])]
            if swap is _Swap.THROWN:
                return [self._discard(lost[0]), OwnMove(Move.DRAW, (gained[0],))]
            if any(move.move is Move.CALL for move in self.moves):
                return [OwnMove(Move.CALL, (lost[0],)), OwnMove(Move.DRAW, (gained[0],))]
            # 沒有副露就不可能加槓:是之前認錯了一張。
            self._correct(lost[0], gained[0])
            return []
        return None

    def _correct(self, wrong: str, right: str) -> None:
        """之前收下的一張牌認錯了:連同它的來源一起改掉。"""
        self.hand[self.hand.index(wrong)] = right
        for index, move in enumerate(self.moves):
            if move.move is Move.DRAW and move.tiles[0] == wrong:
                self.moves[index] = OwnMove(Move.DRAW, (right,))
                return
        tehai = list(self.tehai)
        if wrong in tehai:
            tehai[tehai.index(wrong)] = right
            self.tehai = tuple(tehai)

    def _repair(self, lost: list[str], right: str) -> bool:
        """上一次摸到的牌讀錯了:改回去。``lost`` 裡留下真的打掉的那一張。"""
        last = self.moves[-1] if self.moves else None
        if last is None or last.move is not Move.DRAW or last.tiles[0] not in lost:
            return False
        wrong = last.tiles[0]
        self.moves[-1] = OwnMove(Move.DRAW, (right,))
        self.hand[self.hand.index(wrong)] = right
        self._drawn = right
        lost.remove(wrong)
        return True

    def _discard(self, tile: str) -> OwnMove:
        return OwnMove(Move.DISCARD, (tile,), tsumogiri=tile == self._drawn)

    def _apply(self, move: OwnMove) -> None:
        self.moves.append(move)
        if move.move is Move.DRAW:
            self.hand.append(move.tiles[0])
            self._drawn = move.tiles[0]
            return
        for tile in move.tiles:
            self.hand.remove(tile)
        self._drawn = None
