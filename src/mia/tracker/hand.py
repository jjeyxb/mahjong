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
-4         少 4              暗槓,嶺上摸切了:立直之後兩次讀取之間就做完。河會補上摸切那一手
0          少 1 多 1         河多一張:摸打都沒看到。少的是碰過的牌:加槓 + 嶺上
=========  ================  ==================

一張認錯不能變成一個動作
------------------------
手牌每次畫面停下來才認一次,認錯一張的話,那一張的差會被當成動作。所以:

* 張數對得上但差太多的讀數不收,等下一次;收下的是**推出來的**手牌,不是這次讀到的。
* 同一種牌超過四張的讀數不收。開局前的介面實測被讀成 14 張、裡面有 5 張白。
* 認錯的牌留在手上的話,下一次摸打會多差一張,而且之後每一手都解釋不了,整局就斷了
  (實測 6 索認成 3 索,建議卡到局尾)。所以動作對得上、只差一張時就地改回去,不管
  認錯的是剛摸的還是配牌。連續解釋不了時放寬到 :data:`_MAX_FIXES` 張。
* 13 張換了一張,可能是兩次讀取之間摸打完了,也可能只是認錯。自己的河多一張才算。
  打出去馬上被鳴走的牌進不了河,但別家的副露會多一組,:data:`_STUBBORN` 次讀到一樣
  的又有這個證據也算。只看讀到幾次的話,一張認錯就會變成自己多打一手。

加槓不靠連續讀到
----------------
畫面沒變就不會重讀,同一副手牌要連讀好幾次實戰裡幾乎不會發生。分得開的是**不見的那張
是自己碰過的牌**:別的牌換掉是認錯,碰過的牌不見了才是加槓。只少一張、嶺上還沒來時先
等,也可能是真的把第四張打掉了,河裡多一張才算打牌(:meth:`HandWatcher.catch_up`)。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from mia.mjai.tiles import normalize_red

__all__ = ["HandWatcher", "Move", "OwnMove"]

#: 同一種牌最多幾張。
_COPIES = 4

#: 解釋不了的同一副手牌連續讀到幾次,就相信它。
_STUBBORN = 3

#: 手牌一直解釋不了時,一次最多改回幾張。
_MAX_FIXES = 3

#: 兩張牌湊不成順子時的距離,比任何真的距離都大。
_FAR = 99


def _is_kan(tiles: Sequence[str]) -> bool:
    """手上拿出去的三張是不是一樣的。"""
    return len({normalize_red(tile) for tile in tiles}) == 1


def _gap(a: str, b: str) -> int:
    """兩張牌離多遠。不同花色與字牌湊不成順子,給一個很大的數。"""
    a, b = normalize_red(a), normalize_red(b)
    if a == b:
        return 0
    if len(a) != len(b) or len(a) == 1 or a[1] != b[1]:
        return _FAR
    return abs(int(a[0]) - int(b[0]))


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
    CALLED = "called"


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
    _last_read: tuple[tuple[str, ...], str | None] = ((), None)
    _called: int = 0
    _combined_calls: set[int] = field(default_factory=set)

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
        self._last_read = ((), None)
        self._called = 0
        self._combined_calls.clear()

    def observe(
        self,
        tiles: Sequence[str],
        drawn: str | None = None,
        *,
        thrown: int | None = None,
        called: int = 0,
    ) -> list[OwnMove]:
        """收一副手牌(MJAI 記法,含 ``drawn``),回傳新的動作。

        Args:
            tiles: 手上全部的牌。
            drawn: 其中剛摸到、擺在旁邊的那一張。
            thrown: 河裡看到自己打過幾張。``None`` 表示不知道。
            called: 別家總共鳴過幾組。
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
        discards = sum(move.move is Move.DISCARD for move in self.moves)
        quiet = thrown is not None and thrown <= discards
        read = (tuple(sorted(tiles)), drawn)
        again, self._last_read = read == self._last_read, read
        fresh = self._explain(
            change, lost, gained, drawn, self._swap(tiles, thrown, called), quiet=quiet, again=again
        )
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
        if fresh:
            self._called = called
        for move in fresh:
            self._apply(move)
        return fresh

    def catch_up(self, thrown: int, tile: str | None) -> list[OwnMove]:
        """河裡多了一張,手牌卻沒動:摸切快到手牌從來沒讀到 14 張。

        立直之後是自動摸切,一定會這樣。漏掉的話自己的動作少一組,之後每一手都晚一輪。
        呼叫端要等手牌有時間跟上才叫,不然正常打牌也會走到這裡。

        Args:
            thrown: 河裡看到自己打過幾張。
            tile: 河裡最新的那一張。認不出來給 ``None``,這一次就不補。
        """
        count = sum(move.move is Move.DISCARD for move in self.moves)
        if not self.tehai or tile is None or thrown <= count:
            return []
        if len(self.hand) % 3 == 1:
            fresh = [OwnMove(Move.DRAW, (tile,)), OwnMove(Move.DISCARD, (tile,), tsumogiri=True)]
        elif tile in self.hand and self._can_add_kan(tile):
            # 手牌少了碰過的那張,一直在等嶺上:河裡有它,原來是打掉了。
            fresh = [self._discard(tile)]
        else:
            return []
        for move in fresh:
            self._apply(move)
        return fresh

    def reconcile_discards(self, readings: Mapping[int, str]) -> None:
        """用有把握的牌河讀數,核對一次少三張時猜的吃碰與捨牌。

        五七吃六、打八時,只比手牌會先選較近的七八。河裡確認打的是八之後,
        就把這三張重新分配;總共少的牌不變,不需要改目前手牌。
        分開看到的鳴牌與打牌不改,牌河也不能湊出不合法的搭子。
        """
        serial = 0
        for index, move in enumerate(self.moves):
            if move.move is not Move.DISCARD:
                continue
            pai = readings.get(serial)
            serial += 1
            if index - 1 not in self._combined_calls or pai is None or pai == move.tiles[0]:
                continue
            call = self.moves[index - 1]
            tiles = [*call.tiles, *move.tiles]
            if pai not in tiles:
                continue
            tiles.remove(pai)
            if _gap(*tiles) > 2:
                continue
            self.moves[index - 1] = OwnMove(Move.CALL, tuple(sorted(tiles)))
            self.moves[index] = OwnMove(Move.DISCARD, (pai,))

    def adopt(
        self,
        tehai: Sequence[str],
        moves: Sequence[OwnMove],
        tiles: Sequence[str],
        drawn: str | None,
    ) -> None:
        """換成重寫過的配牌與動作。見 :func:`mia.tracker.stream.resynced`。"""
        self.tehai = tuple(tehai)
        self.moves = list(moves)
        self.hand = list(tiles)
        self.stuck = 0
        self._drawn = drawn
        self._doubted = ()
        self._doubts = 0
        self._unexplained = ()
        self._combined_calls.clear()

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

    def _swap(self, tiles: Sequence[str], thrown: int | None, called: int) -> _Swap:
        """張數一樣、換了一張,憑什麼相信它。"""
        count = sum(move.move is Move.DISCARD for move in self.moves)
        if thrown is not None and thrown > count:
            return _Swap.THROWN
        current = tuple(sorted(tiles))
        self._doubts = self._doubts + 1 if current == self._doubted else 1
        self._doubted = current
        if self._doubts < _STUBBORN:
            return _Swap.NO
        return _Swap.CALLED if called > self._called else _Swap.STUBBORN

    def _explain(
        self,
        change: int,
        lost: list[str],
        gained: list[str],
        drawn: str | None,
        swap: _Swap,
        *,
        quiet: bool = False,
        again: bool = True,
    ) -> list[OwnMove] | None:
        """這兩副手牌的差是哪個動作。解釋不了回 ``None``。

        ``quiet`` 是河讀得到、卻沒有多一張。``again`` 是這副手牌上一次也是這樣讀的。
        """
        waiting = len(self.hand) % 3 == 1
        if change == 1 and waiting:
            if len(gained) == 1 and not lost:
                return [OwnMove(Move.DRAW, (gained[0],))]
            # 舊的有一張這次認對了:摸到的是擺在旁邊的那張,其他的是改回來。
            if drawn in gained and len(gained) == len(lost) + 1 and self._may_fix(lost):
                rest = list(gained)
                rest.remove(drawn)
                self._fix(list(lost), rest)
                return [OwnMove(Move.DRAW, (drawn,))]
            return None
        if change == -1 and not waiting:
            if len(lost) == 1 and not gained:
                if self._can_add_kan(lost[0]) and swap is not _Swap.THROWN:
                    return []
                return [self._discard(lost[0])]
            if len(lost) == len(gained) + 1 and self._may_fix(gained):
                left = list(lost)
                self._fix(left, gained)
                return [self._discard(left[0])]
            return None
        if change == -2 and waiting:
            if len(lost) == 2 and not gained:
                # 要連續讀到兩次。打完牌手牌滑回去的那一幀,一對牌還沒到位沒讀到,看起來就是碰了那一對
                # (selfplay-20261110-035646 南2局1本場:一幀少兩張發,收成碰發,牌桌再把別家的捨牌
                # 改成發,事件流停到局尾)。真的鳴牌之後同一副手牌 0.3 ~ 0.4 秒內一定會再讀到一次
                # (同一場六次吃碰都是),代價就是晚這麼久。
                if not again:
                    return None
                return [OwnMove(Move.CALL, tuple(lost))]
            if len(lost) == 3 and len(gained) == 1:
                draw = OwnMove(Move.DRAW, (gained[0],))
                if _is_kan(lost):
                    return [OwnMove(Move.CALL, tuple(lost)), draw]
                # 鳴完打一張,下一巡又摸了一張,都在兩次讀取之間。
                return [*self._call_and_discard(lost), draw]
            return None
        if change == -3 and waiting and len(lost) == 3 and not gained:
            # 三張一樣是大明槓,嶺上摸切了。理由同下。
            if _is_kan(lost):
                return [OwnMove(Move.CALL, tuple(lost))]
            # 碰牌動畫中途,手上剩下的牌會滑過摸牌的位置,那一瞬間沒讀到就少三張。河裡沒多一張的話
            # 先不收:實測碰發之後讀成「碰、打中」,中滑回來又變成摸中,之後每一手都晚一輪,14 手建議
            # 跟著錯。真的鳴完就打的話,河會跟上,或下一次摸牌時從少三張多一張那條補回來。
            if quiet:
                return None
            return self._call_and_discard(lost)
        if change == -4 and not waiting and len(lost) == 4 and not gained and _is_kan(lost):
            # 立直之後嶺上是自動摸切,一秒不到,手牌從 14 張直接變 10 張。沒有這一條的話,這之後每一
            # 副手牌都解釋不了,實測事件流停在暗槓前,那一局最後十幾手沒有建議。
            return [OwnMove(Move.CALL, tuple(lost))]
        if change == -3 and not waiting and len(lost) == 4 and len(gained) == 1:
            return [OwnMove(Move.CALL, tuple(lost)), OwnMove(Move.DRAW, (gained[0],))]
        if change == 0 and len(lost) == 1 and len(gained) == 1:
            if waiting and swap in (_Swap.THROWN, _Swap.CALLED):
                # 摸打都發生在兩次讀取之間。
                return [OwnMove(Move.DRAW, (gained[0],)), self._discard(lost[0])]
            if swap is _Swap.THROWN:
                return [self._discard(lost[0]), OwnMove(Move.DRAW, (gained[0],))]
            if not waiting and self._can_add_kan(lost[0]):
                return [OwnMove(Move.CALL, (lost[0],)), OwnMove(Move.DRAW, (gained[0],))]
            if swap is _Swap.NO:
                return None
            # 河沒有多,碰過的牌也沒少:是之前認錯了一張。
            self._correct(lost[0], gained[0])
            return []
        if change == 0 and len(lost) == len(gained) > 1 and self._may_fix(gained):
            self._fix(list(lost), gained)
            return []
        return None

    def _can_add_kan(self, tile: str) -> bool:
        """這張是自己碰過、還沒加槓的牌嗎。"""
        plain = normalize_red(tile)
        calls = [move.tiles for move in self.moves if move.move is Move.CALL]
        pon = any(len(t) == 2 and _is_kan(t) and normalize_red(t[0]) == plain for t in calls)
        added = any(len(t) == 1 and normalize_red(t[0]) == plain for t in calls)
        return pon and not added

    def _may_fix(self, rights: list[str]) -> bool:
        """差這幾張能不能當成認錯。一張直接改;更多張要連續兩副手牌都解釋不了才改。

        同一副重讀不算:那只是畫面沒變。
        """
        return len(rights) == 1 or (self.stuck >= 2 and len(rights) <= _MAX_FIXES)

    def _fix(self, wrongs: list[str], rights: list[str]) -> None:
        """把認錯的牌改回去,改掉的從 ``wrongs`` 拿掉,剩下的才是真的動作。

        先懷疑剛摸的那張,再找最像的:認錯多半是同花色的牌。
        """
        for right in rights:
            wrong = min(wrongs, key=lambda tile: (tile != self._drawn, _gap(tile, right)))
            wrongs.remove(wrong)
            self._correct(wrong, right)

    def _call_and_discard(self, lost: list[str]) -> list[OwnMove]:
        """手上少了三張,不是大明槓:鳴牌拿出去兩張,再打掉一張。

        碰是兩張一樣的;吃是兩張最接近的。打掉的是剩下那一張。
        """
        meld = min(
            ((i, j) for i in range(len(lost)) for j in range(i + 1, len(lost))),
            key=lambda pair: _gap(lost[pair[0]], lost[pair[1]]),
        )
        tiles = tuple(lost[index] for index in meld)
        rest = next(tile for index, tile in enumerate(lost) if index not in meld)
        self._combined_calls.add(len(self.moves))
        return [OwnMove(Move.CALL, tiles), OwnMove(Move.DISCARD, (rest,))]

    def _correct(self, wrong: str, right: str) -> None:
        """之前收下的一張牌認錯了:連同它的來源一起改掉。"""
        self.hand[self.hand.index(wrong)] = right
        if self._drawn == wrong:
            self._drawn = right
        for index in range(len(self.moves) - 1, -1, -1):
            move = self.moves[index]
            if move.move is Move.DRAW and move.tiles[0] == wrong:
                self.moves[index] = OwnMove(Move.DRAW, (right,))
                return
        tehai = list(self.tehai)
        if wrong in tehai:
            tehai[tehai.index(wrong)] = right
            self.tehai = tuple(tehai)

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
