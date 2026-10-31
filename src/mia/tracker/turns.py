"""四家的捨牌 → 依序的一串動作:誰摸、誰鳴、誰打。

:class:`~mia.tracker.rivers.RiverWatcher` 是逐家獨立的,而 Mortal 吃的是**有序**的
串流。這一層把順序排回來。見 ``docs/cv_mjai.md`` §4.3。

由捨牌驅動
----------
每張捨牌本身就帶著座位 —— 出現在誰的河裡就是誰打的。所以到達順序就是打牌順序,
同一輪的並列才用輪替決定。第一版反過來,用「輪到誰 + 等副露確認」的狀態機,
失敗了:副露比捨牌晚 2 ~ 4 秒,到的時候順序早就過去了。

兩張捨牌之間發生什麼,由座位與余牌推
------------------------------------
余牌只會在有人摸牌時減少,所以 ``70 - 余牌`` 就是摸過幾張。與「已經發出去的摸牌」
比,差額 ``unseen`` 就是畫面上看得到、事件流還沒交代的摸牌:

====================  ==========  ================================
``a`` 之後 ``s`` 打   unseen      中間發生的事
====================  ==========  ================================
``s`` 是下家          1 以上      正常摸打
``s`` 是下家          0,有副露   ``s`` 吃
``s`` 是下家          2,有副露   暗槓:摸、槓、嶺上
``s`` 不是下家        0           ``s`` 鳴了 ``a`` 的牌
``s`` 不是下家        1 以上      中間有人摸打,**那張一打出就被鳴走**
====================  ==========  ================================

最後一列是每秒讀一次的盲點:牌在河裡待不到一秒,從來沒被讀到過。

**余牌要看上一輪的。** 牌出現在第 ``t - 1`` 與 ``t`` 輪之間,而它下家的摸牌至少
再晚半秒,常常落在同一輪 —— 看第 ``t`` 輪就會多算一張。

沒交代的摸牌會累積(加槓看不見,但嶺上會讓余牌多減一張),所以正常摸打時
把差額吸收掉,不讓一次的誤差變成之後每次都多補一張。

什麼時候要等
------------
絕大多數的牌立刻發。只有 ``unseen`` 不是 1 的下家要等副露確認(最多
:data:`WAIT` 輪),因為吃與摸打的差別只有那一筆紀錄。等不到就當成摸打。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Action", "Step", "TurnTracker"]

#: 開局的余牌。第一張是莊家摸的,畫面顯示 69。
_FULL_WALL = 70

_SEATS = 4

#: 等副露最多幾輪。副露確認比鳴牌晚 2 ~ 4 秒,而捨牌本身也比鳴牌晚一兩秒。
WAIT = 4

#: 副露紀錄最多比它對應的捨牌早幾輪。太舊的是之前的鳴牌,不是這一張的。
_MELD_BEFORE = 4


class Action(Enum):
    """事件流裡的一步。"""

    TSUMO = "tsumo"
    CALL = "call"
    DAHAI = "dahai"


@dataclass(frozen=True, slots=True)
class Step:
    """一步。

    Attributes:
        action: 做了什麼。
        actor: 誰。**依視角**,0 是自己。
        pai: 打出的牌。:attr:`Action.DAHAI` 才有;``None`` 表示這張牌從來沒被
            看到(一打出就被鳴走),或讀不出來。
    """

    action: Action
    actor: int
    pai: str | None = None


@dataclass(slots=True)
class _Pending:
    tick: int
    actor: int
    pai: str | None
    wall: int | None


@dataclass
class TurnTracker:
    """一局的順序。每一局造一個新的。

    Args:
        oya: 莊家,依視角。
    """

    oya: int
    drawn: int = 0
    last: int = field(init=False)
    _queue: list[_Pending] = field(default_factory=list, repr=False)
    _melds: list[tuple[int, int]] = field(default_factory=list, repr=False)
    #: 這幾家的鳴牌已經從「跳過」推出來了,之後才到的副露紀錄要銷帳。
    _owed: list[int] = field(default_factory=lambda: [0] * _SEATS, repr=False)
    _wall: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.last = (self.oya - 1) % _SEATS

    def tick(
        self,
        tick: int,
        discards: Sequence[tuple[int, str | None]] = (),
        melds: Sequence[int] = (),
        wall: int | None = None,
    ) -> list[Step]:
        """收一輪的觀察,回傳這一輪可以確定的動作。

        Args:
            tick: 第幾輪。
            discards: 這一輪新出現的捨牌 ``(座位, 牌)``。牌讀不出來給 ``None``。
            melds: 這一輪副露多了一組的座位,一組一筆。
            wall: 這一輪讀到的余牌,讀不到給 ``None``。
        """
        for seat in melds:
            if self._owed[seat]:
                self._owed[seat] -= 1
            else:
                self._melds.append((tick, seat))
        # 同一輪的並列照輪替排:誰離上一個打的人最近,誰先。
        anchor = self._queue[-1].actor if self._queue else self.last
        for seat, pai in sorted(discards, key=lambda d: self._distance(anchor, d[0])):
            self._queue.append(_Pending(tick, seat, pai, self._wall))
            anchor = seat
        if wall is not None:
            self._wall = wall
        out: list[Step] = []
        while self._queue:
            steps = self._resolve(self._queue[0], tick)
            if steps is None:
                break
            out += steps
            self.last = self._queue.pop(0).actor
        return out

    @staticmethod
    def _distance(anchor: int, seat: int) -> int:
        return (seat - anchor - 1) % _SEATS

    def _take_meld(self, seat: int, tick: int) -> bool:
        for entry in self._melds:
            if entry[1] == seat and entry[0] >= tick - _MELD_BEFORE:
                self._melds.remove(entry)
                return True
        return False

    def _has_meld(self, seat: int, tick: int) -> bool:
        return any(s == seat and t >= tick - _MELD_BEFORE for t, s in self._melds)

    def _resolve(self, item: _Pending, now: int) -> list[Step] | None:
        """這張牌之前發生了什麼。還不能確定就回 ``None``。"""
        seat = item.actor
        known = _FULL_WALL - self.drawn if item.wall is None else item.wall
        unseen = _FULL_WALL - self.drawn - known
        drop = Step(Action.DAHAI, seat, item.pai)
        if seat != (self.last + 1) % _SEATS:
            return [*self._hidden(seat, unseen), Step(Action.CALL, seat), drop] if self._settle(
                seat, item
            ) else None
        if unseen == 1:
            self.drawn += 1
            return [Step(Action.TSUMO, seat), drop]
        if not self._has_meld(seat, item.tick) and now - item.tick < WAIT:
            return None
        if self._take_meld(seat, item.tick):
            if unseen <= 0:
                return [Step(Action.CALL, seat), drop]
            self.drawn += 2
            draw = Step(Action.TSUMO, seat)
            return [draw, Step(Action.CALL, seat), draw, drop]
        self.drawn += max(1, unseen)
        return [Step(Action.TSUMO, seat), drop]

    def _settle(self, seat: int, item: _Pending) -> bool:
        """跳家:鳴牌已經確定了,副露紀錄來了就銷帳,還沒來就記著。"""
        if not self._take_meld(seat, item.tick):
            self._owed[seat] += 1
        return True

    def _hidden(self, seat: int, unseen: int) -> list[Step]:
        """跳過的那幾家裡,余牌有減的就是看不見的摸打。"""
        out: list[Step] = []
        actor = (self.last + 1) % _SEATS
        while unseen > 0 and actor != seat:
            out += [Step(Action.TSUMO, actor), Step(Action.DAHAI, actor)]
            self.drawn += 1
            unseen -= 1
            actor = (actor + 1) % _SEATS
        return out
