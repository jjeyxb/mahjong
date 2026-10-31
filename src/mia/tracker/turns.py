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
正常摸打立刻發。其他情況都要等副露確認,最多 :data:`WAIT` 輪:

* 下家、``unseen`` 不是 1:吃與摸打的差別只有那一筆紀錄。
* 跳家:一定是鳴牌,而鳴牌一定有副露。

等不到副露、余牌也沒減的牌**不是真的捨牌**。和了之後的結算畫面會被讀成一排
``?``,實測一局最多多出 3 張;它們既沒有摸牌也沒有鳴牌,在這裡就留不下來。

**輪到自己就不用等了。** 自己摸到牌,代表前面的事都發生過了,佇列裡的牌可以全部
排出來 —— 這一刻正是 Mortal 要回答的時候,而等四輪的話它看到的局面少一兩張。

河會晚到
--------
某一家的河這一輪讀不到(立繪、動畫),下一輪才出現,下家的牌就先到了。看起來像跳家,
但那一家的牌就排在佇列後面。所以跳家沒有副露時先往後找:找得到就照輪替換回來,
不是真的跳過。不這樣做的話會補一張假的隱藏捨牌,真的牌到了又多一張。
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
        guess: 沒有把握時的最佳猜測。Mortal 不收 ``?``,見 :mod:`mia.tracker.stream`。
        target: :attr:`Action.CALL` 鳴的是誰的牌;``None`` 是暗槓。
        serial: 這是打的人的第幾張捨牌。``None`` 是河裡從沒出現過的牌。
    """

    action: Action
    actor: int
    pai: str | None = None
    guess: str | None = None
    target: int | None = None
    serial: int | None = None


@dataclass(slots=True)
class _Pending:
    tick: int
    actor: int
    pai: str | None
    guess: str | None
    serial: int | None
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
    _wall: int | None = field(default=None, repr=False)
    _wall_tick: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        self.last = (self.oya - 1) % _SEATS

    def tick(
        self,
        tick: int,
        discards: Sequence[tuple[int, str | None, *tuple[str | int | None, ...]]] = (),
        melds: Sequence[int] = (),
        wall: int | None = None,
        *,
        hurry: bool = False,
    ) -> list[Step]:
        """收一輪的觀察,回傳這一輪可以確定的動作。

        Args:
            tick: 第幾輪。
            discards: 這一輪新出現的捨牌 ``(座位, 牌)`` 或
                ``(座位, 牌, 猜測, 第幾張)``。牌讀不出來給 ``None``。
            melds: 這一輪副露多了一組的座位,一組一筆。
            wall: 這一輪讀到的余牌,讀不到給 ``None``。
            hurry: 輪到自己了,不必再等。見模組說明。
        """
        self._melds += [(tick, seat) for seat in melds]
        # 同一輪的並列照輪替排:誰離上一個打的人最近,誰先。
        anchor = self._queue[-1].actor if self._queue else self.last
        for seat, pai, *rest in sorted(discards, key=lambda d: self._distance(anchor, d[0])):
            guess, serial = rest if rest else (None, None)
            self._queue.append(_Pending(tick, seat, pai, guess, serial, self._wall))  # type: ignore[arg-type]
            anchor = seat
        if wall is not None:
            self._wall = wall
            self._wall_tick = tick
        out: list[Step] = []
        while self._queue:
            self._restore_turn()
            steps = self._resolve(self._queue[0], tick + WAIT if hurry else tick)
            if steps is None:
                break
            item = self._queue.pop(0)
            if steps:
                out += steps
                self.last = item.actor
        return out

    def _restore_turn(self) -> None:
        """排頭跳家又沒有副露,而該輪到的那一家就在後面:河晚到了,換回來。"""
        head = self._queue[0]
        turn = (self.last + 1) % _SEATS
        if head.actor == turn or self._has_meld(head.actor, head.tick):
            return
        for index, item in enumerate(self._queue):
            if item.actor == turn:
                # 它一定摸過牌:下家都打了。它讀到的余牌已經是之後的,不能拿來算。
                item.wall = _FULL_WALL - self.drawn - 1
                self._queue.insert(0, self._queue.pop(index))
                return

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
        drop = Step(Action.DAHAI, seat, item.pai, item.guess, serial=item.serial)
        following = seat == (self.last + 1) % _SEATS
        if following and unseen == 1:
            self.drawn += 1
            return [Step(Action.TSUMO, seat), drop]
        if not self._has_meld(seat, item.tick) and now - item.tick < WAIT:
            return None
        if self._take_meld(seat, item.tick):
            if not following:
                # 第一張就被鳴走:桌上還沒有牌,一定是莊家打的,而開局沒有上一輪
                # 的余牌可以比。
                hidden = self._hidden(seat, max(unseen, 1) if self.drawn == 0 else unseen)
                target = hidden[-1].actor if hidden else self.last
                return [*hidden, Step(Action.CALL, seat, target=target), drop]
            if unseen <= 0:
                return [Step(Action.CALL, seat, target=self.last), drop]
            self.drawn += 2
            draw = Step(Action.TSUMO, seat)
            return [draw, Step(Action.CALL, seat), draw, drop]
        # 等完了還是沒有副露。余牌有減就是動作太快、摸牌沒趕上那一輪;沒減就不是真的牌。
        behind = _FULL_WALL - self.drawn - (known if self._wall is None else self._wall)
        if behind < 1:
            if self._wall_tick > item.tick:
                return []
            # 這張牌之後余牌都沒讀到:不知道有沒有摸,不能說它是假的。實測網頁版有一段
            # 余牌整段讀不到,局的後半全部被這樣丟掉。
            behind = 1
        if not following:
            # 跳家卻沒有副露,而余牌有減:中間漏看了。補上再接回去,不然 last 停在
            # 原地,之後每一張都變成「跳家且沒有副露」,整局後面一路被丟。
            lost = (seat - self.last - 1) % _SEATS
            steps = self._hidden(seat, min(lost, max(behind - 1, 0)))
            self.drawn += 1
            return [*steps, Step(Action.TSUMO, seat), drop]
        self.drawn += max(1, unseen)
        return [Step(Action.TSUMO, seat), drop]

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
