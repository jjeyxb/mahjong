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
``s`` 是下家          2,沒有副露  牌讀得晚,再下一家已經摸了:等一下副露,沒有就是摸打
``s`` 不是下家        0           ``s`` 鳴了 ``a`` 的牌,沒有副露紀錄也一樣
``s`` 不是下家        1 以上      中間有人摸打,**那張一打出就被鳴走**
``s`` 槓了別人的牌    0 以上      大明槓:鳴、嶺上、打
====================  ==========  ================================

槓靠副露長度分:碰與吃是 1.0 個組長,槓至少 1.13。``unseen`` 分不開「大明槓的嶺上」
與「被秒鳴的那一家的摸牌」,但長度分得開。

最後一列是每秒讀一次的盲點:牌在河裡待不到一秒,從來沒被讀到過。

中央的燈
--------
上面全是用余牌推的,而余牌只要多算一張,之後正常的摸打看起來就像沒人摸牌。電腦打得快,下一次摸牌
馬上把差額補回來;真人鳴牌前會猶豫好幾秒,這段時間余牌不動,那張牌等不到副露就被當成假的丟掉,
連同後面的鳴牌整串排錯(真人一場東二局,事件流停了 14 秒)。

中央面板的燈跟著「現在輪到誰」(:func:`~mia.vision.hud.read_turn`):打完牌燈留在打牌的人那邊,
下一家摸牌或有人鳴牌才移走。它換到誰,就是誰的回合,不必拿余牌去推:

* 從 ``last`` 直接換到下家,而換到下家之後讀到的余牌比換到 ``last`` 之後的少:摸打。下家吃也是燈直接
  換過來,只有余牌分得開;兩個讀數都在換手之後,所以算出來的張數多算少算都不影響。之後讀到的余牌說
  這次摸牌已經算過了,就是之前多算了一張,在這裡吸收。
* 從 ``last`` 直接換到別家:鳴了 ``last`` 的牌。不必等副露確認。

燈換手比封包晚(摸牌中位 0.3 ~ 0.5 秒、p90 到 1.2 秒;鳴牌 0.15 秒),電腦打得快時牌常常比燈先到。
那時照上面的余牌推,行為與沒有燈時一樣。中間夾著別家,或可能是槓的,也照余牌推。

**余牌要看上一輪的。** 牌出現在第 ``t - 1`` 與 ``t`` 輪之間,而它下家的摸牌至少
再晚半秒,常常落在同一輪 —— 看第 ``t`` 輪就會多算一張。

沒交代的摸牌會累積(加槓不變長,只有嶺上讓余牌多減一張),所以正常摸打時把差額吸收
掉,不讓一次的誤差變成之後每次都多補一張。加槓本身由 :mod:`mia.tracker.stream` 從寶牌
補回來:真人打得快時,那一張差額要晚兩三手才反映在余牌上,這裡分不出是誰。

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

#: 等副露最多幾輪,一輪一秒。副露確認比鳴牌晚 2 ~ 4 秒,而捨牌本身也比鳴牌晚一兩秒。
WAIT = 4

#: 燈換到這一家的紀錄,比上一手牌早這麼多輪就是上一輪的,不算。一輪的四手,真人遠多於兩秒,電腦
#: 也要三秒以上。
_LIGHT_STALE = 2.0

#: 下家打牌而余牌多減一張時等幾輪。多半只是牌讀得晚,再下一家已經摸了,不是暗槓。暗槓一定翻寶牌,
#: 而且少見,不值得為它讓整桌排隊:等滿 :data:`WAIT` 的話,後面每一張都晚 3 秒。
_LATE_WAIT = 1

#: 副露紀錄最多比**上一張捨牌**早幾輪。再舊的是之前的鳴牌,不是這一張的。
#:
#: 不能拿這一張捨牌來比:鳴完之後要想多久沒有上限。CPU 馬上打,真人想十幾秒,
#: 實測那些副露全被當成太舊,排成一次不存在的摸打。
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
        kan: :attr:`Action.CALL` 是不是大明槓。
    """

    action: Action
    actor: int
    pai: str | None = None
    guess: str | None = None
    target: int | None = None
    serial: int | None = None
    kan: bool = False


@dataclass(slots=True)
class _Pending:
    tick: float
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
    #: 多讀出來的格子 ``(誰, 第幾張)`` → 它其實是第幾張。立直記的可能是這一格。
    doubled: dict[tuple[int, int], int] = field(default_factory=dict)
    _queue: list[_Pending] = field(default_factory=list, repr=False)
    _melds: list[tuple[float, int, bool]] = field(default_factory=list, repr=False)
    _wall: int | None = field(default=None, repr=False)
    _wall_tick: float = field(default=-1, repr=False)
    _last_tick: float = field(default=0, repr=False)
    _last_serial: int | None = field(default=None, repr=False)
    #: 已經排出去、副露紀錄還沒到的鳴牌,依座位。
    _owed: list[int] = field(default_factory=lambda: [0] * _SEATS, repr=False)
    #: 這一輪副露區正在長、還沒確認的座位。見 :meth:`tick`。
    _growing: frozenset[int] = field(default=frozenset(), repr=False)
    #: 摸切手切還沒看出來的座位,與它們的捨牌最多等幾輪。見 :meth:`tick`。
    hold: float = 0.0
    _undecided: frozenset[int] = field(default=frozenset(), repr=False)
    _now: float = field(default=0.0, repr=False)
    #: 燈換手的紀錄 ``(第幾輪, 換到誰)``、換手之後第一次讀到的余牌,與排到哪一筆。見模組說明
    #: 「中央的燈」。
    _light: list[tuple[float, int]] = field(default_factory=list, repr=False)
    _light_walls: list[int | None] = field(default_factory=list, repr=False)
    _light_at: int = field(default=-1, repr=False)
    #: 下家摸打時余牌多減的座位 → 那張牌的輪。多減的可能是暗槓的嶺上,見 :meth:`_absorb`。
    _rinshan: dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.last = (self.oya - 1) % _SEATS

    def tick(
        self,
        tick: float,
        discards: Sequence[tuple[int, str | None, *tuple[str | int | None, ...]]] = (),
        melds: Sequence[int] = (),
        wall: int | None = None,
        *,
        hurry: bool = False,
        kans: Sequence[int] = (),
        growing: Sequence[int] = (),
        undecided: Sequence[int] = (),
    ) -> list[Step]:
        """收一輪的觀察,回傳這一輪可以確定的動作。

        Args:
            tick: 第幾輪,一輪一秒,可以有小數。見 :meth:`TableWorker._turn_clock`。
            discards: 這一輪新出現的捨牌 ``(座位, 牌)`` 或
                ``(座位, 牌, 猜測, 第幾張)``。牌讀不出來給 ``None``。
            melds: 這一輪副露多了一組的座位,一組一筆。
            wall: 這一輪讀到的余牌,讀不到給 ``None``。
            hurry: 輪到自己了,不必再等。見模組說明。
            kans: ``melds`` 裡寬度是槓的那幾家。
            growing: 副露區量到長了一組吃碰、還沒確認的座位。打牌之前余牌沒減的話,這兩個證據
                就夠說是鳴牌,不必等確認,見 :meth:`_called_early`。
            undecided: 摸切手切還沒看出來的座位。它們正常摸打的牌最多等 :attr:`hold` 輪再排,
                排進去那一刻就決定了,見 :mod:`mia.tracker.tedashi`。輪到自己時不等。
        """
        self._growing = frozenset(growing)
        self._undecided = frozenset(undecided)
        self._now = tick + WAIT if hurry else tick
        for seat in melds:
            if self._owed[seat]:
                self._owed[seat] -= 1
            else:
                self._melds.append((tick, seat, seat in kans))
        # 同一輪的並列照輪替排:誰離上一個打的人最近,誰先。
        anchor = self._queue[-1].actor if self._queue else self.last
        for seat, pai, *rest in sorted(discards, key=lambda d: self._distance(anchor, d[0])):
            guess, serial = rest if rest else (None, None)
            self._queue.append(_Pending(tick, seat, pai, guess, serial, self._wall))  # type: ignore[arg-type]
            anchor = seat
        if wall is not None:
            self._wall = wall
            self._wall_tick = tick
            self._light_saw_wall(wall)
        out: list[Step] = []
        while self._queue:
            self._restore_turn()
            steps = self._resolve(self._queue[0], tick + WAIT if hurry else tick)
            if steps is None:
                break
            item = self._queue.pop(0)
            if self._rinshan.get(item.actor, item.tick) < item.tick:
                # 只留到這一家的下一手:那時副露還沒出現,就不是暗槓。
                del self._rinshan[item.actor]
            if steps:
                out += steps
                lit = self._lit_for(item)
                if lit is not None:
                    self._light_at = lit
                self.last = item.actor
                self._last_tick = item.tick
                self._last_serial = item.serial
        return out

    def see_wall(self, tick: float, wall: int | None) -> None:
        """沒有整輪走完時也記下余牌。

        排序看的是牌落下**之前**的余牌,讀得太舊的話,打牌的人自己那次摸牌還沒算進去,看起來
        像鳴牌,這張牌與排在後面的牌就要等副露,實測等到 3.9 秒。
        """
        if wall is not None:
            self._wall, self._wall_tick = wall, tick
            self._light_saw_wall(wall)

    def see_light(self, tick: float, seat: int | None) -> None:
        """中央面板的燈亮在哪一家。讀不到(燈閃到暗的那一下、被蓋住)給 ``None``,不算換手。"""
        if seat is None or (self._light and self._light[-1][1] == seat):
            return
        self._light.append((tick, seat))
        self._light_walls.append(None)

    def _light_saw_wall(self, wall: int) -> None:
        """燈換手之後第一次讀到的余牌。燈比摸牌晚 0.3 秒以上才換,所以它一定含這次摸牌。"""
        if self._light_walls and self._light_walls[-1] is None:
            self._light_walls[-1] = wall

    def jump(
        self,
        tick: float,
        discards: Sequence[tuple[int, str | None, *tuple[str | int | None, ...]]],
        wall: int | None = None,
    ) -> list[Step]:
        """一次看到一大批牌:中途才開始看,或畫面被蓋了好幾秒。

        每家打了什麼、依什麼順序都在河裡,只有家與家之間的先後要推:照輪替排。鳴牌看不出來,但
        Mortal 要的是現在的局面,手牌與河都對,就比這一局都沒有建議好。
        """
        pending = [(i.actor, i.pai, i.guess, i.serial) for i in self._queue]
        for seat, pai, *rest in discards:
            guess, serial = rest if rest else (None, None)
            pending.append((seat, pai, guess, serial))  # type: ignore[arg-type]
        self._queue.clear()
        self._melds.clear()
        self._owed = [0] * _SEATS
        self._light_at = len(self._light) - 1
        self._rinshan.clear()
        seats = {seat: [d for d in pending if d[0] == seat] for seat in range(_SEATS)}
        out: list[Step] = []
        seat = (self.last + 1) % _SEATS
        while any(seats.values()):
            if seats[seat]:
                _, pai, guess, serial = seats[seat].pop(0)
                drop = Step(Action.DAHAI, seat, pai, guess, serial=serial)
                out += [Step(Action.TSUMO, seat), drop]
                self.drawn += 1
                self.last = seat
                self._last_serial = serial
            seat = (seat + 1) % _SEATS
        self._last_tick = tick
        if wall is not None:
            self._wall, self._wall_tick = wall, tick
            self.drawn = max(self.drawn, _FULL_WALL - wall)
        return out

    @property
    def next_up(self) -> Step | None:
        """還沒排定、但確定是下一張的捨牌。見 :mod:`mia.tracker.stream`「還沒排定的捨牌」。"""
        if not self._queue:
            return None
        head = self._queue[0]
        if head.pai is None and head.guess is None:
            return None
        if head.actor == 0 or head.actor != (self.last + 1) % _SEATS:
            return None
        if self._holding(head, self._now):
            # 先接在尾巴的牌也會送進引擎,摸切手切在那時就定了。
            return None
        return Step(Action.DAHAI, head.actor, head.pai, head.guess, serial=head.serial)

    def held(self, now: float) -> bool:
        """排頭那張在等摸切手切看出來,而且到 ``now`` 已經等完了:要再走一輪把它排出去,
        不能等到下一張牌落下。"""
        if not self._queue:
            return False
        head = self._queue[0]
        # 只在剛等完的那一小段:排頭可能還在等副露,那不是這裡要催的。
        return head.actor in self._undecided and self.hold <= now - head.tick < 2 * self.hold

    def _holding(self, item: _Pending, now: float) -> bool:
        return item.actor in self._undecided and now - item.tick < self.hold

    def _restore_turn(self) -> None:
        """排頭跳家又沒有副露,而該輪到的那一家就在後面:河晚到了,換回來。"""
        head = self._queue[0]
        turn = (self.last + 1) % _SEATS
        if head.actor == turn or self._has_meld(head.actor, head.tick):
            return
        for index, item in enumerate(self._queue):
            if item.actor == turn:
                # 它一定摸過牌:下家都打了。它讀到的余牌已經是之後的,不能拿來算。
                # 它在鳴牌就不是:吃了上一家的牌沒有摸,改成摸過的話吃就被排成摸打,整組不見
                # (電腦一打出就吃,吃的人的牌比再下一家的牌晚讀到)。
                if not (turn in self._growing or self._has_meld(turn, item.tick)):
                    item.wall = _FULL_WALL - self.drawn - 1
                self._queue.insert(0, self._queue.pop(index))
                return

    @staticmethod
    def _distance(anchor: int, seat: int) -> int:
        return (seat - anchor - 1) % _SEATS

    def _meld_since(self, tick: float) -> float:
        """這一輪的捨牌,副露最早可以記在哪一輪。"""
        return min(tick, self._last_tick) - _MELD_BEFORE

    def _take_meld(self, seat: int, tick: float) -> tuple[float, int, bool] | None:
        for entry in self._melds:
            if entry[1] == seat and entry[0] >= self._meld_since(tick):
                self._melds.remove(entry)
                return entry
        return None

    def _has_meld(self, seat: int, tick: float) -> bool:
        since = self._meld_since(tick)
        return any(s == seat and t >= since for t, s, _ in self._melds)

    def _kan_called(self, seat: int, tick: float) -> bool:
        """``seat`` 有一組寬度是槓的副露還沒排出去。"""
        since = self._meld_since(tick)
        return any(s == seat and t >= since and kan for t, s, kan in self._melds)

    def _skipped_seat_called(self, seat: int, tick: float) -> bool:
        """被跳過的那幾家有人鳴牌:跳家是它造成的,不是 ``seat`` 鳴的。"""
        lost = (seat - self.last - 1) % _SEATS
        skipped = {(self.last + 1 + offset) % _SEATS for offset in range(lost)}
        return any(self._has_meld(other, tick) for other in skipped)

    def _resolve(self, item: _Pending, now: float) -> list[Step] | None:
        """這張牌之前發生了什麼。還不能確定就回 ``None``。"""
        seat = item.actor
        known = _FULL_WALL - self.drawn if item.wall is None else item.wall
        unseen = _FULL_WALL - self.drawn - known
        drop = Step(Action.DAHAI, seat, item.pai, item.guess, serial=item.serial)
        if seat == self.last and self.drawn and unseen <= 0:
            # 同一家沒摸牌又打一張,不可能。橫放的宣言牌比一格寬,實測被讀成兩張。
            if item.serial is not None and self._last_serial is not None:
                self.doubled[seat, item.serial] = self._last_serial
            return []
        following = seat == (self.last + 1) % _SEATS
        lit = self._lit(item, unseen, following=following)
        if lit is not None:
            return lit
        # 自己的牌手牌看得到:沒有鳴牌就是摸打,余牌怎麼說都一樣。不等的話莊家第一張就要等滿四輪,
        # 那時候還沒讀過余牌。
        own = seat == 0 and not self._has_meld(seat, item.tick)
        plain = unseen == 1 or own
        # 下家大明槓也是余牌減一張,長得跟摸打一樣,只有副露的寬度分得開。
        if following and plain and not self._kan_called(seat, item.tick):
            if self._holding(item, now):
                return None
            self.drawn += 1
            return [Step(Action.TSUMO, seat), drop]
        if self._called_early(item, unseen, following=following):
            self._owed[seat] += 1
            return [Step(Action.CALL, seat, target=self.last), drop]
        # 跳到自己:前一家的牌漏讀了。等一下讓晚到的牌有機會排回去,但不能讓整桌等滿。
        patience = _LATE_WAIT if own or (following and unseen > 1) else WAIT
        if not self._has_meld(seat, item.tick) and now - item.tick < patience:
            return None
        meld = self._take_meld(seat, item.tick)
        if meld is not None:
            if meld[2] and following and self._rinshan.pop(seat, None) is not None:
                # 上一手余牌多減的那張是嶺上:暗槓在上一手,副露現在才確認。排在這一手,
                # 牌張與寶牌都對,只是晚一手。
                self.drawn += 1
                draw = Step(Action.TSUMO, seat)
                return [draw, Step(Action.CALL, seat, kan=True), draw, drop]
            if meld[2] and (not following or unseen <= 1):
                return self._open_kan(seat, unseen, drop)
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
        elif (
            not following
            and unseen <= 0
            and behind - (len(self._queue) - 1) <= 1
            and not self._skipped_seat_called(seat, item.tick)
        ):
            # 跳家、打之前沒有人摸牌,之後只減了下一家的那一張:只可能是鳴牌。副露晚到而已,
            # 鳴完打牌之後馬上輪到自己時,副露還沒確認就得排出來。減得比這更多的話,是一輪
            # 湧進好幾張、余牌來不及更新,這時候 ``unseen`` 不可信,中間其實有人摸過。
            self._owed[seat] += 1
            return [Step(Action.CALL, seat, target=self.last), drop]
        if not following:
            # 跳家卻沒有副露,而余牌有減:中間漏看了。補上再接回去,不然 last 停在
            # 原地,之後每一張都變成「跳家且沒有副露」,整局後面一路被丟。
            lost = (seat - self.last - 1) % _SEATS
            steps = self._hidden(seat, min(lost, max(behind - 1, 0)))
            self.drawn += 1
            return [*steps, Step(Action.TSUMO, seat), drop]
        self._absorb(item, unseen, following=following)
        return [Step(Action.TSUMO, seat), drop]

    def _absorb(self, item: _Pending, unseen: int, *, following: bool) -> None:
        """正常摸打,順便吸收沒交代的摸牌。

        佇列裡後面那幾張的摸牌也算在差額裡,要留給它們。全部吸收的話會多算,之後每一張
        看起來都沒人摸,真的捨牌就被當成假的丟掉。

        下家多減的那張可能是加槓或暗槓的嶺上。暗槓的副露比牌晚一輪以上才確認,那時再看是
        摸打,只剩大明槓一種排法,鳴的卻是上家剛打、還在河裡的牌,整局停住或錯一組(011937 東一局、
        164613 南一局)。所以記下來,他下一手看到槓寬的副露就排成暗槓,晚一手但牌張與寶牌都對。
        """
        extra = unseen - (len(self._queue) - 1)
        self.drawn += max(1, extra)
        if following and extra > 1 and item.actor != 0:
            # 自己的槓看手牌就知道,不靠這個。
            self._rinshan[item.actor] = item.tick

    def _lit_for(self, item: _Pending) -> int | None:
        """燈最後一次換到 ``item`` 那一家的那一筆:這張牌看到之前、還沒排過的。

        太舊的不算:它是這一家上一輪的,這一輪的燈還沒換過來(電腦打得快時常常這樣)。上一輪那筆
        的前一家一樣是 ``last``,只看順序分不出來。
        """
        for index in range(len(self._light) - 1, self._light_at, -1):
            tick, seat = self._light[index]
            if seat == item.actor and tick <= item.tick:
                return index if tick >= self._last_tick - _LIGHT_STALE else None
        return None

    def _lit(self, item: _Pending, unseen: int, *, following: bool) -> list[Step] | None:
        """燈說這張牌之前發生了什麼。說不清楚就回 ``None``,照余牌推。見模組說明「中央的燈」。"""
        seat = item.actor
        lit = self._lit_for(item)
        if seat == 0 or not self.drawn or not lit or self._light[lit - 1][1] != self.last:
            # 燈不是從 ``last`` 直接換過來的:中間有看不到的動作,或燈晚到了。
            return None
        drop = Step(Action.DAHAI, seat, item.pai, item.guess, serial=item.serial)
        if following:
            if seat in self._growing or self._has_meld(seat, item.tick):
                return None
            # 下家吃也是燈直接換過來。分得開的是余牌:兩次換手之後各讀到的第一個余牌,摸了就少一張。
            # 只比這兩個讀數,不比算出來的張數,所以之前多算少算都不影響。
            before, after = self._light_walls[lit - 1], self._light_walls[lit]
            if before is None or after is None or after >= before:
                return None
            if self._holding(item, self._now):
                return None
            # 燈換過來之後讀到的余牌一定含這次摸牌。它說摸過的張數沒超過已經算的,就是之前多算了
            # 一張,這次不再加。
            # 少算的(加槓的嶺上讓余牌多減一張)照余牌那條路吸收,見最後的正常摸打。
            wall = self._light_walls[lit]
            if wall is None or _FULL_WALL - wall > self.drawn:
                self._absorb(item, unseen, following=True)
            return [Step(Action.TSUMO, seat), drop]
        if unseen > 0:
            # 余牌有減:可能是大明槓的嶺上,或中間有人摸打,讓余牌那條路分。
            return None
        meld = self._take_meld(seat, item.tick)
        if meld is None:
            self._owed[seat] += 1
        elif meld[2]:
            return None
        return [Step(Action.CALL, seat, target=self.last), drop]

    def _called_early(self, item: _Pending, unseen: int, *, following: bool) -> bool:
        """副露還沒確認,但已經看得出這張牌是鳴了之後打的。

        確認要隔 0.4 秒再讀一次一樣的長度,實測鳴牌的人打的牌晚 1.2 秒才排進去,後面兩家的牌排在
        它之後,也跟著晚。這裡三個證據都要有:打之前余牌沒減(沒人摸牌)、余牌讀過(讀不到的話
        ``unseen`` 一律是 0)、他的副露區正在長一組吃碰的長度。結算畫面的假牌只滿足第一個。

        確認之後的那一筆由 ``_owed`` 抵掉,不會再鳴一次。自己的鳴牌看手牌,不走這裡;第一張就被鳴
        走的話前面還有一次看不到的摸打,也不走這裡。
        """
        seat = item.actor
        return (
            seat != 0
            and seat in self._growing
            and not self._has_meld(seat, item.tick)
            and item.wall is not None
            and unseen <= 0
            and self.drawn > 0
            and (following or not self._skipped_seat_called(seat, item.tick))
        )

    def _open_kan(self, seat: int, unseen: int, drop: Step) -> list[Step]:
        """大明槓:鳴、嶺上、打。``unseen`` 裡最後一張是嶺上。"""
        hidden = self._hidden(seat, unseen - 1)
        target = hidden[-1].actor if hidden else self.last
        self.drawn += 1
        call = Step(Action.CALL, seat, target=target, kan=True)
        return [*hidden, call, Step(Action.TSUMO, seat), drop]

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
