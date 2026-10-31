"""目前知道的一切 → 這一局的 MJAI 事件流。給 Mortal 的。

每一輪重算,不是一路累加
------------------------
事件流一路累加的話,發出去就收不回來;而畫面這條路的知識是**會變多**的:副露的
牌面晚四五秒才認好、立直要點數變了才知道、河裡的牌事後可能被鳴走才知道是什麼。
所以這裡是一個純函式::

    compose(到目前為止知道的一切) -> 這一局完整的事件流

知道得更多,輸出就跟著變。送進引擎的時候只比前綴:新的流接得上舊的就送尾巴,接不上
就從 ``start_game`` 重播這一局(Mortal 只有幾十毫秒的成本,見
:meth:`mia.live.table.TableWorker.events`)。

libriichi 的硬規則
------------------
這些不成立,引擎就報錯、整局沒有建議(``state/update.rs``):

* **別家的牌不能是 ``?``。** 打牌與副露都要是一張確定的牌。讀不出來的格子用最佳
  猜測(沒信心的格子約 13%,猜對約八成),別家的牌猜錯只影響現物判斷,不會破壞狀態。
* **同一種牌不能看到第五張。** 猜錯的牌佔掉名額的話,之後**真的**那一張就違規了。
  所以先替確定的牌(自己的手牌、摸到的牌、寶牌、有信心的捨牌)保留名額,猜測只能用
  剩下的。這也是非得整局重算的理由:要知道「之後」會看到什麼。
* **打的牌一定要在手上**,摸到的牌一定要知道。自己的動作全部來自
  :class:`~mia.tracker.hand.HandWatcher`。

這些規則不只是「盡量遵守」:引擎報錯一次,:class:`~mia.engine.multiplex.EngineGroup`
就把它整場移除,之後再也沒有建議。所以送出去之前要自己驗過
(:func:`_legal_length`),不合法的地方就停在那裡,等更多資訊來改寫。

自己的動作走在 :class:`~mia.tracker.turns.TurnTracker` 前面
---------------------------------------------------------
TurnTracker 是捨牌驅動的,打出去才排得出那一手。但 Mortal 要在**打之前**回答,
所以手牌一變成 14 張,摸牌就接在串流尾巴上。之後 TurnTracker 排出同一步時
內容一樣,前綴接得上。

不需要的事件
------------
``Hora`` / ``Ryukyoku`` 對 Mortal 沒有作用:狀態機直接略過,下一個 ``start_kyoku``
會重設一切,而分數在 HUD 上。所以局尾只送 ``end_kyoku``。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    Kakan,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    StartKyoku,
    Tsumo,
)
from mia.mjai.tiles import HONOR_ORDER, UNKNOWN, normalize_red
from mia.tracker.hand import Move, OwnMove
from mia.tracker.turns import Action, Step
from mia.vision.hud import Hud

__all__ = ["KyokuLog", "Stream", "compose"]

_SEATS = 4
_COPIES = 4
_NUMBERS = range(1, 10)

#: 什麼都不知道時要填的牌,依序試。字牌與么九在前:開局多半打這些,也不會變成
#: 別人的搭子。
_FILLERS = (
    *HONOR_ORDER,
    *(f"{n}{suit}" for n in (1, 9) for suit in "mps"),
    *(f"{n}{suit}" for n in (2, 8, 3, 7, 4, 6, 5) for suit in "mps"),
)


@dataclass
class KyokuLog:
    """一局到目前為止知道的一切。

    Attributes:
        hud: 開局的 HUD。
        tehai: 自己的配牌。
        doras: 寶牌指示牌 ``(翻開時已經排出幾步, 牌)``,第一張是開局的。
        steps: :class:`~mia.tracker.turns.TurnTracker` 排出來的順序。
        melds: 四家的副露,牌面認到多少算多少,依發生順序。
        reaches: 別家立直 ``(誰, 他的第幾張捨牌是宣言牌)``。
        own_reach: 自己立直,宣言牌是自己第幾次打牌。與別家分開數:宣言牌一打出就
            被鳴走時,河裡從沒有過那一張,而手牌一定看得到。
        moves: 自己的動作。
        readings: 捨牌後來重讀的結果,``(誰, 第幾張) -> (牌, 猜測)``。蓋過
            :attr:`~mia.tracker.turns.Step.pai`:立直那一列的格線要等橫放牌找到才
            挪得對,在那之前讀到的是歪的。
    """

    hud: Hud
    tehai: tuple[str, ...] = ()
    doras: list[tuple[int, str]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    melds: list[list[tuple[str, ...]]] = field(
        default_factory=lambda: [[] for _ in range(_SEATS)]
    )
    reaches: list[tuple[int, int]] = field(default_factory=list)
    own_reach: int | None = None
    moves: list[OwnMove] = field(default_factory=list)
    readings: dict[tuple[int, int], tuple[str | None, str | None]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Stream:
    """組出來的事件流。

    Attributes:
        events: 從 ``start_kyoku`` 起的事件。缺開局資訊時是空的。
        stalled: 串流停在這裡,後面的步還排不出來(等自己的手牌)。
        forced: 幾張有信心的牌因為違反張數被換掉。不是零代表有一張讀錯了。
        doubt: 串流對不上實際局面的理由。不是 ``None`` 時不該給建議。
    """

    events: tuple[MjaiEvent, ...] = ()
    stalled: bool = False
    forced: int = 0
    doubt: str | None = None


def compose(log: KyokuLog) -> Stream:
    """見模組說明。開局資訊不齊時回空的 :class:`Stream`。"""
    hud = log.hud
    if (
        len(log.tehai) != 13
        or not log.doras
        or hud.bakaze is None
        or hud.kyoku is None
        or hud.dealer is None
        or not hud.scores_agree
    ):
        return Stream()
    return _Composer(log).run()


class _Composer:
    def __init__(self, log: KyokuLog) -> None:
        self.log = log
        self.steps = [self._updated(step) for step in log.steps]
        self.left = self._budget()
        self.events: list[MjaiEvent] = []
        self.moves = list(log.moves)
        self.calls = [0 for _ in range(_SEATS)]
        self.own_drops = 0
        self.unmatched = sum(s.action is Action.TSUMO and s.actor == 0 for s in self.steps) - sum(
            m.move is Move.DRAW for m in log.moves
        )
        self.doras = list(log.doras[1:])
        self.forced = 0
        self.stalled = False

    def _updated(self, step: Step) -> Step:
        if step.serial is None:
            return step
        reading = self.log.readings.get((step.actor, step.serial))
        return step if reading is None else replace(step, pai=reading[0], guess=reading[1])

    def _budget(self) -> Counter[str]:
        """每種牌還剩幾個名額:確定的牌先佔。"""
        left = Counter({tile: _COPIES for tile in self._all_tiles()})
        certain = [*self.log.tehai, *(pai for _, pai in self.log.doras)]
        certain += [m.tiles[0] for m in self.log.moves if m.move is Move.DRAW]
        certain += [
            s.pai
            for s in self.steps
            if s.action is Action.DAHAI and s.actor != 0 and s.pai is not None
        ]
        left.subtract(normalize_red(tile) for tile in certain)
        return left

    @staticmethod
    def _all_tiles() -> list[str]:
        numbers = [f"{n}{suit}" for suit in "mps" for n in _NUMBERS]
        return [*numbers, *HONOR_ORDER]

    # ------------------------------------------------------------------ 主流程

    def run(self) -> Stream:
        hud = self.log.hud
        assert hud.bakaze is not None and hud.kyoku is not None and hud.dealer is not None
        self.events.append(
            StartKyoku(
                bakaze=hud.bakaze,
                kyoku=hud.kyoku,
                honba=hud.honba or 0,
                kyotaku=hud.kyotaku or 0,
                oya=hud.dealer,
                dora_marker=self.log.doras[0][1],
                tehais=[list(self.log.tehai), *([UNKNOWN] * 13 for _ in range(_SEATS - 1))],
                scores=[score or 0 for score in hud.scores],
            )
        )
        steps = self.steps
        for index, step in enumerate(steps):
            self._flush_doras(index)
            if not self._walk(step, index):
                self.stalled = True
                break
        else:
            self._flush_doras(len(steps))
            self._tail()
        length = _legal_length(self.events)
        return Stream(
            tuple(self.events[:length]),
            self.stalled or length < len(self.events),
            self.forced,
        )

    def _flush_doras(self, index: int) -> None:
        while self.doras and self.doras[0][0] <= index:
            self.events.append(Dora(dora_marker=self.doras.pop(0)[1]))

    def _walk(self, step: Step, index: int) -> bool:
        if step.action is Action.TSUMO:
            return self._tsumo(step, index)
        if step.action is Action.CALL:
            return self._call(step)
        return self._dahai(step)

    def _tail(self) -> None:
        """TurnTracker 還沒排到、但手牌已經看到的動作。"""
        while self.moves and self.moves[0].move is not Move.DISCARD:
            move = self.moves[0]
            if move.move is Move.DRAW:
                self.moves.pop(0)
                self.events.append(Tsumo(actor=0, pai=move.tiles[0]))
                continue
            call = self._own_call(move)
            if call is None:
                return
            self.moves.pop(0)
            self.events.append(call)

    # -------------------------------------------------------------------- 摸打

    def _tsumo(self, step: Step, index: int) -> bool:
        if step.actor != 0:
            self.events.append(Tsumo(actor=step.actor, pai=UNKNOWN))
            return True
        unseen = self._unseen_turn(index)
        if self.moves and self.moves[0].move is Move.DRAW and not unseen:
            self.events.append(Tsumo(actor=0, pai=self.moves.pop(0).tiles[0]))
            return True
        following = self.steps[index + 1] if index + 1 < len(self.steps) else None
        if following is None or following.action is not Action.DAHAI:
            return False
        if self.moves and not unseen:
            return False
        # 手牌完全沒看到這一手:摸到就打掉。立直之後的摸切只有一兩秒。
        pai = following.pai or following.guess
        if pai is None or self.left[normalize_red(pai)] <= 0:
            return False
        self.left[normalize_red(pai)] -= 1
        self.unmatched -= 1
        self.events.append(Tsumo(actor=0, pai=pai))
        self.moves.insert(0, OwnMove(Move.DISCARD, (pai,), tsumogiri=True))
        return True

    def _unseen_turn(self, index: int) -> bool:
        """第 ``index`` 步的摸牌,手牌沒看到。

        河裡有把握的那張不是手牌接下來打掉的那張,有兩種解釋:河裡認錯了,或手牌漏了
        這一手。分得開的是河裡**下一張**:對得上手牌接下來要打的牌,就是漏了一手。
        只看這一張的話,認錯的牌會被當成摸到的牌,實測弄出過第五張東風。
        下一張還沒出現時數摸牌:排出來的比手牌看到的多,才算漏了。
        """
        river = self._own_discards(index)
        if not river or len(self.moves) < 2 or self.moves[1].move is not Move.DISCARD:
            return False
        planned = normalize_red(self.moves[1].tiles[0])
        if normalize_red(river[0]) == planned:
            return False
        return normalize_red(river[1]) == planned if len(river) > 1 else self.unmatched > 0

    def _own_discards(self, index: int) -> list[str]:
        """第 ``index`` 步之後,河裡有把握的自己的牌。第一張要是緊接著的那一步。"""
        following = self.steps[index + 1] if index + 1 < len(self.steps) else None
        if following is None or following.action is not Action.DAHAI or following.pai is None:
            return []
        return [
            step.pai
            for step in self.steps[index + 1 :]
            if step.action is Action.DAHAI and step.actor == 0 and step.pai is not None
        ]

    def _dahai(self, step: Step) -> bool:
        reached = (step.actor, step.serial) in self.log.reaches
        if step.actor == 0:
            reached = self.own_drops == self.log.own_reach
            self.own_drops += 1
        if reached:
            self.events.append(Reach(actor=step.actor))
        if step.actor == 0:
            if not (self.moves and self.moves[0].move is Move.DISCARD):
                return False
            move = self.moves.pop(0)
            self.events.append(Dahai(actor=0, pai=move.tiles[0], tsumogiri=move.tsumogiri))
        else:
            self.events.append(
                Dahai(actor=step.actor, pai=self._discarded(step), tsumogiri=False)
            )
        if reached:
            self.events.append(ReachAccepted(actor=step.actor))
        return True

    def _discarded(self, step: Step) -> str:
        """別家打的這張。有信心的已經佔過名額了。"""
        if step.pai is not None:
            if self.left[normalize_red(step.pai)] >= 0:
                return step.pai
            # 張數超過了:這張一定讀錯了。
            self.left[normalize_red(step.pai)] += 1
            self.forced += 1
        return self._pick(step.guess)

    def _pick(self, *wanted: str | None) -> str:
        """照順序挑第一張還有名額的牌,都沒有就挑一張最不要緊的。"""
        for tile in wanted:
            if tile is not None and tile != UNKNOWN and self.left[normalize_red(tile)] > 0:
                self.left[normalize_red(tile)] -= 1
                return tile
        for tile in _FILLERS:
            if self.left[tile] > 0:
                self.left[tile] -= 1
                return tile
        return _FILLERS[0]

    # -------------------------------------------------------------------- 鳴牌

    def _call(self, step: Step) -> bool:
        if step.actor == 0:
            if not (self.moves and self.moves[0].move is Move.CALL):
                return False
            call = self._own_call(self.moves[0])
            if call is None:
                return False
            self.moves.pop(0)
            self.events.append(call)
            return True
        groups = self.log.melds[step.actor]
        index = self.calls[step.actor]
        self.calls[step.actor] += 1
        faces = groups[index] if index < len(groups) else ()
        known = [tile for tile in faces if tile != UNKNOWN]
        if step.target is None:
            self.events.append(self._ankan(step.actor, known))
            return True
        if not any(isinstance(event, Dahai) for event in self.events):
            # 被鳴的那張還沒排出來,可能是自己打的、手牌還沒跟上。
            return False
        self.events.append(self._open_meld(step, known))
        return True

    def _own_call(self, move: OwnMove) -> MjaiEvent | None:
        """自己的鳴牌。手上拿出去幾張就是哪一種;別人的牌要接在那張牌後面。"""
        tiles = list(move.tiles)
        if len(tiles) == _COPIES:
            return Ankan(actor=0, consumed=sorted(tiles))
        if len(tiles) == 1:
            return Kakan(actor=0, pai=tiles[0], consumed=[normalize_red(tiles[0])] * 3)
        last = self.events[-1]
        if not isinstance(last, Dahai) or last.actor == 0:
            return None
        if len(tiles) == 3:
            return Daiminkan(actor=0, target=last.actor, pai=last.pai, consumed=tiles)
        if normalize_red(tiles[0]) == normalize_red(tiles[1]):
            return Pon(actor=0, target=last.actor, pai=last.pai, consumed=tiles)
        return Chi(actor=0, target=last.actor, pai=last.pai, consumed=tiles)

    def _ankan(self, actor: int, known: list[str]) -> Ankan:
        tile = known[0] if known else None
        if tile is None or self.left[normalize_red(tile)] < _COPIES:
            full = [t for t in _FILLERS if self.left[t] >= _COPIES]
            tile = full[0] if full else _FILLERS[0]
        self.left[normalize_red(tile)] -= _COPIES
        return Ankan(actor=actor, consumed=[tile] * _COPIES)

    def _open_meld(self, step: Step, known: list[str]) -> MjaiEvent:
        """別家的吃碰。``pai`` 是剛打出來的牌;牌面說不是的話聽牌面的。"""
        assert step.target is not None
        at = self._last_dahai()
        pai = self.events[at].pai  # type: ignore[attr-defined]
        plain = normalize_red(pai)
        adjacent = step.actor == (step.target + 1) % _SEATS
        triple = len(known) == 3 and len(set(known)) == 1
        if triple or not adjacent or plain in HONOR_ORDER or (known and known[0] in HONOR_ORDER):
            tile = known[0] if known else plain
            pai = self._replace(at, tile)
            consumed = [self._pick(tile), self._pick(tile)]
            return Pon(actor=step.actor, target=step.target, pai=pai, consumed=consumed)
        pai, others = self._run(at, plain, known)
        consumed = [self._pick(tile) for tile in others]
        return Chi(actor=step.actor, target=step.target, pai=pai, consumed=consumed)

    def _last_dahai(self) -> int:
        for index in range(len(self.events) - 1, -1, -1):
            if isinstance(self.events[index], Dahai):
                return index
        raise AssertionError("鳴牌之前一定有人打牌")

    def _replace(self, at: int, tile: str) -> str:
        """把第 ``at`` 個事件打的牌換掉。牌面認到的比河裡猜的可靠。"""
        event = self.events[at]
        assert isinstance(event, Dahai)
        if normalize_red(event.pai) == normalize_red(tile):
            return event.pai
        self.left[normalize_red(event.pai)] += 1
        pai = self._pick(tile)
        self.events[at] = replace(event, pai=pai)
        return pai

    def _run(self, at: int, plain: str, known: list[str]) -> tuple[str, list[str]]:
        """吃的那一組:要含打出來的牌、也要含認到的牌面。"""
        shapes = _runs_with(plain)
        fitting = [run for run in shapes if set(known) <= set(run)]
        if not fitting and known:
            # 河裡的牌猜錯了。順子有一張是打出來的,剩下的認到了。
            whole = _run_of(known)
            if whole is not None:
                tile = next(t for t in whole if t not in known[:2])
                self._replace(at, tile)
                return self.events[at].pai, [t for t in whole if t != tile]  # type: ignore[attr-defined]
        run = max(fitting or shapes, key=lambda r: sum(self.left[t] for t in r if t != plain))
        return self.events[at].pai, [t for t in run if t != plain]  # type: ignore[attr-defined]


def _legal_length(events: list[MjaiEvent]) -> int:
    """前面幾個事件是 libriichi 收得下的。規則見模組說明。"""
    seen: Counter[str] = Counter()
    hand: Counter[str] = Counter()

    def witness(tiles: Iterable[str]) -> bool:
        for tile in tiles:
            seen[normalize_red(tile)] += 1
        return all(seen[normalize_red(tile)] <= _COPIES for tile in tiles)

    def spend(tiles: Iterable[str]) -> bool:
        for tile in tiles:
            hand[normalize_red(tile)] -= 1
        return all(hand[normalize_red(tile)] >= 0 for tile in tiles)

    for index, event in enumerate(events):
        match event:
            case StartKyoku():
                hand.update(normalize_red(tile) for tile in event.tehais[0])
                fine = witness([*event.tehais[0], event.dora_marker])
            case Tsumo(actor=0):
                hand[normalize_red(event.pai)] += 1
                fine = witness([event.pai])
            case Dahai(actor=0):
                fine = spend([event.pai])
            case Dahai():
                fine = witness([event.pai])
            case Dora():
                fine = witness([event.dora_marker])
            case Chi() | Pon() | Daiminkan() | Ankan():
                fine = (spend if event.actor == 0 else witness)(event.consumed)
            case Kakan(actor=0):
                fine = spend([event.pai])
            case _:
                fine = True
        if not fine:
            return index
    return len(events)


def _runs_with(tile: str) -> list[tuple[str, str, str]]:
    """含這張牌的順子。字牌沒有,但呼叫端已經擋掉了。"""
    number, suit = int(tile[0]), tile[1]
    starts = [start for start in (number - 2, number - 1, number) if 1 <= start <= 7]
    return [(f"{s}{suit}", f"{s + 1}{suit}", f"{s + 2}{suit}") for s in starts]


def _run_of(tiles: list[str]) -> tuple[str, str, str] | None:
    """認到的牌湊出的那個順子,湊不出來回 ``None``。"""
    if len(tiles) < 3:
        return None
    return (tiles[0], tiles[1], tiles[2])
