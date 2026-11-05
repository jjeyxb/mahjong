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
* **牌山只有 70 張。** 嶺上也算在內,多摸一張就報錯。局尾的結算畫面被讀成捨牌時
  踩到過。

這些規則不只是「盡量遵守」:引擎報錯一次,:class:`~mia.engine.multiplex.EngineGroup`
就把它整場移除,之後再也沒有建議。所以送出去之前要自己驗過
(:func:`_legal_length`),不合法的地方就停在那裡,等更多資訊來改寫。

自己的動作走在 :class:`~mia.tracker.turns.TurnTracker` 前面
---------------------------------------------------------
TurnTracker 是捨牌驅動的,打出去才排得出那一手。但 Mortal 要在**打之前**回答,
所以手牌一變成 14 張,摸牌就接在串流尾巴上。之後 TurnTracker 排出同一步時
內容一樣,前綴接得上。

加槓看不到,但寶牌看得到
------------------------
別家加槓的那張牌疊在橫放那張上面,副露不變長,看不出來。余牌會多減一張,但真人打得快
時要晚兩三手才反映出來,那時候已經輪到別人了。可靠的是寶牌:**每一次槓都會翻一張**,
翻出來的當下畫面就看得到。所以一張沒有對應的槓的新寶牌,就是下一個有碰可以加的人加槓。

還沒排定的捨牌
--------------
下家摸完馬上打時余牌還沒更新,排序要等著看是不是鳴牌,最多四輪。但這張牌是真的,不管
他是摸打還是吃完打的,自己都能吃碰它,而吃碰的按鈕只亮幾秒。所以 :attr:`KyokuLog.next_up`
先接在尾巴上問 Mortal。之後排出來的不一樣,下一輪重算就改掉了。

手牌與排序對不上時
------------------
上面每一步都要手牌與排序器說同一件事。兩邊分歧的方式很多(漏讀一副手牌、鳴牌太快、牌認錯),
一種一種補永遠補不完,而沒補到的那種會讓事件流停到局尾,建議就沒了。

但 Mortal 要的是**現在的局面**,不是歷史:手牌對、河對、余牌對,就算過去的摸打不完全對,答案也
差不多。所以分歧持續一陣子之後,:func:`resynced` 用畫面上現在的手牌與河重寫自己的動作:配牌換成
推回去的手牌,之前每一手當成摸切。摸切與手切的標記會失真,其他都是真的,這比沒有建議好得多。

不需要的事件
------------
``Hora`` / ``Ryukyoku`` 對 Mortal 沒有作用:狀態機直接略過,下一個 ``start_kyoku``
會重設一切,而分數在 HUD 上。所以局尾只送 ``end_kyoku``。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Protocol

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

__all__ = ["KyokuLog", "MeldLook", "MeldPick", "MeldShape", "Stream", "compose", "resynced"]

_SEATS = 4
_COPIES = 4
_WALL = 70
_NUMBERS = range(1, 10)
_FIVES = ("5m", "5p", "5s")

#: 什麼都不知道時要填的牌,依序試。字牌與么九在前:開局多半打這些,也不會變成
#: 別人的搭子。
_FILLERS = (
    *HONOR_ORDER,
    *(f"{n}{suit}" for n in (1, 9) for suit in "mps"),
    *(f"{n}{suit}" for n in (2, 8, 3, 7, 4, 6, 5) for suit in "mps"),
)


#: 新寶牌翻開之後,再過幾步還找不到是誰槓的就不找了。排序最多落後一輪。
_KAN_REACH = 12

#: 碰的橫放牌在第幾張,依 ``(被鳴的人 - 鳴的人) % 4``:上家的放最左、對家的放中間、下家的
#: 放最右。吃一定是上家的,排成 [被吃的, 小, 大]。
_PON_SIDEWAYS = {3: 0, 2: 1, 1: 2}


@dataclass(frozen=True, slots=True)
class MeldShape:
    """一組吃碰可能的樣子。

    Attributes:
        tiles: 由左而右的牌,MJAI 記法、不分赤。
        sideways: 橫放的是第幾張,也就是被鳴的那張。
    """

    tiles: tuple[str, ...]
    sideways: int


@dataclass(frozen=True, slots=True)
class MeldPick:
    """畫面選出來的候選。

    Attributes:
        index: 第幾個候選。
        reds: 每一張是不是赤五,由左而右。
    """

    index: int
    reds: tuple[bool, ...]


class MeldLook(Protocol):
    """拿畫面比一組副露的候選,見 :mod:`mia.vision.tiles.meld_fit`。

    不能卡住:一組要比零點幾秒,而事件流每一輪都重算。還沒比完就回 ``None``,下一輪再問。
    """

    def pick(self, shapes: tuple[MeldShape, ...]) -> MeldPick | None: ...

    @property
    def pending(self) -> bool: ...


@dataclass
class KyokuLog:
    """一局到目前為止知道的一切。

    Attributes:
        hud: 開局的 HUD。
        tehai: 自己的配牌。
        doras: 寶牌指示牌 ``(翻開時已經排出幾步, 牌)``,第一張是開局的。
        steps: :class:`~mia.tracker.turns.TurnTracker` 排出來的順序。
        melds: 四家的副露,牌面認到多少算多少,依發生順序。
        meld_reds: 副露裡看到赤五 ``(誰, 第幾組) -> 花色``。見
            :attr:`~mia.vision.tiles.melds.MeldFaces.red_suits`。
        meld_looks: 別家每一組吃碰的畫面 ``(誰, 第幾組)``,拿來比候選。見 :class:`MeldLook`。
        meld_growing: 畫面正在長出新一組、尚未確認的座位。推測的數牌鳴牌等它確認再認牌面。
        meld_sideways: 只認到一張、而且只在橫的框裡認到的副露 ``(誰, 第幾組)``。那張是
            被鳴的牌,分不出吃碰。見 :attr:`~mia.vision.tiles.melds.MeldFaces.upright`。
        tsumogiri: 別家的捨牌是不是摸切 ``(誰, 第幾張) -> 摸切``,排進事件流那一刻看手牌的
            形狀決定,之後不改。見 :mod:`mia.tracker.tedashi`。
        reaches: 別家立直 ``(誰, 他的第幾張捨牌是宣言牌)``。
        own_reach: 自己立直,宣言牌是自己第幾次打牌。與別家分開數:宣言牌一打出就
            被鳴走時,河裡從沒有過那一張,而手牌一定看得到。
        moves: 自己的動作。
        readings: 捨牌後來重讀的結果,``(誰, 第幾張) -> (牌, 猜測)``。蓋過
            :attr:`~mia.tracker.turns.Step.pai`:立直那一列的格線要等橫放牌找到才
            挪得對,在那之前讀到的是歪的。
        next_up: 還沒排定、但確定是下一張的捨牌。
    """

    hud: Hud
    tehai: tuple[str, ...] = ()
    doras: list[tuple[int, str]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    melds: list[list[tuple[str, ...]]] = field(
        default_factory=lambda: [[] for _ in range(_SEATS)]
    )
    meld_reds: dict[tuple[int, int], frozenset[str]] = field(default_factory=dict)
    meld_sideways: set[tuple[int, int]] = field(default_factory=set)
    meld_looks: dict[tuple[int, int], MeldLook] = field(default_factory=dict)
    meld_growing: set[int] = field(default_factory=set)
    tsumogiri: dict[tuple[int, int | None], bool] = field(default_factory=dict)
    reaches: list[tuple[int, int]] = field(default_factory=list)
    own_reach: int | None = None
    moves: list[OwnMove] = field(default_factory=list)
    readings: dict[tuple[int, int], tuple[str | None, str | None]] = field(default_factory=dict)
    next_up: Step | None = None


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


def resynced(
    log: KyokuLog, tiles: Iterable[str], drawn: str | None
) -> tuple[tuple[str, ...], list[OwnMove]] | None:
    """以畫面上現在的手牌為準,重寫配牌與自己的動作。見模組說明。

    Args:
        log: 這一局到目前為止知道的一切。
        tiles: 現在手上的牌,含 ``drawn``。
        drawn: 其中剛摸到的那張。

    Returns:
        ``(配牌, 動作)``。湊不出 13 張配牌回 ``None``:手牌與排序器連張數都對不上,硬寫只會更亂。
    """
    hand = list(tiles)
    if drawn is not None:
        if drawn not in hand:
            return None
        hand.remove(drawn)
    calls = [move for move in log.moves if move.move is Move.CALL]
    moves: list[OwnMove] = []
    spent: list[str] = []
    drew = False
    discards = sum(move.move is Move.DISCARD for move in log.moves)
    for step in _Composer(log, discards=discards).steps:
        if step.actor != 0:
            continue
        if step.action is Action.TSUMO:
            drew = True
        elif step.action is Action.CALL:
            if not calls:
                return None
            moves.append(calls.pop(0))
            spent += moves[-1].tiles
            drew = False
        else:
            pai = step.pai or step.guess
            if pai is None:
                return None
            if drew:
                moves += [OwnMove(Move.DRAW, (pai,)), OwnMove(Move.DISCARD, (pai,), tsumogiri=True)]
            else:
                moves.append(OwnMove(Move.DISCARD, (pai,)))
                spent.append(pai)
            drew = False
    if drawn is not None:
        moves.append(OwnMove(Move.DRAW, (drawn,)))
    tehai = (*hand, *spent)
    return (tehai, moves) if len(tehai) == 13 else None


class _Composer:
    def __init__(self, log: KyokuLog, *, discards: int | None = None) -> None:
        self.log = log
        if discards is None:
            discards = sum(move.move is Move.DISCARD for move in log.moves)
        self.steps = _without_phantom_ankans(
            _without_phantom_turns([self._updated(step) for step in log.steps], discards),
            len(log.doras) - 1,
        )
        self.left = self._budget()
        self.events: list[MjaiEvent] = []
        self.moves = list(log.moves)
        self.calls = [0 for _ in range(_SEATS)]
        self.own_drops = 0
        #: 手牌先排出來、副露區還沒確認的自己的暗槓。見 :meth:`_own_kans`。
        self.early_kans = 0
        self.unmatched = sum(s.action is Action.TSUMO and s.actor == 0 for s in self.steps) - sum(
            m.move is Move.DRAW for m in log.moves
        )
        self.doras, self.loose = self._timed_doras()
        self.forced = 0
        self.stalled = False
        #: 最後打出來的那張,河裡是有把握讀到的。
        self.sure = False

    def _timed_doras(self) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        """每張槓寶牌該接在哪裡,分成看得到是誰槓的,與看不到的(加槓)。

        指示牌一翻開畫面就看得到,槓要等下一張捨牌才排得出來,所以要等 ``at`` 之後
        最近的那一次槓。照翻開的時間放的話會排在槓前面,真人那場差了兩次摸牌。
        """
        doras = list(self.log.doras[1:])
        timed: list[tuple[int, str]] = []
        own = sum(m.move is Move.CALL and len(m.tiles) in (1, _COPIES) for m in self.log.moves)
        for index, step in enumerate(self.steps):
            if step.action is not Action.CALL or not (step.kan or step.target is None):
                continue
            own -= step.actor == 0
            near = [dora for dora in doras if dora[0] <= index + _KAN_REACH]
            if near:
                doras.remove(near[-1])
                timed.append((max(near[-1][0], index + 2), near[-1][1]))
        # 自己的槓手牌看得到,不必猜。
        own = max(own, 0)
        return sorted([*timed, *doras[:own]]), doras[own:]

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
            self.events += [Dora(dora_marker=pai) for _, pai in self.loose]
            self._tail()
            self._next_up()
        length = _legal_length(self.events)
        return Stream(
            tuple(self.events[:length]),
            self.stalled or length < len(self.events),
            self.forced,
        )

    def _flush_doras(self, index: int) -> None:
        while self.doras and self.doras[0][0] <= index:
            self.events.append(Dora(dora_marker=self.doras.pop(0)[1]))
        pon = any(isinstance(event, Pon) for event in self.events)
        while self.loose and self.loose[0][0] + (_KAN_REACH if pon else -1) < index:
            self.events.append(Dora(dora_marker=self.loose.pop(0)[1]))

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

    def _next_up(self) -> None:
        """見模組說明「還沒排定的捨牌」。自己的動作走在前面時,輪不到別人。"""
        step = self.log.next_up
        if step is None or self.moves or isinstance(self.events[-1], Tsumo):
            return
        self.events.append(Tsumo(actor=step.actor, pai=UNKNOWN))
        self._dahai(self._updated(step))

    # -------------------------------------------------------------------- 摸打

    def _tsumo(self, step: Step, index: int) -> bool:
        if step.actor != 0:
            self.events.append(Tsumo(actor=step.actor, pai=UNKNOWN))
            self._added_kan(step.actor, index)
            return True
        if self.moves and self.moves[0].move is Move.CALL and len(self.moves[0].tiles) == 2:
            return self._call_for_draw()
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

    def _call_for_draw(self) -> bool:
        """排序器說自己摸牌,手牌說鳴牌:聽手牌的。

        鳴完打得快時副露還沒確認,排序器只看到河裡多一張,排成摸打。步驟排出去就不會改了,不接上
        的話事件流從這裡停到局尾。
        """
        call = self._own_call(self.moves[0])
        if call is None:
            return False
        self.moves.pop(0)
        self.unmatched -= 1
        self.events.append(call)
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
        if step.actor == 0:
            self._own_kans()
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
            pai = self._discarded(step)
            self.sure = step.pai is not None and normalize_red(step.pai) == normalize_red(pai)
            tsumogiri = self._tsumogiri(step, reached=reached)
            self.events.append(Dahai(actor=step.actor, pai=pai, tsumogiri=tsumogiri))
        if reached:
            self.events.append(ReachAccepted(actor=step.actor))
        return True

    def _tsumogiri(self, step: Step, *, reached: bool) -> bool:
        """別家這張是不是摸切。規則推得出來的照規則,其餘看手牌的形狀(:attr:`KyokuLog.tsumogiri`)。

        鳴了之後打的一定是手切;立直之後(宣言牌以外)一定是摸切。都不是又沒量到就填手切,
        與之前一樣。
        """
        last = self.events[-1] if self.events else None
        if isinstance(last, Chi | Pon) and last.actor == step.actor:
            return False
        accepted = any(
            isinstance(event, ReachAccepted) and event.actor == step.actor for event in self.events
        )
        if accepted and not reached:
            return True
        return self.log.tsumogiri.get((step.actor, step.serial), False)

    def _own_kans(self) -> None:
        """手牌看到自己槓了、排序器不知道的:加槓不變長,而打之前副露還沒確認。"""
        while (
            len(self.moves) >= 2
            and self.moves[0].move is Move.CALL
            and len(self.moves[0].tiles) in (1, _COPIES)
            and self.moves[1].move is Move.DRAW
        ):
            call = self._own_call(self.moves.pop(0))
            assert call is not None
            self.early_kans += isinstance(call, Ankan)
            self.events += [call, Tsumo(actor=0, pai=self.moves.pop(0).tiles[0])]

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
            called = bool(self.moves) and self.moves[0].move is Move.CALL
            if step.kan and self.early_kans and not called:
                # 手牌已經排過的暗槓,副露區晚一輪才確認,排序器在下一手補了一次槓。它不是新的
                # 動作;當成摸牌的話會吃掉下一次真的摸牌,整局停住(044833 東二局)。
                self.early_kans -= 1
                return True
            if not (self.moves and self.moves[0].move is Move.CALL):
                # 手牌沒有鳴牌,這一步是排序器推出來的:順序亂掉時,自己打牌看起來像跳家。自己的
                # 動作聽手牌的:手牌接下來是摸牌,這一步就是摸牌。手牌只是還沒跟上的話,下一輪重算
                # 就接得上。
                if self.moves and self.moves[0].move is Move.DRAW:
                    self.unmatched += 1
                    self.events.append(Tsumo(actor=0, pai=self.moves.pop(0).tiles[0]))
                return True
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
        last = self.events[self._last_dahai()]
        assert isinstance(last, Dahai)
        if (
            index >= len(groups)
            and step.actor in self.log.meld_growing
            and normalize_red(last.pai) not in HONOR_ORDER
        ):
            # 排序器能從副露正在變長先推得有人鳴牌,此時快照還沒建立。
            # 不能因為沒有正在執行的比對,就把尚未確認的順子或普通五送出去。
            return False
        if step.kan and last.actor == 0 and known and not _same_tile(known, last.pai):
            # 槓的是自己打的牌,牌面卻是別的牌:不是大明槓,是他摸牌之後暗槓。自己打的牌來自手牌不會錯,
            # 照大明槓排就得改掉它,整局停在這裡(011937 東一局:下家暗槓 9 筒,副露晚一輪才確認,
            # 那時上一張是自己的 7 筒)。暗槓之前的那次摸牌由這裡補,嶺上與打牌照排序器的。
            self.events.append(Tsumo(actor=step.actor, pai=UNKNOWN))
            self.events.append(self._ankan(step.actor, known))
            return True
        step = self._unseen_call(step, known)
        known = self._agreeing(known)
        if step.kan:
            self.events.append(self._open_kan(step, known))
            return True
        look = self.log.meld_looks.get((step.actor, index))
        meld = self._looked(step, look)
        if meld is None:
            if look is not None and look.pending:
                # 背景比對尚未完成不等於棄權。先組一個普通五或猜錯的順子,
                # 下一輪又改掉,會讓引擎收到當下已知仍待確認的錯誤副露。
                return False
            red_suits = self.log.meld_reds.get((step.actor, index), frozenset())
            sideways = (step.actor, index) in self.log.meld_sideways
            meld = self._with_red(self._open_meld(step, known, sideways=sideways), red_suits)
        self.events.append(meld)
        return True

    def _looked(self, step: Step, look: MeldLook | None) -> Chi | Pon | None:
        """畫面比過候選的吃碰。還沒比完、差距不夠、沒什麼好選的,回 ``None``。

        候選是「被鳴的牌、從誰那裡鳴的」推得出來的每一種,張數湊不出來的不列。這比
        :meth:`_open_meld` 從牌面推準:十四份錄影別家 84 組要選的吃碰,這裡錯 0 組、棄權 2 組,牌面
        認得全的只有 43 組(``docs/cv_mjai.md``)。

        被鳴的那張照河裡的,不聽畫面:相鄰數字的牌面只差上面那個字,河裡的牌讀錯時,
        錯的候選常常跟對的一樣像。
        """
        assert step.target is not None
        if look is None:
            return None
        at = self._last_dahai()
        pai = self.events[at].pai  # type: ignore[attr-defined]
        plain = normalize_red(pai)
        if plain in HONOR_ORDER:
            return None
        shapes = []
        if self.left[plain] >= 2:
            side = _PON_SIDEWAYS[(step.target - step.actor) % _SEATS]
            shapes.append(MeldShape((plain,) * 3, side))
        if step.actor == (step.target + 1) % _SEATS:
            for run in _runs_with(plain):
                others = tuple(tile for tile in run if tile != plain)
                if all(self.left[tile] > 0 for tile in others):
                    shapes.append(MeldShape((plain, *others), 0))
        if not shapes or (len(shapes) == 1 and not set(shapes[0].tiles) & set(_FIVES)):
            # 只剩一種、又沒有五要看赤不赤:畫面沒有要回答的。
            return None
        found = look.pick(tuple(shapes))
        if found is None:
            return None
        shape = shapes[found.index]
        consumed = []
        for index, (tile, red) in enumerate(zip(shape.tiles, found.reds, strict=True)):
            if index == shape.sideways:
                continue
            wanted = f"{tile}r"
            if red and tile in _FIVES and not self._red_seen(wanted) and wanted not in consumed:
                consumed.append(self._pick(wanted, tile))
            else:
                consumed.append(self._pick(tile))
        meld = Pon if len(set(shape.tiles)) == 1 else Chi
        return meld(actor=step.actor, target=step.target, pai=pai, consumed=consumed)

    def _agreeing(self, known: list[str]) -> list[str]:
        """牌面與河裡有把握的那張湊不成一組:牌面讀錯了。

        被鳴的牌一定在這一組裡,而橫放的它常常認不到,所以看的是湊不湊得起來,不是有沒有認到它。
        河裡有把握的牌實測 99.7% 是對的,副露的牌面小而且斜,反過來才對:實測 5 索吃進來被牌面讀成
        三張 2 索,整組變成碰,還把河裡那張改掉。
        """
        event = self.events[self._last_dahai()]
        assert isinstance(event, Dahai)
        if not self.sure or event.actor == 0:
            return known
        plain = normalize_red(event.pai)
        tiles = {normalize_red(tile) for tile in known}
        if tiles <= {plain}:
            return known
        if plain in HONOR_ORDER or not any(tiles <= set(run) for run in _runs_with(plain)):
            return []
        return known

    def _added_kan(self, actor: int, index: int) -> None:
        """``actor`` 剛摸牌。有一張沒有主人的新寶牌,而他有碰可以加,就是他加槓。

        加的那張一定是他碰過的牌。碰了兩組以上時分不出是哪一組,挑還有第四張的第一組。
        """
        if not self.loose or self.loose[0][0] > index:
            return
        added = {
            normalize_red(event.pai)
            for event in self.events
            if isinstance(event, Kakan) and event.actor == actor
        }
        pon = next(
            (
                event
                for event in self.events
                if isinstance(event, Pon)
                and event.actor == actor
                and normalize_red(event.pai) not in added
                and self.left[normalize_red(event.pai)] > 0
            ),
            None,
        )
        if pon is None:
            return
        plain = normalize_red(pon.pai)
        self.left[plain] -= 1
        red = f"{plain}r"
        pai = red if plain in _FIVES and not self._red_seen(red) else plain
        self.events += [
            Kakan(actor=actor, pai=pai, consumed=[pon.pai, *pon.consumed]),
            Tsumo(actor=actor, pai=UNKNOWN),
            Dora(dora_marker=self.loose.pop(0)[1]),
        ]

    def _unseen_call(self, step: Step, known: list[str]) -> Step:
        """牌面對不上自己剛打的牌:鳴的是別人打的、一打出來就被鳴走的那張。

        自己打的牌來自手牌,不能被牌面改掉:改了就是打出手上沒有的牌,事件流從那裡斷掉,
        Steam 實測一局 8 手只排到 1 手。河裡沒看到那張牌,所以補上它的摸打。補給最近
        打過牌的別家,那是猜的,但牌張對了,Mortal 看到的局面就對。
        """
        last = self.events[self._last_dahai()]
        assert isinstance(last, Dahai)
        triple = len(known) == 3 and len(set(known)) == 1
        if last.actor != 0 or not triple or normalize_red(known[0]) == normalize_red(last.pai):
            return step
        others = [
            event.actor
            for event in reversed(self.events)
            if isinstance(event, Dahai) and event.actor not in (0, step.actor)
        ]
        target = others[0] if others else (step.actor + 1) % _SEATS or 1
        self.events.append(Tsumo(actor=target, pai=UNKNOWN))
        self.events.append(Dahai(actor=target, pai=self._pick(known[0]), tsumogiri=False))
        return replace(step, target=target)

    def _own_call(self, move: OwnMove) -> MjaiEvent | None:
        """自己的鳴牌。手上拿出去幾張就是哪一種;別人的牌要接在那張牌後面。"""
        tiles = list(move.tiles)
        if len(tiles) == _COPIES:
            return Ankan(actor=0, consumed=sorted(tiles))
        if len(tiles) == 1:
            return Kakan(actor=0, pai=tiles[0], consumed=[normalize_red(tiles[0])] * 3)
        # 鳴的是宣言牌時,它後面還接著 reach_accepted。
        at = len(self.events) - 1 - isinstance(self.events[-1], ReachAccepted)
        last = self.events[at]
        if not isinstance(last, Dahai) or last.actor == 0:
            return None
        if len(tiles) == 3:
            pai = self._called_by_me(at, [normalize_red(tiles[0])])
            return Daiminkan(actor=0, target=last.actor, pai=pai, consumed=tiles)
        if normalize_red(tiles[0]) == normalize_red(tiles[1]):
            pai = self._called_by_me(at, [normalize_red(tiles[0])])
            return Pon(actor=0, target=last.actor, pai=pai, consumed=tiles)
        low, high = sorted(normalize_red(tile) for tile in tiles)
        fitting = [t for run in _runs_with(low) if high in run for t in run if t not in (low, high)]
        pai = self._called_by_me(at, fitting)
        return Chi(actor=0, target=last.actor, pai=pai, consumed=tiles)

    def _called_by_me(self, at: int, fitting: list[str]) -> str:
        """自己鳴的那張牌。手上拿出去的牌來自手牌,比河裡那張可靠,對不上就聽手牌的。"""
        pai = self.events[at].pai  # type: ignore[attr-defined]
        if normalize_red(pai) in fitting:
            return pai
        tile = max(fitting, key=lambda t: self.left[t])
        return self._replace(at, tile) if self.left[tile] > 0 else pai

    def _ankan(self, actor: int, known: list[str]) -> Ankan:
        tile = known[0] if known else None
        if tile is None or self.left[normalize_red(tile)] < _COPIES:
            full = [t for t in _FILLERS if self.left[t] >= _COPIES]
            tile = full[0] if full else _FILLERS[0]
        self.left[normalize_red(tile)] -= _COPIES
        return Ankan(actor=actor, consumed=self._red_in_kan([tile] * _COPIES))

    def _open_kan(self, step: Step, known: list[str]) -> Daiminkan:
        """別家的大明槓。四張一樣,所以認到一張就夠了。"""
        assert step.target is not None
        at = self._last_dahai()
        plain = normalize_red(self.events[at].pai)  # type: ignore[attr-defined]
        tile = known[0] if known else plain
        pai = self._replace(at, tile)
        consumed = self._red_in_kan([self._pick(tile) for _ in range(_COPIES - 1)], pai)
        return Daiminkan(actor=step.actor, target=step.target, pai=pai, consumed=consumed)

    def _open_meld(self, step: Step, known: list[str], *, sideways: bool = False) -> Chi | Pon:
        """別家的吃碰。``pai`` 是剛打出來的牌;牌面說不是的話聽牌面的。

        ``sideways`` 表示唯一認到的那張是橫放的,見 :attr:`KyokuLog.meld_sideways`。
        """
        assert step.target is not None
        at = self._last_dahai()
        pai = self.events[at].pai  # type: ignore[attr-defined]
        plain = normalize_red(pai)
        adjacent = step.actor == (step.target + 1) % _SEATS
        triple = len(known) == 3 and len(set(known)) == 1
        # 只在直的框認到被鳴的那張:吃的直放牌不會是被鳴的牌,所以是碰。只在橫的框認到就是
        # 被鳴的那張本身,分不出來:對家的直放索子常常認不出來,十二場有 6 組吃因此被當成碰,
        # 而從上家鳴數牌是吃 67 次、碰 0 次。場上已經湊不出兩張時也只可能是吃(嵌張的中間那張)。
        lone = (
            len(known) == 1
            and not sideways
            and normalize_red(known[0]) == plain
            and self.left[plain] >= 2
        )
        if (
            triple
            or lone
            or not adjacent
            or plain in HONOR_ORDER
            or (known and known[0] in HONOR_ORDER)
        ):
            tile = known[0] if known else plain
            pai = self._replace(at, tile)
            consumed = [self._pick(tile), self._pick(tile)]
            return Pon(actor=step.actor, target=step.target, pai=pai, consumed=consumed)
        pai, others = self._run(at, plain, known)
        consumed = [self._pick(tile) for tile in others]
        return Chi(actor=step.actor, target=step.target, pai=pai, consumed=consumed)

    def _with_red(self, meld: Chi | Pon, red_suits: frozenset[str]) -> Chi | Pon:
        """畫面看到這組有赤五:把組裡的五換成赤的。

        花色由事件流推:牌面常常認不全,而赤五本身又常常是認不到的那一張。
        """
        suit = meld.pai[1:2]
        five = f"5{suit}"
        red = f"{five}r"
        if suit not in red_suits or self._red_seen(red):
            return meld
        if five in meld.consumed:
            consumed = list(meld.consumed)
            consumed[consumed.index(five)] = red
            return replace(meld, consumed=consumed)
        if meld.pai != five:
            return meld
        at = self._last_dahai()
        event = self.events[at]
        assert isinstance(event, Dahai)
        self.events[at] = replace(event, pai=red)
        return replace(meld, pai=red)

    def _red_in_kan(self, tiles: list[str], called: str = "") -> list[str]:
        """四張五裡一定有一張赤的:被鳴的那張不是,就在這幾張裡。不必看畫面。"""
        plain = normalize_red(tiles[0])
        red = f"{plain}r"
        if plain not in _FIVES or called == red or self._red_seen(red):
            return tiles
        return [red, *tiles[1:]]

    def _red_seen(self, red: str) -> bool:
        """這張赤五已經在事件流裡了嗎。每種只有一張。"""
        return any(
            red in tiles
            for event in self.events
            for tiles in (
                *getattr(event, "tehais", ())[:1],
                getattr(event, "consumed", ()),
                (getattr(event, "pai", ""),),
            )
        )

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


def _without_phantom_turns(steps: list[Step], discards: int) -> list[Step]:
    """拿掉排序器補出來、但自己其實沒有的回合。

    別家的牌一打出來就被鳴走時,河裡少一張,排序器會補隱藏的捨牌來對齊,有時補到自己
    頭上。留著假的回合,手牌的動作就全部晚一輪對不上,Steam 實測因此在第三巡卡死。

    自己河裡沒讀到的回合也長得一樣,但那是真的,拿掉一樣會晚一輪。分得開:手牌看得到
    自己打過幾張(``discards``),比河裡讀到的多幾張,就有幾個隱藏的回合是真的。
    """

    def seen(index: int) -> bool:
        step = steps[index]
        hidden = step.serial is None and step.pai is None and step.guess is None
        after = steps[index + 1] if index + 1 < len(steps) else None
        return not hidden or after is None or after.action is Action.CALL

    own = [
        index
        for index, step in enumerate(steps)
        if step.action is Action.DAHAI and step.actor == 0
    ]
    real = discards - sum(seen(index) for index in own)
    drop: set[int] = set()
    for index in own:
        if seen(index) or index == 0 or steps[index - 1].action is not Action.TSUMO:
            continue
        if real > 0:
            real -= 1
        else:
            drop |= {index - 1, index}
    return [step for index, step in enumerate(steps) if index not in drop]


def _same_tile(known: list[str], pai: str) -> bool:
    """認到的牌面全是 ``pai``(赤五算同一張)。"""
    return {normalize_red(tile) for tile in known} == {normalize_red(pai)}


def _without_phantom_ankans(steps: list[Step], flips: int) -> list[Step]:
    """拿掉別家沒有翻寶牌的暗槓。

    暗槓一定翻一張新寶牌,翻不出來就是副露長度讀錯了:別家的手伸過副露帶,量成多了四張。實測
    一局憑空多了下家暗槓四張北,張數一滿,之後真的北就全變成別的牌。寶牌比副露確認得早,所以真的
    暗槓不會被擋;就算晚一步,下一輪重算也會補回來。
    """
    out: list[Step] = []
    skip = False
    for index, step in enumerate(steps):
        if skip:
            skip = False
            continue
        if step.action is Action.CALL and (step.kan or step.target is None):
            hidden = step.target is None and step.actor != 0
            if hidden and flips <= 0:
                after = steps[index + 1] if index + 1 < len(steps) else None
                skip = after is not None and after.action is Action.TSUMO
                continue
            flips -= 1
        out.append(step)
    return out


def _legal_length(events: list[MjaiEvent]) -> int:
    """前面幾個事件是 libriichi 收得下的。規則見模組說明。"""
    seen: Counter[str] = Counter()
    hand: Counter[str] = Counter()
    drawn = 0
    # 立直之後不能鳴牌。libriichi 遇到這個不是報錯,是整個行程崩潰:實測一次假立直接上真的碰,
    # 引擎當場掉線,後面一局半都沒有建議。
    reached: set[int] = set()

    def witness(tiles: Iterable[str]) -> bool:
        for tile in tiles:
            seen[normalize_red(tile)] += 1
        return all(seen[normalize_red(tile)] <= _COPIES for tile in tiles)

    def spend(tiles: Iterable[str]) -> bool:
        for tile in tiles:
            hand[normalize_red(tile)] -= 1
        return all(hand[normalize_red(tile)] >= 0 for tile in tiles)

    for index, event in enumerate(events):
        drawn += isinstance(event, Tsumo)
        if drawn > _WALL:
            return index
        if isinstance(event, Chi | Pon | Daiminkan | Kakan) and event.actor in reached:
            return index
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
            case Kakan():
                fine = witness([event.pai])
            case ReachAccepted():
                reached.add(event.actor)
                fine = True
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
