"""混合辨識:樣板與 YOLO 各自組出完整的事件流,在事件層合併成一條。

兩邊錯的地方幾乎不重疊。七份錄影(2026-11-08):

* 樣板讀自己又快又穩(自己的回合 705/705 排得到,引擎 0 次報錯),弱點是別家:偶爾把一張牌讀錯、
  多讀一張,或讀出假的吃碰,那一局之後的建議全跟著錯。
* YOLO 讀別家幾乎不錯(別家捨牌 0 錯 0 多),弱點是自己:手牌或摸牌偶爾讀錯,引擎拒收,
  整個回合沒有建議;也比樣板慢,偶爾整局卡住十幾秒。

所以以樣板為主幹,YOLO 追上之後拿來核對別家:

1. 兩條事件流逐事件對齊(:func:`difflib.SequenceMatcher`)。一致的段落照用,只有別家的摸切
   與否換成 YOLO 的(:func:`_yolo_cuts`)。對齊之前,立直成立先排回宣言牌後面(:func:`canonical`)。
2. 夾在兩個一致點之間、不一致的一段,整段挑一邊。兩邊各自都合乎輪次,一致點上局面相同,
   整段換掉仍然合法。

   * 有自己動作的段落用樣板:自己的手牌樣板讀得準,YOLO 讀錯會讓引擎拒收。別家鳴自己打的牌不算,
     那張牌自己早就打出去了。
   * 別家的段落用 YOLO,除非 YOLO 在那一段落後太多(:data:`LAG_OK`):那是 YOLO 卡住、整局錯位。
   * 樣板有吃碰、YOLO 整段什麼都沒有,用樣板。被立刻鳴走的牌只在河裡停一下,YOLO 要牌停穩才收,
     捨牌、鳴牌、鳴後打的那張整段都不在;樣板的假吃碰則是 YOLO 在同一個位置看到那家照常摸打。
     立直成立也一樣:立直棒放上桌之前 YOLO 不送。樣板有槓、YOLO 那段沒有槓、整局那家的槓也比樣板少
     (:func:`_missed_kans`),也用樣板。

3. 最後一個一致點之後用樣板:YOLO 還沒追上,不等它,建議的速度由樣板決定。
4. 拼起來會被引擎拒收的(:func:`plausible`),一段一段換,換了會拒收的那段退回樣板。

離線用七份錄影的兩條事件流驗證,自己的回合建議與封包相同:樣板 697、YOLO 688、合併 702 / 705;
出牌那一刻看得到的建議:686、645、692。YOLO 再慢 1 秒,合併仍有 701 / 690。
"""

from __future__ import annotations

import difflib
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
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
from mia.mjai.tiles import normalize_red
from mia.tracker import Stream

_OWN = 0
_CALLS = (Chi, Pon, Daiminkan)
_KANS = (Ankan, Kakan, Daiminkan)

#: 一致點上 YOLO 比樣板晚超過這麼多秒,中間那段不信 YOLO。
#:
#: 平常 YOLO 晚 0.2 ~ 0.6 秒(離線,沒算推論時間)。卡住的那兩局晚 6 ~ 21 秒,
#: 而且整家錯位一格,牌面全錯。
LAG_OK = 2.5

Token = tuple[object, ...]


def _token(event: MjaiEvent) -> Token:
    """對齊用的代號。摸切與否兩邊常不一致而且不影響局面,不放進來。"""
    match event:
        case StartKyoku():
            return ("start", event.bakaze, event.kyoku, event.honba)
        case Tsumo():
            return ("tsumo", event.actor, event.pai if event.actor == _OWN else "?")
        case Dahai():
            return ("dahai", event.actor, event.pai)
        case Chi() | Pon() | Daiminkan():
            return (event.TYPE, event.actor, event.target, event.pai, tuple(sorted(event.consumed)))
        case Ankan():
            return ("ankan", event.actor, tuple(sorted(event.consumed)))
        case Kakan():
            return ("kakan", event.actor, event.pai)
        case Reach() | ReachAccepted():
            return (event.TYPE, event.actor)
        case Dora():
            return ("dora", event.dora_marker)
        case _:
            return (event.TYPE,)


def tokens(events: Sequence[MjaiEvent]) -> tuple[Token, ...]:
    """整條事件流的代號。

    別家摸牌看不到牌面,單獨比的話每一個都長得一樣,錯位的兩段也會到處「對上」。
    綁上它後面那個事件(通常就是摸完打的那張)。

    翻寶牌綁上它前面那個事件。實測 YOLO 漏了一個暗槓、只讀到翻寶牌,兩邊同一張翻寶牌被當成
    一致點,把槓切成前後兩段,兩段都換成 YOLO,暗槓就不見了。
    """
    out = []
    for index, event in enumerate(events):
        token = _token(event)
        if isinstance(event, Tsumo) and event.actor != _OWN:
            after = events[index + 1] if index + 1 < len(events) else None
            token = (*token, _token(after) if after is not None else None)
        elif isinstance(event, Dora):
            token = (*token, _token(events[index - 1]) if index else None)
        out.append(token)
    return tuple(out)


def _own(event: MjaiEvent) -> bool:
    return getattr(event, "actor", None) == _OWN


def canonical(events: Sequence[MjaiEvent]) -> tuple[MjaiEvent, ...]:
    """立直成立排回宣言牌的正後面;宣言牌還沒讀到的,先拿掉。

    封包裡一定是「立直、宣言牌、立直成立」,宣言牌被吃碰時也一樣。YOLO 有時把立直成立排在
    下一家摸牌之後,也有時立直棒先讀到、宣言牌還沒有。兩種都會讓兩邊對不齊:前者錯開的那一小段
    整段換成 YOLO 的順序;後者更糟,立直成立被當成一致點,樣板讀到的宣言牌夾在中間被刪掉。
    """
    out: list[MjaiEvent] = []
    for event in events:
        if not isinstance(event, ReachAccepted):
            out.append(event)
            continue
        # 只找同一家立直之後的那張,不然會插到他上一巡的捨牌後面,改寫整段歷史。
        for n in range(len(out) - 1, -1, -1):
            earlier = out[n]
            if isinstance(earlier, Reach) and earlier.actor == event.actor:
                break
            if isinstance(earlier, Dahai) and earlier.actor == event.actor:
                out.insert(n + 1, event)
                break
    return tuple(out)


@dataclass
class Side:
    """一個後端目前的事件流,與每個事件(連同它前面的一切)第一次出現的時間。"""

    events: tuple[MjaiEvent, ...] = ()
    tokens: tuple[Token, ...] = ()
    seen: list[float] = field(default_factory=list)

    def update(self, events: Sequence[MjaiEvent], wall: float) -> None:
        events = canonical(events)
        new = tokens(events)
        keep = 0
        for old, now in zip(self.tokens, new, strict=False):
            if old != now:
                break
            keep += 1
        self.seen = self.seen[:keep] + [wall] * (len(new) - keep)
        self.events, self.tokens = tuple(events), new


def merge(
    template: Side, yolo: Side, *, lag_ok: float = LAG_OK, stats: Counter[str] | None = None
) -> tuple[MjaiEvent, ...]:
    """見模組說明。``template`` 有事件流時才呼叫。"""
    if not yolo.events or template.tokens[0] != yolo.tokens[0]:
        return template.events
    t, y = template, yolo
    ruled = _ruled_cuts(t.events)
    missed = _missed_kans(t.events, y.events)
    # 每一段:(樣板那段, YOLO 那段;不換的話是 None)。
    pieces: list[tuple[Sequence[MjaiEvent], Sequence[MjaiEvent] | None]] = []
    matcher = difflib.SequenceMatcher(None, t.tokens, y.tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pieces.append((_yolo_cuts(t.events, y.events, i1, i2, j1, ruled, stats), None))
            continue
        if i2 == len(t.tokens) or j2 == len(y.tokens):
            pieces.append((t.events[i1:i2], None))
            continue
        mine, theirs = t.events[i1:i2], y.events[j1:j2]
        own = any(_own(event) for event in mine + theirs)
        flash = not theirs and any(isinstance(event, _CALLS) for event in mine)
        # 槓也是:YOLO 讀副露區比較會漏,樣板要讀出假的槓得同時看到四張牌與翻寶牌。
        flash = flash or (
            any(isinstance(event, _KANS) and event.actor in missed for event in mine)
            and not any(isinstance(event, _KANS) for event in theirs)
        )
        # 立直棒放上桌之前 YOLO 不送立直成立,樣板已經有了:YOLO 是還沒看到,不是看到沒有。
        flash = flash or (not theirs and all(isinstance(event, ReachAccepted) for event in mine))
        late = y.seen[j2] - t.seen[i2] > lag_ok
        if own or flash or late:
            pieces.append((mine, None))
            reason = "own" if own else "flash" if flash else "late"
        else:
            pieces.append((mine, theirs))
            reason = "yolo"
        if stats is not None:
            stats[reason] += 1
    merged = _assemble(pieces, range(len(pieces)))
    if plausible(merged):
        return merged
    # 拼起來不合理:一段一段換,換了會不合理的那段退回樣板。實測 YOLO 把上家一張 8p 讀成 E,
    # E 變 5 張;整條退回的話,同一局 YOLO 讀對、樣板讀錯的另一張也跟著丟了。
    chosen: list[int] = []
    for index, (_, correction) in enumerate(pieces):
        if correction is None:
            continue
        if plausible(_assemble(pieces, [*chosen, index])):
            chosen.append(index)
        elif stats is not None:
            stats["implausible"] += 1
    return _assemble(pieces, chosen)


def _missed_kans(template: Sequence[MjaiEvent], yolo: Sequence[MjaiEvent]) -> set[int]:
    """YOLO 整局讀到的槓比樣板少的那幾家。

    只看那一段沒有槓不夠:錄影 053001 西二局三本場,樣板把上家的暗槓晚讀一巡,YOLO 在對的位置
    讀到,兩邊的槓落在不同段,兩個都收進來就多一個槓。
    """
    def count(events: Sequence[MjaiEvent]) -> Counter[int]:
        return Counter(event.actor for event in events if isinstance(event, _KANS))

    mine, theirs = count(template), count(yolo)
    return {actor for actor, kans in mine.items() if kans > theirs[actor]}


def _ruled_cuts(events: Sequence[MjaiEvent]) -> set[int]:
    """規則就決定了摸切與否的別家捨牌:鳴牌後打的一定手切,立直成立之後一定摸切。"""
    out = set()
    reached: set[int] = set()
    for index, event in enumerate(events):
        if isinstance(event, ReachAccepted):
            reached.add(event.actor)
        elif isinstance(event, Dahai) and event.actor != _OWN:
            before = events[index - 1] if index else None
            called = isinstance(before, Chi | Pon) and before.actor == event.actor
            if called or event.actor in reached:
                out.add(index)
    return out


def _yolo_cuts(
    template: Sequence[MjaiEvent],
    yolo: Sequence[MjaiEvent],
    i1: int,
    i2: int,
    j1: int,
    ruled: set[int],
    stats: Counter[str] | None,
) -> Sequence[MjaiEvent]:
    """一致的一段,別家捨牌的摸切與否換成 YOLO 的。

    樣板看別家手牌的形狀估(:mod:`mia.tracker.tedashi`),YOLO 看摸進來的那張還在不在。八份
    錄影、別家 2200 張自己決定的捨牌:樣板 92 ~ 95%,YOLO 98 ~ 99%。規則決定的(:func:`_ruled_cuts`)
    樣板全對,不換。
    """
    out = list(template[i1:i2])
    for offset, event in enumerate(out):
        other = yolo[j1 + offset]
        if (
            isinstance(event, Dahai)
            and isinstance(other, Dahai)
            and event.actor != _OWN
            and i1 + offset not in ruled
            and event.tsumogiri != other.tsumogiri
        ):
            out[offset] = replace(event, tsumogiri=other.tsumogiri)
            if stats is not None:
                stats["cut"] += 1
    return out


def _assemble(
    pieces: Sequence[tuple[Sequence[MjaiEvent], Sequence[MjaiEvent] | None]],
    yolo: Sequence[int] | range,
) -> tuple[MjaiEvent, ...]:
    """把各段接起來,``yolo`` 裡的那幾段用 YOLO 的。"""
    use = set(yolo)
    out: list[MjaiEvent] = []
    for index, (mine, theirs) in enumerate(pieces):
        out.extend(theirs if index in use and theirs is not None else mine)
    return tuple(out)


def plausible(events: Sequence[MjaiEvent]) -> bool:
    """拼起來的事件流引擎收不收:暗槓之後、下一張捨牌之前翻寶牌,同一種牌看得見的不超過 4 張。

    實測拼接過一次「上家暗槓 P」接樣板的嶺上摸牌與捨牌,YOLO 那時還沒送出翻寶牌,引擎拒收。
    翻寶牌排在嶺上摸牌前後都可以:樣板排在後面,YOLO 排在前面。
    """
    seen: Counter[str] = Counter()
    kan = False
    for event in events:
        match event:
            case StartKyoku():
                seen.update(normalize_red(tile) for tile in event.tehais[_OWN])
                seen[normalize_red(event.dora_marker)] += 1
            case Ankan():
                kan = True
                if event.actor != _OWN:
                    seen.update(normalize_red(tile) for tile in event.consumed)
            case Dahai() if kan:
                return False
            case Tsumo() if event.actor == _OWN:
                seen[normalize_red(event.pai)] += 1
            case Dahai() if event.actor != _OWN:
                seen[normalize_red(event.pai)] += 1
            case Chi() | Pon() | Daiminkan() if event.actor != _OWN:
                seen.update(normalize_red(tile) for tile in event.consumed)
            case Dora():
                kan = False
                seen[normalize_red(event.dora_marker)] += 1
    return all(count <= 4 for tile, count in seen.items() if tile != "?")


class StreamMerger:
    """兩個後端的事件流進來,合併後的事件流出去。兩條執行緒各自呼叫 :meth:`offer`。

    Args:
        publish: 合併結果變了就呼叫 ``publish(第幾局, 事件流)``。局數是這裡自己數的:
            兩個後端的局數各數各的,而建議那一端靠局數變了才收掉上一局。
        clock: 測試用。
    """

    TEMPLATE = "template"
    YOLO = "yolo"

    def __init__(
        self,
        publish: Callable[[int, Stream], None],
        *,
        lag_ok: float = LAG_OK,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._lag_ok = lag_ok
        self._clock = clock
        self._lock = threading.Lock()
        self._sides = {self.TEMPLATE: Side(), self.YOLO: Side()}
        self._streams = {self.TEMPLATE: Stream(), self.YOLO: Stream()}
        self._kyoku: dict[str, int | None] = {self.TEMPLATE: None, self.YOLO: None}
        #: 樣板上一局的開局代號。樣板換局後還沒讀到開局資訊時,YOLO 還停在上一局的話不能拿來用。
        self._template_before: Token | None = None
        self._out: Stream | None = None
        self._out_start: Token | None = None
        self._out_kyoku = 0
        self.stats: Counter[str] = Counter()

    def offer(self, source: str, kyoku: int, stream: Stream) -> None:
        with self._lock:
            wall = self._clock()
            if kyoku != self._kyoku[source]:
                if source == self.TEMPLATE and self._sides[source].tokens:
                    self._template_before = self._sides[source].tokens[0]
                self._sides[source] = Side()
                self._kyoku[source] = kyoku
            self._sides[source].update(stream.events, wall)
            self._streams[source] = stream
            events, base = self._merged()
            merged = Stream(
                events=events, stalled=base.stalled, forced=base.forced, doubt=base.doubt
            )
            if merged == self._out:
                return
            start = tokens(events[:1])[0] if events else None
            if events and start != self._out_start:
                self._out_kyoku += 1
                self._out_start = start
            self._out = merged
            kyoku_out = self._out_kyoku
        self._publish(kyoku_out, merged)

    def _merged(self) -> tuple[tuple[MjaiEvent, ...], Stream]:
        template, yolo = self._sides[self.TEMPLATE], self._sides[self.YOLO]
        if template.events:
            events = merge(template, yolo, lag_ok=self._lag_ok, stats=self.stats)
            return events, self._streams[self.TEMPLATE]
        if yolo.events and yolo.tokens[0] != self._template_before:
            return yolo.events, self._streams[self.YOLO]
        return (), self._streams[self.TEMPLATE]
