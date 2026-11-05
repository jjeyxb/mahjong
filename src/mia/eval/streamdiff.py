"""封包事件流對畫面事件流:一場即時對局,畫面這條路錯在哪、晚多少。

:mod:`mia.eval.rivers` 量「每一幀讀對幾格」,這裡量的是更後面的東西:送進引擎的那串事件。
答案是同一場的封包,兩邊都記了牆上時鐘,所以每張牌都算得出「封包到了之後多久畫面才排到」。

三件事分開報:

* **局末對不對**:每局最後那份事件流,與封包逐項比。
* **當下對不對**:每張牌**第一次**出現時對不對。事件流會改寫,局末是對的不代表當下給的
  建議是對的,所以「改過」要另外算。
* **給的建議一不一樣**:同一個引擎分別看封包與畫面的當下局面,見 :func:`agreement`。

座位一律用視角:自己是 0。畫面這條路本來就是這樣,封包在這裡換算。
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any

from mia.engine.base import AIEngine, EngineError
from mia.groundtruth.stream import MjaiDecoder
from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Hora,
    Kakan,
    MjaiEvent,
    Pon,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
    parse_event,
)

__all__ = [
    "Agreement",
    "Issue",
    "KyokuResult",
    "Run",
    "TruthKyoku",
    "Version",
    "agreement",
    "compare",
    "load_run",
    "load_truth",
    "summary",
]

#: 記錄檔的檔名,放在 session 目錄裡。
VISION_LOG = "vision.jsonl"

_MELDS = (Chi, Pon, Daiminkan, Ankan, Kakan)
_OWN = 0
_SEATS = ("自己", "下家", "對家", "上家")

#: 一局的鍵:風、局、本場。
Key = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class TruthKyoku:
    """封包的一局。``events`` 是 ``(牆上時鐘, 事件)``,座位已換成視角。"""

    key: Key
    events: tuple[tuple[float, MjaiEvent], ...]

    @property
    def stream(self) -> list[MjaiEvent]:
        return [event for _, event in self.events]

    @property
    def start(self) -> float:
        return self.events[0][0]


@dataclass(frozen=True, slots=True)
class Version:
    """畫面事件流的一個版本。``kyoku`` 是 :attr:`TableWorker.kyoku`,換局就加一。"""

    wall: float
    kyoku: int
    events: tuple[MjaiEvent, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class Issue:
    """一個錯誤。

    Attributes:
        kyoku: 哪一局,給人看的。
        kind: 哪一類。
        seat: 哪一家,0 是自己。
        index: 那一家的第幾項,從 0 算。
        expected: 封包的答案。
        got: 畫面給的。
        wall: 封包那一刻。多出來的東西沒有封包,用畫面第一次出現的時間。
    """

    kyoku: str
    kind: str
    seat: int | None = None
    index: int | None = None
    expected: str = ""
    got: str = ""
    wall: float | None = None

    def __str__(self) -> str:
        seat = "" if self.seat is None else f" {_SEATS[self.seat]}"
        index = "" if self.index is None else f" 第{self.index + 1}項"
        return f"{self.kyoku} {self.kind}{seat}{index}:封包 {self.expected} / 畫面 {self.got}"


@dataclass
class KyokuResult:
    """一局的比對結果。"""

    key: Key
    label: str
    versions: int = 0
    rewrites: int = 0
    #: 帶著「對不上」或「等手牌」的秒數。
    noted: float = 0.0
    counts: Counter[str] = field(default_factory=Counter)
    issues: list[Issue] = field(default_factory=list)
    #: 封包到了之後多久排到,秒。``T`` 是自己摸牌,``D`` 是別家打牌。
    lags: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass(frozen=True, slots=True)
class Run:
    """一次實打留下的東西。"""

    versions: tuple[Version, ...]
    advices: tuple[dict[str, Any], ...] = ()
    statuses: tuple[dict[str, Any], ...] = ()


# ------------------------------------------------------------------ 讀檔


def load_truth(dump: Path) -> list[TruthKyoku]:
    """把封包錄影轉成一局一局的事件。座位換成視角。"""
    kyokus: list[TruthKyoku] = []
    current: list[tuple[float, MjaiEvent]] = []
    seat = 0

    def flush() -> None:
        if current:
            start = current[0][1]
            assert isinstance(start, StartKyoku)
            key = (start.bakaze, start.kyoku, start.honba)
            kyokus.append(TruthKyoku(key, tuple(current)))
            current.clear()

    for decoded in MjaiDecoder().decode_file(dump):
        wall = decoded.frame.wall_clock
        for event in decoded.events:
            match event:
                case StartGame():
                    seat = event.id
                case StartKyoku():
                    flush()
                    current.append((wall, _view_start(event, seat)))
                case EndGame() | EndKyoku() | Hora() | Ryukyoku():
                    pass
                case _ if current:
                    current.append((wall, _view(event, seat)))
    flush()
    return kyokus


def _view_start(event: StartKyoku, seat: int) -> StartKyoku:
    def rotate(values: list[Any]) -> list[Any]:
        return [values[(seat + offset) % 4] for offset in range(4)]

    return replace(
        event,
        oya=(event.oya - seat) % 4,
        tehais=rotate(event.tehais),
        scores=rotate(event.scores),
    )


def _view(event: MjaiEvent, seat: int) -> MjaiEvent:
    changes = {
        name: (getattr(event, name) - seat) % 4
        for name in ("actor", "target")
        if getattr(event, name, None) is not None
    }
    return replace(event, **changes)


def load_run(path: Path) -> Run:
    """讀 ``vision.jsonl``,見 ``tools/selfplay.py``。"""
    versions: list[Version] = []
    advices: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            match row["kind"]:
                case "stream":
                    events = tuple(parse_event(event) for event in row["events"])
                    versions.append(
                        Version(row["wall"], row["kyoku"], events, row.get("note", ""))
                    )
                case "advice":
                    advices.append(row)
                case "status":
                    statuses.append(row)
    return Run(tuple(versions), tuple(advices), tuple(statuses))


# ------------------------------------------------------------------ 比對


def compare(truth: Sequence[TruthKyoku], versions: Sequence[Version]) -> list[KyokuResult]:
    """每一局比一次。封包有、畫面沒有的局也會列出來。"""
    groups = _group(truth, versions)
    results = []
    for index, kyoku in enumerate(truth):
        label = _label(kyoku.key)
        result = KyokuResult(kyoku.key, label)
        results.append(result)
        group = groups.get(index, [])
        if not group or not group[-1].events:
            result.issues.append(Issue(label, "整局沒有事件流", wall=kyoku.start))
            continue
        _compare_one(kyoku, group, result)
    return results


def _label(key: Key) -> str:
    return f"{key[0]}{key[1]}局{key[2]}本"


def _group(
    truth: Sequence[TruthKyoku], versions: Sequence[Version]
) -> dict[int, list[Version]]:
    """畫面的每個版本屬於封包的哪一局。

    **靠時間,不靠局名。** 本場數讀錯的話局名就對不上,而那正是要報出來的錯,不能因此整局
    變成「沒有事件流」。一個換局計數落在封包某一局的時間範圍內就算那一局。
    """
    starts = [kyoku.start for kyoku in truth]
    by_counter: dict[int, list[Version]] = defaultdict(list)
    for version in versions:
        if version.events:
            by_counter[version.kyoku].append(version)
    groups: dict[int, list[Version]] = defaultdict(list)
    for group in by_counter.values():
        middle = group[len(group) // 2].wall
        index = next(
            (i for i in range(len(starts) - 1, -1, -1) if starts[i] <= middle), None
        )
        if index is not None:
            groups[index].extend(group)
    return groups


def _compare_one(kyoku: TruthKyoku, group: Sequence[Version], result: KyokuResult) -> None:
    result.versions = len(group)
    result.rewrites = sum(
        1
        for before, after in pairwise(group)
        if after.events[: len(before.events)] != before.events
    )
    result.noted = sum(
        after.wall - before.wall
        for before, after in pairwise(group)
        if before.note
    )
    final = group[-1].events
    first = _first_seen(group)
    _compare_start(kyoku, final, result)
    _compare_discards(kyoku, final, first, result)
    _compare_draws(kyoku, final, first, result)
    _compare_melds(kyoku, final, result)
    _compare_doras(kyoku, final, result)
    _compare_reaches(kyoku, final, result)


def _first_seen(group: Sequence[Version]) -> dict[tuple[Any, ...], tuple[float, str]]:
    """每張牌、每個版本第一次出現的時間與當時的值。

    **錯的值也算第一次**,對的值另外用 ``(*key, 值)`` 記:兩者的差就是「改過」。
    """
    first: dict[tuple[Any, ...], tuple[float, str]] = {}
    for version in group:
        for key, value in _keyed(version.events):
            first.setdefault(key, (version.wall, value))
            first.setdefault((*key, value), (version.wall, value))
    return first


def _keyed(events: Iterable[MjaiEvent]) -> Iterable[tuple[tuple[Any, ...], str]]:
    discards: Counter[int] = Counter()
    draws = 0
    for event in events:
        if isinstance(event, Dahai):
            yield ("D", event.actor, discards[event.actor]), event.pai
            discards[event.actor] += 1
        elif isinstance(event, Tsumo) and event.actor == _OWN:
            yield ("T", draws), event.pai
            draws += 1


def _compare_start(kyoku: TruthKyoku, final: Sequence[MjaiEvent], result: KyokuResult) -> None:
    truth, seen = kyoku.events[0][1], final[0]
    assert isinstance(truth, StartKyoku)
    assert isinstance(seen, StartKyoku)
    for name in ("bakaze", "kyoku", "honba", "kyotaku", "oya", "dora_marker", "scores"):
        expected, got = getattr(truth, name), getattr(seen, name)
        result.counts["開局欄位"] += 1
        if expected == got:
            result.counts["開局欄位對"] += 1
        else:
            result.issues.append(
                Issue(result.label, f"開局 {name}", None, None, str(expected), str(got),
                      kyoku.start)
            )
    result.counts["配牌"] += 1
    # 莊家開局時畫面上就是 14 張,哪一張算第一摸是封包才知道的事,兩邊分得不一樣不算錯。
    if _with_first_draw(truth, kyoku.stream) == _with_first_draw(seen, final):
        result.counts["配牌對"] += 1
    else:
        result.issues.append(
            Issue(result.label, "配牌", _OWN, None, " ".join(sorted(truth.tehais[_OWN])),
                  " ".join(sorted(seen.tehais[_OWN])), kyoku.start)
        )


def _with_first_draw(start: StartKyoku, events: Sequence[MjaiEvent]) -> list[str]:
    tiles = list(start.tehais[_OWN])
    if start.oya == _OWN:
        first = next((e.pai for e in events if isinstance(e, Tsumo) and e.actor == _OWN), None)
        if first is not None:
            tiles.append(first)
    return sorted(tiles)


def _compare_discards(
    kyoku: TruthKyoku,
    final: Sequence[MjaiEvent],
    first: dict[tuple[Any, ...], tuple[float, str]],
    result: KyokuResult,
) -> None:
    for seat in range(4):
        truth = [(wall, e.pai) for wall, e in kyoku.events
                 if isinstance(e, Dahai) and e.actor == seat]
        seen = [e.pai for e in final if isinstance(e, Dahai) and e.actor == seat]
        tiles = [pai for _, pai in truth]
        matcher = SequenceMatcher(a=tiles, b=seen, autojunk=False)
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag == "equal":
                result.counts["捨牌對"] += a1 - a0
                continue
            for offset in range(max(a1 - a0, b1 - b0)):
                expected = tiles[a0 + offset] if a0 + offset < a1 else ""
                got = seen[b0 + offset] if b0 + offset < b1 else ""
                kind = "捨牌錯" if expected and got else "捨牌漏" if expected else "捨牌多"
                result.counts[kind] += 1
                wall = truth[a0 + offset][0] if expected else None
                index = a0 + offset if expected else b0 + offset
                result.issues.append(
                    Issue(result.label, kind, seat, index, expected, got, wall)
                )
        for index, (wall, pai) in enumerate(truth):
            key = ("D", seat, index)
            _timing(result, "D" if seat != _OWN else "O", key, pai, wall, first)


def _compare_draws(
    kyoku: TruthKyoku,
    final: Sequence[MjaiEvent],
    first: dict[tuple[Any, ...], tuple[float, str]],
    result: KyokuResult,
) -> None:
    truth = [(wall, e.pai) for wall, e in kyoku.events
             if isinstance(e, Tsumo) and e.actor == _OWN]
    seen = [e.pai for e in final if isinstance(e, Tsumo) and e.actor == _OWN]
    start = kyoku.events[0][1]
    assert isinstance(start, StartKyoku)
    for index, (wall, pai) in enumerate(truth):
        got = seen[index] if index < len(seen) else ""
        if got == pai or (index == 0 and got and start.oya == _OWN):
            # 莊家第一摸見 _compare_start,對不對是一起算的。
            result.counts["自摸對"] += 1
        else:
            kind = "自摸錯" if got else "自摸漏"
            result.counts[kind] += 1
            result.issues.append(Issue(result.label, kind, _OWN, index, pai, got, wall))
        _timing(result, "T", ("T", index), pai, wall, first)


def _timing(
    result: KyokuResult,
    name: str,
    key: tuple[Any, ...],
    pai: str,
    wall: float,
    first: dict[tuple[Any, ...], tuple[float, str]],
) -> None:
    """這張牌晚多少、第一次出現時對不對。"""
    right = first.get((*key, pai))
    if right is None:
        return
    result.lags[name].append(right[0] - wall)
    if first[key][1] != pai:
        result.counts["當下錯後來改對"] += 1


def _meld_tuple(event: Any) -> tuple[Any, ...]:
    return (type(event).__name__, event.actor, getattr(event, "pai", None),
            tuple(sorted(event.consumed)))


def _compare_melds(kyoku: TruthKyoku, final: Sequence[MjaiEvent], result: KyokuResult) -> None:
    truth = [(wall, _meld_tuple(e)) for wall, e in kyoku.events if isinstance(e, _MELDS)]
    rest = [_meld_tuple(e) for e in final if isinstance(e, _MELDS)]
    missed: list[tuple[float, tuple[Any, ...]]] = []
    for wall, meld in truth:
        result.counts["副露"] += 1
        if meld in rest:
            rest.remove(meld)
            result.counts["副露全對"] += 1
        else:
            missed.append((wall, meld))
    for wall, meld in missed:
        same_kind = next((x for x in rest if x[:2] == meld[:2]), None)
        same_seat = next((x for x in rest if x[1] == meld[1]), None)
        other = same_kind or same_seat
        kind = "副露牌面差" if same_kind else "副露類型錯" if same_seat else "副露漏"
        result.counts[kind] += 1
        if other is not None:
            rest.remove(other)
        result.issues.append(
            Issue(result.label, kind, meld[1], None, _meld_text(meld),
                  _meld_text(other) if other else "", wall)
        )
    for meld in rest:
        result.counts["副露多"] += 1
        result.issues.append(Issue(result.label, "副露多", meld[1], None, "", _meld_text(meld)))


def _meld_text(meld: tuple[Any, ...]) -> str:
    return f"{meld[0]} {meld[2] or ''} {' '.join(meld[3])}".replace("  ", " ")


def _compare_doras(kyoku: TruthKyoku, final: Sequence[MjaiEvent], result: KyokuResult) -> None:
    truth = [(wall, e.dora_marker) for wall, e in kyoku.events if isinstance(e, Dora)]
    seen = [e.dora_marker for e in final if isinstance(e, Dora)]
    for index in range(max(len(truth), len(seen))):
        expected = truth[index][1] if index < len(truth) else ""
        got = seen[index] if index < len(seen) else ""
        result.counts["槓寶牌"] += bool(expected)
        if expected == got:
            result.counts["槓寶牌對"] += 1
        else:
            wall = truth[index][0] if index < len(truth) else None
            result.issues.append(Issue(result.label, "槓寶牌", None, index, expected, got, wall))


def _compare_reaches(kyoku: TruthKyoku, final: Sequence[MjaiEvent], result: KyokuResult) -> None:
    truth = [(wall, e.actor) for wall, e in kyoku.events if isinstance(e, ReachAccepted)]
    seen = [e.actor for e in final if isinstance(e, ReachAccepted)]
    for wall, seat in truth:
        result.counts["立直"] += 1
        if seat in seen:
            seen.remove(seat)
            result.counts["立直對"] += 1
        else:
            result.issues.append(Issue(result.label, "立直漏", seat, None, "立直", "", wall))
    for seat in seen:
        result.issues.append(Issue(result.label, "立直多", seat, None, "", "立直"))


# ------------------------------------------------------------------ 引擎


@dataclass
class Agreement:
    """同一個引擎看封包與看畫面,給的答案一不一樣。"""

    #: 自己摸牌之後的決策。
    turns: int = 0
    turns_seen: int = 0
    turns_same: int = 0
    #: 別家打牌時有得鳴、鳴完要打牌。
    calls: int = 0
    calls_seen: int = 0
    calls_same: int = 0
    #: 引擎對畫面的事件流報錯的次數。真的跑起來那個引擎會掉隊。
    errors: int = 0
    differ: list[str] = field(default_factory=list)


Decision = tuple[Any, ...]


def _decisions(
    engine: AIEngine, events: Sequence[MjaiEvent]
) -> tuple[dict[Decision, tuple[Any, ...]], str | None]:
    """把一局餵給引擎,回傳每個決策點的答案。"""
    engine.react(StartGame(id=_OWN))
    answers: dict[Decision, tuple[Any, ...]] = {}
    draws = 0
    discards: Counter[int] = Counter()
    for event in events:
        if isinstance(event, Tsumo) and event.actor == _OWN:
            draws += 1
        if isinstance(event, Dahai):
            discards[event.actor] += 1
        try:
            advice = engine.react(event)
        except EngineError as exc:
            return answers, f"{event}:{str(exc).splitlines()[0]}"
        if not advice.is_decision:
            continue
        action = advice.action
        answer = ("pass",) if action is None else (action.TYPE, getattr(action, "pai", None))
        if isinstance(event, Tsumo) and event.actor == _OWN:
            answers["T", draws] = answer
        elif isinstance(event, Dahai) and event.actor != _OWN:
            answers["D", event.actor, discards[event.actor]] = answer
        elif isinstance(event, Chi | Pon) and event.actor == _OWN:
            answers["C", draws, discards[_OWN]] = answer
    try:
        engine.react(EndKyoku())
    except EngineError as exc:
        return answers, f"局尾:{str(exc).splitlines()[0]}"
    return answers, None


def _prefixes(
    group: Sequence[Version],
) -> dict[Decision, tuple[MjaiEvent, ...]]:
    """每個決策點**第一次排到**時的事件流,就是當下引擎看到的東西。"""
    prefixes: dict[Decision, tuple[MjaiEvent, ...]] = {}
    for version in group:
        draws = 0
        discards: Counter[int] = Counter()
        for index, event in enumerate(version.events):
            if isinstance(event, Tsumo) and event.actor == _OWN:
                draws += 1
                prefixes.setdefault(("T", draws), version.events[: index + 1])
            elif isinstance(event, Dahai):
                discards[event.actor] += 1
                if event.actor != _OWN:
                    key = ("D", event.actor, discards[event.actor])
                    prefixes.setdefault(key, version.events[: index + 1])
            elif isinstance(event, Chi | Pon) and event.actor == _OWN:
                key = ("C", draws, discards[_OWN])
                prefixes.setdefault(key, version.events[: index + 1])
    return prefixes


def agreement(
    truth: Sequence[TruthKyoku],
    versions: Sequence[Version],
    make_engine: Callable[[], AIEngine],
) -> Agreement:
    """逐局比對引擎的答案。

    Args:
        make_engine: 造一個**已啟動**的引擎。報錯之後引擎的內部狀態不能用,要換一個新的。
    """
    groups = _group(truth, versions)
    result = Agreement()
    engine = make_engine()
    # 鍵用字串:start_kyoku 帶著 list,事件不能雜湊。
    memo: dict[tuple[str, ...], dict[Decision, tuple[Any, ...]]] = {}

    def run(events: tuple[MjaiEvent, ...]) -> dict[Decision, tuple[Any, ...]]:
        nonlocal engine
        key = tuple(str(event) for event in events)
        if key not in memo:
            memo[key], error = _decisions(engine, events)
            if error is not None:
                result.errors += 1
                result.differ.append(f"引擎報錯 {error}")
                engine.close()
                engine = make_engine()
        return memo[key]

    try:
        for index, kyoku in enumerate(truth):
            memo.clear()
            answers = run(tuple(kyoku.stream))
            prefixes = _prefixes(groups.get(index, []))
            label = _label(kyoku.key)
            for key, answer in answers.items():
                own_turn = key[0] == "T"
                result.turns += own_turn
                result.calls += not own_turn
                prefix = prefixes.get(key)
                got = run(prefix).get(key) if prefix is not None else None
                if got is None:
                    continue
                result.turns_seen += own_turn
                result.calls_seen += not own_turn
                if got == answer:
                    result.turns_same += own_turn
                    result.calls_same += not own_turn
                else:
                    result.differ.append(f"{label} {key}:封包 {answer} / 畫面 {got}")
    finally:
        engine.close()
    return result


# ------------------------------------------------------------------ 報告


def _percent(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "0/0"


def _lag_line(name: str, lags: Sequence[float]) -> str:
    if not lags:
        return f"  {name}:沒有樣本"
    ordered = sorted(lags)
    tail = ordered[min(len(ordered) - 1, round(len(ordered) * 0.9))]
    return (
        f"  {name}:中位 {statistics.median(ordered):.2f} 秒,p90 {tail:.2f} 秒,"
        f"最久 {ordered[-1]:.2f} 秒,{len(ordered)} 張"
    )


def summary(
    results: Sequence[KyokuResult],
    agreed: Agreement | None = None,
    run: Run | None = None,
    truth: Sequence[TruthKyoku] = (),
) -> str:
    """給人看的報告。"""
    counts: Counter[str] = Counter()
    lags: dict[str, list[float]] = defaultdict(list)
    for result in results:
        counts.update(result.counts)
        for name, values in result.lags.items():
            lags[name].extend(values)
    discards = sum(counts[k] for k in ("捨牌對", "捨牌錯", "捨牌漏"))
    draws = sum(counts[k] for k in ("自摸對", "自摸錯", "自摸漏"))
    lines = [
        f"局數 {len(results)}",
        f"開局欄位 {_percent(counts['開局欄位對'], counts['開局欄位'])},"
        f"配牌 {_percent(counts['配牌對'], counts['配牌'])}",
        f"捨牌 {_percent(counts['捨牌對'], discards)}"
        f"(錯 {counts['捨牌錯']},漏 {counts['捨牌漏']},多 {counts['捨牌多']})",
        f"自摸 {_percent(counts['自摸對'], draws)}"
        f"(錯 {counts['自摸錯']},漏 {counts['自摸漏']})",
        f"副露 {_percent(counts['副露全對'], counts['副露'])}"
        f"(牌面差 {counts['副露牌面差']},類型錯 {counts['副露類型錯']},"
        f"漏 {counts['副露漏']},多 {counts['副露多']})",
        f"立直 {_percent(counts['立直對'], counts['立直'])},"
        f"槓寶牌 {_percent(counts['槓寶牌對'], counts['槓寶牌'])}",
        f"當下錯後來改對 {counts['當下錯後來改對']} 張",
        "延遲(封包到了之後多久排進事件流):",
        _lag_line("自己摸牌", lags["T"]),
        _lag_line("別家打牌", lags["D"]),
        _lag_line("自己打牌", lags["O"]),
    ]
    if run is not None and truth:
        lines.append(_lag_line("自己摸牌到建議", _advice_lags(run, truth)))
    if agreed is not None:
        lines += [
            f"引擎:自己的回合排到 {_percent(agreed.turns_seen, agreed.turns)},"
            f"同 {_percent(agreed.turns_same, agreed.turns_seen)};"
            f"鳴牌與鳴後排到 {_percent(agreed.calls_seen, agreed.calls)},"
            f"同 {_percent(agreed.calls_same, agreed.calls_seen)};報錯 {agreed.errors} 次",
        ]
    lines.append("")
    lines.append("逐局:")
    for result in results:
        lines.append(
            f"  {result.label}:{result.versions} 版,改寫 {result.rewrites} 次,"
            f"對不上或等手牌 {result.noted:.0f} 秒,問題 {len(result.issues)} 個"
        )
    issues = [issue for result in results for issue in result.issues]
    if issues:
        lines += ["", "問題:", *(f"  {issue}" for issue in issues)]
    if agreed is not None and agreed.differ:
        lines += ["", "引擎答案不同:", *(f"  {text}" for text in agreed.differ)]
    return "\n".join(lines)


def _advice_lags(run: Run, truth: Sequence[TruthKyoku]) -> list[float]:
    """每次自己摸牌之後,多久收到建議。下一次自己動作之前沒收到的不算。"""
    walls = sorted(row["wall"] for row in run.advices if row.get("action") is not None)
    lags = []
    for kyoku in truth:
        for index, (start, event) in enumerate(kyoku.events):
            if not isinstance(event, Tsumo) or event.actor != _OWN:
                continue
            end = next(
                (wall for wall, later in kyoku.events[index + 1 :]
                 if getattr(later, "actor", None) == _OWN),
                kyoku.events[-1][0],
            )
            answer = next((w for w in walls if start <= w < end), None)
            if answer is not None:
                lags.append(answer - start)
    return lags
