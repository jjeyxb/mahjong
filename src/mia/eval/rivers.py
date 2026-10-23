"""量四家牌河的 CV 準確率:拿同時錄下的封包當標準答案。

與 :mod:`~mia.eval.report`(手牌)是平行的兩件事,但量的東西不同:

* 手牌只有自己一副,畫面上正對鏡頭,而且**使用者自己看得到** ——
  認錯了他一眼就發現。
* 牌河有四家,躺在桌上、各轉九十度,而且**沒人會去核對** ——
  放銃分析憑它算危險度,算錯了使用者不會知道。

所以牌河這邊更需要「錯的時候知道自己錯了」,而不是更高的總體準確率。

索引位移是這裡最嚴重的失敗
--------------------------
牌河是有序的:第 k 格對應第 k 張捨牌。少讀一格,後面**整串往前位移**,
那一家就從「錯一張」變成「幾乎全錯」。所以張數先對,才輪得到比牌面 ——
:meth:`RiverReport.count_accuracy` 要跟每張牌的準確率一起看。

標準答案要用「看得見的河」
--------------------------
被別家鳴走的牌已經移到鳴牌者的副露去了,河裡那一格會空出來、後面往前遞補。
拿 :attr:`~mia.mjai.table.Player.river` 當答案的話,每被鳴一次就多一張對不上
的牌,而症狀看起來只是「準確率有點低」 —— 真正的原因與辨識完全無關。
見 :attr:`~mia.mjai.table.Player.visible_river`。
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mia.eval.align import SETTLE, stable_index
from mia.groundtruth.dump import parse_dump
from mia.groundtruth.liqi import LiqiParser
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.to_mjai import MajsoulToMjai
from mia.mjai.table import TableTracker
from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.recorder.session import SessionReader
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import (
    MIN_BRIGHTNESS,
    MIN_MARGIN,
    MIN_SCORE,
    TemplateSet,
)
from mia.vision.tiles.river import (
    CONFIDENT,
    DEFAULT_CORNERS,
    SEATS,
    classify_river_tile,
    face_size,
    read_river,
)

__all__ = [
    "OFF_SCREEN",
    "SWEEP",
    "AlignedRivers",
    "RiverFrameResult",
    "RiverReport",
    "RiverState",
    "RiverTimeline",
    "align_rivers",
    "build_river_timeline",
    "evaluate_rivers",
    "summarize_rivers",
]

#: 這些事件之後牌河就不在畫面上了(和了動畫、結算、下一局發牌)。
#: 與手牌那邊的 ``HAND_OFF_SCREEN`` 是同一個用意:時間軸必須說得出
#: 「現在沒有東西可以評分」,否則那十幾秒會每一幀都被判成 CV 認錯。
OFF_SCREEN = frozenset({"hora", "ryukyoku", "end_kyoku", "end_game"})

#: 門檻掃描用的 ``(最低分數, 最小差距)``。第一組是手牌在用的現行值。
#:
#: 牌河比手牌小、斜、而且沒人會核對,所以它**未必該用同一組門檻**。這個掃描
#: 就是在問:把門檻拉高,換得到多少靜默錯誤的減少,又要付出多少棄權?
SWEEP = (
    (MIN_SCORE, MIN_MARGIN),
    (0.55, 0.08),
    CONFIDENT,
    (0.65, 0.12),
    (0.70, 0.15),
)


@dataclass(frozen=True, slots=True)
class RiverState:
    """某個時刻四家的牌河。

    Attributes:
        wall: 這個狀態成立的牆上時鐘。
        rivers: 依**自己的視角**排好的四家牌河 —— ``own``、``shimocha``、
            ``toimen``、``kamicha``,與 :data:`~mia.vision.tiles.river.SEATS`
            同序。存成視角而不是絕對座位,是因為畫面上的位置跟著視角走,
            而比對的是畫面。
        boundary: 這不是一個可以評分的狀態,是「從這裡開始畫面上沒有牌河」
            的界標。見 :data:`OFF_SCREEN`。
    """

    wall: float
    rivers: tuple[tuple[str, ...], ...]
    boundary: bool = False


@dataclass(slots=True)
class RiverTimeline:
    """一整場的牌河變化,依時間排序。"""

    states: list[RiverState] = field(default_factory=list)
    seat: int | None = None
    _walls: list[float] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def span(self) -> tuple[float, float]:
        if not self.states:
            return (0.0, 0.0)
        return (self.states[0].wall, self.states[-1].wall)

    def at(self, wall: float) -> RiverState | None:
        index = bisect.bisect_right(self._walls, wall) - 1
        return self.states[index] if index >= 0 else None

    def stable_at(self, wall: float, settle: float = SETTLE) -> RiverState | None:
        """這個時刻的牌河,但只在畫面「應該已經演完」時才回答。"""
        index = stable_index(self._walls, wall, settle)
        if index is None:
            return None
        state = self.states[index]
        return None if state.boundary else state

    def freeze(self) -> RiverTimeline:
        self.states.sort(key=lambda s: s.wall)
        self._walls = [s.wall for s in self.states]
        return self


def build_river_timeline(dump: Path | str) -> RiverTimeline:
    """從封包錄影建出牌河的時間軸。

    與 :func:`~mia.eval.align.build_timeline` 吃同一份檔案,但追的是
    :class:`~mia.mjai.table.TableTracker` 而不是手牌。
    """
    schema = LiqiSchema.load()
    converters: dict[str, MajsoulToMjai] = {}
    tracker = TableTracker()
    timeline = RiverTimeline()
    seen_wall = False

    for frame, message in parse_dump(Path(dump), schema):
        seen_wall = seen_wall or frame.wall_clock > 0
        converter = converters.get(frame.flow)
        if converter is None:
            converter = converters[frame.flow] = MajsoulToMjai(LiqiParser(schema))

        for event in converter.handle(message):
            tracker.handle(event)
            if event.TYPE in OFF_SCREEN:
                timeline.states.append(RiverState(frame.wall_clock, (), boundary=True))
                continue
            if tracker.seat is None:
                continue
            rivers = tuple(
                tuple(tracker.players[(tracker.seat + offset) % 4].visible_river)
                for offset in range(4)
            )
            if timeline.states and timeline.states[-1].rivers == rivers:
                continue  # 牌河沒變的事件不必記一個狀態
            timeline.states.append(RiverState(frame.wall_clock, rivers))

    if timeline.states and not seen_wall:
        raise ValueError(f"{dump} 沒有 wall 欄位(舊格式),無法與畫面對齊。")

    timeline.seat = tracker.seat
    return timeline.freeze()


@dataclass(frozen=True, slots=True)
class AlignedRivers:
    """一幀畫面配上它的標準答案。"""

    index: int
    wall: float
    truth: RiverState


def align_rivers(
    session: SessionReader, timeline: RiverTimeline, *, settle: float = SETTLE
) -> Iterator[AlignedRivers]:
    """把錄下的畫面與牌河時間軸配對。轉場期間的幀直接跳過。"""
    for record in session.manifest.frames:
        wall = session.wall_at(record.timestamp)
        state = timeline.stable_at(wall, settle)
        if state is not None:
            yield AlignedRivers(index=record.index, wall=wall, truth=state)


@dataclass(frozen=True, slots=True)
class RiverFrameResult:
    """一幀、一家的比對結果。

    Attributes:
        index: 幀序號。
        seat: 視角座位名(:data:`~mia.vision.tiles.river.SEATS` 之一)。
        truth: 標準答案,依序。
        predicted: CV 讀出來的,依序。
        scores: 每一張的相關係數。與 :attr:`predicted` 等長。
        margins: 每一張與次高分的差距。
        brightness: 每一張的比對區與模板的亮度比。
        contiguous: 有牌的格子是不是從頭連續 —— 不連續代表攤平歪了。

    Note:
        存的是**分數本身**而不是「有沒有信心」的布林值。門檻是可以調的,
        而調門檻正是這份報告要回答的問題 —— 存成布林值就只能看到現行門檻下
        的樣子,想問「拉高會怎樣」得重跑一次完整辨識。

        手牌那邊踩過一次:``Match`` 加了亮度檢查、誤判也確實被擋下了,但統計
        那層沒有這個欄位,於是報告重算信心時完全不知道有這回事,數字一個字
        都沒變,看起來像修了個沒用的東西。
    """

    index: int
    seat: str
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    scores: tuple[float, ...]
    margins: tuple[float, ...]
    brightness: tuple[float, ...]
    contiguous: bool

    def confident_at(
        self, index: int, min_score: float = CONFIDENT[0], min_margin: float = CONFIDENT[1]
    ) -> bool:
        """第 ``index`` 張在指定門檻下算不算有信心。

        亮度**不隨掃描而變** —— 分數與差距問的是「像不像」,可以在準確率與
        棄權之間換;亮度問的是「這裡到底是不是一張牌」,那沒有什麼好換的。
        """
        return (
            self.scores[index] >= min_score
            and self.margins[index] >= min_margin
            and self.brightness[index] >= MIN_BRIGHTNESS
        )

    @property
    def count_ok(self) -> bool:
        """張數對不對。不對的話後面整串索引位移,比對沒有意義。"""
        return len(self.truth) == len(self.predicted)

    @property
    def accepted(self) -> bool:
        """這一家這一幀,即時路徑會不會採用。

        兩個條件都是**不需要標準答案**就能判的,所以它們在實際執行時也一樣
        攔得住 —— 這正是把它們當成採用條件而不是扣分項的理由:

        * :attr:`contiguous` —— 牌河中間有洞,代表攤平歪了或畫面上根本不是
          牌桌(實測素材的開頭有一段選單畫面,它就是被這一條擋下來的)。
        * :attr:`count_ok` —— 張數不對,後面整串索引已經位移。

        ⚠ 量準確率時要**兩個數字一起看**:採用率低而準確率高,代表系統只是
        很會挑容易的幀,那不等於它可靠。
        """
        return self.contiguous and self.count_ok

    @property
    def correct(self) -> int:
        """逐格對上幾張。**逐格**比對是刻意的 —— 牌河是有序的,位置也要對。"""
        return sum(a == b for a, b in zip(self.truth, self.predicted, strict=False))

    def silent_errors(
        self, min_score: float = CONFIDENT[0], min_margin: float = CONFIDENT[1]
    ) -> int:
        """有信心卻認錯。自檢攔不住的那一格。"""
        return sum(
            1
            for i, (a, b) in enumerate(zip(self.truth, self.predicted, strict=False))
            if a != b and self.confident_at(i, min_score, min_margin)
        )

    def abstained(self, min_score: float = CONFIDENT[0], min_margin: float = CONFIDENT[1]) -> int:
        """其實是對的,卻被判為沒信心 —— 棄權的代價。"""
        return sum(
            1
            for i, (a, b) in enumerate(zip(self.truth, self.predicted, strict=False))
            if a == b and not self.confident_at(i, min_score, min_margin)
        )


def evaluate_rivers(
    image: np.ndarray,
    frame: AlignedRivers,
    templates: TemplateSet,
    table: Rect,
) -> list[RiverFrameResult]:
    """對一幀跑四家的牌河辨識,與標準答案比對。

    Args:
        image: **整張**擷取影像 —— 攤平是從整張畫面取四邊形,不是從 ROI。
        frame: 配好的幀。
        templates: 牌面模板集。
        table: 牌桌矩形。
    """
    size = face_size()
    results = []
    for position, seat in enumerate(SEATS):
        flat = DEFAULT_CORNERS[seat].rectify(image, table)
        river = read_river(flat)
        matches = [
            classify_river_tile(flat[river.cells[i].as_slice()], templates, size=size)
            for i in river.filled
        ]
        results.append(
            RiverFrameResult(
                index=frame.index,
                seat=seat,
                truth=frame.truth.rivers[position],
                predicted=tuple(normalize_red(ms_to_mjai(m.label)) for m in matches),
                scores=tuple(m.score for m in matches),
                margins=tuple(m.margin for m in matches),
                brightness=tuple(m.brightness for m in matches),
                contiguous=river.is_plausible(),
            )
        )
    return results


@dataclass(slots=True)
class RiverReport:
    """整份報告。每一家分開看 —— 四家的難度差很多。"""

    results: list[RiverFrameResult] = field(default_factory=list)

    def _for(self, seat: str | None) -> list[RiverFrameResult]:
        return [r for r in self.results if seat is None or r.seat == seat]

    def _accepted(self, seat: str | None) -> list[RiverFrameResult]:
        return [r for r in self._for(seat) if r.accepted]

    def accept_rate(self, seat: str | None = None) -> float:
        """有多少幀通得過 :attr:`RiverFrameResult.accepted`。"""
        rows = self._for(seat)
        return _ratio(len(self._accepted(seat)), len(rows))

    def count_accuracy(self, seat: str | None = None) -> float:
        rows = self._for(seat)
        return _ratio(sum(r.count_ok for r in rows), len(rows))

    def tile_accuracy(self, seat: str | None = None) -> float:
        """逐格對上的比例,**只算採用的幀**。見 :attr:`RiverFrameResult.accepted`。"""
        rows = self._accepted(seat)
        return _ratio(sum(r.correct for r in rows), sum(len(r.truth) for r in rows))

    def contiguous_rate(self, seat: str | None = None) -> float:
        rows = self._for(seat)
        return _ratio(sum(r.contiguous for r in rows), len(rows))

    def silent_error_rate(
        self,
        seat: str | None = None,
        min_score: float = CONFIDENT[0],
        min_margin: float = CONFIDENT[1],
    ) -> float:
        """高信心卻認錯,佔全部讀出來的牌的比例。

        只算**張數正確**的幀:張數不對時整串索引已經位移,那時候的「錯」
        與辨識無關,算進去會把這個數字灌爆,而它正是用來決定「要不要相信
        牌河」的。
        """
        rows = self._accepted(seat)
        total = sum(len(r.predicted) for r in rows)
        return _ratio(sum(r.silent_errors(min_score, min_margin) for r in rows), total)

    def abstain_rate(
        self,
        seat: str | None = None,
        min_score: float = CONFIDENT[0],
        min_margin: float = CONFIDENT[1],
    ) -> float:
        """其實是對的卻被判為沒信心,佔全部讀出來的牌的比例。

        與 :meth:`silent_error_rate` 是同一個門檻的兩端。單看任一邊都會得到
        錯的結論:門檻拉到極高能讓靜默錯誤歸零,代價是什麼都不敢講。
        """
        rows = self._accepted(seat)
        total = sum(len(r.predicted) for r in rows)
        return _ratio(sum(r.abstained(min_score, min_margin) for r in rows), total)

    def confusions(self, seat: str | None = None) -> Counter[tuple[str, str]]:
        """``(標準答案, 認成什麼)`` → 次數。只在張數正確的幀上統計。"""
        table: Counter[tuple[str, str]] = Counter()
        for row in self._accepted(seat):
            for truth, got in zip(row.truth, row.predicted, strict=False):
                if truth != got:
                    table[(truth, got)] += 1
        return table

    def summary(self) -> str:
        if not self.results:
            return "沒有任何可評分的幀 —— 轉場期間的幀不算數。"
        lines = [
            f"可評分的幀 {len(self.results) // len(SEATS)}",
            "",
            "採用率 = 連續且張數正確。準確率只算採用的幀 —— 兩個要一起看,",
            "採用率低而準確率高只代表它很會挑容易的幀。",
            "",
            f"{'家':<10}{'採用率':>10}{'連續':>10}{'張數正確':>10}{'每張牌':>10}{'靜默錯誤':>10}",
        ]
        for seat in (*SEATS, None):
            name = seat if seat else "合計"
            lines.append(
                f"{name:<10}{self.accept_rate(seat):>10.1%}"
                f"{self.contiguous_rate(seat):>10.1%}"
                f"{self.count_accuracy(seat):>10.1%}"
                f"{self.tile_accuracy(seat):>10.1%}"
                f"{self.silent_error_rate(seat):>10.2%}"
            )
        lines += ["", "門檻掃描(分數/差距 → 靜默錯誤 / 棄權成本):"]
        for min_score, min_margin in SWEEP:
            lines.append(
                f"  {min_score:.2f} / {min_margin:.2f}"
                f"  {self.silent_error_rate(None, min_score, min_margin):>8.2%}"
                f"  {self.abstain_rate(None, min_score, min_margin):>8.2%}"
            )
        worst = self.confusions().most_common(8)
        if worst:
            lines += ["", "  最常認錯的:"]
            lines += [f"    {t} → {g}  ×{n}" for (t, g), n in worst]
        return "\n".join(lines)


def summarize_rivers(results: Sequence[RiverFrameResult]) -> RiverReport:
    return RiverReport(results=list(results))


def _ratio(part: int, total: int) -> float:
    return part / total if total else 0.0
