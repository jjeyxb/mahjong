"""把錄影當成畫面來源,餵進真正的 :class:`~mia.live.table.TableWorker`。

與 :mod:`~mia.eval.rivers` 量的是**不同的東西**,這一點是整個模組存在的理由:

* ``eval.rivers`` 離線逐幀叫 :func:`~mia.vision.tiles.river.read_river`,
  量的是「**讀圖的函式**對不對」。
* 這裡跑的是 app 裡真正那條迴圈,量的是「**跑起來**對不對」。

兩者之間差了四件離線評測看不到的事:

挑哪一幀
    離線評測自己挑「動畫已經演完」的幀(:func:`~mia.eval.rivers.align_rivers`
    會跳過轉場)。實際跑的時候沒人幫它挑 —— 它一秒醒一次,拿到的就是當下
    最新的那一張,可能正好是翻牌動畫的中途,而且**不會重試**。

增量讀取會累積
    ``_read_seat`` 只認新增的那幾格,舊的沿用快取。所以某一輪認錯一格,
    那個錯會**留在狀態裡**直到被鳴走或換局。離線評測每一幀從頭認,
    結構上量不到這種累積。

棄權會讓事件流變短而不是變歪
    認不出來的格子不會送出 ``Dahai``(見 ``TableWorker._read_seat``),
    所以 :class:`~mia.mjai.table.TableTracker` 的河比畫面上的**短** ——
    位置對不上,但成員還是對的。放銃分析只問「這張是不是現物」,
    問的是成員。所以這裡格線比位置、tracker 比集合,刻意用兩把尺。

來不來得及
    1 Hz 的迴圈要在一秒內讀完四家。離線評測開多行程衝吞吐量,
    量不出單一輪的延遲。

Note:
    **解碼不算在 tick 的時間裡。** 實際跑的時候解碼是擷取那條路的工作,
    畫面已經是 ``np.ndarray`` 才丟進來 —— 這裡照著做,不然量到的是
    JPEG 解碼而不是辨識。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median

from mia.config.models import RoiConfig
from mia.eval.align import SETTLE
from mia.eval.rivers import RiverState, RiverTimeline
from mia.live.bus import UpdateBus
from mia.live.table import TableWorker
from mia.mjai.tiles import normalize_red
from mia.recorder.session import FrameRecord, SessionReader
from mia.tracker import UNKNOWN
from mia.tracker.rivers import WatchStats
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import DEFAULT_SKIN, TemplateSet
from mia.vision.tiles.river import SEATS

__all__ = [
    "ReplayReport",
    "SeatScore",
    "Tick",
    "replay",
    "schedule",
]


@dataclass(frozen=True, slots=True)
class Tick:
    """模擬的一次 ``tick()``。

    Attributes:
        wall: 這一輪醒來的牆上時鐘。
        record: 這一輪手上那一幀 —— **上一輪到這一輪之間最後抵達的那張**。
            這段時間內沒有新畫面就是 ``None``,而那是真的會發生的
            (擷取那條路卡住、視窗被遮)。``TableWorker.tick`` 對
            ``None`` 的反應也要一起量到,所以不過濾掉。
    """

    wall: float
    record: FrameRecord | None


def schedule(session: SessionReader, *, interval: float) -> Iterator[Tick]:
    """按實際的節奏排出每一輪拿到哪一幀。

    重現的是 ``offer_frame`` + ``tick`` 這一對的語意:投遞只留最新的,
    而 ``tick`` 取走之後把格子清空。所以一輪之內來了十五張,看得到的只有
    第十五張;一張都沒來,看到的是 ``None``。

    **不是「每隔 n 幀取一張」。** 那會均勻取樣,而真正的迴圈是照時間醒的,
    錄影的幀間隔並不均勻(去重規則是「畫面變了才存」,靜止時同一個檔案
    會被好幾幀共用)。
    """
    records = session.manifest.frames
    if not records:
        return
    walls = [session.wall_at(r.timestamp) for r in records]
    deadline = walls[0] + interval
    latest: FrameRecord | None = None
    cursor = 0
    while cursor < len(records) or latest is not None:
        while cursor < len(records) and walls[cursor] <= deadline:
            latest = records[cursor]
            cursor += 1
        yield Tick(wall=deadline, record=latest)
        latest = None
        deadline += interval


@dataclass(slots=True)
class SeatScore:
    """一家的比對結果。

    Attributes:
        right: 格線上那一格與標準答案相同。
        abstained: 那一格 CV 說不知道(``?``)。**這不是錯誤** ——
            放銃分析會把它當成危險,方向是保守的。
        wrong: 有信心卻認錯了。**這一欄才是真正的失敗**:它會把危險牌
            說成現物,而且沒有任何症狀。
        short: 格線比標準答案少了幾格(累計)。
        long: 格線比標準答案多了幾格(累計)。
        leaked: tracker 的河裡有、但標準答案沒有的牌(以集合比)。
            這是「放銃分析真的看到了假現物」的張數。
        held: tracker 的河裡正確的牌 —— 真的派上用場的現物。
    """

    right: int = 0
    abstained: int = 0
    wrong: int = 0
    short: int = 0
    long: int = 0
    leaked: int = 0
    held: int = 0

    @property
    def graded(self) -> int:
        return self.right + self.abstained + self.wrong


@dataclass(slots=True)
class ReplayReport:
    """一整場重播的結果。"""

    ticks: int = 0
    starved: int = 0
    scored: int = 0
    unstable: int = 0
    seats: list[SeatScore] = field(default_factory=lambda: [SeatScore() for _ in SEATS])
    durations: list[float] = field(default_factory=list)
    interval: float = 1.0
    reaches: list[str] = field(default_factory=list)
    doras: list[str] = field(default_factory=list)
    kyokus: int = 0
    posts: int = 0
    watch: WatchStats = field(default_factory=WatchStats)
    yardstick: str = "真時間"

    @property
    def overrun(self) -> int:
        """有幾輪讀得比 ``interval`` 還久 —— 那幾輪跟不上,下一輪會立刻接上。"""
        return sum(1 for d in self.durations if d > self.interval)

    def totals(self) -> SeatScore:
        out = SeatScore()
        for seat in self.seats:
            out.right += seat.right
            out.abstained += seat.abstained
            out.wrong += seat.wrong
            out.short += seat.short
            out.long += seat.long
            out.leaked += seat.leaked
            out.held += seat.held
        return out

    def summary(self) -> str:
        total = self.totals()
        lines = [
            f"醒了 {self.ticks} 輪,其中 {self.starved} 輪手上沒有新畫面",
            f"可評分 {self.scored} 輪({self.unstable} 輪沒有答案,跳過)"
            f"  答案來自{self.yardstick}",
            "",
            "每一輪的格線(位置 × 牌面)",
        ]
        if total.graded:
            lines += [
                f"  對       {total.right:7d}  {100 * total.right / total.graded:5.1f}%",
                f"  棄權     {total.abstained:7d}  {100 * total.abstained / total.graded:5.1f}%",
                f"  **錯**   {total.wrong:7d}  {100 * total.wrong / total.graded:5.1f}%",
                f"  少了格子 {total.short:7d}   多了格子 {total.long}",
            ]
        else:
            lines.append("  沒有一輪評得到分。")
        lines += ["", "tracker 的河(成員,放銃分析真正吃的)"]
        consumed = total.held + total.leaked
        if consumed:
            lines.append(
                f"  真現物   {total.held:7d}   **假現物 {total.leaked}** "
                f"({100 * total.leaked / consumed:.2f}%)"
            )
        else:
            lines.append("  tracker 的河一直是空的。")
        lines += ["", "每一家"]
        for name, seat in zip(SEATS, self.seats, strict=True):
            lines.append(
                f"  {name:<9} 對 {seat.right:6d}  棄權 {seat.abstained:6d}  "
                f"錯 {seat.wrong:4d}  少 {seat.short:5d}  假現物 {seat.leaked}"
            )
        lines += ["", "事件"]
        lines.append(f"  立直 {len(self.reaches)} 次:{'、'.join(self.reaches) or '(無)'}")
        lines.append(f"  寶牌 {len(self.doras)} 張:{'、'.join(self.doras) or '(無)'}")
        lines.append(f"  局邊界 {self.kyokus} 次   投出危險度 {self.posts} 次")
        # ``confused`` 與 ``deferred`` 是 tracker 那一層唯一的症狀 ——
        # 格線可以每一格都對,而這兩欄還是爆的(實測:認牌全對,但 watcher
        # 把「少讀一格」當成鳴牌,整排挪位)。所以一定要印出來。
        w = self.watch
        lines.append(
            f"  觀察 {w.observations} 次 → 捨牌 {w.emitted} 個,"
            f"補上 {w.resolved},鳴牌 {w.calls}"
        )
        # 要看的是 confused - collapsed:換局在清桌、立繪蓋住牌河的時候
        # 看不懂是**對的反應**,那些次數會把這一欄稀釋掉。
        lines.append(
            f"  **看不懂 {w.confused - w.collapsed}**"
            f"(另有 {w.collapsed} 次是河整個空掉,換局/立繪)"
            f"   先不動 {w.deferred}"
        )
        if self.durations:
            ordered = sorted(self.durations)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
            lines += [
                "",
                f"一輪要多久(預算 {self.interval:.1f}s)",
                f"  中位 {1000 * median(ordered):.0f}ms   "
                f"p95 {1000 * p95:.0f}ms   最久 {1000 * ordered[-1]:.0f}ms",
                f"  超出預算 {self.overrun} 輪 "
                f"({100 * self.overrun / len(self.durations):.1f}%)",
            ]
        return "\n".join(lines)


def _grade_grid(score: SeatScore, truth: Sequence[str], known: Sequence[str]) -> None:
    """格線逐格比 —— 位置也要對。

    少讀一格會讓後面**整串往前位移**,所以位置錯了就是錯了,不能只比集合。
    """
    for index, want in enumerate(truth):
        if index >= len(known):
            score.short += 1
            continue
        got = known[index]
        if got == UNKNOWN:
            score.abstained += 1
        elif got == normalize_red(want):
            score.right += 1
        else:
            score.wrong += 1
    score.long += max(0, len(known) - len(truth))


def _grade_members(score: SeatScore, truth: Sequence[str], river: Sequence[str]) -> None:
    """tracker 的河比**集合** —— 棄權的牌根本沒進去,位置本來就對不上。

    放銃分析只問「這張是不是現物」,問的是成員。多出來的那幾張是真正的
    危險:它們會被當成現物,而現物是「一定安全」。

    Note:
        標準答案要用 :attr:`~mia.mjai.table.Player.river` 而**不是**
        ``visible_river``。這裡踩過一次:第一版拿 ``visible_river`` 當答案,
        量到 4.98% 的「假現物」,而其中幾乎全部是**被別家鳴走的牌** ——
        那些牌畫面上確實不在了,但他**打過**,所以振聽成立、現物成立,
        tracker 留著是對的。症狀只是一個不高不低的百分比,完全看不出
        原因與辨識無關。``eval.rivers`` 的模組說明警告過反過來的那一半。
    """
    want = Counter(normalize_red(t) for t in truth)
    for tile in (normalize_red(t) for t in river):
        if want[tile] > 0:
            want[tile] -= 1
            score.held += 1
        else:
            score.leaked += 1


def replay(
    session: SessionReader,
    timeline: RiverTimeline,
    table: Rect,
    *,
    config: RoiConfig,
    interval: float = 1.0,
    skin: str = DEFAULT_SKIN,
    settle: float = SETTLE,
    limit: int = 0,
    on_tick: object = None,
    answers: Mapping[int, RiverState] | None = None,
) -> ReplayReport:
    """把整份錄影餵過 :class:`~mia.live.table.TableWorker`,逐輪對分。

    Args:
        session: 錄影。
        timeline: 封包建出來的牌河時間軸,當標準答案。
        table: 牌桌矩形。所有正規化座標相對它,**錯了整份報告會很正常地騙人**。
        config: ROI 設定(寶牌面板要用)。
        interval: worker 的節奏,秒。
        skin: 牌面皮膚。
        settle: 答案要穩定多久才算得分。轉場中的幀照樣餵進去 ——
            實際跑的時候躲不掉 —— 只是不拿來對分。
        limit: 最多跑幾輪,0 表示全部。
        on_tick: 每輪叫一次的進度回呼,收 ``(已完成, 總數)``。
        answers: 幀序號 → 標準答案,給了就**取代** ``stable_at``。
            見 :func:`~mia.eval.sequence.align_session`。

            兩個理由要它:牌譜素材沒有時間,``stable_at`` 什麼都答不出來;
            而即時素材上 ``stable_at`` 會跳過「狀態不到 ``settle`` 秒」的那幾輪
            —— 出牌快的時候、局尾最後一張 —— 那裡正好有它量不到的錯
            (``docs/cv_mjai.md`` §3.4)。

    Note:
        **worker 的執行緒不啟動**,直接叫 :meth:`~mia.live.table.TableWorker.tick`。
        要量的是每一輪讀出什麼、花多久,而不是 ``threading`` 會不會排程 ——
        起執行緒只會讓結果變得不可重現。
    """
    from mia.calibration.table import Calibration

    worker = TableWorker(
        UpdateBus(),
        config=config,
        skin=skin,
        interval=interval,
        templates=TemplateSet.load(skin),
    )
    calibration = Calibration(
        table_rect=table, image_size=session.image_size, source="manual"
    )

    report = ReplayReport(interval=interval, yardstick="單調對齊" if answers else "真時間")
    ticks = list(schedule(session, interval=interval))
    if limit:
        ticks = ticks[:limit]

    seen_reach = [False for _ in SEATS]
    seen_dora: list[str] = []
    kyokus = 0

    for done, tick in enumerate(ticks, 1):
        report.ticks += 1
        if tick.record is None:
            report.starved += 1
        else:
            worker.offer_frame(session.load_image(tick.record), calibration)
            # 手牌不餵。這一支量的是牌河與事件流,而手牌有自己的評測
            # (``tools/evaluate.py``)—— 從封包灌一副進來只會讓兩邊的
            # 誤差混在一起,分不出是誰的。
        started = time.perf_counter()
        worker.tick()
        report.durations.append(time.perf_counter() - started)

        if worker._kyoku.started != kyokus:  # noqa: SLF001
            kyokus = worker._kyoku.started  # noqa: SLF001
            report.kyokus += 1
            # 換局會把 ``_sideways`` 清回 None —— 下一局的立直要能再數到一次,
            # 所以「這一家立直過了」這件事也得跟著清。
            seen_reach = [False for _ in SEATS]

        for position in range(len(SEATS)):
            if worker._sideways[position] is not None and not seen_reach[position]:  # noqa: SLF001
                seen_reach[position] = True
                report.reaches.append(f"{SEATS[position]}@{done}")
        # ``DoraWatcher.indicators`` 存的是牌名字串,而且**換局會被清空**
        # (``_new_kyoku`` 會 reset)—— 所以不能拿累積長度當游標,
        # 要比對上一輪的快照。縮短了就是換局,重新開始數。
        current = list(worker._dora.indicators)  # noqa: SLF001
        if len(current) < len(seen_dora):
            seen_dora = []
        for indicator in current[len(seen_dora) :]:
            report.doras.append(f"{indicator}@{done}")
        seen_dora = current
        if answers is None:
            truth = timeline.stable_at(tick.wall, settle)
        else:
            truth = answers.get(tick.record.index) if tick.record is not None else None
        if truth is None:
            report.unstable += 1
        else:
            report.scored += 1
            for position, want in enumerate(truth.rivers):
                _grade_grid(report.seats[position], want, worker._known[position])  # noqa: SLF001
                _grade_members(
                    report.seats[position],
                    truth.full[position],
                    worker._table.players[position].river,  # noqa: SLF001
                )
        if callable(on_tick):
            on_tick(done, len(ticks))

    report.posts = worker.posts
    report.watch = worker._rivers.stats  # noqa: SLF001
    return report
