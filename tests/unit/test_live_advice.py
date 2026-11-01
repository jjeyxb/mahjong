"""事件流接到引擎。理由見 ``mia/live/advice.py``。"""

from __future__ import annotations

from mia.config.loader import load_config
from mia.engine.base import Advice
from mia.engine.multiplex import EngineGroup
from mia.live.advice import StreamFeeder, VisionAdviceWorker
from mia.live.bus import EventStream, UpdateBus
from mia.live.table import TableLink
from mia.mjai.events import Dahai, EndKyoku, MjaiEvent, StartGame, Tsumo
from mia.tracker import Stream


class Recorder:
    name = "記錄"

    def __init__(self) -> None:
        self.seen: list[MjaiEvent] = []

    def start(self) -> None: ...

    def close(self) -> None: ...

    def react(self, event: MjaiEvent) -> Advice:
        self.seen.append(event)
        return Advice(self.name)

    def peek(self, event: MjaiEvent) -> Advice:  # noqa: ARG002
        return Advice(self.name)


DRAW = Tsumo(actor=1, pai="?")
DROP = Dahai(actor=1, pai="1m", tsumogiri=False)
OTHER = Dahai(actor=1, pai="9m", tsumogiri=False)
NEXT = Tsumo(actor=2, pai="?")


def feeder() -> tuple[StreamFeeder, Recorder]:
    engine = Recorder()
    return StreamFeeder(EngineGroup([engine], parallel=False)), engine


class TestFeeding:
    def test_the_first_stream_starts_the_game(self) -> None:
        stream, engine = feeder()
        stream.update(1, [DRAW])
        assert engine.seen == [StartGame(id=0), DRAW]

    def test_only_the_tail_is_sent(self) -> None:
        stream, engine = feeder()
        stream.update(1, [DRAW])
        stream.update(1, [DRAW, DROP])
        assert engine.seen == [StartGame(id=0), DRAW, DROP]
        assert stream.rewinds == 0

    def test_nothing_new_is_nothing(self) -> None:
        stream, engine = feeder()
        stream.update(1, [DRAW])
        assert stream.update(1, [DRAW]) is None
        assert len(engine.seen) == 2

    def test_an_empty_stream_sends_nothing(self) -> None:
        stream, engine = feeder()
        assert stream.update(1, []) is None
        assert engine.seen == []


class TestRewriting:
    def test_a_changed_past_replays_the_kyoku(self) -> None:
        """牌面晚到,前面的牌換了:引擎不能改過去,只能重來。"""
        stream, engine = feeder()
        stream.update(1, [DRAW, DROP])
        stream.update(1, [DRAW, OTHER, NEXT])
        assert engine.seen == [
            StartGame(id=0),
            DRAW,
            DROP,
            StartGame(id=0),
            DRAW,
            OTHER,
            NEXT,
        ]
        assert stream.rewinds == 1

    def test_a_shorter_stream_also_replays(self) -> None:
        stream, engine = feeder()
        stream.update(1, [DRAW, DROP])
        stream.update(1, [DRAW])
        assert engine.seen[-2:] == [StartGame(id=0), DRAW]


class TestKyoku:
    def test_a_new_kyoku_ends_the_old_one_without_a_rewind(self) -> None:
        stream, engine = feeder()
        stream.update(1, [DRAW])
        stream.update(2, [NEXT])
        assert engine.seen == [StartGame(id=0), DRAW, EndKyoku(), NEXT]
        assert stream.rewinds == 0

    def test_an_empty_kyoku_ends_nothing(self) -> None:
        stream, engine = feeder()
        stream.update(1, [])
        stream.update(2, [DRAW])
        assert engine.seen == [StartGame(id=0), DRAW]


class TestLog:
    """Log 視窗要看得到事件流,對不上而不給建議的時候尤其要看得到。"""

    @staticmethod
    def _worker() -> tuple[VisionAdviceWorker, UpdateBus]:
        bus = UpdateBus()
        link = TableLink()
        worker = VisionAdviceWorker(bus, config=load_config().roi, engines=[], link=link)
        return worker, bus

    def test_the_stream_is_posted(self) -> None:
        worker, bus = self._worker()
        events = (Tsumo(actor=1, pai="?"),)
        worker._log(Stream(events=events))  # noqa: SLF001
        update = bus.drain()[0]
        assert update == EventStream("圖像辨識", 0, events)

    def test_why_it_stopped_goes_with_it(self) -> None:
        worker, bus = self._worker()
        worker._log(Stream(events=(Tsumo(actor=1, pai="?"),), stalled=True))  # noqa: SLF001
        assert bus.drain()[0].note == "等手牌"  # type: ignore[union-attr]

    def test_a_doubt_outranks_a_stall(self) -> None:
        worker, bus = self._worker()
        stream = Stream(events=(Tsumo(actor=1, pai="?"),), stalled=True, doubt="余牌對不上")
        worker._log(stream)  # noqa: SLF001
        assert bus.drain()[0].note == "余牌對不上"  # type: ignore[union-attr]

    def test_it_is_posted_even_when_advice_stays_quiet(self) -> None:
        worker, bus = self._worker()
        stream = Stream(events=(Tsumo(actor=1, pai="?"),), doubt="手牌對不上")
        worker._feed(1, stream)  # noqa: SLF001
        assert any(isinstance(update, EventStream) for update in bus.drain())

    def test_an_empty_stream_says_nothing(self) -> None:
        worker, bus = self._worker()
        worker._log(Stream())  # noqa: SLF001
        assert bus.drain() == ()
