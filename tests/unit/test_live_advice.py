"""事件流接到引擎。理由見 ``mia/live/advice.py``。"""

from __future__ import annotations

from mia.engine.base import Advice
from mia.engine.multiplex import EngineGroup
from mia.live.advice import StreamFeeder
from mia.mjai.events import Dahai, EndKyoku, MjaiEvent, StartGame, Tsumo


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
