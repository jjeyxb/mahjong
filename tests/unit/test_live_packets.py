"""封包執行緒:跟著正在被寫入的錄影檔走,一路到引擎建議。

用真實錄影當素材(``tests/fixtures/real_game_full.jsonl``,一場完整的東風戰)。
自己造的假封包只能驗「程式沒當」,驗不到協定本身 —— 而協定就是這條路上最容易
錯的地方。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from mia.engine.dummy import DummyEngine
from mia.live.bus import Advices, PacketHand, UpdateBus
from mia.live.packets import DumpTail, PacketWorker

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_game_full.jsonl"


@pytest.fixture
def raw_lines() -> list[str]:
    return [line for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _drain_until(
    bus: UpdateBus, predicate: object, *, timeout: float = 30.0
) -> list[object]:
    """收郵箱直到 ``predicate`` 對收到的清單成立,或逾時。

    不能只 sleep 一次就檢查:工作執行緒的進度取決於機器負載,而引擎啟動本身
    要好幾秒。逾時值訂得寬,失敗時才是真的失敗。
    """
    collected: list[object] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        collected.extend(bus.drain())
        if predicate(collected):  # type: ignore[operator]
            return collected
        time.sleep(0.02)
    collected.extend(bus.drain())
    return collected


class TestDumpTail:
    def test_reads_a_file_that_is_already_complete(self, tmp_path: Path, raw_lines) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        stop = threading.Event()
        tail = DumpTail(path)
        frames = []
        for frame in tail.frames(stop):
            frames.append(frame)
            if len(frames) == len(raw_lines):
                stop.set()
        assert len(frames) == len(raw_lines)

    def test_waits_for_a_file_that_does_not_exist_yet(self, tmp_path: Path) -> None:
        """擷取子程序通常比封包執行緒晚幾秒才建檔。"""
        path = tmp_path / "later.jsonl"
        stop = threading.Event()
        tail = DumpTail(path, poll=0.01)

        got: list[object] = []
        thread = threading.Thread(target=lambda: got.extend(tail.frames(stop)), daemon=True)
        thread.start()
        time.sleep(0.05)
        assert got == []  # 還沒有檔案,也還沒有炸掉

        path.write_text(_one_line() + "\n", encoding="utf-8")
        deadline = time.monotonic() + 3
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(2)
        assert len(got) == 1

    def test_picks_up_lines_appended_while_reading(self, tmp_path: Path, raw_lines) -> None:
        path = tmp_path / "growing.jsonl"
        path.write_text(raw_lines[0] + "\n", encoding="utf-8")

        stop = threading.Event()
        tail = DumpTail(path, poll=0.01)
        got: list[object] = []
        thread = threading.Thread(target=lambda: got.extend(tail.frames(stop)), daemon=True)
        thread.start()

        with path.open("a", encoding="utf-8") as handle:
            for line in raw_lines[1:20]:
                handle.write(line + "\n")
                handle.flush()

        deadline = time.monotonic() + 3
        while len(got) < 20 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(2)
        assert len(got) == 20

    def test_a_half_written_line_is_not_parsed_until_complete(self, tmp_path: Path) -> None:
        """``DumpWriter`` 每行都 flush,但一次 write 在底層仍可能被切開。

        半截的 JSON 若被當成一行處理,會被記成壞行 —— 而它其實是好的,
        只是還沒寫完。
        """
        path = tmp_path / "partial.jsonl"
        line = _one_line()
        path.write_text(line[:30], encoding="utf-8")  # 沒有換行

        stop = threading.Event()
        tail = DumpTail(path, poll=0.01)
        got: list[object] = []
        thread = threading.Thread(target=lambda: got.extend(tail.frames(stop)), daemon=True)
        thread.start()
        time.sleep(0.08)
        assert got == []
        assert tail.bad_lines == 0  # 關鍵:沒有被誤判成壞行

        with path.open("a", encoding="utf-8") as handle:
            handle.write(line[30:] + "\n")
            handle.flush()

        deadline = time.monotonic() + 3
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(2)
        assert len(got) == 1
        assert tail.bad_lines == 0

    def test_garbage_lines_are_counted_and_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.jsonl"
        path.write_text(f"這不是 JSON\n{_one_line()}\n", encoding="utf-8")

        stop = threading.Event()
        tail = DumpTail(path, poll=0.01)
        frames = []
        for frame in tail.frames(stop):
            frames.append(frame)
            stop.set()
        assert len(frames) == 1
        assert tail.bad_lines == 1

    def test_from_start_false_skips_existing_content(self, tmp_path: Path, raw_lines) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text("\n".join(raw_lines[:50]) + "\n", encoding="utf-8")

        stop = threading.Event()
        tail = DumpTail(path, from_start=False, poll=0.01)
        got: list[object] = []
        thread = threading.Thread(target=lambda: got.extend(tail.frames(stop)), daemon=True)
        thread.start()
        time.sleep(0.08)
        assert got == []

        with path.open("a", encoding="utf-8") as handle:
            handle.write(raw_lines[50] + "\n")
            handle.flush()
        deadline = time.monotonic() + 3
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        thread.join(2)
        assert len(got) == 1

    def test_stop_ends_the_iteration_promptly(self, tmp_path: Path) -> None:
        path = tmp_path / "idle.jsonl"
        path.write_text("", encoding="utf-8")
        stop = threading.Event()
        tail = DumpTail(path, poll=0.01)

        done = threading.Event()

        def consume() -> None:
            list(tail.frames(stop))
            done.set()

        threading.Thread(target=consume, daemon=True).start()
        time.sleep(0.03)
        stop.set()
        assert done.wait(2), "設了 stop 之後執行緒沒有結束"


class TestPacketWorker:
    """整條路:錄影檔 → liqi → MJAI → 手牌 + 引擎建議。

    引擎用 ``DummyEngine``(行程內、無 torch)—— 這裡驗的是接線,
    Mortal 子程序本身在 ``test_engine_mortal.py`` 驗過了。
    """

    def test_produces_hands_and_advice_from_a_real_game(
        self, tmp_path: Path, raw_lines
    ) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        bus = UpdateBus()
        worker = PacketWorker(bus, dump=path, engines=[DummyEngine()])
        worker.start()
        try:
            collected = _drain_until(
                bus,
                lambda got: any(isinstance(u, Advices) for u in got) and worker.decisions >= 20,
            )
        finally:
            worker.stop()
            worker.join(5)

        hands = [u for u in collected if isinstance(u, PacketHand)]
        advices = [u for u in collected if isinstance(u, Advices)]
        assert hands, "沒有推出任何手牌"
        assert advices, "沒有產生任何建議"
        # 錄影裡自己坐 2,而封包只看得到自己的手牌
        assert worker.seat == 2
        assert worker.decisions >= 20

    def test_the_hands_it_posts_are_legal(self, tmp_path: Path, raw_lines) -> None:
        """張數必須是 13/14 那一族(扣掉副露)。合併過的郵箱只留最新一筆,
        所以這裡看的是抽樣 —— 但錯的張數會讓向聽算不出來,是很值得釘住的。
        """
        path = tmp_path / "ws.jsonl"
        path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        bus = UpdateBus()
        seen: list[PacketHand] = []
        worker = PacketWorker(bus, dump=path, engines=[DummyEngine()])
        worker.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and worker.decisions < 40:
            seen.extend(u for u in bus.drain() if isinstance(u, PacketHand))
            time.sleep(0.005)
        worker.stop()
        worker.join(5)

        assert len(seen) >= 10
        for hand in seen:
            assert len(hand.tiles) % 3 in (1, 2), f"不合法的張數: {hand.tiles}"

    def test_does_not_feed_engines_before_start_game(self, tmp_path: Path, raw_lines) -> None:
        """中途接上時要什麼都不做,而不是把 start_kyoku 硬餵進引擎。

        MJAI 引擎沒有 ``start_game`` 就不知道自己是誰,硬餵下去會報錯,
        然後被整組移出這一場 —— 那一場就再也不會有建議了。
        """
        events = _mjai_events_of(raw_lines)
        first_kyoku = next(i for i, e in enumerate(events) if e["type"] == "start_kyoku")
        assert first_kyoku > 0  # 錄影裡 start_game 真的在前面

        bus = UpdateBus()
        engine = DummyEngine()
        # 直接餵一個沒有 start_game 的事件流:用 _handle 而不是整條 tail,
        # 因為要測的正是「armed 之前」那個狀態
        worker = PacketWorker(bus, dump=tmp_path / "unused.jsonl", engines=[engine])
        from mia.mjai import parse_event

        worker._handle(parse_event(events[first_kyoku]))  # noqa: SLF001
        assert bus.drain() == ()
        assert "中途接上" in worker.status.read()

    def test_start_game_resets_the_hand_tracker(self, tmp_path: Path, raw_lines) -> None:
        """連打兩場時座位會變,上一場的手牌不能留下來。"""
        from mia.mjai import parse_event

        events = _mjai_events_of(raw_lines)
        bus = UpdateBus()
        worker = PacketWorker(bus, dump=tmp_path / "unused.jsonl", engines=[])
        for event in events[:60]:
            worker._handle(parse_event(event))  # noqa: SLF001
        assert worker._tracker.tiles  # noqa: SLF001

        start_game = next(e for e in events if e["type"] == "start_game")
        worker._handle(parse_event(start_game))  # noqa: SLF001
        assert worker._tracker.tiles == []  # noqa: SLF001

    def test_only_posts_a_hand_when_it_actually_changed(
        self, tmp_path: Path, raw_lines
    ) -> None:
        """別家打牌不動到自己的手牌,重投只會讓 UI 白算一次向聽。"""
        from mia.mjai import parse_event

        events = _mjai_events_of(raw_lines)
        bus = UpdateBus()
        worker = PacketWorker(bus, dump=tmp_path / "unused.jsonl", engines=[])
        for event in events[:80]:
            worker._handle(parse_event(event))  # noqa: SLF001
            bus.drain()

        posted_before = bus.stats.posted
        # 別家的打牌事件
        other = next(
            e
            for e in events[80:]
            if e["type"] == "dahai" and e.get("actor") != worker.seat
        )
        worker._handle(parse_event(other))  # noqa: SLF001
        assert bus.stats.posted == posted_before


def _one_line() -> str:
    """一行合法的錄影檔內容 —— 從真實錄影取,不是自己編的。"""
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line
    raise AssertionError("fixture 是空的")


def _mjai_events_of(raw_lines: list[str]) -> list[dict]:
    """把錄影解成 MJAI 事件的 dict。"""
    from mia.groundtruth.dump import DumpFrame
    from mia.groundtruth.stream import MjaiDecoder

    decoder = MjaiDecoder()
    events: list[dict] = []
    for line in raw_lines:
        decoded = decoder.feed(DumpFrame.from_json(json.loads(line)))
        if decoded is not None:
            events.extend(e.to_dict() for e in decoded.events)
    return events
