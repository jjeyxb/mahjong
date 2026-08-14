"""放銃分析的執行緒。

它跟 AI 建議走同一份錄影檔、同一套解碼,但**不開引擎子程序** —— 那正是把
功能 2 拆成兩層的理由。

內容的斷言一律**同步驅動**,不從郵箱撈:郵箱每個 slot 只留最新一筆(那是
它的設計,見 bus.py),所以整場投出的一百多份報告,消費端只會看到兩三份。
拿那個抽樣去斷言內容,測試會照著執行緒排程時好時壞。
"""

from __future__ import annotations

import time
from itertools import pairwise
from pathlib import Path

import pytest

from mia.analysis.danger import DangerLevel
from mia.groundtruth.stream import MjaiDecoder
from mia.live.bus import Dangers, UpdateBus
from mia.live.danger import DangerWorker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DUMP = FIXTURES / "real_game_full.jsonl"


@pytest.fixture(scope="module")
def events() -> list:
    return [e for d in MjaiDecoder().decode_file(DUMP) for e in d.events]


def _drive(events: list) -> tuple[DangerWorker, list]:
    """同步把整場餵進去,把**每一份**投出來的報告都收下來。"""
    bus = UpdateBus()
    worker = DangerWorker(bus, dump=DUMP)
    reports: list = []
    for event in events:
        worker._handle(event)  # noqa: SLF001 - 就是要繞過執行緒與郵箱的合併
        reports.extend(u.report for u in bus.drain() if isinstance(u, Dangers))
    return worker, reports


@pytest.fixture(scope="module")
def driven(events: list) -> tuple[DangerWorker, list]:
    return _drive(events)


class TestReplayingARealGame:
    def test_it_finds_our_seat_from_the_packets(self, driven) -> None:
        worker, _ = driven
        assert worker.seat == 2

    def test_it_does_not_post_on_every_event(self, driven) -> None:
        """別家摸牌、寶牌翻開這些事件不會改變任何一張牌的危險度。
        每個事件都投的話,UI 每一輪都要重畫一次一模一樣的東西。"""
        worker, reports = driven
        assert 0 < len(reports) < worker.events

    def test_consecutive_reports_are_never_identical(self, driven) -> None:
        """只在真的變了才投 —— 這是上一條的嚴格版。"""
        _worker, reports = driven
        assert all(a != b for a, b in pairwise(reports))

    def test_the_reports_are_about_our_own_hand(self, driven) -> None:
        _worker, reports = driven
        assert all(1 <= len(r.tiles) <= 14 for r in reports)

    def test_threats_are_picked_up(self, driven) -> None:
        """那場有四次立直、也有人副露到三組,兩種都該被指名。"""
        _worker, reports = driven
        labels = {t.label for r in reports for t in r.threats}
        assert "立直" in labels
        assert "3副露" in labels

    def test_every_report_is_sorted_safest_first(self, driven) -> None:
        _worker, reports = driven
        for report in reports:
            levels = [t.level for t in report.tiles]
            assert levels == sorted(levels)

    def test_it_finds_safe_tiles_somewhere_in_the_game(self, driven) -> None:
        _worker, reports = driven
        assert any(t.level is DangerLevel.SAFE for r in reports for t in r.tiles)

    def test_a_new_game_forgets_the_previous_table(self, events: list) -> None:
        """``start_game`` 之後座位可能變了,兩個追蹤器都要從頭來 ——
        留著上一場的安全牌會安靜地把危險牌說成安全。"""
        worker, _ = _drive(events)
        worker._handle(events[0])  # noqa: SLF001 - 再來一次 start_game
        assert worker._table.players[0].safe == set()  # noqa: SLF001


class TestThreadedRun:
    def test_it_really_runs_and_posts(self) -> None:
        """真的開執行緒跑一次 —— 上面那些繞過了執行緒,這條負責證明它會轉。"""
        bus = UpdateBus()
        worker = DangerWorker(bus, dump=DUMP)
        worker.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and worker.posts == 0:
            time.sleep(0.02)
        worker.stop()
        worker.join(3)
        assert worker.posts > 0
        assert not worker.is_alive()

    def test_a_missing_dump_does_not_crash_the_thread(self, tmp_path: Path) -> None:
        """錄影檔還沒出現是正常的(先開功能再按開始遊戲),要等它,不是死掉。"""
        bus = UpdateBus()
        worker = DangerWorker(bus, dump=tmp_path / "nope.jsonl")
        worker.start()
        time.sleep(0.2)
        assert worker.is_alive()
        worker.stop()
        worker.join(3)
        assert len(bus) == 0


class TestItNeedsNoEngine:
    def test_constructing_it_takes_no_engines(self) -> None:
        """簽名裡沒有 engines —— 這是「不載 130MB 權重」在型別上的保證。"""
        import inspect

        assert "engines" not in inspect.signature(DangerWorker.__init__).parameters
