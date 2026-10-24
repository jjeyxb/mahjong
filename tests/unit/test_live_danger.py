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
    """同步把整場餵進去,把**每一次**投出來的東西整個收下來。

    收的是 :class:`Dangers` 本身而不是 ``.report`` —— 那班車上還有 ``seen``,
    而兩者是分開變的。只收 report 的話,「報告沒變但看得見的牌變了」這種投遞
    在測試眼裡會變成一次重複投遞。
    """
    bus = UpdateBus()
    worker = DangerWorker(bus, dump=DUMP)
    posts: list[Dangers] = []
    for event in events:
        worker._handle(event)  # noqa: SLF001 - 就是要繞過執行緒與郵箱的合併
        posts.extend(u for u in bus.drain() if isinstance(u, Dangers))
    return worker, posts


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
        worker, posts = driven
        assert 0 < len(posts) < worker.events

    def test_consecutive_posts_are_never_identical(self, driven) -> None:
        """只在真的變了才投 —— 這是上一條的嚴格版。

        比的是整班車(報告 + 看得見的牌)。**連續兩份報告相同是允許的** ——
        別人打掉一張我要的進張時,那張牌對我的安全度可能毫無影響,報告一字
        不變,但我的進張少了一張。那一次必須投,不然畫面上的枚數會停在舊值
        而且沒有任何徵兆。
        """
        _worker, posts = driven
        assert all(a != b for a, b in pairwise(posts))

    def test_some_posts_carry_the_same_report(self, driven) -> None:
        """而且上面那種情況真的會發生 —— 不是理論上的擔心。

        這條是上一條的前提檢查:如果整場下來每次投遞的報告都不同,那
        `seen` 根本沒派上用場,上一條就只是換個寫法重測同一件事。
        """
        _worker, posts = driven
        assert any(a.report == b.report for a, b in pairwise(posts))

    def test_every_post_carries_what_is_visible(self, driven) -> None:
        """看得見的牌要跟著報告一起走,不能是 None。

        進張枚數靠它扣準(decisions.md 未解 #4)。少了它那條路會安靜地
        退回估計值 —— 畫面照樣有數字,只是偏高。
        """
        _worker, posts = driven
        assert all(p.seen for p in posts)

    def test_the_reports_are_about_our_own_hand(self, driven) -> None:
        _worker, posts = driven
        assert all(1 <= len(p.report.tiles) <= 14 for p in posts)

    def test_threats_are_picked_up(self, driven) -> None:
        """那場有四次立直、也有人副露到三組,兩種都該被指名。"""
        _worker, posts = driven
        labels = {t.label for p in posts for t in p.report.threats}
        assert "立直" in labels
        assert "3副露" in labels

    def test_every_report_is_sorted_safest_first(self, driven) -> None:
        _worker, posts = driven
        for post in posts:
            levels = [t.level for t in post.report.tiles]
            assert levels == sorted(levels)

    def test_it_finds_safe_tiles_somewhere_in_the_game(self, driven) -> None:
        _worker, posts = driven
        assert any(t.level is DangerLevel.SAFE for p in posts for t in p.report.tiles)

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
