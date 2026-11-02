"""封包對畫面:一場實打的錯誤與延遲。

這份比對是之後「修什麼」的依據,算錯的話會去修不存在的問題,或漏掉真的問題。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mia.engine.base import Advice, EngineError
from mia.eval.streamdiff import (
    TruthKyoku,
    Version,
    agreement,
    compare,
    load_run,
    load_truth,
    summary,
)
from mia.mjai.events import (
    Dahai,
    Dora,
    EndKyoku,
    Kakan,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    StartGame,
    StartKyoku,
    Tsumo,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "S"]
START = StartKyoku(
    bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0, dora_marker="1p",
    tehais=[HAND, ["?"] * 13, ["?"] * 13, ["?"] * 13],
    scores=[25000, 25000, 25000, 25000],
)


def _truth(*events: MjaiEvent, start: float = 100.0) -> TruthKyoku:
    timed = [(start + index, event) for index, event in enumerate((START, *events))]
    return TruthKyoku(("E", 1, 0), tuple(timed))


def _version(*events: MjaiEvent, wall: float = 200.0, kyoku: int = 1, note: str = "") -> Version:
    return Version(wall, kyoku, (START, *events), note)


TURN = (
    Tsumo(actor=0, pai="N"),
    Dahai(actor=0, pai="N", tsumogiri=True),
    Tsumo(actor=1, pai="?"),
    Dahai(actor=1, pai="9m", tsumogiri=False),
)


class TestFinal:
    def test_a_perfect_stream_has_no_issues(self) -> None:
        result = compare([_truth(*TURN)], [_version(*TURN)])[0]
        assert result.issues == []
        assert result.counts["捨牌對"] == 2
        assert result.counts["自摸對"] == 1

    def test_tsumogiri_is_not_an_error(self) -> None:
        """別家摸切看不出來,畫面一律報手切。那不是讀錯。"""
        seen = (*TURN[:3], replace(TURN[3], tsumogiri=True))
        assert compare([_truth(*TURN)], [_version(*seen)])[0].issues == []

    def test_a_misread_discard(self) -> None:
        seen = (*TURN[:3], replace(TURN[3], pai="8m"))
        issue = compare([_truth(*TURN)], [_version(*seen)])[0].issues[0]
        assert (issue.kind, issue.seat, issue.index) == ("捨牌錯", 1, 0)
        assert (issue.expected, issue.got) == ("9m", "8m")
        assert issue.wall == 104.0

    def test_a_missing_discard_does_not_shift_the_rest(self) -> None:
        """少排一張的話,後面每張都對。不能算成一路錯到底。"""
        tiles = ["9m", "8m", "7m", "6m"]
        truth = [Dahai(actor=1, pai=p, tsumogiri=False) for p in tiles]
        seen = [e for e in truth if e.pai != "8m"]
        result = compare([_truth(*truth)], [_version(*seen)])[0]
        assert [issue.kind for issue in result.issues] == ["捨牌漏"]
        assert result.counts["捨牌對"] == 3

    def test_an_extra_discard(self) -> None:
        extra = Dahai(actor=2, pai="9m", tsumogiri=False)
        result = compare([_truth(*TURN)], [_version(*TURN, extra)])[0]
        assert [issue.kind for issue in result.issues] == ["捨牌多"]

    def test_a_wrong_draw(self) -> None:
        """第二次摸牌:莊家第一摸與配牌算在一起,見下面。"""
        turns = (*TURN, Tsumo(actor=0, pai="W"))
        seen = (*TURN, Tsumo(actor=0, pai="F"))
        assert compare([_truth(*turns)], [_version(*seen)])[0].counts["自摸錯"] == 1

    def test_a_field_of_the_opening(self) -> None:
        wrong = replace(START, honba=1)
        result = compare([_truth(*TURN)], [Version(200.0, 1, (wrong, *TURN))])[0]
        assert [issue.kind for issue in result.issues] == ["開局 honba"]

    def test_which_of_the_dealers_14_is_the_draw_does_not_matter(self) -> None:
        """莊家開局畫面上是 14 張,分不出哪一張是摸的。實測報成配牌與自摸各錯一張。"""
        truth_tiles = [*HAND[:-1], "P"]
        seen_tiles = [*HAND[:-1], "4p"]
        start = replace(START, tehais=[truth_tiles, *START.tehais[1:]])
        truth = TruthKyoku(
            ("E", 1, 0), ((100.0, start), (101.0, Tsumo(actor=0, pai="4p")))
        )
        wrong = replace(START, tehais=[seen_tiles, *START.tehais[1:]])
        got = compare([truth], [Version(200.0, 1, (wrong, Tsumo(actor=0, pai="P")))])[0]
        assert got.issues == []

    def test_a_dealers_hand_can_still_be_wrong(self) -> None:
        truth = TruthKyoku(("E", 1, 0), ((100.0, START), (101.0, Tsumo(actor=0, pai="N"))))
        wrong = replace(START, tehais=[[*HAND[:-1], "W"], *START.tehais[1:]])
        got = compare([truth], [Version(200.0, 1, (wrong, Tsumo(actor=0, pai="N")))])[0]
        assert [issue.kind for issue in got.issues] == ["配牌"]

    def test_a_kyoku_without_a_stream(self) -> None:
        result = compare([_truth(*TURN)], [])[0]
        assert [issue.kind for issue in result.issues] == ["整局沒有事件流"]

    def test_the_kyoku_is_found_by_time_not_by_name(self) -> None:
        """本場讀錯的話局名對不上,但那是要報的錯,不是「沒有事件流」。"""
        other = TruthKyoku(("E", 2, 0), ((500.0, replace(START, kyoku=2)),))
        wrong = replace(START, honba=3)
        results = compare([_truth(*TURN), other], [Version(300.0, 1, (wrong, *TURN))])
        assert [issue.kind for issue in results[0].issues] == ["開局 honba"]
        assert [issue.kind for issue in results[1].issues] == ["整局沒有事件流"]


class TestMelds:
    PON = Pon(actor=1, target=0, pai="N", consumed=["N", "N"])

    def test_correct(self) -> None:
        result = compare([_truth(self.PON)], [_version(self.PON)])[0]
        assert result.counts["副露全對"] == 1
        assert result.issues == []

    def test_the_consumed_order_does_not_matter(self) -> None:
        chi_a = Pon(actor=1, target=0, pai="5m", consumed=["5mr", "5m"])
        chi_b = Pon(actor=1, target=0, pai="5m", consumed=["5m", "5mr"])
        assert compare([_truth(chi_a)], [_version(chi_b)])[0].issues == []

    def test_wrong_faces(self) -> None:
        seen = replace(self.PON, pai="W", consumed=["W", "W"])
        assert compare([_truth(self.PON)], [_version(seen)])[0].issues[0].kind == "副露牌面差"

    def test_wrong_kind(self) -> None:
        seen = Kakan(actor=1, pai="N", consumed=["N", "N", "N"])
        assert compare([_truth(self.PON)], [_version(seen)])[0].issues[0].kind == "副露類型錯"

    def test_missing_and_extra(self) -> None:
        assert compare([_truth(self.PON)], [_version()])[0].issues[0].kind == "副露漏"
        assert compare([_truth()], [_version(self.PON)])[0].issues[0].kind == "副露多"


class TestOthers:
    def test_doras_and_reaches(self) -> None:
        truth = _truth(Dora(dora_marker="P"), Reach(actor=2), ReachAccepted(actor=2))
        kinds = [issue.kind for issue in compare([truth], [_version()])[0].issues]
        assert kinds == ["槓寶牌", "立直漏"]

    def test_an_extra_reach(self) -> None:
        kinds = compare([_truth()], [_version(ReachAccepted(actor=3))])[0].issues
        assert [issue.kind for issue in kinds] == ["立直多"]


class TestOverTime:
    def test_lag_is_measured_from_the_packet(self) -> None:
        versions = [_version(TURN[0], wall=101.5), _version(*TURN, wall=110.0)]
        result = compare([_truth(*TURN)], versions)[0]
        assert result.lags["T"] == [0.5]

    def test_a_tile_that_was_fixed_counts_as_wrong_at_the_time(self) -> None:
        """局末是對的,但當下給的建議是看著錯的牌算的。"""
        wrong = (*TURN[:3], replace(TURN[3], pai="8m"))
        versions = [_version(*wrong, wall=105.0), _version(*TURN, wall=106.0)]
        result = compare([_truth(*TURN)], versions)[0]
        assert result.issues == []
        assert result.counts["當下錯後來改對"] == 1
        assert result.lags["D"] == [2.0], "延遲算到讀對的那一刻"

    def test_rewrites_and_stalls(self) -> None:
        wrong = (*TURN[:3], replace(TURN[3], pai="8m"))
        versions = [
            _version(TURN[0], wall=105.0, note="等手牌"),
            _version(*wrong, wall=108.0),
            _version(*TURN, wall=109.0),
        ]
        result = compare([_truth(*TURN)], versions)[0]
        assert result.rewrites == 1
        assert result.noted == pytest.approx(3.0)


class _Engine:
    """照自己手上的規則回答:看到自己摸牌就打那一張。"""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.closed = False

    def react(self, event: MjaiEvent) -> Advice:
        if self.fail_on and getattr(event, "pai", None) == self.fail_on:
            raise EngineError("不合法\n細節")
        if isinstance(event, Tsumo) and event.actor == 0:
            return Advice("x", action=Dahai(actor=0, pai=event.pai, tsumogiri=True))
        assert isinstance(event, StartGame | StartKyoku | Dahai | Tsumo | EndKyoku)
        return Advice("x")

    def close(self) -> None:
        self.closed = True


class TestAgreement:
    def test_same_answer(self) -> None:
        got = agreement([_truth(*TURN)], [_version(*TURN)], _Engine)
        assert (got.turns, got.turns_seen, got.turns_same) == (1, 1, 1)

    def test_it_uses_what_was_on_screen_at_that_moment(self) -> None:
        """第一次排到時摸的是 W,之後才改對。當下給的建議是打 W。"""
        wrong = replace(TURN[0], pai="W")
        versions = [_version(wrong, wall=101.0), _version(*TURN, wall=110.0)]
        got = agreement([_truth(*TURN)], versions, _Engine)
        assert (got.turns_seen, got.turns_same) == (1, 0)
        assert got.differ

    def test_a_turn_that_never_came(self) -> None:
        got = agreement([_truth(*TURN)], [_version()], _Engine)
        assert (got.turns, got.turns_seen) == (1, 0)

    def test_an_engine_error_is_counted_and_replaced(self) -> None:
        made = []

        def make() -> _Engine:
            made.append(_Engine(fail_on="W" if not made else None))
            return made[-1]

        wrong = replace(TURN[0], pai="W")
        got = agreement([_truth(*TURN)], [_version(wrong)], make)
        assert got.errors == 1
        assert len(made) == 2
        assert made[0].closed


    def test_a_crash_at_the_end_of_the_kyoku_is_counted(self) -> None:
        """引擎崩潰可能到局尾才發作,實測報告因此整個中斷。"""
        wrong = (replace(TURN[0], pai="W"),)
        made: list[_Engine] = []

        class Crashes(_Engine):
            last: MjaiEvent | None = None

            def react(self, event: MjaiEvent) -> Advice:
                if isinstance(event, EndKyoku) and getattr(self.last, "pai", "") == "W":
                    raise EngineError("崩潰")
                self.last = event
                return super().react(event)

        def make() -> _Engine:
            made.append(Crashes())
            return made[-1]

        got = agreement([_truth(*TURN)], [_version(*wrong)], make)
        assert got.errors == 1
        assert len(made) == 2


class TestFiles:
    def test_a_recorded_game_loads(self) -> None:
        truth = load_truth(FIXTURES / "real_game_full.jsonl")
        assert truth
        assert all(isinstance(kyoku.events[0][1], StartKyoku) for kyoku in truth)
        walls = [wall for wall, _ in truth[0].events]
        assert walls == sorted(walls)

    def test_seats_are_turned_to_the_own_view(self) -> None:
        """封包是絕對座位,畫面是視角。自己的手牌一定要在第 0 家。"""
        for kyoku in load_truth(FIXTURES / "real_game_full.jsonl"):
            start = kyoku.events[0][1]
            assert isinstance(start, StartKyoku)
            assert "?" not in start.tehais[0]
            assert start.tehais[1][0] == "?"
            own = [e for _, e in kyoku.events if isinstance(e, Tsumo) and e.pai != "?"]
            assert all(event.actor == 0 for event in own)

    def test_the_run_log_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "vision.jsonl"
        rows = [
            {"kind": "stream", "wall": 1.0, "kyoku": 2, "note": "等手牌",
             "events": [e.to_dict() for e in (START, *TURN)]},
            {"kind": "advice", "wall": 2.0, "engine": "m", "action": None},
            {"kind": "status", "wall": 3.0, "name": "畫面", "message": "x"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        run = load_run(path)
        assert run.versions[0].events == (START, *TURN)
        assert (run.versions[0].kyoku, run.versions[0].note) == (2, "等手牌")
        assert len(run.advices) == 1
        assert len(run.statuses) == 1


def test_the_summary_names_the_numbers_that_matter() -> None:
    seen = (*TURN[:3], replace(TURN[3], pai="8m"))
    truth = [_truth(*TURN)]
    results = compare(truth, [_version(*seen)])
    text = summary(results, agreement(truth, [_version(*seen)], _Engine))
    assert "捨牌 1/2(錯 1" in text
    assert "東" in text or "E1局0本" in text
    assert "封包 9m / 畫面 8m" in text
