"""放銃危險度:排除法。

輸出的不是機率而是「還有幾種待牌型沒被排除」。所以測試驗的也是排除:
**哪一型為什麼不見了**,而不是某個數字對不對。

最後拿真實牌譜收尾 —— 那場裡我們自己放銃過一次,那張牌不能被說成安全。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.analysis.danger import DangerLevel, _worst, assess
from mia.groundtruth.stream import MjaiDecoder
from mia.mjai import Dahai, Pon, Reach, StartGame, StartKyoku
from mia.mjai.table import TableTracker
from mia.mjai.tiles import normalize_red

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _table(seat: int = 0) -> TableTracker:
    t = TableTracker()
    t.handle(StartGame(id=seat))
    t.handle(
        StartKyoku(
            bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0, dora_marker="1z",
            tehais=[["?"] * 13 for _ in range(4)], scores=[25000] * 4,
        )
    )
    return t


def _meld(table: TableTracker, seat: int, count: int) -> TableTracker:
    """讓某一家碰到 ``count`` 組。用字牌,免得順手動到數牌的壁與筋。"""
    for pai in ("1z", "2z", "3z", "4z")[:count]:
        table.handle(Pon(actor=seat, target=(seat + 3) % 4, pai=pai, consumed=[pai, pai]))
    return table


def _danger(table: TableTracker, tile: str):
    report = assess([tile], table)
    return next(t for t in report.tiles if t.tile == normalize_red(tile))


def _kinds(table: TableTracker, tile: str, seat: int) -> set[str]:
    seats = _danger(table, tile).seats
    return {str(w) for w in next(s for s in seats if s.seat == seat).waits}


class TestFuriten:
    """打過的牌不能榮和。這一條是規則,不是估計 —— 對**所有人**成立,
    不是只對立直的人。"""

    def test_a_tile_they_discarded_has_no_waits_left(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        seat = next(s for s in _danger(t, "3m").seats if s.seat == 1)
        assert seat.waits == ()
        assert seat.furiten
        assert seat.level is DangerLevel.SAFE

    def test_it_says_why(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        assert "現物" in next(s for s in _danger(t, "3m").seats if s.seat == 1).reason

    def test_furiten_against_one_seat_is_not_safe_overall(self) -> None:
        """**實測踩過的坑。** 真實牌譜裡有一張牌對立直的 0 家是現物,
        而榮和的是沒立直的 3 家。只看其中一家就寫「安全」是在騙人。
        """
        t = _table()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        assert _danger(t, "3m").level is not DangerLevel.SAFE

    def test_safe_means_safe_against_everyone(self) -> None:
        t = _table()
        for seat in (1, 2, 3):
            t.handle(Dahai(actor=seat, pai="3m", tsumogiri=False))
        assert _danger(t, "3m").level is DangerLevel.SAFE


class TestSuji:
    """筋不是另外寫的一條規則,是排除法自然掉出來的:那一型也會和另一張,
    而他打過那一張 —— 他若真有那一型早就振聽了。"""

    def test_discarding_the_partner_kills_the_ryanmen(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="4m", tsumogiri=False))  # 1m 的筋
        assert "両面(2m3m)" not in _kinds(t, "1m", 1)

    def test_the_other_side_too(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="4m", tsumogiri=False))  # 7m 的筋
        assert "両面(5m6m)" not in _kinds(t, "7m", 1)

    def test_a_kanchan_is_not_suji(self) -> None:
        """嵌張只和這一張,沒有「另一張」可以構成振聽 —— 筋殺不掉它。"""
        t = _table()
        t.handle(Dahai(actor=1, pai="1m", tsumogiri=False))
        t.handle(Dahai(actor=1, pai="7m", tsumogiri=False))
        assert "嵌張(3m5m)" in _kinds(t, "4m", 1)

    def test_suji_works_without_a_reach(self) -> None:
        """振聽與聽不聽牌無關 —— 他若現在聽那一型,就同時聽著自己打過的牌。"""
        t = _table()
        t.handle(Dahai(actor=1, pai="4m", tsumogiri=False))
        assert not t.players[1].reach
        assert "両面(2m3m)" not in _kinds(t, "1m", 1)


class TestWall:
    """四張全見的牌對方不可能持有,需要它的那一型就不成立。"""

    def test_four_visible_kills_the_shapes_that_need_it(self) -> None:
        t = _table()
        for seat in (1, 2, 3):
            t.handle(Dahai(actor=seat, pai="6m", tsumogiri=False))
        # 第四張在自己手上
        report = assess(["4m", "6m"], t)
        kinds = {str(w) for w in next(
            d for d in report.tiles if d.tile == "4m"
        ).seats[0].waits}
        assert "両面(5m6m)" not in kinds
        assert "両面(2m3m)" in kinds  # 另一邊沒有被擋

    def test_an_edge_wait_is_labelled_penchan(self) -> None:
        """(8m9m) 只和 7m,沒有另一張可以構成筋 —— 型的名字要分得開。"""
        assert "辺張(8m9m)" in _kinds(_table(), "7m", 1)


class TestHonours:
    """字牌沒有順子,只剩単騎與雙碰 —— 所以它天生比數牌安全,
    而理由是「能打中它的型比較少」,不是統計印象。"""

    def test_an_untouched_honour_has_only_two_shapes(self) -> None:
        assert _kinds(_table(), "1z", 1) == {"単騎(東)", "雙碰(東東)"}

    def test_three_visible_leaves_only_tanki(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="2z", tsumogiri=False))
        t.handle(Dahai(actor=2, pai="2z", tsumogiri=False))
        assert _kinds(t, "2z", 3) == {"単騎(南)"}

    def test_four_visible_is_genuinely_safe(self) -> None:
        """四張全見的字牌不可能被和 —— 単騎要他手上有一張,雙碰要兩張,
        而一張都不剩。這與現物一樣是規則保證的。"""
        t = _table()
        for seat in (1, 2, 3):
            t.handle(Dahai(actor=seat, pai="3z", tsumogiri=False))
        assert _danger(t, "3z").level is DangerLevel.SAFE


class TestReachIsSharperNotBroader:
    def test_reach_seats_are_reported(self) -> None:
        t = _table()
        t.handle(Reach(actor=2))
        t.handle(Dahai(actor=2, pai="1z", tsumogiri=False))
        threats = assess(["3m"], t).threats
        assert [(x.seat, x.label) for x in threats] == [(2, "立直")]

    def test_against_threats_only_looks_at_the_named(self) -> None:
        """對立直家是現物、對別家全新 —— 兩個數字要分得開,
        因為「對立直家安全」與「對誰都安全」是很不一樣的處境。"""
        t = _table()
        t.handle(Reach(actor=1))
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        danger = _danger(t, "3m")
        assert danger.against_threats is DangerLevel.SAFE
        assert danger.level is not DangerLevel.SAFE


class TestMeldsCountAsThreats:
    """三副露與立直平起平坐 —— **在指名這件事上**,不在等級上。

    實測那場最重的一次榮和,和牌的是一個三副露、沒立直的人;而三副露的
    曝光量與立直是同一個量級(53 對 59 個捨牌時點)。
    """

    def test_three_melds_gets_named(self) -> None:
        t = _meld(_table(), seat=2, count=3)
        assert [(x.seat, x.label) for x in assess(["3m"], t).threats] == [(2, "3副露")]

    def test_two_melds_does_not(self) -> None:
        """門檻在三組。兩組還在「可能只是想做個役」的範圍。"""
        t = _meld(_table(), seat=2, count=2)
        assert assess(["3m"], t).threats == ()

    def test_melds_do_not_change_the_level(self) -> None:
        """**這一條是這個設計的界線。** 等級由排除法數出來,副露是估計 ——
        混進去的話畫面上會出現「非常危險」配著「還有 2 型」,而「每一條都能
        指回它憑什麼」是這個模組唯一的賣點。
        """
        plain = _danger(_table(), "3m").level
        melded = _danger(_meld(_table(), seat=2, count=3), "3m").level
        assert melded is plain

    def test_it_sharpens_against_threats(self) -> None:
        """三副露的家要算進那個銳利指標。只認立直會把他整個漏掉。"""
        t = _meld(_table(), seat=2, count=3)
        assert _danger(t, "3m").against_threats is not DangerLevel.SAFE

    def test_a_reach_outranks_melds_when_both_apply(self) -> None:
        """兩者都成立時寫立直 —— 確定聽牌比推定聽牌硬。"""
        t = _meld(_table(), seat=2, count=3)
        t.handle(Reach(actor=2))
        assert assess(["3m"], t).threats[0].label == "立直"

    def test_the_melded_seat_is_the_one_worth_naming(self) -> None:
        """平手時要講露出馬腳的那個,不是座位編號最小的那個 —— 三家的待牌型
        數一樣時,``max`` 預設會回 1 家,而那只是編號順序。"""
        t = _meld(_table(), seat=3, count=3)
        assert _worst(_danger(t, "3m").seats).seat == 3


class TestNoReach:
    def test_no_reach_still_produces_a_report(self) -> None:
        """沒有人立直**不代表安全** —— 實測三次榮和裡兩次是沒立直的人和的。"""
        assert assess(["3m"], _table()).tiles


class TestReportShape:
    def test_tiles_are_sorted_safest_first(self) -> None:
        t = _table()
        t.handle(Dahai(actor=1, pai="9p", tsumogiri=False))
        t.handle(Dahai(actor=2, pai="9p", tsumogiri=False))
        t.handle(Dahai(actor=3, pai="9p", tsumogiri=False))
        levels = [d.level for d in assess(["5m", "9p", "1z"], t).tiles]
        assert levels == sorted(levels)

    def test_an_empty_hand_gives_nothing(self) -> None:
        assert not assess([], _table())

    def test_red_fives_collapse_onto_plain_fives(self) -> None:
        report = assess(["5mr", "5m"], _table())
        assert [d.tile for d in report.tiles] == ["5m"]


class TestAgainstARealGame:
    @pytest.fixture(scope="class")
    @classmethod
    def replay(cls) -> list:
        path = FIXTURES / "real_game_full.jsonl"
        return [e for d in MjaiDecoder().decode_file(path) for e in d.events]

    def _walk(self, replay: list):
        """重播,並在每張牌打出去**之前**評估它。"""
        table = TableTracker()
        pending = None
        for event in replay:
            if event.TYPE == "hora" and event.target != event.actor and pending:
                yield ("hora", event, pending)
            if event.TYPE == "start_kyoku":
                pending = None
            if event.TYPE == "dahai":
                pai = normalize_red(event.pai)
                got = next(
                    (d for d in assess([pai], table).tiles if d.tile == pai), None
                )
                if got is not None:
                    pending = (event.actor, pai, got)
                    yield ("dahai", event, pending)
            table.handle(event)

    def test_no_tile_called_safe_was_ever_ronned(self, replay: list) -> None:
        """**這一組的重點。** 說「安全」就是說「三家都不可能榮和」——
        那是振聽與剩餘張數保證的,不是估計。錯一次就是這個模組沒有價值。
        """
        rons = 0
        for kind, event, (actor, pai, danger) in self._walk(replay):
            if kind != "hora":
                continue
            rons += 1
            assert danger.level is not DangerLevel.SAFE, (
                f"{actor} 家打的 {pai} 被 {event.actor} 家榮和,"
                f"而它被評成「安全」:{danger.reason}"
            )
        assert rons == 3

    def test_our_own_deal_in_was_flagged_with_its_reason(self, replay: list) -> None:
        """那一場我們自己放銃過一次(8s)。它被收斂到只剩一型,而那一型
        指名了對方真正的待牌 —— 這正是這個做法該有的樣子:不說安全,
        說還剩什麼。
        """
        found = []
        for kind, event, (_actor, pai, danger) in self._walk(replay):
            if kind == "hora" and event.target == 2:  # 座位 2 是自己
                found.append((pai, danger, event.actor))
        assert found, "這場我們應該放銃過一次"
        pai, danger, winner = found[0]
        assert pai == "8s"
        assert danger.level is DangerLevel.LIKELY_SAFE
        seat = next(s for s in danger.seats if s.seat == winner)
        assert "6s7s" in seat.reason

    def test_it_does_find_genuinely_safe_tiles(self, replay: list) -> None:
        """反面:只要一路回答「危險」,上面那條也會過。"""
        safe = sum(
            1
            for kind, _e, (_a, _p, d) in self._walk(replay)
            if kind == "dahai" and d.level is DangerLevel.SAFE
        )
        assert safe > 0
