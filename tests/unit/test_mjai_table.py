"""整桌公開資訊的追蹤。

這一層錯掉的後果特別嚴重:放銃分析會把危險牌說成安全,而使用者會照著打。
所以除了單元測試,最後還拿**真實牌譜**重播 —— 那場裡真的有人榮和了三次,
那三張牌若被判成安全,就是違反振聽,必錯無疑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.groundtruth.stream import MjaiDecoder
from mia.mjai import Chi, Dahai, Kakan, Pon, Reach, StartGame, StartKyoku
from mia.mjai.table import TableTracker
from mia.mjai.tiles import normalize_red

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _start(dora: str = "1z") -> StartKyoku:
    return StartKyoku(
        bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0,
        dora_marker=dora, tehais=[["?"] * 13 for _ in range(4)],
        scores=[25000] * 4,
    )


def _tracker(seat: int = 0) -> TableTracker:
    tracker = TableTracker()
    tracker.handle(StartGame(id=seat))
    tracker.handle(_start())
    return tracker


class TestOwnDiscards:
    def test_a_players_own_discard_is_safe_against_them(self) -> None:
        """振聽:打過的牌自己不能榮和。整局有效,不是只有那一巡。"""
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        assert "3m" in t.players[1].safe

    def test_it_is_not_safe_against_anyone_else(self) -> None:
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        assert "3m" not in t.players[2].safe

    def test_a_called_tile_stays_in_the_river(self) -> None:
        """被鳴走的牌仍然算他打過 —— 振聽看的是「他打過」,不是那張牌現在在哪。"""
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        t.handle(Chi(actor=2, target=1, pai="3m", consumed=["1m", "2m"]))
        assert "3m" in t.players[1].safe
        assert t.players[1].river == ["3m"]

    def test_red_fives_count_as_plain_fives(self) -> None:
        """赤五與普通五在振聽上是同一種牌。不正規化就會漏判安全牌。"""
        t = _tracker()
        t.handle(Dahai(actor=1, pai="5mr", tsumogiri=False))
        assert "5m" in t.players[1].safe


class TestAfterReach:
    """立直之後手牌不能再變 —— 那一巡能和就一定和了,沒和就永遠不能和。

    這是安全牌的主要來源,而且**只對立直的人成立**。
    """

    def test_everything_discarded_after_a_reach_is_safe_against_the_reacher(self) -> None:
        t = _tracker()
        t.handle(Reach(actor=1))
        t.handle(Dahai(actor=1, pai="1z", tsumogiri=False))  # 宣言牌
        t.handle(Dahai(actor=2, pai="7p", tsumogiri=False))
        assert "7p" in t.players[1].safe

    def test_nothing_accumulates_for_a_player_who_has_not_reached(self) -> None:
        """沒立直的人可以隨時換聽,別人剛打過的牌對他完全不安全。
        把這兩件事混為一談是最常見的錯。"""
        t = _tracker()
        t.handle(Dahai(actor=2, pai="7p", tsumogiri=False))
        assert t.players[1].safe == set()

    def test_a_tile_discarded_before_the_reach_is_not_safe(self) -> None:
        """立直**之前**別人打過的牌對立直者不安全 —— 他當時可能還沒聽這張。"""
        t = _tracker()
        t.handle(Dahai(actor=2, pai="7p", tsumogiri=False))
        t.handle(Reach(actor=1))
        t.handle(Dahai(actor=1, pai="1z", tsumogiri=False))
        assert "7p" not in t.players[1].safe

    def test_the_declaration_tile_itself_is_safe(self) -> None:
        t = _tracker()
        t.handle(Reach(actor=1))
        t.handle(Dahai(actor=1, pai="1z", tsumogiri=False))
        assert "1z" in t.players[1].safe

    def test_a_reacher_is_a_threat_and_others_are_not(self) -> None:
        t = _tracker(seat=0)
        t.handle(Reach(actor=2))
        assert t.threats == [2]

    def test_you_are_never_a_threat_to_yourself(self) -> None:
        """不會放銃給自己。自己立直也不該出現在要防的名單裡。"""
        t = _tracker(seat=1)
        t.handle(Reach(actor=1))
        assert t.threats == []


class TestMeldsAreThreatsToo:
    """三副露也算威脅。實測那場最重的一次榮和,和牌的就是一個三副露、
    沒立直的人,而三副露的曝光量與立直是同一個量級。"""

    @staticmethod
    def _pon(t, seat: int, count: int):
        for pai in ("1z", "2z", "3z")[:count]:
            t.handle(Pon(actor=seat, target=(seat + 3) % 4, pai=pai, consumed=[pai, pai]))
        return t

    def test_three_melds_is_a_threat(self) -> None:
        t = self._pon(_tracker(), 2, 3)
        assert t.threats == [2]

    def test_two_melds_is_not(self) -> None:
        t = self._pon(_tracker(), 2, 2)
        assert t.threats == []

    def test_we_are_never_our_own_threat(self) -> None:
        """自己副露三組不會變成要防的對象 —— 不會放銃給自己。"""
        t = self._pon(_tracker(), 0, 3)
        assert t.threats == []



class TestNewHand:
    def test_a_new_kyoku_forgets_everything(self) -> None:
        """上一局的安全牌在這一局毫無意義 —— 留著不會有任何症狀,
        只會安靜地把危險牌說成安全。"""
        t = _tracker()
        t.handle(Reach(actor=1))
        t.handle(Dahai(actor=1, pai="1z", tsumogiri=False))
        t.handle(Dahai(actor=2, pai="7p", tsumogiri=False))
        t.handle(_start())
        assert t.players[1].safe == set()
        assert not t.players[1].reach
        assert t.players[1].river == []

    def test_the_new_dora_marker_replaces_the_old(self) -> None:
        t = _tracker()
        t.handle(_start(dora="9s"))
        assert t.dora_markers == ["9s"]


class TestVisibleTiles:
    def test_it_counts_rivers_melds_dora_and_your_own_hand(self) -> None:
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        t.handle(Pon(actor=2, target=1, pai="3m", consumed=["3m", "3m"]))
        seen = t.visible(["3m"])
        # 河 1(被鳴走的那張)+ 副露中從手上拿出的 2 + 自己手上 1 = 4
        assert seen["3m"] == 4

    def test_a_called_tile_is_not_counted_twice(self) -> None:
        """被鳴的那張已經在打者的河裡。兩邊都數會得到五張 3m ——
        而高估看得見的張數會讓還有活牌的危險牌被判成安全,方向是錯的。"""
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        t.handle(Pon(actor=2, target=1, pai="3m", consumed=["3m", "3m"]))
        assert t.visible()["3m"] == 3
        assert t.remaining("3m") == 1

    def test_an_added_kan_only_reveals_one_more(self) -> None:
        """加槓是在原本那組碰上多亮一張,不是新的一組四張。"""
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        t.handle(Pon(actor=2, target=1, pai="3m", consumed=["3m", "3m"]))
        t.handle(Kakan(actor=2, pai="3m", consumed=["3m", "3m", "3m"]))
        assert t.visible()["3m"] == 4
        assert len(t.players[2].melds) == 1

    def test_remaining_never_goes_negative(self) -> None:
        t = _tracker()
        t.handle(Dahai(actor=1, pai="3m", tsumogiri=False))
        assert t.remaining("3m", ["3m", "3m", "3m", "3m"]) == 0

    def test_an_unseen_tile_has_all_four(self) -> None:
        assert _tracker().remaining("6p") == 4


class TestAgainstARealGame:
    """一整場東風戰,550 個事件、259 張捨牌、4 次立直、3 次榮和。"""

    @pytest.fixture(scope="class")
    @classmethod
    def replay(cls) -> list:
        path = FIXTURES / "real_game_full.jsonl"
        return [e for d in MjaiDecoder().decode_file(path) for e in d.events]

    def test_no_deal_in_tile_was_ever_called_safe(self, replay: list) -> None:
        """**這一組測試的重點。**

        在每一張牌被打出去**之前**問「它對每一家安全嗎」—— 那正是實戰時會問
        的時點。之後若有人榮和了它,而我說過它安全,那就是違反振聽。
        """
        tracker = TableTracker()
        pending: tuple[str, list[bool]] | None = None
        rons = 0

        for event in replay:
            if event.TYPE == "hora" and event.target != event.actor and pending:
                rons += 1
                _pai, said_safe = pending
                assert not said_safe[event.actor], (
                    f"{event.actor} 家榮和了 {_pai},而它被判定為對他安全 —— 違反振聽"
                )
            if event.TYPE == "dahai":
                pai = normalize_red(event.pai)
                pending = (pai, [pai in p.safe for p in tracker.players])
            tracker.handle(event)

        assert rons == 3, f"這場應該有 3 次榮和,只看到 {rons}"

    def test_it_actually_finds_safe_tiles(self, replay: list) -> None:
        """反面:上一條測試只要「永遠說不安全」就會過。這條確認它真的有在
        判斷,不是一路回答不知道。"""
        tracker = TableTracker()
        flagged = 0
        for event in replay:
            if event.TYPE == "dahai" and tracker.threats:
                pai = normalize_red(event.pai)
                flagged += any(pai in tracker.players[i].safe for i in tracker.threats)
            tracker.handle(event)
        assert flagged > 0, "整場沒有認出任何一張安全牌"

    def test_the_seat_comes_from_start_game(self, replay: list) -> None:
        tracker = TableTracker()
        for event in replay:
            tracker.handle(event)
        assert tracker.seat == 2
