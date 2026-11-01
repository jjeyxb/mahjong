"""事件的中文說明。"""

from __future__ import annotations

import pytest

from mia.mjai.describe import describe
from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Hora,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)


def _kyoku() -> StartKyoku:
    return StartKyoku(
        bakaze="S",
        kyoku=2,
        honba=1,
        kyotaku=2,
        oya=3,
        dora_marker="5pr",
        tehais=[["1m", "E"], ["?"], ["P", "9s"], ["?"]],
        scores=[25000, 24000, 26000, 23000],
    )


class TestSeats:
    def test_seats_are_relative_to_me(self) -> None:
        """封包的 ``actor`` 是絕對座位。我坐 2,0 號就是對家、3 號是下家。"""
        assert describe(Reach(actor=2), seat=2) == "自家 宣告立直"
        assert describe(Reach(actor=3), seat=2) == "下家 宣告立直"
        assert describe(Reach(actor=0), seat=2) == "對家 宣告立直"
        assert describe(Reach(actor=1), seat=2) == "上家 宣告立直"

    def test_an_unknown_seat_uses_the_number(self) -> None:
        assert describe(Reach(actor=2)) == "2 家 宣告立直"


class TestTiles:
    @pytest.mark.parametrize(
        ("pai", "name"),
        [("3m", "三萬"), ("9p", "九筒"), ("1s", "一索"), ("5sr", "赤五索"), ("E", "東"),
         ("P", "白"), ("F", "發"), ("C", "中")],
    )
    def test_names(self, pai: str, name: str) -> None:
        assert describe(Dora(dora_marker=pai)).endswith(name)

    def test_a_tile_we_do_not_know_is_kept(self) -> None:
        assert describe(Dora(dora_marker="xx")).endswith("xx")


class TestDraws:
    def test_my_draw_names_the_tile(self) -> None:
        assert describe(Tsumo(actor=0, pai="3p"), seat=0) == "自家 摸 三筒"

    def test_a_hidden_draw_does_not(self) -> None:
        assert describe(Tsumo(actor=1, pai="?"), seat=0) == "下家 摸牌"


class TestDiscards:
    def test_packets_know_how_it_was_thrown(self) -> None:
        assert describe(Dahai(actor=1, pai="E", tsumogiri=True), seat=0) == "下家 摸切 東"
        assert describe(Dahai(actor=1, pai="E", tsumogiri=False), seat=0) == "下家 手切 東"

    def test_the_screen_does_not_know_for_other_seats(self) -> None:
        """畫面分不出來,一律填 False。照實寫成手切就是在說謊。"""
        event = Dahai(actor=1, pai="E", tsumogiri=False)
        assert describe(event, seat=0, exact_discards=False) == "下家 打 東"

    def test_the_screen_knows_for_me(self) -> None:
        event = Dahai(actor=0, pai="E", tsumogiri=True)
        assert describe(event, seat=0, exact_discards=False) == "自家 摸切 東"


class TestCalls:
    def test_chi(self) -> None:
        event = Chi(actor=1, target=0, pai="3s", consumed=["4s", "5sr"])
        assert describe(event, seat=0) == "下家 吃 自家的 三索,用 四索、赤五索"

    def test_pon(self) -> None:
        event = Pon(actor=2, target=3, pai="E", consumed=["E", "E"])
        assert describe(event, seat=0) == "對家 碰 上家的 東,用 東、東"

    def test_an_open_kan(self) -> None:
        event = Daiminkan(actor=3, target=1, pai="9m", consumed=["9m", "9m", "9m"])
        assert describe(event, seat=0) == "上家 大明槓 下家的 九萬,用 九萬、九萬、九萬"

    def test_a_closed_kan(self) -> None:
        assert describe(Ankan(actor=1, consumed=["C"] * 4), seat=0) == "下家 暗槓 中 中 中 中"

    def test_an_added_kan(self) -> None:
        event = Kakan(actor=1, pai="E", consumed=["E", "E", "E"])
        assert describe(event, seat=0) == "下家 加槓 東"


class TestRounds:
    def test_a_round_opens_with_everything_mortal_gets(self) -> None:
        text = describe(_kyoku(), seat=2)
        assert text.startswith("南二局 1 本場,供託 2,莊家是下家,寶牌指示牌 赤五筒")
        assert "點數 自家 26000、下家 23000、對家 25000、上家 24000" in text
        assert text.endswith("配牌 白 九索")

    def test_without_a_seat_there_is_no_hand_to_show(self) -> None:
        assert "配牌" not in describe(_kyoku())

    def test_a_ron_and_its_points(self) -> None:
        event = Hora(actor=1, target=0, deltas=[-3900, 3900, 0, 0])
        assert describe(event, seat=0) == "下家 榮和 自家,自家 -3900、下家 +3900"

    def test_a_tsumo(self) -> None:
        assert describe(Hora(actor=1, target=1), seat=0) == "下家 自摸"

    def test_an_exhaustive_draw(self) -> None:
        assert describe(Ryukyoku(deltas=[1500, -1500, 0, 0]), seat=0).startswith("流局,")

    @pytest.mark.parametrize(
        "event",
        [
            StartGame(id=1),
            EndGame(),
            EndKyoku(),
            ReachAccepted(actor=1),
            Kita(actor=1),
            Ryukyoku(),
        ],
    )
    def test_nothing_falls_through_to_the_raw_type(self, event: MjaiEvent) -> None:
        assert describe(event, seat=0) != event.TYPE
