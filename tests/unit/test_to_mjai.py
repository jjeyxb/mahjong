"""liqi → MJAI 狀態機測試。

重點放在協定文件明確點名「其他實作會做錯」的幾處:
寶牌翻牌時機、``reach_accepted`` 的插入位置、榮和放銃者的判定。
"""

from __future__ import annotations

from typing import Any

import pytest

from mia.groundtruth.liqi import LiqiParser
from mia.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema
from mia.groundtruth.to_mjai import DoraTiming, MajsoulToMjai
from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndKyoku,
    Hora,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartKyoku,
    Tsumo,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_LIQI_PATH.is_file(), reason="需要 assets/proto/liqi.json"
)


@pytest.fixture(scope="module")
def schema() -> LiqiSchema:
    return LiqiSchema.load()


@pytest.fixture
def conv(schema: LiqiSchema) -> MajsoulToMjai:
    """已完成 authGame、自己坐 0 號位的四人局。"""
    converter = MajsoulToMjai(LiqiParser(schema))
    converter._account_id = 1001  # noqa: SLF001 - 免去組一則 authGame 請求
    res = schema.message_class("lq.ResAuthGame")(
        seat_list=[1001, 1002, 1003, 1004],
        players=[
            schema.message_class("lq.PlayerGameView")(account_id=1001, nickname="我"),
            schema.message_class("lq.PlayerGameView")(account_id=1002, nickname="下家"),
            schema.message_class("lq.PlayerGameView")(account_id=1003, nickname="對家"),
            schema.message_class("lq.PlayerGameView")(account_id=1004, nickname="上家"),
        ],
    )
    converter._on_auth_game(res)  # noqa: SLF001
    return converter


def act(conv: MajsoulToMjai, name: str, message: Any) -> list[MjaiEvent]:
    """直接餵一個動作,跳過線路層。"""
    return conv._on_action(f".lq.{name}", message)  # noqa: SLF001


def new_round(schema: LiqiSchema, *, oya: int = 0, tiles: list[str] | None = None) -> Any:
    return schema.message_class("lq.ActionNewRound")(
        chang=0, ju=oya, ben=0, liqibang=0,
        tiles=tiles if tiles is not None else ["1m"] * 13,
        doras=["3p"], scores=[25000] * 4,
    )


def types_of(events: list[MjaiEvent]) -> list[str]:
    return [e.TYPE for e in events]


class TestStartGame:
    def test_seat_and_names_from_auth_game(self, conv: MajsoulToMjai) -> None:
        assert conv.seat == 0
        assert conv.state.num_players == 4
        assert conv.state.names == ["我", "下家", "對家", "上家"]

    def test_three_player_table(self, schema: LiqiSchema) -> None:
        """三麻的 seat_list 長度是 3,不可補到 4。"""
        converter = MajsoulToMjai(LiqiParser(schema))
        converter._account_id = 7  # noqa: SLF001
        res = schema.message_class("lq.ResAuthGame")(seat_list=[5, 7, 9])
        events = converter._on_auth_game(res)  # noqa: SLF001

        assert converter.state.num_players == 3
        assert converter.seat == 1
        assert len(events[0].names) == 3


class TestNewRound:
    def test_dealer_hand_is_split_into_13_plus_tsumo(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """雀魂發莊家 14 張;MJAI 是 13 張 + 一個明確的 tsumo。"""
        hand = [f"{n}m" for n in range(1, 10)] + ["1p", "2p", "3p", "4p", "9s"]
        events = act(conv, "ActionNewRound", new_round(schema, oya=0, tiles=hand))

        assert types_of(events) == ["start_kyoku", "tsumo"]
        start: StartKyoku = events[0]
        assert len(start.tehais[0]) == 13
        assert events[1].actor == 0
        assert events[1].pai == "9s", "第 14 張才是莊家摸到的牌"

    def test_non_dealer_sees_unknown_dealer_draw(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        events = act(conv, "ActionNewRound", new_round(schema, oya=2, tiles=["1m"] * 13))

        start: StartKyoku = events[0]
        assert start.oya == 2
        assert start.kyoku == 3
        assert len(start.tehais[0]) == 13
        assert events[1] == Tsumo(actor=2, pai="?"), "看不到別人摸什麼"

    def test_other_hands_are_hidden(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        start: StartKyoku = act(conv, "ActionNewRound", new_round(schema))[0]
        for seat in (1, 2, 3):
            assert start.tehais[seat] == ["?"] * 13

    def test_round_fields(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        action = schema.message_class("lq.ActionNewRound")(
            chang=1, ju=3, ben=2, liqibang=1, tiles=["1m"] * 13,
            doras=["0s"], scores=[24000, 25000, 26000, 25000],
        )
        start: StartKyoku = act(conv, "ActionNewRound", action)[0]

        assert (start.bakaze, start.kyoku, start.honba, start.kyotaku) == ("S", 4, 2, 1)
        assert start.dora_marker == "5sr"
        assert start.scores == [24000, 25000, 26000, 25000]

    def test_resets_state_from_previous_kyoku(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        conv.state.pending_reach = 2
        conv.state.deferred_dora = "1m"
        conv.state.last_revealed_by = 3

        act(conv, "ActionNewRound", new_round(schema))

        assert conv.state.pending_reach is None
        assert conv.state.deferred_dora is None
        assert conv.state.last_revealed_by is None


class TestDiscardAndTsumo:
    def test_own_draw_is_visible(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionDealTile", schema.message_class("lq.ActionDealTile")(
            seat=0, tile="0m"))
        assert events == [Tsumo(actor=0, pai="5mr")]

    def test_other_draw_is_hidden(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionDealTile", schema.message_class("lq.ActionDealTile")(
            seat=2, tile="3p"))
        assert events == [Tsumo(actor=2, pai="?")], "不該洩漏別人摸到的牌"

    def test_discard(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=1, tile="7s", moqie=True))
        assert events == [Dahai(actor=1, pai="7s", tsumogiri=True)]

    def test_discard_without_tile_is_skipped(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(
            conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(seat=1)
        )
        assert events == [], "缺牌的切牌是壞資料,略過勝於送出錯的事件"


class TestReach:
    def _reach_discard(self, schema: LiqiSchema, seat: int) -> Any:
        return schema.message_class("lq.ActionDiscardTile")(seat=seat, tile="1z", is_liqi=True)

    def test_declaration_emits_reach_then_dahai(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionDiscardTile", self._reach_discard(schema, 1))

        assert events == [Reach(actor=1), Dahai(actor=1, pai="E", tsumogiri=False)]
        assert conv.state.pending_reach == 1, "此時還不能記點"

    def test_accepted_is_prepended_to_the_next_action(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """宣言牌安全通過後才扣點,且要排在下一個動作的事件之前。"""
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", self._reach_discard(schema, 1))

        events = act(conv, "ActionDealTile", schema.message_class("lq.ActionDealTile")(seat=2))

        assert events == [ReachAccepted(actor=1), Tsumo(actor=2, pai="?")]
        assert conv.state.pending_reach is None

    def test_double_riichi_flag_also_counts(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=3, tile="1z", is_wliqi=True))
        assert types_of(events) == ["reach", "dahai"]

    def test_ron_on_declaration_tile_voids_the_riichi(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """被榮和則立直不成立,不可發 reach_accepted。"""
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", self._reach_discard(schema, 1))

        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=2, zimo=False)],
            delta_scores=[0, -8000, 8000, 0],
        )
        events = act(conv, "ActionHule", hule)

        assert "reach_accepted" not in types_of(events)
        assert types_of(events) == ["hora", "end_kyoku"]
        assert conv.state.pending_reach is None

    def test_drained_before_ryukyoku(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        """最後一張切牌宣言立直又流局:宣言牌安全通過,立直成立。"""
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", self._reach_discard(schema, 1))

        events = act(conv, "ActionNoTile", schema.message_class("lq.ActionNoTile")())

        assert types_of(events) == ["reach_accepted", "ryukyoku", "end_kyoku"]


class TestMelds:
    def test_chi(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionChiPengGang")(
            seat=1, type=0, tiles=["3m", "4m", "5m"], froms=[0, 1, 1])
        assert act(conv, "ActionChiPengGang", action) == [
            Chi(actor=1, target=0, pai="3m", consumed=["4m", "5m"])
        ]

    def test_pon(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionChiPengGang")(
            seat=2, type=1, tiles=["7p", "7p", "7p"], froms=[2, 2, 3])
        assert act(conv, "ActionChiPengGang", action) == [
            Pon(actor=2, target=3, pai="7p", consumed=["7p", "7p"])
        ]

    def test_daiminkan_defers_dora(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionChiPengGang")(
            seat=1, type=2, tiles=["2s"] * 4, froms=[1, 1, 1, 0])
        events = act(conv, "ActionChiPengGang", action)

        assert events == [Daiminkan(actor=1, target=0, pai="2s", consumed=["2s", "2s", "2s"])]
        assert conv.state.dora_timing is DoraTiming.AFTER_RINSHAN

    def test_chi_suppressed_in_three_player(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        conv.state.num_players = 3
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionChiPengGang")(
            seat=1, type=0, tiles=["3m", "4m", "5m"], froms=[0, 1, 1])
        assert act(conv, "ActionChiPengGang", action) == [], "三麻沒有吃"

    def test_ankan_puts_red_five_first(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        """五的四枚必然包含唯一的赤五,MJAI 慣例把它排在索引 0。"""
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionAnGangAddGang")(seat=0, type=3, tiles="5p")
        events = act(conv, "ActionAnGangAddGang", action)

        assert events == [Ankan(actor=0, consumed=["5pr", "5p", "5p", "5p"])]
        assert conv.state.dora_timing is DoraTiming.BEFORE_RINSHAN

    def test_ankan_of_non_five(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionAnGangAddGang")(seat=0, type=3, tiles="1z")
        assert act(conv, "ActionAnGangAddGang", action) == [
            Ankan(actor=0, consumed=["E", "E", "E", "E"])
        ]

    def test_kakan_with_normal_five_means_pon_held_the_red(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionAnGangAddGang")(seat=1, type=2, tiles="5s")
        assert act(conv, "ActionAnGangAddGang", action) == [
            Kakan(actor=1, pai="5s", consumed=["5sr", "5s", "5s"])
        ]

    def test_kakan_with_red_five_means_pon_was_all_normal(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        action = schema.message_class("lq.ActionAnGangAddGang")(seat=1, type=2, tiles="0s")
        assert act(conv, "ActionAnGangAddGang", action) == [
            Kakan(actor=1, pai="5sr", consumed=["5s", "5s", "5s"])
        ]


class TestDoraTiming:
    """暗槓是即乗り,加槓與大明槓是後乗り。"""

    def _rinshan(self, schema: LiqiSchema, seat: int, doras: list[str]) -> Any:
        return schema.message_class("lq.ActionDealTile")(seat=seat, tile="9m", doras=doras)

    def test_ankan_flips_dora_before_the_rinshan_draw(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))  # doras=["3p"]
        act(conv, "ActionAnGangAddGang",
            schema.message_class("lq.ActionAnGangAddGang")(seat=0, type=3, tiles="1m"))

        events = act(conv, "ActionDealTile", self._rinshan(schema, 0, ["3p", "7z"]))

        assert events == [Dora(dora_marker="C"), Tsumo(actor=0, pai="9m")]

    def test_kakan_flips_dora_after_the_rinshan_draw(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionAnGangAddGang",
            schema.message_class("lq.ActionAnGangAddGang")(seat=0, type=2, tiles="1m"))

        after_draw = act(conv, "ActionDealTile", self._rinshan(schema, 0, ["3p", "7z"]))
        assert after_draw == [Tsumo(actor=0, pai="9m")], "加槓時摸牌之前不可翻寶牌"

        after_discard = act(conv, "ActionDiscardTile",
                            schema.message_class("lq.ActionDiscardTile")(seat=0, tile="9m"))
        assert after_discard == [
            Dora(dora_marker="C"),
            Dahai(actor=0, pai="9m", tsumogiri=False),
        ], "寶牌要在打牌之前才翻"

    def test_daiminkan_also_flips_after(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionChiPengGang", schema.message_class("lq.ActionChiPengGang")(
            seat=0, type=2, tiles=["2s"] * 4, froms=[0, 0, 0, 1]))

        assert act(conv, "ActionDealTile", self._rinshan(schema, 0, ["3p", "1m"])) == [
            Tsumo(actor=0, pai="9m")
        ]
        assert conv.state.deferred_dora == "1m"

    def test_repeated_dora_list_is_not_re_emitted(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """雀魂在多個動作上重複帶完整的 doras 清單,不能每次都當成新的。"""
        act(conv, "ActionNewRound", new_round(schema))  # doras=["3p"]
        first = act(conv, "ActionDealTile", self._rinshan(schema, 1, ["3p"]))
        assert types_of(first) == ["tsumo"], "既有的寶牌不該再發一次"

    def test_kita_does_not_flip_dora(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        conv.state.num_players = 3
        act(conv, "ActionNewRound", new_round(schema))
        events = act(conv, "ActionBaBei", schema.message_class("lq.ActionBaBei")(seat=1))

        assert events == [Kita(actor=1, pai="N")]
        assert conv.state.dora_timing is DoraTiming.NONE


class TestHora:
    def test_tsumo_targets_self(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=2, zimo=True)],
            delta_scores=[-2000, -1000, 4000, -1000])
        events = act(conv, "ActionHule", hule)

        assert events[0] == Hora(
            actor=2, target=2, deltas=[-2000, -1000, 4000, -1000]
        )
        assert types_of(events) == ["hora", "end_kyoku"]

    def test_ron_targets_the_discarder(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=3, tile="5m"))

        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=0, zimo=False)],
            delta_scores=[8000, 0, 0, -8000])
        assert act(conv, "ActionHule", hule)[0].target == 3

    def test_chankan_targets_the_kan_declarer_not_the_last_discarder(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """搶槓:放銃的是加槓的人,不是上一個切牌的人。只看切牌會判錯。"""
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=1, tile="9p"))
        act(conv, "ActionAnGangAddGang", schema.message_class("lq.ActionAnGangAddGang")(
            seat=2, type=2, tiles="3m"))

        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=0, zimo=False)])
        assert act(conv, "ActionHule", hule)[0].target == 2

    def test_kokushi_robbing_ankan(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=1, tile="9p"))
        act(conv, "ActionAnGangAddGang", schema.message_class("lq.ActionAnGangAddGang")(
            seat=3, type=3, tiles="1z"))

        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=0, zimo=False)])
        assert act(conv, "ActionHule", hule)[0].target == 3

    def test_ron_on_kita(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        conv.state.num_players = 3
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionBaBei", schema.message_class("lq.ActionBaBei")(seat=1))

        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=0, zimo=False)])
        assert act(conv, "ActionHule", hule)[0].target == 1

    def test_ron_without_any_reveal_is_skipped(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        hule = schema.message_class("lq.ActionHule")(
            hules=[schema.message_class("lq.HuleInfo")(seat=0, zimo=False)])
        events = act(conv, "ActionHule", hule)

        assert types_of(events) == ["end_kyoku"], "判不出放銃者時寧可不發 hora"

    def test_ura_markers_distinguish_riichi_from_no_riichi(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """立直但無裏寶 → 空 list;沒立直 → None。兩者語意不同。"""
        act(conv, "ActionNewRound", new_round(schema))
        hule = schema.message_class("lq.ActionHule")(hules=[
            schema.message_class("lq.HuleInfo")(seat=0, zimo=True, liqi=True, li_doras=["0m"]),
            schema.message_class("lq.HuleInfo")(seat=1, zimo=True, liqi=True),
            schema.message_class("lq.HuleInfo")(seat=2, zimo=True, liqi=False),
        ])
        events = act(conv, "ActionHule", hule)

        assert events[0].ura_markers == ["5mr"]
        assert events[1].ura_markers == []
        assert events[2].ura_markers is None

    def test_multi_ron_emits_one_hora_each(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=3, tile="5m"))
        hule = schema.message_class("lq.ActionHule")(hules=[
            schema.message_class("lq.HuleInfo")(seat=0, zimo=False),
            schema.message_class("lq.HuleInfo")(seat=1, zimo=False),
        ])
        assert types_of(act(conv, "ActionHule", hule)) == ["hora", "hora", "end_kyoku"]


class TestRyukyoku:
    def test_no_tile_sums_delta_records(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        """ActionNoTile.scores 是多筆點數變動紀錄,要依座位加總。"""
        act(conv, "ActionNewRound", new_round(schema))
        info = schema.message_class("lq.NoTileScoreInfo")
        action = schema.message_class("lq.ActionNoTile")(scores=[
            info(delta_scores=[1000, -1000, 0, 0]),
            info(delta_scores=[500, 0, -500, 0]),
        ])
        events = act(conv, "ActionNoTile", action)

        assert events == [Ryukyoku(deltas=[1500, -1000, -500, 0]), EndKyoku()]

    def test_no_tile_without_scores(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        act(conv, "ActionNewRound", new_round(schema))
        assert act(conv, "ActionNoTile", schema.message_class("lq.ActionNoTile")()) == [
            Ryukyoku(deltas=None), EndKyoku()
        ]

    def test_liuju_has_no_deltas(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        """途中流局(九種九牌等)沒有點數變動。"""
        act(conv, "ActionNewRound", new_round(schema))
        assert act(conv, "ActionLiuJu", schema.message_class("lq.ActionLiuJu")(type=1)) == [
            Ryukyoku(deltas=None), EndKyoku()
        ]


class TestUnknownActions:
    def test_action_mj_start_is_a_noop(self, conv: MajsoulToMjai, schema: LiqiSchema) -> None:
        assert act(conv, "ActionMJStart", schema.message_class("lq.ActionMJStart")()) == []

    def test_unhandled_action_does_not_swallow_reach_accepted(
        self, conv: MajsoulToMjai, schema: LiqiSchema
    ) -> None:
        """沒有對應處理的動作仍要讓立直成立事件流出去,不能一起吞掉。"""
        act(conv, "ActionNewRound", new_round(schema))
        act(conv, "ActionDiscardTile", schema.message_class("lq.ActionDiscardTile")(
            seat=1, tile="1z", is_liqi=True))

        events = act(conv, "ActionMJStart", schema.message_class("lq.ActionMJStart")())

        assert events == [ReachAccepted(actor=1)]
