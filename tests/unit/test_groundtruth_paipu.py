"""牌譜紀錄 → 一步一步的 MJAI 事件。"""

from __future__ import annotations

from typing import Any

import pytest

from mia.groundtruth.paipu import decode_records, replay_steps
from mia.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema
from mia.mjai.events import Ankan, Chi, Dahai, StartGame, StartKyoku, Tsumo

pytestmark = pytest.mark.skipif(
    not DEFAULT_LIQI_PATH.is_file(), reason="需要 assets/proto/liqi.json"
)


@pytest.fixture(scope="module")
def schema() -> LiqiSchema:
    return LiqiSchema.load()


def wrap(schema: LiqiSchema, name: str, message: Any) -> bytes:
    wrapper = schema.message_class("lq.Wrapper")(
        name=f".lq.{name}", data=message.SerializeToString()
    )
    return wrapper.SerializeToString()


def game(schema: LiqiSchema, *, version: int = 210715) -> bytes:
    """一局的開頭:配牌、莊家打一張、上家吃、上家暗槓。"""
    rec = schema.message_class
    records = [
        wrap(schema, "RecordNewRound", rec("lq.RecordNewRound")(
            chang=0, ju=0, ben=0, doras=["3p"], scores=[25000] * 4,
            tiles0=["1m"] * 14, tiles1=["2m"] * 13, tiles2=["3m"] * 13, tiles3=["4m"] * 13,
        )),
        wrap(schema, "RecordDiscardTile", rec("lq.RecordDiscardTile")(seat=0, tile="3s")),
        wrap(schema, "RecordChiPengGang", rec("lq.RecordChiPengGang")(
            seat=1, type=0, tiles=["3s", "4s", "5s"], froms=[0, 1, 1],
        )),
        wrap(schema, "RecordAnGangAddGang", rec("lq.RecordAnGangAddGang")(
            seat=1, type=3, tiles="1z",
        )),
    ]
    if version:
        details = rec("lq.GameDetailRecords")(
            version=version,
            actions=[
                rec("lq.GameAction")(type=1, result=records[0]),
                rec("lq.GameAction")(type=2),  # 玩家輸入,不是牌局動作
                *[rec("lq.GameAction")(type=1, result=r) for r in records[1:]],
            ],
        )
    else:
        details = rec("lq.GameDetailRecords")(records=records)
    return wrap(schema, "GameDetailRecords", details)


class TestDecode:
    @pytest.mark.parametrize("version", [210715, 0])
    def test_both_formats_give_the_same_steps(self, schema: LiqiSchema, version: int) -> None:
        names = [name for name, _ in decode_records(schema, game(schema, version=version))]
        assert names == [
            ".lq.RecordNewRound",
            ".lq.RecordDiscardTile",
            ".lq.RecordChiPengGang",
            ".lq.RecordAnGangAddGang",
        ]


class TestReplay:
    def test_first_step_is_a_synthesised_start_game(self, schema: LiqiSchema) -> None:
        steps = replay_steps(schema, game(schema), seat=2, names=["a", "b", "c", "d"])
        assert steps[0].events == (StartGame(id=2, names=["a", "b", "c", "d"]),)
        assert [s.index for s in steps] == list(range(len(steps)))

    def test_only_the_viewpoint_hand_is_known(self, schema: LiqiSchema) -> None:
        """畫面上別家的手牌是背面,素材的答案也不能知道。"""
        steps = replay_steps(schema, game(schema), seat=1)
        start = steps[1].events[0]
        assert isinstance(start, StartKyoku)
        assert start.tehais[1] == ["2m"] * 13
        assert start.tehais[0] == ["?"] * 13
        assert steps[1].events[1] == Tsumo(actor=0, pai="?")

    def test_dealer_viewpoint_sees_its_own_draw(self, schema: LiqiSchema) -> None:
        steps = replay_steps(schema, game(schema), seat=0)
        assert steps[1].events[1] == Tsumo(actor=0, pai="1m")

    def test_calls_go_through_the_live_converter(self, schema: LiqiSchema) -> None:
        steps = replay_steps(schema, game(schema), seat=0)
        assert steps[2].events == (Dahai(actor=0, pai="3s", tsumogiri=False),)
        assert steps[3].events == (Chi(actor=1, target=0, pai="3s", consumed=["4s", "5s"]),)
        assert steps[4].events == (Ankan(actor=1, consumed=["E"] * 4),)
