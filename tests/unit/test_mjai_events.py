"""MJAI 事件的序列化與反序列化。

反序列化是引擎回覆的入口,格式錯誤在這裡沒擋住,錯的動作就會一路送到 UI。
"""

from __future__ import annotations

import pytest

from mia.mjai import (
    NONE_ACTION,
    Chi,
    Dahai,
    Hora,
    MjaiFormatError,
    Reach,
    Ryukyoku,
    StartGame,
    Tsumo,
    parse_event,
)


class TestToDict:
    def test_none_fields_are_omitted(self) -> None:
        """消費端以「鍵不存在」判斷選填欄位,送 null 進去會被當成有值。"""
        assert "deltas" not in Ryukyoku().to_dict()

    def test_empty_list_is_not_omitted(self) -> None:
        """空 list 與 None 語意不同 —— 立直但無裏寶是 [],非立直是 None。"""
        assert Hora(actor=0, target=1, ura_markers=[]).to_dict()["ura_markers"] == []

    def test_start_game_without_names_omits_the_key(self) -> None:
        """libriichi 的規則是「可以沒有 names,有就必須剛好四個」。

        先前 ``names`` 預設是空 list,``to_dict`` 會送出 ``"names": []``,
        libriichi 直接拒收(invalid length 0, expected an array of length 4)。
        這個測試釘住「不知道就別送」。
        """
        assert StartGame(id=0).to_dict() == {"type": "start_game", "id": 0}

    def test_start_game_with_names_keeps_them(self) -> None:
        names = ["a", "b", "c", "d"]
        assert StartGame(id=2, names=names).to_dict()["names"] == names


class TestParseEvent:
    def test_a_dahai_round_trips(self) -> None:
        event = Dahai(actor=1, pai="3s", tsumogiri=True)
        assert parse_event(event.to_dict()) == event

    @pytest.mark.parametrize(
        "event",
        [
            Tsumo(actor=0, pai="5mr"),
            Reach(actor=3),
            Chi(actor=2, target=1, pai="4p", consumed=["3p", "5p"]),
            Hora(actor=0, target=0, deltas=[8000, -8000, 0, 0]),
            StartGame(id=1, names=["a", "b", "c", "d"]),
        ],
    )
    def test_every_event_round_trips(self, event: object) -> None:
        assert parse_event(event.to_dict()) == event  # type: ignore[attr-defined]

    def test_kyushukyuhai_is_a_ryukyoku_with_a_reason(self) -> None:
        """引擎宣告九種九牌時回的是 ryukyoku,帶 actor 與 reason。

        同一個 type 也用來表示「實際流局」(只有 deltas),兩者共用一個型別
        是刻意的 —— 見 Ryukyoku 的說明。
        """
        event = parse_event({"type": "ryukyoku", "actor": 0, "reason": "kyushukyuhai"})
        assert isinstance(event, Ryukyoku)
        assert event.reason == "kyushukyuhai"
        assert event.deltas is None

    def test_unknown_keys_are_ignored(self) -> None:
        """引擎會夾帶協定之外的欄位(Mortal 的 meta),不該因此解析失敗。"""
        data = {"type": "reach", "actor": 1, "meta": {"q_values": [1.0]}}
        assert parse_event(data) == Reach(actor=1)

    def test_none_action_is_rejected(self) -> None:
        """``none`` 是「不動作」,不是事件 —— 呼叫端要先自己判斷。"""
        with pytest.raises(MjaiFormatError, match=NONE_ACTION):
            parse_event({"type": NONE_ACTION})

    def test_a_missing_type_is_rejected(self) -> None:
        with pytest.raises(MjaiFormatError, match="type"):
            parse_event({"actor": 0})

    def test_an_unknown_type_is_rejected(self) -> None:
        with pytest.raises(MjaiFormatError, match="認不得"):
            parse_event({"type": "teleport", "actor": 0})

    def test_a_missing_required_field_is_rejected(self) -> None:
        """少了 pai 的 dahai。放行的話會在 UI 才炸出 AttributeError,追不回來源。"""
        with pytest.raises(MjaiFormatError, match="欄位不完整"):
            parse_event({"type": "dahai", "actor": 0, "tsumogiri": False})

    def test_a_non_object_is_rejected(self) -> None:
        with pytest.raises(MjaiFormatError):
            parse_event(["dahai"])  # type: ignore[arg-type]
