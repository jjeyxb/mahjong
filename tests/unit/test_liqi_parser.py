"""liqi 線路格式解析測試。

用真實的 liqi schema 組出合法訊息再解回來 —— 比手寫 schema 更能確保
與實際協定對得上。
"""

from __future__ import annotations

import struct

import pytest

from majsoul_copilot.groundtruth.liqi import (
    LiqiParseError,
    LiqiParser,
    MessageKind,
    deobfuscate,
)
from majsoul_copilot.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema

pytestmark = pytest.mark.skipif(
    not DEFAULT_LIQI_PATH.is_file(), reason="需要 assets/proto/liqi.json"
)


@pytest.fixture(scope="module")
def schema() -> LiqiSchema:
    return LiqiSchema.load()


@pytest.fixture
def parser(schema: LiqiSchema) -> LiqiParser:
    return LiqiParser(schema)


def _wrap(schema: LiqiSchema, name: str, data: bytes) -> bytes:
    return schema.message_class("lq.Wrapper")(name=name, data=data).SerializeToString()


def _notify(schema: LiqiSchema, name: str, data: bytes) -> bytes:
    return b"\x01" + _wrap(schema, name, data)


def _request(schema: LiqiSchema, msg_id: int, method: str, data: bytes) -> bytes:
    return b"\x02" + struct.pack("<H", msg_id) + _wrap(schema, method, data)


def _response(schema: LiqiSchema, msg_id: int, data: bytes) -> bytes:
    return b"\x03" + struct.pack("<H", msg_id) + _wrap(schema, "", data)


def _action(schema: LiqiSchema, step: int, name: str, payload: bytes) -> bytes:
    prototype = schema.message_class("lq.ActionPrototype")(step=step, name=name, data=payload)
    return _notify(schema, ".lq.ActionPrototype", prototype.SerializeToString())


class TestDeobfuscate:
    def test_is_an_involution(self) -> None:
        """XOR 反混淆是對合運算,套兩次會回到原狀。"""
        for payload in (b"", b"\x00", b"hello", bytes(range(256))):
            assert deobfuscate(deobfuscate(payload)) == payload

    def test_actually_changes_content(self) -> None:
        assert deobfuscate(b"hello world") != b"hello world"

    def test_output_length_preserved(self) -> None:
        assert len(deobfuscate(bytes(500))) == 500


class TestFraming:
    def test_notify_has_no_msg_id(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        heartbeat = schema.message_class("lq.ActionMJStart")()
        message = parser.parse(
            _action(schema, 1, ".lq.ActionMJStart", heartbeat.SerializeToString()),
            from_client=False,
        )
        assert message.kind is MessageKind.NOTIFY
        assert message.msg_id is None

    def test_empty_payload_rejected(self, parser: LiqiParser) -> None:
        with pytest.raises(LiqiParseError, match="訊息為空"):
            parser.parse(b"", from_client=False)

    def test_unknown_kind_byte_rejected(self, parser: LiqiParser) -> None:
        with pytest.raises(LiqiParseError, match="未知的訊息類型"):
            parser.parse(b"\x09\x00", from_client=False)

    def test_truncated_request_rejected(self, parser: LiqiParser) -> None:
        with pytest.raises(LiqiParseError, match="缺少 msg_id"):
            parser.parse(b"\x02\x01", from_client=True)

    def test_unknown_notify_type_rejected(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        with pytest.raises(LiqiParseError, match="未知的型別"):
            parser.parse(_notify(schema, ".lq.NotARealMessage", b""), from_client=False)


class TestRequestResponsePairing:
    METHOD = ".lq.Lobby.fetchQueueInfo"

    def test_response_type_recovered_from_msg_id(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """回應的 Wrapper.name 是空的,只能靠 msg_id 回推方法。"""
        signature = schema.methods[self.METHOD]
        request = schema.message_class(signature.request_type)()
        parser.parse(
            _request(schema, 42, self.METHOD, request.SerializeToString()), from_client=True
        )

        response = schema.message_class(signature.response_type)()
        message = parser.parse(
            _response(schema, 42, response.SerializeToString()), from_client=False
        )

        assert message.kind is MessageKind.RESPONSE
        assert message.method == self.METHOD

    def test_pending_count_tracks_open_requests(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        signature = schema.methods[self.METHOD]
        request = schema.message_class(signature.request_type)().SerializeToString()
        assert parser.pending_count == 0
        parser.parse(_request(schema, 1, self.METHOD, request), from_client=True)
        parser.parse(_request(schema, 2, self.METHOD, request), from_client=True)
        assert parser.pending_count == 2

        response = schema.message_class(signature.response_type)().SerializeToString()
        parser.parse(_response(schema, 1, response), from_client=False)
        assert parser.pending_count == 1

    def test_orphan_response_raises_and_is_counted(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """從連線中途開始錄製時很常見,要能明確辨識而不是誤解成別的錯。"""
        with pytest.raises(LiqiParseError, match="配不到請求"):
            parser.parse(_response(schema, 999, b""), from_client=False)
        assert parser.unmatched_responses == 1

    def test_reset_clears_state(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        signature = schema.methods[self.METHOD]
        request = schema.message_class(signature.request_type)().SerializeToString()
        parser.parse(_request(schema, 5, self.METHOD, request), from_client=True)
        parser.reset()
        assert parser.pending_count == 0

    def test_unknown_method_rejected(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        with pytest.raises(LiqiParseError, match="未知的方法"):
            parser.parse(_request(schema, 1, ".lq.Lobby.notAMethod", b""), from_client=True)


class TestActionNameQualification:
    """``ActionPrototype.name`` 的實際寫法。

    迴歸測試:真實流量裡內層動作名是**未限定的裸名**(``"ActionDiscardTile"``),
    跟外層 ``Wrapper.name`` 的完整寫法(``".lq.ActionPrototype"``)不一樣。

    最初的實作只認完整名,結果整場對局 242 個動作**一個都沒解析出來** ——
    而合成測試全數通過,因為測試資料是照著同一個錯誤假設造的。
    這種 bug 只有真實流量抓得到。
    """

    def _prototype(self, schema: LiqiSchema, name: str) -> bytes:
        inner = schema.message_class("lq.ActionDiscardTile")(seat=1, tile="5m")
        return _action(schema, 7, name, deobfuscate(inner.SerializeToString()))

    def test_bare_name_is_resolved(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        """真實流量的寫法。"""
        message = parser.parse(self._prototype(schema, "ActionDiscardTile"), from_client=False)

        assert message.is_action
        assert message.action_name == ".lq.ActionDiscardTile", "應正規化成完整名"
        assert message.action.tile == "5m"

    def test_qualified_name_still_works(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        """兩種寫法都要接受,協定哪天改了也不會壞。"""
        message = parser.parse(self._prototype(schema, ".lq.ActionDiscardTile"), from_client=False)

        assert message.action_name == ".lq.ActionDiscardTile"
        assert message.action.tile == "5m"

    def test_package_qualified_without_dot(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        message = parser.parse(self._prototype(schema, "lq.ActionDiscardTile"), from_client=False)
        assert message.action_name == ".lq.ActionDiscardTile"

    def test_genuinely_unknown_name_is_reported_once(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """未知型別每種只警告一次 —— 否則一場對局會刷出上百行相同訊息。"""
        for _ in range(3):
            message = parser.parse(self._prototype(schema, "ActionNotReal"), from_client=False)
            assert message.action is None
            assert message.action_name == "ActionNotReal", "認不得就原樣回報,不要亂補前綴"


class TestActionPrototype:
    def _discard(self, schema: LiqiSchema, **kwargs: object):
        return schema.message_class("lq.ActionDiscardTile")(**kwargs)

    def test_obfuscated_action_is_decoded(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        inner = self._discard(schema, seat=1, tile="3p", moqie=True)
        raw = _action(schema, 12, ".lq.ActionDiscardTile", deobfuscate(inner.SerializeToString()))

        message = parser.parse(raw, from_client=False)

        assert message.is_action
        assert message.deobfuscated is True
        assert message.step == 12
        assert message.action_name == ".lq.ActionDiscardTile"
        assert (message.action.seat, message.action.tile, message.action.moqie) == (1, "3p", True)

    def test_plain_action_also_works(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        """混淆演算法可能被官方移除,未混淆的 data 也必須解得開。

        迴歸測試:這裡的 ActionDiscardTile 只有 6 位元組,把它做 XOR 之後的
        位元組**照樣能被 protobuf「成功」解析**(所有位元組都落進 unknown
        fields,已知欄位全空,連 round-trip 都一致)。只看「有沒有拋例外」
        會選到錯的那一邊,得到一個空的動作。判準必須是「有沒有解出已知欄位」。
        """
        inner = self._discard(schema, seat=3, tile="7s")
        raw = _action(schema, 13, ".lq.ActionDiscardTile", inner.SerializeToString())

        message = parser.parse(raw, from_client=False)

        assert message.deobfuscated is False
        assert message.action.tile == "7s"
        assert message.action.seat == 3

    def test_restore_actions_are_not_obfuscated(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """GameRestore 內夾帶的動作沒有經過 XOR,套反混淆會把位元組弄壞。"""
        inner = self._discard(schema, seat=2, tile="1p", is_liqi=True)
        prototype = schema.message_class("lq.ActionPrototype")(
            step=5, name=".lq.ActionDiscardTile", data=inner.SerializeToString()
        )

        name, action, deobfuscated = parser.parse_action_prototype(prototype, obfuscated=False)

        assert name == ".lq.ActionDiscardTile"
        assert deobfuscated is False
        assert (action.seat, action.tile, action.is_liqi) == (2, "1p", True)

    def test_expected_branch_wins_when_both_parse_empty(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """本來就沒有欄位的動作,兩條路都解出空訊息,應採用預期的那一條。"""
        prototype = schema.message_class("lq.ActionPrototype")(
            step=1, name=".lq.ActionMJStart", data=b""
        )
        name, action, deobfuscated = parser.parse_action_prototype(prototype)

        assert name == ".lq.ActionMJStart"
        assert action is not None
        assert deobfuscated is True

    def test_unparseable_action_data_does_not_raise(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        """外層仍要能解,只是內層動作為 None —— 一則壞訊息不該中斷整份錄影的解析。"""
        raw = _action(schema, 14, ".lq.ActionDiscardTile", b"\xff\xff\xff\xff\xff\xff\xff\xff")

        message = parser.parse(raw, from_client=False)

        assert message.action is None
        assert message.deobfuscated is None
        assert message.step == 14

    def test_unknown_action_name_is_tolerated(
        self, parser: LiqiParser, schema: LiqiSchema
    ) -> None:
        raw = _action(schema, 15, ".lq.ActionFromTheFuture", b"")
        message = parser.parse(raw, from_client=False)
        assert message.action is None
        assert message.action_name == ".lq.ActionFromTheFuture"

    def test_non_action_notify_has_no_action(self, parser: LiqiParser, schema: LiqiSchema) -> None:
        notify = schema.message_class("lq.NotifyRoomGameStart")(game_url="x", connect_token="y")
        message = parser.parse(
            _notify(schema, ".lq.NotifyRoomGameStart", notify.SerializeToString()),
            from_client=False,
        )
        assert not message.is_action
        assert message.method == ".lq.NotifyRoomGameStart"
