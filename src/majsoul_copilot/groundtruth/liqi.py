"""雀魂 WebSocket 訊息的框架解析。

線路格式
--------
每一則 WebSocket binary frame 的結構::

    NOTIFY    : [0x01]                    [Wrapper protobuf]
    REQUEST   : [0x02] [msg_id uint16 LE] [Wrapper protobuf]
    RESPONSE  : [0x03] [msg_id uint16 LE] [Wrapper protobuf]

``Wrapper`` 是 ``{ string name = 1; bytes data = 2; }``:

* **NOTIFY** 的 ``name`` 是訊息型別,例如 ``.lq.ActionPrototype``。
* **REQUEST** 的 ``name`` 是方法名,例如 ``.lq.FastTest.inputOperation``。
* **RESPONSE** 的 ``name`` 是**空字串** —— 客戶端靠 ``msg_id`` 跟先前的請求配對。
  因此解析器必須是有狀態的:記住每個 ID 對應哪個方法,才知道回應要用什麼型別解。

牌局事件包在 ``.lq.ActionPrototype`` 裡(``{step, name, data}``),``data`` 才是
真正的 ``ActionDealTile`` / ``ActionDiscardTile`` 等訊息。

關於 data 的混淆
----------------
雀魂會對 ``ActionPrototype.data`` 做一層 XOR 混淆。本模組的處理方式是
**先直接解析,失敗才套用 XOR 再解析**(見 :func:`deobfuscate`),並在
:attr:`LiqiMessage.deobfuscated` 記錄實際走了哪條路。

這樣設計是刻意的:XOR 的金鑰表是社群逆向得來的,官方隨時可能改動或移除。
與其寫死一種假設然後在某天無聲地全部解析失敗,不如兩條路都試、記錄結果 ——
第一次接到真實流量時就能從日誌看出到底哪一種才對。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from google.protobuf.message import DecodeError, Message

from majsoul_copilot.groundtruth.schema import LiqiSchema
from majsoul_copilot.utils.logging import logger

__all__ = [
    "ACTION_PROTOTYPE",
    "LiqiMessage",
    "LiqiParseError",
    "LiqiParser",
    "MessageKind",
    "deobfuscate",
]

ACTION_PROTOTYPE = ".lq.ActionPrototype"

# 社群逆向得到的混淆金鑰表。見模組 docstring 對於「為什麼不寫死」的說明。
_XOR_KEYS = (0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C)


class MessageKind(IntEnum):
    """線路上的第一個位元組。"""

    NOTIFY = 1
    REQUEST = 2
    RESPONSE = 3


class LiqiParseError(Exception):
    """訊息不符合 liqi 的線路格式。"""


def _has_content(message: Message) -> bool:
    """訊息是否解出了任何已知欄位。

    protobuf 對無法辨識的位元組相當寬容 —— 它們會被收進 unknown fields,
    解析照樣「成功」,但已知欄位一個都沒填。這是判斷「解對了沒」的實用依據。
    """
    return bool(message.ListFields())


def deobfuscate(data: bytes) -> bytes:
    """對 ``ActionPrototype.data`` 套用 XOR 反混淆。

    這是對合運算(自身的反運算),再套一次即可還原。
    """
    if not data:
        return data
    out = bytearray(data)
    length = len(out)
    key_count = len(_XOR_KEYS)
    for i in range(length):
        out[i] ^= ((23 ^ length) + 5 * i + _XOR_KEYS[i % key_count]) & 0xFF
    return bytes(out)


@dataclass(frozen=True, slots=True)
class LiqiMessage:
    """一則解析完成的 liqi 訊息。

    Attributes:
        kind: NOTIFY / REQUEST / RESPONSE。
        method: NOTIFY 為訊息型別名,REQUEST/RESPONSE 為方法名(皆含前導點)。
            回應是靠 msg_id 回推的,若配不到請求則為空字串。
        payload: 解析後的 protobuf 訊息。
        msg_id: REQUEST / RESPONSE 的配對 ID;NOTIFY 為 None。
        from_client: True 表示客戶端送出,False 表示伺服器送來。
        action_name: 若這是一則 ``ActionPrototype``,此處為內層動作型別名。
        action: 內層動作訊息本身。
        step: ``ActionPrototype.step``,牌局內的事件序號。
        deobfuscated: 內層動作是否需要 XOR 反混淆才解得開。None 表示不適用。
    """

    kind: MessageKind
    method: str
    payload: Message
    msg_id: int | None = None
    from_client: bool = False
    action_name: str = ""
    action: Message | None = None
    step: int | None = None
    deobfuscated: bool | None = None
    raw: bytes = field(default=b"", repr=False)

    @property
    def is_action(self) -> bool:
        """是否為牌局內的動作事件(GT 事件流真正需要的東西)。"""
        return self.action is not None

    def __str__(self) -> str:
        arrow = "→" if self.from_client else "←"
        if self.is_action:
            return f"{arrow} [{self.step}] {self.action_name}"
        ident = f"#{self.msg_id}" if self.msg_id is not None else ""
        return f"{arrow} {self.kind.name}{ident} {self.method}"


class LiqiParser:
    """有狀態的 liqi 訊息解析器。

    狀態只有一項:``msg_id → 方法名`` 的待回應表。回應本身不帶方法名,
    必須靠這張表才能知道要用哪個型別去解。

    一場對局用一個解析器實例;重新連線時呼叫 :meth:`reset`。
    """

    def __init__(self, schema: LiqiSchema | None = None) -> None:
        self.schema = schema or LiqiSchema.load()
        self._wrapper_cls = self.schema.message_class("lq.Wrapper")
        self._action_prototype_cls = self.schema.message_class("lq.ActionPrototype")
        self._pending: dict[int, str] = {}
        self._unmatched_responses = 0
        # 未知型別每種只警告一次 —— 否則一場對局會刷出上百行相同訊息
        self._unknown_actions: set[str] = set()

    def reset(self) -> None:
        """清空待回應表。重新連線或換一局時呼叫。"""
        self._pending.clear()
        self._unmatched_responses = 0

    @property
    def pending_count(self) -> int:
        """尚未收到回應的請求數。持續成長代表有訊息被漏掉了。"""
        return len(self._pending)

    @property
    def unmatched_responses(self) -> int:
        """配不到請求的回應數。非零通常表示錄製是從連線中途開始的。"""
        return self._unmatched_responses

    # ------------------------------------------------------------------

    def parse(self, data: bytes, *, from_client: bool) -> LiqiMessage:
        """解析一則 WebSocket binary frame。

        Args:
            data: frame 的完整位元組。
            from_client: 方向。決定 REQUEST 該登記還是該忽略。

        Raises:
            LiqiParseError: 長度不足、未知的訊息類型,或 protobuf 解不開。
        """
        if len(data) < 1:
            raise LiqiParseError("訊息為空")

        try:
            kind = MessageKind(data[0])
        except ValueError as exc:
            raise LiqiParseError(f"未知的訊息類型位元組 0x{data[0]:02x}") from exc

        if kind is MessageKind.NOTIFY:
            return self._parse_notify(self._unwrap(data[1:]), data, from_client)

        if len(data) < 3:
            raise LiqiParseError(f"{kind.name} 訊息過短({len(data)} 位元組),缺少 msg_id")
        msg_id = int(struct.unpack_from("<H", data, 1)[0])
        wrapper = self._unwrap(data[3:])

        if kind is MessageKind.REQUEST:
            return self._parse_request(wrapper, msg_id, data, from_client)
        return self._parse_response(wrapper, msg_id, data, from_client)

    def _unwrap(self, body: bytes) -> Any:
        # 回傳型別是 Any 而非 Message:訊息類別是執行期從 liqi.json 產生的,
        # 靜態型別檢查無從得知 .name / .data 這些欄位存在。
        wrapper = self._wrapper_cls()
        try:
            wrapper.ParseFromString(body)
        except DecodeError as exc:
            raise LiqiParseError(f"Wrapper 解析失敗({len(body)} 位元組)") from exc
        return wrapper

    # ------------------------------------------------------------------

    def _parse_notify(self, wrapper: Any, raw: bytes, from_client: bool) -> LiqiMessage:
        type_name = wrapper.name
        if not self.schema.has_message(type_name):
            raise LiqiParseError(f"NOTIFY 帶了未知的型別 {type_name!r}")
        payload = self.schema.parse(type_name, wrapper.data)

        if type_name == ACTION_PROTOTYPE:
            action_name, action, deobfuscated = self.parse_action_prototype(payload)
            return LiqiMessage(
                kind=MessageKind.NOTIFY,
                method=type_name,
                payload=payload,
                from_client=from_client,
                action_name=action_name,
                action=action,
                step=payload.step,
                deobfuscated=deobfuscated,
                raw=raw,
            )
        return LiqiMessage(
            kind=MessageKind.NOTIFY,
            method=type_name,
            payload=payload,
            from_client=from_client,
            raw=raw,
        )

    def _parse_request(
        self, wrapper: Any, msg_id: int, raw: bytes, from_client: bool
    ) -> LiqiMessage:
        method = wrapper.name
        signature = self.schema.methods.get(method)
        if signature is None:
            raise LiqiParseError(f"REQUEST 帶了未知的方法 {method!r}")
        self._pending[msg_id] = method
        return LiqiMessage(
            kind=MessageKind.REQUEST,
            method=method,
            payload=self.schema.parse(signature.request_type, wrapper.data),
            msg_id=msg_id,
            from_client=from_client,
            raw=raw,
        )

    def _parse_response(
        self, wrapper: Any, msg_id: int, raw: bytes, from_client: bool
    ) -> LiqiMessage:
        method = self._pending.pop(msg_id, "")
        if not method:
            # 從連線中途開始錄製時很正常 —— 對應的請求發生在錄製之前。
            self._unmatched_responses += 1
            raise LiqiParseError(
                f"RESPONSE #{msg_id} 配不到請求,無從得知回應型別"
                "(從連線中途開始錄製時屬正常現象)"
            )
        signature = self.schema.methods[method]
        return LiqiMessage(
            kind=MessageKind.RESPONSE,
            method=method,
            payload=self.schema.parse(signature.response_type, wrapper.data),
            msg_id=msg_id,
            from_client=from_client,
            raw=raw,
        )

    def parse_action_prototype(
        self, prototype: Any, *, obfuscated: bool = True
    ) -> tuple[str, Message | None, bool | None]:
        """解析 ``ActionPrototype`` 內層的動作訊息。

        Args:
            prototype: 已解析的 ``ActionPrototype``。
            obfuscated: 預期 ``data`` 是否經過 XOR 混淆,決定**先試哪一條路**。

                即時的 ``.lq.ActionPrototype`` Notify 是混淆的;但斷線重連時
                ``GameRestore.actions[]`` 裡夾帶的動作**沒有**混淆,對它們套
                XOR 反而會把位元組弄壞。

                順序有實質意義:protobuf 的解析相當寬鬆,短訊息的錯誤位元組
                有機率被誤判成合法訊息。先試對的那一條可以避開這種假陽性。

        Returns:
            (動作型別名, 解析結果或 None, 是否實際做了反混淆)。
            第三個值為 None 代表兩條路都失敗。
        """
        raw_name = prototype.name
        if not raw_name:
            return "", None, None

        action_name = self._qualify_action(raw_name)
        if action_name is None:
            if raw_name not in self._unknown_actions:
                self._unknown_actions.add(raw_name)
                logger.warning("未知的動作型別 {!r},略過內層解析", raw_name)
            return raw_name, None, None

        raw = prototype.data
        expected = (obfuscated, deobfuscate(raw) if obfuscated else raw)
        alternate = (not obfuscated, raw if obfuscated else deobfuscate(raw))

        # 「能解析」不足以判斷正確。protobuf 很寬鬆:短訊息的錯誤位元組常常
        # 也能「成功」解析,只是所有位元組都落進 unknown fields、已知欄位全空。
        # (實測 6 位元組的 ActionDiscardTile 就會這樣,連 round-trip 都一致。)
        # 所以要看有沒有真的解出東西。
        expected_parsed = self._try_parse(action_name, expected[1])
        if expected_parsed is not None and _has_content(expected_parsed):
            return action_name, expected_parsed, expected[0]

        alternate_parsed = self._try_parse(action_name, alternate[1])
        if alternate_parsed is not None and _has_content(alternate_parsed):
            logger.info(
                "動作 {} 的混淆狀態與預期不符(預期 obfuscated={},實際 {});協定可能有變動",
                action_name,
                obfuscated,
                alternate[0],
            )
            return action_name, alternate_parsed, alternate[0]

        # 兩邊都沒有內容:可能本來就是空訊息(例如 ActionMJStart)。
        # 這種情況下兩者等價,採用預期的那一條。
        if expected_parsed is not None:
            return action_name, expected_parsed, expected[0]
        if alternate_parsed is not None:
            return action_name, alternate_parsed, alternate[0]

        logger.warning(
            "動作 {} 的 data 兩種解法都失敗({} 位元組);混淆演算法可能已變動",
            action_name,
            len(raw),
        )
        return action_name, None, None

    def _qualify_action(self, name: str) -> str | None:
        """把 ``ActionPrototype.name`` 補成完整型別名。

        實測真實流量:內層動作名是**未限定的裸名**(``"ActionDiscardTile"``),
        跟外層 ``Wrapper.name`` 的完整名(``".lq.ActionPrototype"``)寫法不同。
        兩種都接受,回傳統一成帶前導點的完整名,下游比對才有單一形式可依賴。

        Returns:
            正規化後的型別名,或 None(協定中沒有這個型別)。
        """
        if self.schema.has_message(name):
            return name if name.startswith(".") else f".{name}"
        package = self.schema.package
        if package:
            qualified = f"{package}.{name}"
            if self.schema.has_message(qualified):
                return f".{qualified}"
        return None

    def _try_parse(self, type_name: str, payload: bytes) -> Message | None:
        try:
            return self.schema.parse(type_name, payload)
        except DecodeError:
            return None

    def __repr__(self) -> str:
        return f"<LiqiParser pending={len(self._pending)} unmatched={self._unmatched_responses}>"
