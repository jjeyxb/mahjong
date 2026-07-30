"""MJAI 事件型別。

MJAI 是 JSON Lines 協定,每個事件是一個帶 ``type`` 欄位的物件。這裡用
dataclass 而不是直接傳 dict:事件流會同時被 AI 引擎消費與被測試比對,有明確
的型別,除錯與斷言都好做得多;真正要送進引擎時再 :meth:`MjaiEvent.to_dict`
轉成 JSON。

目前事件由封包解析(:mod:`mia.groundtruth.to_mjai`)產生 ——
視覺辨識不再重建完整對局狀態,見 ``docs/decisions.md``。

省略 ``None`` 欄位是刻意的:MJAI 的消費端(libriichi / Mortal)以「鍵不存在」
而非「值為 null」來判斷選填欄位。

兩個方向
--------
事件流是**雙向**的:``to_dict`` 把我們產生的事件送進引擎,:func:`parse_event`
把引擎回覆的動作讀回來。同一組 dataclass 兩邊共用 —— 引擎回的「打 3s」與我們
送出的「某家打 3s」本來就是同一件事,分成兩套型別只會在對照時多一層轉換。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

__all__ = [
    "NONE_ACTION",
    "Ankan",
    "Chi",
    "Dahai",
    "Daiminkan",
    "Dora",
    "EndGame",
    "EndKyoku",
    "Hora",
    "Kakan",
    "Kita",
    "MjaiEvent",
    "MjaiFormatError",
    "Pon",
    "Reach",
    "ReachAccepted",
    "Ryukyoku",
    "StartGame",
    "StartKyoku",
    "Tsumo",
    "parse_event",
]

#: 引擎用來表示「這一手我不動作」的 ``type`` 值。它不是對局事件,沒有對應的
#: dataclass —— :func:`parse_event` 會拒絕它,呼叫端應先自行判斷。
NONE_ACTION = "none"


class MjaiFormatError(ValueError):
    """收到的 JSON 不是合法的 MJAI 事件。"""


@dataclass(frozen=True, slots=True)
class MjaiEvent:
    """所有 MJAI 事件的基底。"""

    #: JSON 中的 ``type`` 值,由各子類別指定
    TYPE: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        """轉成可直接 ``json.dumps`` 的 dict。值為 None 的欄位會被略去。"""
        data: dict[str, Any] = {"type": self.TYPE}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                data[f.name] = value
        return data

    def __str__(self) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.to_dict().items() if k != "type")
        return f"{self.TYPE}({body})" if body else self.TYPE


# --------------------------------------------------------------------- 對局層級


@dataclass(frozen=True, slots=True)
class StartGame(MjaiEvent):
    """一場對局開始。``id`` 是自己的座位(0~3)。

    Attributes:
        id: 自己的座位。
        names: 四家的暱稱。**不知道時要留 ``None``,不能給空 list** ——
            libriichi 對這個欄位的規則是「可以不存在,存在就必須剛好四個」,
            送 ``[]`` 進去會被拒絕(``invalid length 0, expected an array of
            length 4``)。``None`` 會被 :meth:`~MjaiEvent.to_dict` 省略掉,
            正好對應「沒有這個鍵」。
    """

    TYPE: ClassVar[str] = "start_game"

    id: int
    names: list[str] | None = None


@dataclass(frozen=True, slots=True)
class EndGame(MjaiEvent):
    TYPE: ClassVar[str] = "end_game"


# --------------------------------------------------------------------- 局層級


@dataclass(frozen=True, slots=True)
class StartKyoku(MjaiEvent):
    """一局開始。

    Attributes:
        bakaze: 場風,``E`` / ``S`` / ``W`` / ``N``。
        kyoku: 局數,1 起算。
        honba: 本場數。
        kyotaku: 供托(立直棒)數。
        oya: 莊家座位。
        dora_marker: 寶牌指示牌。
        tehais: 每家的配牌;不知道的牌以 ``?`` 表示。
        scores: 每家點數。
    """

    TYPE: ClassVar[str] = "start_kyoku"

    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    oya: int
    dora_marker: str
    tehais: list[list[str]]
    scores: list[int]


@dataclass(frozen=True, slots=True)
class EndKyoku(MjaiEvent):
    TYPE: ClassVar[str] = "end_kyoku"


# --------------------------------------------------------------------- 基本動作


@dataclass(frozen=True, slots=True)
class Tsumo(MjaiEvent):
    """摸牌。看不到別家摸什麼時 ``pai`` 為 ``?``。"""

    TYPE: ClassVar[str] = "tsumo"

    actor: int
    pai: str


@dataclass(frozen=True, slots=True)
class Dahai(MjaiEvent):
    """打牌。``tsumogiri`` 為摸切。"""

    TYPE: ClassVar[str] = "dahai"

    actor: int
    pai: str
    tsumogiri: bool


@dataclass(frozen=True, slots=True)
class Dora(MjaiEvent):
    """翻開新的寶牌指示牌。翻牌時機見 to_mjai 的說明。"""

    TYPE: ClassVar[str] = "dora"

    dora_marker: str


# --------------------------------------------------------------------- 鳴牌


@dataclass(frozen=True, slots=True)
class Chi(MjaiEvent):
    TYPE: ClassVar[str] = "chi"

    actor: int
    target: int
    pai: str
    consumed: list[str]


@dataclass(frozen=True, slots=True)
class Pon(MjaiEvent):
    TYPE: ClassVar[str] = "pon"

    actor: int
    target: int
    pai: str
    consumed: list[str]


@dataclass(frozen=True, slots=True)
class Daiminkan(MjaiEvent):
    TYPE: ClassVar[str] = "daiminkan"

    actor: int
    target: int
    pai: str
    consumed: list[str]


@dataclass(frozen=True, slots=True)
class Ankan(MjaiEvent):
    """暗槓。``consumed`` 為四張同牌;有赤五時赤五排在索引 0。"""

    TYPE: ClassVar[str] = "ankan"

    actor: int
    consumed: list[str]


@dataclass(frozen=True, slots=True)
class Kakan(MjaiEvent):
    """加槓。``pai`` 是加上去的那張,``consumed`` 是原本碰的三張。"""

    TYPE: ClassVar[str] = "kakan"

    actor: int
    pai: str
    consumed: list[str]


# --------------------------------------------------------------------- 立直


@dataclass(frozen=True, slots=True)
class Reach(MjaiEvent):
    """立直宣言。緊接著會有宣言牌的 :class:`Dahai`。"""

    TYPE: ClassVar[str] = "reach"

    actor: int


@dataclass(frozen=True, slots=True)
class ReachAccepted(MjaiEvent):
    """立直成立(扣 1000 點)。宣言牌通過、沒被榮和之後才發生。"""

    TYPE: ClassVar[str] = "reach_accepted"

    actor: int


# --------------------------------------------------------------------- 終局


@dataclass(frozen=True, slots=True)
class Hora(MjaiEvent):
    """和了。自摸時 ``target`` 等於 ``actor``。

    Attributes:
        deltas: 各家點數增減。
        ura_markers: 裏寶牌指示牌;立直但無裏寶時為空 list,非立直為 None
            —— 兩者語意不同,消費端要分得出來。
    """

    TYPE: ClassVar[str] = "hora"

    actor: int
    target: int
    deltas: list[int] | None = None
    ura_markers: list[str] | None = None


@dataclass(frozen=True, slots=True)
class Ryukyoku(MjaiEvent):
    """流局。``deltas`` 為聽牌罰符等點數變動,無變動時為 None。

    這個型別身兼兩用。從封包來的是**結果**(荒牌平局,只有 ``deltas``);
    從引擎回來的是**宣告**(九種九牌,帶 ``actor`` 與 ``reason``)。
    MJAI 用同一個 ``type`` 表示兩者,這裡跟著它 —— 拆成兩個型別會讓
    「引擎宣告的流局」與「實際發生的流局」在事件流上對不起來。
    """

    TYPE: ClassVar[str] = "ryukyoku"

    deltas: list[int] | None = None
    actor: int | None = None
    reason: str | None = None


# --------------------------------------------------------------------- 三麻


@dataclass(frozen=True, slots=True)
class Kita(MjaiEvent):
    """北抜き(拔北寶牌)。僅三人麻將有,``pai`` 恆為 ``N``。"""

    TYPE: ClassVar[str] = "kita"

    actor: int
    pai: str = "N"


# --------------------------------------------------------------------- 反序列化

_EVENT_TYPES: tuple[type[MjaiEvent], ...] = (
    StartGame,
    EndGame,
    StartKyoku,
    EndKyoku,
    Tsumo,
    Dahai,
    Dora,
    Chi,
    Pon,
    Daiminkan,
    Ankan,
    Kakan,
    Reach,
    ReachAccepted,
    Hora,
    Ryukyoku,
    Kita,
)

_BY_TYPE: dict[str, type[MjaiEvent]] = {cls.TYPE: cls for cls in _EVENT_TYPES}


def parse_event(data: dict[str, Any]) -> MjaiEvent:
    """MJAI 的 JSON 物件 → 對應的 dataclass。

    引擎的回覆走這裡進來。解析而不是原封不動傳 dict,是為了讓格式錯誤在
    **收到的當下**就炸掉 —— 一個少了 ``pai`` 的 ``dahai`` 如果一路傳到 UI
    才出事,現場只會看到一個 KeyError,追不回是哪個引擎送錯的。

    Args:
        data: 已 ``json.loads`` 的物件,須含 ``type``。

    Returns:
        對應的事件。

    Raises:
        MjaiFormatError: 缺 ``type``、type 不認得、或缺必填欄位。

    Note:
        認不得的鍵會被**忽略**。引擎會在回覆裡夾帶自訂欄位(Mortal 的
        ``meta`` 帶著 Q 值與候選遮罩),那些不屬於協定本身,由
        :mod:`mia.engine` 在解析前先取走。

        ``{"type": "none"}`` **不是**事件,這裡會拒絕 —— 見 :data:`NONE_ACTION`。
    """
    if not isinstance(data, dict):
        raise MjaiFormatError(f"MJAI 事件必須是物件,收到 {type(data).__name__}")

    kind = data.get("type")
    if kind is None:
        raise MjaiFormatError(f"缺少 type 欄位: {data!r}")
    if kind == NONE_ACTION:
        raise MjaiFormatError(
            f"{NONE_ACTION!r} 表示「不動作」,不是事件。呼叫端應先判斷這個值"
        )

    cls = _BY_TYPE.get(kind)
    if cls is None:
        raise MjaiFormatError(f"認不得的事件型別: {kind!r}")

    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
    try:
        return cls(**kwargs)
    except TypeError as exc:
        # dataclass 少了必填參數 —— 訊息是 Python 的,補上事件內容才追得回來
        raise MjaiFormatError(f"{kind} 事件的欄位不完整: {data!r}") from exc
