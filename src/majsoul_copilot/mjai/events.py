"""MJAI 事件型別。

MJAI 是 JSON Lines 協定,每個事件是一個帶 ``type`` 欄位的物件。這裡用
dataclass 表示,原因是這些事件會由**兩條路徑**產生 —— Ground Truth
(:mod:`majsoul_copilot.groundtruth.to_mjai`)與視覺辨識(M4 的 tracker)——
而 M5 的評測要把兩條流對齊比較。有明確的型別,比對與除錯都好做得多;
真正要送進 AI 引擎時再 :meth:`MjaiEvent.to_dict` 轉成 JSON。

省略 ``None`` 欄位是刻意的:MJAI 的消費端(libriichi / Mortal)以「鍵不存在」
而非「值為 null」來判斷選填欄位。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

__all__ = [
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
    "Pon",
    "Reach",
    "ReachAccepted",
    "Ryukyoku",
    "StartGame",
    "StartKyoku",
    "Tsumo",
]


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
    """一場對局開始。``id`` 是自己的座位(0~3)。"""

    TYPE: ClassVar[str] = "start_game"

    id: int
    names: list[str] = field(default_factory=list)


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
    """流局。``deltas`` 為聽牌罰符等點數變動,無變動時為 None。"""

    TYPE: ClassVar[str] = "ryukyoku"

    deltas: list[int] | None = None


# --------------------------------------------------------------------- 三麻


@dataclass(frozen=True, slots=True)
class Kita(MjaiEvent):
    """北抜き(拔北寶牌)。僅三人麻將有,``pai`` 恆為 ``N``。"""

    TYPE: ClassVar[str] = "kita"

    actor: int
    pai: str = "N"
