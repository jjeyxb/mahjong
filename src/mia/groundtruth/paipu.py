"""牌譜(``fetchGameRecord`` 的回應)→ 一步一步的 MJAI 事件。

為什麼要有這個
--------------
副露的素材要從牌譜來(見 ``docs/cv_mjai.md`` §3.3):跟機器人打錄不到槓、
錄不到下家鳴牌,牌譜是挑的。但牌譜的整場紀錄在一開始一次送完,即時對局那條
「每一幀配上那一刻的封包」的路走不通。

所以這裡不管時間,只管**順序**:把紀錄拆成一步一步,每一步附上它產生的
MJAI 事件。畫面那邊由 ``tools/paipu.py`` 一步一步按、一步截一張,兩邊用
步數對起來 —— 不必再靠單調對齊去猜哪一幀是哪個狀態。

紀錄長什麼樣
------------
``ResGameRecord.data`` 是一個 ``Wrapper``,裡面是 ``GameDetailRecords``。
它有兩種版本:

* 新版(``version`` 不為 0):``actions[]``,每個 ``GameAction`` 的 ``type``
  為 1 時 ``result`` 才是一個動作(又是一個 ``Wrapper``);其他 type 是玩家
  輸入、表情這類東西,不是牌局動作。
* 舊版:``records[]``,每一筆直接是一個 ``Wrapper``。

內層的型別叫 ``RecordDiscardTile``、``RecordChiPengGang`` …,欄位與即時的
``ActionDiscardTile``、``ActionChiPengGang`` 幾乎一樣,所以這裡**不另寫一套
轉換**,改名之後直接交給 :class:`~mia.groundtruth.to_mjai.MajsoulToMjai`。
即時與牌譜走同一份轉換程式碼,理由與 :mod:`mia.groundtruth.stream` 相同。

唯一不一樣的是配牌:即時的 ``ActionNewRound.tiles`` 只有自己的手牌,
牌譜的 ``RecordNewRound`` 有四家的 ``tiles0`` ~ ``tiles3``。這裡依視角挑一家
當 ``tiles``,其餘照即時的規則一律 ``?`` —— 素材要模擬的是**那個視角在畫面
上看得到的東西**,別家的手牌畫面上是背面。

紀錄很大時 ``data`` 是空的、改給 ``data_url``,要另外下載。這個模組只吃位元組,
下載交給呼叫端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mia.groundtruth.liqi import LiqiParser
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.to_mjai import MajsoulToMjai
from mia.mjai.events import MjaiEvent, StartGame

__all__ = ["RecordStep", "decode_records", "replay_steps"]

#: ``GameAction.type`` 裡代表「這是一個牌局動作」的值。
_ACTION_RESULT = 1


@dataclass(frozen=True, slots=True)
class RecordStep:
    """牌譜裡的一步。

    Attributes:
        index: 第幾步,從 0 起算,只數牌局動作。
        name: 內層型別名,例如 ``".lq.RecordDiscardTile"``。
        events: 這一步產生的 MJAI 事件。可能是空的(例如 ``RecordMJStart``)。
    """

    index: int
    name: str
    events: tuple[MjaiEvent, ...]


class _WithTiles:
    """把 ``RecordNewRound`` 裝成 ``ActionNewRound`` 的樣子:只多一個 ``tiles``。

    protobuf 訊息不能隨便加屬性,所以包一層。其餘欄位原樣轉給內層。
    """

    def __init__(self, record: Any, tiles: list[str]) -> None:
        self._record = record
        self.tiles = tiles

    def __getattr__(self, name: str) -> Any:
        return getattr(self._record, name)


def decode_records(schema: LiqiSchema, data: bytes) -> list[tuple[str, Any]]:
    """把 ``ResGameRecord.data`` 拆成 ``(型別名, 訊息)``,依序。

    型別名一律帶前導點(``".lq.RecordNewRound"``),與 :class:`LiqiParser` 一致。
    認不得的型別直接略過 —— 與即時那條路一樣,協定多了新東西不該讓整份紀錄報廢。
    """
    wrapper_cls = schema.message_class("lq.Wrapper")

    def unwrap(raw: bytes) -> tuple[str, Any] | None:
        # 與 LiqiParser._unwrap 同一個理由標成 Any:類別是執行期產生的。
        wrapper: Any = wrapper_cls()
        wrapper.ParseFromString(raw)
        name = wrapper.name if wrapper.name.startswith(".") else f".{wrapper.name}"
        if not schema.has_message(name):
            return None
        return name, schema.parse(name, wrapper.data)

    outer = unwrap(data)
    if outer is None:
        return []
    _, details = outer

    raws: list[bytes]
    if getattr(details, "version", 0):
        raws = [a.result for a in details.actions if a.type == _ACTION_RESULT and a.result]
    else:
        raws = list(details.records)

    steps = []
    for raw in raws:
        item = unwrap(raw)
        if item is not None:
            steps.append(item)
    return steps


def replay_steps(
    schema: LiqiSchema, data: bytes, *, seat: int, names: list[str] | None = None
) -> list[RecordStep]:
    """從某一家的視角重播整份紀錄,每一步附上它產生的事件。

    Args:
        schema: 協定 schema。
        data: ``ResGameRecord.data``(或從 ``data_url`` 下載回來的位元組)。
        seat: 視角。決定 ``start_game.id``、看得到誰的配牌與摸牌。
        names: 四家的名稱,依座位。缺的補空字串。

    Returns:
        第 0 步是合成的 ``start_game``(紀錄裡沒有這一步,但下游的 tracker
        要它),之後一筆紀錄一步。
    """
    records = decode_records(schema, data)
    num_players = _num_players(records)
    names = list(names or [])[:num_players]
    names += [""] * (num_players - len(names))

    converter = MajsoulToMjai(LiqiParser(schema))
    state = converter.state
    state.seat = seat
    state.num_players = num_players
    state.names = names

    steps = [RecordStep(0, "", (StartGame(id=seat, names=names),))]
    for name, record in records:
        action_name = name.replace(".lq.Record", ".lq.Action", 1)
        action = record
        if action_name == ".lq.ActionNewRound":
            action = _WithTiles(record, list(getattr(record, f"tiles{seat}", [])))
        events = converter._on_action(action_name, action)  # noqa: SLF001
        steps.append(RecordStep(len(steps), name, tuple(events)))
    return steps


def _num_players(records: list[tuple[str, Any]]) -> int:
    """從第一個配牌看是幾人麻將:三麻的 ``tiles3`` 是空的。"""
    for name, record in records:
        if name == ".lq.RecordNewRound":
            return 4 if list(getattr(record, "tiles3", [])) else 3
    return 4
