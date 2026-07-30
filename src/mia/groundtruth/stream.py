"""錄影 frame → liqi 訊息 → MJAI 事件,一步到底。

為什麼要有這一層
----------------
「每條 WebSocket 連線各自一個 ``LiqiParser`` 與一個 ``MajsoulToMjai``」這個
規則原本被抄了三份 —— ``tools/gt.py inspect``、``tools/advise.py``、以及即時
模式。三份各自維護 ``dict[flow, …]``,而抄漏一份就會出現最難查的那種 bug:
兩條連線共用解析器,請求與回應互相配錯,回應被拿去用錯誤的型別解讀。

錄影與即時,同一條路
--------------------
:meth:`MjaiDecoder.feed` 一次吃一個 frame,不假設檔案已經寫完;
:meth:`MjaiDecoder.decode_file` 只是「把整個檔案餵進去」的便利包裝。
所以離線重播與即時跟播走的是**同一份轉換程式碼** —— 離線驗過的東西,
即時模式不會用另一個實作把它推翻。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from mia.groundtruth.dump import DumpFrame, DumpStats, iter_frames
from mia.groundtruth.liqi import LiqiMessage, LiqiParseError, LiqiParser
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.to_mjai import MajsoulToMjai
from mia.mjai import MjaiEvent

__all__ = ["DecodedFrame", "MjaiDecoder"]


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """一個 frame 解出來的東西。

    Attributes:
        frame: 原始 frame,含時間戳 —— 與畫面錄影對齊時要用。
        message: 解析出的 liqi 訊息。
        events: 這則訊息產生的 MJAI 事件。**常常是空的** —— 大部分流量是
            心跳、大廳查詢這類與牌局無關的訊息。
    """

    frame: DumpFrame
    message: LiqiMessage
    events: tuple[MjaiEvent, ...] = ()


@dataclass
class MjaiDecoder:
    """把 frame 流轉成 MJAI 事件流,依連線分流。

    Args:
        schema: 共用的協定 schema。載入一次約數百毫秒,重複建立 decoder 時傳入。

    Note:
        **有狀態,而且狀態很重要。** liqi 的回應要靠先前的請求才知道該用哪個
        型別解;MJAI 的轉換要記住立直、寶牌時機、最後亮牌的人。所以一個
        decoder 對應一條 frame 流,不能中途換人也不能倒帶。
    """

    schema: LiqiSchema | None = None
    stats: DumpStats = field(default_factory=DumpStats)
    _parsers: dict[str, LiqiParser] = field(default_factory=dict, repr=False)
    _converters: dict[str, MajsoulToMjai] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.schema is None:
            self.schema = LiqiSchema.load()

    @property
    def seat(self) -> int | None:
        """自己的座位,``authGame`` 出現之後才有值。

        多條連線時取第一條認得座位的 —— 實務上只有一條是對局伺服器,
        其餘(大廳、公告)不會產生 ``start_game``。
        """
        for converter in self._converters.values():
            if converter.state.names:
                return converter.seat
        return None

    def feed(self, frame: DumpFrame) -> DecodedFrame | None:
        """吃一個 frame。回傳解出來的東西,或 ``None`` 表示這個 frame 解不開。

        解不開**不拋例外**:錄影裡本來就會混進解不開的 frame(協定改版、
        非對局訊息),為此中斷整條流是不對的。累計在 :attr:`stats` 裡。
        """
        self.stats.total += 1
        parser = self._parsers.get(frame.flow)
        if parser is None:
            parser = self._parsers[frame.flow] = LiqiParser(self.schema)
            self.stats.flows = len(self._parsers)

        try:
            message = parser.parse(frame.payload, from_client=frame.from_client)
        except LiqiParseError as exc:
            self.stats.record_failure(str(exc))
            return None

        self.stats.parsed += 1
        if message.is_action:
            self.stats.actions += 1
            if message.deobfuscated:
                self.stats.obfuscated_actions += 1

        converter = self._converters.get(frame.flow)
        if converter is None:
            # 與訊息流共用同一個 parser。轉換器只會用它的
            # parse_action_prototype 去解 GameRestore 夾帶的動作,而那個方法
            # 不碰請求/回應的配對狀態,所以共用不會讓兩邊互相干擾。
            converter = self._converters[frame.flow] = MajsoulToMjai(parser)
        return DecodedFrame(frame, message, tuple(converter.handle(message)))

    def decode_file(self, path: Path | str) -> Iterator[DecodedFrame]:
        """把整個錄影檔餵進去。解不開的 frame 直接跳過。"""
        for frame in iter_frames(path):
            decoded = self.feed(frame)
            if decoded is not None:
                yield decoded
