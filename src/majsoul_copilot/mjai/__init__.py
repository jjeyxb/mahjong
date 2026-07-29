"""MJAI 協定的事件型別與牌表示法轉換。

這一層由兩條路徑共用:Ground Truth(封包解析)與視覺辨識(M4 的 tracker)。
M5 的準確率評測會把兩條事件流對齊比較,所以它們必須產出同一種型別。
"""

from majsoul_copilot.mjai.events import (
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
from majsoul_copilot.mjai.tiles import TileError, mjai_to_ms, ms_to_mjai

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
    "TileError",
    "Tsumo",
    "mjai_to_ms",
    "ms_to_mjai",
]
