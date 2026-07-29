"""MJAI 協定的事件型別與牌表示法轉換。

這一層目前由封包解析路徑(:mod:`majsoul_copilot.groundtruth.to_mjai`)產生。
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
