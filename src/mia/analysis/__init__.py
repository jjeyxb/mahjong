"""手牌分析:向聽數、進張、打牌建議。

這一層只吃「有哪些牌」,不管那些牌是怎麼來的 —— 視覺辨識(功能 1)與封包
(功能 2)都可以餵進來。
"""

from mia.analysis.danger import (
    DangerLevel,
    DangerReport,
    SeatDanger,
    TileDanger,
    assess,
)
from mia.analysis.shanten import (
    AGARI,
    TENPAI,
    DiscardOption,
    HandAnalysis,
    HandError,
    Ukeire,
    analyse,
    suggest_discards,
)

__all__ = [
    "AGARI",
    "TENPAI",
    "DangerLevel",
    "DangerReport",
    "DiscardOption",
    "HandAnalysis",
    "HandError",
    "SeatDanger",
    "TileDanger",
    "Ukeire",
    "analyse",
    "assess",
    "suggest_discards",
]
