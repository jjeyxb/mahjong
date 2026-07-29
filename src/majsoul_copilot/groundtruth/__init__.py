"""Ground Truth:從雀魂的 WebSocket 流量取得完美的牌局狀態。

**這一層不參與線上決策。** 它只有兩個用途:

1. 自動產生帶標註的資料集(畫面 ↔ 完美狀態的成對資料)
2. 量化 CV 辨識的準確率

線上路徑一律走視覺辨識,這是專題的主軸;GT 只是用來證明它到底準不準。
"""

from majsoul_copilot.groundtruth.dump import DumpFrame, DumpStats, iter_frames, parse_dump
from majsoul_copilot.groundtruth.liqi import (
    LiqiMessage,
    LiqiParseError,
    LiqiParser,
    MessageKind,
    deobfuscate,
)
from majsoul_copilot.groundtruth.schema import LiqiSchema, MethodSignature

__all__ = [
    "DumpFrame",
    "DumpStats",
    "LiqiMessage",
    "LiqiParseError",
    "LiqiParser",
    "LiqiSchema",
    "MessageKind",
    "MethodSignature",
    "deobfuscate",
    "iter_frames",
    "parse_dump",
]
