"""即時模式:把畫面與封包接到 UI 上。

兩條路各有一條執行緒,交會點只有一個郵箱:

* :mod:`mia.live.vision` —— 擷取 → CV → 手牌(功能 1)
* :mod:`mia.live.packets` —— 封包 → MJAI → 引擎建議(功能 2)
* :mod:`mia.live.bus` —— 兩者共用的郵箱,每個 slot 只留最新一筆
* :mod:`mia.live.runtime` —— 在 UI 執行緒上把郵箱套進 ViewModel
* :mod:`mia.live.source` —— 封包擷取子程序的生命週期

**這個套件不 import Qt。** 呈現層的責任只有「定期呼叫
:meth:`~mia.live.runtime.LiveRuntime.pump`」,所以整條即時路徑不需要事件迴圈
就能測。
"""

from __future__ import annotations

from mia.live.bus import Advices, CvHand, PacketHand, Update, UpdateBus
from mia.live.packets import DumpTail, PacketWorker
from mia.live.runtime import LiveRuntime
from mia.live.source import CaptureProcess, capture_command
from mia.live.vision import VisionWorker

__all__ = [
    "Advices",
    "CaptureProcess",
    "CvHand",
    "DumpTail",
    "LiveRuntime",
    "PacketHand",
    "PacketWorker",
    "Update",
    "UpdateBus",
    "VisionWorker",
    "capture_command",
]
