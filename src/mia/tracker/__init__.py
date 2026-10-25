"""把逐幀的觀察拼成事件流。

畫面辨識回答的是「**現在**桌上有什麼」,MJAI 要的是「**誰剛剛做了什麼**」。
中間這一層負責的就是那個轉換,而它不只是 diff —— 見 :mod:`mia.tracker.rivers`。
"""

from mia.tracker.rivers import UNKNOWN, Discard, RiverWatcher

__all__ = ["UNKNOWN", "Discard", "RiverWatcher"]
