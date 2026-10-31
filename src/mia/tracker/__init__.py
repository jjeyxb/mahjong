"""把逐幀的觀察拼成事件流。

畫面辨識回答的是「**現在**桌上有什麼」,MJAI 要的是「**誰剛剛做了什麼**」。
中間這一層負責的就是那個轉換,而它不只是 diff —— 見 :mod:`mia.tracker.rivers`。
"""

from mia.tracker.dora import DoraWatcher, NewDora
from mia.tracker.kyoku import KyokuSignal, KyokuWatcher
from mia.tracker.melds import MeldWatcher, guess_meld
from mia.tracker.rivers import UNKNOWN, Discard, RiverWatcher
from mia.tracker.round import RoundWatcher
from mia.tracker.turns import Action, Step, TurnTracker

__all__ = [
    "UNKNOWN",
    "Action",
    "Discard",
    "DoraWatcher",
    "KyokuSignal",
    "KyokuWatcher",
    "MeldWatcher",
    "NewDora",
    "RiverWatcher",
    "RoundWatcher",
    "Step",
    "TurnTracker",
    "guess_meld",
]
