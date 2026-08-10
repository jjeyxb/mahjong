"""工作執行緒 → UI 執行緒的單向郵箱。

為什麼不是佇列
--------------
兩個生產者的節奏差很多:擷取執行緒 15 fps 定速,封包執行緒事件驅動、一陣一陣。
消費端是 Qt 主執行緒,而它會被自己的事情打斷(重繪、選單、視窗拖曳)。

用佇列的話,主執行緒卡住 0.5 秒就積了 7 幀,恢復後會**照順序把 7 幀都套一遍**
—— 使用者看到手牌快速跳過一段歷史才追上現在。那比「掉幀」糟得多:掉幀只是
少看一眼,補播是在螢幕上顯示已經不成立的局面。

所以這裡**每個 slot 只留最新一筆**。舊的直接丟掉,因為它已經沒有意義了。

slot 就是型別
-------------
一個 dataclass 一個 slot,合併的鍵直接用 ``type(update)``。這樣不需要另外
維護一組字串常數,消費端也能用 ``match`` 窮盡地分派。

為什麼建議與手牌是兩個 slot(而不是一個大 update):大部分 MJAI 事件不產生
建議。塞在同一個 slot 裡的話,「摸牌→建議切 3s」之後緊接著別人打牌的事件
會把那個建議連帶覆蓋掉 —— 明明還該顯示著。分開就沒有這個問題:手牌合併成
最新的手牌,建議合併成最新的建議,兩者互不影響。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from mia.engine.base import Advice

__all__ = ["Advices", "CvHand", "PacketHand", "Update", "UpdateBus"]


@dataclass(frozen=True, slots=True)
class CvHand:
    """CV 從畫面認到的手牌。

    Attributes:
        tiles: 暗手牌,**雀魂記法**(``0m`` / ``1z``)—— 那是 classify 的輸出,
            轉成 MJAI 記法是 ViewModel 的事。
        drawn: 剛摸進來那張;沒有就是 ``None``。
        confident: 比對分數是否都夠有把握。
    """

    tiles: tuple[str, ...]
    drawn: str | None = None
    confident: bool = True


@dataclass(frozen=True, slots=True)
class PacketHand:
    """封包推出來的手牌(MJAI 記法,含剛摸進來那張)。"""

    tiles: tuple[str, ...]
    drawn: str | None = None


@dataclass(frozen=True, slots=True)
class Advices:
    """引擎這一手的建議。"""

    advices: tuple[Advice, ...]


Update = CvHand | PacketHand | Advices


@dataclass
class BusStats:
    """郵箱的流量。``dropped`` 不是零就代表 UI 跟不上生產者。

    這個數字**該被看到**:它是「畫面上的手牌落後於實際」的唯一線索,
    不然那種延遲只會表現成使用者說不清楚的「怪怪的」。
    """

    posted: int = 0
    dropped: int = 0
    drained: int = 0

    @property
    def drop_ratio(self) -> float:
        return self.dropped / self.posted if self.posted else 0.0


class UpdateBus:
    """執行緒安全、每個 slot 只留最新一筆的郵箱。

    生產者呼叫 :meth:`post`,消費者(UI 執行緒)定期呼叫 :meth:`drain`。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # dict 保持插入順序,而重新指派同一個 key 不會改變它的位置 ——
        # 所以 drain 出來的順序是「這一輪各 slot 第一次被投遞的順序」,
        # 對同一組生產者是穩定的。
        self._pending: dict[type[Update], Update] = {}
        self.stats = BusStats()

    def post(self, update: Update) -> None:
        """投一筆更新。同一個型別未被取走時直接覆蓋。"""
        key = type(update)
        with self._lock:
            if key in self._pending:
                self.stats.dropped += 1
            self._pending[key] = update
            self.stats.posted += 1

    def drain(self) -> tuple[Update, ...]:
        """取走所有待處理的更新並清空。沒有東西時回空的 tuple。"""
        with self._lock:
            if not self._pending:
                return ()
            items = tuple(self._pending.values())
            self._pending.clear()
            self.stats.drained += len(items)
        return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def __repr__(self) -> str:
        return (
            f"<UpdateBus pending={len(self)} posted={self.stats.posted} "
            f"dropped={self.stats.dropped}>"
        )


@dataclass
class WorkerStatus:
    """一個工作執行緒對外的一句話。

    刻意只有一句話,而不是一組結構化欄位:它唯一的用途是顯示在狀態列上,
    而狀態列只有一行。工作執行緒內部的細節走日誌。

    ``message`` 為空字串表示「一切正常,沒有話要說」—— 正常運作時不該佔用
    狀態列,那個位置要留給真的出問題的那個執行緒。
    """

    name: str
    message: str = ""
    alive: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def say(self, message: str) -> None:
        with self._lock:
            self.message = message

    def read(self) -> str:
        with self._lock:
            return self.message

    def clear_if(self, message: str) -> bool:
        """訊息還是 ``message`` 的話就清掉。回傳有沒有真的清掉。

        用比對而不是直接清:要清的那一刻,寫的那一邊可能已經換成別的話
        (「子程序已結束」之類),那句是壞消息,不能被一個「一切正常」蓋掉。
        比對與清除必須在同一個鎖裡,分兩次呼叫就留了一道縫。
        """
        with self._lock:
            if self.message != message:
                return False
            self.message = ""
            return True
