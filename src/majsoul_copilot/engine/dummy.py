"""規則式 baseline 引擎:只看向聽與進張,不用類神經網路。

存在的理由是**比較的基準**。「Mortal 打得好」這句話要有意義,得先有一個
「不好但不亂」的對照組 —— 隨機打牌太弱,比了不說明任何事;這個引擎會挑
向聽數最小、進張最多的那一張,已經是不少人類玩家的水準,Mortal 贏它多少
才是有資訊量的數字。功能 3 比較不同風格的權重時,它同樣是那條基準線。

計算本體直接用功能 1 的 :mod:`majsoul_copilot.analysis.shanten` —— 同一份
程式碼在兩個功能裡跑,CV 那條路與封包這條路的建議因此保證一致。

刻意不做的事
------------
**不鳴牌、不立直、不宣告和了。** 這個引擎只回答一個問題:「輪到我打牌時,
純粹以聽牌速度而言該切哪張?」打點、安全度、場況、鳴牌的取捨全部不管。
把它擴充成會鳴牌的版本沒有意義 —— 那些判斷正是類神經網路的價值所在,
baseline 應該停在「規則能簡單算出來」的邊界上,才看得出差距來自哪裡。

因此它**不能拿去真的打一場牌**(不會和牌),只能作為打牌選擇的對照。
"""

from __future__ import annotations

from majsoul_copilot.analysis import HandError, suggest_discards
from majsoul_copilot.engine.base import Advice, EngineError
from majsoul_copilot.mjai import Chi, Dahai, MjaiEvent, Pon, Tsumo, mjai_to_ms, ms_to_mjai
from majsoul_copilot.mjai.handstate import HandTracker, UnknownSeatError
from majsoul_copilot.mjai.tiles import UNKNOWN
from majsoul_copilot.utils.logging import logger

__all__ = ["DummyEngine"]


class DummyEngine:
    """向聽最小化的規則式引擎。

    Args:
        name: 引擎名稱。

    Note:
        與子程序引擎不同,這個是**行程內**執行的 —— 它只用到 ``mahjong``
        套件,沒有 cp312 或 torch 的限制,隔一層行程只會增加延遲。
        兩者滿足同一個 :class:`~majsoul_copilot.engine.base.AIEngine` 介面,
        呼叫端不需要知道差別。
    """

    def __init__(self, name: str = "baseline") -> None:
        self._name = name
        # 手牌追蹤與 eval 那邊共用同一份實作 —— 兩邊各寫一份的話,CV 準確率
        # 報告會拿這裡的錯誤當標準答案,那個數字就完全沒有意義了
        self._tracker = HandTracker(_name=name)
        self._started = False

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    def __enter__(self) -> DummyEngine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def react(self, event: MjaiEvent) -> Advice:
        if not self._started:
            raise EngineError(f"{self._name}: 尚未啟動 —— 先呼叫 start()")

        try:
            self._tracker.feed(event)
        except UnknownSeatError as exc:
            raise EngineError(f"{self._name}: {exc}") from exc

        # 摸牌之後要打一張,鳴牌之後**也**要 —— 吃碰不從牌山補牌,鳴完直接輪到
        # 自己打。槓不在這裡:槓會從嶺上補一張,那張會以 tsumo 的形式再送進來。
        if isinstance(event, Tsumo | Chi | Pon) and event.actor == self._tracker.seat:
            return Advice(self._name, self._choose_discard())
        return Advice(self._name, None)

    def __repr__(self) -> str:
        tracker = self._tracker
        return f"<DummyEngine {self._name} seat={tracker.seat} tiles={len(tracker.tiles)}>"

    # ------------------------------------------------------------------ 決策

    def _choose_discard(self) -> MjaiEvent | None:
        hand = self._tracker.tiles
        if UNKNOWN in hand:
            logger.warning(f"{self._name}: 手牌含未知牌,跳過這一手 — {hand}")
            return None

        try:
            options = suggest_discards([mjai_to_ms(t) for t in hand])
        except HandError as exc:
            logger.warning(f"{self._name}: 算不出打牌建議({exc})— 手牌 {hand}")
            return None
        if not options:
            return None

        pai = self._pick_tile(options[0].tile)
        actor = self._tracker.seat or 0
        return Dahai(actor=actor, pai=pai, tsumogiri=pai == self._tracker.drawn)

    def _pick_tile(self, label: str) -> str:
        """把「切 5m」對應到手上實際的那一張,普通牌優先於赤寶牌。

        向聽計算把赤五與普通五視為同一種(它只影響打點),所以建議回來的是
        ``5m``。但手上可能同時有 ``5m`` 與 ``5mr``,而它們**不能互換** ——
        切掉赤五等於白丟一番。這裡固定留下赤五。
        """
        hand = self._tracker.tiles
        wanted = ms_to_mjai(label)
        if wanted in hand:
            return wanted
        red = f"{wanted}r"
        if red in hand:
            return red
        # 對不上只可能是手牌追蹤壞了。回原本的建議讓錯誤在下游顯現,
        # 而不是在這裡挑一張無關的牌把問題蓋掉。
        logger.warning(f"{self._name}: 建議切 {wanted} 但手上沒有 — {hand}")
        return wanted
