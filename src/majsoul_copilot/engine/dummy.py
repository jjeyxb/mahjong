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
from majsoul_copilot.mjai import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    StartKyoku,
    Tsumo,
    mjai_to_ms,
    ms_to_mjai,
)
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
        self._seat: int | None = None
        self._hand: list[str] = []
        self._drawn: str | None = None
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

        self._track(event)
        # 摸牌之後要打一張,鳴牌之後**也**要 —— 吃碰不從牌山補牌,鳴完直接輪到
        # 自己打。槓不在這裡:槓會從嶺上補一張,那張會以 tsumo 的形式再送進來。
        if isinstance(event, Tsumo | Chi | Pon) and event.actor == self._seat:
            return Advice(self._name, self._choose_discard())
        return Advice(self._name, None)

    def __repr__(self) -> str:
        return f"<DummyEngine {self._name} seat={self._seat} tiles={len(self._hand)}>"

    # ------------------------------------------------------------------ 手牌追蹤

    def _track(self, event: MjaiEvent) -> None:
        """從事件流維護自己的暗手牌。

        只追蹤自己 —— baseline 不需要知道別家的牌河與副露(它也不會拿來用)。
        別家的手牌本來就看不到。
        """
        if event.TYPE == "start_game":
            self._seat = getattr(event, "id", None)
            return

        if isinstance(event, StartKyoku):
            if self._seat is None:
                raise EngineError(
                    f"{self._name}: 收到 start_kyoku 前沒有 start_game,不知道自己是誰"
                )
            self._hand = list(event.tehais[self._seat])
            self._drawn = None
            return

        actor = getattr(event, "actor", None)
        if actor is None or actor != self._seat:
            return

        match event:
            case Tsumo():
                self._hand.append(event.pai)
                self._drawn = event.pai
            case Dahai():
                self._remove(event.pai)
                self._drawn = None
            case Chi() | Pon() | Daiminkan():
                # 被鳴的那張來自別家,不在自己手上;只扣掉自己拿出來的
                for tile in event.consumed:
                    self._remove(tile)
                # 鳴牌打出的不會是剛摸的那張 —— 根本沒摸
                self._drawn = None
            case Ankan():
                for tile in event.consumed:
                    self._remove(tile)
                self._drawn = None
            case Kakan():
                self._remove(event.pai)
                self._drawn = None
            case Kita():
                self._remove(event.pai)
                self._drawn = None

    def _remove(self, tile: str) -> None:
        try:
            self._hand.remove(tile)
        except ValueError:
            # 手牌對不上代表事件流漏了或重了。這個引擎自己不會因此崩潰,但
            # 之後的建議全都不能信,所以要留下痕跡而不是靜靜跳過。
            logger.warning(f"{self._name}: 手上沒有 {tile},事件流可能不完整 — 目前 {self._hand}")

    # ------------------------------------------------------------------ 決策

    def _choose_discard(self) -> MjaiEvent | None:
        if UNKNOWN in self._hand:
            logger.warning(f"{self._name}: 手牌含未知牌,跳過這一手 — {self._hand}")
            return None

        try:
            options = suggest_discards([mjai_to_ms(t) for t in self._hand])
        except HandError as exc:
            logger.warning(f"{self._name}: 算不出打牌建議({exc})— 手牌 {self._hand}")
            return None
        if not options:
            return None

        pai = self._pick_tile(options[0].tile)
        return Dahai(actor=self._seat or 0, pai=pai, tsumogiri=pai == self._drawn)

    def _pick_tile(self, label: str) -> str:
        """把「切 5m」對應到手上實際的那一張,普通牌優先於赤寶牌。

        向聽計算把赤五與普通五視為同一種(它只影響打點),所以建議回來的是
        ``5m``。但手上可能同時有 ``5m`` 與 ``5mr``,而它們**不能互換** ——
        切掉赤五等於白丟一番。這裡固定留下赤五。
        """
        wanted = ms_to_mjai(label)
        if wanted in self._hand:
            return wanted
        red = f"{wanted}r"
        if red in self._hand:
            return red
        # 對不上只可能是手牌追蹤壞了。回原本的建議讓錯誤在下游顯現,
        # 而不是在這裡挑一張無關的牌把問題蓋掉。
        logger.warning(f"{self._name}: 建議切 {wanted} 但手上沒有 — {self._hand}")
        return wanted
