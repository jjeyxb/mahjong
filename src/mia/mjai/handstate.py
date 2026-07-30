"""從 MJAI 事件流追蹤自己的暗手牌。

兩個地方需要同一份邏輯,所以抽出來共用:

* :class:`~mia.engine.dummy.DummyEngine` 要知道手上有什麼才能
  算該切哪張。
* :mod:`mia.eval` 要拿封包推出來的手牌當 **ground truth**,
  用來量 CV 認得對不對。

兩邊各寫一份的話,CV 準確率報告會拿「baseline 的錯誤」當標準答案 —— 那個數字
就完全沒有意義了。

只追蹤自己
----------
別家的手牌永遠蓋牌,封包也不會送(除非終局亮牌)。這個追蹤器只回答
「**我**手上有哪些牌」,那正好也是功能 1 的 CV 唯一看得到的東西。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mia.mjai.events import (
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
)
from mia.utils.logging import logger

__all__ = ["HandTracker", "UnknownSeatError"]


class UnknownSeatError(ValueError):
    """還不知道自己坐哪 —— ``start_game`` 沒來過。"""


@dataclass(slots=True)
class HandTracker:
    """自己的暗手牌,隨事件流更新。

    Attributes:
        seat: 自己的座位,由 ``start_game`` 決定。
        tiles: 目前的暗手牌(MJAI 表示法),**含剛摸進來那張**。
        drawn: 這一巡摸到的牌;打過牌或鳴牌之後是 ``None``。
        melds: 副露組數。由張數反推,不記內容 —— 向聽計算只需要組數。
        desyncs: 對不上的次數(打了一張手上沒有的牌)。不是零就代表事件流
            有缺漏,這之後的手牌都不能信。
    """

    seat: int | None = None
    tiles: list[str] = field(default_factory=list)
    drawn: str | None = None
    desyncs: int = 0
    _name: str = "handstate"

    @property
    def melds(self) -> int:
        """副露組數。手牌張數不合法時回 -1。

        由**張數本身**判斷該對照 13 還是 14,不看 :attr:`drawn` —— 吃碰之後
        沒有摸牌(``drawn`` 是 None),但手上是 11 張、還欠一張沒打,狀態等同
        14 張。用 ``drawn`` 判斷會在每次鳴牌後算錯一組副露。

        張數除以 3 的餘數就足以區分:餘 1 是「等別人打」(13/10/7/4/1),
        餘 2 是「該我打了」(14/11/8/5/2),餘 0 不是合法的暗手牌張數。
        """
        remainder = len(self.tiles) % 3
        if remainder == 0:
            return -1
        expected = 13 if remainder == 1 else 14
        return (expected - len(self.tiles)) // 3

    @property
    def in_sync(self) -> bool:
        return self.desyncs == 0

    def feed(self, event: MjaiEvent) -> None:
        """吃一個事件並更新手牌。

        Raises:
            UnknownSeatError: 在 ``start_game`` 之前收到 ``start_kyoku``。
        """
        if event.TYPE == "start_game":
            self.seat = getattr(event, "id", None)
            return

        if isinstance(event, StartKyoku):
            if self.seat is None:
                raise UnknownSeatError("收到 start_kyoku 前沒有 start_game,不知道自己是誰")
            self.tiles = list(event.tehais[self.seat])
            self.drawn = None
            self.desyncs = 0
            return

        if getattr(event, "actor", None) != self.seat:
            return

        match event:
            case Tsumo():
                self.tiles.append(event.pai)
                self.drawn = event.pai
            case Dahai():
                self._remove(event.pai)
                self.drawn = None
            case Chi() | Pon() | Daiminkan():
                # 被鳴的那張來自別家,不在自己手上;只扣掉自己拿出來的。
                # 吃碰不補牌,鳴完直接輪到自己打 —— 所以 drawn 是 None。
                for tile in event.consumed:
                    self._remove(tile)
                self.drawn = None
            case Ankan():
                for tile in event.consumed:
                    self._remove(tile)
                self.drawn = None
            case Kakan():
                self._remove(event.pai)
                self.drawn = None
            case Kita():
                self._remove(event.pai)
                self.drawn = None

    def _remove(self, tile: str) -> None:
        try:
            self.tiles.remove(tile)
        except ValueError:
            # 對不上代表事件流漏了或重了。不拋例外 —— 一場錄影裡有一段壞掉,
            # 其餘的部分仍然有用;但要留下痕跡,而且 in_sync 之後恆為 False,
            # 免得有人拿一份已經歪掉的手牌當標準答案。
            self.desyncs += 1
            logger.warning(f"{self._name}: 手上沒有 {tile},事件流可能不完整 — 目前 {self.tiles}")
