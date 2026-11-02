"""局的邊界:四家牌河**同時清空**,然後重新開始長。

訊號長什麼樣(實測,不是推測)
----------------------------
量 ``20261024-123746`` 的每一幀,四家的張數與連續性:

==================  ==================  ========
時刻                四家張數            連續
==================  ==================  ========
局進行中            11 10 9 12 = 43     4
**結算畫面**        **5 3 1 4 = 13**    **0**
新局開始            0 0 0 0 = 0         4
==================  ==================  ========

**和了之後張數不是掉到零,是亂跳。** 整桌被立繪與結算面板蓋住,而角色的膚色
與牌面的判斷條件一模一樣(亮、不飽和),所以讀出來是一堆假牌。文件原本寫著
「四家牌河同時清空,那個訊號很強」—— 一半對:清空確實發生,但**在那之前有
十幾秒的垃圾**,只看張數會在那段期間亂宣告邊界。

分得開的是**連續性**:結算畫面四家全部不連續(假牌散落在格子裡),而真正的
空桌是「零張而且四家都連續」。兩個條件一起用,實測整份錄影只在四次局開始
(加上開局前)命中,結算畫面一次都沒誤中。

為什麼要講兩件事
----------------
邊界不是一個瞬間,是一段**空白**:桌子清掉了,過一會兒新的牌才出現。
中間那段可以很長 —— 實測有一局是 **56 秒**(有人長考,畫面上是東 3 局、
手牌已發、四家牌河空無一物,計時器還在倒數)。

所以 :meth:`KyokuWatcher.observe` 回兩種訊號:

* :attr:`KyokuSignal.ENDED` —— 桌面清空了。**這時候就該把舊狀態丟掉**,
  不要等。不丟的話那 56 秒裡 :class:`~mia.tracker.rivers.RiverWatcher`
  還握著上一局的河,而新的觀察是空的 —— 對不上,於是每一幀都記一次
  「看不懂」。無害,但噪音會蓋掉真正的異常。
* :attr:`KyokuSignal.STARTED` —— 新的一局真的開始了(第一張牌出現)。
  **宣告新局要等到這裡**,因為對局結束後的空桌也會給 ``ENDED``,
  而那之後沒有牌跟上來(實測連續四十秒的「零張、四家連續」)。

桌面空了不一定是換局
--------------------
**第一張牌被鳴走時,四家的河也是全空的。** 實測東 3 局:上家第一張 9 索被自己碰走,張數從
1 掉到 0,四家連續,於是整局的配牌與進度全被清掉,那一局 95 秒只排出 3 個事件。

分得開的是 HUD:鳴牌不會改局數,換局一定會改(局或本場)。所以呼叫端可以告訴這裡 ``taken``,
意思是「局數本場沒變」。而一次鳴牌只拿得走一張,桌上有兩張以上就空了,那一定是換局。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from mia.vision.tiles.river import River

__all__ = ["KyokuSignal", "KyokuWatcher"]


class KyokuSignal(Enum):
    """一次觀察帶來的訊號。"""

    #: 沒有變化。絕大多數的幀都是這個。
    NONE = "none"
    #: 桌面清空了 —— 丟掉舊狀態。
    ENDED = "ended"
    #: 清空之後第一次看到牌 —— 新的一局開始。
    STARTED = "started"


@dataclass
class KyokuWatcher:
    """看四家的牌河,說出「新的一局開始了」。

    用法::

        if watcher.observe(rivers):
            river_watcher.reset()
            dora_watcher.reset()

    Attributes:
        empty: 目前停在「桌面是乾淨的」狀態。
        started: 到目前為止判定了幾局。
    """

    empty: bool = True
    started: int = 0
    #: 上一次四家都連續時桌上有幾張。動畫中的假牌不算。
    _last: int = field(default=0, repr=False)

    def observe(self, rivers: Sequence[River], *, taken: bool = False) -> KyokuSignal:
        """收一次觀察。

        Args:
            rivers: 四家的牌河,:func:`~mia.vision.tiles.river.read_river`
                的輸出。順序不重要 —— 這裡只看總量與連續性。
            taken: 局數本場沒有變。見模組說明的「桌面空了不一定是換局」。

        Returns:
            見 :class:`KyokuSignal`。``ENDED`` 與 ``STARTED`` 各自只在
            **轉折的那一刻**回一次,中間都是 ``NONE``。
        """
        total = sum(river.count for river in rivers)
        consistent = all(river.is_plausible() for river in rivers)

        last, self._last = self._last, total if consistent else self._last

        if total == 0 and consistent:
            if self.empty or (taken and last <= 1):
                return KyokuSignal.NONE
            self.empty = True
            return KyokuSignal.ENDED
        if not self.empty:
            return KyokuSignal.NONE
        if not consistent:
            # 清空之後、新局之前的垃圾(結算畫面、發牌動畫)。**不要收** ——
            # 收了會把假牌當成第一張捨牌,而那個錯誤會跟著整局。
            return KyokuSignal.NONE

        self.empty = False
        self.started += 1
        return KyokuSignal.STARTED

    def reset(self) -> None:
        """回到「還沒看過任何東西」。"""
        self.empty = True
        self.started = 0
        self._last = 0
