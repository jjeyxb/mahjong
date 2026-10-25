"""寶牌指示牌的逐幀觀察 → ``Dora`` 事件。

一條規則撐起整個設計
--------------------
**同一局之內,翻開的指示牌只會增加。** 槓會多翻一張,但不會有東西把已經翻開
的蓋回去 —— 所以「這一幀比上一幀少」一定是畫面的問題,不是牌桌的事件。

實測這件事確實會發生:``20260801-110639`` 有 11 幀把已經翻開的指示牌讀成
「沒翻開」,而那 11 幀**全部**落在指示牌變化後的 0.2~1.6 秒內 —— 翻牌與發牌
的動畫。沒有一次散在局中。

這跟牌河的「讀不到」剛好相反,值得並排記著:

* **牌河**的棄權是**永久**的。牌不會動,同一塊像素讀一百次還是同一個答案
  (實測:觀察次數翻四倍,補上的未知一個都沒多)。
* **寶牌**的讀不到是**瞬間**的。它是動畫造成的,而動畫會結束。

兩者症狀一樣(「這一幀讀不到」),處理方式相反:牌河要把未知帶進事件流讓
下游處理,寶牌直接忽略就好。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mia.vision.tiles.classify import MIN_BRIGHTNESS, Match

__all__ = ["DoraWatcher", "NewDora"]


@dataclass(frozen=True, slots=True)
class NewDora:
    """翻開了一張新的指示牌。

    Attributes:
        index: 第幾槽(0 是起始指示牌,之後是槓寶)。
        indicator: **指示牌**本身,不是寶牌。換算是規則的事
            —— ``9m`` 的下一張是 ``1m`` 而不是 ``10m``,那屬於 ``mia.mjai``。
    """

    index: int
    indicator: str


@dataclass
class DoraWatcher:
    """盯著寶牌面板,把新翻開的指示牌講成事件。

    與 :class:`~mia.tracker.rivers.RiverWatcher` 的差別是這裡**不需要未知**:
    實測讀到的時候 363 次全對,一次都沒認錯。所以規則是「讀得有把握就收,
    沒把握就當這一幀沒看到」——反正下一幀它還在那裡,而且不會變。
    """

    indicators: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """新的一局。指示牌全部重來。"""
        self.indicators = []

    def observe(self, slots: Sequence[Match | None]) -> list[NewDora]:
        """收一次觀察,回傳新翻開的指示牌。

        Args:
            slots: :func:`~mia.vision.tiles.dora.read_dora` 的輸出。

        Returns:
            新的指示牌,依槽序。沒有新的就是空串列。
        """
        fresh: list[NewDora] = []
        for index, match in enumerate(slots):
            if index < len(self.indicators):
                continue  # 已經收過的槽不再看 —— 它不會變
            if match is None or not self._trust(match):
                # 讀不到就停在這裡。**不要跳過這一槽去看下一槽** ——
                # 指示牌是從左往右翻的,第 n 槽還沒翻開的話第 n+1 槽
                # 不可能已經翻開,讀到的話那是雜訊。
                break
            self.indicators.append(match.label)
            fresh.append(NewDora(index=index, indicator=match.label))
        return fresh

    @staticmethod
    def _trust(match: Match) -> bool:
        """這個讀數可以收嗎。

        用 ``classify`` 自己的門檻就夠 —— 寶牌是正面朝鏡頭、沒有遮擋,
        實測分數 0.929、差距 0.391,遠高於牌河(0.6~0.7)。不需要牌河那組
        刻意嚴格的門檻。
        """
        return match.is_confident and match.brightness >= MIN_BRIGHTNESS
