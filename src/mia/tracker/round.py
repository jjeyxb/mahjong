"""HUD 的讀數 → 「這一局是哪一局」。

HUD 不是事件,是一直掛在畫面上的狀態,所以問題只有一個:**哪一次讀數可以信**。
結算面板、立繪蓋住的時候,欄位會各自棄權,或者讀到上一局的點數。

採信的條件與副露同一套:連續兩次讀到一模一樣,而且要過點數的檢查碼
(:attr:`~mia.vision.hud.Hud.scores_agree`)。一局只採信一次 —— 開局資訊
不會變;之後點數因為立直而變動,是別的事件的事。

場風**可以**是 ``None``:現在只認得東,南場時不能因此整局沒有莊家。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mia.vision.hud import Hud

__all__ = ["RoundWatcher"]


@dataclass
class RoundWatcher:
    """盯著 HUD,回報這一局的開局資訊。"""

    #: 連續幾次讀到一樣才採信。
    confirm: int = 2
    #: 採信的那一次。還沒有就是 ``None``。
    settled: Hud | None = None
    _last: Hud | None = field(default=None, repr=False)
    _same: int = field(default=0, repr=False)

    def reset(self) -> None:
        """新的一局。"""
        self.settled = None
        self._last = None
        self._same = 0

    def observe(self, hud: Hud) -> Hud | None:
        """收一次讀數。**這一次**剛好採信了就回傳它,其他時候回 ``None``。"""
        if self.settled is not None:
            return None
        if not _complete(hud):
            self._last = None
            self._same = 0
            return None
        self._same = self._same + 1 if _same_round(hud, self._last) else 1
        self._last = hud
        if self._same < self.confirm:
            return None
        self.settled = hud
        return hud


def _complete(hud: Hud) -> bool:
    return (
        hud.scores_agree
        and hud.kyoku is not None
        and hud.honba is not None
        and hud.dealer is not None
    )


def _same_round(hud: Hud, last: Hud | None) -> bool:
    """余牌每一巡都在變,不能拿來比。其他欄位一局之內要一樣。"""
    return last is not None and (
        (hud.bakaze, hud.kyoku, hud.honba, hud.kyotaku, hud.scores, hud.dealer)
        == (last.bakaze, last.kyoku, last.honba, last.kyotaku, last.scores, last.dealer)
    )
