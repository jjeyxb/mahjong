"""HUD 的讀數 → 「這一局是哪一局」、「誰立直了」。

HUD 不是事件,是一直掛在畫面上的狀態,所以問題只有一個:**哪一次讀數可以信**。
結算面板、立繪蓋住的時候,欄位會各自棄權,或者讀到上一局的點數。

採信的條件與副露同一套:連續兩次讀到一模一樣,而且要過點數的檢查碼
(:attr:`~mia.vision.hud.Hud.scores_agree`)。一局只採信一次 —— 開局資訊
不會變;之後點數因為立直而變動,是別的事件的事。

場風**可以**是 ``None``:只認得東與南,西入時不能因此整局沒有莊家。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mia.vision.hud import Hud

__all__ = ["ReachWatcher", "RoundWatcher"]


#: 立直要放一根 1000 點的棒子。
_REACH_COST = 1000


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


@dataclass
class ReachWatcher:
    """點數少 1000、供託多 1 —— 有人立直了。

    **一局之內點數只有立直會動。** 所以這裡完全不看立直棒:棒子的造型每個玩家不同
    (白底紅點、紫色箭頭、木紋),而它底下那圈面板在 Steam 上常有別的東西掠過。
    點數有檢查碼,讀錯過不了;也看得到自己的立直。

    實測三份素材 19 次立直全抓到、沒有誤報。橫放牌那條路在 13 局的半莊上只抓到
    7 / 13 —— 它只在有後續的牌可以驗時才判得出來。橫放牌還是要找,牌河的格線要靠
    它挪開,但「誰立直了」改由這裡說。

    局尾的結算也會動點數,但那不是「剛好少 1000 而供託加同樣多」,會被擋掉。
    連續兩次讀到一樣才採信,與 :class:`RoundWatcher` 同一個理由。
    """

    confirm: int = 2
    _base: tuple[tuple[int | None, ...], int] | None = field(default=None, repr=False)
    _last: tuple[tuple[int | None, ...], int] | None = field(default=None, repr=False)
    _same: int = field(default=0, repr=False)

    def reset(self) -> None:
        """新的一局。"""
        self._base = None
        self._last = None
        self._same = 0

    def start(self, hud: Hud) -> None:
        """這一局開局的點數,之後都跟它比。"""
        self.reset()
        self._base = (hud.scores, hud.kyotaku or 0)

    def observe(self, hud: Hud) -> list[int]:
        """收一次讀數,回傳**這一次**剛立直的座位(依視角,通常是空的)。"""
        if self._base is None or not hud.scores_agree:
            self._last = None
            self._same = 0
            return []
        current = (hud.scores, hud.kyotaku or 0)
        if current == self._base:
            self._last = None
            self._same = 0
            return []
        self._same = self._same + 1 if current == self._last else 1
        self._last = current
        if self._same < self.confirm:
            return []
        before, sticks = self._base
        self._base = current
        changed = [i for i, (a, b) in enumerate(zip(before, current[0], strict=True)) if a != b]
        lost = [
            i for i in changed if (before[i] or 0) - (current[0][i] or 0) == _REACH_COST
        ]
        return lost if lost == changed and len(lost) == current[1] - sticks else []
