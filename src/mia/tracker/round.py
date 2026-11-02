"""HUD 的讀數 → 「這一局是哪一局」、「誰立直了」。

HUD 不是事件,是一直掛在畫面上的狀態,所以問題只有一個:**哪一次讀數可以信**。
結算面板、立繪蓋住的時候,欄位會各自棄權,或者讀到上一局的點數。

採信的條件與副露同一套:連續兩次讀到一模一樣,而且要過點數的檢查碼
(:attr:`~mia.vision.hud.Hud.scores_agree`)。一局只採信一次 —— 開局資訊
不會變;之後點數因為立直而變動,是別的事件的事。

場風**可以**是 ``None``:北場沒有模板,不能因此整局沒有莊家。

本場數也可以,只是要多等幾次:換局動畫中也會讀不到,而且動畫還沒演完時其他欄位可能是上一局的。

場風與本場數不靠模板也算得出來
------------------------------
模板只收得到打過的東西,而 9 本場、西入、北入錄不到素材。但這兩個數字都有規則:

* **本場數只有兩種可能。** 閒家和牌歸零,其他結果一定是上一局加一:莊家連莊、流局、中途流局都
  是。所以讀不到就是加一,沒有上限;讀到的數字不是這兩個的話就是看錯了。同一局(莊家沒換)
  不可能歸零,要加一。
* **局數變小就是換場風。** 南四局之後是西一局,西四局之後是北一局。

只有第一局看不到上一局。那時讀不到場風就當東場,場風錯了只是少算役牌,比沒有建議好。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mia.vision.hud import Hud

__all__ = ["ReachWatcher", "RoundWatcher"]


#: 立直要放一根 1000 點的棒子。
_REACH_COST = 1000

#: 本場數讀不到時,其他欄位要連續幾次一樣才採信。
_NO_HONBA_CONFIRM = 4

#: 場風的順序。
_WINDS = ("E", "S", "W", "N")


@dataclass
class RoundWatcher:
    """盯著 HUD,回報這一局的開局資訊。"""

    #: 連續幾次讀到一樣才採信。
    confirm: int = 2
    #: 採信的那一次。還沒有就是 ``None``。
    settled: Hud | None = None
    _last: Hud | None = field(default=None, repr=False)
    _same: int = field(default=0, repr=False)
    #: 上一局採信的。:meth:`reset` 不清:場風與本場數要靠它。
    _previous: Hud | None = field(default=None, repr=False)
    #: 這一場的總點數。:meth:`reset` 不清,見 :meth:`_total_of`。
    _total: int | None = field(default=None, repr=False)

    def reset(self) -> None:
        """新的一局。"""
        self.settled = None
        self._last = None
        self._same = 0

    def observe(self, hud: Hud) -> Hud | None:
        """收一次讀數。**這一次**剛好採信了就回傳它,其他時候回 ``None``。"""
        if self.settled is not None:
            return None
        hud = self._total_of(hud).filled()
        if not _complete(hud):
            self._last = None
            self._same = 0
            return None
        self._same = self._same + 1 if _same_round(hud, self._last) else 1
        self._last = hud
        if self._same < (self.confirm if hud.honba is not None else _NO_HONBA_CONFIRM):
            return None
        self.settled = self._previous = _carried(hud, self._previous)
        return self.settled


    def _total_of(self, hud: Hud) -> Hud:
        """總點數是算出來的:四家一樣多就是開局,乘四就是這一場的總點數。

        友人場可以改起始點數,寫死 100000 的話檢查碼永遠過不了,整場沒有事件流。只有四家完全相同
        才算:一位數讀錯湊不出四個一樣的數字,檢查碼的本事不會打折。
        """
        first = hud.scores[0]
        if first is not None and first > 0 and first % 100 == 0 and len(set(hud.scores)) == 1:
            self._total = first * len(hud.scores)
        return hud if self._total is None else replace(hud, total=self._total)


def _carried(hud: Hud, previous: Hud | None) -> Hud:
    """用上一局補上、校正這一局的場風與本場數。見模組說明。"""
    if previous is None or previous.kyoku is None or hud.kyoku is None:
        return replace(hud, bakaze=hud.bakaze or _WINDS[0])
    wind = hud.bakaze
    if wind is None:
        at = _WINDS.index(previous.bakaze or _WINDS[0])
        wind = _WINDS[(at + (hud.kyoku < previous.kyoku)) % len(_WINDS)]
    same = (wind, hud.kyoku) == (previous.bakaze, previous.kyoku)
    honba = hud.honba
    if not (same and honba == previous.honba):
        following = (previous.honba or 0) + 1
        if honba is None or same or honba not in (0, following):
            honba = following
    return replace(hud, bakaze=wind, honba=honba)


def _complete(hud: Hud) -> bool:
    return (
        hud.scores_agree
        and hud.kyoku is not None
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
    (白底紅點、紫色箭頭、木紋),而它底下那圈面板在即時對局常有別的東西掠過。
    點數有檢查碼,讀錯過不了;也看得到自己的立直。

    實測三份素材 19 次立直全抓到、沒有誤報。橫放牌那條路在 13 局的半莊上只抓到
    7 / 13 —— 它只在有後續的牌可以驗時才判得出來。橫放牌還是要找,牌河的格線要靠
    它挪開,但「誰立直了」改由這裡說。

    局尾的結算也會動點數,但那不是「剛好少 1000 而供託加同樣多」,會被擋掉。
    連續兩次讀到一樣才採信,與 :class:`RoundWatcher` 同一個理由。
    """

    confirm: int = 2
    _total: int | None = field(default=None, repr=False)
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
        self._total = hud.total

    def observe(self, hud: Hud) -> list[int]:
        """收一次讀數,回傳**這一次**剛立直的座位(依視角,通常是空的)。

        有一家一直讀不到時,當作他沒變再驗檢查碼。別家立直對得起來;棒子先動、點數還沒跟上的
        那一刻對不起來,不會算到他頭上。他自己立直看不出來,那要靠橫放牌。
        """
        if self._total is not None:
            hud = replace(hud, total=self._total)
        hud = hud.with_sticks()
        if self._base is not None and hud.scores.count(None) == 1:
            known = zip(hud.scores, self._base[0], strict=True)
            hud = replace(hud, scores=tuple(b if a is None else a for a, b in known))
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
