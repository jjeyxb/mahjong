"""放銃危險度:切這張牌會不會點炮。

不編機率
--------
「放銃率 12.3%」這種數字要有意義,得從一份夠大的牌譜語料統計出來。手上只有
兩場,差三四個數量級,而 ``docs/decisions.md`` 未解 #5 已經記著訓練語料這件事
本身就卡著。硬湊一個公式生出百分比,會得到一個**看起來精確、實際上沒有來源**
的數字 —— 正是這個專案第十七節在講的那種錯誤:不會報錯,只會給你一個合理的
錯答案。

所以這裡輸出的是**還有幾種待牌型沒被排除**。那是推論不是估計,每一條都能指回
它憑什麼,而且極端情況剛好落在對的地方:現物是 0 種,那不是「很低」,是規則
保證的不可能。

怎麼數
------
一張牌 ``T`` 可能被哪些待牌型榮和,是可以窮舉的:

======  ==============================  ==========================
型      牌                              也會和的另一張
======  ==============================  ==========================
両面    ``(T+1, T+2)``                  ``T+3``
両面    ``(T-2, T-1)``                  ``T-3``
嵌張    ``(T-1, T+1)``                  ——
単騎    ``T``                           ——
雙碰    ``T, T``                        ——
======  ==============================  ==========================

(``T±3`` 超出 1~9 時那一型是**辺張**,只和 ``T`` 一張。)

然後一型一型排除:

* **對方打過這張** → 振聽,一型都不剩。這是規則,不是估計。
* **那一型也會和的另一張,對方打過** → 他若有那一型早就和了 → 排除。
  這就是**筋**,而它在這裡是自然掉出來的,不必特別寫一條規則。
* **那一型需要的牌四張全見** → 他不可能持有 → 排除。這就是**壁**。
* **単騎需要他手上有一張、雙碰需要兩張** → 剩餘張數不夠就排除。字牌的
  「三枚見え」「四枚見え」就是這一條。

剩下幾型,就是這張牌現在還有幾條路會中。

對「每一家」都要算,不是只算立直的
------------------------------------
這一節是被真實牌譜逼出來的。``tests/fixtures/real_game_full.jsonl`` 裡有三次
榮和,**和牌的人一個都沒有立直**。第一版只評估立直家,結果對其中一次給出
「8s 安全 —— 現物」,而那張牌正是放銃牌 —— 它對立直的 0 家確實是現物,
榮和的卻是沒立直的 3 家。模組沒說錯話,但畫面上寫「安全」就是在騙人。

關鍵是把規則想窄了。上面那四條排除法,**只有一條是立直專屬的**:

* 振聽(他自己打過)—— 對**所有人**成立,而且整局有效。
* 筋(那一型也會和的另一張他打過)—— 對**所有人**成立。理由同上:他若現在
  聽那一型,就同時聽著自己打過的牌,那是振聽。
* 壁(四張全見)—— 與聽不聽牌無關。
* **立直之後別人打過的** —— 這條才是立直專屬的。沒立直的人隨時可以換聽。

所以評估的對象是**三家對手**,立直只是其中一種「確定聽牌」的標記。

沒立直的人危險度會偏高,因為我們不知道他到底聽不聽牌。**那是誠實的**:
不知道就是不知道,而把不知道畫成安全正是這個模組要避免的事。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from mia.mjai.table import TableTracker
from mia.mjai.tiles import normalize_red, tile_name

__all__ = [
    "DangerLevel",
    "DangerReport",
    "SeatDanger",
    "TileDanger",
    "Wait",
    "assess",
]

_SUITS = ("m", "p", "s")


class DangerLevel(IntEnum):
    """危險度。由「還有幾型沒被排除」換算,不是機率。"""

    SAFE = 0
    LIKELY_SAFE = 1
    RISKY = 2
    VERY_RISKY = 3

    @property
    def label(self) -> str:
        return _LEVEL_LABELS[self]


_LEVEL_LABELS = {
    DangerLevel.SAFE: "安全",
    DangerLevel.LIKELY_SAFE: "比較安全",
    DangerLevel.RISKY: "危險",
    DangerLevel.VERY_RISKY: "非常危險",
}

#: 剩幾型 → 危險度。
#:
#: 上限隨牌而異,那是**對的**:中張最多五型(兩個両面 + 嵌張 + 単騎 + 雙碰),
#: 么九牌少一個両面與嵌張,字牌只有単騎與雙碰。老頭牌與字牌本來就比較安全,
#: 而這裡的理由是「能打中它的型比較少」,不是統計上的印象。
def _level(waits: int) -> DangerLevel:
    if waits == 0:
        return DangerLevel.SAFE
    if waits <= 2:
        return DangerLevel.LIKELY_SAFE
    if waits <= 4:
        return DangerLevel.RISKY
    return DangerLevel.VERY_RISKY


@dataclass(frozen=True, slots=True)
class Wait:
    """一種還沒被排除的待牌型。

    Attributes:
        kind: ``両面`` / ``辺張`` / ``嵌張`` / ``単騎`` / ``雙碰``。
        tiles: 對方手上要有的那幾張。
    """

    kind: str
    tiles: tuple[str, ...]

    def __str__(self) -> str:
        if not self.tiles:
            return self.kind
        return f"{self.kind}({''.join(tile_name(t) for t in self.tiles)})"


@dataclass(frozen=True, slots=True)
class SeatDanger:
    """對某一家的危險度。

    Attributes:
        seat: 哪一家。
        waits: 還沒被排除的待牌型。空的代表**不可能**放銃給他。
        furiten: 他打過這張牌 —— 振聽,規則保證的安全。
        reach: 他有沒有立直。立直代表**確定聽牌**;沒立直不代表沒聽,
            只代表我們不知道。
    """

    seat: int
    waits: tuple[Wait, ...]
    furiten: bool = False
    reach: bool = False

    @property
    def level(self) -> DangerLevel:
        return _level(len(self.waits))

    @property
    def reason(self) -> str:
        """為什麼是這個等級。給人看的一句話。"""
        if self.furiten:
            return "現物"
        if not self.waits:
            return "所有待牌型都排除了"
        return "、".join(str(w) for w in self.waits)


@dataclass(frozen=True, slots=True)
class TileDanger:
    """切某一張牌的危險度,對所有要防的家合起來看。

    Attributes:
        tile: 這張牌(正規化過的 MJAI 記法)。
        seats: 每一家一份。空的代表現在沒有人需要防。
    """

    tile: str
    seats: tuple[SeatDanger, ...]

    @property
    def level(self) -> DangerLevel:
        """取最危險的那一家 —— 放銃只需要中一個人。

        **含沒立直的家。** 只看立直家的話,一張對立直者是現物、對別家全新的
        牌會被標成「安全」,而使用者會照著打出去。實測那正是真實牌譜裡發生
        過的事,見模組說明。
        """
        return max((s.level for s in self.seats), default=DangerLevel.SAFE)

    @property
    def against_reach(self) -> DangerLevel:
        """只看已立直的家。沒有人立直時是 :attr:`DangerLevel.SAFE`。

        比 :attr:`level` 銳利,因為立直是**確定聽牌**。兩個一起看才完整:
        「對立直家安全、對其他家危險」與「對誰都危險」是很不一樣的處境。
        """
        return max((s.level for s in self.seats if s.reach), default=DangerLevel.SAFE)

    @property
    def waits(self) -> int:
        """三家加起來還有幾型。**排序用的總數**,不是等級的依據 ——
        等級取的是最危險那一家(放銃只需要中一個人)。"""
        return sum(len(s.waits) for s in self.seats)

    @property
    def worst_waits(self) -> int:
        """最危險那一家還有幾型。這個才與 :attr:`level` 對得起來。"""
        return max((len(s.waits) for s in self.seats), default=0)

    @property
    def reason(self) -> str:
        """最危險那一家的理由。全部安全時說為什麼安全。"""
        if not self.seats:
            return "沒有人需要防"
        return _worst(self.seats).reason

    def __str__(self) -> str:
        return f"{tile_name(self.tile)} {self.level.label}(最多 {self.worst_waits} 型)"


@dataclass(frozen=True, slots=True)
class DangerReport:
    """一次評估的結果。

    Attributes:
        reached: 已宣告立直的家。空的**不代表安全** —— 只代表沒有人確定聽牌,
            而實測放銃給沒立直的人比給立直的人更常見。
        tiles: 手上每一張牌一份,由安全到危險排序。
        seat: 自己坐哪。UI 要靠它把絕對座位換成「上家 / 對家 / 下家」——
            「對 3 家危險」要在腦裡換算,「對下家危險」直接就能用。
    """

    reached: tuple[int, ...]
    tiles: tuple[TileDanger, ...]
    seat: int | None = None

    def __bool__(self) -> bool:
        return bool(self.tiles)

    def __str__(self) -> str:
        who = "".join(f"{s}家" for s in self.reached) or "無人"
        return f"立直 {who}:" + " ".join(str(t) for t in self.tiles[:5])


def _worst(seats: Sequence[SeatDanger]) -> SeatDanger:
    """最該提的那一家。

    平手時**優先講立直的那個**:危險度一樣的時候,「確定聽牌的人」比「可能
    根本沒聽的人」值得寫在那一行上。不寫死的話 ``max`` 會回第一個,而那只是
    座位編號的順序 —— 看起來有道理,其實是巧合。
    """
    return max(seats, key=lambda s: (s.level, len(s.waits), s.reach))


def _ranks(tile: str) -> tuple[int, str] | None:
    """``3m`` → ``(3, "m")``。字牌與認不得的東西回 ``None``。"""
    if len(tile) == 2 and tile[1] in _SUITS and tile[0].isdigit() and tile[0] != "0":
        return int(tile[0]), tile[1]
    return None


def _sequence_waits(
    tile: str, safe: set[str], remaining: dict[str, int]
) -> list[Wait]:
    """順子系的待牌型(両面 / 辺張 / 嵌張)裡,還沒被排除的那些。"""
    parsed = _ranks(tile)
    if parsed is None:
        return []  # 字牌沒有順子
    rank, suit = parsed

    def name(n: int) -> str:
        return f"{n}{suit}"

    def usable(*ns: int) -> bool:
        """對方有沒有可能同時持有這幾張。四張全見就是不可能 —— 壁。"""
        return all(1 <= n <= 9 and remaining.get(name(n), 0) > 0 for n in ns)

    waits: list[Wait] = []
    # 両面 / 辺張:(rank+1, rank+2) 也和 rank+3;(rank-2, rank-1) 也和 rank-3
    for low, high, partner in ((rank + 1, rank + 2, rank + 3), (rank - 2, rank - 1, rank - 3)):
        if not usable(low, high):
            continue
        if 1 <= partner <= 9:
            # 它也會和 partner。對方打過 partner 就代表他沒有這一型 —— 筋。
            if name(partner) in safe:
                continue
            waits.append(Wait("両面", (name(low), name(high))))
        else:
            waits.append(Wait("辺張", (name(low), name(high))))
    # 嵌張:只和這一張,沒有筋可言
    if usable(rank - 1, rank + 1):
        waits.append(Wait("嵌張", (name(rank - 1), name(rank + 1))))
    return waits


def _pair_waits(tile: str, remaining: dict[str, int]) -> list[Wait]:
    """単騎與雙碰。要對方手上真的有牌才成立,所以看剩餘張數。"""
    left = remaining.get(tile, 0)
    waits: list[Wait] = []
    if left >= 1:
        waits.append(Wait("単騎", (tile,)))
    if left >= 2:
        waits.append(Wait("雙碰", (tile, tile)))
    return waits


def assess(
    hand: list[str] | tuple[str, ...], table: TableTracker
) -> DangerReport:
    """評估手上每一張牌切出去的危險度。

    Args:
        hand: 自己的暗手牌(MJAI 記法,含剛摸的那張)。赤五會被正規化。
        table: 目前的桌面狀態。

    Returns:
        每張牌一份,由安全到危險排序。**三家對手都會算**,不是只算立直的家
        —— 理由見模組說明,那是被真實牌譜逼出來的。

    Note:
        「安全」只有一個意思:**對三家都不可能放銃**。對其中一家是現物但對
        另外兩家全新的牌不算安全,那正是實測踩過的坑。
    """
    seats_to_check = table.opponents
    if not seats_to_check or not hand:
        return DangerReport((), ())

    tiles = sorted({normalize_red(t) for t in hand})
    remaining = {t: table.remaining(t, hand) for t in _candidate_tiles(tiles)}

    report = []
    for tile in tiles:
        seats = []
        for seat in seats_to_check:
            player = table.players[seat]
            if tile in player.safe:
                seats.append(SeatDanger(seat, (), furiten=True, reach=player.reach))
                continue
            waits = (
                *_sequence_waits(tile, player.safe, remaining),
                *_pair_waits(tile, remaining),
            )
            seats.append(SeatDanger(seat, tuple(waits), reach=player.reach))
        report.append(TileDanger(tile, tuple(seats)))

    report.sort(key=lambda d: (d.level, d.against_reach, d.waits, d.tile))
    return DangerReport(tuple(table.threats), tuple(report), table.seat)


def _candidate_tiles(tiles: list[str]) -> set[str]:
    """要查剩餘張數的牌:手上那些,加上它們前後兩張(壁與両面會用到)。"""
    needed = set(tiles)
    for tile in tiles:
        parsed = _ranks(tile)
        if parsed is None:
            continue
        rank, suit = parsed
        for delta in (-2, -1, 1, 2):
            if 1 <= rank + delta <= 9:
                needed.add(f"{rank + delta}{suit}")
    return needed
