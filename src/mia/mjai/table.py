"""從 MJAI 事件流追蹤**整桌**的公開資訊。

:mod:`~mia.mjai.handstate` 只追自己的暗手牌,理由是別家的手牌永遠蓋著。
這個模組追的是另一半:**攤在桌上、每個人都看得到的東西** —— 每一家的牌河、
副露、有沒有立直、寶牌指示牌。放銃分析需要的正是這些。

為什麼要分成兩個追蹤器
----------------------
不是為了檔案大小,是因為**它們的可信度不同**。自家手牌有兩個來源(畫面與
封包)可以互相對照;桌面資訊只有封包一個來源,畫面那條路根本看不到
(``RoiSet`` 只有 ``own_hand`` 一個區域,見 ``docs/decisions.md``)。
混在一起會讓「這份資料從哪來的」變得不明確,而放銃分析的每一條結論都要
能指回它憑什麼。

安全牌是**累積**的,不是查表查出來的
------------------------------------
:attr:`Player.safe` 是這個模組唯一需要小心的東西。一張牌對某一家安全,只有
兩種來源,兩種都是**振聽**這條規則直接保證的,不是估計:

1. **他自己打過。** 打過的牌不能榮和,整局有效。
2. **他立直之後,任何人打過。** 立直之後手牌不能再變,所以那一巡他若能和
   就一定和了 —— 沒和就代表不是他的待牌(之後也永遠不能和)。

第二條是安全牌的主要來源,而且**只對立直的人成立**。沒立直的人可以隨時換
聽,別人剛剛打過的牌對他完全不安全。把這兩者混為一談是初學者最常見的錯,
也是這個模組刻意把 ``safe`` 存成每家一份而不是全域一份的原因。

**沒有實作同巡振聽。** 見 :attr:`Player.safe` 的說明。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    StartGame,
    StartKyoku,
)
from mia.mjai.tiles import normalize_red

__all__ = ["SEATS", "Player", "TableTracker"]

#: 四人麻將。三麻不在範圍內 —— 北拔きドラ 與座位數都不一樣。
SEATS = 4

#: 一種牌總共幾張。
COPIES = 4


@dataclass(slots=True)
class Player:
    """一家攤在桌上的資訊。牌一律是**正規化過的** MJAI 記法(赤五當普通五)。

    Attributes:
        river: 打出去的牌,依序。被別家鳴走的也留著 —— 振聽看的是「他打過」,
            不是「那張牌現在在哪」。
        melds: 副露,每組一個 tuple(完整內容,給人看的)。
        revealed: 副露之中**從自己手上拿出來**的那幾張。數「場上看得見幾張」
            時要用這個而不是 ``melds``:被鳴走的那張已經記在打者的 ``river``
            裡,兩邊都數就會得到五張 3m。而高估看得見的張數會讓還有活牌的
            危險牌被判成安全 —— 錯的方向。
        reach: 有沒有立直(以 ``reach`` 宣告為準,不等宣告成立)。
        reach_turn: 立直宣言牌是他的第幾張捨牌。``None`` 表示還沒立直。
        safe: 對這一家**確定安全**的牌。見模組說明 —— 這是振聽規則保證的,
            不是估計出來的。

    Note:
        **沒有實作同巡振聽。** 一家放過一次榮和機會之後,到他下次摸牌為止是
        暫時振聽的。那段時間很短,而且要正確追蹤得知道他「能和而沒和」——
        那需要推測他的手牌。少算的後果是**把安全牌當成不安全**,方向是保守的,
        所以先不做。
    """

    river: list[str] = field(default_factory=list)
    melds: list[tuple[str, ...]] = field(default_factory=list)
    revealed: list[str] = field(default_factory=list)
    reach: bool = False
    reach_turn: int | None = None
    safe: set[str] = field(default_factory=set)

    @property
    def is_threat(self) -> bool:
        """需不需要防他。

        目前只認立直 —— 那是封包直接宣告的,毫無歧義。副露聽牌的判斷要靠
        捨牌與副露去推,那是估計,混進來會讓「安全」這兩個字失去意義。
        """
        return self.reach


@dataclass(slots=True)
class TableTracker:
    """整桌的公開資訊,隨 MJAI 事件流更新。

    Attributes:
        seat: 自己的座位。``start_game`` 之前是 ``None``。
        players: 四家,索引即 ``actor``。
        dora_markers: 寶牌指示牌(不是寶牌本身)。它們也是**看得見的牌**,
            要算進 :meth:`visible`。
    """

    seat: int | None = None
    players: list[Player] = field(default_factory=lambda: [Player() for _ in range(SEATS)])
    dora_markers: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 更新

    def handle(self, event: MjaiEvent) -> None:
        """吃一個事件。認不得的型別直接忽略。"""
        match event:
            case StartGame():
                self.seat = event.id
            case StartKyoku():
                self._reset(event)
            case Dahai():
                self._discard(event.actor, event.pai)
            case Reach():
                self._reach(event.actor)
            case ReachAccepted():
                pass  # 宣告當下就開始防了,不等它成立
            case Chi() | Pon() | Daiminkan():
                # 被鳴的那張來自別人的河,那裡已經數過了
                self._meld(event.actor, (event.pai, *event.consumed), event.consumed)
            case Ankan():
                # 四張全部來自手上
                self._meld(event.actor, tuple(event.consumed), event.consumed)
            case Kakan():
                # 加槓只多亮一張(手上那張第四張);原本那組碰早就記過了
                self._upgrade(event.actor, (event.pai, *event.consumed), event.pai)
            case Dora():
                self.dora_markers.append(normalize_red(event.dora_marker))
            case Kita():
                pass  # 三麻才有,不在範圍內
            case _:
                pass

    def _reset(self, event: StartKyoku) -> None:
        """新的一局。**所有東西都要清掉** —— 上一局的安全牌在這一局毫無意義,
        而留著不會有任何症狀,只會安靜地把危險牌說成安全。"""
        self.players = [Player() for _ in range(SEATS)]
        self.dora_markers = [normalize_red(event.dora_marker)]

    def _discard(self, actor: int, tile: str) -> None:
        pai = normalize_red(tile)
        if not self._valid(actor):
            return
        self.players[actor].river.append(pai)
        # 自己打過的牌自己不能榮和,整局有效
        self.players[actor].safe.add(pai)
        # 立直的人若能和這張就一定和了 —— 沒和就代表永遠不能和
        for index, player in enumerate(self.players):
            if index != actor and player.reach:
                player.safe.add(pai)

    def _reach(self, actor: int) -> None:
        if not self._valid(actor):
            return
        player = self.players[actor]
        player.reach = True
        # 宣言牌是「這一次 dahai」,而 reach 事件在它之前 —— 所以記的是
        # 目前的長度,那正好是宣言牌之後的索引
        player.reach_turn = len(player.river)

    def _meld(self, actor: int, tiles: tuple[str, ...], from_hand: Sequence[str]) -> None:
        if not self._valid(actor):
            return
        player = self.players[actor]
        player.melds.append(tuple(normalize_red(t) for t in tiles))
        player.revealed.extend(normalize_red(t) for t in from_hand)

    def _upgrade(self, actor: int, tiles: tuple[str, ...], added: str) -> None:
        """加槓:把原本那組碰換成四張,只多亮一張。

        換而不是新增 —— 新增的話同一組會在畫面上出現兩次,而張數也會多數三張。
        """
        if not self._valid(actor):
            return
        player = self.players[actor]
        upgraded = tuple(normalize_red(t) for t in tiles)
        base = normalize_red(added)
        for index, meld in enumerate(player.melds):
            if len(meld) == 3 and all(t == base for t in meld):
                player.melds[index] = upgraded
                break
        else:  # pragma: no cover - 沒看到對應的碰,只能當成新的一組
            player.melds.append(upgraded)
        player.revealed.append(base)

    def _valid(self, actor: int) -> bool:
        return 0 <= actor < SEATS

    # ------------------------------------------------------------------ 查詢

    @property
    def threats(self) -> list[int]:
        """需要防的座位。自己不算 —— 不會放銃給自己。"""
        return [
            i for i, p in enumerate(self.players) if p.is_threat and i != self.seat
        ]

    def visible(self, own_hand: list[str] | tuple[str, ...] = ()) -> Counter[str]:
        """場上看得見的每種牌各幾張。

        含自己的手牌:自己手上那幾張別人一定沒有,那是壁與字牌判斷的一半依據。
        **不含**別家的暗手牌與牌山(那正是要推測的東西)。
        """
        seen: Counter[str] = Counter(normalize_red(t) for t in own_hand)
        seen.update(self.dora_markers)
        for player in self.players:
            seen.update(player.river)
            seen.update(player.revealed)
        return seen

    def remaining(self, tile: str, own_hand: list[str] | tuple[str, ...] = ()) -> int:
        """這種牌還有幾張沒露面。0 表示四張都看得到了。"""
        return max(0, COPIES - self.visible(own_hand)[normalize_red(tile)])

    def __str__(self) -> str:
        reached = ",".join(str(i) for i in self.threats) or "無"
        turns = max((len(p.river) for p in self.players), default=0)
        return f"TableTracker(座位 {self.seat} 第 {turns} 巡 立直 {reached})"
