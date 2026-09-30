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
from mia.mjai.tiles import normalize_honor, normalize_red

__all__ = [
    "DRAGONS",
    "MELD_THREAT",
    "MELD_THREAT_WITH_YAKUHAI",
    "SEATS",
    "SUIT_READ_MELDS",
    "Player",
    "TableTracker",
]

#: 四人麻將。三麻不在範圍內 —— 北拔きドラ 與座位數都不一樣。
SEATS = 4

#: 一種牌總共幾張。
COPIES = 4

#: 幾組副露之後視為**推定聽牌**。
#:
#: 三組是有理由的:副露三組之後暗手牌只剩四張,能組的東西已經很少,而且
#: 願意鳴到第三下的人手上通常有役、正在趕聽。實測那場東風戰裡三副露的曝光量
#: 與立直是同一個量級(53 對 59 個捨牌時點),而三次榮和裡最重的那一次
#: 正是一個三副露、沒立直的人和的。
#:
#: 這**是估計不是推論**,與這個模組其他部分不同。所以它只影響「要不要把他
#: 當成威脅」(標題、排序、對誰),不影響危險度等級 —— 等級仍然只由「還剩
#: 幾種待牌型」決定,那條鏈不能斷。
#:
#: 2026-09-28 拿 14,544 場鳳凰卓牌譜量過,這個門檻**漏掉一個威脅度相同的狀態**,
#: 見 :data:`MELD_THREAT_WITH_YAKUHAI`。
MELD_THREAT = 3

#: 有役牌副露時,幾組就算威脅。
#:
#: 原本只有 :data:`MELD_THREAT` 這個純計數門檻。14,544 場鳳凰卓、每一個捨牌
#: 時點對每一家各算一筆(共約 400 萬筆)量出來的條件放銃率:
#:
#: ====================  ==========  ===============
#: 對手狀態              放銃率      95% CI
#: ====================  ==========  ===============
#: 1 副露、無役牌        0.811%      ——
#: 1 副露、**有**役牌    0.750%      ——
#: 2 副露、無役牌        1.257%      ——
#: 2 副露、**有**役牌    1.488%      [1.459, 1.518]
#: 3 副露、無役牌        1.465%      [1.358, 1.579]
#: ====================  ==========  ===============
#:
#: **「2 副露 + 役牌」與「3 副露無役牌」的信賴區間重疊** —— 威脅度是同一級,
#: 但舊門檻只認得後者。實測牌譜裡我們自己那次放銃,和牌的人正是 2 副露
#: (``docs/decisions.md`` 未解 #15)。那不是運氣不好,是門檻有洞。
#:
#: 注意**一副露有役牌反而是 0.93x**(0.750% 對 0.811%,樣本各 125 萬與 150 萬,
#: 不是雜訊)。所以「碰了役牌就有役、門檻該掉到一副露」是錯的 —— 單獨一個
#: 役牌碰多半是慢手的起手,不是聽牌訊號。門檻只降到 2,不再往下。
MELD_THREAT_WITH_YAKUHAI = 2

#: 三元牌。役牌永遠包含這三種,與場風自風無關。
DRAGONS = ("P", "F", "C")

#: 風牌依座位順序。``_WINDS[(座位 - 莊家) % 4]`` 就是那一家的自風。
_WINDS = ("E", "S", "W", "N")

#: 總副露幾組之後,「數牌全同色」才算染手(混一色 / 清一色)的形狀。
#:
#: 看的是**總組數**,不是同色的數牌組數 —— 這是量出來的,不是想出來的。
#: 14,544 場裡取**自摸**和牌(和牌張是從牌山摸的,沒有被任何人挑選過,
#: 所以是對他待牌組成的無偏估計),看和牌張落在他集中的那個花色的比例,
#: 虛無假設是 33.3%:
#:
#: =========================  ======  ============  ================
#: 對手副露形狀               樣本    同色佔數牌    95% CI
#: =========================  ======  ============  ================
#: 1 組數牌(無字牌)         5192    **25.4%**     [24.2, 26.6]
#: 1 組數牌 + 1 組字牌        ——      31.2%         ——
#: 2 組同色(無字牌)         715     36.8%         [33.3, 40.4]
#: 2 組同色 + 1 組字牌        436     **50.5%**     [45.8, 55.1]
#: 1 組數牌 + 2 組字牌        ——      **50.3%**     ——
#: 3 組同色                   91      **81.3%**     [72.1, 88.0]
#: =========================  ======  ============  ================
#:
#: 兩件事跟直覺不同:
#:
#: 1. **單獨一組副露是負訊號**(25.4%,信賴區間完全不碰 33.3%)。吃一組萬子
#:    之後他的待牌**更不可能**在萬子 —— 那個面子已經做完了,剩下的形在別處。
#: 2. **帶訊號的是字牌組,不是同色的數牌組數。** 2 組同色沒字牌只有 36.8%,
#:    加一組字牌跳到 50.5%;而 1 組數牌配 2 組字牌同樣是 50.3%。混一色允許
#:    字牌面子,所以字牌組與數牌組在這裡是等價的。
#:
#: 所以門檻是「總副露 ≥ 3 且數牌全同色」。3 組同色(81.3%)更強,但沒有另外
#: 分級 —— 兩種都遠高於基準,而多一級只會讓畫面多一種要解釋的狀態。
SUIT_READ_MELDS = 3


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
        yakuhai: 對**這一家**來說算役牌的字牌:三元牌 + 場風 + 他的自風。
            自風要知道莊家是誰才算得出來,所以由 :meth:`TableTracker._reset`
            在每一局開始時填進來。預設只有三元牌 —— 那是在沒收到
            ``start_kyoku`` 時最保守的值(少算風牌會**低估**威脅,不會高估)。

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
    yakuhai: frozenset[str] = frozenset(DRAGONS)

    @property
    def melded(self) -> int:
        """副露幾組。"""
        return len(self.melds)

    @property
    def has_yakuhai(self) -> bool:
        """副露裡有沒有役牌。有就代表他**已經有役**,隨時可以聽牌。

        這是門檻降到 :data:`MELD_THREAT_WITH_YAKUHAI` 的依據,而那個門檻是
        量出來的,不是推的 —— 數字見該常數的說明。

        Note:
            比對前一定要過 :func:`~mia.mjai.tiles.normalize_honor`。副露記下來
            的可能是 ``1z`` 也可能是 ``E``(看事件來源),而 :attr:`yakuhai`
            存的是字母式。少了這一步門檻會**靜靜地不生效**。
        """
        return any(normalize_honor(m[0]) in self.yakuhai for m in self.melds)

    @property
    def suit_read(self) -> str | None:
        """他的副露像不像染手,是的話回那個花色(``m`` / ``p`` / ``s``)。

        條件是「總副露 ≥ :data:`SUIT_READ_MELDS` 組,且數牌副露全在同一色」。
        字牌組不破壞條件 —— 混一色本來就允許字牌面子,而實測**帶訊號的正是
        字牌組**(見 :data:`SUIT_READ_MELDS`)。

        Note:
            這**是估計不是推論**。三組同色萬子的人仍然可以做斷么九或對對和,
            那時他的待牌可以在任何花色 —— 實測也確實只有 81.3% 落在同色,
            不是 100%。所以它跟 :data:`MELD_THREAT` 一樣只影響「怎麼描述他、
            怎麼排序」,**不影響危險度等級**。
        """
        if len(self.melds) < SUIT_READ_MELDS:
            return None
        suits = {s for m in self.melds if (s := _suit_of(m[0])) is not None}
        if len(suits) != 1:
            return None
        return next(iter(suits))

    @property
    def is_threat(self) -> bool:
        """需不需要防他:立直、副露到 :data:`MELD_THREAT` 組,或
        :data:`MELD_THREAT_WITH_YAKUHAI` 組而其中有役牌。

        立直是封包直接宣告的,毫無歧義;後兩者是估計(見那兩個常數)。
        兩者混在同一個屬性裡是刻意的 —— 它回答的是「防不防他」,而那個答案
        兩種情況都是要。**「安全」兩個字的意思不受影響**:那是由振聽與排除法
        決定的,與他聽不聽牌無關。

        Note:
            立直家會隨著局面推進**越來越好防** —— 宣告之後別人打過的牌對他
            都安全,那份安全牌一路累積。三副露的人拿不到這個扣抵,所以到中盤
            之後他每一張牌的危險度通常**高於**立直家,不只是持平。
        """
        if self.reach or self.melded >= MELD_THREAT:
            return True
        return self.melded >= MELD_THREAT_WITH_YAKUHAI and self.has_yakuhai


def _suit_of(tile: str) -> str | None:
    """``3m`` → ``"m"``。字牌回 ``None``(它不屬於任何數牌花色)。"""
    suit = tile[-1]
    return suit if suit in ("m", "p", "s") else None


@dataclass(slots=True)
class TableTracker:
    """整桌的公開資訊,隨 MJAI 事件流更新。

    Attributes:
        seat: 自己的座位。``start_game`` 之前是 ``None``。
        players: 四家,索引即 ``actor``。
        dora_markers: 寶牌指示牌(不是寶牌本身)。它們也是**看得見的牌**,
            要算進 :meth:`visible`。
        bakaze: 場風。``start_kyoku`` 之前是 ``None``。
        oya: 這一局的莊家。``start_kyoku`` 之前是 ``None``。
    """

    seat: int | None = None
    players: list[Player] = field(default_factory=lambda: [Player() for _ in range(SEATS)])
    dora_markers: list[str] = field(default_factory=list)
    bakaze: str | None = None
    oya: int | None = None

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
        而留著不會有任何症狀,只會安靜地把危險牌說成安全。

        順便發役牌:自風是座位相對莊家算出來的,每一局莊家會換,所以這件事
        只能在這裡做。場風與莊家也要留著,因為 :attr:`Player.yakuhai` 是每家
        一份而不是全桌一份 —— 同一張東,對莊家是役牌,對西家不是。
        """
        self.bakaze = event.bakaze
        self.oya = event.oya
        self.players = [
            Player(yakuhai=frozenset((*DRAGONS, event.bakaze, _WINDS[(seat - event.oya) % SEATS])))
            for seat in range(SEATS)
        ]
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
    def opponents(self) -> list[int]:
        """除了自己以外的座位。座位未知時是全部四家。"""
        return [i for i in range(SEATS) if i != self.seat]

    @property
    def threats(self) -> list[int]:
        """**確定或推定聽牌**的座位(立直、或三副露)。自己不算。

        這仍然不等於「要防的人」:沒立直也沒副露的人一樣會榮和,而實測三次
        榮和裡有兩次是沒立直的人和的(見 ``docs/decisions.md`` 第十八節)。
        這個屬性回答的是「誰值得被指名」,不是「只有這些人要防」——
        :func:`~mia.analysis.danger.assess` 一律評估三家。
        """
        return [i for i in self.opponents if self.players[i].is_threat]

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
