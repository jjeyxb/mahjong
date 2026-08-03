"""UI 要顯示什麼 —— 與 Qt 完全無關的那一半。

**這個模組不 import Qt。** 兩個原因:

1. 側邊視窗與 Overlay(M7)是兩種呈現方式,共用同一份狀態。狀態綁在某個
   widget 上的話,第二種呈現就得把它抄一遍。
2. 沒有 Qt 才測得動。GUI 測試要起事件迴圈、要有顯示裝置,在 CI 上很痛苦;
   而真正容易錯的是「該顯示什麼」而不是「像素畫在哪」。

兩個來源,兩種資訊
------------------
* **功能 1(CV)**:從畫面認出手牌 → 向聽、進張、該切哪張。不碰連線。
* **功能 2(封包)**:MJAI 事件流 → 引擎建議 + Q 值。

兩者的手牌**可能不一致** —— CV 認錯,或畫面還在演動畫。這種時候不去猜哪邊對:
:attr:`ViewState.hand_conflict` 標出來讓 UI 顯示警示,因為「兩邊對不上」本身
就是使用者該知道的資訊。

刻意不做的事
------------
不在這裡做節流或去抖動。畫面 15 fps 進來、封包事件驅動,兩者的更新頻率差很多,
但「多久更新一次畫面」是呈現層的問題(Qt 有自己的機制),不是狀態的問題。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

from mia.analysis import (
    AGARI,
    TENPAI,
    DiscardOption,
    HandAnalysis,
    HandError,
    analyse,
    suggest_discards,
)
from mia.engine.actions import Candidate, action_label, action_tiles, decode_candidates
from mia.engine.base import Advice
from mia.mjai.tiles import UNKNOWN, mjai_to_ms, ms_to_mjai

__all__ = ["EngineView", "ViewModel", "ViewState", "q_fraction", "shanten_text"]


def q_fraction(q: float, *, low: float, high: float) -> float:
    """一個 Q 值的長條要畫多滿(0~1)。

    Q 值是**相對的** —— 同一手之內互相比較才有意義,不同局面之間的絕對值不可比
    (實測見過 +2.7 也見過 -6.8)。所以每一手都拿這一手的最高/最低重新正規化,
    不用固定刻度:固定刻度的話大部分局面的長條會全部擠在同一端。

    只有一個候選時 ``high == low``,畫滿而不是除以零。
    """
    span = high - low
    return 1.0 if span <= 0 else (q - low) / span


def shanten_text(shanten: int) -> str:
    """向聽數要寫成什麼字。

    放在這裡而不是各自寫在 widget 裡:側邊視窗的標題、打牌選項清單、Overlay
    三個地方都要這一句,而「0 是聽牌、-1 是和了」寫錯了畫面上看起來仍然正常
    —— 只是數字差一。
    """
    if shanten == AGARI:
        return "和了"
    if shanten == TENPAI:
        return "聽牌"
    return f"{shanten} 向聽"


@dataclass(frozen=True, slots=True)
class EngineView:
    """一個引擎在這一手給的東西。

    Attributes:
        name: 引擎名稱。
        action: 建議的動作,人看的字串。``None`` 表示這一手不需要動作。
        tile: 這個動作**在講哪張牌**(MJAI 記法)。切牌是切掉的那張,鳴牌是
            被鳴的那張。UI 靠它決定要不要畫牌面圖。
        consumed: 鳴牌時要從**自己手上**拿出來的那幾張。
            吃牌一定要顯示 —— ``吃 3m`` 可以是 1m2m、2m4m 或 4m5m 三種吃法,
            不寫出來使用者不知道該點哪兩張。
        is_discard: 這個動作是不是切牌。與 :attr:`tile` 分開,因為鳴牌的
            :attr:`tile` **不在自己手上**(那是別家打出來的),
            :attr:`ViewState.advice_is_stale` 不能拿它去比手牌。
        follow_up_tile: 立直之後要切的那張。見 :attr:`Advice.follow_up`。
        candidates: 所有合法動作與 Q 值,最高分在前。規則式引擎沒有,是空的。
        latency_ms: 這一手花了多久。
    """

    name: str
    action: str | None = None
    tile: str | None = None
    candidates: tuple[Candidate, ...] = ()
    latency_ms: float = 0.0
    declined: bool = False
    consumed: tuple[str, ...] = ()
    is_discard: bool = False
    follow_up_tile: str | None = None

    @property
    def shown_tiles(self) -> tuple[str, ...]:
        """這一手要畫成牌面圖的全部牌,由左而右。

        立直的後續切牌也算進來 —— 使用者按下立直之後馬上就要選那張牌。
        """
        tiles = [t for t in (self.tile,) if t]
        tiles.extend(self.consumed)
        if self.follow_up_tile:
            tiles.append(self.follow_up_tile)
        return tuple(tiles)

    @property
    def has_reasoning(self) -> bool:
        """有沒有 Q 值可以解釋「為什麼」。"""
        return bool(self.candidates)

    @property
    def headline(self) -> str:
        """要放在最上面那行的字。

        三種狀態,而不是兩種:有動作、**跳過**、沒人在問。中間那個原本被歸到
        「不需要動作」,於是遊戲跳出「碰 / 槓 / 跳過」時畫面上什麼都不說。
        """
        if self.action is not None:
            return self.action
        return "跳過" if self.declined else "不需要動作"

    # 底下四個屬性合起來就是「大字 + 一排牌」那一行。側邊視窗與 Overlay 都用
    # 它們,而不是各自去拆 :attr:`action` 的字串 —— 「吃要寫出 consumed」與
    # 「立直要畫出後續切牌」這兩件事都是實機打過一場才發現的,抄成兩份遲早
    # 會有一份漏掉,而漏掉的那一份看起來仍然正常。

    @property
    def verb(self) -> str:
        """大字那個動詞。牌交給圖去講,所以這裡不含牌名。

        與 :attr:`headline` 的差別:那個是完整的一句話(``吃 3m ← 1m 2m``),
        用在「其他引擎怎麼說」那種並排比較;這個只有動詞,因為旁邊就畫著牌。
        """
        if self.action is None:
            return self.headline
        if self.is_discard:
            return "切"
        if self.follow_up_tile:
            # 立直:動詞就是「立直」,重點的牌是後續要切的那張(在 own_tiles)
            return self.action
        return self.action.split(" ")[0]

    @property
    def subject(self) -> str | None:
        """動詞後面那一張 —— 動作在講的牌。沒有就是 ``None``。

        立直沒有 subject:立直本身不指向任何一張牌。
        """
        if self.action is None or self.follow_up_tile:
            return None
        return self.tile

    @property
    def joiner(self) -> str:
        """:attr:`subject` 與 :attr:`own_tiles` 中間的那個字。

        它是唯一讓「桌上那張」與「自己手上那幾張」分得開的東西 —— 三張一樣
        大小排在一起,使用者看不出該點哪幾張。
        """
        if self.action is None:
            return ""
        if self.follow_up_tile:
            return "切"
        return "用" if self.consumed else ""

    @property
    def own_tiles(self) -> tuple[str, ...]:
        """要從**自己手上**拿出來的那幾張,由左而右。"""
        if self.action is None:
            return ()
        if self.follow_up_tile:
            return (self.follow_up_tile,)
        return self.consumed


@dataclass(frozen=True, slots=True)
class ViewState:
    """UI 在某個瞬間要顯示的全部內容。

    是 frozen 的:每次更新產生一個新的 state,UI 拿到的東西不會在畫的中途被改掉。

    Attributes:
        hand: 目前的手牌(MJAI 記法),含剛摸進來那張。
        drawn: 剛摸進來那張。**由來源明確告知,不從 hand 猜** —— CV 那邊
            ``read_hand`` 本來就把它與暗手牌分開(槽位上真的有間隔),
            封包那邊 ``HandTracker`` 也記著。用「最後一張」去猜的話,
            排序過或副露後就會指錯。
        hand_source: 手牌是哪來的 —— ``cv`` / ``packet`` / ``both``。
        cv_hand: CV 認到的手牌。與 :attr:`hand` 分開存,才比較得出衝突。
        packet_hand: 封包推出來的手牌。
        analysis: 向聽與進張。手牌不合法時是 ``None``。
        discards: 純以聽牌速度而言的打牌選項,最好的在前。
        engines: 各引擎的建議,加入順序 —— 那個順序就是 :attr:`primary` 的優先序。
        preferred_engine: 使用者點名要當主角的引擎;``None`` 表示照順序取。
        notices: 要顯示給使用者的警示(校正未鎖定、引擎掉隊、CV 沒把握…)。
    """

    hand: tuple[str, ...] = ()
    drawn: str | None = None
    hand_source: str = ""
    cv_hand: tuple[str, ...] = ()
    cv_drawn: str | None = None
    packet_hand: tuple[str, ...] = ()
    packet_drawn: str | None = None
    analysis: HandAnalysis | None = None
    discards: tuple[DiscardOption, ...] = ()
    engines: tuple[EngineView, ...] = ()
    preferred_engine: str | None = None
    notices: tuple[str, ...] = ()

    @property
    def hand_conflict(self) -> bool:
        """CV 與封包認到的手牌不一致。

        兩邊都有值才算。只有一邊時沒有東西可比 —— 那不是衝突,是資訊不足。
        """
        if not self.cv_hand or not self.packet_hand:
            return False
        return sorted(self.cv_hand) != sorted(self.packet_hand)

    @property
    def best_discard(self) -> DiscardOption | None:
        """純速度最優的那一張。功能 1 分頁的主角。"""
        return self.discards[0] if self.discards else None

    @property
    def primary(self) -> EngineView | None:
        """要放在最上面的那個引擎。

        指定了 :attr:`preferred_engine` 就用它,即使它這一手不動作 —— 使用者
        點名要看某個引擎,那就該一直看它,不該因為它剛好沒事做就跳去別人。

        沒指定的話取**第一個有動作的**,所以 :attr:`engines` 的順序就是優先序,
        由建立引擎的那一端決定。規則式 baseline 要排在模型後面 —— 排前面的話
        headline 會顯示 baseline、而真正的模型被擠到下面那排,Q 值長條也會整段
        消失(baseline 沒有 meta)。
        """
        if self.preferred_engine:
            for engine in self.engines:
                if engine.name == self.preferred_engine:
                    return engine
        # 「跳過」也算有答案 —— 不然遊戲在問要不要鳴牌時,主角會被讓給一個
        # 什麼都沒說的引擎
        for engine in self.engines:
            if engine.action is not None or engine.declined:
                return engine
        return self.engines[0] if self.engines else None

    @property
    def advice_is_stale(self) -> bool:
        """建議切的那張牌已經不在手上 —— 那一手已經打完了。

        手牌與建議是**兩個獨立的更新槽**(理由見 :mod:`mia.live.bus`),所以
        即時模式下兩者最多會差一個事件:打牌的事件到了、手牌變成 13 張,但
        建議還是上一巡算出來的那個。差一個事件是刻意接受的代價 —— 合併成一個
        槽的話,別人打牌的事件會把還該顯示著的建議連帶蓋掉。

        代價換成這個旗標讓 UI 標出來:使用者看到「切 3s」而手上沒有 3s 時,
        該知道那是「剛剛打掉了」而不是程式算錯。

        沒有手牌或建議不是切牌時一律 False —— 立直、吃碰沒有對應的單張牌可比。
        """
        primary = self.primary
        # 只對切牌成立。鳴牌的那張是**別家打出來的**,本來就不在自己手上,
        # 拿它去比會讓每一個吃碰建議都被誤標成「已打出」。
        if primary is None or not primary.is_discard or not primary.tile or not self.hand:
            return False
        return primary.tile not in self.hand

    @property
    def is_unanimous(self) -> bool:
        """有動作的引擎是否給了同一個答案。分歧的那幾手才值得停下來看。"""
        actions = {e.action for e in self.engines if e.action is not None}
        return len(actions) <= 1


class ViewModel:
    """收集各來源的更新,產生 :class:`ViewState`。

    Args:
        on_change: 每次狀態變化時呼叫。用普通的 callable 而不是 Qt signal
            —— 這一層不該知道 Qt 的存在。

    Note:
        **不是執行緒安全的。** 擷取執行緒與封包執行緒都會來更新,呼叫端要自己
        序列化(Qt 那邊的做法是把更新丟回主執行緒)。這裡不加鎖是刻意的:
        加了也擋不住「UI 讀到一半狀態被換掉」,而 frozen state 才真的擋得住。
    """

    def __init__(self, on_change: Callable[[ViewState], None] | None = None) -> None:
        self._state = ViewState()
        self._listeners: list[Callable[[ViewState], None]] = []
        self._depth = 0
        self._dirty = False
        if on_change is not None:
            self._listeners.append(on_change)

    @property
    def state(self) -> ViewState:
        return self._state

    def subscribe(self, listener: Callable[[ViewState], None]) -> None:
        """加一個變化通知。側邊視窗與 Overlay 各自訂閱同一個 ViewModel。"""
        self._listeners.append(listener)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """把區塊內的多筆更新合成**一次**通知。

        即時模式一輪會一次套用「CV 手牌 + 封包手牌 + 引擎建議」三筆更新。
        逐筆通知的話 UI 每輪重畫三次,而中間那兩次畫的是不完整的組合
        —— 例如手牌已經換成下一巡、建議還是上一巡的。

        區塊內沒有任何更新時不會發通知。
        """
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0 and self._dirty:
                self._dirty = False
                self._notify()

    # ------------------------------------------------------------------ 更新

    def update_cv_hand(
        self,
        tiles: Sequence[str],
        *,
        drawn: str | None = None,
        confident: bool = True,
    ) -> None:
        """CV 認到的手牌。

        Args:
            tiles: **雀魂記法**(``0m`` / ``1z``)的暗手牌 —— 那是 classify 的
                輸出。這裡轉成 MJAI 記法,UI 以下一律只有一種記法。
            drawn: 剛摸進來那張(同樣是雀魂記法)。``read_hand`` 本來就把它
                與暗手牌分開。
            confident: 比對分數是否都夠有把握。不夠的話會掛一個警示,
                但**仍然顯示** —— 藏起來只會讓使用者以為程式當了。
        """
        try:
            mjai = tuple(ms_to_mjai(t) for t in tiles)
            drawn_mjai = ms_to_mjai(drawn) if drawn else None
        except ValueError:
            self._notice("CV 讀到認不得的牌,這一幀跳過")
            return
        notice = "CV 對部分牌沒有把握,建議僅供參考" if not confident else None
        # 內部一律存完整手牌(含摸的那張),兩個來源才比較得起來
        full = (*mjai, drawn_mjai) if drawn_mjai else mjai
        self._recompute(cv_hand=full, cv_drawn=drawn_mjai, notice=notice)

    def update_packet_hand(self, tiles: Sequence[str], *, drawn: str | None = None) -> None:
        """封包推出來的手牌(已是 MJAI 記法)。

        Args:
            tiles: 暗手牌,**含**剛摸進來那張 —— ``HandTracker.tiles`` 就是這樣。
            drawn: 那一張是哪張,見 :attr:`ViewState.drawn`。
        """
        self._recompute(packet_hand=tuple(tiles), packet_drawn=drawn)

    def update_advices(self, advices: Sequence[Advice]) -> None:
        """引擎這一手的建議。

        全部引擎都「不動作」時也要更新 —— 那代表「現在不該我動」,是有效資訊,
        而不是「還沒算出來」。
        """
        views = tuple(_to_view(a) for a in advices)
        self._emit(replace(self._state, engines=views))

    def set_preferred_engine(self, name: str | None) -> None:
        """點名哪個引擎當主角。``None`` 回到照 :attr:`ViewState.engines` 順序取。"""
        self._emit(replace(self._state, preferred_engine=name))

    def set_notices(self, notices: Sequence[str]) -> None:
        """整批換掉警示。校正狀態、引擎存活這些由外部持續回報。"""
        self._emit(replace(self._state, notices=tuple(notices)))

    def clear(self) -> None:
        """回到空白 —— 換局或斷線時用。"""
        self._emit(ViewState())

    # ------------------------------------------------------------------ 內部

    def _recompute(
        self,
        *,
        cv_hand: tuple[str, ...] | None = None,
        cv_drawn: str | None = None,
        packet_hand: tuple[str, ...] | None = None,
        packet_drawn: str | None = None,
        notice: str | None = None,
    ) -> None:
        state = self._state
        updating_cv = cv_hand is not None
        cv = cv_hand if cv_hand is not None else state.cv_hand
        cv_draw = cv_drawn if updating_cv else state.cv_drawn
        packet = packet_hand if packet_hand is not None else state.packet_hand
        packet_draw = packet_drawn if packet_hand is not None else state.packet_drawn

        # 封包優先:它是精確的,CV 是估計的。兩邊都有時衝突另外標示。
        # drawn 跟著同一個優先序,否則會出現「手牌來自封包、摸的那張來自 CV」
        # 這種混血狀態,而那張牌可能根本不在手牌裡。
        hand = packet or cv
        drawn = packet_draw if packet else cv_draw
        source = "both" if (cv and packet) else ("packet" if packet else "cv" if cv else "")

        analysis, discards = _analyse(hand)
        notices = list(state.notices)
        if notice and notice not in notices:
            notices.append(notice)

        self._emit(
            replace(
                state,
                hand=hand,
                drawn=drawn,
                hand_source=source,
                cv_hand=cv,
                cv_drawn=cv_draw,
                packet_hand=packet,
                packet_drawn=packet_draw,
                analysis=analysis,
                discards=discards,
                notices=tuple(notices),
            )
        )

    def _notice(self, message: str) -> None:
        if message in self._state.notices:
            return
        self._emit(replace(self._state, notices=(*self._state.notices, message)))

    def _emit(self, state: ViewState) -> None:
        self._state = state
        if self._depth:
            self._dirty = True
            return
        self._notify()

    def _notify(self) -> None:
        for listener in self._listeners:
            listener(self._state)


def _analyse(hand: tuple[str, ...]) -> tuple[HandAnalysis | None, tuple[DiscardOption, ...]]:
    """算向聽與打牌建議。算不出來就回 ``None`` —— 那不是錯誤。

    手牌張數落在 3 的倍數是常態(辨識抓在動畫中間、或剛好在切牌那一刻),
    這時該讓 UI 保持上一次的顯示,而不是彈一個錯誤出來。
    """
    if not hand or UNKNOWN in hand:
        return None, ()
    try:
        ms = [mjai_to_ms(t) for t in hand]
        analysis = analyse(ms)
    except (HandError, ValueError):
        return None, ()

    discards: tuple[DiscardOption, ...] = ()
    if analysis.needs_discard:
        try:
            discards = suggest_discards(ms)
        except HandError:
            discards = ()
    return analysis, discards


def _to_view(advice: Advice) -> EngineView:
    action = advice.action
    if action is None:
        # 跳過的那一手要保留 candidates —— 使用者最需要看到的正是「碰 -0.11
        # 對 跳過 +0.10」這個對比,那才回答得了「為什麼不鳴」
        return EngineView(
            advice.engine,
            None,
            None,
            decode_candidates(advice.meta),
            advice.latency_ms,
            declined=advice.declined,
        )

    tile, consumed = action_tiles(action)
    follow = advice.follow_up
    # 立直與吃碰的 Q 值也要看得到,所以 candidates 不因為「不是切牌」而省略
    return EngineView(
        name=advice.engine,
        action=action_label(action),
        tile=tile,
        candidates=decode_candidates(advice.meta),
        latency_ms=advice.latency_ms,
        consumed=consumed,
        is_discard=action.TYPE == "dahai",
        follow_up_tile=getattr(follow, "pai", None) if follow is not None else None,
    )
