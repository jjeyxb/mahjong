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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from mia.analysis import (
    DiscardOption,
    HandAnalysis,
    HandError,
    analyse,
    suggest_discards,
)
from mia.engine.actions import Candidate, action_label, decode_candidates
from mia.engine.base import Advice
from mia.mjai.tiles import UNKNOWN, mjai_to_ms, ms_to_mjai

__all__ = ["EngineView", "ViewModel", "ViewState"]


@dataclass(frozen=True, slots=True)
class EngineView:
    """一個引擎在這一手給的東西。

    Attributes:
        name: 引擎名稱。
        action: 建議的動作,人看的字串。``None`` 表示這一手不需要動作。
        tile: 若建議是切牌,那張牌(MJAI 記法);否則 ``None``。
            UI 靠它決定要不要畫牌面圖。
        candidates: 所有合法動作與 Q 值,最高分在前。規則式引擎沒有,是空的。
        latency_ms: 這一手花了多久。
    """

    name: str
    action: str | None = None
    tile: str | None = None
    candidates: tuple[Candidate, ...] = ()
    latency_ms: float = 0.0

    @property
    def has_reasoning(self) -> bool:
        """有沒有 Q 值可以解釋「為什麼」。"""
        return bool(self.candidates)


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
        for engine in self.engines:
            if engine.action is not None:
                return engine
        return self.engines[0] if self.engines else None

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
        if on_change is not None:
            self._listeners.append(on_change)

    @property
    def state(self) -> ViewState:
        return self._state

    def subscribe(self, listener: Callable[[ViewState], None]) -> None:
        """加一個變化通知。側邊視窗與 Overlay 各自訂閱同一個 ViewModel。"""
        self._listeners.append(listener)

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
        for listener in self._listeners:
            listener(state)


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
        return EngineView(advice.engine, None, None, (), advice.latency_ms)

    tile = getattr(action, "pai", None)
    # 立直與吃碰的 Q 值也要看得到,所以 candidates 不因為「不是切牌」而省略
    return EngineView(
        name=advice.engine,
        action=action_label(action),
        tile=tile if action.TYPE == "dahai" else None,
        candidates=decode_candidates(advice.meta),
        latency_ms=advice.latency_ms,
    )
