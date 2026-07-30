"""AI 引擎的共同介面。

一個引擎就是一個函式:餵它一串 MJAI 事件,它在輪到自己時回一個動作。
Mortal、規則式 baseline、日後微調出來的權重,都只要滿足這個介面就能互換
—— 功能 3 的「風格比較」本質上就是同時跑好幾個引擎、看它們在同一個局面
給出什麼不同的答案,所以可替換性不是設計潔癖,是需求。

為什麼是「餵事件」而不是「問局面」
----------------------------------
MJAI 的引擎是**有狀態**的:它自己從事件流重建局面,記得誰立直了、牌河長什麼樣。
所以介面是 :meth:`AIEngine.react` 一路餵,不是每次傳一個完整的 GameState 進去。
這也代表事件**不能漏、不能重送** —— 漏一個 dahai,引擎眼中的牌河就永遠少一張。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mia.engine.actions import is_decision
from mia.mjai import MjaiEvent

__all__ = ["AIEngine", "Advice", "EngineError", "EngineTimeout"]


class EngineError(RuntimeError):
    """引擎壞了 —— 啟動失敗、崩潰、或回了認不得的東西。"""


class EngineTimeout(EngineError):
    """引擎在時限內沒有回覆。

    與 :class:`EngineError` 分開是因為兩者的處置不同:逾時通常是這一手太慢
    (第一次推論要載入權重、或機器正在忙),重試可能就好;崩潰則必須重啟。
    """


@dataclass(frozen=True, slots=True)
class Advice:
    """引擎對某個局面給出的建議。

    Attributes:
        engine: 是哪個引擎給的。多引擎並排顯示時要靠它區分。
        action: 建議的動作。``None`` 表示「這一手不需要我動作」——
            大部分事件都是這個結果,只有輪到自己摸打或可以鳴牌時才有值。
        meta: 引擎自訂的附加資訊,不屬於 MJAI 協定本身。Mortal 會在這裡放
            各候選動作的 Q 值與遮罩,那是 UI 上「為什麼這樣打」的唯一素材。
        latency_ms: 從送出事件到收到回覆的時間。多引擎比較時要看得到誰慢。
        follow_up: 做了 :attr:`action` 之後緊接著要做的事。

            **只有立直會有。** MJAI 的引擎回 ``reach`` 之後,要等它**看到**
            ``reach`` 事件才會說切哪一張;而那個事件要等使用者真的宣言完、
            牌也打出去了才會從封包送來 —— 那時候「該切哪張」已經沒有意義了。
            所以呼叫端要用 :meth:`AIEngine.peek` 先問一步,答案放在這裡。
    """

    engine: str
    action: MjaiEvent | None = None
    meta: dict[str, Any] | None = None
    latency_ms: float = 0.0
    follow_up: MjaiEvent | None = None

    @property
    def is_action(self) -> bool:
        return self.action is not None

    @property
    def is_decision(self) -> bool:
        """這一手是不是一個**決策點** —— 即使結論是「跳過」。

        有動作時當然是。沒有動作時要看引擎是不是本來就有得選:遊戲跳出
        「碰 / 槓 / 跳過」時引擎回 ``none``,那個 ``none`` 的意思是「跳過」,
        是使用者要的答案;而別人在摸打時的 ``none`` 只是「沒人在問」。
        兩者都是 ``action is None``,靠 ``meta`` 裡的合法動作數才分得開。

        規則式引擎沒有 ``meta``,所以它的「不動作」一律不算決策點 ——
        它本來就不鳴牌(見 :class:`~mia.engine.dummy.DummyEngine`),
        把它算進來只會讓畫面在「跳過」與切牌建議之間閃。
        """
        return self.is_action or is_decision(self.meta)

    @property
    def declined(self) -> bool:
        """被問了,但選擇不動作 —— 也就是「跳過」。"""
        return not self.is_action and is_decision(self.meta)

    def __str__(self) -> str:
        if self.action is not None:
            body = str(self.action)
        else:
            body = "跳過" if self.declined else "不動作"
        return f"[{self.engine}] {body} ({self.latency_ms:.0f} ms)"


@runtime_checkable
class AIEngine(Protocol):
    """AI 引擎。

    生命週期是 :meth:`start` → 多次 :meth:`react` → :meth:`close`,
    :meth:`close` 之後不可再用。實作也應支援 ``with`` 語法。

    Note:
        實作**不必**是執行緒安全的。事件流本來就有嚴格順序,並行送入沒有意義;
        要並行的是「多個引擎」而不是「一個引擎的多個請求」,那由
        :class:`~mia.engine.multiplex.EngineGroup` 負責。
    """

    @property
    def name(self) -> str:
        """人看的名字,會出現在 :attr:`Advice.engine` 與 UI 上。"""
        ...

    def start(self) -> None:
        """啟動引擎。可重複呼叫,已啟動時是 no-op。

        Raises:
            EngineError: 啟動失敗。
        """
        ...

    def react(self, event: MjaiEvent) -> Advice:
        """送一個事件進去,拿回這一手的建議。

        Raises:
            EngineError: 引擎崩潰或回覆無法解析。
            EngineTimeout: 逾時。
        """
        ...

    def peek(self, event: MjaiEvent) -> Advice:
        """問一個**假設性**的後續:「如果這件事發生了,你接下來會做什麼?」

        問完引擎的狀態必須與問之前完全相同 —— 這個事件**沒有真的發生**,
        它只是一個假設。實作要自己負責復原。

        存在的唯一理由是立直:引擎回 ``reach`` 之後不會順便說切哪一張,
        而使用者在按下立直的那一刻就需要知道。見 :attr:`Advice.follow_up`。

        Raises:
            EngineError: 引擎崩潰,或復原失敗。復原失敗必須拋例外而不是默默
                回傳 —— 帶著被污染的狀態繼續跑會給出看起來正常但其實錯的建議。
        """
        ...

    def close(self) -> None:
        """關掉引擎並釋放資源。可重複呼叫。"""
        ...
