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
    """

    engine: str
    action: MjaiEvent | None = None
    meta: dict[str, Any] | None = None
    latency_ms: float = 0.0

    @property
    def is_action(self) -> bool:
        return self.action is not None

    def __str__(self) -> str:
        body = str(self.action) if self.action is not None else "不動作"
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

    def close(self) -> None:
        """關掉引擎並釋放資源。可重複呼叫。"""
        ...
