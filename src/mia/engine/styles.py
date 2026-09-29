"""使用者能在 UI 上挑的「風格」—— 一個名字對應一組要同時載入的權重。

為什麼一個選項可以帶**多份**權重
--------------------------------
功能 3 的重點是**比較**:「這份微調權重打得比較兇」不能靠感覺,得讓兩個引擎
看同一個局面、把兩邊的答案並排,分歧的那幾手才是風格差異的實際內容。
所以「愛鳴牌 vs 不愛鳴牌」本身就該是一個可以選的項目,而不是要使用者開兩次
程式再自己對照。:class:`~mia.engine.multiplex.EngineGroup` 本來就吃一組引擎,
這裡只是把「哪一組」變成執行期可以改的東西。

為什麼 :class:`StyleChoice` 是可變物件而不是傳值
------------------------------------------------
與 :class:`~mia.calibration.canvas.CanvasChoice` 同一個理由:讀它的是
``PacketWorker`` 的建構參數,而那是**延遲建構**的(功能打開才建)。傳值的話
使用者改了選擇之後,還沒建起來的那一邊會拿到舊值,而那不會有任何症狀 ——
畫面照常更新,只是跑的是上一個風格。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["StyleChoice", "StyleProfile"]


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """一個可選的風格。

    Attributes:
        name: 給人看的名字,同時是識別字。設定檔裡一個就是一項,所以不另外
            分 key 與 label —— 兩個名字只會讓人改了一個忘了另一個。
        weights: 要同時載入的權重。**順序就是引擎的優先序** ——
            :attr:`~mia.ui.viewmodel.ViewState.primary` 取第一個有動作的,
            所以想當主角的那份排前面。
    """

    name: str
    weights: tuple[Path, ...]

    @property
    def missing(self) -> tuple[Path, ...]:
        """設定裡寫了但檔案不在的那幾份。"""
        return tuple(w for w in self.weights if not w.is_file())

    @property
    def available(self) -> bool:
        """權重全都在,選了才真的跑得起來。

        權重不隨專案散布(Mortal 是 AGPL-3.0),所以設定檔列出來的東西在別人
        的機器上很可能不存在。缺的那些不該出現在選單裡 —— 可選而選了就報錯,
        比一開始就看不到糟。
        """
        return bool(self.weights) and not self.missing


class StyleChoice:
    """使用者現在選的風格,**一個可變的共用格子**。

    Args:
        profiles: 全部可選項。不在這裡面的名字設不進去。
        name: 初始選擇;``None`` 或找不到時取第一個**可用**的。

    Note:
        空的 ``profiles``(設定沒寫、或權重全都不在)是合法狀態 ——
        那時 :attr:`value` 是 ``None``,UI 把選單畫成停用。
    """

    def __init__(self, profiles: tuple[StyleProfile, ...] = (), name: str | None = None) -> None:
        self._profiles = tuple(p for p in profiles if p.available)
        self._value = self._find(name) or (self._profiles[0] if self._profiles else None)

    @property
    def profiles(self) -> tuple[StyleProfile, ...]:
        """可用的選項。權重缺檔的已經被濾掉。"""
        return self._profiles

    @property
    def value(self) -> StyleProfile | None:
        return self._value

    @property
    def name(self) -> str | None:
        return self._value.name if self._value else None

    @property
    def weights(self) -> tuple[Path, ...]:
        """現在該載哪幾份。沒有選項時是空的 —— 呼叫端只跑規則式 baseline。"""
        return self._value.weights if self._value else ()

    def set(self, name: str | None) -> bool:
        """依名字設定。回傳**有沒有真的變**,讓呼叫端決定要不要重啟引擎。

        重啟要付一次完整的載入成本(130MB 權重,實測 0.5~0.6 秒/份),
        而使用者在選單上重選同一項是很常見的動作。
        """
        profile = self._find(name)
        if profile is None or profile is self._value:
            return False
        self._value = profile
        return True

    def _find(self, name: str | None) -> StyleProfile | None:
        if name is None:
            return None
        return next((p for p in self._profiles if p.name == name), None)

    def __str__(self) -> str:
        return self._value.name if self._value else "(沒有可用的風格)"

    def __repr__(self) -> str:
        return f"<StyleChoice {self} of {len(self._profiles)}>"
