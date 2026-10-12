"""使用者能在 UI 上挑的「風格」—— 一個名字對應一組要同時載入的權重。

比較的對象由使用者自己挑
------------------------
功能 3 的重點是**比較**:「這份微調權重打得比較兇」不能靠感覺,得讓兩個引擎
看同一個局面、把兩邊的答案並排,分歧的那幾手才是風格差異的實際內容。

所以 :class:`StyleChoice` 有**兩個**格子:主要風格,加上一個可選的比較對象。
原本的做法是在設定檔裡寫死幾組(``比較:鳴牌傾向`` 之類),而那只覆蓋得到
事先想到的那幾種組法 —— 想比「愛鳴牌 vs 愛立直」就得自己去編 YAML。
兩個格子的話 N 份權重自然就有 N×(N−1) 種組法,而設定檔只要列出單獨的那幾份。

**不能跟自己比。** 同一份權重載兩次會得到兩個答案完全相同的引擎,
永遠沒有分歧 —— 看起來像「這兩個風格沒差別」,實際上是選錯了。
所以 :meth:`StyleChoice.set_compare` 擋掉它。

為什麼一個選項仍然可以帶**多份**權重
------------------------------------
:class:`StyleProfile.weights` 是個 tuple 而不是單一路徑,所以設定檔裡仍然寫得出
三份以上的組合給特殊用途。UI 上的兩個選單是常用的那條路,不是唯一的路 ——
:class:`~mia.engine.multiplex.EngineGroup` 本來就吃任意數量的引擎。

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
        name: 初始的主要風格;``None`` 或找不到時取第一個**可用**的。
        compare: 初始的比較對象;``None`` 代表不比較。找不到或與 ``name``
            撞在一起時也是不比較 —— 那是從磁碟讀回來的舊選擇,設定檔改過之後
            對不上是正常的,不該讓程式起不來。

    Note:
        空的 ``profiles``(設定沒寫、或權重全都不在)是合法狀態 ——
        那時 :attr:`value` 是 ``None``,UI 把選單畫成停用。
    """

    def __init__(
        self,
        profiles: tuple[StyleProfile, ...] = (),
        name: str | None = None,
        compare: str | None = None,
    ) -> None:
        self._profiles = tuple(p for p in profiles if p.available)
        self._value = self._find(name) or (self._profiles[0] if self._profiles else None)
        self._compare: StyleProfile | None = None
        self.set_compare(compare)

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
    def compare(self) -> StyleProfile | None:
        """要並排比較的那一個。``None`` 代表只跑主要風格。"""
        return self._compare

    @property
    def compare_name(self) -> str | None:
        return self._compare.name if self._compare else None

    @property
    def weights(self) -> tuple[Path, ...]:
        """現在該載哪幾份。沒有選項時是空的 —— 呼叫端只跑規則式 baseline。

        主要風格排在前面:順序就是引擎的優先序,而
        :attr:`~mia.ui.viewmodel.ViewState.primary` 取第一個有動作的。
        比較對象接在後面,所以 headline 永遠是主要那一份,不會因為開了比較
        就換人當主角。
        """
        primary = self._value.weights if self._value else ()
        return primary + (self._compare.weights if self._compare else ())

    def set(self, name: str | None) -> bool:
        """換主要風格。回傳**有沒有真的變**,讓呼叫端決定要不要重啟引擎。

        重啟要付一次完整的載入成本(130MB 權重,實測 0.5~0.6 秒/份),
        而使用者在選單上重選同一項是很常見的動作。

        選到的正好是目前的比較對象時,**比較會被關掉** —— 不然就變成跟自己比,
        兩個引擎的答案永遠相同。這是靜靜地改掉使用者的另一個選擇,所以 UI 要
        在這之後重新同步那個選單(見 :meth:`set_compare`)。
        """
        profile = self._find(name)
        if profile is None or profile is self._value:
            return False
        self._value = profile
        if self._compare is profile:
            self._compare = None
        return True

    def set_compare(self, name: str | None) -> bool:
        """換比較對象。``None`` 關掉比較。回傳**有沒有真的變**。

        擋兩種情況,兩種都回 ``False``:

        * **跟主要風格同一個。** 同一份權重載兩次會得到兩個答案完全相同的
          引擎,永遠沒有分歧 —— 畫面看起來像「這兩個風格沒差別」,
          而那是最糟的一種錯:它長得像一個結論。
        * **名字不在選項裡。** 當成不比較。那通常是從 ``ui_state.json`` 讀回來
          的舊選擇,而設定檔改過之後對不上是正常的。
        """
        profile = self._find(name) if name is not None else None
        if profile is not None and profile is self._value:
            return False
        if profile is self._compare:
            return False
        self._compare = profile
        return True

    def _find(self, name: str | None) -> StyleProfile | None:
        if name is None:
            return None
        return next((p for p in self._profiles if p.name == name), None)

    def __str__(self) -> str:
        if self._value is None:
            return "(沒有可用的風格)"
        if self._compare is None:
            return self._value.name
        return f"{self._value.name} vs {self._compare.name}"

    def __repr__(self) -> str:
        return f"<StyleChoice {self} of {len(self._profiles)}>"
