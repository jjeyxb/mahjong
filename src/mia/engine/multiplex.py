"""把同一條事件流餵給多個引擎。

這是功能 3 的骨架。「這份微調權重打得比較兇」不能靠感覺,得讓兩個引擎看同一
個局面、把兩邊的答案並排 —— 分歧的那幾手才是風格差異的實際內容。同樣的機制
也用來與規則式 baseline 對照。

一次的意外
----------
引擎有狀態,事件流必須完整送到**每一個**引擎。所以其中一個壞掉時不能直接跳過
它繼續跑:那個引擎會少掉一段歷史,之後給的建議看起來正常卻是錯的。
:class:`EngineGroup` 的處置是把壞掉的引擎**整個移出這一場**並記錄原因,
而不是讓它帶著破掉的狀態繼續回答。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from mia.engine.base import Advice, AIEngine, EngineError
from mia.mjai import MjaiEvent, Reach
from mia.utils.logging import logger

__all__ = ["EngineGroup", "GroupResult"]


@dataclass(frozen=True, slots=True)
class GroupResult:
    """一個事件送進所有引擎之後的結果。

    Attributes:
        advices: 各引擎的建議,順序與加入時相同。已經掉隊的引擎不在裡面。
        failures: 這一手壞掉的引擎 → 原因。
    """

    advices: tuple[Advice, ...] = ()
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def actions(self) -> tuple[Advice, ...]:
        """只留真的有動作的建議。大部分事件會是空的。"""
        return tuple(a for a in self.advices if a.is_action)

    @property
    def decisions(self) -> tuple[Advice, ...]:
        """這一手構成決策點的建議 —— **含結論是「跳過」的**。

        與 :attr:`actions` 分開,是因為「跳過」也是答案。只看 actions 的話,
        遊戲跳出「碰 / 槓 / 跳過」而引擎說不鳴時,呼叫端會以為這一手沒事發生。
        """
        return tuple(a for a in self.advices if a.is_decision)

    @property
    def is_unanimous(self) -> bool:
        """所有有動作的引擎是否給了同一個答案。

        少於兩個引擎有動作時為 True —— 沒有分歧可言。**分歧才是要看的東西**,
        UI 應該把 False 的那幾手標出來。
        """
        actions = {str(a.action) for a in self.actions}
        return len(actions) <= 1

    def __str__(self) -> str:
        if not self.advices:
            return "(沒有引擎回應)"
        body = " | ".join(str(a) for a in self.advices)
        return f"{body}{'  ⚠分歧' if not self.is_unanimous else ''}"


class EngineGroup:
    """一組並行運作的引擎。

    Args:
        engines: 要納入的引擎。
        parallel: 是否用執行緒同時送。子程序引擎大部分時間都卡在等回覆,
            並行能把 N 個引擎的延遲從相加變成取最大值。單一引擎時無意義。

    Note:
        並行安全的前提是**每個引擎各自獨立** —— 它們之間不共用任何狀態,
        每個都只跟自己的子程序講話。同一個引擎不會被兩個執行緒同時呼叫。
    """

    def __init__(self, engines: Iterable[AIEngine], *, parallel: bool = True) -> None:
        self._engines: list[AIEngine] = list(engines)
        self._failed: dict[str, str] = {}
        self.parallel = parallel

    def __len__(self) -> int:
        return len(self._engines)

    def __iter__(self) -> Iterator[AIEngine]:
        return iter(self._engines)

    def __enter__(self) -> EngineGroup:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        names = ", ".join(e.name for e in self._engines)
        return f"<EngineGroup [{names}] failed={len(self._failed)}>"

    @property
    def failed(self) -> dict[str, str]:
        """已經被移出的引擎 → 移出的原因。"""
        return dict(self._failed)

    def start(self) -> None:
        """啟動所有引擎。啟動不了的直接移出,不讓它拖著整組不能跑。

        Raises:
            EngineError: 一個都啟動不了。
        """
        alive = []
        for engine in self._engines:
            try:
                engine.start()
            except EngineError as exc:
                self._drop(engine.name, f"啟動失敗: {exc}")
            else:
                alive.append(engine)
        self._engines = alive
        if not self._engines:
            raise EngineError(f"沒有任何引擎啟動成功: {self._failed}")

    def react(self, event: MjaiEvent) -> GroupResult:
        """把一個事件送給所有還活著的引擎。

        Returns:
            各引擎的建議與這一手的失敗紀錄。**不會拋例外** —— 單一引擎壞掉
            不該中斷整場,呼叫端從 :attr:`GroupResult.failures` 得知。
        """
        if not self._engines:
            return GroupResult()

        # 這一手的失敗,與 self._failed(整場累計)分開 —— GroupResult 只報
        # 新壞掉的,否則掉隊過的引擎會在後續每一手重複出現在 failures 裡
        failures: dict[str, str] = {}

        def run(engine: AIEngine) -> Advice | None:
            try:
                return engine.react(event)
            except EngineError as exc:
                failures[engine.name] = str(exc)
                return None

        if self.parallel and len(self._engines) > 1:
            with ThreadPoolExecutor(max_workers=len(self._engines)) as pool:
                results = list(pool.map(run, self._engines))
        else:
            results = [run(engine) for engine in self._engines]

        for name, reason in failures.items():
            self._drop(name, reason)
        self._engines = [e for e in self._engines if e.name not in failures]
        return GroupResult(tuple(a for a in results if a is not None), failures)

    def resolve_follow_ups(self, result: GroupResult) -> GroupResult:
        """替回了「立直」的引擎補上「然後切哪一張」。

        MJAI 的引擎回 ``reach`` 之後不會順便說切哪一張,要等它看到 ``reach``
        事件才會說 —— 而那個事件要等使用者真的宣言完、牌也打出去了才會從封包
        送來。所以這裡先用 :meth:`AIEngine.peek` 問一步。

        只對真的回了 ``reach`` 的引擎問,因為 ``peek`` 對子程序引擎要付一次
        重啟 + 重播的代價。問失敗**不算引擎掉隊** —— 主要建議(立直)已經拿到
        手了,少一個後續切牌不值得把整個引擎移出這一場。
        """
        if not any(a.action is not None and a.action.TYPE == "reach" for a in result.advices):
            return result

        by_name = {e.name: e for e in self._engines}
        resolved: list[Advice] = []
        for advice in result.advices:
            action = advice.action
            engine = by_name.get(advice.engine)
            if action is None or action.TYPE != "reach" or engine is None:
                resolved.append(advice)
                continue
            try:
                follow = engine.peek(Reach(actor=int(getattr(action, "actor", 0))))
            except EngineError as exc:
                logger.warning(f"{advice.engine}: 問不到立直後要切哪張 — {exc}")
                resolved.append(advice)
            else:
                resolved.append(replace(advice, follow_up=follow.action))
        return GroupResult(tuple(resolved), dict(result.failures))

    def close(self) -> None:
        for engine in self._engines:
            try:
                engine.close()
            except Exception as exc:  # noqa: BLE001 - 關閉失敗不該蓋掉其他引擎的關閉
                logger.warning(f"{engine.name}: 關閉時出錯 — {exc}")

    # ------------------------------------------------------------------ 內部

    def _drop(self, name: str, reason: str) -> None:
        self._failed[name] = reason
        logger.error(f"{name}: 移出這一場 — {reason}")
