"""多引擎並排。

重點不是「能同時跑」,而是**壞掉時的處置**:引擎有狀態,漏掉一段事件的引擎
之後給的建議看起來正常卻是錯的,所以必須整個移出而不是跳過這一手繼續用。
"""

from __future__ import annotations

import pytest

from mia.engine import Advice, EngineError, EngineGroup
from mia.mjai import Dahai, MjaiEvent, Reach, Tsumo

TSUMO = Tsumo(actor=0, pai="3s")


class FakeEngine:
    """可以指定「第幾次呼叫壞掉」的引擎,用來測 EngineGroup 的容錯。"""

    def __init__(
        self,
        name: str,
        action: MjaiEvent | None = None,
        *,
        fail_at: int | None = None,
        fail_start: bool = False,
    ) -> None:
        self._name = name
        self.action = action
        self.fail_at = fail_at
        self.fail_start = fail_start
        self.seen: list[MjaiEvent] = []
        self.started = False
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        if self.fail_start:
            raise EngineError(f"{self._name} 起不來")
        self.started = True

    def react(self, event: MjaiEvent) -> Advice:
        self.seen.append(event)
        if self.fail_at is not None and len(self.seen) >= self.fail_at:
            raise EngineError(f"{self._name} 崩了")
        return Advice(self._name, self.action)

    def close(self) -> None:
        self.closed = True


DAHAI_A = Dahai(actor=0, pai="1m", tsumogiri=False)
DAHAI_B = Dahai(actor=0, pai="9p", tsumogiri=False)


class TestBroadcast:
    @pytest.mark.parametrize("parallel", [True, False])
    def test_every_engine_sees_every_event(self, parallel: bool) -> None:
        """引擎有狀態,漏一個事件它眼中的牌河就永遠少一張。"""
        engines = [FakeEngine("a"), FakeEngine("b")]
        with EngineGroup(engines, parallel=parallel) as group:
            group.react(TSUMO)
            group.react(Reach(actor=1))
        assert [e.seen for e in engines] == [[TSUMO, Reach(actor=1)]] * 2

    def test_advices_keep_the_order_engines_were_added_in(self) -> None:
        """UI 會並排顯示,順序跳動會讓人看錯是哪個引擎給的建議。"""
        group = EngineGroup([FakeEngine("a", DAHAI_A), FakeEngine("b", DAHAI_B)])
        group.start()
        assert [a.engine for a in group.react(TSUMO).advices] == ["a", "b"]

    def test_close_reaches_every_engine(self) -> None:
        engines = [FakeEngine("a"), FakeEngine("b")]
        with EngineGroup(engines):
            pass
        assert all(e.closed for e in engines)


class TestAgreement:
    def test_the_same_action_is_unanimous(self) -> None:
        group = EngineGroup([FakeEngine("a", DAHAI_A), FakeEngine("b", DAHAI_A)])
        group.start()
        assert group.react(TSUMO).is_unanimous

    def test_different_actions_are_flagged(self) -> None:
        """分歧的那幾手才是風格比較的實際內容 —— UI 要標出來。"""
        group = EngineGroup([FakeEngine("a", DAHAI_A), FakeEngine("b", DAHAI_B)])
        group.start()
        result = group.react(TSUMO)
        assert not result.is_unanimous
        assert len(result.actions) == 2

    def test_no_action_at_all_is_unanimous(self) -> None:
        """大部分事件不需要任何人動作,那不算分歧。"""
        group = EngineGroup([FakeEngine("a"), FakeEngine("b")])
        group.start()
        result = group.react(TSUMO)
        assert result.is_unanimous
        assert result.actions == ()


class TestFailure:
    def test_a_crashed_engine_is_dropped_not_retried(self) -> None:
        """壞掉的引擎少了一段歷史,再用它只會拿到看似正常的錯誤建議。"""
        good, bad = FakeEngine("good", DAHAI_A), FakeEngine("bad", fail_at=1)
        group = EngineGroup([good, bad])
        group.start()

        first = group.react(TSUMO)
        assert first.failures == {"bad": "bad 崩了"}
        assert [a.engine for a in first.advices] == ["good"]

        second = group.react(TSUMO)
        assert second.failures == {}, "先前就掉隊的不該每一手重報一次"
        assert len(bad.seen) == 1, "掉隊之後不該再被呼叫"
        assert len(good.seen) == 2

    def test_the_group_records_why_each_engine_left(self) -> None:
        group = EngineGroup([FakeEngine("a", DAHAI_A), FakeEngine("bad", fail_at=1)])
        group.start()
        group.react(TSUMO)
        assert "崩了" in group.failed["bad"]

    def test_an_engine_that_cannot_start_is_dropped(self) -> None:
        group = EngineGroup([FakeEngine("ok"), FakeEngine("nope", fail_start=True)])
        group.start()
        assert len(group) == 1
        assert "起不來" in group.failed["nope"]

    def test_starting_with_no_working_engine_raises(self) -> None:
        """一個都起不來時要當場失敗 —— 靜靜地跑一個空的組合最糟。"""
        group = EngineGroup([FakeEngine("nope", fail_start=True)])
        with pytest.raises(EngineError, match="沒有任何引擎啟動成功"):
            group.start()

    def test_reacting_with_an_empty_group_is_harmless(self) -> None:
        assert EngineGroup([]).react(TSUMO).advices == ()
