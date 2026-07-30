"""即時模式的協調層。

沒有 Qt、沒有執行緒 —— pump 是同步的,所以整條路可以直接呼叫來驗。
那正是把它做成「郵箱 + pump」而不是 Qt signal 的目的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia import features
from mia.engine.base import Advice
from mia.live.bus import Advices, CvHand, PacketHand, UpdateBus, WorkerStatus
from mia.live.runtime import Feature, LiveRuntime
from mia.live.source import CaptureProcess, capture_command
from mia.mjai import Dahai
from mia.ui.viewmodel import ViewModel


class FakeWorker:
    """滿足 :class:`~mia.live.runtime.Worker` 的最小實作。"""

    def __init__(self, name: str) -> None:
        self.status = WorkerStatus(name)
        self.started = False
        self.stopped = False
        self.joined = 0

    def start(self) -> None:
        self.started = True
        self.status.alive = True

    def stop(self) -> None:
        self.stopped = True
        self.status.alive = False

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
        self.joined += 1

    def is_alive(self) -> bool:
        return self.status.alive and not self.stopped


class Spy:
    """記錄工廠被叫了幾次,並讓測試拿得到造出來的 worker。"""

    def __init__(self, name: str = "畫面") -> None:
        self.name = name
        self.made: list[FakeWorker] = []

    def __call__(self) -> FakeWorker:
        worker = FakeWorker(self.name)
        self.made.append(worker)
        return worker

    @property
    def latest(self) -> FakeWorker:
        return self.made[-1]


def _feature(key: str = features.VISION, spy: Spy | None = None) -> tuple[Feature, Spy]:
    spy = spy or Spy()
    return Feature(key, spy), spy


@pytest.fixture
def model() -> ViewModel:
    return ViewModel()


@pytest.fixture
def bus() -> UpdateBus:
    return UpdateBus()


class TestLifecycle:
    def test_start_does_not_start_any_feature(self, model: ViewModel, bus: UpdateBus) -> None:
        """預設全部關著。開視窗不該是「開始擷取螢幕 + 載入 130MB 權重」的副作用。"""
        feature, spy = _feature()
        LiveRuntime(model, bus, features=[feature]).start()
        assert spy.made == []
        assert not feature.enabled

    def test_start_is_idempotent(self, model: ViewModel, bus: UpdateBus) -> None:
        capture = CaptureProcess(["true"])
        runtime = LiveRuntime(model, bus, capture=capture)
        runtime.start()
        first = capture.command
        runtime.start()
        assert capture.command is first  # 沒有再開第二個子程序

    def test_enabling_creates_and_starts_a_worker(self, model: ViewModel, bus: UpdateBus) -> None:
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.start()
        runtime.set_enabled(features.VISION, True)
        assert len(spy.made) == 1
        assert spy.latest.started
        assert runtime.is_enabled(features.VISION)

    def test_disabling_stops_and_joins(self, model: ViewModel, bus: UpdateBus) -> None:
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.start()
        runtime.set_enabled(features.VISION, True)
        runtime.set_enabled(features.VISION, False)
        assert spy.latest.stopped
        assert spy.latest.joined == 1
        assert not runtime.is_enabled(features.VISION)

    def test_enabling_twice_does_not_make_a_second_worker(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.VISION, True)
        runtime.set_enabled(features.VISION, True)
        assert len(spy.made) == 1

    def test_re_enabling_builds_a_fresh_worker(self, model: ViewModel, bus: UpdateBus) -> None:
        """不重用停掉的 worker —— Python 的 Thread 不能重新 start(),
        而且它的狀態(手牌追蹤、解析器、引擎子程序)已經沒有意義了。
        """
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.VISION, True)
        runtime.set_enabled(features.VISION, False)
        runtime.set_enabled(features.VISION, True)
        assert len(spy.made) == 2
        assert spy.made[0] is not spy.made[1]

    def test_an_unknown_key_is_ignored(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime = LiveRuntime(model, bus)
        runtime.set_enabled("telepathy", True)
        assert not runtime.is_enabled("telepathy")

    def test_stop_disables_everything(self, model: ViewModel, bus: UpdateBus) -> None:
        vision, vision_spy = _feature(features.VISION)
        advice, advice_spy = _feature(features.ADVICE, Spy("\u5c01\u5305"))
        runtime = LiveRuntime(model, bus, features=[vision, advice])
        runtime.start()
        runtime.set_enabled(features.VISION, True)
        runtime.set_enabled(features.ADVICE, True)
        runtime.stop()
        assert vision_spy.latest.stopped
        assert advice_spy.latest.stopped
        assert not runtime.running

    def test_stop_before_start_does_nothing(self, model: ViewModel, bus: UpdateBus) -> None:
        feature, spy = _feature()
        LiveRuntime(model, bus, features=[feature]).stop()
        assert spy.made == []

    def test_context_manager_starts_and_stops(self, model: ViewModel, bus: UpdateBus) -> None:
        feature, spy = _feature()
        with LiveRuntime(model, bus, features=[feature]) as runtime:
            assert runtime.running
            runtime.set_enabled(features.VISION, True)
        assert spy.latest.stopped


class TestClearingOnDisable:
    """關掉功能要把它在畫面上的產出一起清掉。

    留著的話最後一次的手牌或建議會停在畫面上,看起來像還在運作 —— 那比空白
    糟得多,因為使用者會照著一個已經不再更新的建議打牌。
    """

    def test_disabling_vision_clears_the_cv_hand(self, model: ViewModel, bus: UpdateBus) -> None:
        feature, _ = _feature(features.VISION)
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.VISION, True)
        bus.post(CvHand(("1m", "2m", "3m")))
        runtime.pump()
        assert model.state.cv_hand

        runtime.set_enabled(features.VISION, False)
        assert model.state.cv_hand == ()

    def test_disabling_advice_clears_the_engines_and_packet_hand(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        feature, _ = _feature(features.ADVICE)
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.ADVICE, True)
        bus.post(PacketHand(("1m", "2m")))
        bus.post(Advices((Advice("m", Dahai(actor=0, pai="2m", tsumogiri=False)),)))
        runtime.pump()
        assert model.state.engines and model.state.packet_hand

        runtime.set_enabled(features.ADVICE, False)
        assert model.state.engines == ()
        assert model.state.packet_hand == ()

    def test_disabling_one_path_leaves_the_other_alone(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """兩個功能刻意解耦 —— 關掉一邊不該動到另一邊。"""
        vision, _ = _feature(features.VISION)
        advice, _ = _feature(features.ADVICE, Spy("\u5c01\u5305"))
        runtime = LiveRuntime(model, bus, features=[vision, advice])
        runtime.set_enabled(features.VISION, True)
        runtime.set_enabled(features.ADVICE, True)
        bus.post(CvHand(("1m", "2m", "3m")))
        bus.post(PacketHand(("4p", "5p")))
        runtime.pump()

        runtime.set_enabled(features.ADVICE, False)
        assert model.state.cv_hand == ("1m", "2m", "3m")
        assert model.state.packet_hand == ()


class TestPump:
    def test_cv_hand_reaches_the_viewmodel(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime = LiveRuntime(model, bus)
        bus.post(CvHand(("1m", "2m", "3m"), drawn="5z"))
        runtime.pump()
        # ViewModel 負責轉成 MJAI 記法 —— 1z..7z 對應 E S W N P F C
        assert model.state.cv_hand == ("1m", "2m", "3m", "P")
        assert model.state.cv_drawn == "P"

    def test_packet_hand_reaches_the_viewmodel(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime = LiveRuntime(model, bus)
        bus.post(PacketHand(("1m", "1m", "2m"), drawn="2m"))
        runtime.pump()
        assert model.state.packet_hand == ("1m", "1m", "2m")
        assert model.state.packet_drawn == "2m"

    def test_advices_reach_the_viewmodel(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime = LiveRuntime(model, bus)
        advice = Advice("mortal", Dahai(actor=0, pai="3s", tsumogiri=False), None, 12.0)
        bus.post(Advices((advice,)))
        runtime.pump()
        assert [e.name for e in model.state.engines] == ["mortal"]
        assert model.state.engines[0].tile == "3s"

    def test_one_pump_notifies_once(self, bus: UpdateBus) -> None:
        """三筆更新一次重畫。逐筆通知的話中間那兩次畫的是不完整的組合。"""
        seen: list[int] = []
        model = ViewModel(lambda _s: seen.append(1))
        runtime = LiveRuntime(model, bus)
        bus.post(CvHand(("1m",)))
        bus.post(PacketHand(("2m",)))
        bus.post(Advices((Advice("m", None),)))
        runtime.pump()
        assert len(seen) == 1

    def test_pump_with_an_empty_box_notifies_nobody(self, bus: UpdateBus) -> None:
        seen: list[int] = []
        model = ViewModel(lambda _s: seen.append(1))
        LiveRuntime(model, bus).pump()
        assert seen == []

    def test_both_sources_together_flag_a_conflict(self, model: ViewModel, bus: UpdateBus) -> None:
        """兩邊都有手牌而且對不上時,狀態要標出來 —— 那是使用者該知道的事。"""
        runtime = LiveRuntime(model, bus)
        bus.post(PacketHand(("1m", "2m", "3m")))
        runtime.pump()
        bus.post(CvHand(("1m", "2m", "4m")))
        runtime.pump()
        assert model.state.hand_conflict


class TestNotices:
    def _running(self, model: ViewModel, bus: UpdateBus) -> tuple[LiveRuntime, FakeWorker]:
        feature, spy = _feature(features.VISION)
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.VISION, True)
        return runtime, spy.latest

    def test_worker_messages_become_notices(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime, worker = self._running(model, bus)
        worker.status.say("找不到遊戲視窗")
        runtime.pump()
        assert model.state.notices == ("畫面:找不到遊戲視窗",)

    def test_a_silent_worker_takes_no_room(self, model: ViewModel, bus: UpdateBus) -> None:
        """正常運作時不該佔用狀態列 —— 那個位置要留給真的出問題的那個。"""
        runtime, _ = self._running(model, bus)
        runtime.pump()
        assert model.state.notices == ()

    def test_a_disabled_feature_takes_no_room_either(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """「它是關著的」由開關本身表達,不需要在狀態列再寫一句話。"""
        feature, _ = _feature(features.VISION)
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.pump()
        assert model.state.notices == ()

    def test_notices_are_replaced_not_accumulated(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime, worker = self._running(model, bus)
        worker.status.say("校正中 3/10")
        runtime.pump()
        worker.status.say("")
        runtime.pump()
        assert model.state.notices == ()

    def test_the_viewmodels_own_sticky_notice_does_not_survive(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """ViewModel 自己附加的訊息是黏住的(去重後永久留著)。

        即時模式下一幀認錯就會留一句話到程式關掉,所以狀態列由 runtime 獨佔。
        """
        feature, _ = _feature(features.VISION)
        runtime = LiveRuntime(model, bus, features=[feature])
        runtime.set_enabled(features.VISION, True)
        bus.post(CvHand(("99z",)))  # 認不得的牌 → ViewModel 會 _notice 一句
        runtime.pump()
        assert model.state.notices == ()

    def test_set_notices_is_not_called_when_nothing_changed(self, bus: UpdateBus) -> None:
        """每輪都換一次會讓 UI 每 100 ms 重畫一次,即使什麼都沒發生。"""
        seen: list[int] = []
        model = ViewModel(lambda _s: seen.append(1))
        runtime, worker = self._running(model, bus)
        worker.status.say("校正中 3/10")
        runtime.pump()
        before = len(seen)
        runtime.pump()
        runtime.pump()
        assert len(seen) == before

    def test_capture_status_comes_first(self, model: ViewModel, bus: UpdateBus) -> None:
        """擷取子程序掛掉的話後面兩個都沒有輸入,先看到根本原因比先看到症狀有用。"""
        capture = CaptureProcess(["true"], name="封包擷取")
        capture.status.say("啟動失敗")
        feature, spy = _feature(features.ADVICE, Spy("封包"))
        runtime = LiveRuntime(model, bus, features=[feature], capture=capture)
        runtime.set_enabled(features.ADVICE, True)
        spy.latest.status.say("等待對局開始")
        runtime.pump()
        assert model.state.notices[0].startswith("封包擷取:")


class TestCaptureCommand:
    def test_cdp_is_the_default(self) -> None:
        command = capture_command("/tmp/ws.jsonl")
        assert "cdp" in command
        assert command[-2:] == ["--out", "/tmp/ws.jsonl"]

    def test_it_invokes_the_gt_tool_that_actually_exists(self) -> None:
        """路徑寫錯的話子程序會在啟動的瞬間就死掉,而那個錯誤很難從 UI 看出來。"""
        command = capture_command("/tmp/ws.jsonl")
        assert Path(command[1]).is_file(), f"{command[1]} 不存在"

    @pytest.mark.parametrize("mode", ["cdp", "proxy", "local"])
    def test_all_three_capture_modes_are_accepted(self, mode: str) -> None:
        assert mode in capture_command("/tmp/ws.jsonl", mode=mode)

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知的擷取模式"):
            capture_command("/tmp/ws.jsonl", mode="telepathy")

    def test_cdp_only_flags_are_not_passed_to_the_mitm_modes(self) -> None:
        """``gt.py proxy`` 沒有 --url,傳過去它會直接 argparse 錯誤退出。"""
        command = capture_command("/tmp/ws.jsonl", mode="proxy", url="https://example.com")
        assert "--url" not in command

    def test_url_and_profile_are_passed_through_for_cdp(self) -> None:
        command = capture_command(
            "/tmp/ws.jsonl", url="https://example.com", user_data_dir="/tmp/profile"
        )
        assert "--url" in command and "https://example.com" in command
        assert "--user-data-dir" in command and "/tmp/profile" in command


class TestCaptureProcess:
    def test_a_process_that_exits_immediately_is_reported_as_a_startup_failure(self) -> None:
        process = CaptureProcess(["true"])
        process.start()
        deadline_hits = 0
        while process.status.alive and deadline_hits < 500:
            process.poll()
            deadline_hits += 1
        assert "啟動" in process.status.read()

    def test_starting_a_nonexistent_command_says_so(self) -> None:
        process = CaptureProcess(["/definitely/not/a/real/binary"])
        process.start()
        assert not process.status.alive
        assert "失敗" in process.status.read()

    def test_stop_on_a_process_that_never_started_is_harmless(self) -> None:
        CaptureProcess(["true"]).stop()
