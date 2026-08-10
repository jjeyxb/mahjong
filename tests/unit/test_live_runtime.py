"""即時模式的協調層。

沒有 Qt、沒有執行緒 —— pump 是同步的,所以整條路可以直接呼叫來驗。
那正是把它做成「郵箱 + pump」而不是 Qt signal 的目的。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mia import features
from mia.calibration.canvas import Canvas, CanvasChoice
from mia.engine.base import Advice
from mia.live.bus import Advices, CvHand, PacketHand, UpdateBus, WorkerStatus
from mia.live.control import ControlFile
from mia.live.runtime import Feature, LiveRuntime
from mia.live.source import (
    STARTUP_HINT,
    CaptureLauncher,
    CaptureProcess,
    capture_command,
)
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


def _launcher(command: list[str] | None = None, root: Path | None = None) -> CaptureLauncher:
    """一個不會真的開瀏覽器的 launcher。

    ``sleep`` 而不是 ``true``:要驗「已經在跑的時候不再開一個」就需要一個
    活得夠久的子程序。呼叫端記得 ``stop()``。
    """
    return CaptureLauncher(
        lambda _dump: command or ["sleep", "30"],
        root=root or Path("/tmp/mia-test-live"),
    )


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

    def test_start_does_not_open_the_browser(self, model: ViewModel, bus: UpdateBus) -> None:
        """開一個瀏覽器並開始往磁碟寫錄影檔,跟載 130MB 權重是同一個層級的事
        —— 該是使用者按下「開始遊戲」的結果,不是打開視窗的副作用。
        """
        capture = _launcher()
        LiveRuntime(model, bus, capture=capture).start()
        assert not capture.running
        assert capture.status.read() == ""

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
        capture = _launcher()
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

    def test_canvas_reaches_the_subprocess(self) -> None:
        """畫布尺寸是**開視窗時下的命令**,只能靠命令列送過去。

        漏掉的話 UI 上選了尺寸、瀏覽器照樣用預設大小開,而 CV 會說「畫布
        對不上」—— 症狀看起來像畫布功能壞了,實際上是這一行沒接。
        """
        command = capture_command("/tmp/ws.jsonl", canvas="1920x1080")
        assert command[command.index("--canvas") + 1] == "1920x1080"

    def test_canvas_is_not_sent_when_attaching_to_someone_elses_browser(self) -> None:
        """``--connect`` 過去的視窗不是我們開的,不該去改它的大小。"""
        command = capture_command(
            "/tmp/ws.jsonl", canvas="1920x1080", connect="http://localhost:9222"
        )
        assert "--canvas" not in command

    def test_the_control_file_reaches_the_subprocess(self) -> None:
        """開下去之後主程式還想改設定,只能靠這個檔案。"""
        command = capture_command("/tmp/ws.jsonl", control="/tmp/control.json")
        assert command[command.index("--control") + 1] == "/tmp/control.json"

    def test_no_control_file_when_attaching_to_someone_elses_browser(self) -> None:
        command = capture_command(
            "/tmp/ws.jsonl", control="/tmp/c.json", connect="http://localhost:9222"
        )
        assert "--control" not in command

    def test_canvas_is_a_cdp_only_flag(self) -> None:
        command = capture_command("/tmp/ws.jsonl", mode="proxy", canvas="1920x1080")
        assert "--canvas" not in command

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


class TestStartGame:
    """「開始遊戲」按下去之後。"""

    @pytest.fixture
    def wired(self, model: ViewModel, bus: UpdateBus, tmp_path: Path):
        capture = _launcher(root=tmp_path)
        feature, spy = _feature(features.ADVICE, Spy("封包"))
        runtime = LiveRuntime(model, bus, features=[feature], capture=capture)
        runtime.start()
        yield runtime, capture, spy
        runtime.stop()

    def test_it_launches(self, wired) -> None:
        runtime, capture, _ = wired
        runtime.start_game()
        assert capture.running
        assert runtime.game_running()

    def test_pressing_it_twice_does_not_open_a_second_browser(self, wired) -> None:
        """按這個按鈕的意思從來不是「把現在這場砍了重開」。"""
        runtime, capture, _ = wired
        runtime.start_game()
        first = capture.dump
        runtime.start_game()
        assert capture.dump == first

    def test_each_game_gets_its_own_dump_file(self, model, bus, tmp_path) -> None:
        """DumpWriter 是 append 模式開的 —— 兩場寫進同一個檔案的話,從檔頭讀的
        封包執行緒會先把上一場整個重播一遍,然後拿著一個已經結束的牌局給建議。
        """
        capture = _launcher(command=["true"], root=tmp_path)
        runtime = LiveRuntime(model, bus, capture=capture)
        runtime.start_game()
        first = capture.dump
        _wait_until_dead(capture)
        runtime.start_game()
        assert capture.dump != first

    def test_the_advice_worker_is_restarted_for_a_new_game(self, model, bus, tmp_path) -> None:
        """它的錄影檔路徑是建構時決定的。不重開的話畫面會停在上一場的最後一手,
        **而且不會有任何錯誤訊息**。
        """
        capture = _launcher(command=["true"], root=tmp_path)
        feature, spy = _feature(features.ADVICE, Spy("封包"))
        runtime = LiveRuntime(model, bus, features=[feature], capture=capture)
        runtime.start_game()
        runtime.set_enabled(features.ADVICE, True)
        _wait_until_dead(capture)
        runtime.start_game()
        assert len(spy.made) == 2
        runtime.stop()

    def test_the_first_game_does_not_restart_anything(self, wired) -> None:
        """先打開 AI 建議再按開始遊戲也要成立 —— 錄影檔路徑是懶決定的,
        那個 worker 等的正是這一場要寫的檔案(DumpTail 會等檔案出現)。
        """
        runtime, _, spy = wired
        runtime.set_enabled(features.ADVICE, True)
        runtime.start_game()
        assert len(spy.made) == 1

    def test_a_disabled_advice_feature_is_left_alone(self, model, bus, tmp_path) -> None:
        """換場不該順手把使用者關著的功能打開。"""
        capture = _launcher(command=["true"], root=tmp_path)
        feature, spy = _feature(features.ADVICE, Spy("封包"))
        runtime = LiveRuntime(model, bus, features=[feature], capture=capture)
        runtime.start_game()
        _wait_until_dead(capture)
        runtime.start_game()
        assert spy.made == []

    def test_without_a_capture_it_is_a_no_op(self, model: ViewModel, bus: UpdateBus) -> None:
        """--tail / --no-packets:遊戲不是 MIA 開的。"""
        runtime = LiveRuntime(model, bus)
        runtime.start_game()
        assert not runtime.can_start_game()
        assert not runtime.game_running()

    def test_stop_takes_the_browser_with_it(self, wired) -> None:
        runtime, capture, _ = wired
        runtime.start_game()
        runtime.stop()
        capture.poll()
        assert not capture.running


class TestCaptureLauncher:
    def test_the_dump_path_is_decided_lazily_but_stays_put(self, tmp_path: Path) -> None:
        """先問路徑、之後才開始 —— 「先打開 AI 建議再按開始遊戲」靠的就是這個。"""
        launcher = _launcher(root=tmp_path)
        assert launcher.dump == launcher.dump

    def test_the_command_is_built_with_that_path(self, tmp_path: Path) -> None:
        seen: list[Path] = []
        launcher = CaptureLauncher(
            lambda dump: seen.append(dump) or ["true"],  # type: ignore[func-returns-value]
            root=tmp_path,
        )
        launcher.launch()
        assert seen == [launcher.dump]

    def test_two_games_in_the_same_second_do_not_collide(self, tmp_path: Path) -> None:
        """時間戳只到秒。撞到就等於兩場寫進同一個檔案,正是要避免的事。

        不能靠擷取子程序去建目錄來佔名字 —— 它要啟動、連上、收到第一個 frame
        才會建,而在那之前這個名字看起來還是空的。
        """
        launcher = _launcher(command=["true"], root=tmp_path)
        launcher.launch()
        first = launcher.dump
        _wait_until_dead(launcher)
        launcher.launch()
        assert launcher.dump != first

    def test_the_status_object_survives_a_new_game(self, tmp_path: Path) -> None:
        """狀態列盯著的是同一個物件 —— 換場之後換掉的話,UI 會一直看著上一場
        那個已經不會再更新的狀態。
        """
        launcher = _launcher(command=["true"], root=tmp_path)
        status = launcher.status
        launcher.launch()
        _wait_until_dead(launcher)
        launcher.launch()
        assert launcher.status is status
        launcher.stop()

    def test_it_says_nothing_before_the_first_game(self, tmp_path: Path) -> None:
        """狀態列那個位置要留給真的出問題的那個 —— 還沒開始不是問題。"""
        assert _launcher(root=tmp_path).status.read() == ""


def _wait_until_dead(launcher: CaptureLauncher, timeout: float = 5.0) -> None:
    """等子程序真的結束。

    要 sleep:不 sleep 的話 500 次 poll 在一毫秒內就跑完了,而子程序還沒被
    排到 —— 那是個會在別台機器上偶發的假失敗。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        launcher.poll()
        if not launcher.running:
            return
        time.sleep(0.01)
    raise AssertionError("子程序沒有結束")


class TestCanvas:
    """畫布尺寸的選擇怎麼傳下去。見 :mod:`mia.calibration.canvas`。"""

    def test_no_choice_means_the_option_is_unavailable(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """重播與示範模式沒有畫布可選,UI 要分得出來才能把選單畫成停用。"""
        feature, _ = _feature()
        runtime = LiveRuntime(model, bus, features=[feature])
        assert not runtime.can_pick_canvas()
        assert runtime.canvas() is None

    def test_unavailable_without_vision(self, model: ViewModel, bus: UpdateBus) -> None:
        """``--no-vision`` 時牌桌矩形沒有人要用 —— 選了也不影響任何東西。"""
        feature, _ = _feature(features.ADVICE)
        runtime = LiveRuntime(model, bus, features=[feature], canvas=CanvasChoice())
        assert not runtime.can_pick_canvas()

    def test_setting_it_restarts_vision(self, model: ViewModel, bus: UpdateBus) -> None:
        """校正器是 worker **建構時**吃設定的,不重開就還在用舊的畫布。

        不重開的話症狀是「選了沒反應」,而使用者唯一能做的事就是再選一次
        —— 那次會因為「值沒變」被忽略掉。
        """
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature], canvas=CanvasChoice())
        runtime.start()
        runtime.set_enabled(features.VISION, True)
        assert len(spy.made) == 1

        runtime.set_canvas("1920x1080")
        assert runtime.canvas() == "1920x1080"
        assert len(spy.made) == 2, "改了畫布卻沒有重建 worker"
        runtime.stop()

    def test_a_running_browser_is_told_to_resize(self, model: ViewModel, bus: UpdateBus) -> None:
        """改了尺寸,**正在跑的瀏覽器要當場跟著變**。

        使用者的原話:「我還希望能在設定尺寸後能自動把瀏覽器視窗尺寸也跟著修改」。
        尺寸是開子程序時用命令列傳的,開下去之後就只剩控制檔這條路。
        """
        feature, _ = _feature()
        launcher = _launcher()
        runtime = LiveRuntime(
            model, bus, features=[feature], capture=launcher, canvas=CanvasChoice()
        )
        runtime.start()
        runtime.start_game()
        try:
            runtime.set_canvas("1920x1080")
            command = ControlFile(launcher.control.path).poll()
            assert command is not None, "沒有把新尺寸送給擷取子程序"
            assert command["canvas"] == "1920x1080"
        finally:
            runtime.stop()

    def test_nothing_is_pushed_when_no_game_is_running(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """沒在跑就不必推 —— 下次「開始遊戲」時新尺寸本來就會走命令列過去。"""
        feature, _ = _feature()
        launcher = _launcher()
        runtime = LiveRuntime(
            model, bus, features=[feature], capture=launcher, canvas=CanvasChoice()
        )
        runtime.start()
        runtime.set_canvas("1920x1080")
        assert not launcher.control.path.exists()
        runtime.stop()

    def test_setting_the_same_value_changes_nothing(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """重開一次要付重建 worker 的成本,值沒變就不該付。"""
        feature, spy = _feature()
        runtime = LiveRuntime(
            model, bus, features=[feature], canvas=CanvasChoice(Canvas(1920, 1080))
        )
        runtime.start()
        runtime.set_enabled(features.VISION, True)
        runtime.set_canvas("1920x1080")
        assert len(spy.made) == 1
        runtime.stop()

    def test_a_disabled_feature_is_not_started_by_picking(
        self, model: ViewModel, bus: UpdateBus
    ) -> None:
        """選一個尺寸不等於「請開始擷取螢幕」。"""
        feature, spy = _feature()
        runtime = LiveRuntime(model, bus, features=[feature], canvas=CanvasChoice())
        runtime.start()
        runtime.set_canvas("1280x720")
        assert spy.made == []
        assert not feature.enabled


class TestStartupHint:
    """「請在瀏覽器裡登入並開始對局」是**指示**,不是狀態。

    它原本永遠不會消失:CaptureProcess.poll 只在子程序死掉時改寫訊息,所以
    只要瀏覽器活著,那句話就一路掛到程式關掉 —— 打到南四局了還在叫人登入。
    而 WorkerStatus 的約定是「沒話說就留空,那個位置要留給真的出問題的那個」。
    """

    def _started(self, model: ViewModel, bus: UpdateBus) -> tuple[LiveRuntime, CaptureLauncher]:
        capture = _launcher()
        runtime = LiveRuntime(model, bus, capture=capture)
        capture.launch()
        return runtime, capture

    def test_it_shows_while_nothing_has_arrived(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime, capture = self._started(model, bus)
        try:
            runtime.pump()
            assert model.state.notices == (f"封包擷取:{STARTUP_HINT}",)
        finally:
            capture.stop()

    def test_a_hand_from_the_packets_clears_it(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime, capture = self._started(model, bus)
        try:
            bus.post(PacketHand(("1m", "1m", "1m")))
            runtime.pump()
            assert model.state.notices == ()
        finally:
            capture.stop()

    def test_a_hand_from_the_screen_clears_it_too(self, model: ViewModel, bus: UpdateBus) -> None:
        """``--no-packets`` 時只有畫面會有東西進來 —— 只認封包的話那個模式下
        這句提示永遠收不掉。"""
        runtime, capture = self._started(model, bus)
        try:
            bus.post(CvHand(("1m", "1m", "1m")))
            runtime.pump()
            assert model.state.notices == ()
        finally:
            capture.stop()

    def test_it_stays_cleared(self, model: ViewModel, bus: UpdateBus) -> None:
        runtime, capture = self._started(model, bus)
        try:
            bus.post(CvHand(("1m",)))
            runtime.pump()
            runtime.pump()
            assert model.state.notices == ()
        finally:
            capture.stop()

    def test_bad_news_is_not_cleared(self, model: ViewModel, bus: UpdateBus) -> None:
        """子程序在這中間死掉的話,狀態列上是「已結束」—— 那句不能被蓋掉。"""
        runtime, capture = self._started(model, bus)
        capture.status.say("封包擷取已結束(exit 1),不會再有新的建議")
        bus.post(CvHand(("1m",)))
        runtime.pump()
        assert model.state.notices == ("封包擷取:封包擷取已結束(exit 1),不會再有新的建議",)
        capture.stop()

    def test_settling_without_a_process_does_nothing(self) -> None:
        """還沒按開始遊戲就有畫面進來(接已在跑的瀏覽器)—— 沒有提示要收。"""
        _launcher().settle()


class TestClearIf:
    def test_it_clears_a_matching_message(self) -> None:
        status = WorkerStatus("封包擷取", message="哈囉")
        assert status.clear_if("哈囉")
        assert status.read() == ""

    def test_it_leaves_a_different_message_alone(self) -> None:
        status = WorkerStatus("封包擷取", message="出事了")
        assert not status.clear_if("哈囉")
        assert status.read() == "出事了"

    def test_clearing_twice_is_harmless(self) -> None:
        status = WorkerStatus("封包擷取", message="哈囉")
        status.clear_if("哈囉")
        assert not status.clear_if("哈囉")
        assert status.read() == ""
