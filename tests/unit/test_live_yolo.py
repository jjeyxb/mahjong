from mia import features
from mia.features import RecognitionChoice
from mia.live.bus import UpdateBus, WorkerStatus
from mia.live.runtime import Feature, LiveRuntime
from mia.live.yolo import YoloHub, YoloStreamSource, device_name
from mia.tracker import Stream
from mia.ui.state import UiState
from mia.ui.viewmodel import ViewModel


def test_shared_source_subscription_replay_and_detach():
    hub = YoloHub()
    received = []
    source = YoloStreamSource(hub, lambda k, s: received.append((k, s)))
    hub.publish(1, Stream(stalled=True))
    source.start()
    hub.publish(1, Stream())
    source.stop()
    hub.publish(2, Stream())
    assert received == [(1, Stream(stalled=True)), (1, Stream())]
    hub.clear()
    source.start()
    assert len(received) == 2


def test_backend_state_persists_and_unknown_names_fall_back(tmp_path):
    path = tmp_path / "ui.json"
    UiState(recognition_backend="yolo").save(path)
    assert UiState.load(path).recognition_backend == "yolo"
    assert RecognitionChoice("retired-backend").value == "template"


def test_switch_restarts_visual_workers_and_keeps_packet_worker_running():
    made = []

    class Worker:
        def __init__(self, key, source):
            self.key, self.SOURCE = key, source
            self.status = WorkerStatus(key)
            self.stopped = False

        def start(self):
            self.status.alive = True

        def stop(self):
            self.stopped = True
            self.status.alive = False

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return self.status.alive

    def factory(key, source):
        def build():
            w = Worker(key, source)
            made.append(w)
            return w

        return build

    vision = Feature(features.VISION, factory(features.VISION, "圖像辨識"))
    advice = Feature(features.ADVICE, factory(features.ADVICE, "封包"))
    danger = Feature(features.DANGER, factory(features.DANGER, "圖像辨識"))
    runtime = LiveRuntime(
        ViewModel(), UpdateBus(), features=[vision, advice, danger], recognition=RecognitionChoice()
    )
    runtime.start()
    for key in (features.VISION, features.ADVICE, features.DANGER):
        runtime.set_enabled(key, True)
    old_vision, old_advice, old_danger = vision.worker, advice.worker, danger.worker
    try:
        runtime.set_recognition_backend("yolo")
        assert runtime.recognition_backend() == "yolo"
        assert old_vision.stopped and old_danger.stopped
        assert advice.worker is old_advice and not old_advice.stopped
        assert vision.worker is not old_vision and danger.worker is not old_danger
        size = len(made)
        runtime.set_recognition_backend("yolo")
        assert len(made) == size
    finally:
        runtime.stop()


def test_panel_exposes_the_backend_choice(qtbot):
    from mia.ui.panel.window import PanelWindow

    class Picker:
        value = "template"

        def can_pick_recognition(self):
            return True

        def recognition_backend(self):
            return self.value

        def set_recognition_backend(self, value):
            self.value = value

    picker = Picker()
    panel = PanelWindow(ViewModel(), recognition=picker)
    qtbot.addWidget(panel)
    panel._recognition_picker.setCurrentIndex(1)  # noqa: SLF001
    assert picker.value == "yolo"


def test_the_three_feature_pages_share_one_backend_choice(qtbot):
    """三個功能頁各有一個「畫面辨識方式」,改任何一個,另外兩個跟著變。"""
    from mia.ui.panel.window import PanelWindow

    class Picker:
        value = "template"
        calls = 0

        def can_pick_recognition(self):
            return True

        def recognition_backend(self):
            return self.value

        def set_recognition_backend(self, value):
            self.value = value
            self.calls += 1

    picker = Picker()
    panel = PanelWindow(ViewModel(), recognition=picker)
    qtbot.addWidget(panel)
    pickers = panel._recognition_pickers  # noqa: SLF001
    pages = (panel._advice_page, panel._analysis_page, panel._danger_page)  # noqa: SLF001
    assert len(pickers) == 3
    for page, combo in zip(pages, pickers, strict=True):
        assert page.settings.isAncestorOf(combo)
    assert not panel._settings_page.isAncestorOf(pickers[0])  # noqa: SLF001

    pickers[2].setCurrentIndex(1)
    assert picker.value == "yolo"
    assert [c.currentData() for c in pickers] == ["yolo"] * 3
    assert picker.calls == 1, "同步另外兩頁時不該再重建一次畫面追蹤"

    pickers[0].setCurrentIndex(0)
    assert [c.currentData() for c in pickers] == ["template"] * 3
    assert picker.calls == 2


def test_a_long_wait_for_the_start_is_logged_with_the_reason(monkeypatch):
    """實打時 YOLO 整局停在「等待開局資訊」,重播卻讀得到。等太久要記下它卡在哪。"""
    from mia.live import yolo
    from mia.utils.logging import logger
    from mia.yolo.stream import LedgerStream

    clock = [100.0]
    monkeypatch.setattr(yolo.time, "monotonic", lambda: clock[0])
    stream = LedgerStream()
    worker = object.__new__(yolo.YoloVisionWorker)
    vars(worker).update(
        _stream=stream, _waiting_since=None, _waiting_logged="", _waiting_logged_at=0.0
    )
    watch = worker._watch_start  # noqa: SLF001
    patience = yolo._START_WARN  # noqa: SLF001
    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        stream.waiting = "start_kyoku 未齊:missing=['tehais']"
        watch(False)
        clock[0] += patience - 1
        watch(False)
        assert messages == []
        clock[0] += 2
        watch(False)
        watch(False)
        assert len(messages) == 1
        assert "tehais" in messages[0]
        clock[0] += 3
        watch(True)
    finally:
        logger.remove(sink)
    assert len(messages) == 2
    assert "等了 12.0 秒" in messages[1]


def test_a_dead_inference_process_is_restarted_a_bounded_number_of_times(monkeypatch):
    """子程序死掉以前讓整條執行緒結束,要手動關掉再開;混合辨識連樣板那條也一起收掉。"""
    from mia.engine.base import EngineError
    from mia.live import yolo

    made = []

    class FakeClient:
        def __init__(self, device=None):
            self.closed = False
            self.device = device
            made.append(self)

        def start(self):
            # made[0] 是原本那個;前兩次重啟起不來,第三次才成。
            if len(made) <= 3:
                raise EngineError("起不來")

        def close(self):
            self.closed = True

    monkeypatch.setattr(yolo, "YoloClient", FakeClient)
    worker = object.__new__(yolo.YoloVisionWorker)
    original = FakeClient()
    vars(worker).update(
        _client=original, _restarts=0, status=WorkerStatus("畫面"), label="混合辨識",
        _device="gpu",
    )
    assert worker._revive(EngineError("死了"))  # noqa: SLF001
    assert original.closed
    assert worker._client is made[-1]  # noqa: SLF001
    # 重開的子程序沿用使用者選的裝置,不會悄悄退回預設。
    assert all(client.device == "gpu" for client in made[1:])
    assert worker._restarts == 3  # noqa: SLF001
    assert not worker._revive(EngineError("又死了"))  # noqa: SLF001
    assert worker._restarts == 3  # noqa: SLF001


def test_abort_wakes_a_waiting_receive_without_closing_the_process():
    """UI 執行緒按關閉只送中斷訊號,不等子程序;等的事交給工作執行緒。"""
    import queue
    import threading

    import pytest

    from mia.engine.base import EngineError
    from mia.yolo.client import YoloClient

    client = object.__new__(YoloClient)
    vars(client).update(_closed=threading.Event(), _lines=queue.Queue(), _timeout=5.0)
    client.abort()
    with pytest.raises(EngineError, match="已停止"):
        client._receive()  # noqa: SLF001


def test_a_table_wider_than_the_screen_is_padded_not_wrapped_around():
    """Steam 全螢幕的牌桌矩形左右超出畫面。以前負的 x 讓切片從右邊數,只切到幾像素寬。"""
    import numpy as np

    from mia.live.yolo import table_frame
    from mia.utils.geometry import Rect

    image = np.full((1184, 1888, 3), 200, np.uint8)
    frame = table_frame(image, Rect(-5, 64, 1898, 1067))
    assert frame is not None
    assert frame.shape == (1080, 1920, 3)
    assert frame[540, 0].tolist() == [0, 0, 0], "超出畫面的那幾像素補黑"
    assert frame[540, 960].tolist() == [200, 200, 200]
    assert frame[540, 1919].tolist() == [0, 0, 0]
    assert table_frame(image, Rect(0, 0, 1888, 1062)) is not None
    assert table_frame(image, Rect(2000, 0, 100, 100)) is None


def test_the_steam_corner_panel_is_moved_to_where_yolo_expects_it():
    """Steam 的寶牌面板貼著視窗角落,一半在牌桌矩形上面。YOLO 要在網頁版的位置看到它。"""
    import numpy as np

    from mia.live.yolo import table_frame
    from mia.utils.geometry import Rect
    from mia.vision.hud import CORNER_PANEL

    image = np.full((1898, 3024, 3), 100, np.uint8)
    panel = Rect(30, 65, 470, 215)
    image[panel.as_slice()] = 250
    frame = table_frame(image, Rect(-8, 102, 3040, 1710), panel)
    assert frame is not None
    x, y, w, h = CORNER_PANEL
    middle = (round((y + h / 2) * 1080), round((x + w / 2) * 1920))
    assert frame[middle].tolist() == [250, 250, 250]
    assert frame[540, 960].tolist() == [100, 100, 100]
    plain = table_frame(image, Rect(-8, 102, 3040, 1710))
    assert plain is not None
    assert plain[middle].tolist() != [250, 250, 250]


def test_the_status_bar_says_gpu_not_the_torch_device_name():
    """AMD 顯示卡的 ROCm 版 torch 也回 ``cuda``,狀態列寫「CUDA」會讓人以為裝錯了。"""
    assert device_name("cuda") == "GPU"
    assert device_name("mps") == "GPU"
    assert device_name("xpu") == "GPU"  # Intel 顯示卡
    assert device_name("cpu") == "CPU"
    assert device_name("npu") == "NPU"
