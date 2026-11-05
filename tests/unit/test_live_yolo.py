from mia import features
from mia.features import RecognitionChoice
from mia.live.bus import UpdateBus, WorkerStatus
from mia.live.runtime import Feature, LiveRuntime
from mia.live.yolo import YoloHub, YoloStreamSource
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
