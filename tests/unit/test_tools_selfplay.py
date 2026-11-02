"""自己打一場:記錄的東西要對得上時間,重播要與實打看到同樣的畫面。

打牌與擷取本身要真的跑才驗得到。這裡測的是接起來的那幾段,它們錯了不會有症狀,只會讓
報告裡的延遲是錯的。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mia.capture.base import Frame, WindowInfo
from mia.engine.base import Advice
from mia.live.bus import Advices
from mia.mjai.events import Dahai
from mia.tracker import Stream
from mia.utils.geometry import Rect

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def selfplay():
    spec = importlib.util.spec_from_file_location("_selfplay", ROOT / "tools" / "selfplay.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_selfplay"] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


WINDOW = WindowInfo(handle=1, title="雀魂麻將", owner="Google Chrome for Testing",
                    bounds=Rect(0, 0, 100, 100))


class TestRunLog:
    def test_the_same_stream_is_written_once(self, selfplay, tmp_path) -> None:
        """一秒讀好幾次,事件流大多沒變。"""
        log = selfplay.RunLog(tmp_path / "v.jsonl")
        stream = Stream(events=(Dahai(actor=0, pai="1m", tsumogiri=True),))
        log.stream(1, stream)
        log.stream(1, stream)
        log.close()
        assert len(_rows(tmp_path / "v.jsonl")) == 1

    def test_a_new_note_is_a_new_row(self, selfplay, tmp_path) -> None:
        """事件沒變但卡住了,那段時間要算得出來。"""
        log = selfplay.RunLog(tmp_path / "v.jsonl")
        events = (Dahai(actor=0, pai="1m", tsumogiri=True),)
        log.stream(1, Stream(events=events))
        log.stream(1, Stream(events=events, stalled=True))
        log.stream(1, Stream(events=events, doubt="對不上"))
        log.close()
        assert [row["note"] for row in _rows(tmp_path / "v.jsonl")] == ["", "等手牌", "對不上"]

    def test_the_clock_is_the_wall_clock(self, selfplay, tmp_path) -> None:
        """要與封包比,兩邊都要是牆上時鐘。"""
        log = selfplay.RunLog(tmp_path / "v.jsonl")
        log.write(kind="status", name="x", message="y")
        log.close()
        assert _rows(tmp_path / "v.jsonl")[0]["wall"] == pytest.approx(time.time(), abs=5)


class TestWhatGetsLogged:
    def test_only_real_advice(self, selfplay, tmp_path) -> None:
        """開局會投一份空的建議,那不是給出去的答案。"""
        log = selfplay.RunLog(tmp_path / "v.jsonl")
        bus = selfplay.TapBus(log)
        action = Dahai(actor=0, pai="1m", tsumogiri=True)
        bus.post(Advices((Advice("mortal"),)))
        bus.post(Advices((Advice("mortal", action=action),)))
        log.close()
        rows = _rows(tmp_path / "v.jsonl")
        assert [row["action"]["pai"] for row in rows] == ["1m"]
        assert bus.drain(), "郵箱照常要收到"

    def test_a_status_line_is_kept_once(self, selfplay, tmp_path) -> None:
        log = selfplay.RunLog(tmp_path / "v.jsonl")
        status = selfplay.LoggedStatus("畫面", log)
        status.say("校正中")
        status.say("校正中")
        status.say("")
        log.close()
        assert [row["message"] for row in _rows(tmp_path / "v.jsonl")] == ["校正中", ""]
        assert status.read() == ""


def _session(stamps):
    """假的 session:每個時間一張圖,圖的亮度是第幾張。"""
    records = [SimpleNamespace(timestamp=t, image=f"{i}.jpg", scale=2.0)
               for i, t in enumerate(stamps)]
    manifest = SimpleNamespace(frames=records, first_frame_wall=1000.0, duration=stamps[-1])

    def load(record):
        return np.full((4, 4, 3), int(record.image.split(".")[0]), np.uint8)

    return SimpleNamespace(manifest=manifest, load_image=load)


class TestReplay:
    def test_it_starts_at_the_requested_moment(self, selfplay) -> None:
        backend = selfplay.ReplayBackend(_session([0.0, 10.0, 20.0]), 1010.0, 1020.0)
        assert backend.capture(WINDOW).image[0, 0, 0] == 1
        assert backend.wall == pytest.approx(1010.0, abs=0.2)

    def test_time_moves_at_the_real_pace(self, selfplay, monkeypatch) -> None:
        backend = selfplay.ReplayBackend(_session([0.0, 10.0, 20.0]), 1000.0, 1020.0)
        now = [100.0]
        monkeypatch.setattr(selfplay.time, "monotonic", lambda: now[0])
        assert backend.capture(WINDOW).image[0, 0, 0] == 0
        now[0] = 111.0
        assert backend.capture(WINDOW).image[0, 0, 0] == 1
        assert not backend.finished
        now[0] = 121.0
        assert backend.capture(WINDOW).image[0, 0, 0] == 2
        assert backend.finished

    def test_an_unchanged_frame_is_the_same_bytes(self, selfplay) -> None:
        """手牌區要位元組相同才認,同一個檔案不能解碼出不同的東西。"""
        backend = selfplay.ReplayBackend(_session([0.0, 10.0]), 1000.0, 1010.0)
        first: Frame = backend.capture(WINDOW)
        assert backend.capture(WINDOW).image is first.image

    def test_the_log_uses_the_recording_clock(self, selfplay, tmp_path) -> None:
        """重播現在發生,但要對的封包是當時的。"""
        backend = selfplay.ReplayBackend(_session([0.0, 10.0]), 1005.0, 1010.0)
        log = selfplay.ReplayLog(tmp_path / "v.jsonl", backend)
        log.write(kind="status", name="x", message="y")
        log.close()
        assert _rows(tmp_path / "v.jsonl")[0]["wall"] == pytest.approx(1005.0)


class TestSpan:
    TRUTH = (SimpleNamespace(start=1100.0), SimpleNamespace(start=1200.0))

    def _session(self):
        return SimpleNamespace(
            manifest=SimpleNamespace(first_frame_wall=1000.0, duration=300.0)
        )

    def test_the_whole_game(self, selfplay) -> None:
        assert selfplay._span(self.TRUTH, None, self._session()) == (1000.0, 1300.0)  # noqa: SLF001

    def test_one_kyoku_starts_a_bit_early(self, selfplay) -> None:
        assert selfplay._span(self.TRUTH, 1, self._session()) == (1095.0, 1200.0)  # noqa: SLF001

    def test_the_last_kyoku_runs_to_the_end(self, selfplay) -> None:
        assert selfplay._span(self.TRUTH, 2, self._session()) == (1195.0, 1300.0)  # noqa: SLF001

    def test_a_kyoku_that_does_not_exist(self, selfplay) -> None:
        with pytest.raises(SystemExit):
            selfplay._span(self.TRUTH, 3, self._session())  # noqa: SLF001


class TestWhichWindow:
    def _backend(self, *windows):
        return SimpleNamespace(list_windows_everywhere=lambda: list(windows))

    def _config(self):
        from mia.config.loader import load_config

        return load_config()

    def test_a_github_page_is_not_the_game(self, selfplay, monkeypatch) -> None:
        """實測鎖到 Safari 開著這個專案的 GitHub 頁面,標題也有「雀魂」。"""
        safari = WindowInfo(handle=2, title="jjeyxb/mahjong: MIA:雀魂輔助工具", owner="Safari",
                            bounds=Rect(0, 0, 1512, 870))
        monkeypatch.setattr(selfplay, "_WINDOW_TIMEOUT", 0.2)
        monkeypatch.setattr(selfplay.time, "sleep", lambda _: None)
        got = selfplay._game_window(self._backend(safari, WINDOW), self._config())  # noqa: SLF001
        assert got is WINDOW

    def test_no_browser_yet(self, selfplay, monkeypatch) -> None:
        monkeypatch.setattr(selfplay, "_WINDOW_TIMEOUT", 0.05)
        assert selfplay._game_window(self._backend(), self._config()) is None  # noqa: SLF001


class TestABlackScreen:
    """視窗沒被畫出來時擷取拿到全黑的圖,不會報錯。實測一場前 23 分鐘都是這樣。"""

    @staticmethod
    def _frame(level: int, at: float) -> Frame:
        image = np.full((64, 64, 3), level, np.uint8)
        return Frame(image, WINDOW, captured_at=at)

    def test_black_and_not(self, selfplay) -> None:
        assert selfplay.is_black(np.zeros((64, 64, 3), np.uint8))
        assert not selfplay.is_black(np.full((64, 64, 3), 70, np.uint8))

    def test_it_adds_up_the_seconds(self, selfplay) -> None:
        recorder = selfplay.FrameRecorder(SimpleNamespace())
        for level, at in ((70, 1.0), (0, 2.0), (0, 5.0), (0, 7.0), (70, 8.0)):
            recorder._watch_black(self._frame(level, at))  # noqa: SLF001
        assert recorder.black == pytest.approx(5.0)
        assert recorder.black_since is None

    def test_it_remembers_when_it_went_black(self, selfplay) -> None:
        recorder = selfplay.FrameRecorder(SimpleNamespace())
        for at in (2.0, 3.0):
            recorder._watch_black(self._frame(0, at))  # noqa: SLF001
        assert recorder.black_since == 2.0


class TestKeepingTheScreenOn:
    def test_it_lasts_as_long_as_this_process(self, selfplay, monkeypatch) -> None:
        import os

        started = []
        monkeypatch.setattr(selfplay.sys, "platform", "darwin")
        monkeypatch.setattr(selfplay.subprocess, "Popen", lambda args: started.append(args))
        selfplay._keep_awake()  # noqa: SLF001
        assert started == [["caffeinate", "-d", "-i", "-w", str(os.getpid())]]

    def test_other_platforms_do_nothing(self, selfplay, monkeypatch) -> None:
        started = []
        monkeypatch.setattr(selfplay.sys, "platform", "win32")
        monkeypatch.setattr(selfplay.subprocess, "Popen", lambda args: started.append(args))
        selfplay._keep_awake()  # noqa: SLF001
        assert started == []
