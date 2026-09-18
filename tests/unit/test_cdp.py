"""CDP 擷取的 frame 處理測試。

用假的 WebSocket 物件驗證邏輯,不需要真的起瀏覽器 —— 瀏覽器那段已在
開發時以本機 WebSocket 伺服器做過端到端驗證。這裡確保的是網址過濾、
方向判斷、文字 frame 略過、flow 識別這些自己寫的部分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mia.groundtruth.cdp import DEFAULT_URL_PATTERNS, CdpCapture
from mia.groundtruth.dump import DumpWriter, iter_frames

GAME_URL = "wss://gateway-hw.maj-soul.com:443/game-gateway"
OTHER_URL = "wss://analytics.example.com/track"


class FakeWebSocket:
    """模仿 Playwright 的 WebSocket:只提供 CdpCapture 真正用到的介面。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self._handlers: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload: Any) -> None:
        for handler in self._handlers.get(event, []):
            handler(payload)

    @property
    def listening(self) -> bool:
        return bool(self._handlers)


@pytest.fixture
def capture(tmp_path: Path) -> CdpCapture:
    writer = DumpWriter(tmp_path / "ws.jsonl")
    return CdpCapture(writer, url_patterns=DEFAULT_URL_PATTERNS)


def frames_of(capture: CdpCapture) -> list:
    capture.writer.close()
    return list(iter_frames(capture.writer.path))


class TestUrlFiltering:
    def test_majsoul_socket_is_attached(self, capture: CdpCapture) -> None:
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)

        assert ws.listening
        assert capture.stats.sockets_matched == 1

    def test_unrelated_socket_is_ignored(self, capture: CdpCapture) -> None:
        """頁面上常有分析、廣告之類的 WebSocket,不該混進錄影檔。"""
        ws = FakeWebSocket(OTHER_URL)
        capture.attach_websocket(ws)

        assert not ws.listening
        assert capture.stats.sockets_matched == 0
        assert capture.stats.sockets_seen == 1
        assert OTHER_URL in capture.stats.urls_ignored

    def test_custom_patterns(self, tmp_path: Path) -> None:
        capture = CdpCapture(DumpWriter(tmp_path / "d.jsonl"), url_patterns=["example.com"])
        ws = FakeWebSocket(OTHER_URL)
        capture.attach_websocket(ws)
        assert capture.stats.sockets_matched == 1

    def test_empty_patterns_match_everything(self, tmp_path: Path) -> None:
        capture = CdpCapture(DumpWriter(tmp_path / "d.jsonl"), url_patterns=[])
        ws = FakeWebSocket(OTHER_URL)
        capture.attach_websocket(ws)
        assert capture.stats.sockets_matched == 1


class TestFrameHandling:
    def test_binary_frames_recorded_with_direction(self, capture: CdpCapture) -> None:
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)

        ws.emit("framereceived", b"\x01server")
        ws.emit("framesent", b"\x02client")

        frames = frames_of(capture)
        assert [f.payload for f in frames] == [b"\x01server", b"\x02client"]
        assert [f.from_client for f in frames] == [False, True]

    def test_text_frames_are_skipped(self, capture: CdpCapture) -> None:
        """雀魂協定走 binary;Playwright 會把文字 frame 給成 str。"""
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)

        ws.emit("framereceived", "ping")
        ws.emit("framereceived", b"\x01real")

        assert capture.stats.text_frames == 1
        assert capture.stats.frames == 1
        assert [f.payload for f in frames_of(capture)] == [b"\x01real"]

    def test_binary_payload_is_byte_exact(self, capture: CdpCapture) -> None:
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)
        payload = bytes(range(256))

        ws.emit("framereceived", payload)

        assert frames_of(capture)[0].payload == payload


class TestFlowIdentity:
    def test_same_socket_keeps_one_flow_id(self, capture: CdpCapture) -> None:
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)

        ws.emit("framereceived", b"\x01a")
        ws.emit("framereceived", b"\x01b")

        assert len({f.flow for f in frames_of(capture)}) == 1

    def test_different_sockets_get_different_flow_ids(self, capture: CdpCapture) -> None:
        """每條連線有自己的 msg_id 序列,識別必須分得開。"""
        first, second = FakeWebSocket(GAME_URL), FakeWebSocket(GAME_URL)
        capture.attach_websocket(first)
        capture.attach_websocket(second)

        first.emit("framereceived", b"\x01a")
        second.emit("framereceived", b"\x01b")

        flows = [f.flow for f in frames_of(capture)]
        assert len(set(flows)) == 2


class TestStats:
    def test_summary_reports_counts(self, capture: CdpCapture) -> None:
        ws = FakeWebSocket(GAME_URL)
        capture.attach_websocket(ws)
        ws.emit("framereceived", b"\x01a")
        ws.emit("framereceived", "text")

        summary = capture.stats.summary()
        assert "1 個 binary frame" in summary
        assert "略過文字 frame 1 個" in summary

    def test_ignored_urls_listed_for_debugging(self, capture: CdpCapture) -> None:
        capture.attach_websocket(FakeWebSocket(OTHER_URL))
        assert OTHER_URL in capture.stats.summary()


class FakeWindow:
    """模仿高 DPI 螢幕上的瀏覽器視窗,重現兩個真機量到的行為。

    1. **DIP → CSS 像素不是 1:1。** 視窗尺寸是 DIP,換算成 CSS 像素中間要過一次
       實體像素的四捨五入。125% 下實測差一格 DIP 可能讓 CSS 寬不動、也可能跳 2。
    2. **``getWindowBounds`` 不會原樣回報 ``setWindowBounds`` 送進去的值。**
       實測送 1614 進去,問回來是 1615、再問是 1617,每問一次漂一點。

    第 2 點是真正咬人的那個:拿回報值當基準去算下一步,基準會一路漂走,
    最後掃描範圍涵蓋不到正確答案 —— 而畫面上只會看到「調不到畫布」,
    完全看不出是這個原因。
    """

    def __init__(self, dpr: float = 1.25, chrome_dip: int = 14) -> None:
        self.dpr = dpr
        self.chrome_dip = chrome_dip  # 工具列等等佔掉的 DIP 高度
        self.width = 1000
        self.height = 800
        self._reported_drift = 0

    def _css(self, dip: int) -> int:
        """DIP → CSS 像素,經過實體像素的四捨五入。"""
        return int(round(dip * self.dpr) / self.dpr)

    @property
    def viewport(self) -> list[int]:
        return [self._css(self.width), self._css(self.height - self.chrome_dip)]

    @property
    def reported_bounds(self) -> dict[str, int]:
        self._reported_drift += 1
        return {
            "width": self.width + self._reported_drift,
            "height": self.height + self._reported_drift,
        }


class FakePage:
    """只提供 _fit_canvas 真正用到的那幾個介面。"""

    def __init__(self, window: FakeWindow) -> None:
        self.window = window
        self.context = self
        self.resizes = 0

    # --- page ---
    def evaluate(self, _script: str) -> list[int]:
        return self.window.viewport

    def wait_for_timeout(self, _ms: float) -> None:
        return None

    # --- context ---
    def new_cdp_session(self, _page: object) -> FakePage:
        return self

    # --- cdp session ---
    def send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if method == "Browser.getWindowForTarget":
            return {"windowId": 1}
        if method == "Browser.getWindowBounds":
            return {"bounds": self.window.reported_bounds}
        if method == "Browser.setWindowBounds":
            assert params is not None
            bounds = params["bounds"]
            # windowState 一定要一起送,否則最大化的視窗會默默忽略寬高
            assert bounds["windowState"] == "normal"
            self.window.width = bounds["width"]
            self.window.height = bounds["height"]
            self.resizes += 1
            return {}
        raise AssertionError(f"沒預期到的 CDP 方法:{method}")


class TestFitCanvas:
    """高 DPI 下把 viewport 調到剛好等於畫布。

    對應 2026-09-18 真機驗證抓到的失敗:125% 螢幕上收斂不到 1600×900、
    停在 1602×902,於是 ``Canvas.table_rect`` 判定對不上,畫面辨識鎖不上。
    """

    @pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
    def test_hits_the_canvas_exactly(self, dpr: float) -> None:
        from mia.calibration.canvas import Canvas
        from mia.groundtruth.cdp import _fit_canvas

        page = FakePage(FakeWindow(dpr=dpr))
        _fit_canvas(page, Canvas(1600, 900))
        assert page.window.viewport == [1600, 900], f"dpr={dpr} 收斂不到"

    def test_does_not_trust_reported_bounds(self) -> None:
        """回報值每問一次漂一點,而結果仍然要精確 —— 這是那個 bug 的本體。"""
        from mia.calibration.canvas import Canvas
        from mia.groundtruth.cdp import _fit_canvas

        window = FakeWindow(dpr=1.25)
        page = FakePage(window)
        _fit_canvas(page, Canvas(1280, 720))
        assert window.viewport == [1280, 720]
