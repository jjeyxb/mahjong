"""CDP 擷取的 frame 處理測試。

用假的 WebSocket 物件驗證邏輯,不需要真的起瀏覽器 —— 瀏覽器那段已在
開發時以本機 WebSocket 伺服器做過端到端驗證。這裡確保的是網址過濾、
方向判斷、文字 frame 略過、flow 識別這些自己寫的部分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from majsoul_copilot.groundtruth.cdp import DEFAULT_URL_PATTERNS, CdpCapture
from majsoul_copilot.groundtruth.dump import DumpWriter, iter_frames

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
