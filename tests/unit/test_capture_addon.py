"""mitmproxy addon 的測試 —— 用假的 flow 物件,不需要真的起代理。

真實流量能不能攔到,取決於憑證信任與客戶端是否遵循系統代理設定,那是環境問題;
這裡驗證的是 addon 自己的邏輯:網址過濾、方向判斷、輸出格式。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from majsoul_copilot.groundtruth.capture_addon import WebSocketDump


@dataclass
class FakeMessage:
    content: bytes
    from_client: bool
    is_text: bool = False


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[FakeMessage] = []


class FakeRequest:
    def __init__(self, url: str) -> None:
        self.pretty_url = url


class FakeFlow:
    """模仿 mitmproxy 的 HTTPFlow:只提供 addon 真正會用到的屬性。"""

    def __init__(self, url: str) -> None:
        self.request = FakeRequest(url)
        self.websocket = FakeWebSocket()

    def send(self, content: bytes, *, from_client: bool = False, is_text: bool = False) -> None:
        self.websocket.messages.append(FakeMessage(content, from_client, is_text))


GAME_URL = "wss://gateway-hw.maj-soul.com:443/game-gateway"
OTHER_URL = "wss://example.com/socket"


@pytest.fixture
def dump(tmp_path: Path) -> WebSocketDump:
    return WebSocketDump(output=tmp_path / "ws.jsonl")


def _lines(dump: WebSocketDump) -> list[dict]:
    if not dump.path.is_file():
        return []
    return [json.loads(line) for line in dump.path.read_text(encoding="utf-8").splitlines() if line]


class TestFiltering:
    def test_target_url_is_recorded(self, dump: WebSocketDump) -> None:
        flow = FakeFlow(GAME_URL)
        flow.send(b"\x01payload")
        dump.websocket_message(flow)

        assert dump.frames == 1
        assert len(_lines(dump)) == 1

    def test_unrelated_url_is_ignored(self, dump: WebSocketDump) -> None:
        """避免把其他應用程式的 WebSocket 混進錄影檔。"""
        flow = FakeFlow(OTHER_URL)
        flow.send(b"\x01payload")
        dump.websocket_message(flow)

        assert dump.frames == 0
        assert _lines(dump) == []

    def test_custom_filter(self, tmp_path: Path) -> None:
        dump = WebSocketDump(output=tmp_path / "ws.jsonl", url_filter="example.com")
        flow = FakeFlow(OTHER_URL)
        flow.send(b"\x01x")
        dump.websocket_message(flow)
        assert dump.frames == 1

    def test_text_frames_skipped(self, dump: WebSocketDump) -> None:
        """雀魂協定走 binary frame;文字訊息不是我們要的東西。"""
        flow = FakeFlow(GAME_URL)
        flow.send(b"ping", is_text=True)
        dump.websocket_message(flow)
        assert dump.frames == 0


class TestOutputFormat:
    def test_payload_roundtrips_through_base64(self, dump: WebSocketDump) -> None:
        payload = bytes(range(256))
        flow = FakeFlow(GAME_URL)
        flow.send(payload)
        dump.websocket_message(flow)

        record = _lines(dump)[0]
        assert base64.b64decode(record["b64"]) == payload
        assert record["len"] == 256

    def test_direction_recorded(self, dump: WebSocketDump) -> None:
        flow = FakeFlow(GAME_URL)
        flow.send(b"\x02up", from_client=True)
        dump.websocket_message(flow)
        flow.send(b"\x03down", from_client=False)
        dump.websocket_message(flow)

        assert [r["dir"] for r in _lines(dump)] == ["c2s", "s2c"]

    def test_timestamps_are_monotonic_and_relative(self, dump: WebSocketDump) -> None:
        flow = FakeFlow(GAME_URL)
        for i in range(3):
            flow.send(bytes([1, i]))
            dump.websocket_message(flow)

        times = [r["t"] for r in _lines(dump)]
        assert times[0] >= 0.0
        assert times == sorted(times)

    def test_flushed_immediately(self, dump: WebSocketDump) -> None:
        """錄製通常是被 Ctrl-C 中斷的,不能等到關檔才落地。"""
        flow = FakeFlow(GAME_URL)
        flow.send(b"\x01x")
        dump.websocket_message(flow)
        assert len(_lines(dump)) == 1, "尚未 close 就該讀得到"


class TestParseIntegration:
    def test_dump_is_readable_by_the_offline_parser(self, dump: WebSocketDump) -> None:
        """addon 寫出的格式必須能被 groundtruth.dump 直接讀回去。"""
        from majsoul_copilot.groundtruth.dump import iter_frames

        flow = FakeFlow(GAME_URL)
        flow.send(b"\x01hello", from_client=False)
        dump.websocket_message(flow)
        dump.done()

        frames = list(iter_frames(dump.path))
        assert len(frames) == 1
        assert frames[0].payload == b"\x01hello"
        assert frames[0].from_client is False
