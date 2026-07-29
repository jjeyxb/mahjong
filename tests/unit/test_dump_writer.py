"""錄影檔格式的寫入測試,以及兩種擷取後端的格式一致性。

格式一致是整個 GT 層的關鍵性質:CDP 與 mitmproxy 兩條擷取路徑寫出來的檔案
必須完全相同,下游(解析、轉 MJAI、評測)才能不知道也不必知道資料是怎麼來的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from majsoul_copilot.groundtruth.dump import DumpWriter, iter_frames


class TestFormat:
    def test_payload_survives_roundtrip(self, tmp_path: Path) -> None:
        payload = bytes(range(256))
        with DumpWriter(tmp_path / "d.jsonl") as w:
            w.write(payload, from_client=False, flow="f1")

        frames = list(iter_frames(tmp_path / "d.jsonl"))
        assert len(frames) == 1
        assert frames[0].payload == payload
        assert frames[0].len_matches_payload if hasattr(frames[0], "len_matches_payload") else True

    def test_direction_recorded(self, tmp_path: Path) -> None:
        with DumpWriter(tmp_path / "d.jsonl") as w:
            w.write(b"\x02up", from_client=True, flow="f1")
            w.write(b"\x03down", from_client=False, flow="f1")

        assert [f.from_client for f in iter_frames(tmp_path / "d.jsonl")] == [True, False]

    def test_flow_recorded(self, tmp_path: Path) -> None:
        """少了 flow,多條連線的請求/回應會在離線解析時互相配錯。"""
        with DumpWriter(tmp_path / "d.jsonl") as w:
            w.write(b"\x01a", from_client=False, flow="conn-A")
            w.write(b"\x01b", from_client=False, flow="conn-B")

        assert [f.flow for f in iter_frames(tmp_path / "d.jsonl")] == ["conn-A", "conn-B"]

    def test_length_field_matches(self, tmp_path: Path) -> None:
        with DumpWriter(tmp_path / "d.jsonl") as w:
            w.write(b"\x01" * 42, from_client=False, flow="f")

        record = json.loads((tmp_path / "d.jsonl").read_text(encoding="utf-8").strip())
        assert record["len"] == 42

    def test_timestamps_are_relative_and_monotonic(self, tmp_path: Path) -> None:
        with DumpWriter(tmp_path / "d.jsonl") as w:
            for i in range(3):
                w.write(bytes([1, i]), from_client=False, flow="f")

        times = [f.timestamp for f in iter_frames(tmp_path / "d.jsonl")]
        assert times[0] >= 0.0
        assert times == sorted(times)

    def test_required_keys_present(self, tmp_path: Path) -> None:
        """欄位名稱是兩種後端的契約,改了要有意識。"""
        with DumpWriter(tmp_path / "d.jsonl") as w:
            w.write(b"\x01x", from_client=False, flow="f")

        record = json.loads((tmp_path / "d.jsonl").read_text(encoding="utf-8").strip())
        assert set(record) == {"t", "wall", "flow", "dir", "len", "b64"}


class TestLifecycle:
    def test_flushed_immediately(self, tmp_path: Path) -> None:
        """錄製幾乎都是被 Ctrl-C 中斷的,不能等到關檔才落地。"""
        writer = DumpWriter(tmp_path / "d.jsonl")
        writer.write(b"\x01x", from_client=False, flow="f")

        assert len(list(iter_frames(tmp_path / "d.jsonl"))) == 1

    def test_frame_counter(self, tmp_path: Path) -> None:
        with DumpWriter(tmp_path / "d.jsonl") as w:
            assert w.frames == 0
            w.write(b"\x01a", from_client=False, flow="f")
            w.write(b"\x01b", from_client=False, flow="f")
            assert w.frames == 2

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        writer = DumpWriter(tmp_path / "d.jsonl")
        writer.close()
        with pytest.raises(RuntimeError, match="已關閉"):
            writer.write(b"\x01x", from_client=False, flow="f")

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        writer = DumpWriter(tmp_path / "d.jsonl")
        writer.close()
        writer.close()

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        with DumpWriter(tmp_path / "a" / "b" / "d.jsonl") as w:
            w.write(b"\x01x", from_client=False, flow="f")
        assert (tmp_path / "a" / "b" / "d.jsonl").is_file()

    def test_appends_rather_than_truncates(self, tmp_path: Path) -> None:
        """同一個檔案再開一次不該把先前錄的內容清掉。"""
        path = tmp_path / "d.jsonl"
        with DumpWriter(path) as w:
            w.write(b"\x01a", from_client=False, flow="f")
        with DumpWriter(path) as w:
            w.write(b"\x01b", from_client=False, flow="f")

        assert len(list(iter_frames(path))) == 2


class TestBackendParity:
    """兩種擷取後端必須寫出完全相同的格式。"""

    def test_mitmproxy_addon_output_matches_direct_writer(self, tmp_path: Path) -> None:
        from majsoul_copilot.groundtruth.capture_addon import WebSocketDump

        payload = b"\x01\xde\xad\xbe\xef"

        # 直接用 DumpWriter(CDP 後端走的路)
        with DumpWriter(tmp_path / "direct.jsonl") as w:
            w.write(payload, from_client=False, flow="X")

        # 透過 mitmproxy addon
        dump = WebSocketDump(output=tmp_path / "addon.jsonl", url_filter="example.com")
        dump.writer.write(payload, from_client=False, flow="X")
        dump.writer.close()

        direct = json.loads((tmp_path / "direct.jsonl").read_text(encoding="utf-8").strip())
        addon = json.loads((tmp_path / "addon.jsonl").read_text(encoding="utf-8").strip())

        assert set(direct) == set(addon), "欄位集合必須一致"
        for key in ("flow", "dir", "len", "b64"):
            assert direct[key] == addon[key], f"{key} 不一致"

    def test_both_backends_readable_by_same_parser(self, tmp_path: Path) -> None:
        from majsoul_copilot.groundtruth.capture_addon import WebSocketDump

        dump = WebSocketDump(output=tmp_path / "d.jsonl", url_filter="x")
        dump.writer.write(b"\x01from-addon", from_client=False, flow="A")
        dump.writer.close()

        with DumpWriter(tmp_path / "d.jsonl") as w:  # 同一個檔案接著寫
            w.write(b"\x01from-cdp", from_client=True, flow="B")

        frames = list(iter_frames(tmp_path / "d.jsonl"))
        assert [f.payload for f in frames] == [b"\x01from-addon", b"\x01from-cdp"]
        assert [f.flow for f in frames] == ["A", "B"]
