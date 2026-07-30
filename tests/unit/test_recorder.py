"""錄製 session 的寫入 / 讀取 / 去重測試。"""

from __future__ import annotations

import base64
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from mia.capture.base import Frame, WindowInfo
from mia.groundtruth.dump import DumpStats, iter_frames, parse_dump
from mia.recorder import SessionReader, SessionWriter
from mia.utils.geometry import Rect


def _frame(window: WindowInfo, value: int, t: float, *, size: tuple[int, int] = (120, 80)) -> Frame:
    """造一張純色畫面。純色是刻意的:去重判斷只看縮圖差異,純色最好控制。

    ``size`` 是 (width, height),與專案其他地方的 :class:`Size` 一致。
    """
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(image=image, window=window, scale=2.0, captured_at=t)


class TestChangeDetection:
    def test_identical_frames_stored_once(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s") as writer:
            for i in range(5):
                writer.add_frame(_frame(window, 100, i * 0.1))

        assert writer.manifest.frame_count == 5
        assert writer.manifest.stored_frames == 1, "靜止畫面只該寫一張"
        assert len(list((tmp_path / "s" / "frames").iterdir())) == 1

    def test_changed_frames_each_stored(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s") as writer:
            for i, value in enumerate((10, 90, 170, 250)):
                writer.add_frame(_frame(window, value, i * 0.1))
        assert writer.manifest.stored_frames == 4

    def test_unchanged_records_point_at_previous_image(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """沒變化的幀仍要留在索引裡(時間軸完整),只是共用同一個檔名。"""
        with SessionWriter(tmp_path, session_id="s") as writer:
            first = writer.add_frame(_frame(window, 50, 0.0))
            second = writer.add_frame(_frame(window, 50, 0.1))
            third = writer.add_frame(_frame(window, 200, 0.2))

        assert (first.changed, second.changed, third.changed) == (True, False, True)
        assert second.image == first.image
        assert third.image != first.image

    def test_threshold_zero_stores_everything(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
            for i in range(3):
                writer.add_frame(_frame(window, 100, i * 0.1))
        assert writer.manifest.stored_frames == 3


class TestManifest:
    def test_metadata_captured_from_first_frame(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s", backend="macos", notes="測試") as writer:
            writer.add_frame(_frame(window, 100, 5.0), table_rect=Rect(0, 56, 120, 24))

        manifest = writer.manifest
        assert manifest.backend == "macos"
        assert manifest.notes == "測試"
        assert manifest.window_title == window.title
        assert manifest.image_size == [120, 80]
        assert manifest.table_rect == [0, 56, 120, 24]

    def test_table_rect_can_arrive_after_the_first_frame(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """StableCalibrator 要蒐集數幀才鎖定,前幾幀傳進來的是 None。

        把寫入 table_rect 綁在「第一幀」分支裡的話,manifest 會永遠沒有校正
        結果 —— 而且不會有任何錯誤,只是後續所有 ROI 都失去座標基準。
        """
        with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
            writer.add_frame(_frame(window, 100, 0.0))          # 還在蒐集
            writer.add_frame(_frame(window, 110, 0.1))          # 還在蒐集
            writer.add_frame(_frame(window, 120, 0.2), table_rect=Rect(0, 56, 120, 24))

        assert writer.manifest.table_rect == [0, 56, 120, 24]

    def test_table_rect_is_not_overwritten_once_set(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """校正一旦鎖定就不會變,後來的值不該覆寫掉已經定案的那個。"""
        with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
            writer.add_frame(_frame(window, 100, 0.0), table_rect=Rect(0, 56, 120, 24))
            writer.add_frame(_frame(window, 110, 0.1), table_rect=Rect(9, 9, 9, 9))

        assert writer.manifest.table_rect == [0, 56, 120, 24]

    def test_timestamps_are_relative_to_session_start(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """單調時鐘的絕對值沒有意義,存進去的必須是相對秒數。"""
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 10, 1000.0))
            last = writer.add_frame(_frame(window, 200, 1002.5))

        assert writer.manifest.frames[0].timestamp == 0.0
        assert last.timestamp == pytest.approx(2.5)
        assert writer.manifest.duration == pytest.approx(2.5)

    def test_manifest_written_on_close(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 100, 0.0))
        assert (tmp_path / "s" / "manifest.json").is_file()

    def test_writing_after_close_raises(self, window: WindowInfo, tmp_path: Path) -> None:
        writer = SessionWriter(tmp_path, session_id="s")
        writer.add_frame(_frame(window, 100, 0.0))
        writer.close()
        with pytest.raises(RuntimeError, match="已關閉"):
            writer.add_frame(_frame(window, 200, 0.1))


class TestWallClockAlignment:
    """畫面與 GT 是兩個獨立行程錄的,只有牆上時鐘能把它們對起來。

    畫面錄製器與 ``gt.py`` 各自以自己的 :func:`time.monotonic` 為原點,那兩個
    原點毫無關係。manifest 的 ``first_frame_wall`` 是唯一的橋樑 —— 少了它,
    整個成對資料集就只能靠人眼對齊。
    """

    def test_anchor_matches_real_wall_clock(self, window: WindowInfo, tmp_path: Path) -> None:
        before = time.time()
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 100, time.monotonic()))
        after = time.time()
        assert before <= writer.manifest.first_frame_wall <= after

    def test_anchor_is_the_first_frame_not_construction(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """錨點必須綁在第一幀,不是建構時刻。

        在 tools/record.py 目前的呼叫順序下兩者只差 21 ms(實測),但那是偶然
        —— 呼叫端只要把建構提前,落差就會無聲地長大。這個測試把「錨點跟著第
        一幀走」釘成契約,而不是依賴呼叫順序碰巧正確。
        """
        writer = SessionWriter(tmp_path, session_id="s")
        constructed = time.time()
        time.sleep(0.05)  # 模擬後端暖機與第一次擷取
        writer.add_frame(_frame(window, 100, time.monotonic()))
        writer.close()

        assert writer.manifest.first_frame_wall - constructed >= 0.05

    def test_wall_at_converts_relative_to_absolute(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        base = time.monotonic()
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 10, base))
            writer.add_frame(_frame(window, 200, base + 2.5))

        reader = SessionReader(tmp_path / "s")
        assert reader.can_align
        anchor = reader.manifest.first_frame_wall
        assert reader.wall_at(0.0) == pytest.approx(anchor)
        assert reader.wall_at(2.5) == pytest.approx(anchor + 2.5)

    def test_frame_wall_times_line_up_with_gt_wall_times(
        self, window: WindowInfo, tmp_path: Path
    ) -> None:
        """端到端:錄下的幀換算成牆上時鐘後,要落在同時段 GT 事件的秒級鄰域內。"""
        base = time.monotonic()
        with SessionWriter(tmp_path, session_id="s") as writer:
            for i in range(5):
                writer.add_frame(_frame(window, 10 * i, base + i * 0.1))
        gt_wall = time.time()  # 另一個行程在同一時刻寫下的 "wall" 欄位

        reader = SessionReader(tmp_path / "s")
        deltas = [abs(reader.wall_at(f.timestamp) - gt_wall) for f in reader.manifest.frames]
        assert min(deltas) < 1.0

    def test_legacy_session_reports_unalignable(self, window: WindowInfo, tmp_path: Path) -> None:
        """舊 session 缺欄位時要明確拒絕,而不是預設 0.0 悄悄算出 1970 年。"""
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 100, time.monotonic()))

        manifest_path = tmp_path / "s" / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        del data["first_frame_wall"]
        manifest_path.write_text(json.dumps(data), encoding="utf-8")

        reader = SessionReader(tmp_path / "s")
        assert not reader.can_align
        with pytest.raises(ValueError, match="無法自動對齊"):
            reader.wall_at(0.0)


class TestRoundTrip:
    @pytest.fixture
    def session(self, window: WindowInfo, tmp_path: Path) -> Path:
        with SessionWriter(tmp_path, session_id="s", image_format="png") as writer:
            for i, value in enumerate((10, 10, 200, 200, 90)):
                writer.add_frame(_frame(window, value, i * 0.5), table_rect=Rect(2, 4, 100, 60))
            writer.add_event({"timestamp": 0.7, "kind": "dahai", "tile": "5m"})
            writer.add_event({"timestamp": 1.4, "kind": "tsumo", "tile": "1p"})
        return tmp_path / "s"

    def test_frame_count(self, session: Path) -> None:
        assert len(SessionReader(session)) == 5

    def test_frames_reconstruct_as_capture_frames(self, session: Path) -> None:
        """回放出來的必須跟線上擷取同型別,視覺管線才能不分來源共用。"""
        frames = list(SessionReader(session).frames())
        assert len(frames) == 5
        assert all(isinstance(f, Frame) for f in frames)
        assert frames[0].size.width == 120
        assert frames[0].scale == 2.0
        assert frames[0].window.title  # WindowInfo 由 manifest 還原

    def test_pixels_survive_roundtrip_with_png(self, session: Path) -> None:
        first = next(SessionReader(session).frames())
        assert int(first.image[0, 0, 0]) == 10

    def test_changed_only_skips_duplicates(self, session: Path) -> None:
        reader = SessionReader(session)
        assert len(list(reader.frames())) == 5
        assert len(list(reader.frames(changed_only=True))) == 3

    def test_timestamps_preserved(self, session: Path) -> None:
        times = [f.captured_at for f in SessionReader(session).frames()]
        assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_table_rect_preserved(self, session: Path) -> None:
        assert SessionReader(session).table_rect == Rect(2, 4, 100, 60)

    def test_events_roundtrip(self, session: Path) -> None:
        events = list(SessionReader(session).events())
        assert [e["kind"] for e in events] == ["dahai", "tsumo"]

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest"):
            SessionReader(tmp_path)


class TestNoEventsFile:
    def test_events_empty_when_never_written(self, window: WindowInfo, tmp_path: Path) -> None:
        with SessionWriter(tmp_path, session_id="s") as writer:
            writer.add_frame(_frame(window, 100, 0.0))
        assert list(SessionReader(tmp_path / "s").events()) == []


class TestDumpReading:
    """WebSocket 原始錄影的讀取與解析。"""

    def _write_dump(self, path: Path, entries: list[tuple[float, str, bytes]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for t, direction, payload in entries:
                fh.write(
                    json.dumps(
                        {
                            "t": t,
                            "wall": 1700000000.0 + t,
                            "dir": direction,
                            "len": len(payload),
                            "b64": base64.b64encode(payload).decode(),
                        }
                    )
                    + "\n"
                )

    def test_iter_frames_decodes_payloads(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        self._write_dump(path, [(0.5, "s2c", b"\x01abc"), (1.0, "c2s", b"\x02xyz")])

        frames = list(iter_frames(path))

        assert [f.timestamp for f in frames] == [0.5, 1.0]
        assert [f.from_client for f in frames] == [False, True]
        assert frames[0].payload == b"\x01abc"

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """錄製常被 Ctrl-C 中斷,最後一行可能是半截的,不該讓整份檔案讀不了。"""
        path = tmp_path / "d.jsonl"
        self._write_dump(path, [(0.1, "s2c", b"\x01ok")])
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"t": 0.2, "dir": "s2c"')  # 截斷

        assert len(list(iter_frames(path))) == 1

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        self._write_dump(path, [(0.1, "s2c", b"\x01ok")])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n")
        assert len(list(iter_frames(path))) == 1

    def test_parse_failures_are_counted_not_raised(self, tmp_path: Path) -> None:
        from mia.groundtruth.schema import DEFAULT_LIQI_PATH

        if not DEFAULT_LIQI_PATH.is_file():
            pytest.skip("需要 assets/proto/liqi.json")

        path = tmp_path / "d.jsonl"
        # 兩則垃圾:一則類型位元組未知,一則是配不到請求的回應
        self._write_dump(
            path,
            [(0.1, "s2c", b"\x09bad"), (0.2, "s2c", b"\x03" + struct.pack("<H", 1))],
        )

        stats = DumpStats()
        results = list(parse_dump(path, stats=stats))

        assert results == []
        assert stats.total == 2
        assert stats.failed == 2
        assert stats.parsed == 0
        assert sum(stats.failures.values()) == 2
        assert "frame 2 個" in stats.summary()
