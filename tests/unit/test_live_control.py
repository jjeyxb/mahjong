"""主程式 → 擷取子程序的控制通道。

存在的理由:只有那個子程序握著瀏覽器,而它的參數是**啟動時**用命令列傳的。
使用者在 UI 上換了畫布尺寸之後,瀏覽器要當場跟著變,就只能靠這條路。
"""

from __future__ import annotations

import json

from mia.live.control import ControlFile


class TestRoundTrip:
    def test_a_written_command_is_read_back(self, tmp_path) -> None:
        path = tmp_path / "control.json"
        ControlFile(path).write(canvas="1920x1080")
        assert ControlFile(path).poll() == {"seq": 0, "canvas": "1920x1080"}

    def test_nothing_written_yet_reads_as_nothing(self, tmp_path) -> None:
        assert ControlFile(tmp_path / "nope.json").poll() is None

    def test_the_same_command_is_only_delivered_once(self, tmp_path) -> None:
        """讀端每 250 ms 輪詢一次,重複套用同一個指令等於一直去調視窗大小。"""
        path = tmp_path / "control.json"
        ControlFile(path).write(canvas="1280x720")
        reader = ControlFile(path)
        assert reader.poll() is not None
        assert reader.poll() is None

    def test_a_newer_command_is_delivered(self, tmp_path) -> None:
        path = tmp_path / "control.json"
        writer, reader = ControlFile(path), ControlFile(path)
        writer.write(canvas="1280x720")
        reader.poll()
        writer.write(canvas="1920x1080")
        command = reader.poll()
        assert command is not None
        assert command["canvas"] == "1920x1080"

    def test_sequence_not_mtime_decides_freshness(self, tmp_path) -> None:
        """檔案系統的時間解析度在某些平台上只到秒 —— 同一秒內的兩次變更
        用 mtime 判斷會被當成沒變,而使用者連按兩次選單是很正常的事。"""
        path = tmp_path / "control.json"
        writer, reader = ControlFile(path), ControlFile(path)
        writer.write(canvas="1280x720")
        reader.poll()
        writer.write(canvas="1600x900")  # 同一秒內
        assert reader.poll() is not None


class TestRobustness:
    """這條通道壞掉的後果只是視窗沒跟著調整,不該讓正在錄的那一場掛掉。"""

    def test_broken_json_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "control.json"
        path.write_text("{ 不是 JSON", encoding="utf-8")
        assert ControlFile(path).poll() is None

    def test_a_json_list_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "control.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert ControlFile(path).poll() is None

    def test_a_missing_sequence_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "control.json"
        path.write_text(json.dumps({"canvas": "1280x720"}), encoding="utf-8")
        assert ControlFile(path).poll() is None

    def test_writing_into_an_unwritable_place_does_not_raise(self, tmp_path) -> None:
        blocked = tmp_path / "file"
        blocked.write_text("", encoding="utf-8")
        ControlFile(blocked / "sub" / "control.json").write(canvas="1280x720")

    def test_no_temporary_file_survives(self, tmp_path) -> None:
        """先寫暫存檔再 rename —— 讀端可能剛好讀到寫了一半的 JSON。"""
        path = tmp_path / "control.json"
        ControlFile(path).write(canvas="1280x720")
        assert [p.name for p in tmp_path.iterdir()] == ["control.json"]
