"""一個指令同時錄畫面與封包。

這裡只測 :class:`GameWatcher` —— 「什麼時候該停」是這支工具唯一的判斷,
其他部分是把兩個既有工具接起來。

為什麼值得測:上次把素材錄壞就是停的時機錯了(畫面 848 秒、封包 1433 秒,
那一局 7 次立直只涵蓋到一家)。而那件事**幾個月後**才被發現。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def capture():
    """用檔案路徑載入 —— tools/ 不是套件。"""
    spec = importlib.util.spec_from_file_location("_capture", ROOT / "tools" / "capture.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_capture"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    return tmp_path / "ws.jsonl"


def _append(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)


class TestGameWatcher:
    def test_a_missing_file_is_not_an_ending(self, capture, dump) -> None:
        """封包檔還沒出現 ≠ 對局結束。

        瀏覽器要開、使用者要登入,那段時間檔案根本不存在。把它當成結束的話,
        錄影會在按下指令的瞬間就停掉。
        """
        watcher = capture.GameWatcher(dump)
        assert watcher.poll() is False

    def test_an_empty_file_is_not_an_ending(self, capture, dump) -> None:
        dump.write_text("", encoding="utf-8")
        assert capture.GameWatcher(dump).poll() is False

    def test_it_only_reads_the_new_lines(self, capture, dump) -> None:
        """每次重讀整個檔會隨著對局變長越來越慢。

        而它跑在錄影的同一條路上 —— 那不只是慢,是會變成掉幀的原因。
        所以這裡釘的是「讀過的不再讀」。
        """
        real = (FIXTURES / "real_game_full.jsonl").read_text(encoding="utf-8").splitlines(True)
        _append(dump, real[:20])
        watcher = capture.GameWatcher(dump)
        watcher.poll()
        assert watcher._seen == 20  # noqa: SLF001 - 就是要確認它記住了進度

        _append(dump, real[20:40])
        watcher.poll()
        assert watcher._seen == 40  # noqa: SLF001

    def test_a_real_game_ends(self, capture, dump) -> None:
        """整場餵完要說「結束了」,而且數得出打了幾局。"""
        dump.write_bytes((FIXTURES / "real_game_full.jsonl").read_bytes())
        watcher = capture.GameWatcher(dump)
        assert watcher.poll() is True
        assert watcher.kyoku > 0

    def test_it_does_not_end_early(self, capture, dump) -> None:
        """**這是整支工具的重點。** 對局還在打的時候不可以說結束。

        錄影停早了的症狀不是「錯誤」,是一份看起來正常、但後半段不見了的
        素材 —— 而那要等到有人去量的時候才會發現。
        """
        real = (FIXTURES / "real_game_full.jsonl").read_text(encoding="utf-8").splitlines(True)
        watcher = capture.GameWatcher(dump)
        half = len(real) // 2
        _append(dump, real[:half])
        assert watcher.poll() is False, "打到一半不算結束"
        assert watcher.kyoku > 0, "前提:這半場真的有打到局"

    def test_once_ended_it_stays_ended(self, capture, dump) -> None:
        """說過結束就不會改口 —— 停止判斷不可以抖。"""
        dump.write_bytes((FIXTURES / "real_game_full.jsonl").read_bytes())
        watcher = capture.GameWatcher(dump)
        assert watcher.poll() is True
        dump.unlink()
        assert watcher.poll() is True

    def test_a_broken_line_does_not_end_the_recording(self, capture, dump) -> None:
        """壞掉的一行要略過,不能當成結束。

        封包錄影是**邊寫邊讀**的,所以一定會讀到寫到一半的那一行。
        把它當成結束,錄影會在隨機的時間點停掉。
        """
        real = (FIXTURES / "real_game_full.jsonl").read_text(encoding="utf-8").splitlines(True)
        _append(dump, [*real[:10], json.dumps({"不是": "frame"}) + "\n", "{壞掉的 json"])
        assert capture.GameWatcher(dump).poll() is False
