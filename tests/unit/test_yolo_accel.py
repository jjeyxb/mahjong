"""YOLO 的 GPU 加速:從顯示卡名稱決定裝哪一種 torch、讀 probe.py 的回報、取消安裝。

名稱都是 Windows ``Win32_VideoController.Name`` 實際會出現的樣子。
"""

from __future__ import annotations

import dataclasses
import sys
import threading

import pytest

from mia.yolo import accel
from mia.yolo.accel import EnvStatus, parse_probe, plan_for

#: 這台開發機實際讀到的清單:虛擬螢幕驅動混在裡面。
DEV_BOX = ["Virtual Desktop Monitor", "AMD Radeon RX 9070", "MuMuPlayer Virtual Display Adapter"]


class TestPlan:
    def test_the_dev_box_gets_rocm(self) -> None:
        plan = plan_for(DEV_BOX, system="win32")
        assert (plan.backend, plan.gpu) == ("rocm", "AMD Radeon RX 9070")
        assert plan.download_mb > 1000

    @pytest.mark.parametrize(
        ("names", "backend"),
        [
            (["NVIDIA GeForce RTX 4070"], "cuda"),
            (["NVIDIA GeForce RTX 5090"], "cuda"),
            (["AMD Radeon RX 7900 XTX"], "rocm"),
            (["AMD Radeon RX 7800 XT"], "rocm"),
            (["AMD Radeon RX 9060 XT"], "rocm"),
            (["AMD Radeon PRO W7900"], "rocm"),
            (["Intel(R) Arc(TM) A770 Graphics"], "xpu"),
            (["Intel(R) Arc(TM) Graphics"], "xpu"),  # Core Ultra 內建
        ],
    )
    def test_supported_cards(self, names, backend) -> None:
        assert plan_for(names, system="win32").backend == backend

    def test_a_laptop_prefers_the_nvidia_card_over_the_integrated_one(self) -> None:
        names = ["Intel(R) Arc(TM) Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"]
        assert plan_for(names, system="win32").backend == "cuda"

    def test_an_unsupported_amd_card_is_not_downloaded_for(self) -> None:
        """ROCm 只支援特定幾代。看到 AMD 就裝的話,RX 6600 會白下載 2 GB。"""
        plan = plan_for(["AMD Radeon RX 6600"], system="win32")
        assert plan.backend is None
        assert "RX 6600" in plan.reason

    def test_an_amd_apu_with_an_nvidia_card_uses_the_nvidia_card(self) -> None:
        names = ["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 3060"]
        assert plan_for(names, system="win32").backend == "cuda"

    def test_old_intel_graphics_are_not_supported(self) -> None:
        plan = plan_for(["Intel(R) Iris(R) Xe Graphics"], system="win32")
        assert plan.backend is None
        assert "Iris" in plan.reason

    def test_no_gpu(self) -> None:
        plan = plan_for(["Microsoft Basic Display Adapter"], system="win32")
        assert plan.backend is None
        assert plan.reason

    def test_apple_silicon_needs_no_download(self) -> None:
        plan = plan_for(["Apple Silicon"], system="darwin")
        assert plan.backend == "mps"
        assert plan.download_mb == 0

    def test_an_intel_mac_has_no_gpu_path(self) -> None:
        assert plan_for([], system="darwin").backend is None

    def test_linux_is_not_offered(self) -> None:
        assert plan_for(["NVIDIA GeForce RTX 4070"], system="linux").backend is None


class TestProbe:
    def test_a_working_rocm_build(self) -> None:
        line = (
            '{"torch": "2.9.1+rocm7.2.1", "build": "rocm", "device": "cuda",'
            ' "name": "AMD Radeon RX 9070", "ok": true}'
        )
        status = parse_probe(line)
        assert status.gpu_ready
        assert (status.build, status.device, status.name) == ("rocm", "cuda", "AMD Radeon RX 9070")

    def test_only_the_last_json_line_counts(self) -> None:
        """torch 載入時可能先印警告。"""
        output = 'UserWarning: something\n{"build": "cpu", "device": null, "ok": false}\n'
        status = parse_probe(output)
        assert status.installed
        assert not status.gpu_ready
        assert status.build == "cpu"

    def test_a_gpu_build_that_cannot_compute(self) -> None:
        status = parse_probe(
            '{"build": "cuda", "device": "cuda", "ok": false,'
            ' "error": "CUDA error: no kernel image is available"}'
        )
        assert not status.gpu_ready
        assert "no kernel image" in status.error

    def test_garbage_is_reported_not_raised(self) -> None:
        status = parse_probe("Traceback (most recent call last):\n  ...")
        assert status.installed
        assert not status.gpu_ready
        assert status.error

    def test_no_environment(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MIA_YOLO_PYTHON", str(tmp_path / "missing.exe"))
        assert accel.probe() == EnvStatus(installed=False)


class TestInstall:
    """跑一個假的 setup_yolo:真的那支會下載好幾 GB。"""

    @pytest.fixture
    def fake_setup(self, monkeypatch, tmp_path):
        def use(script: str) -> None:
            path = tmp_path / "fake_setup.py"
            path.write_text(script, encoding="utf-8")
            monkeypatch.setattr(accel, "PROJECT_ROOT", tmp_path)
            (tmp_path / "tools").mkdir(exist_ok=True)
            path.replace(tmp_path / "tools" / "setup_yolo.py")

        return use

    def test_lines_are_passed_on_and_success_is_the_exit_code(self, fake_setup) -> None:
        fake_setup("import sys\nprint('== 1/2 下載')\nprint('YOLO 環境已就緒')\nsys.exit(0)\n")
        lines: list[str] = []
        ok, last = accel.install("cuda", lines.append, threading.Event())
        assert ok
        assert lines == ["== 1/2 下載", "YOLO 環境已就緒"]
        assert last == "YOLO 環境已就緒"

    def test_a_failure_returns_the_last_line(self, fake_setup) -> None:
        fake_setup("import sys\nprint('xpu 版 torch 裝好了,但在顯示卡上算不動')\nsys.exit(1)\n")
        ok, last = accel.install("xpu", lambda _line: None, threading.Event())
        assert not ok
        assert "算不動" in last

    def test_cancelling_while_downloading_stops_it(self, fake_setup) -> None:
        fake_setup("import time\nprint('Downloading torch', flush=True)\ntime.sleep(60)\n")
        cancelled = threading.Event()

        def on_line(_line: str) -> None:
            cancelled.set()

        ok, last = accel.install("cuda", on_line, cancelled)
        assert not ok
        assert "取消" in last

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows 才有子程序留著繼續跑的問題")
    def test_cancelling_also_stops_the_child_pip(self, fake_setup, tmp_path) -> None:
        """setup_yolo.py 另外開 pip。只停它的話 pip 會繼續下載、繼續裝。"""
        marker = tmp_path / "child-still-running.txt"
        fake_setup(
            "import subprocess, sys, time\n"
            "child = (\n"
            "    'import time, pathlib\\n'\n"
            "    'time.sleep(3)\\n'\n"
            f"    'pathlib.Path(r\"{marker}\").write_text(\"x\")\\n'\n"
            ")\n"
            "subprocess.Popen([sys.executable, '-c', child])\n"
            "print('Downloading torch', flush=True)\n"
            "time.sleep(60)\n"
        )
        cancelled = threading.Event()
        accel.install("cuda", lambda _line: cancelled.set(), cancelled)
        threading.Event().wait(5)
        assert not marker.exists()

    def test_cancelling_after_pip_started_installing_lets_it_finish(self, fake_setup) -> None:
        """裝到一半的 torch 是壞的環境 —— 開始裝之後取消也不停。"""
        fake_setup(
            "import time\n"
            f"print('{accel.INSTALL_PHASE}: torch', flush=True)\n"
            "time.sleep(1)\n"
            "print('YOLO 環境已就緒', flush=True)\n"
        )
        cancelled = threading.Event()
        ok, last = accel.install("cuda", lambda _line: cancelled.set(), cancelled)
        assert ok
        assert last == "YOLO 環境已就緒"


class TestDefaultDevice:
    """使用者還沒選過時 YOLO 跑在哪。"""

    def test_apple_silicon_defaults_to_the_gpu(self) -> None:
        """以前 Mac 一律自己用 MPS;預設 CPU 會讓 Mac 使用者更新之後不知不覺變慢。"""
        from mia.features import default_yolo_device

        assert default_yolo_device("darwin", "arm64") == "gpu"
        assert default_yolo_device("darwin", "x86_64") == "cpu"
        assert default_yolo_device("win32", "AMD64") == "cpu"

    def test_an_unchosen_or_unknown_value_uses_the_default(self, monkeypatch) -> None:
        from mia import features

        monkeypatch.setattr(features, "default_yolo_device", lambda: "gpu")
        assert features.YoloDeviceChoice(None).value == "gpu"
        assert features.YoloDeviceChoice("tpu").value == "gpu"
        assert features.YoloDeviceChoice("cpu").value == "cpu"

    def test_not_chosen_is_remembered_as_not_chosen(self, tmp_path) -> None:
        from mia.ui.state import UiState

        path = tmp_path / "ui.json"
        UiState().save(path)
        assert UiState.load(path).yolo_device is None
        UiState(yolo_device="gpu").save(path)
        assert UiState.load(path).yolo_device == "gpu"


class TestPackagedProbe:
    """打包版沒有 engines/yolo/.venv,要問打包好的推論程序。"""

    @pytest.fixture
    def packaged(self, monkeypatch):
        from mia.yolo import client as client_module

        made: list[object] = []

        class FakeClient:
            device_reply = "mps"

            def __init__(self, *, device=None, timeout=45.0):  # noqa: ARG002 —— 照真的簽名
                self.asked = device
                self.closed = False
                made.append(self)

            def start(self):
                self.device = FakeClient.device_reply

            def close(self):
                self.closed = True

        monkeypatch.setattr(accel, "LAYOUT", dataclasses.replace(accel.LAYOUT, packaged=True))
        monkeypatch.setattr(client_module, "YoloClient", FakeClient)
        return FakeClient, made

    def test_apple_silicon_is_ready_without_a_download(self, packaged) -> None:
        _fake, made = packaged
        status = accel.probe()
        assert status.gpu_ready
        assert status.device == "mps"
        assert made[0].asked == "gpu"
        assert made[0].closed

    def test_a_cpu_only_worker_is_not_ready(self, packaged) -> None:
        fake, _ = packaged
        fake.device_reply = "cpu"
        assert not accel.probe().gpu_ready

    def test_a_worker_that_does_not_start_is_reported(self, packaged, monkeypatch) -> None:
        from mia.engine.base import EngineError

        fake, _ = packaged

        def broken(_self):
            raise EngineError("YOLO 程序已結束")

        monkeypatch.setattr(fake, "start", broken)
        status = accel.probe()
        assert not status.gpu_ready
        assert "已結束" in status.error
