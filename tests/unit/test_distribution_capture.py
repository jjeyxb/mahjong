from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path
from types import ModuleType

from mia.distribution.layout import Layout
from mia.live import source


def test_packaged_capture_keeps_traceback_without_a_terminal(tmp_path, monkeypatch):
    layout = Layout(tmp_path / "resources", tmp_path, True)
    monkeypatch.setattr(source, "LAYOUT", layout)
    process = source.CaptureProcess(
        [sys.executable, "-c", "raise RuntimeError('capture-startup-evidence')"]
    )
    process.start()
    deadline = time.monotonic() + 5
    while process.status.alive and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    assert not process.status.alive
    assert process.log_path is not None
    assert "capture-startup-evidence" in process.log_path.read_text()
    assert str(process.log_path) in process.status.read()


def test_frozen_yolo_first_run_keeps_import_messages_off_protocol(
    tmp_path, monkeypatch, capsys
):
    """First-run Ultralytics announcements must never precede the JSON handshake."""
    import multiprocessing

    settings = tmp_path / "userdata/yolo"
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources/release.json").write_text('{"platform":"macos-arm64"}')
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "argv", ["MIA-yolo"])
    monkeypatch.setattr(multiprocessing, "freeze_support", lambda: None)
    for key, value in {
        "YOLO_CONFIG_DIR": str(settings),
        "YOLO_AUTOINSTALL": "false",
        "MIA_RESOURCE_ROOT": str(tmp_path / "MIA.app/Contents/Resources/resources"),
        "MIA_YOLO_MODELS": str(tmp_path / "models"),
        "MJ_DEVICE": "cpu",
    }.items():
        monkeypatch.setenv(key, value)

    worker = ModuleType("worker")
    vars(worker)["main"] = lambda: print('{"ready":true,"protocol":1}')
    compat = ModuleType("platform_compat")
    vars(compat)["pick_device"] = lambda: "cpu"

    class Model:
        def predict(self):
            pass

    def imported(name):
        if name == "Model":
            assert settings.is_dir()
            print("Creating new Ultralytics Settings")
            return Model
        raise AttributeError(name)

    model = ModuleType("ultralytics.engine.model")
    vars(model)["__getattr__"] = imported
    for name, module in {
        "worker": worker,
        "platform_compat": compat,
        "ultralytics.engine.model": model,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    entry = Path(__file__).resolve().parents[2] / "packaging/entrypoints/yolo.py"
    runpy.run_path(str(entry))
    output = capsys.readouterr()
    assert json.loads(output.out) == {"ready": True, "protocol": 1}
    assert "Creating new Ultralytics Settings" in output.err


def test_the_released_capture_entry_only_allows_browser_capture() -> None:
    """自動打牌、開電腦場與 MITM 都不隨發布版提供。"""
    import pytest

    from mia.distribution.capture import CaptureRefused, checked_capture_args

    normal = [
        "cdp", "--out", "/x/ws.jsonl", "--url", "https://game.maj-soul.com/1/",
        "--user-data-dir", "/x/profile", "--canvas", "1280x720", "--control", "/x/c.json",
    ]
    assert checked_capture_args(normal) == normal
    assert checked_capture_args(["--log-level", "INFO", "cdp", "--headless"])
    for refused in (
        ["proxy", "--out", "/x"],
        ["local", "--out", "/x"],
        ["cdp", "--autoplay", "mortal"],
        ["cdp", "--autoplay=mortal"],
        ["cdp", "--cpu-room"],
        ["cdp", "--mortal", "w.pth"],
        ["cdp", "--overtime"],
        ["inspect", "ws.jsonl"],
        ["cdp", "extra"],
        ["--autoplay", "mortal", "cdp"],
        ["cdp", "--out"],
        [],
    ):
        with pytest.raises(CaptureRefused):
            checked_capture_args(refused)
