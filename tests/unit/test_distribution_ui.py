from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication, QMainWindow

from mia.distribution import ui
from mia.distribution.release import Release


def settle(app, dialog):
    deadline = time.monotonic() + 3
    while dialog.task is not None and dialog.task.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert dialog.task is None or not dialog.task.isRunning()


def test_check_download_then_wait_for_install_confirmation(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    monkeypatch.setattr(ui, "configuration", lambda: {"version": "0.1.0"})
    release = Release("0.2.0", "macos-arm64", "app", "https://example.org/x", 32, "a" * 64, 100)
    monkeypatch.setattr(ui, "check", lambda *_args, **_kw: (release, b"signed"))
    plan = tmp_path / "plan.json"
    monkeypatch.setattr(ui, "prepare_update", lambda *_args, **_kw: plan)
    monkeypatch.setattr(
        ui,
        "launch_installer",
        lambda *_args: (_ for _ in ()).throw(AssertionError("no confirmation")),
    )
    dialog = ui.UpdateDialog(window)
    dialog.act()
    settle(app, dialog)
    assert dialog.envelope == b"signed"
    dialog.act()
    settle(app, dialog)
    assert dialog.plan == plan
    assert dialog.button.text() == "安裝並重新啟動"
    dialog.reject()
    window.close()


def test_missing_release_configuration_is_shown_without_crash(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    monkeypatch.setattr(
        ui, "configuration", lambda: {"version": "0.1.0", "public_key": "", "feed_url": ""}
    )
    dialog = ui.UpdateDialog(window)
    dialog.act()
    settle(app, dialog)
    assert "尚未設定" in dialog.label.text()
    dialog.reject()
    window.close()
