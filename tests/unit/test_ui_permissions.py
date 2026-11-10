from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QMainWindow

from mia import features
from mia.live.bus import UpdateBus, WorkerStatus
from mia.live.runtime import Feature, LiveRuntime
from mia.ui import permissions
from mia.ui.viewmodel import ViewModel


def worker(source, started):
    status = WorkerStatus("test")

    def start():
        started.append(True)
        status.alive = True

    def stop():
        status.alive = False

    return SimpleNamespace(
        SOURCE=source, status=status, start=start, stop=stop,
        join=lambda _timeout: None, is_alive=lambda: status.alive,
    )


@pytest.mark.parametrize(
    ("key", "source", "requires_permission"),
    [
        (features.VISION, "", True),
        (features.ADVICE, "圖像辨識", True),
        (features.DANGER, "圖像辨識", True),
        (features.ADVICE, "封包", False),
        (features.DANGER, "封包", False),
    ],
)
def test_only_screen_workers_are_blocked_until_authorized(key, source, requires_permission):
    started, checked = [], []
    feature = Feature(key, lambda: worker(source, started))
    model = ViewModel()
    runtime = LiveRuntime(model, UpdateBus(), features=[feature])

    def deny():
        checked.append(True)
        return False

    runtime.set_screen_capture_permission_check(deny)
    runtime.set_enabled(key, True)
    assert runtime.is_enabled(key) is not requires_permission
    assert bool(started) is not requires_permission
    assert bool(checked) is requires_permission
    if requires_permission:
        assert any("缺少螢幕錄製權限" in s for s in model.state.notices)
        runtime.set_screen_capture_permission_check(lambda: True)
        runtime.set_enabled(key, True)
        assert runtime.is_enabled(key)
        assert started == [True]
    runtime.set_enabled(key, False)


def test_permission_request_and_dialog_follow_actual_feature_toggle(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    granted, requests, started, urls = [False], [], [], []
    monkeypatch.setattr(permissions, "_has_permission", lambda: granted[0])
    monkeypatch.setattr(permissions, "_request_permission", lambda: requests.append(True) or False)
    monkeypatch.setattr(
        permissions.QDesktopServices, "openUrl", lambda url: urls.append(url.toString()) or True
    )
    prompt = permissions.ScreenCapturePermissionPrompt(window)
    feature = Feature(features.VISION, lambda: worker("", started))
    runtime = LiveRuntime(ViewModel(), UpdateBus(), features=[feature])
    runtime.set_screen_capture_permission_check(prompt.check)
    runtime.set_enabled(features.VISION, True)
    app.processEvents()
    dialog = prompt.dialog
    assert dialog is not None and dialog.isVisible()
    assert not runtime.is_enabled(features.VISION) and not started
    assert requests == [True]
    assert not prompt.check()
    assert prompt.dialog is dialog and requests == [True]
    settings = next(button for button in dialog.buttons() if button.text() == "開啟系統設定")
    settings.click()
    app.processEvents()
    assert urls == [permissions.SETTINGS_URL]
    assert prompt.dialog is None
    granted[0] = True
    runtime.set_enabled(features.VISION, True)
    assert runtime.is_enabled(features.VISION) and started == [True]
    assert prompt.dialog is None and requests == [True]
    runtime.set_enabled(features.VISION, False)
    window.close()


def test_os_grants_immediately_without_an_extra_reminder(monkeypatch):
    _app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    monkeypatch.setattr(permissions, "_has_permission", lambda: False)
    monkeypatch.setattr(permissions, "_request_permission", lambda: True)
    prompt = permissions.ScreenCapturePermissionPrompt(window)
    assert prompt.check()
    assert prompt.dialog is None
    window.close()


def test_other_platforms_do_not_install_a_screen_permission_guard(monkeypatch):
    _app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    monkeypatch.setattr(permissions.sys, "platform", "win32")
    runtime = SimpleNamespace(
        set_screen_capture_permission_check=lambda _check: pytest.fail("Mac only")
    )
    assert permissions.install_permission_prompt(window, runtime) is None
    window.close()
