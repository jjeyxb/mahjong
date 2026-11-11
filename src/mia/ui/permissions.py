"""Request macOS screen access from the UI thread before starting recognition."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMainWindow, QMessageBox

from mia.distribution.layout import LAYOUT
from mia.i18n import tr
from mia.utils.logging import logger

if TYPE_CHECKING:
    from mia.live.runtime import LiveRuntime

SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"


def _has_permission() -> bool:
    from mia.capture.macos import MacOSCaptureBackend

    return MacOSCaptureBackend.has_permission()


def _request_permission() -> bool:
    from mia.capture.macos import MacOSCaptureBackend

    return MacOSCaptureBackend.request_permission()


class ScreenCapturePermissionPrompt(QObject):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.dialog: QMessageBox | None = None
        self._requested = False

    def check(self) -> bool:
        if _has_permission():
            if self.dialog is not None:
                self.dialog.close()
            return True
        if not self._requested:
            self._requested = True
            if _request_permission():
                return True
        if self.dialog is None:
            self._show()
        else:
            self.dialog.raise_()
            self.dialog.activateWindow()
        return False

    def _show(self) -> None:
        dialog = QMessageBox(self.window)
        dialog.setWindowTitle(tr("需要螢幕錄製權限"))
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(tr("請允許 MIA 擷取遊戲畫面，才能啟用圖像辨識。"))
        application = (
            str(LAYOUT.bundle or LAYOUT.install)
            if LAYOUT.packaged
            else tr("執行 MIA 的終端機或 VS Code")
        )
        dialog.setInformativeText(
            tr(
                "請到「系統設定 → 隱私權與安全性 → 螢幕錄製」開啟權限。\n若清單沒有列出，按「＋」"
                "加入：\n{application}\n\n授權後請完整退出並重新開啟 MIA，再啟用需要圖像辨識的功"
                "能。",
                application=application,
            )
        )
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        settings = dialog.addButton(tr("開啟系統設定"), QMessageBox.ButtonRole.AcceptRole)
        settings.clicked.connect(self._open_settings)
        dialog.addButton(tr("稍後"), QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(settings)
        dialog.finished.connect(self._finished)
        self.dialog = dialog
        dialog.show()

    def _open_settings(self) -> None:
        if not QDesktopServices.openUrl(QUrl(SETTINGS_URL)):
            logger.warning("無法開啟螢幕錄製設定，請從系統設定手動開啟")

    def _finished(self, _result: int) -> None:
        dialog, self.dialog = self.dialog, None
        if dialog is not None:
            dialog.deleteLater()


def install_permission_prompt(
    window: QMainWindow, runtime: LiveRuntime
) -> ScreenCapturePermissionPrompt | None:
    if sys.platform != "darwin":
        return None
    prompt = ScreenCapturePermissionPrompt(window)
    runtime.set_screen_capture_permission_check(prompt.check)
    return prompt
