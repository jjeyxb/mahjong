"""Small update dialog. Background I/O never blocks screen recognition."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from mia.distribution.layout import LAYOUT
from mia.distribution.release import ReleaseError
from mia.distribution.service import (
    atomic_json,
    check,
    configuration,
    launch_installer,
    prepare_update,
)
from mia.i18n import tr, translate


class _Task(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, action: Callable[[], Any], parent: QObject) -> None:
        super().__init__(parent)
        self.action = action

    def run(self) -> None:
        try:
            self.result.emit(self.action())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class UpdateDialog(QDialog):
    def __init__(
        self, parent: QMainWindow, runtime: Any = None, *, feed: str | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("MIA 更新") if feed is None else tr("MIA GPU 加速元件"))
        self.resize(480, 300)
        self.runtime, self.feed = runtime, feed
        self.config = configuration()
        self.cancel = threading.Event()
        self.task: _Task | None = None
        self.envelope: bytes | None = None
        self.plan: Path | None = None
        layout = QVBoxLayout(self)
        self.label = QLabel(tr("目前版本 {value}", value=self.config['version']))
        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(False)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.button = QPushButton(tr("檢查更新"))
        self.button.clicked.connect(self.act)
        for widget in (self.label, self.notes, self.bar, self.button):
            layout.addWidget(widget)

    def _run(self, action: Callable[[], Any], done: Callable[[Any], None]) -> None:
        self.button.setEnabled(False)
        self.task = _Task(action, self)
        self.task.result.connect(done)
        self.task.failed.connect(self._failed)
        self.task.progress.connect(self.bar.setValue)
        self.task.start()

    def _failed(self, message: str) -> None:
        self.label.setText(translate(message))
        self.button.setEnabled(True)

    def act(self) -> None:
        if self.plan is not None:
            if self.runtime is not None and (
                self.runtime.game_running() or any(f.enabled for f in self.runtime.feature_list)
            ):
                QMessageBox.information(
                    self, tr("稍後安裝"), tr("請先結束對局／錄製並關閉辨識功能，再安裝更新。")
                )
                return
            answer = QMessageBox.question(
                self, tr("安裝更新"), tr("將關閉 MIA 並安裝，設定與錄影會保留。繼續嗎？")
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            try:
                launch_installer(LAYOUT, self.plan)
            except (OSError, ReleaseError) as exc:
                self._failed(str(exc))
                return
            app = QApplication.instance()
            if app is not None:
                app.quit()
        elif self.envelope is not None:
            envelope = self.envelope
            self.label.setText(tr("正在下載…"))
            self.cancel.clear()

            def progress(done: int, total: int) -> None:
                if self.task is not None:
                    self.task.progress.emit(int(done * 100 / total))

            def action() -> Path:
                return prepare_update(
                    LAYOUT,
                    self.config,
                    envelope,
                    cancelled=self.cancel.is_set,
                    progress=progress,
                )

            self._run(action, self._downloaded)
        else:
            self.label.setText(tr("正在檢查…"))
            self._run(lambda: check(self.config, feed=self.feed), self._checked)

    def _checked(self, result: Any) -> None:
        if result is None:
            self.label.setText(tr("目前已是最新版本"))
        else:
            release, self.envelope = result
            self.label.setText(
                tr(
                    "新版 {version}，下載 {value:.1f} MB",
                    version=release.version,
                    value=release.size / 1024**2,
                )
            )
            self.notes.setPlainText(release.notes)
            self.button.setText(tr("下載"))
        self.button.setEnabled(True)

    def _downloaded(self, plan: Path) -> None:
        self.plan = plan
        self.label.setText(tr("下載與驗證完成，請選擇適合的時間安裝"))
        self.button.setText(tr("安裝並重新啟動"))
        self.button.setEnabled(True)

    def reject(self) -> None:
        if self.task is not None and self.task.isRunning():
            self.cancel.set()
            self.label.setText(tr("正在停止下載，請稍候…"))
            return
        super().reject()


def install_menu(
    window: QMainWindow, app: QApplication, runtime: Any = None
) -> Callable[[], None]:
    """「說明」選單。回傳打開更新視窗的函式,設定頁的「檢查更新」也用它,兩邊開的是同一種視窗。"""
    from mia.distribution.launcher import HEALTH

    config = configuration()
    menu = window.menuBar().addMenu(tr("說明"))
    update = menu.addAction(tr("檢查更新…"))
    dialogs: list[UpdateDialog] = []

    def show(feed: str | None = None) -> None:
        dialog = UpdateDialog(window, runtime, feed=feed)
        dialogs.append(dialog)
        dialog.show()

    update.triggered.connect(lambda: show())
    components = menu.addMenu(tr("GPU 加速元件"))
    feeds = config.get("components", {})
    if not feeds:
        components.addAction(tr("此版本尚未提供額外加速元件")).setEnabled(False)
    for name, feed in feeds.items():
        components.addAction(name).triggered.connect(lambda _checked=False, url=feed: show(url))
    about = menu.addAction(tr("關於 MIA"))
    about.triggered.connect(
        lambda: QMessageBox.information(
            window,
            tr("關於 MIA"),
            tr(
                "MIA {value}\n程式位置：{install}\n資料位置：{data}",
                value=config["version"],
                install=LAYOUT.bundle or LAYOUT.install,
                data=LAYOUT.data,
            ),
        )
    )

    def stop_tasks() -> None:
        for dialog in dialogs:
            dialog.cancel.set()
        for dialog in dialogs:
            if dialog.task is not None:
                dialog.task.wait()

    app.aboutToQuit.connect(stop_tasks)
    if HEALTH is not None:
        path, nonce = HEALTH
        QTimer.singleShot(
            0, lambda: atomic_json(path, {"version": config["version"], "nonce": nonce})
        )
    return show
