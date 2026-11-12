"""「YOLO 運算裝置」選 GPU 時跳出的對話框:檢查顯示卡,需要的話下載並安裝 GPU 版 torch。

流程::

    檢查(背景)──┬─ 環境裡的 torch 已經算得動顯示卡 ── 已就緒 ── [完成] → 接受
                 ├─ 有支援的顯示卡、可以自己裝 ─────── [下載並安裝] ── 安裝(背景)── 成功 → 接受
                 │                                                          └ 失敗 → 拒絕
                 ├─ 有支援的顯示卡、但沒辦法自動裝 ─── 說明 → 拒絕
                 └─ 沒有支援的顯示卡 ────────────────── 說明為什麼 → 拒絕

呼叫端只看結果:接受就把裝置設成 GPU,其他情況維持 CPU。**沒有「裝好一半但設成 GPU」的狀態**
—— 那樣 YOLO 會安靜地退回 CPU,使用者以為在用顯示卡。

真正的檢查與安裝在 :mod:`mia.yolo.accel`;這裡只管畫面,所以測試可以換一個假的
:class:`~mia.ui.switchboard.GpuSetup` 進來。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mia.i18n import tr, translate
from mia.ui.switchboard import GpuSetup
from mia.yolo.accel import BACKEND_NAMES, INSTALL_PHASE, EnvStatus, Plan

__all__ = ["GpuSetupDialog"]


class _Task(QThread):
    """在背景跑一件事,做完把結果送回主執行緒。形狀同 :mod:`mia.distribution.ui` 的 ``_Task``。"""

    result = Signal(object)
    line = Signal(str)

    def __init__(self, action: Callable[[_Task], Any], parent: QObject) -> None:
        super().__init__(parent)
        self._action = action

    def run(self) -> None:
        try:
            self.result.emit(self._action(self))
        except Exception as exc:  # noqa: BLE001 —— 背景執行緒的例外要變成畫面上的一句話
            self.result.emit(exc)


class GpuSetupDialog(QDialog):
    """見模組說明。

    Args:
        setup: 檢查與安裝。
        busy: YOLO 現在有沒有在跑。在跑的話不能裝 —— Windows 上 torch 的 DLL 被占著,
            pip 換不掉,會裝到一半失敗。
    """

    def __init__(
        self, parent: QWidget | None, setup: GpuSetup, *, busy: Callable[[], bool] = lambda: False
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("YOLO 使用 GPU"))
        self.setMinimumWidth(460)
        self._setup = setup
        self._busy = busy
        self._plan: Plan | None = None
        self._task: _Task | None = None
        self._cancel = threading.Event()
        self._installing = False

        layout = QVBoxLayout(self)
        self.message = QLabel(tr("正在檢查顯示卡與 YOLO 環境…"), self)
        self.message.setWordWrap(True)
        self.detail = QLabel("", self)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: gray;")
        self.detail.hide()
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)  # 不知道總量:pip 不會給可靠的百分比
        self.bar.setTextVisible(False)
        self.install_button = QPushButton(tr("下載並安裝"), self)
        self.install_button.hide()
        self.install_button.clicked.connect(self._on_install)
        self.close_button = QPushButton(tr("取消"), self)
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.install_button)
        buttons.addWidget(self.close_button)
        for widget in (self.message, self.detail, self.bar):
            layout.addWidget(widget)
        layout.addLayout(buttons)

        self._run(lambda _task: setup.inspect(), self._inspected)

    # ------------------------------------------------------------------ 背景工作

    def _run(self, action: Callable[[_Task], Any], done: Callable[[Any], None]) -> None:
        self._task = _Task(action, self)
        self._task.result.connect(done)
        self._task.line.connect(self._on_line)
        self._task.start()

    def _inspected(self, result: object) -> None:
        self.bar.hide()
        if isinstance(result, Exception):
            self._finish_failed(tr("檢查失敗:{error}", error=result))
            return
        assert isinstance(result, tuple)
        plan, status = result
        assert isinstance(plan, Plan) and isinstance(status, EnvStatus)
        self._plan = plan
        if status.gpu_ready:
            self.message.setText(
                tr("{gpu} 已就緒，YOLO 會改用顯示卡執行。", gpu=status.name or plan.gpu)
            )
            self._offer_done()
            return
        if plan.backend is None:
            self._finish_failed(plan.reason)
            return
        backend = BACKEND_NAMES[plan.backend]
        if plan.download_mb == 0:
            # Apple MPS:基本環境就該有,算不動是環境本身的問題,下載也救不了。
            self._finish_failed(
                tr("{gpu} 目前用不了:{error}", gpu=plan.gpu, error=translate(status.error) or "?")
            )
            return
        if not self._setup.can_install():
            self._finish_failed(
                tr("偵測到 {gpu},但這個版本的 MIA 沒辦法自動安裝 {backend} 加速。",
                   gpu=plan.gpu, backend=backend)
            )
            return
        self.message.setText(
            tr("偵測到 {gpu}。要用顯示卡跑 YOLO,需要下載 {backend} 版的 PyTorch"
               "(約 {size:.1f} GB),裝進 YOLO 的環境。",
               gpu=plan.gpu, backend=backend, size=plan.download_mb / 1000)
        )
        if status.error and status.installed and status.build != "cpu":
            # 裝過 GPU 版卻算不動:多半是驅動太舊。重裝不一定有用,先講清楚。
            self.detail.setText(
                tr("目前環境的顯示卡檢查失敗:{error}", error=translate(status.error))
            )
            self.detail.show()
        self.install_button.show()

    def _on_install(self) -> None:
        if self._plan is None or self._plan.backend is None:
            return
        if self._busy():
            self.detail.setText(
                tr("YOLO 正在執行。請先關掉使用 YOLO 的功能"
                   "(畫面辨識、AI 建議、放銃分析),再按安裝。")
            )
            self.detail.show()
            return
        backend = self._plan.backend
        self.install_button.hide()
        self.bar.show()
        self.detail.setText("")
        self.detail.show()
        self.message.setText(tr("下載中…下載完成前可以取消，YOLO 環境不會被動到。"))
        self._installing = True
        cancel = self._cancel

        def action(task: _Task) -> tuple[bool, str]:
            return self._setup.install(backend, task.line.emit, cancel)

        self._run(action, self._installed)

    def _on_line(self, line: str) -> None:
        self.detail.setText(line[-200:])
        if INSTALL_PHASE in line and self.close_button.isEnabled():
            # pip 開始換套件了。這時停下來會留下裝到一半的 torch,所以不給取消。
            self.message.setText(tr("安裝中，請不要關閉 MIA…"))
            self.close_button.setEnabled(False)

    def _installed(self, result: object) -> None:
        self._installing = False
        self.bar.hide()
        self.close_button.setEnabled(True)
        if isinstance(result, Exception):
            self._finish_failed(tr("安裝失敗:{error}", error=result))
            return
        assert isinstance(result, tuple)
        ok, last = result
        if not ok:
            self._finish_failed(tr("沒有裝成:{error}", error=translate(str(last))))
            return
        assert self._plan is not None
        self.message.setText(
            tr("安裝完成:{gpu} 已就緒，YOLO 會改用顯示卡執行。", gpu=self._plan.gpu)
        )
        self._offer_done()

    # ------------------------------------------------------------------ 收尾

    def _offer_done(self) -> None:
        self.close_button.setText(tr("完成"))
        self.close_button.clicked.disconnect()
        self.close_button.clicked.connect(self.accept)

    def _finish_failed(self, message: str) -> None:
        self.message.setText(message + "\n" + tr("YOLO 會繼續用 CPU。"))
        self.install_button.hide()
        self.close_button.setText(tr("關閉"))

    def reject(self) -> None:
        """關掉視窗。安裝中按取消只是**要求**停止:要等子程序真的停了才關,不然它在背後繼續裝。"""
        if self._installing:
            self._cancel.set()
            self.message.setText(tr("正在停止下載…"))
            self.close_button.setEnabled(False)
            return
        if self._task is not None and self._task.isRunning():
            # 還在檢查:檢查不會動到任何東西,不必讓使用者等它。但執行緒不能跟著對話框
            # 被回收 —— Qt 遇到還在跑的 QThread 被銷毀會直接結束整個程式。交給模組保管到它跑完。
            task = self._task
            task.result.disconnect()
            task.line.disconnect()
            task.setParent(None)
            _ORPHANS.add(task)
            task.finished.connect(lambda: _ORPHANS.discard(task))
            self._task = None
        super().reject()


#: 對話框關掉時還沒跑完的檢查。見 :meth:`GpuSetupDialog.reject`。
_ORPHANS: set[_Task] = set()
