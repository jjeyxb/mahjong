"""設定頁的「YOLO 運算裝置」與選 GPU 時的檢查/安裝對話框。

檢查與安裝換成假的:真的那套會開 PowerShell 問顯示卡、下載好幾 GB。
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6", reason="UI 測試需要 PySide6")

from mia.ui.panel import window as window_module
from mia.ui.panel.gpu import GpuSetupDialog
from mia.ui.panel.window import PanelWindow
from mia.ui.viewmodel import ViewModel
from mia.yolo.accel import INSTALL_PHASE, EnvStatus, Plan

RX9070 = Plan("rocm", "AMD Radeon RX 9070")
CPU_ENV = EnvStatus(installed=True, build="cpu")
READY_ENV = EnvStatus(
    installed=True, build="rocm", device="cuda", name="AMD Radeon RX 9070", ok=True
)


class FakeSetup:
    def __init__(
        self,
        plan: Plan = RX9070,
        status: EnvStatus = CPU_ENV,
        *,
        installable: bool = True,
        lines: tuple[str, ...] = ("== 1/2 rocm 版 torch", "YOLO 環境已就緒"),
        result: tuple[bool, str] = (True, "YOLO 環境已就緒"),
    ) -> None:
        self.plan, self.status, self.installable = plan, status, installable
        self.lines, self.result = lines, result
        self.installed: list[str] = []
        #: 設了之後安裝會停在這裡,等 cancelled 或 release。
        self.hold = False
        self.release = threading.Event()

    def inspect(self) -> tuple[Plan, EnvStatus]:
        return self.plan, self.status

    def can_install(self) -> bool:
        return self.installable

    def install(self, backend, on_line, cancelled):
        self.installed.append(backend)
        for line in self.lines:
            on_line(line)
        if self.hold:
            while not cancelled.is_set() and not self.release.is_set():
                cancelled.wait(0.05)
            if cancelled.is_set():
                return False, "已取消,YOLO 環境維持原本的版本。"
        return self.result


def _open(qtbot, setup: FakeSetup, *, busy=lambda: False) -> GpuSetupDialog:
    dialog = GpuSetupDialog(None, setup, busy=busy)
    qtbot.addWidget(dialog)
    # 還沒 show 的視窗 isVisible() 永遠是 False,要看 isHidden():檢查完才會把進度條藏起來。
    qtbot.waitUntil(dialog.bar.isHidden, timeout=5000)
    return dialog


class TestDialog:
    def test_a_ready_environment_needs_no_download(self, qtbot) -> None:
        dialog = _open(qtbot, FakeSetup(status=READY_ENV))
        assert "AMD Radeon RX 9070 已就緒" in dialog.message.text()
        assert not dialog.install_button.isVisible()
        with qtbot.waitSignal(dialog.accepted):
            dialog.close_button.click()

    def test_an_unsupported_card_says_why_and_stays_on_cpu(self, qtbot) -> None:
        plan = Plan(None, "AMD Radeon RX 6600", "AMD Radeon RX 6600 不在 AMD ROCm 的支援清單上")
        dialog = _open(qtbot, FakeSetup(plan=plan))
        assert "RX 6600" in dialog.message.text()
        assert "YOLO 會繼續用 CPU" in dialog.message.text()
        assert dialog.install_button.isHidden()
        with qtbot.waitSignal(dialog.rejected):
            dialog.close_button.click()

    def test_the_download_size_is_shown_before_anything_is_downloaded(self, qtbot) -> None:
        setup = FakeSetup()
        dialog = _open(qtbot, setup)
        assert "ROCm" in dialog.message.text()
        assert "GB" in dialog.message.text()
        assert not dialog.install_button.isHidden()
        assert setup.installed == []

    def test_installing_then_done(self, qtbot) -> None:
        setup = FakeSetup()
        dialog = _open(qtbot, setup)
        dialog.install_button.click()
        qtbot.waitUntil(lambda: dialog.close_button.text() == "完成", timeout=5000)
        assert setup.installed == ["rocm"]
        assert "安裝完成" in dialog.message.text()
        with qtbot.waitSignal(dialog.accepted):
            dialog.close_button.click()

    def test_a_failed_install_stays_on_cpu(self, qtbot) -> None:
        setup = FakeSetup(result=(False, "rocm 版 torch 裝好了,但在顯示卡上算不動。"))
        dialog = _open(qtbot, setup)
        dialog.install_button.click()
        qtbot.waitUntil(lambda: dialog.close_button.text() == "關閉", timeout=5000)
        assert "算不動" in dialog.message.text()
        with qtbot.waitSignal(dialog.rejected):
            dialog.close_button.click()

    def test_nothing_is_installed_while_yolo_is_running(self, qtbot) -> None:
        """Windows 上 torch 的 DLL 被推論子程序占著,pip 換不掉,會裝到一半失敗。"""
        setup = FakeSetup()
        dialog = _open(qtbot, setup, busy=lambda: True)
        dialog.install_button.click()
        assert setup.installed == []
        assert "請先關掉" in dialog.detail.text()

    def test_cancelling_during_the_download(self, qtbot) -> None:
        setup = FakeSetup(lines=("Downloading torch",))
        setup.hold = True
        dialog = _open(qtbot, setup)
        dialog.install_button.click()
        qtbot.waitUntil(lambda: dialog.detail.text() == "Downloading torch", timeout=5000)
        dialog.close_button.click()  # 取消:只是要求停止,視窗等它真的停了才關
        qtbot.waitUntil(lambda: dialog.close_button.text() == "關閉", timeout=5000)
        assert "已取消" in dialog.message.text()

    def test_cancel_is_disabled_once_pip_starts_installing(self, qtbot) -> None:
        setup = FakeSetup(lines=(f"{INSTALL_PHASE}: torch",))
        setup.hold = True
        dialog = _open(qtbot, setup)
        dialog.install_button.click()
        qtbot.waitUntil(lambda: not dialog.close_button.isEnabled(), timeout=5000)
        assert "請不要關閉" in dialog.message.text()
        setup.release.set()
        qtbot.waitUntil(lambda: dialog.close_button.text() == "完成", timeout=5000)

    def test_a_packaged_build_points_to_the_component_menu(self, qtbot) -> None:
        dialog = _open(qtbot, FakeSetup(installable=False))
        assert "加速元件" in dialog.message.text()
        assert dialog.install_button.isHidden()

    def test_closing_while_still_checking_does_not_crash(self, qtbot) -> None:
        """執行緒還在跑時對話框被回收,Qt 會直接結束整個程式。"""
        gate = threading.Event()

        class Slow(FakeSetup):
            def inspect(self):
                gate.wait(5)
                return super().inspect()

        dialog = GpuSetupDialog(None, Slow())
        dialog.reject()
        dialog.deleteLater()
        del dialog
        gate.set()
        qtbot.wait(200)


class FakeDevice:
    def __init__(self) -> None:
        self.value = "cpu"
        self.running = False

    def can_pick_yolo_device(self) -> bool:
        return True

    def yolo_device(self) -> str:
        return self.value

    def set_yolo_device(self, value: str) -> None:
        self.value = value

    def yolo_running(self) -> bool:
        return self.running


class TestSettingsPage:
    @pytest.fixture
    def setup(self, qtbot, monkeypatch):
        answer = {"ok": True}
        asked: list[bool] = []

        def run(_parent, _setup, busy) -> bool:
            asked.append(busy())
            return answer["ok"]

        monkeypatch.setattr(window_module, "_run_gpu_setup", run)
        device = FakeDevice()
        window = PanelWindow(ViewModel(), yolo_device=device, gpu_setup=FakeSetup())
        qtbot.addWidget(window)
        # 視窗本身也要一起回傳(最後一項):只留按鈕的話視窗被回收,按鈕跟著被刪。
        return window._yolo_device_picker, device, answer, asked, window  # noqa: SLF001

    def test_gpu_is_set_only_after_the_setup_succeeds(self, setup) -> None:
        picker, device, _, asked, _window = setup
        picker.button("gpu").click()
        assert asked == [False]
        assert device.value == "gpu"
        assert picker.current() == "gpu"

    def test_a_declined_setup_goes_back_to_cpu(self, setup) -> None:
        picker, device, answer, _, _window = setup
        answer["ok"] = False
        picker.button("gpu").click()
        assert device.value == "cpu"
        assert picker.current() == "cpu"

    def test_going_back_to_cpu_needs_no_dialog(self, setup) -> None:
        picker, device, _, asked, _window = setup
        picker.button("gpu").click()
        picker.button("cpu").click()
        assert asked == [False]
        assert device.value == "cpu"

    def test_disabled_without_a_runtime(self, qtbot) -> None:
        window = PanelWindow(ViewModel())
        qtbot.addWidget(window)
        assert not window._yolo_device_picker.isEnabled()  # noqa: SLF001
