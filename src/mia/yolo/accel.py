"""YOLO 的 GPU 加速:這台電腦是哪一家的顯示卡、該裝哪一種 torch、YOLO 環境現在裝的是哪一種。

不碰 Qt 也不 import torch(torch 在另一個 Python 3.12 的環境裡,見 ``engines/yolo``)。
設定頁在背景執行緒呼叫這裡,所以每個函式都可以慢、可以等子程序。

四家顯示卡各走哪條路
--------------------
================  ============  ==============================================================
顯示卡             torch 版本     說明
================  ============  ==============================================================
Apple Silicon     (不用換)       PyPI 的 torch 本來就有 MPS,基本環境就會用
NVIDIA            ``cuda``      CUDA 12.8 版,約 2.9 GB
AMD               ``rocm``      AMD 官方的 Windows ROCm 版,約 2.0 GB;**只有支援清單上的卡**
Intel Arc         ``xpu``       PyTorch 的 XPU 版,約 1.4 GB;舊的 Iris Xe 不支援
================  ============  ==============================================================

AMD 不能看到品牌就裝:ROCm 只支援特定幾代(下面的 :data:`_ROCM_CARDS`),裝了不支援的卡只會白下載
兩 GB。NVIDIA 與 Intel 沒有這麼細的清單,裝完由 ``engines/yolo/probe.py`` 實際在卡上算一次,
算不動就明講。

一張筆電常常同時有內顯和獨顯:NVIDIA > AMD > Intel,先挑效能通常比較好的那張。
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mia.distribution.layout import LAYOUT
from mia.i18n import tr
from mia.utils.paths import PROJECT_ROOT

__all__ = [
    "BACKEND_NAMES",
    "INSTALL_PHASE",
    "EnvStatus",
    "LocalGpuSetup",
    "Plan",
    "can_install",
    "install",
    "list_gpus",
    "parse_probe",
    "plan_for",
    "probe",
]

#: torch 版本的顯示名稱。
BACKEND_NAMES: dict[str, str] = {
    "cuda": "CUDA",
    "rocm": "ROCm",
    "xpu": "Intel XPU",
    "mps": "Apple MPS",
    "cpu": "CPU",
}

#: 下載大小(MB),2026-11-12 從各來源的檔案大小加總。顯示在「要下載嗎」的確認框。
_DOWNLOAD_MB: dict[str, int] = {"cuda": 2900, "rocm": 2000, "xpu": 1400}

#: Windows 上 ROCm 7.2.1 支援的 AMD 顯示卡。AMD 官方清單(2026-11-12)列的是 gfx1201 / 1200 / 1100 /
#: 1101 四個架構上的 RX 9070 / 9070 XT / 9060 XT、AI PRO R9700、RX 7900 XTX、PRO W7900、RX 7700;
#: 這裡放寬到同架構的其他型號(RX 7900 XT / GRE、7800 XT、7700 XT、PRO W7800)—— 它們跑的是同一套
#: GPU 核心,不支援的話裝完的實際運算檢查會抓到。RX 7600(gfx1102)以下不在支援的架構裡。
_ROCM_CARDS = re.compile(
    r"RX\s*9070|RX\s*9060|AI\s*PRO\s*R9700|RX\s*7900|RX\s*7800|RX\s*7700|PRO\s*W7[89]00",
    re.IGNORECASE,
)

#: Intel 的 XPU 版 torch 支援 Arc 獨顯與 Core Ultra 內建的 Arc 顯示晶片,名稱裡都有「Arc」。
_INTEL_ARC = re.compile(r"\bArc\b", re.IGNORECASE)


@dataclass(frozen=True)
class Plan:
    """這台電腦該用哪一種 torch。

    Attributes:
        backend: ``cuda`` / ``rocm`` / ``xpu`` / ``mps``;沒有能用的顯示卡就是 ``None``。
        gpu: 挑中的那張顯示卡的名稱。
        reason: ``backend`` 是 ``None`` 時,為什麼不行 —— 直接顯示給使用者看。
    """

    backend: str | None
    gpu: str = ""
    reason: str = ""

    @property
    def download_mb(self) -> int:
        """要下載多少。``mps`` 是 0:基本環境就有。"""
        return _DOWNLOAD_MB.get(self.backend or "", 0)


@dataclass(frozen=True)
class EnvStatus:
    """YOLO 環境現在裝的 torch。對應 ``engines/yolo/probe.py`` 印的那一行。

    Attributes:
        installed: 有沒有 YOLO 環境(沒有的話其他欄位都沒意義)。
        build: torch 是哪一種版本(``cpu`` / ``cuda`` / ``rocm`` / ``xpu``)。
        device: 可以用的顯示卡裝置名(``cuda`` / ``xpu`` / ``mps``),沒有就是 ``None``。
        name: 顯示卡名稱。
        ok: 真的在顯示卡上算過一次而且算對。
        error: 算不動的原因。
    """

    installed: bool
    build: str = "cpu"
    device: str | None = None
    name: str = ""
    ok: bool = False
    error: str = ""

    @property
    def gpu_ready(self) -> bool:
        return self.installed and self.ok and self.device is not None


def list_gpus() -> list[str]:
    """這台電腦的顯示卡名稱。問不到就是空的(當成沒有顯示卡,只能用 CPU)。"""
    if sys.platform == "darwin":
        # Apple Silicon 的 GPU 不會出現在任何「顯示卡清單」裡,看 CPU 架構就知道。
        return ["Apple Silicon"] if platform.machine() == "arm64" else []
    if sys.platform != "win32":
        return []
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def plan_for(gpus: list[str], *, system: str | None = None) -> Plan:
    """從顯示卡名稱決定該裝哪一種 torch。純函式,測試直接餵名稱。"""
    system = system or sys.platform
    if system == "darwin":
        if gpus:
            return Plan("mps", gpus[0])
        return Plan(None, reason=tr("Intel 處理器的 Mac 沒有 YOLO 能用的顯示卡加速,請用 CPU。"))
    if system != "win32":
        return Plan(None, reason=tr("GPU 加速目前只支援 Windows 與 Apple Silicon 的 Mac。"))
    for name in gpus:
        if "nvidia" in name.lower():
            return Plan("cuda", name)
    amd = [name for name in gpus if re.search(r"\bAMD\b|Radeon", name, re.IGNORECASE)]
    for name in amd:
        if _ROCM_CARDS.search(name):
            return Plan("rocm", name)
    for name in gpus:
        if "intel" in name.lower() and _INTEL_ARC.search(name):
            return Plan("xpu", name)
    if amd:
        return Plan(None, amd[0], tr(
            "{gpu} 不在 AMD ROCm 的支援清單上(Windows 只支援 RX 7700 以上與 RX 9000 系列),"
            "請用 CPU。", gpu=amd[0]))
    intel = [name for name in gpus if "intel" in name.lower()]
    if intel:
        return Plan(None, intel[0], tr(
            "{gpu} 不是 Arc 顯示晶片,PyTorch 不支援,請用 CPU。", gpu=intel[0]))
    return Plan(None, reason=tr(
        "找不到支援的顯示卡(NVIDIA、AMD Radeon RX 7000/9000、Intel Arc),請用 CPU。"))


def _environment() -> tuple[Path, Path]:
    """YOLO 環境的 Python 與 engines/yolo 目錄。只有原始碼版本有。"""
    base = PROJECT_ROOT / "engines/yolo"
    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return Path(os.environ.get("MIA_YOLO_PYTHON", base / ".venv" / executable)), base


def can_install() -> bool:
    """這個 MIA 能不能自己裝 GPU 版 torch。

    只有原始碼版本可以:它會在 ``engines/yolo/.venv`` 跑 pip。打包版不在使用者電腦上跑 pip
    (``docs/packaging.md``「可選 GPU 加速包」),要等發布方提供簽過名的加速元件。
    """
    return not LAYOUT.packaged and (PROJECT_ROOT / "tools/setup_yolo.py").is_file()


def probe(timeout: float = 180.0) -> EnvStatus:
    """問 YOLO 環境裡的 torch。第一次在 AMD 卡上跑要編譯 GPU 核心,可能要幾十秒。"""
    if LAYOUT.packaged:
        return _probe_worker(timeout)
    python, base = _environment()
    if not python.is_file():
        return EnvStatus(installed=False)
    try:
        result = subprocess.run(
            [str(python), str(base / "probe.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            cwd=PROJECT_ROOT,
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EnvStatus(installed=True, error=tr("檢查 YOLO 環境失敗:{error}", error=exc))
    return parse_probe(result.stdout)


def _probe_worker(timeout: float) -> EnvStatus:
    """打包版:沒有 engines/yolo/.venv 與 probe.py,直接把打包好的推論程序叫起來問。

    要 GPU 啟動,看它回報選了哪個裝置(``ready`` 那一行的 ``device``)。它啟動時已經把四個
    模型載到那個裝置上,所以回報 GPU 就代表那張卡載得動模型。Apple Silicon 的打包版走這條:
    基本的推論程序就有 MPS。
    """
    from mia.engine.base import EngineError
    from mia.yolo.client import YoloClient

    try:
        client = YoloClient(device="gpu", timeout=timeout)
        client.start()
    except EngineError as exc:
        return EnvStatus(installed=True, error=str(exc))
    try:
        device = client.device
    finally:
        client.close()
    if device in ("", "cpu"):
        return EnvStatus(installed=True, build="cpu")
    name = "Apple GPU" if device == "mps" else device.upper()
    return EnvStatus(installed=True, build=device, device=device, name=name, ok=True)


def parse_probe(output: str) -> EnvStatus:
    """``probe.py`` 的輸出。只看最後一行 JSON:torch 載入時可能先印別的東西。"""
    for line in reversed(output.strip().splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            device = data.get("device")
            return EnvStatus(
                installed=True,
                build=str(data.get("build") or "cpu"),
                device=str(device) if device else None,
                name=str(data.get("name") or ""),
                ok=bool(data.get("ok")),
                error=str(data.get("error") or ""),
            )
    return EnvStatus(installed=True, error=tr("檢查 YOLO 環境沒有回應"))


#: pip 開始動環境的那一行。在這之前取消,環境還是原來那一套;之後就不能中斷了。
INSTALL_PHASE = "Installing collected packages"


def install(
    backend: str,
    on_line: Callable[[str], None],
    cancelled: threading.Event,
) -> tuple[bool, str]:
    """跑 ``tools/setup_yolo.py --gpu <backend>``,一行一行回報輸出。

    回 ``(成功與否, 最後一行說明)``。``cancelled`` 被設起來時,只要 pip 還沒開始裝
    (見 :data:`INSTALL_PHASE`)就停掉子程序;開始裝之後不停 —— 裝到一半的 torch 是壞的環境。
    """
    command = [sys.executable, str(PROJECT_ROOT / "tools/setup_yolo.py"), "--gpu", backend]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
            # 自成一個程序群組,取消時才能連 pip 一起停(見 _kill_tree)。
            start_new_session=sys.platform != "win32",
        )
    except OSError as exc:
        return False, tr("無法執行安裝程式:{error}", error=exc)
    assert process.stdout is not None
    last = ""
    installing = False
    stopper = threading.Thread(
        target=_stop_when_cancelled, args=(process, cancelled, lambda: installing), daemon=True
    )
    stopper.start()
    for raw in iter(process.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        if INSTALL_PHASE in line:
            installing = True
        last = line
        on_line(line)
    code = process.wait()
    if cancelled.is_set() and code != 0:
        return False, tr("已取消,YOLO 環境維持原本的版本。")
    return code == 0, last


def _stop_when_cancelled(
    process: subprocess.Popen[bytes], cancelled: threading.Event, installing: Callable[[], bool]
) -> None:
    while process.poll() is None:
        if cancelled.wait(0.3) and not installing():
            _kill_tree(process)
            return


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """連子程序一起停。只停 setup_yolo.py 的話,它開的 pip 會繼續下載、繼續裝。"""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
    else:
        import signal

        os.killpg(process.pid, signal.SIGTERM)


class LocalGpuSetup:
    """:class:`~mia.ui.switchboard.GpuSetup` 的實作:這台電腦、這份原始碼的 YOLO 環境。"""

    def inspect(self) -> tuple[Plan, EnvStatus]:
        return plan_for(list_gpus()), probe()

    def can_install(self) -> bool:
        return can_install()

    def install(
        self, backend: str, on_line: Callable[[str], None], cancelled: threading.Event
    ) -> tuple[bool, str]:
        return install(backend, on_line, cancelled)
