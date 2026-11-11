"""tools/setup_yolo.py:換 GPU 版本時,舊版本的執行環境只拿掉沒人在用的。

在一個只放假套件的資料夾上跑 ``_UNUSED``(它的第二個參數:只在那裡找套件)。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import setup_yolo

ROCM = ["rocm", "rocm-sdk-core", "rocm-sdk-libraries-custom"]


def _unused(tmp_path: Path, packages: dict[str, list[str]], stale: list[str]) -> list[str]:
    for name, requires in packages.items():
        info = tmp_path / f"{name.replace('-', '_')}-1.0.dist-info"
        info.mkdir()
        lines = ["Metadata-Version: 2.1", f"Name: {name}", "Version: 1.0"]
        lines += [f"Requires-Dist: {r}" for r in requires]
        (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    script = setup_yolo._UNUSED  # noqa: SLF001
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(stale), json.dumps([str(tmp_path)])],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture
def rocm_runtime() -> dict[str, list[str]]:
    return {
        "rocm": [
            "rocm-sdk-core==7.2.1",
            'rocm-sdk-libraries-custom==7.2.1; extra == "libraries"',
        ],
        "rocm-sdk-core": [],
        "rocm-sdk-libraries-custom": [],
    }


def test_after_switching_to_cuda_the_rocm_runtime_goes(tmp_path, rocm_runtime) -> None:
    packages = {"torch": ["filelock"], "filelock": [], **rocm_runtime}
    assert _unused(tmp_path, packages, ROCM) == sorted(ROCM)


def test_nothing_goes_while_torch_still_needs_it(tmp_path, rocm_runtime) -> None:
    """torch 要的是 rocm[libraries]:附加依賴那個也要留。"""
    packages = {"torch": ["rocm[libraries]==7.2.1"], **rocm_runtime}
    assert _unused(tmp_path, packages, ROCM) == []


def test_a_shared_package_stays_with_what_it_needs(tmp_path) -> None:
    """XPU 清單裡的 mkl 若有清單外的套件要,mkl 與它要的 tbb 都留;沒人要的照樣拿掉。"""
    packages = {
        "numpy": ["mkl"],
        "mkl": ["tbb", "intel-openmp"],
        "tbb": [],
        "intel-openmp": [],
        "intel-sycl-rt": [],
    }
    stale = ["mkl", "tbb", "intel-openmp", "intel-sycl-rt"]
    assert _unused(tmp_path, packages, stale) == ["intel-sycl-rt"]


def test_platform_markers_follow_this_machine(tmp_path) -> None:
    """只在 Linux 才要的依賴,在別的平台上不算有人要。"""
    other = "Windows" if sys.platform != "win32" else "Linux"
    packages = {"torch": [f'oneccl; platform_system == "{other}"'], "oneccl": []}
    assert _unused(tmp_path, packages, ["oneccl"]) == ["oneccl"]
