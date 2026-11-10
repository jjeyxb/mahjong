"""Independent update helper. Only signed, owned payloads may replace a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mia.distribution import APP_ID
from mia.distribution.archive import extract
from mia.distribution.release import ReleaseError, verified_release
from mia.distribution.service import atomic_json


def _settings(root: Path, job: Path, *, restore: bool = False) -> None:
    """Only small settings migrate with code; recordings and login profiles stay put."""
    backup = job / "settings-backup"
    if not restore:
        backup.mkdir()
    for name in ("ui_state.json", "config"):
        current, saved = root / "userdata" / name, backup / name
        if restore:
            if current.is_symlink() or current.is_file():
                current.unlink()
            elif current.is_dir():
                shutil.rmtree(current)
            source, target = saved, current
        else:
            if current.is_symlink():
                raise ReleaseError("設定路徑為符號連結，請先改用 MIA 自有資料夾")
            source, target = current, saved
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def wait_for_parent(pid: int, timeout: float = 180) -> None:
    if pid <= 0 or pid == os.getpid():
        raise ReleaseError("不合法的父程序")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle, int(timeout * 1000)) != 0:
                    raise ReleaseError("MIA 尚未退出，取消安裝")
            finally:
                kernel.CloseHandle(handle)
        elif vars(ctypes)["get_last_error"]() not in (0, 87):  # Exited PID: error 87.
            raise ReleaseError("無法確認 MIA 已退出，取消安裝")
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise ReleaseError("MIA 尚未退出，取消安裝")
        time.sleep(0.2)


def _health(command: list[str], health: Path, version: str, nonce: str) -> None:
    health.unlink(missing_ok=True)
    process = subprocess.Popen(
        [*command, "--update-health", str(health), nonce],
        env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if health.is_file():
            result = json.loads(health.read_text(encoding="utf-8"))
            if result.get("version") == version and result.get("nonce") == nonce:
                return
        if process.poll() is not None:
            break
        time.sleep(0.2)
    # Only terminate the exact new child we just started, never another Python/app.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    raise ReleaseError("新版啟動檢查失敗")


def install(
    plan_path: Path,
    config: dict[str, Any],
    *,
    wait: Callable[[int], None] = wait_for_parent,
    health: Callable[[list[str], Path, str, str], None] = _health,
) -> Path:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"]).resolve()
    if plan.get("app_id") != APP_ID or not plan_path.is_relative_to(root / "updates"):
        raise ReleaseError("安裝計畫不在 MIA 專用目錄")
    info_path = root / "install-info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("app_id") != APP_ID or info.get("version") != plan["from_version"]:
        raise ReleaseError("安裝識別或來源版本已變更")
    release = verified_release(
        (plan_path.parent / "release.json").read_bytes(), config["public_key"], config["platform"]
    )
    archive_name = plan["archive"]
    if Path(archive_name).name != archive_name or archive_name != f"{release.sha256}.zip":
        raise ReleaseError("封裝檔名不符")
    archive = plan_path.parent / archive_name
    with archive.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    if archive.stat().st_size != release.size or digest != release.sha256:
        raise ReleaseError("封裝驗證失敗")
    lock = root / "updates" / "install.lock"
    lock.mkdir()  # An existing installation is never overwritten by a second helper.
    try:
        if release.kind == "component":
            return _component(plan_path, archive, release, config, wait)
        from packaging.version import Version

        if Version(release.version) <= Version(info["version"]):
            raise ReleaseError("不接受較舊的程式更新")
        name = "MIA.app" if config["platform"] == "macos-arm64" else "program"
        staged = extract(archive, plan_path.parent / "staged", name, release.unpacked_size)
        target = root / name
        if target.is_symlink() or not target.is_dir():
            raise ReleaseError("程式目錄不是有效的 MIA 安裝")
        executable = Path("Contents/MacOS/MIA") if name == "MIA.app" else Path("MIA.exe")
        if not (staged / executable).is_file():
            raise ReleaseError("新版缺少啟動檔")
        wait(int(plan["parent_pid"]))
        _settings(root, plan_path.parent)
        backup = plan_path.parent / "previous"
        atomic_json(
            plan_path.parent / "transaction.json",
            {
                "target": str(target),
                "backup": str(backup),
                "version": release.version,
            },
        )
        target.rename(backup)
        try:
            staged.rename(target)
            health(
                [str(target / executable), "--live"],
                plan_path.parent / "health.json",
                release.version,
                uuid.uuid4().hex,
            )
            atomic_json(info_path, {**info, "version": release.version})
        except BaseException:
            if target.exists():
                target.rename(plan_path.parent / "failed")
            backup.rename(target)
            _settings(root, plan_path.parent, restore=True)
            atomic_json(info_path, info)
            raise
        atomic_json(plan_path.parent / "result.json", {"installed": release.version})
        return target
    finally:
        lock.rmdir()


def _component(
    plan_path: Path,
    archive: Path,
    release: Any,
    config: dict[str, Any],
    wait: Callable[[int], None],
) -> Path:
    if release.app_version != config["version"]:
        raise ReleaseError("加速元件與 App 版本不相容")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"]).resolve()
    staged = extract(archive, plan_path.parent / "staged", "component", release.unpacked_size)
    name = "MIA-yolo.exe" if config["platform"] == "windows-x64" else "MIA-yolo"
    worker = staged / name
    if worker.is_symlink() or not worker.is_file():
        raise ReleaseError("加速元件缺少推論程序")
    resources = (
        root / "MIA.app/Contents/Resources/resources"
        if config["platform"] == "macos-arm64"
        else root / "program/_internal/resources"
    )
    check = subprocess.run(
        [str(worker), "--self-check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={
            **os.environ,
            "PYINSTALLER_RESET_ENVIRONMENT": "1",
            "MIA_RESOURCE_ROOT": str(resources),
            "MJ_DEVICE": "cuda",
        },
    )
    result = json.loads(check.stdout)
    backend = release.component.split("-", 1)[1]
    if result.get("backend") != backend or not result.get("ok"):
        raise ReleaseError("加速元件無法在這台電腦上執行")
    wait(int(plan["parent_pid"]))
    target = root / "components" / release.platform / release.component / release.version
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ReleaseError("此版本元件已存在")
    staged.rename(target)
    (target / "release.json").write_bytes((plan_path.parent / "release.json").read_bytes())
    atomic_json(
        root / "components" / "active.json",
        {
            "component": release.component,
            "version": release.version,
        },
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    # Trust configuration is copied with this helper from the installed build.
    config_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "resources/release.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target = install(args.plan, config)
        envelope = verified_release(
            (args.plan.parent / "release.json").read_bytes(),
            config["public_key"],
            config["platform"],
        )
        if envelope.kind == "component":
            root = Path(json.loads(args.plan.read_text())["root"])
            exe = (
                root / "MIA.app/Contents/MacOS/MIA"
                if config["platform"] == "macos-arm64"
                else root / "program/MIA.exe"
            )
            subprocess.Popen(
                [str(exe), "--live"], env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}
            )
        print(f"已安裝:{target}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"更新未完成，保留原版本:{exc}", file=sys.stderr)
        return 1
