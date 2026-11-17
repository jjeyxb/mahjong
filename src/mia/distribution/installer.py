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


def _settings(data: Path, job: Path, *, restore: bool = False) -> None:
    """Only small settings migrate with code; recordings and login profiles stay put."""
    backup = job / "settings-backup"
    if not restore:
        backup.mkdir()
    for name in ("ui_state.json", "config"):
        current, saved = data / name, backup / name
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


def _rename(source: Path, target: Path, *, attempts: int = 30) -> None:
    """改名整個資料夾;Windows 上被拒絕就等一下再試。

    MIA 剛退出時,它開的推論程序與瀏覽器可能還要一兩秒才結束;在那之前它們開著的檔案或
    工作資料夾會讓整個 program 資料夾改不了名(WinError 32)。30 秒還不行才算失敗。
    """
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == attempts - 1:
                raise
            time.sleep(1)


def _health(command: list[str], health: Path, version: str, nonce: str) -> None:
    health.unlink(missing_ok=True)
    process = subprocess.Popen(
        [*command, "--update-health", str(health), nonce],
        # 新版就是使用者接下來用的 MIA:工作資料夾跟捷徑一樣放在程式旁邊,不要留在這次更新的
        # 暫存資料夾裡(否則之後清不掉它)。
        cwd=Path(command[0]).parent,
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


def _data_dir(plan: dict[str, Any], root: Path) -> Path:
    """設定放在哪。舊的計畫沒寫這一欄,那時一律是 root/userdata。"""
    data = Path(plan.get("data", root / "userdata")).resolve()
    if not data.is_relative_to(root):
        raise ReleaseError("設定目錄不在 MIA 專用目錄")
    return data


def _app_target(plan: dict[str, Any], root: Path, platform: str) -> Path:
    """要換掉的程式。Mac 的 MIA.app 可以在任何地方(通常是「應用程式」),所以由計畫指定。"""
    if platform != "macos-arm64":
        return root / "program"
    target = Path(plan.get("app", root / "MIA.app"))
    if target.suffix != ".app" or target.is_symlink():
        raise ReleaseError("程式目錄不是有效的 MIA 安裝")
    target = target.resolve()
    try:
        bundled = json.loads(
            (target / "Contents/Resources/resources/release.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ReleaseError("程式目錄不是有效的 MIA 安裝") from exc
    if bundled.get("app_id") != APP_ID:
        raise ReleaseError("程式目錄不是有效的 MIA 安裝")
    return target


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
        from packaging.version import Version

        if Version(release.version) <= Version(info["version"]):
            raise ReleaseError("不接受較舊的程式更新")
        name = "MIA.app" if config["platform"] == "macos-arm64" else "program"
        target = _app_target(plan, root, config["platform"])
        if target.is_symlink() or not target.is_dir():
            raise ReleaseError("程式目錄不是有效的 MIA 安裝")
        data = _data_dir(plan, root)
        # 換程式靠 rename,跨磁碟做不到;App 放在外接碟時寧可不更新,也不要換到一半。
        if target.parent.stat().st_dev != plan_path.parent.stat().st_dev:
            raise ReleaseError("MIA.app 與資料不在同一顆磁碟，請手動下載新版")
        staged = extract(archive, plan_path.parent / "staged", name, release.unpacked_size)
        executable = Path("Contents/MacOS/MIA") if name == "MIA.app" else Path("MIA.exe")
        if not (staged / executable).is_file():
            raise ReleaseError("新版缺少啟動檔")
        wait(int(plan["parent_pid"]))
        _settings(data, plan_path.parent)
        backup = plan_path.parent / "previous"
        atomic_json(
            plan_path.parent / "transaction.json",
            {
                "target": str(target),
                "backup": str(backup),
                "version": release.version,
            },
        )
        _rename(target, backup)
        try:
            _rename(staged, target)
            health(
                [str(target / executable), "--live"],
                plan_path.parent / "health.json",
                release.version,
                uuid.uuid4().hex,
            )
            atomic_json(info_path, {**info, "version": release.version})
        except BaseException:
            if target.exists():
                _rename(target, plan_path.parent / "failed")
            _rename(backup, target)
            _settings(data, plan_path.parent, restore=True)
            atomic_json(info_path, info)
            raise
        atomic_json(plan_path.parent / "result.json", {"installed": release.version})
        return target
    finally:
        lock.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    # 工作資料夾不能在要換掉的 program 裡:Windows 上有程序的工作資料夾在裡面,整個資料夾就
    # 改不了名(WinError 32)。捷徑的「開始位置」就是 program,MIA 與這個程序都會繼承它
    # —— 0.1.0 → 0.1.1 的 App 內更新就是這樣失敗的(2026-11-17)。
    os.chdir(args.plan.resolve().parent)
    # Trust configuration is copied with this helper from the installed build.
    config_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "resources/release.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target = install(args.plan, config)
        print(f"已安裝:{target}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"更新未完成，保留原版本:{exc}", file=sys.stderr)
        return 1
