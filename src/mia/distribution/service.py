"""Application-facing update service; no recognition or Qt dependencies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from packaging.version import Version

from mia.distribution import APP_ID, VERSION
from mia.distribution.download import download, fetch_metadata
from mia.distribution.layout import LAYOUT, Layout, platform_key
from mia.distribution.release import Release, ReleaseError, verified_release


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configuration(layout: Layout = LAYOUT) -> dict[str, Any]:
    path = layout.resources / "release.json"
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("app_id") != APP_ID:
            raise ReleaseError("發布設定的應用程式識別不符")
        return config
    return dict(app_id=APP_ID, version=VERSION, platform=platform_key(), public_key="", feed_url="")


def check(config: dict[str, Any]) -> tuple[Release, bytes] | None:
    if not config.get("public_key") or not config.get("feed_url"):
        raise ReleaseError("這個版本尚未設定更新來源與驗證金鑰")
    envelope = fetch_metadata(config["feed_url"])
    release = verified_release(envelope, config["public_key"], config["platform"])
    if release.kind == "app" and Version(release.version) <= Version(config["version"]):
        return None
    return release, envelope


def prepare_update(
    layout: Layout, config: dict[str, Any], envelope: bytes, **download_options: Any
) -> Path:
    release = verified_release(envelope, config["public_key"], config["platform"])
    if release.kind == "app" and Version(release.version) <= Version(config["version"]):
        raise ReleaseError("不接受較舊的程式更新")
    if release.kind == "app" and (blocker := layout.update_blocker()) is not None:
        raise ReleaseError(blocker)
    folder = layout.updates / uuid.uuid4().hex
    archive = download(release, folder, **download_options)
    (folder / "release.json").write_bytes(envelope)
    atomic_json(
        folder / "plan.json",
        {
            "app_id": APP_ID,
            "root": str(layout.root),
            # Mac 的 App 不在資料根目錄底下,要明說換哪一個;Windows 換 root/program。
            **({"app": str(layout.bundle)} if layout.bundle is not None else {}),
            "data": str(layout.data),
            "archive": archive.name,
            "from_version": config["version"],
            "parent_pid": os.getpid(),
        },
    )
    return folder / "plan.json"


def clean_updates(updates: Path, *, stale_after: float = 24 * 3600) -> None:
    """清掉之前更新留下的暫存:下載的壓縮檔、解開的新版、換下來的舊版(各 1~3 GB)。

    安裝裝完(或失敗)之後就用不到了,但原本沒有人刪,每更新一次多約 4 GB。
    更新程序正在換程式時(install.lock 在)一律不動;還沒進到安裝的(例如正在下載)只刪
    放了超過一天的。
    """
    if not updates.is_dir() or (updates / "install.lock").exists():
        return
    now = time.time()
    for job in updates.iterdir():
        if job.is_symlink() or not job.is_dir() or job.name == "install.lock":
            continue
        finished = (job / "transaction.json").exists() or (job / "result.json").exists()
        if finished or now - job.stat().st_mtime > stale_after:
            shutil.rmtree(job, ignore_errors=True)


def launch_installer(layout: Layout, plan: Path) -> subprocess.Popen[bytes]:
    helper = layout.worker("updater")
    if not helper.is_file():
        raise ReleaseError("找不到更新安裝程序，請重新下載安裝包")
    # Windows cannot replace the executable or DLLs still used by its updater.
    copied = plan.parent / "helper"
    shutil.copytree(helper.parent, copied, symlinks=True)
    executable = copied / helper.name
    log = layout.logs / "update-installer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        return subprocess.Popen(
            [str(executable), "--plan", str(plan)],
            # 不要繼承 MIA 的工作資料夾(捷徑設在 program 裡),否則 Windows 上 program 改不了名。
            cwd=plan.parent,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
