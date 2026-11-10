"""Application-facing update service; no recognition or Qt dependencies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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


def check(config: dict[str, Any], *, feed: str | None = None) -> tuple[Release, bytes] | None:
    if not config.get("public_key") or not (feed or config.get("feed_url")):
        raise ReleaseError("這個版本尚未設定更新來源與驗證金鑰")
    envelope = fetch_metadata(feed or config["feed_url"])
    release = verified_release(envelope, config["public_key"], config["platform"])
    if release.kind == "app" and Version(release.version) <= Version(config["version"]):
        return None
    if release.kind == "component" and release.app_version != config["version"]:
        raise ReleaseError("加速元件與目前 App 版本不相容")
    return release, envelope


def prepare_update(
    layout: Layout, config: dict[str, Any], envelope: bytes, **download_options: Any
) -> Path:
    release = verified_release(envelope, config["public_key"], config["platform"])
    if release.kind == "app" and Version(release.version) <= Version(config["version"]):
        raise ReleaseError("不接受較舊的程式更新")
    if release.kind == "component" and release.app_version != config["version"]:
        raise ReleaseError("加速元件與目前 App 版本不相容")
    folder = layout.updates / uuid.uuid4().hex
    archive = download(release, folder, **download_options)
    (folder / "release.json").write_bytes(envelope)
    atomic_json(
        folder / "plan.json",
        {
            "app_id": APP_ID,
            "root": str(layout.install),
            "archive": archive.name,
            "from_version": config["version"],
            "parent_pid": os.getpid(),
        },
    )
    return folder / "plan.json"


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
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
