"""Downloads are staged; failure never replaces an installed executable."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import certifi

from mia.distribution.release import MAX_METADATA, Release, ReleaseError, https_url


def fetch_metadata(url: str) -> bytes:
    with urllib.request.urlopen(
        https_url(url), timeout=20, context=ssl.create_default_context(cafile=certifi.where())
    ) as response:
        https_url(response.geturl())
        data = response.read(MAX_METADATA + 1)
    if len(data) > MAX_METADATA:
        raise ReleaseError("更新資訊超過大小限制")
    return data


def download(
    release: Release,
    folder: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda _done, _total: None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    # Signed expected length is also a hard transfer limit.
    if shutil.disk_usage(folder).free < release.size + release.unpacked_size + 64 * 1024**2:
        raise ReleaseError("安裝目錄可用空間不足")
    target = folder / f"{release.sha256}.zip"
    temporary = target.with_suffix(".part")
    digest, received = hashlib.sha256(), 0
    try:
        with (
            opener(
                https_url(release.url),
                timeout=30,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as response,
            temporary.open("wb") as fh,
        ):
            https_url(response.geturl())
            while True:
                if cancelled():
                    raise ReleaseError("已取消下載")
                chunk = response.read(min(1024 * 1024, release.size - received + 1))
                if not chunk:
                    break
                received += len(chunk)
                if received > release.size:
                    raise ReleaseError("下載大小與發布資訊不符")
                fh.write(chunk)
                digest.update(chunk)
                progress(received, release.size)
        if received != release.size or digest.hexdigest() != release.sha256:
            raise ReleaseError("下載檔案不完整或已損壞")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
