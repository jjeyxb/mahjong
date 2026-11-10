"""Bounded ZIP extraction that preserves in-bundle macOS symbolic links."""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

from mia.distribution.release import ReleaseError


def extract(archive: Path, destination: Path, expected: str, maximum: int) -> Path:
    if destination.exists():
        raise ReleaseError("解壓目的地必須是新目錄")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 100_000 or sum(m.file_size for m in members) > maximum:
            raise ReleaseError("解壓內容超過發布資訊的限制")
        names: set[str] = set()
        links: dict[str, str] = {}
        for member in members:
            name = member.filename.rstrip("/")
            parts = PurePosixPath(name).parts
            if (
                not parts
                or parts[0] != expected
                or ".." in parts
                or "\\" in name
                or ":" in name
                or "\x00" in name
                or PurePosixPath(name).is_absolute()
                or name in names
            ):
                raise ReleaseError("封裝含不合法或重複路徑")
            names.add(name)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                if name == expected:
                    raise ReleaseError("封裝根目錄不能是符號連結")
                if member.file_size > 4096:
                    raise ReleaseError("不合法的符號連結")
                target = zipped.read(member).decode("utf-8")
                if "\\" in target or ":" in target or PurePosixPath(target).is_absolute():
                    raise ReleaseError("符號連結必須留在封裝內")
                links[name] = target
            elif mode and stat.S_IFMT(mode) not in (0, stat.S_IFDIR, stat.S_IFREG):
                raise ReleaseError("不支援的封裝檔案類型")
        for name in names:
            if any(str(parent) in links for parent in PurePosixPath(name).parents):
                raise ReleaseError("不能寫入符號連結底下的路徑")
        destination.mkdir(parents=True)
        for member in members:
            name = member.filename.rstrip("/")
            if name in links:
                continue
            path = destination / name
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as src, path.open("xb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            path.chmod((member.external_attr >> 16) & 0o777 or 0o644)
        if links and os.name == "nt":
            raise ReleaseError("Windows 加速包不可含符號連結")
        for name, target in links.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        root = (destination / expected).resolve()
        for name in links:
            try:
                if not (destination / name).resolve().is_relative_to(root):
                    raise ReleaseError("符號連結越出封裝範圍")
            except (RuntimeError, OSError) as exc:
                raise ReleaseError("無法解析符號連結") from exc
    return destination / expected
