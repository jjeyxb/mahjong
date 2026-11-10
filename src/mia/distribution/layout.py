"""Resolve owned files without relying on the build machine or current directory."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64"
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x64"
    raise RuntimeError(f"尚未支援的發布平台: {sys.platform}/{machine}")


@dataclass(frozen=True)
class Layout:
    resources: Path
    install: Path
    packaged: bool = False

    @classmethod
    def detect(cls) -> Layout:
        if not getattr(sys, "frozen", False):
            root = Path(__file__).resolve().parents[3]
            return cls(root, root)
        executable = Path(sys.executable).resolve()
        bundle = next((p for p in executable.parents if p.suffix == ".app"), None)
        install = bundle.parent if bundle else executable.parent.parent
        resources = Path(vars(sys)["_MEIPASS"]) / "resources"
        return cls(resources.resolve(), install.resolve(), True)

    @property
    def data(self) -> Path:
        return self.install / ("userdata" if self.packaged else "data")

    @property
    def config(self) -> Path:
        return self.data / "config" if self.packaged else self.resources / "config"

    @property
    def logs(self) -> Path:
        return self.data / "logs" if self.packaged else self.install / "logs"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def components(self) -> Path:
        return self.install / "components"

    @property
    def updates(self) -> Path:
        return self.install / "updates"

    def worker(self, role: str) -> Path:
        if role not in ("mortal", "yolo", "updater"):
            raise ValueError(role)
        suffix = ".exe" if sys.platform == "win32" else ""
        return self.resources / "engines" / role / f"MIA-{role}{suffix}"

    def prepare(self) -> None:
        for p in (self.data, self.config, self.logs, self.cache, self.updates):
            p.mkdir(parents=True, exist_ok=True)
        # Private to this process and its children; never modify global PATH.
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(self.resources / "browsers")
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(self.cache)
        os.environ["XDG_CACHE_HOME"] = str(self.cache)
        os.environ["TORCH_HOME"] = str(self.cache / "torch")
        os.environ["YOLO_CONFIG_DIR"] = str(self.data / "yolo")
        os.environ["MIA_YOLO_MODELS"] = str(self.resources / "models" / "yolo")
        os.environ["MIA_RESOURCE_ROOT"] = str(self.resources)


LAYOUT = Layout.detect()
