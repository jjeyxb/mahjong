"""Resolve owned files without relying on the build machine or current directory."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

#: Mac 上 Application Support、Caches、Logs 底下的資料夾名稱。
APP_NAME = "MIA"


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64"
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x64"
    raise RuntimeError(f"尚未支援的發布平台: {sys.platform}/{machine}")


@dataclass(frozen=True)
class Layout:
    """打包版與原始碼版各自的檔案位置。

    ``install`` 是程式所在的資料夾。``home`` 是 MIA 自己的資料根目錄,裝著設定、
    錄影、更新暫存與 install-info.json:Windows 與原始碼版就是 ``install``,
    Mac 打包版是 ``~/Library/Application Support/MIA``。Mac 的 App 照一般習慣拖進
    「應用程式」,而那裡不該放使用者資料,換新的 App 時也不能把資料一起換掉。
    """

    resources: Path
    install: Path
    packaged: bool = False
    #: Mac 打包版的 MIA.app 本身。更新時換掉的就是它。
    bundle: Path | None = None
    home: Path | None = None
    #: Mac 打包版的 ``~/Library``。快取與 log 照 Mac 慣例放在 Caches 與 Logs。
    library: Path | None = None

    @classmethod
    def detect(cls) -> Layout:
        if not getattr(sys, "frozen", False):
            root = Path(__file__).resolve().parents[3]
            return cls(root, root)
        executable = Path(sys.executable).resolve()
        bundle = next((p for p in executable.parents if p.suffix == ".app"), None)
        install = bundle.parent if bundle else executable.parent.parent
        resources = (Path(vars(sys)["_MEIPASS"]) / "resources").resolve()
        if bundle is None:
            return cls(resources, install.resolve(), True)
        library = Path.home() / "Library"
        return cls(
            resources,
            install.resolve(),
            True,
            bundle=bundle.resolve(),
            home=library / "Application Support" / APP_NAME,
            library=library,
        )

    @property
    def root(self) -> Path:
        """MIA 自有資料的根目錄:更新暫存、元件與 install-info.json 都在這裡。"""
        return self.home if self.home is not None else self.install

    @property
    def data(self) -> Path:
        if self.home is not None:
            return self.home
        return self.install / ("userdata" if self.packaged else "data")

    @property
    def config(self) -> Path:
        return self.data / "config" if self.packaged else self.resources / "config"

    @property
    def logs(self) -> Path:
        if self.library is not None:
            return self.library / "Logs" / APP_NAME
        return self.data / "logs" if self.packaged else self.install / "logs"

    @property
    def cache(self) -> Path:
        if self.library is not None:
            return self.library / "Caches" / APP_NAME
        return self.data / "cache"

    @property
    def components(self) -> Path:
        return self.root / "components"

    @property
    def updates(self) -> Path:
        return self.root / "updates"

    @property
    def install_info(self) -> Path:
        return self.root / "install-info.json"

    def owned_data(self) -> tuple[Path, ...]:
        """「清除所有資料」要刪的資料夾。原始碼版不刪任何東西:那裡的 data/ 是開發用的。

        Windows 只刪 userdata/:程式與 install-info.json 由安裝程式管理。
        """
        if not self.packaged:
            return ()
        if self.home is not None:
            return (self.home, self.cache, self.logs)
        return (self.data,)

    def update_blocker(self) -> str | None:
        """這份 App 能不能就地換成新版。不能的話回一句給使用者看的原因。"""
        if self.bundle is None:
            return None
        if "AppTranslocation" in self.bundle.parts or not os.access(self.bundle.parent, os.W_OK):
            return "請先把 MIA.app 拖進「應用程式」資料夾，再從那裡開啟，才能自動更新"
        return None

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
