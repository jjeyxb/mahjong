from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _package():
    spec = importlib.util.spec_from_file_location("package_tool", ROOT / "tools/package.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_assembly_links_instead_of_writing_new_files(tmp_path):
    """Windows 組裝寫好幾 GB 新檔案會讓整台電腦的記憶體被吃光(見 copy 的說明),改用硬連結。"""
    package = _package()
    source = tmp_path / "frozen/MIA"
    (source / "_internal").mkdir(parents=True)
    (source / "_internal/torch.dll").write_bytes(b"x" * 10)
    (source / "MIA.exe").write_bytes(b"exe")
    target = tmp_path / "stage/program"
    package.copy(source, target, link=True)
    assert os.path.samefile(source / "_internal/torch.dll", target / "_internal/torch.dll")
    single = tmp_path / "stage/LICENSE"
    (tmp_path / "LICENSE").write_text("license")
    package.copy(tmp_path / "LICENSE", single, link=True)
    package.copy(tmp_path / "LICENSE", single, link=True)  # 已存在也不會失敗
    assert os.path.samefile(tmp_path / "LICENSE", single)


def test_mac_assembly_still_copies(tmp_path):
    """Mac 組完要改 Info.plist 再簽章;硬連結會連 dist/frozen 的原檔一起改。"""
    package = _package()
    source = tmp_path / "frozen/MIA.app/Contents"
    source.mkdir(parents=True)
    (source / "Info.plist").write_text("plist")
    target = tmp_path / "stage/MIA.app"
    package.copy(tmp_path / "frozen/MIA.app", target)
    assert not os.path.samefile(source / "Info.plist", target / "Contents/Info.plist")
