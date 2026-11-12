"""產生打包版「YOLO GPU 加速」的下載清單:每個檔案的官方網址、SHA-256 與大小。

打包版不帶 GPU 版的 torch(一種就 1.4~2.9 GB,而且要看玩家是哪一家的顯示卡)。玩家在設定裡
選 GPU 時,MIA 照這裡產生的清單從**官方來源**下載,自己不架任何下載站:

* Python 本體:python.org 的 Windows embeddable 版(3.12,GPU 版 torch 只有 3.12 的)。
* torch / torchvision:NVIDIA 與 Intel 用 download.pytorch.org,AMD 用 repo.radeon.com。
* 其餘套件(ultralytics、numpy……):PyPI。

清單把每個檔案的版本與雜湊都釘死,玩家電腦上的 pip 只照清單裝(``--require-hashes --no-deps``),
不會自己挑版本,也不會換成「最新版」。任何一個檔案的內容跟這裡記的不一樣,安裝就失敗。

用法(在發布用的 Windows 電腦上,需要 Python 3.12)::

    python tools/lock_gpu.py --python downloads/python-3.12.10/python.exe

結果寫到 ``packaging/gpu/``,打包時(``tools/package.py``)整個資料夾放進安裝檔。
換 torch 版本時改 ``engines/yolo/requirements-<backend>.txt``,再重跑這支。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "engines/yolo"
OUT = ROOT / "packaging/gpu"

#: python.org 的 Windows embeddable 版。3.12.10 是 3.12 最後一個有官方 Windows 安裝檔的版本;
#: 3.12.0 不行(PyInstaller 那邊撞過的推導式 bug,見 packaging/windows/README.md)。
PYTHON_EMBED = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"

#: 玩家電腦上跑安裝用的 pip。挑發布超過兩週的版本(2026-10-20),不追最新。
PIP = "pip==26.2.2"

BACKENDS = ("cuda", "rocm", "xpu")

#: 唯一允許的原始碼套件。AMD 的 ``rocm`` 只以原始碼發布(純 Python,裝的時候要 setuptools 建置);
#: 其他套件一律要 wheel —— 在玩家電腦上編譯 C 擴充是不會成功的。
SDIST_ALLOWED = {"rocm"}


def _requirements(backend: str) -> list[str]:
    """``requirements-<backend>.txt`` 的 torch 加上 ``requirements.txt`` 裡 torch 以外的套件。

    跟開發版 ``tools/setup_yolo.py --gpu`` 裝的是同一組,只是一次解完。
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from setup_yolo import _is_torch

    lines = []
    for name, keep in ((f"requirements-{backend}.txt", None), ("requirements.txt", _is_torch)):
        for line in (YOLO / name).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not (keep and keep(line)):
                lines.append(line)
    return lines


def _tested_versions(python: Path) -> list[str]:
    """已經實測過的 YOLO 環境裡各套件的版本,當作解版本時的限制。

    ``requirements.txt`` 只釘了幾個主要套件,其餘(pillow、polars……)pip 會挑當天最新的。
    照實測過的環境釘住,清單才是「真的跑過的那一組」,也不會因為哪天重跑就換掉一半。
    torch 這幾個不從這裡拿:它們由 ``requirements-<backend>.txt`` 決定。
    """
    frozen = subprocess.run(
        [str(python), "-I", "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    skip = re.compile(r"^(pip|torch|torchvision|rocm[-\w]*)==", re.IGNORECASE)
    return [line for line in frozen if "==" in line and not skip.match(line)]


def _resolve(
    python: Path, requirements: list[str], work: Path, constraints: Sequence[str] = ()
) -> list[dict]:
    """讓 pip 解出完整的套件清單(不安裝)。回傳 ``pip --report`` 的 ``install`` 段。"""
    request = work / "request.txt"
    request.write_text("\n".join(requirements) + "\n", encoding="ascii")
    limits = work / "constraints.txt"
    limits.write_text("\n".join(constraints) + "\n", encoding="ascii")
    report = work / "report.json"
    env = {
        **os.environ,
        "PIP_NO_CACHE_DIR": "1",
        # 不讀使用者自己的 pip 設定:那裡可能指向別的套件來源。
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_INDEX_URL": "https://pypi.org/simple",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--dry-run", "--ignore-installed",
         "--quiet", "--prefer-binary", "--report", str(report), "-r", str(request),
         "-c", str(limits)],
        env=env,
        check=True,
    )
    return json.loads(report.read_text(encoding="utf-8"))["install"]


def _entry(item: dict) -> tuple[str, str, str]:
    """報告裡的一個套件 → ``(名稱, 網址, sha256)``。缺雜湊或是不該出現的原始碼套件就停。"""
    name = item["metadata"]["name"].lower()
    info = item["download_info"]
    url = info["url"].split("#", 1)[0]
    archive = info.get("archive_info", {})
    sha = archive.get("hashes", {}).get("sha256")
    if not sha and str(archive.get("hash", "")).startswith("sha256="):
        sha = archive["hash"].split("=", 1)[1]
    if not sha or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise SystemExit(f"{name} 沒有 SHA-256,不能放進清單:{url}")
    if not url.endswith(".whl") and name not in SDIST_ALLOWED:
        raise SystemExit(f"{name} 只有原始碼版本,玩家電腦上裝不起來:{url}")
    if not url.startswith("https://"):
        raise SystemExit(f"{name} 不是 https 來源:{url}")
    return name, url, sha


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "mia-lock"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def _sha256_of(url: str, work: Path) -> tuple[str, int]:
    target = work / url.rsplit("/", 1)[1]
    request = urllib.request.Request(url, headers={"User-Agent": "mia-lock"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as fh:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            fh.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", type=Path, required=True, help="Windows x64 的 Python 3.12")
    parser.add_argument("--backend", choices=BACKENDS, action="append",
                        help="只重產這幾種(預設全部)")
    parser.add_argument("--tested", type=Path, default=YOLO / ".venv/Scripts/python.exe",
                        help="實測過的 YOLO 環境的 Python,其餘套件照它的版本釘")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("清單裡的 wheel 是照執行這支的電腦挑的,請在 Windows x64 上跑")
    version = subprocess.run(
        [str(args.python), "-I", "-c", "import sys, platform;"
         "print('%d.%d' % sys.version_info[:2], platform.machine())"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if version != ["3.12", "AMD64"]:
        parser.error(f"--python 要是 Windows x64 的 Python 3.12,這個是 {' '.join(version)}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "windows-x64.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {"backends": {}}
    )
    if not args.tested.is_file():
        parser.error(f"找不到 {args.tested}:先用 tools/setup_yolo.py 建好並實測 YOLO 環境")
    tested = _tested_versions(args.tested)
    with tempfile.TemporaryDirectory(prefix="mia-lock-") as temp:
        work = Path(temp)
        sha, size = _sha256_of(PYTHON_EMBED, work)
        manifest["python"] = {"url": PYTHON_EMBED, "sha256": sha, "size": size}
        _, pip_url, pip_sha = _entry(_resolve(args.python, [PIP], work)[0])
        manifest["pip"] = {"url": pip_url, "sha256": pip_sha, "size": _size(pip_url)}
        for backend in args.backend or BACKENDS:
            print(f"== {backend}", flush=True)
            resolved = _resolve(args.python, _requirements(backend), work, tested)
            entries = sorted(_entry(item) for item in resolved)
            names = [name for name, _, _ in entries]
            if "setuptools" not in names:
                raise SystemExit("清單裡沒有 setuptools:rocm 原始碼套件要靠它建置")
            total = sum(_size(url) for _, url, _ in entries)
            lock = f"windows-x64-{backend}.txt"
            # 只寫 ASCII:pip 讀需求檔的編碼問題見 engines/yolo/requirements-rocm.txt 開頭。
            lines = [
                f"# Generated by tools/lock_gpu.py on {date.today().isoformat()}. Do not edit.",
                *(f"{name} @ {url} --hash=sha256:{sha}" for name, url, sha in entries),
            ]
            (OUT / lock).write_text("\n".join(lines) + "\n", encoding="ascii")
            manifest["backends"][backend] = {"lock": lock, "size": total}
            print(f"   {len(entries)} 個檔案,{total / 1024**3:.2f} GB", flush=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(text, encoding="utf-8")
    print(f"已寫入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
