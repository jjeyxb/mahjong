"""打包版(Windows)的 YOLO GPU 加速:照安裝檔附的清單,從官方來源下載並驗證。

打包版的 YOLO 推論程序(``MIA-yolo.exe``)帶的是 CPU 版 torch。GPU 版一種就 1.4~2.9 GB,
而且看玩家是哪一家的顯示卡,所以不放進安裝檔,要用時才下載:

* 清單在安裝檔裡(``resources/gpu/``,由 ``tools/lock_gpu.py`` 產生):python.org 的 embeddable
  Python 3.12、PyPI 的 pip,與每一種顯示卡要的全部套件 —— 每個檔案都有官方網址與 SHA-256。
* 下載到 ``components/gpu/`` 底下的暫存資料夾,建一個獨立的 Python,用 pip 照清單裝
  (``--require-hashes --no-deps --no-index``:只裝清單上的檔案,內容不符就失敗,不會自己挑版本)。
* 用新環境實際跑一次推論程序的自我檢查(載入模型、在顯示卡上推論一幀),通過才搬到正式位置
  並設為使用中。**任何一步失敗都只刪暫存資料夾**,原本能用的(CPU 或舊的 GPU 元件)不受影響。

推論時用這個 Python 執行打包版附的原始碼(``MIA-yolo`` 旁邊的 ``yolo.py`` / ``worker.py`` /
``agent/``),跟 ``MIA-yolo.exe`` 是同一份程式,只是 torch 換成 GPU 版。

資料夾::

    components/gpu/
        active.json              {"backend": "rocm", "id": "…"} 目前使用中的那一份
        rocm-1a2b3c4d/           embeddable Python + Lib/site-packages(id 由清單內容算出)
        .stage-…/                安裝中的暫存;失敗或取消就整個刪掉

清單換了(MIA 更新後 torch 版本不同)id 就不同,舊的那份不再被當成可用 —— 推論程序退回 CPU,
設定頁再選一次 GPU 就會照新清單重裝。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mia.distribution.layout import LAYOUT, Layout
from mia.distribution.release import https_url
from mia.i18n import tr

__all__ = ["available", "install", "worker_command"]

_MANIFEST = "windows-x64.json"
_PYTHON_FLAGS = ["-I", "-B", "-X", "utf8"]
"""``-I``:不讀使用者電腦上的 PYTHONPATH 等設定,也不看使用者自己的 site-packages ——
這個環境只能有清單上的東西。``-B``:不在 MIA 的程式資料夾寫 ``__pycache__``(可能沒有寫入權限,
也會變成解除安裝時的殘留)。``-X utf8``:pip 的輸出用 UTF-8,MIA 才解得開。"""

#: 打包版 MIA-yolo 旁邊附的原始碼入口(``tools/package.py`` 放的)。
_ENTRY = "yolo.py"


class _Failed(Exception):
    """安裝失敗,訊息直接給使用者看。"""


def _folder(layout: Layout) -> Path:
    return layout.components / "gpu"


def _manifest(layout: Layout) -> dict[str, Any] | None:
    path = layout.resources / "gpu" / _MANIFEST
    if not layout.packaged or sys.platform != "win32" or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _entry(layout: Layout) -> Path:
    return layout.worker("yolo").parent / "_internal" / _ENTRY


def available(layout: Layout = LAYOUT) -> bool:
    """這個 MIA 能不能自己下載 GPU 加速:Windows 打包版、安裝檔附了清單與原始碼入口。"""
    try:
        return _manifest(layout) is not None and _entry(layout).is_file()
    except (OSError, ValueError):
        return False


def download_size(backend: str, layout: Layout = LAYOUT) -> int | None:
    """這種顯示卡要下載多少位元組(清單算好的)。沒有清單就是 ``None``。"""
    try:
        manifest = _manifest(layout)
    except (OSError, ValueError):
        return None
    if manifest is None or backend not in manifest.get("backends", {}):
        return None
    return int(manifest["backends"][backend]["size"]) + int(manifest["python"]["size"]) + int(
        manifest["pip"]["size"]
    )


def _component_id(layout: Layout, manifest: dict[str, Any], backend: str) -> str:
    """清單內容決定的 id。清單任何一個字變了,舊的安裝就不算數。"""
    lock = (layout.resources / "gpu" / manifest["backends"][backend]["lock"]).read_bytes()
    digest = hashlib.sha256()
    for part in (manifest["python"]["sha256"], manifest["pip"]["sha256"]):
        digest.update(part.encode())
    digest.update(lock)
    return digest.hexdigest()[:8]


def worker_command(layout: Layout = LAYOUT) -> list[str] | None:
    """使用中的 GPU 元件的推論程序指令;沒有(或已經跟這版 MIA 的清單對不上)就是 ``None``。"""
    try:
        manifest = _manifest(layout)
        if manifest is None:
            return None
        active = json.loads((_folder(layout) / "active.json").read_text(encoding="utf-8"))
        backend = active["backend"]
        if backend not in manifest["backends"] or active["id"] != _component_id(
            layout, manifest, backend
        ):
            return None
        python = _folder(layout) / f"{backend}-{active['id']}" / "python.exe"
        entry = _entry(layout)
        if not python.is_file() or not entry.is_file():
            return None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return [str(python), *_PYTHON_FLAGS, "-u", str(entry)]


# ---------------------------------------------------------------------------- 安裝


def install(
    backend: str,
    on_line: Callable[[str], None],
    cancelled: threading.Event,
    layout: Layout = LAYOUT,
) -> tuple[bool, str]:
    """下載、安裝、檢查,通過才設為使用中。回 ``(成功與否, 最後一句說明)``。

    隨時可以取消:裝的東西都在暫存資料夾,取消或失敗就整個刪掉。
    """
    manifest = _manifest(layout)
    if manifest is None or backend not in manifest.get("backends", {}):
        return False, tr("這個版本的 MIA 沒有附 {backend} 的下載清單。", backend=backend)
    root = _folder(layout)
    stage = root / f".stage-{uuid.uuid4().hex[:8]}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        _clean_stages(root)
        component = _component_id(layout, manifest, backend)
        need = (download_size(backend, layout) or 0) * 3 + 1024**3
        if shutil.disk_usage(root).free < need:
            raise _Failed(tr("磁碟空間不足:需要約 {size:.1f} GB。", size=need / 1024**3))
        stage.mkdir()
        _install_python(manifest, stage, on_line, cancelled, layout)
        lock = layout.resources / "gpu" / manifest["backends"][backend]["lock"]
        _pip_install(manifest, stage, lock, on_line, cancelled)
        _self_check(stage, backend, on_line, layout)
        target = root / f"{backend}-{component}"
        if target.exists():
            # 同一份清單裝過但沒通過(或被手動改過):換成剛剛驗證過的這份。
            shutil.rmtree(target)
        stage.rename(target)
        _write_json(root / "active.json", {"backend": backend, "id": component})
        _remove_others(root, keep=target)
        return True, tr("GPU 加速已安裝。")
    except _Failed as exc:
        return False, str(exc)
    except OSError as exc:
        return False, tr("安裝失敗:{error}", error=exc)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _install_python(
    manifest: dict[str, Any],
    stage: Path,
    on_line: Callable[[str], None],
    cancelled: threading.Event,
    layout: Layout,
) -> None:
    """embeddable Python 解到暫存資料夾,讓它看得到自己的 site-packages。"""
    archive = _download(manifest["python"], stage / "python.zip", "Python 3.12", on_line, cancelled)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise _Failed(tr("下載的 Python 內容不正確。"))
        bundle.extractall(stage)
    archive.unlink()
    # embeddable 版靠 ._pth 決定 sys.path,預設不含 site-packages,pip 裝的東西會找不到。
    pth = next(stage.glob("python3*._pth"), None)
    if pth is None:
        raise _Failed(tr("下載的 Python 內容不正確。"))
    zipped = pth.with_suffix(".zip").name
    pth.write_text(f"{zipped}\n.\nLib\\site-packages\nimport site\n", encoding="ascii")
    # torch 的 DLL 需要 Microsoft C++ 執行階段(msvcp140.dll 等)。embeddable 版只帶了
    # vcruntime140*.dll;沒裝 Visual C++ 可轉散發套件的電腦會載不起 torch。打包版的
    # MIA-yolo 本來就帶著這幾個檔(PyInstaller 收的),複製一份過來。
    bundled = layout.worker("yolo").parent / "_internal"
    for dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"):
        if (bundled / dll).is_file() and not (stage / dll).exists():
            shutil.copy2(bundled / dll, stage / dll)


#: pip 的 ``--progress-bar raw`` 每下載一段印一行。
_PROGRESS = re.compile(r"Progress (\d+) of (\d+)")
_DOWNLOADING = re.compile(r"Downloading (\S+\.(?:whl|tar\.gz|zip))")


def _pip_install(
    manifest: dict[str, Any],
    stage: Path,
    lock: Path,
    on_line: Callable[[str], None],
    cancelled: threading.Event,
) -> None:
    """照清單裝。先單獨裝 setuptools:AMD 的 ``rocm`` 只有原始碼版,建置時要它,
    而 ``--no-build-isolation`` 是用這個環境裡的 setuptools(不准 pip 另外上網抓)。"""
    pip = _download(manifest["pip"], stage / "pip.whl", "pip", on_line, cancelled)
    lines = [line for line in lock.read_text(encoding="ascii").splitlines()
             if line and not line.startswith("#")]
    tools = stage / "setuptools.txt"
    tools.write_text("\n".join(line for line in lines if line.startswith("setuptools ")) + "\n",
                     encoding="ascii")
    temp = stage / "tmp"
    temp.mkdir()
    env = {
        **os.environ,
        # pip 的下載暫存放暫存資料夾裡:取消或失敗時一起刪,不留在使用者的 TEMP。
        # 三個都要設:Python 先看 TMPDIR,而打包版 MIA 啟動時把它設成自己的快取資料夾。
        "TMPDIR": str(temp),
        "TMP": str(temp),
        "TEMP": str(temp),
        "PIP_NO_CACHE_DIR": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    base = [str(stage / "python.exe"), *_PYTHON_FLAGS, str(pip / "pip"), "install",
            "--no-index", "--no-deps", "--require-hashes", "--progress-bar", "raw",
            "--no-warn-script-location"]
    _run([*base, "-r", str(tools)], env, on_line, cancelled)
    _run([*base, "--no-build-isolation", "-r", str(lock)], env, on_line, cancelled)
    pip.unlink()


def _run(
    command: list[str],
    env: dict[str, str],
    on_line: Callable[[str], None],
    cancelled: threading.Event,
) -> None:
    """跑 pip,把輸出換成看得懂的進度。取消就連子程序一起停。"""
    from mia.yolo.accel import _kill_tree

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise _Failed(tr("無法執行 pip:{error}", error=exc)) from exc
    assert process.stdout is not None
    watcher = threading.Thread(target=_stop_on_cancel, args=(process, cancelled, _kill_tree),
                               daemon=True)
    watcher.start()
    current = ""
    last = ""
    shown = 0.0
    for raw in iter(process.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if match := _DOWNLOADING.search(line):
            current = match.group(1)
        if match := _PROGRESS.search(line):
            done, total = int(match.group(1)), int(match.group(2))
            if time.monotonic() - shown >= 0.5 or done == total:
                shown = time.monotonic()
                on_line(tr("下載 {name}:{done:.0f} / {total:.0f} MB",
                           name=current, done=done / 1024**2, total=total / 1024**2))
            continue
        last = line
        on_line(line)
    code = process.wait()
    if cancelled.is_set():
        raise _Failed(tr("已取消，沒有更動任何設定。"))
    if code != 0:
        raise _Failed(tr("pip 安裝失敗:{error}", error=last))


def _stop_on_cancel(
    process: subprocess.Popen[bytes],
    cancelled: threading.Event,
    kill: Callable[[subprocess.Popen[bytes]], None],
) -> None:
    while process.poll() is None:
        if cancelled.wait(0.3):
            kill(process)
            return


def _download(
    item: dict[str, Any],
    target: Path,
    label: str,
    on_line: Callable[[str], None],
    cancelled: threading.Event,
) -> Path:
    """下載清單上的一個檔案,大小與 SHA-256 都要對。"""
    import certifi

    size, expected = int(item["size"]), str(item["sha256"])
    digest, received, shown = hashlib.sha256(), 0, 0.0
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(https_url(item["url"]), headers={"User-Agent": "MIA"})
    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response, \
                target.open("wb") as fh:
            https_url(response.geturl())
            while chunk := response.read(1024 * 1024):
                if cancelled.is_set():
                    raise _Failed(tr("已取消，沒有更動任何設定。"))
                received += len(chunk)
                if received > size:
                    break
                fh.write(chunk)
                digest.update(chunk)
                if time.monotonic() - shown >= 0.5:
                    shown = time.monotonic()
                    on_line(tr("下載 {name}:{done:.0f} / {total:.0f} MB", name=label,
                               done=received / 1024**2, total=size / 1024**2))
    except OSError as exc:
        raise _Failed(tr("下載 {name} 失敗:{error}", name=label, error=exc)) from exc
    if received != size or digest.hexdigest() != expected:
        raise _Failed(tr("下載的 {name} 跟清單不符(檔案不完整或被改過)。", name=label))
    return target


def _self_check(stage: Path, backend: str, on_line: Callable[[str], None], layout: Layout) -> None:
    """用新環境跑推論程序的自我檢查:載入四個模型,在顯示卡上推論一幀。

    第一次在 AMD 卡上跑要現場編譯 GPU 核心,可能要幾分鐘。
    """
    on_line(tr("檢查顯示卡能不能跑 YOLO(第一次可能要幾分鐘)…"))
    command = [str(stage / "python.exe"), *_PYTHON_FLAGS, str(_entry(layout)), "--self-check"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
            env={**os.environ, "MJ_DEVICE": "auto"},
            creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise _Failed(tr("顯示卡檢查逾時。")) from exc
    report: dict[str, Any] = {}
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            report = json.loads(line)
            break
        except ValueError:
            continue
    if not report.get("ok") or report.get("backend") != backend:
        detail = (result.stderr.strip().splitlines() or [""])[-1]
        raise _Failed(tr("裝好了，但顯示卡跑不動 YOLO({backend}):{error}",
                         backend=report.get("backend") or "?", error=detail[-300:]))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _clean_stages(root: Path) -> None:
    """上次被強制結束留下的暫存資料夾(例如安裝到一半關機)。"""
    for stale in root.glob(".stage-*"):
        shutil.rmtree(stale, ignore_errors=True)


def _remove_others(root: Path, *, keep: Path) -> None:
    """換了一份元件之後,舊的就用不到了(一份好幾 GB)。刪不掉(檔案被占用)就下次再說。"""
    for folder in root.iterdir():
        if folder.is_dir() and folder != keep and not folder.name.startswith("."):
            shutil.rmtree(folder, ignore_errors=True)
