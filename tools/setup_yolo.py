"""建 YOLO 的獨立推論環境 ``engines/yolo/.venv``,並把權重放進 ``models/yolo``。

    .venv/bin/python tools/setup_yolo.py --python python3.12            # macOS
    .venv\\Scripts\\python.exe tools\\setup_yolo.py                        # Windows,CPU
    .venv\\Scripts\\python.exe tools\\setup_yolo.py --gpu rocm             # Windows,AMD 顯示卡

``--gpu rocm`` 把環境裡的 torch 換成 AMD 的 ROCm 版,YOLO 就會跑在顯示卡上:RX 9070 實測
每幀約 150~200 ms,CPU 版約 650 ms 以上(``docs/yolo.md``)。**對已經裝好的 CPU 環境直接再跑
一次就好** —— 換的是環境裡的套件,不搬資料夾,MIA 找環境的路徑不變。反過來 ``--gpu cpu``
會換回 CPU 版,並移除 ROCm 那幾 GB 的執行環境。

macOS 不需要這個選項:Apple Silicon 的 torch 本來就會用 MPS。NVIDIA 顯示卡還沒有實測過,
所以沒有提供。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "engines/yolo"

#: ROCm 版 torch 的版本。要與 ``engines/yolo/requirements-rocm.txt`` 一致。
ROCM_PINS = ("torch==2.9.1+rocm7.2.1", "torchvision==0.24.1+rocm7.2.1")

#: 換回 CPU 版時要一起拿掉的 ROCm 執行環境。torch 換掉之後它們沒有人用,而且佔好幾 GB。
ROCM_RUNTIME = ("rocm", "rocm-sdk-core", "rocm-sdk-libraries-custom")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python",
        default=None,
        help="用哪個 Python 3.12 建環境。預設 Windows 用 `py -3.12`,其他平台用 python3.12",
    )
    parser.add_argument(
        "--gpu",
        choices=("cpu", "rocm"),
        default="cpu",
        help="rocm:改用 AMD 顯示卡(僅 Windows + Python 3.12)。預設 cpu",
    )
    parser.add_argument(
        "--models", type=Path, default=ROOT / "downloads/majongsoul ai agent_mac/app/Model"
    )
    args = parser.parse_args()

    environment = BASE / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not environment.exists():
        subprocess.run([*_base_python(args.python), "-m", "venv", str(environment)], check=True)

    if args.gpu == "rocm":
        _check_rocm_target(python)
        _install_rocm(python)
    else:
        # 沒裝過 ROCm 的環境上這行只會說「沒裝,略過」,所以不檢查結束碼。
        _pip(python, "uninstall", "-y", *ROCM_RUNTIME, check=False)
        _pip(python, "install", "-r", str(BASE / "requirements.txt"))
        _hint_gpu()

    _copy_models(args.models)
    _report(python, args.gpu)


def _base_python(chosen: str | None) -> list[str]:
    """建環境用的 Python。Windows 上通常沒有 ``python3.12`` 這個指令,用 py 啟動器。"""
    if chosen:
        return [chosen]
    return ["py", "-3.12"] if sys.platform == "win32" else ["python3.12"]


def _pip(python: Path, *arguments: str, check: bool = True) -> None:
    """跑一次環境裡的 pip,**不留快取**。

    用環境變數 ``PIP_NO_CACHE_DIR`` 而不是 ``--no-cache-dir`` 參數:參數只管這一個 pip,
    而 ``rocm`` 是原始碼套件,pip 會另開子程序去建置,那個子程序看不到參數,照樣往
    使用者目錄(``%LOCALAPPDATA%\\pip\\Cache``)寫快取 —— 實測寫了 1.1 MB。環境變數會被
    子程序繼承。不留快取是為了「刪掉 engines/yolo/.venv 就清乾淨」。
    """
    env = {**os.environ, "PIP_NO_CACHE_DIR": "1"}
    result = subprocess.run([str(python), "-m", "pip", *arguments], env=env, check=False)
    if result.returncode != 0 and check:
        raise SystemExit(f"pip {arguments[0]} 失敗(結束碼 {result.returncode})")


def _check_rocm_target(python: Path) -> None:
    """ROCm 的 wheel 只有 Windows、cp312 這一種。不對就在下載幾 GB 之前停下來。"""
    if sys.platform != "win32":
        raise SystemExit(
            "--gpu rocm 只支援 Windows。macOS 的 torch 本來就會用 MPS,不需要這個選項。"
        )
    version = subprocess.run(
        [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if version != "3.12":
        raise SystemExit(
            f"engines/yolo/.venv 是 Python {version},ROCm 版 torch 只有 3.12 的。"
            "請刪掉 engines/yolo/.venv 後重跑,讓它用 Python 3.12 重建。"
        )


def _install_rocm(python: Path) -> None:
    """兩步:先裝 ROCm 版 torch,再照 requirements.txt 裝其餘套件,**但不准動 torch**。

    第二步不能直接 ``-r requirements.txt``:那裡釘的是 PyPI 的 CPU 版 torch 2.5.1,pip 會把
    剛裝好的 ROCm 版換回去。所以 torch 那兩行跳過,其餘照裝,再用 constraints 鎖住 ROCm
    版 —— ultralytics 若要求更新的 torch,pip 會直接報衝突,而不是安靜地換掉。
    """
    print("== 1/2 ROCm 版 torch 與 AMD 執行環境(約數 GB,第一次要下載一陣子)")
    _pip(python, "install", "-r", str(BASE / "requirements-rocm.txt"))

    print("== 2/2 其餘套件(照 requirements.txt,torch 鎖在 ROCm 版)")
    others = [
        line
        for line in (BASE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not _is_torch(line)
    ]
    cache = BASE / ".cache"
    cache.mkdir(exist_ok=True)
    constraints = cache / "rocm-constraints.txt"
    # 只寫 ASCII:這個檔也是 pip 讀的,理由見 requirements-rocm.txt 開頭。
    constraints.write_text("\n".join(ROCM_PINS) + "\n", encoding="ascii")
    _pip(python, "install", "-c", str(constraints), *others)


def _is_torch(line: str) -> bool:
    name = re.split(r"[\s=<>!~;\[@]", line.strip(), maxsplit=1)[0].lower()
    return name in ("torch", "torchvision")


def _hint_gpu() -> None:
    """Windows 上看到 AMD 顯示卡就提一句。只提示、不自動換:要下載幾 GB,也不是每張卡都支援。"""
    if sys.platform != "win32":
        return
    try:
        query = "(Get-CimInstance Win32_VideoController).Name"
        names = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return
    if "Radeon" in names:
        print("偵測到 AMD 顯示卡。加上 --gpu rocm 重跑,YOLO 可以改跑在顯示卡上(約快 4 倍)。")


def _copy_models(source: Path) -> None:
    models = ROOT / "models/yolo"
    models.mkdir(parents=True, exist_ok=True)
    for name in (
        "region_model_best.pt",
        "river_model_best.pt",
        "hand_model_best.pt",
        "meld_model_v32_1280.pt",
        "river_model_best.infer.json",
        "meld_model_v32_1280.infer.json",
        "meld_model_v32_1280.crop_meta.json",
    ):
        dest = models / name
        if source.joinpath(name).is_file():
            if source.joinpath(name).resolve() != dest.resolve():
                shutil.copy2(source / name, dest)
        elif not dest.is_file():
            raise SystemExit(
                f"缺少 YOLO 模型檔:{dest}。先跑 tools/fetch_models.py,或用 --models 指定來源。"
            )


def _report(python: Path, gpu: str) -> None:
    """最後問一次環境:torch 看不看得到顯示卡。``--gpu rocm`` 卻看不到的話要明講。"""
    probe = (
        "import torch\n"
        "ok = torch.cuda.is_available()\n"
        "print(torch.__version__, '|', torch.cuda.get_device_name(0) if ok else 'CPU')\n"
    )
    # 子程序的輸出指定 UTF-8,理由同 engines/yolo/worker.py:Windows 上被接管線時預設是 cp950。
    result = subprocess.run(
        [str(python), "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    errors = result.stderr.strip().splitlines()
    line = result.stdout.strip() or (errors[-1] if errors else "問不到")
    print(f"torch:{line}")
    if result.returncode != 0:
        raise SystemExit("環境裡的 torch 載入失敗,見上一行。")
    if gpu == "rocm" and line.endswith("CPU"):
        raise SystemExit(
            "ROCm 版 torch 裝好了,但看不到顯示卡。這張卡可能不在 ROCm 支援清單上,"
            "或顯示卡驅動太舊。要換回 CPU 版:加上 --gpu cpu 重跑。"
        )
    print("YOLO 環境已就緒;可在 MIA 設定頁選擇 YOLO。")


if __name__ == "__main__":
    main()
