"""建 YOLO 的獨立推論環境 ``engines/yolo/.venv``,並把權重放進 ``models/yolo``。

    .venv/bin/python tools/setup_yolo.py --python python3.12            # macOS
    .venv\\Scripts\\python.exe tools\\setup_yolo.py                        # Windows,CPU
    .venv\\Scripts\\python.exe tools\\setup_yolo.py --gpu auto             # Windows,看顯示卡自己選

``--gpu`` 把環境裡的 torch 換成顯示卡的版本,YOLO 就會跑在顯示卡上:

* ``cuda``:NVIDIA(CUDA 12.8,約 2.9 GB)。**還沒有在 NVIDIA 顯示卡上實測過。**
* ``rocm``:AMD(Windows ROCm 7.2.1,約 2.0 GB)。RX 9070 實測每幀約 150~200 ms,CPU 版約
  650 ms 以上(``docs/yolo.md``)。
* ``xpu``:Intel Arc(約 1.4 GB)。**還沒有在 Intel 顯示卡上實測過。**
* ``auto``:看這台電腦的顯示卡選上面其中一個(規則見 ``mia.yolo.accel``);找不到能用的就裝 CPU 版。
* ``cpu``:換回 CPU 版,並移除上面幾種的執行環境。

**對已經裝好的環境直接再跑一次就好** —— 換的是環境裡的套件,不搬資料夾,MIA 找環境的路徑不變。
MIA 設定頁的「YOLO 運算裝置」選 GPU 時,跑的也是這支。

macOS 不需要這個選項:Apple Silicon 的 torch 本來就會用 MPS。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "engines/yolo"

#: 各 GPU 版 torch 的本地版本標記。要與 ``engines/yolo/requirements-<版本>.txt`` 一致。
GPU_TAGS = {"cuda": "cu128", "rocm": "rocm7.2.1", "xpu": "xpu"}

#: 各 GPU 版**額外**帶進來的執行環境。換成別的版本時要拿掉:torch 換掉之後它們沒有人用,
#: 而且佔好幾百 MB 到幾 GB。CUDA 版在 Windows 上把執行環境包在 torch 裡,沒有額外的。
#: XPU 那串是 torch 2.9.1+xpu 的依賴(它的 metadata 裡釘死的那些)。
GPU_RUNTIMES = {
    "rocm": ("rocm", "rocm-sdk-core", "rocm-sdk-libraries-custom"),
    "xpu": (
        "pytorch-triton-xpu", "intel-cmplr-lib-rt", "intel-cmplr-lib-ur", "intel-cmplr-lic-rt",
        "intel-sycl-rt", "onemkl-sycl-blas", "onemkl-sycl-dft", "onemkl-sycl-lapack",
        "onemkl-sycl-rng", "onemkl-sycl-sparse", "dpcpp-cpp-rt", "intel-opencl-rt", "mkl",
        "intel-openmp", "tbb", "tcmlib", "umf", "intel-pti",
    ),
}


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
        choices=("cpu", "auto", *GPU_TAGS),
        default="cpu",
        help="cuda / rocm / xpu:改用 NVIDIA / AMD / Intel 顯示卡(僅 Windows + Python 3.12);"
        "auto:看顯示卡自己選。預設 cpu",
    )
    parser.add_argument(
        "--models", type=Path, default=ROOT / "downloads/majongsoul ai agent_mac/app/Model"
    )
    args = parser.parse_args()

    environment = BASE / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not environment.exists():
        subprocess.run([*_base_python(args.python), "-m", "venv", str(environment)], check=True)

    gpu = _resolve(args.gpu)
    if gpu in GPU_TAGS:
        _check_gpu_target(python, gpu)
        _install_gpu(python, gpu)
    else:
        _pip(python, "install", "-r", str(BASE / "requirements.txt"))
    # 舊版本的執行環境**最後**才拿掉:前面任何一步失敗或被取消,環境都還是原來那一套能用的。
    _drop_runtimes(python, keep=gpu)

    _copy_models(args.models)
    _report(python, gpu)


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


def _resolve(choice: str) -> str:
    """``auto`` 換成這台電腦該用的版本。Mac 的 MPS 不用換 torch,所以是 ``cpu``(基本環境)。"""
    if choice != "auto":
        return choice
    sys.path.insert(0, str(ROOT / "src"))
    from mia.yolo.accel import list_gpus, plan_for

    plan = plan_for(list_gpus())
    if plan.backend in GPU_TAGS:
        print(f"偵測到 {plan.gpu},改裝 {plan.backend} 版。")
        return plan.backend
    if plan.backend == "mps":
        print("Apple Silicon:基本環境就會用 MPS,不需要另外裝。")
    else:
        print(plan.reason)
    return "cpu"


def _check_gpu_target(python: Path, gpu: str) -> None:
    """GPU 版 wheel 只有 Windows、cp312 這一種。不對就在下載幾 GB 之前停下來。"""
    if sys.platform != "win32":
        raise SystemExit(
            f"--gpu {gpu} 只支援 Windows。macOS 的 torch 本來就會用 MPS,不需要這個選項。"
        )
    version = subprocess.run(
        [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if version != "3.12":
        raise SystemExit(
            f"engines/yolo/.venv 是 Python {version},GPU 版 torch 只有 3.12 的。"
            "請刪掉 engines/yolo/.venv 後重跑,讓它用 Python 3.12 重建。"
        )


def _install_gpu(python: Path, gpu: str) -> None:
    """兩步:先裝 GPU 版 torch,再照 requirements.txt 裝其餘套件,**但不准動 torch**。

    第二步不能直接 ``-r requirements.txt``:那裡釘的是 PyPI 的 CPU 版 torch 2.5.1,pip 會把
    剛裝好的 GPU 版換回去。所以 torch 那兩行跳過,其餘照裝,再用 constraints 鎖住 GPU
    版 —— ultralytics 若要求更新的 torch,pip 會直接報衝突,而不是安靜地換掉。
    """
    print(f"== 1/2 {gpu} 版 torch 與執行環境(1~3 GB,第一次要下載一陣子)")
    _pip(python, "install", "-r", str(BASE / f"requirements-{gpu}.txt"))

    print(f"== 2/2 其餘套件(照 requirements.txt,torch 鎖在 {gpu} 版)")
    others = [
        line
        for line in (BASE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not _is_torch(line)
    ]
    cache = BASE / ".cache"
    cache.mkdir(exist_ok=True)
    constraints = cache / "gpu-constraints.txt"
    tag = GPU_TAGS[gpu]
    # 只寫 ASCII:這個檔也是 pip 讀的,理由見 requirements-rocm.txt 開頭。
    constraints.write_text(f"torch==2.9.1+{tag}\ntorchvision==0.24.1+{tag}\n", encoding="ascii")
    _pip(python, "install", "-c", str(constraints), *others)


#: 在 YOLO 環境裡跑:清單上哪些套件裝著、而且**沒有清單外的套件需要它**。
#: 清單上的彼此需要不算(XPU 版的 torch 已經換掉了,它們之間的依賴一起走)。
_UNUSED = r"""
import json, sys
from importlib.metadata import distributions
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name as norm

stale = {norm(name) for name in json.loads(sys.argv[1])}
# 第二個參數(測試用):只在這些路徑找套件。平常不給,找的就是這個環境。
search = {"path": json.loads(sys.argv[2])} if len(sys.argv) > 2 else {}
installed = {norm(d.metadata["Name"]): d for d in distributions(**search) if d.metadata["Name"]}

def requires(name, extras):
    # 這個套件(連同被要求的附加依賴,例如 rocm[libraries])需要哪些。平台條件照這台電腦算。
    for text in installed[name].requires or []:
        req = Requirement(text)
        if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in extras | {""}):
            yield norm(req.name), frozenset(req.extras)

wanted = [want for name in installed if name not in stale for want in requires(name, frozenset())]
needed = {}
while wanted:
    name, extras = wanted.pop()
    if extras <= needed.get(name, frozenset()) and name in needed:
        continue
    needed[name] = needed.get(name, frozenset()) | extras
    # 留下來的清單套件,它需要的也要留(例:torch 還要 rocm[libraries],那又要 rocm-sdk-*)。
    if name in stale and name in installed:
        wanted.extend(requires(name, needed[name]))
print(json.dumps(sorted(name for name in stale if name in installed and name not in needed)))
"""


def _drop_runtimes(python: Path, *, keep: str) -> None:
    """拿掉其他 GPU 版留下的執行環境 —— 只拿**沒有人在用**的。

    不能照清單全刪:XPU 那串裡有 ``mkl``、``tbb``、``intel-openmp`` 這種通用套件,
    別的版本的 torch 或其他套件可能正需要它們,刪了會把能用的環境弄壞。
    """
    stale = [name for gpu, names in GPU_RUNTIMES.items() if gpu != keep for name in names]
    result = subprocess.run(
        [str(python), "-c", _UNUSED, json.dumps(stale)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        unused = json.loads(result.stdout)
    except ValueError:
        print("查不到哪些舊的執行環境沒人用,這次先不清。")
        return
    if unused:
        print(f"== 移除用不到的執行環境:{' '.join(unused)}")
        _pip(python, "uninstall", "-y", *unused)


def _is_torch(line: str) -> bool:
    name = re.split(r"[\s=<>!~;\[@]", line.strip(), maxsplit=1)[0].lower()
    return name in ("torch", "torchvision")


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
    """最後問一次環境(``engines/yolo/probe.py``):torch 是哪一版、在顯示卡上算不算得動。

    裝了 GPU 版卻算不動的話要明講,並以失敗結束 —— MIA 設定頁看結束碼決定成功與否。
    """
    # 子程序的輸出指定 UTF-8,理由同 engines/yolo/worker.py:Windows 上被接管線時預設是 cp950。
    result = subprocess.run(
        [str(python), str(BASE / "probe.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    sys.path.insert(0, str(ROOT / "src"))
    from mia.yolo.accel import parse_probe

    status = parse_probe(result.stdout)
    where = status.name if status.gpu_ready else "CPU"
    print(f"torch {status.build}:{where}" + (f"({status.error})" if status.error else ""))
    if result.returncode != 0 or status.error.startswith("torch 載入失敗"):
        raise SystemExit("環境裡的 torch 載入失敗,見上一行。")
    if gpu in GPU_TAGS and not status.gpu_ready:
        raise SystemExit(
            f"{gpu} 版 torch 裝好了,但在顯示卡上算不動。可能是顯示卡不在支援清單上,"
            "或顯示卡驅動太舊(請更新驅動後再試)。YOLO 仍可用 CPU 執行。"
        )
    print("YOLO 環境已就緒;可在 MIA 設定頁選擇 YOLO。")


if __name__ == "__main__":
    main()
