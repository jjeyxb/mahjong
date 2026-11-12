"""YOLO 推論程序的入口;上游的推論程式不改。

兩種跑法:

* 打包成 ``MIA-yolo.exe``(PyInstaller),帶 CPU 版 torch(Mac 是含 MPS 的版本)。
* Windows 的 GPU 加速元件:``tools/package.py`` 把這個檔與 ``worker.py`` 一起放在 ``MIA-yolo``
  的 ``_internal/`` 裡,由下載來的 GPU 版 Python 直接執行(見 ``mia.yolo.component``)。
  跟 exe 用的是同一份程式與 ``agent/``,只有 torch 不同。
"""
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

multiprocessing.freeze_support()
os.environ["YOLO_AUTOINSTALL"] = "false"
meipass = getattr(sys, "_MEIPASS", None)
frozen = meipass is not None
here = Path(__file__).resolve().parent
if meipass is not None:
    base = Path(meipass)
    # exe 在 resources/engines/yolo/ 底下。
    default_resources = Path(sys.executable).resolve().parents[2]
else:
    base = here
    # 這個檔在 resources/engines/yolo/_internal/ 底下。GPU 元件的 Python 以 -I 執行,
    # 不會自動把這個資料夾放進 sys.path,worker.py 就在這裡。
    default_resources = here.parents[2]
    sys.path.insert(0, str(here))
    if sys.platform == "win32":
        # 打包版 MIA 的 PyInstaller 啟動器呼叫過 SetDllDirectoryW(指向 MIA 自己的 _internal),
        # 子程序會繼承。不清掉的話,torch 載入 DLL 時可能先找到 MIA 帶的同名檔(版本不同)。
        # 必須在 import torch 之前。
        import ctypes

        ctypes.windll.kernel32.SetDllDirectoryW(None)
config = json.loads((base / "resources/release.json").read_text(encoding="utf-8"))
resources = Path(os.environ.get("MIA_RESOURCE_ROOT", default_resources))
os.environ.setdefault("MIA_YOLO_MODELS", str(resources / "models/yolo"))
install = resources.parents[3 if config["platform"] == "macos-arm64" else 2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(install / "userdata/yolo"))
# Windows 的 exe 只有 CPU 版 torch;GPU 元件就是為了用顯示卡才裝的,有哪張用哪張。
os.environ.setdefault(
    "MJ_DEVICE", "cpu" if frozen and config["platform"] == "windows-x64" else "auto"
)
# Ultralytics falls back to /tmp if this directory does not already exist.
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

# Ultralytics may announce first-run settings while it is imported. stdout is
# exclusively the JSON protocol, including before worker.main starts.
with contextlib.redirect_stdout(sys.stderr):
    import worker
    import platform_compat
    from ultralytics.engine.model import Model

original = Model.predict


def predict(self, *args, **kwargs):
    kwargs.setdefault("device", platform_compat.pick_device())
    return original(self, *args, **kwargs)


Model.predict = predict
if "--self-check" in sys.argv:
    with contextlib.redirect_stdout(sys.stderr):
        detector = worker.Detector()
        import numpy as np
        detector.feed(np.zeros((1080, 1920, 3), dtype=np.uint8), 0, raw=True)
        import torch
        backend = "rocm" if torch.version.hip and detector.device == "cuda" else detector.device
    print(json.dumps(dict(ok=True, backend=backend, models=4)))
else:
    worker.main()
