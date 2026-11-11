"""跨平台的小東西:辨識用哪個裝置跑(CUDA / Apple 的 MPS / CPU)。

ultralytics 預設只會自動用 CUDA,Mac 的 GPU(MPS)不會自己用 -> 這裡把
YOLO 的 predict() 包一層,沒指定 device 的時候補上。

設定:config.json 的 "device"(agent 會轉成環境變數 MJ_DEVICE)
    "auto"(預設):有 CUDA 用 CUDA,有 Intel XPU 用 XPU,Mac 有 MPS 用 MPS,不然 CPU
    "mps" / "cpu" / "cuda" / "xpu":強制(★ 2026-11-12 MIA 加 xpu = Intel 顯示卡)
★ MPS 的結果理論上跟 CPU 一樣,但沒有在真的 Mac 上驗過。辨識怪怪的話先改成
  "cpu" 比對看看(tools/check_device.py 可以直接比)。
"""
import os
import sys

_patched = False


def pick_device(pref=None):
    pref = (pref or os.environ.get("MJ_DEVICE") or "auto").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"
    # ★ 2026-11-12 MIA:加 Intel 顯示卡的 "xpu"(PyTorch 的 XPU 版)。
    if pref in ("cpu", "mps", "cuda", "xpu"):
        if pref == "mps" and not _mps_ok(torch):
            return "cpu"
        if pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if pref == "xpu" and not _xpu_ok(torch):
            return "cpu"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_ok(torch):
        return "xpu"
    if sys.platform == "darwin" and _mps_ok(torch):
        return "mps"
    return "cpu"


def _xpu_ok(torch):
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def _mps_ok(torch):
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def device_label(dev=None):
    dev = dev or pick_device()
    if dev == "cuda":
        try:
            import torch
            return "GPU " + torch.cuda.get_device_name(0)
        except Exception:
            return "GPU"
    if dev == "mps":
        return "GPU (Apple MPS)"
    if dev == "xpu":
        try:
            import torch
            return "GPU " + torch.xpu.get_device_name(0)
        except Exception:
            return "GPU"
    return "CPU"


def setup_device():
    """record_obs 一開始呼叫一次。選到 CUDA 以外的裝置都要把它補進 ultralytics 的 predict()。

    ★ 2026-11-12 MIA:原本只補 MPS。但 ultralytics 沒指定 device 時有 CUDA 就用 CUDA ——
    所以 MJ_DEVICE=cpu 在 CUDA / ROCm 版 torch 上擋不住它跑去用顯示卡,只有回報寫著 CPU;
    Intel 的 XPU 則相反,不明講就不會用。現在 CUDA 以外一律明講。
    """
    global _patched
    dev = pick_device()
    if dev == "cuda" or _patched:
        return dev
    try:
        from ultralytics.engine.model import Model
    except Exception:
        return dev
    orig = Model.predict

    def predict(self, *a, **kw):
        kw.setdefault("device", dev)
        return orig(self, *a, **kw)

    Model.predict = predict
    _patched = True
    return dev
