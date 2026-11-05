"""跨平台的小東西:辨識用哪個裝置跑(CUDA / Apple 的 MPS / CPU)。

ultralytics 預設只會自動用 CUDA,Mac 的 GPU(MPS)不會自己用 -> 這裡把
YOLO 的 predict() 包一層,沒指定 device 的時候補上。

設定:config.json 的 "device"(agent 會轉成環境變數 MJ_DEVICE)
    "auto"(預設):有 CUDA 用 CUDA,Mac 有 MPS 用 MPS,不然 CPU
    "mps" / "cpu" / "cuda":強制
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
    if pref in ("cpu", "mps", "cuda"):
        if pref == "mps" and not _mps_ok(torch):
            return "cpu"
        if pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if sys.platform == "darwin" and _mps_ok(torch):
        return "mps"
    return "cpu"


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
    return "CPU"


def setup_device():
    """record_obs 一開始呼叫一次。只有選到 MPS 才需要動 ultralytics。"""
    global _patched
    dev = pick_device()
    if dev != "mps" or _patched:
        return dev
    try:
        from ultralytics.engine.model import Model
    except Exception:
        return dev
    orig = Model.predict

    def predict(self, *a, **kw):
        kw.setdefault("device", "mps")
        return orig(self, *a, **kw)

    Model.predict = predict
    _patched = True
    return dev
