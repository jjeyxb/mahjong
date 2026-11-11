"""在 YOLO 環境裡問 torch:是哪一種版本、看不看得到顯示卡、在上面算不算得動。

印一行 JSON 到 stdout,給 MIA 的設定頁與 ``tools/setup_yolo.py`` 最後的檢查用::

    {"torch": "2.9.1+rocm7.2.1", "build": "rocm", "device": "cuda",
     "name": "AMD Radeon RX 9070", "ok": true}

``build`` 是 torch 是哪一種版本(``cpu`` / ``cuda`` / ``rocm`` / ``xpu``);``device`` 是 YOLO 該用的
裝置(ROCm 版也叫 ``cuda``,那是 PyTorch 的命名),沒有可用的顯示卡就是 ``null``。

**要真的在顯示卡上算一次,不能只看 is_available()。** 驅動太舊、或顯示卡比 torch 編進去的架構舊時,
is_available() 照樣回 True,要到第一次跑運算才報「no kernel image」—— 那會發生在 YOLO 開局的第一幀,
使用者只看到辨識起不來。這裡先撞,設定頁才能當場講清楚。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> None:
    # 理由同 worker.py:Windows 上被接管線的 stdout 是 cp950,顯示卡名稱不一定是 ASCII。
    sys.stdout.reconfigure(encoding="utf-8")
    # 與 worker.py 同一組快取位置:AMD 第一次跑會編譯 GPU 核心,不要寫進使用者目錄。
    for name, sub in (("AMD_COMGR_CACHE_DIR", "comgr"), ("MIOPEN_USER_DB_PATH", "miopen"),
                      ("MIOPEN_CUSTOM_CACHE_DIR", "miopen")):
        if name not in os.environ:
            (BASE / ".cache" / sub).mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(BASE / ".cache" / sub)
    report: dict[str, object] = {
        "torch": None, "build": "cpu", "device": None, "name": "", "ok": False,
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 —— 環境壞掉也要回一行看得懂的,不要只丟 traceback
        report["error"] = f"torch 載入失敗:{exc}"
        print(json.dumps(report, ensure_ascii=False))
        return
    report["torch"] = torch.__version__
    if getattr(torch.version, "hip", None):
        report["build"] = "rocm"
    elif getattr(torch.version, "cuda", None):
        report["build"] = "cuda"
    elif getattr(torch.version, "xpu", None) or "+xpu" in torch.__version__:
        report["build"] = "xpu"
    try:
        if torch.cuda.is_available():
            report["device"], report["name"] = "cuda", torch.cuda.get_device_name(0)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            report["device"], report["name"] = "xpu", torch.xpu.get_device_name(0)
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            report["device"], report["name"] = "mps", "Apple GPU"
        if report["device"] is not None:
            value = (torch.ones(64, device=str(report["device"])) * 2).sum().item()
            report["ok"] = value == 128
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["error"] = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
