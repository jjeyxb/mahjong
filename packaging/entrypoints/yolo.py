"""Frozen worker wrapper; upstream inference stays unchanged."""
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

multiprocessing.freeze_support()
os.environ["YOLO_AUTOINSTALL"] = "false"
base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
config = json.loads((base / "resources/release.json").read_text(encoding="utf-8"))
resources = Path(os.environ.get("MIA_RESOURCE_ROOT", Path(sys.executable).resolve().parents[2]))
os.environ.setdefault("MIA_YOLO_MODELS", str(resources / "models/yolo"))
install = resources.parents[3 if config["platform"] == "macos-arm64" else 2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(install / "userdata/yolo"))
os.environ.setdefault("MJ_DEVICE", "cpu" if config["platform"] == "windows-x64" else "auto")
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
