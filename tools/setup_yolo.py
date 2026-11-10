"""Create the optional YOLO environment and install/copy provided weights.

.venv/bin/python tools/setup_yolo.py --python python3.12
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument(
        "--models", type=Path, default=ROOT / "downloads/majongsoul ai agent_mac/app/Model"
    )
    args = parser.parse_args()
    base = ROOT / "engines/yolo"
    environment = base / ".venv"
    if not environment.exists():
        subprocess.run([args.python, "-m", "venv", str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    # --no-cache-dir:不然 pip 會把 torch 等套件另外快取一份到使用者目錄
    # (%LOCALAPPDATA%\pip\Cache 或 ~/.cache/pip),實測約 336 MB,刪掉 engines/yolo/.venv
    # 也清不到。重裝時多下載一次,換到的是「刪資料夾就乾淨」。
    subprocess.run(
        [
            str(python), "-m", "pip", "install", "--no-cache-dir",
            "-r", str(base / "requirements.txt"),
        ],
        check=True,
    )
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
        if args.models.joinpath(name).is_file():
            if args.models.joinpath(name).resolve() != dest.resolve():
                shutil.copy2(args.models / name, dest)
        elif not dest.is_file():
            raise SystemExit(f"缺少 YOLO 模型檔:{args.models / name}")
    print("YOLO 環境已就緒；可在 MIA 設定頁選擇 YOLO。")


if __name__ == "__main__":
    main()
