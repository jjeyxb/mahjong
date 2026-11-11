"""Build isolated frozen workers and an offline, relocatable desktop bundle.

Run using the main development Python. Install PyInstaller in a separate build
venv and pass its site-packages via --build-tools; source environments stay intact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 不隨發布版提供的模組:自動打牌、牌譜與自打工具、MITM。見 mia.distribution.capture。
_NOT_SHIPPED = (
    "autoplay",
    "tools.autoplay",
    "paipu",
    "tools.paipu",
    "selfplay",
    "tools.selfplay",
    "mitmproxy",
    "mitmproxy_rs",
)


def copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-source", type=Path, default=ROOT)
    parser.add_argument("--build-tools", type=Path, required=True)
    parser.add_argument("--main-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mortal-python", type=Path, required=True)
    parser.add_argument("--yolo-python", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--only", choices=("mortal", "yolo", "updater", "gui", "assemble"))
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    target = (
        "macos-arm64"
        if sys.platform == "darwin" and platform.machine() == "arm64"
        else "windows-x64"
    )
    if not (sys.platform == "win32" or target == "macos-arm64"):
        parser.error("請在對應的 Windows x64 或 Apple Silicon Mac 上建置")
    version = (
        args.version or tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    )
    config = dict(
        app_id="io.github.jjeyxb.mia",
        version=version,
        platform=target,
        public_key=args.public_key.read_text().strip() if args.public_key else "",
        feed_url=f"https://github.com/jjeyxb/mahjong/releases/latest/download/update-{target}.json",
        components={},
    )
    generated = args.out / "build-config"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "release.json").write_text(json.dumps(config, indent=2))
    suffix = ".exe" if sys.platform == "win32" else ""
    roles = {
        "mortal": args.mortal_python,
        "yolo": args.yolo_python,
        "updater": args.main_python,
        "gui": args.main_python,
    }
    for role, python in roles.items():
        if args.only and args.only != role:
            continue
        name = "MIA" if role == "gui" else f"MIA-{role}"
        command = [
            str(python),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--name",
            name,
            "--distpath",
            str(args.out / "frozen"),
            "--workpath",
            str(args.out / "work" / role),
            "--specpath",
            str(args.out / "specs"),
            "--paths",
            str(ROOT),
            "--paths",
            str(ROOT / "src"),
            "--paths",
            str(ROOT / "tools"),
            "--add-data",
            f"{generated / 'release.json'}:resources",
            "--exclude-module",
            "pytest",
            "--exclude-module",
            "IPython",
        ]
        if role == "mortal":
            upstream = args.asset_source / "engines/mortal/Mortal/mortal"
            command += [
                "--paths",
                str(ROOT / "engines/mortal"),
                "--paths",
                str(upstream),
                "--hidden-import",
                "torch",
                "--hidden-import",
                "numpy",
                "--hidden-import",
                "libriichi",
                "--exclude-module",
                "tensorboard",
            ]
            for filename in ("engine.py", "model.py"):
                command += ["--add-data", f"{upstream / filename}:Mortal/mortal"]
            library = upstream / ("libriichi.pyd" if sys.platform == "win32" else "libriichi.so")
            command += ["--add-binary", f"{library}:Mortal/mortal"]
        elif role == "yolo":
            agent = ROOT / "engines/yolo/agent"
            command += [
                "--paths",
                str(ROOT / "engines/yolo"),
                "--paths",
                str(agent),
                "--paths",
                str(agent / "live_v7"),
                "--add-data",
                f"{agent}:agent",
                "--collect-all",
                "ultralytics",
                "--hidden-import",
                "torchvision",
            ]
            # Their __file__-relative assets must resolve beside the copied source.
            for module in agent.rglob("*.py"):
                command += ["--exclude-module", module.stem]
        elif role == "gui":
            command += [
                "--windowed",
                "--exclude-module",
                "torch",
                "--exclude-module",
                "torchvision",
                "--add-data",
                f"{ROOT / 'assets'}:resources/assets",
                "--add-data",
                f"{ROOT / 'config/default.yaml'}:resources/config",
                "--hidden-import",
                "tools.gt",
                "--hidden-import",
                "tools.ui",
                # 自動打牌只給對電腦的房間錄素材,MITM 要使用者自己信任 CA 憑證:都不隨發布版提供。
                # tools/gt.py 只在函式裡引用它們,排除之後一般的瀏覽器擷取不受影響。
                # 入口另有白名單,見 mia.distribution.capture。
                *(
                    flag
                    for module in _NOT_SHIPPED
                    for flag in ("--exclude-module", module)
                ),
            ]
            # 圖示由 tools/make_icons.py 從 assets/icon.svg 產生。
            icon = "MIA.icns" if sys.platform == "darwin" else "MIA.ico"
            command += ["--icon", str(ROOT / "packaging/icons" / icon)]
            if sys.platform == "darwin":
                command += ["--osx-bundle-identifier", "io.github.jjeyxb.mia"]
        command += [str(ROOT / "packaging/entrypoints" / f"{role}.py")]
        env = {
            **os.environ,
            "YOLO_AUTOINSTALL": "false",
            "YOLO_CONFIG_DIR": str(args.out / "yolo-config"),
            "PYINSTALLER_CONFIG_DIR": str(args.out / "pyinstaller-cache" / role),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), str(ROOT), str(args.build_tools.resolve())]
            ),
        }
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.only and args.only != "assemble":
        return 0
    destination = args.out / target / "MIA"
    destination.mkdir(parents=True, exist_ok=True)
    stage = args.out / target / f".stage-{uuid.uuid4().hex}"
    stage.mkdir()
    if target == "macos-arm64":
        program = stage / "MIA.app"
        copy(args.out / "frozen/MIA.app", program)
        resources = program / "Contents/Resources/resources"
    else:
        program = stage / "program"
        copy(args.out / "frozen/MIA", program)
        resources = program / "_internal/resources"
    if json.loads((resources / "release.json").read_text()) != config:
        raise SystemExit("主程式版本／更新金鑰與組裝參數不同，請以相同參數重建")
    for role in ("mortal", "yolo", "updater"):
        source = args.out / "frozen" / f"MIA-{role}"
        if not (source / f"MIA-{role}{suffix}").exists():
            raise SystemExit(f"缺少建置產物: {source}")
        worker_config = json.loads((source / "_internal/resources/release.json").read_text())
        if worker_config != config:
            raise SystemExit(f"{role} 的版本／更新金鑰不同，請以相同參數重建")
        copy(source, resources / "engines" / role)
    for filename in (
        "mortal_298k.pth",
        "style_call_high.pth",
        "style_call_low.pth",
        "style_reach_high.pth",
        "style_reach_low.pth",
    ):
        copy(args.asset_source / "models" / filename, resources / "models" / filename)
    yolo_models = args.asset_source / "models/yolo"
    for filename in (
        "region_model_best.pt",
        "river_model_best.pt",
        "hand_model_best.pt",
        "meld_model_v32_1280.pt",
        "river_model_best.infer.json",
        "meld_model_v32_1280.infer.json",
        "meld_model_v32_1280.crop_meta.json",
    ):
        copy(yolo_models / filename, resources / "models/yolo" / filename)
    for path in args.browser_root.iterdir():
        if path.name.startswith(("chromium-", "ffmpeg-")):
            copy(path, resources / "browsers" / path.name)
    if not list((resources / "browsers").glob("chromium-*")):
        raise SystemExit("缺少離線 Chromium")
    for filename in ("LICENSE", "NOTICE"):
        copy(ROOT / filename, resources / filename)
    if target == "macos-arm64":
        info_path = program / "Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info["NSScreenCaptureUsageDescription"] = (
            "MIA 需要擷取遊戲畫面，以辨識麻將牌並提供 AI 建議與放銃分析。"
        )
        info_path.write_bytes(plistlib.dumps(info))
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(program)], check=True)
    installed = destination / program.name
    previous = stage / "previous-program"
    if installed.is_symlink():
        raise SystemExit("不覆寫符號連結形式的組裝目錄")
    if installed.exists():
        entry = (
            installed / "Contents/MacOS/MIA" if target == "macos-arm64" else installed / "MIA.exe"
        )
        if not entry.is_file():
            raise SystemExit("既有組裝目錄不是有效的 MIA，不予覆寫")
        installed.rename(previous)
    try:
        program.rename(installed)
    except BaseException:
        if previous.exists():
            previous.rename(installed)
        raise
    if target != "macos-arm64":
        # Mac 的 install-info.json 在使用者的 Application Support,由 App 第一次啟動時寫。
        (destination / "install-info.json").write_text(
            json.dumps({"app_id": config["app_id"], "version": version})
        )
    if previous.exists():
        shutil.rmtree(previous)
    stage.rmdir()
    print(f"已建置 {destination}（內測版，尚未公證／發布）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
