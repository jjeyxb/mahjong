"""Frozen GUI entry; dispatch capture before importing Qt's UI entry point."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

from mia.distribution import APP_ID
from mia.distribution.layout import LAYOUT
from mia.distribution.service import atomic_json, configuration

HEALTH: tuple[Path, str] | None = None
_MUTEX: object | None = None


def main() -> int:
    global HEALTH, _MUTEX
    multiprocessing.freeze_support()
    args = sys.argv[1:]
    if LAYOUT.packaged:
        LAYOUT.prepare()
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateMutexW.restype = wintypes.HANDLE
            _MUTEX = kernel.CreateMutexW(None, False, APP_ID)
    if args[:1] == ["--self-check"]:
        config = configuration()
        result = {
            "app_id": APP_ID,
            "version": config["version"],
            "packaged": LAYOUT.packaged,
            "install": str(LAYOUT.install),
            "resources": str(LAYOUT.resources),
            "data": str(LAYOUT.data),
            "logs": str(LAYOUT.logs),
            "cache": str(LAYOUT.cache),
            "browser": os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
            "mortal": LAYOUT.worker("mortal").is_file(),
            "yolo": LAYOUT.worker("yolo").is_file(),
            "models": [p.name for p in (LAYOUT.resources / "models").glob("*.pth")],
        }
        if "--engines" in args:
            import numpy as np

            from mia.engine.mortal import mortal_engine
            from mia.mjai import StartGame, StartKyoku, Tsumo
            from mia.yolo.client import YoloClient

            hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"]
            with mortal_engine(LAYOUT.resources / "models/mortal_298k.pth") as engine:
                engine.react(StartGame(id=0))
                engine.react(
                    StartKyoku("E", 1, 0, 0, 0, "2s", [hand, *[["?"] * 13] * 3], [25000] * 4)
                )
                result["mortal_decision"] = engine.react(Tsumo(actor=0, pai="9s")).is_action
            client = YoloClient()
            try:
                client.start()
                client.feed(np.zeros((1080, 1920, 3), dtype=np.uint8), 0)
                result["yolo_device"] = client.device
            finally:
                client.close()
        if "--browser" in args:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chromium", headless=True)
                page = browser.new_page()
                page.set_content("<title>MIA offline browser check</title>")
                result["browser_check"] = page.title()
                browser.close()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args[:1] == ["--capture"]:
        capture_args = args[1:]
        if LAYOUT.packaged:
            from mia.distribution.capture import CaptureRefused, checked_capture_args

            try:
                capture_args = checked_capture_args(capture_args)
            except CaptureRefused as exc:
                print(exc, file=sys.stderr)
                return 2
        return int(importlib.import_module("tools.gt").main(capture_args))
    if "--update-health" in args:
        index = args.index("--update-health")
        path, nonce = Path(args[index + 1]).resolve(), args[index + 2]
        if not path.is_relative_to(LAYOUT.updates.resolve()):
            raise ValueError("啟動檢查檔案必須在更新目錄")
        HEALTH = path, nonce
        del args[index : index + 3]
    if LAYOUT.packaged and HEALTH is None:
        # Mac 上換新版可以直接把新的 MIA.app 拖進去蓋掉,不經過更新程式,所以每次
        # 啟動都對一次版本。啟動檢查那一次不寫:那是更新程式的交易,由它決定。
        _record_install(LAYOUT.install_info, configuration()["version"])
    return int(importlib.import_module("tools.ui").main(args or ["--live"]))


def _record_install(marker: Path, version: str) -> None:
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = None
    if recorded != {"app_id": APP_ID, "version": version} and (
        recorded is None or recorded.get("app_id") in (None, APP_ID)
    ):
        atomic_json(marker, {"app_id": APP_ID, "version": version})
