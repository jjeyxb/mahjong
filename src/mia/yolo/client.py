"""Bounded request/reply pipe to the optional inference interpreter."""

from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from mia.distribution.layout import LAYOUT
from mia.engine.base import EngineError
from mia.utils.geometry import Rect
from mia.utils.paths import LOG_DIR, PROJECT_ROOT
from mia.vision.hud import HudDigits, read_turn, read_wall

_FRAME = Rect(0, 0, 1920, 1080)


class YoloClient:
    def __init__(self, *, python: Path | None = None, timeout: float = 45.0) -> None:
        base = PROJECT_ROOT / "engines/yolo"
        executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        self._packaged = LAYOUT.packaged and python is None
        if self._packaged:
            from mia.distribution.components import yolo_worker

            self.python = yolo_worker(LAYOUT)
        else:
            self.python = python or Path(
                os.environ.get("MIA_YOLO_PYTHON", base / ".venv" / executable)
            )
        if not self.python.is_file():
            raise EngineError("YOLO 尚未安裝，請執行 .venv/bin/python tools/setup_yolo.py")
        self._base = base
        self._timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._closed = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self.device = ""
        self.supports_observations = False
        self._digits: HudDigits | None = None

    def start(self) -> None:
        try:
            self._start_process()
        except EngineError:
            base = LAYOUT.worker("yolo")
            if not self._packaged or self.python == base or self._shutdown.is_set():
                raise
            self._dispose_process()
            self._lines = queue.Queue()
            self.python = base
            self._start_process()

    def _start_process(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._closed.is_set() or self._shutdown.is_set():
                raise EngineError("YOLO 已停止")
            self._log = (LOG_DIR / "yolo-worker.log").open("ab")
            child_env = {**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}
            if self._packaged and self.python != LAYOUT.worker("yolo"):
                # ROCm also uses PyTorch's cuda device API. Preserve an explicit CPU override.
                child_env.setdefault("MJ_DEVICE", "cuda")
            try:
                self._process = subprocess.Popen(
                    (
                        [str(self.python)]
                        if self._packaged
                        else [str(self.python), "-u", str(self._base / "worker.py")]
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._log,
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    creationflags=vars(subprocess).get("CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                self._log.close()
                raise EngineError(f"YOLO 無法啟動:{exc}") from exc
        assert self._process.stdout is not None
        reader = self._process.stdout
        lines = self._lines

        def read() -> None:
            try:
                for line in iter(reader.readline, b""):
                    lines.put(line)
            finally:
                lines.put(None)

        threading.Thread(target=read, name="yolo-output", daemon=True).start()
        ready = self._receive()
        if not ready.get("ready") or ready.get("protocol") != 1:
            raise EngineError("YOLO 通訊版本不相容")
        self.device = str(ready.get("device", "cpu"))
        self.supports_observations = bool(ready.get("observations"))

    def _receive(self) -> dict[str, Any]:
        end = time.monotonic() + self._timeout
        while not self._closed.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise EngineError("YOLO 回應逾時，詳見 logs/yolo-worker.log")
            try:
                line = self._lines.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise EngineError("YOLO 程序已結束，詳見 logs/yolo-worker.log")
            try:
                result = json.loads(line)
            except ValueError as exc:
                raise EngineError("YOLO 傳回無效資料") from exc
            if not isinstance(result, dict):
                raise EngineError("YOLO 傳回的資料不是物件")
            return result
        raise EngineError("YOLO 已停止")

    def hud(self, image: np.ndarray) -> dict[str, int | None]:
        """中央面板的余牌數與燈(輪到誰,視角座位)。YOLO 不讀這兩樣,追蹤器拿來對帳摸牌次數。"""
        if self._digits is None:
            self._digits = HudDigits.load()
        return {"wall": read_wall(image, _FRAME, self._digits), "turn": read_turn(image, _FRAME)}

    def feed(self, image: np.ndarray, wall: float, *, raw: bool = False) -> dict[str, Any]:
        """``raw`` 只回原始觀測(含 :meth:`hud`);否則回追蹤後的事件快照。"""
        if self.supports_observations:
            observation = self._send(image, wall, raw=True)
            observation["hud"] = self.hud(image)
            return observation if raw else self.track(observation)
        return self._send(image, wall, raw=raw)

    def _send(self, image: np.ndarray, wall: float, *, raw: bool) -> dict[str, Any]:
        if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
            raise ValueError("YOLO requires 1920×1080 uint8 BGR")
        process = self._process
        if process is None or process.stdin is None:
            raise EngineError("YOLO 尚未啟動")
        fields: dict[str, Any] = {"shape": list(image.shape), "wall": wall}
        if raw:
            fields["raw_observations"] = True
        header = json.dumps(fields).encode()
        try:
            process.stdin.write(struct.pack("!I", len(header)))
            process.stdin.write(header)
            process.stdin.write(image.tobytes())
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError("YOLO 輸入已中斷") from exc
        return self._receive()

    def track(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Submit fused observations before the child updates its only tracker."""
        process = self._process
        if not self.supports_observations or process is None or process.stdin is None:
            raise EngineError("YOLO 程序不支援混合觀測，請更新推論程序")
        payload = json.dumps(observation, ensure_ascii=False, separators=(",", ":")).encode()
        header = json.dumps({"kind": "observation", "size": len(payload)}).encode()
        try:
            process.stdin.write(struct.pack("!I", len(header)))
            process.stdin.write(header)
            process.stdin.write(payload)
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError("YOLO 混合觀測輸入已中斷") from exc
        return self._receive()

    def abort(self) -> None:
        """讓卡在 :meth:`_receive` 的那條執行緒馬上收到「已停止」,自己不等子程序。

        :meth:`close` 會等子程序結束,最多四秒;UI 執行緒按關閉時只該呼叫這個,等的事交給
        工作執行緒的 finally。
        """
        self._closed.set()
        vars(self).setdefault("_shutdown", threading.Event()).set()

    def close(self) -> None:
        self._closed.set()
        self._shutdown.set()
        self._dispose_process()

    def _dispose_process(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        # stop() may interrupt a large in-flight frame write.
                        with suppress(OSError, ValueError):
                            pipe.close()
            if self._log is not None:
                self._log.close()
