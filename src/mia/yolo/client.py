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

from mia.engine.base import EngineError
from mia.utils.paths import LOG_DIR, PROJECT_ROOT


class YoloClient:
    def __init__(self, *, python: Path | None = None, timeout: float = 45.0) -> None:
        base = PROJECT_ROOT / "engines/yolo"
        executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        self.python = python or Path(os.environ.get("MIA_YOLO_PYTHON", base / ".venv" / executable))
        if not self.python.is_file():
            raise EngineError("YOLO 尚未安裝，請執行 .venv/bin/python tools/setup_yolo.py")
        self._base = base
        self._timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self.device = ""
        self.supports_observations = False

    def start(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._closed.is_set():
                raise EngineError("YOLO 已停止")
            self._log = (LOG_DIR / "yolo-worker.log").open("ab")
            try:
                self._process = subprocess.Popen(
                    [str(self.python), "-u", str(self._base / "worker.py")],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._log,
                    cwd=PROJECT_ROOT,
                )
            except OSError as exc:
                self._log.close()
                raise EngineError(f"YOLO 無法啟動:{exc}") from exc
        assert self._process.stdout is not None
        reader = self._process.stdout

        def read() -> None:
            try:
                for line in iter(reader.readline, b""):
                    self._lines.put(line)
            finally:
                self._lines.put(None)

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

    def feed(self, image: np.ndarray, wall: float, *, raw: bool = False) -> dict[str, Any]:
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

    def close(self) -> None:
        self._closed.set()
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
