"""依設定與平台建立擷取後端。"""

from __future__ import annotations

import sys

from mia.capture.base import BackendUnavailableError, CaptureBackend
from mia.capture.mss_fallback import MSSCaptureBackend
from mia.config.models import BackendName, CaptureConfig
from mia.utils.logging import logger

__all__ = ["available_backends", "create_backend"]


def _make(name: BackendName, config: CaptureConfig) -> CaptureBackend:
    if name == "macos":
        from mia.capture.macos import MacOSCaptureBackend

        return MacOSCaptureBackend(retina=config.retina)
    if name == "windows":
        from mia.capture.windows import WindowsCaptureBackend

        return WindowsCaptureBackend()
    if name == "mss":
        return MSSCaptureBackend()
    raise BackendUnavailableError(f"未知的擷取後端: {name!r}")


def _native_backend_name() -> BackendName | None:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return None


def available_backends() -> list[str]:
    """列出當前平台實際可用的後端名稱。"""
    from mia.capture.macos import MacOSCaptureBackend
    from mia.capture.windows import WindowsCaptureBackend

    names = []
    for backend in (MacOSCaptureBackend, WindowsCaptureBackend, MSSCaptureBackend):
        if backend.is_available():
            names.append(backend.name)
    return names


def create_backend(config: CaptureConfig) -> CaptureBackend:
    """建立擷取後端。

    ``backend="auto"`` 時優先使用平台原生的單視窗擷取,失敗才退回 mss
    —— 因為 mss 會把 Overlay 拍進去且解析度減半,是有實質代價的降級。
    """
    if config.backend != "auto":
        return _make(config.backend, config)

    native = _native_backend_name()
    if native is not None:
        try:
            backend = _make(native, config)
            if backend.is_available():
                return backend
            logger.warning("原生後端 {} 不可用,退回 mss", native)
        except Exception as exc:  # noqa: BLE001 - 任何失敗都該能降級,不該讓程式起不來
            logger.warning("建立原生後端 {} 失敗({}),退回 mss", native, exc)
    else:
        logger.warning("平台 {} 沒有原生單視窗擷取實作,使用 mss", sys.platform)

    logger.warning(
        "使用 mss fallback:解析度較低,且 Overlay 會被拍進辨識輸入,請讓 HUD 避開牌桌區域"
    )
    return MSSCaptureBackend()
