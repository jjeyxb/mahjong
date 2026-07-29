"""螢幕擷取層。

以「視窗」為擷取單位而非螢幕矩形,詳見 :mod:`majsoul_copilot.capture.base`。
"""

from majsoul_copilot.capture.base import (
    BackendUnavailableError,
    CaptureBackend,
    CaptureError,
    CaptureFailedError,
    Frame,
    PermissionDeniedError,
    WindowInfo,
    WindowNotFoundError,
)
from majsoul_copilot.capture.factory import available_backends, create_backend

__all__ = [
    "BackendUnavailableError",
    "CaptureBackend",
    "CaptureError",
    "CaptureFailedError",
    "Frame",
    "PermissionDeniedError",
    "WindowInfo",
    "WindowNotFoundError",
    "available_backends",
    "create_backend",
]
