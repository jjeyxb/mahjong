"""螢幕擷取層。

以「視窗」為擷取單位而非螢幕矩形,詳見 :mod:`mia.capture.base`。
"""

from mia.capture.base import (
    BackendUnavailableError,
    CaptureBackend,
    CaptureError,
    CaptureFailedError,
    Frame,
    PermissionDeniedError,
    WindowInfo,
    WindowNotFoundError,
)
from mia.capture.factory import available_backends, create_backend

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
