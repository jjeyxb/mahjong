"""Signed, bounded metadata. The trust key comes from the installed build."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import Version

from mia.distribution import APP_ID

MAX_METADATA = 256 * 1024


class ReleaseError(ValueError):
    """Untrusted or incompatible distribution."""


def https_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ReleaseError("下載來源必須是沒有帳密的 HTTPS URL")
    return url


@dataclass(frozen=True)
class Release:
    version: str
    platform: str
    kind: str
    url: str
    size: int
    sha256: str
    unpacked_size: int
    notes: str = ""
    component: str = ""
    app_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {"schema": 1, "app_id": APP_ID, **asdict(self)}


def verified_release(envelope: bytes, public_key: str, target: str) -> Release:
    if len(envelope) > MAX_METADATA:
        raise ReleaseError("更新資訊超過大小限制")
    try:
        body = json.loads(envelope)
        payload = base64.b64decode(body["payload"], validate=True)
        signature = base64.b64decode(body["signature"], validate=True)
        key = base64.b64decode(public_key, validate=True)
        Ed25519PublicKey.from_public_bytes(key).verify(signature, payload)
        data = json.loads(payload)
        if data.pop("schema") != 1 or data.pop("app_id") != APP_ID:
            raise ReleaseError("不是相容的 MIA 發布資訊")
        release = Release(**data)
        Version(release.version)
        if release.platform != target or release.kind not in ("app", "component"):
            raise ReleaseError("更新的平台或類型不相容")
        if type(release.size) is not int or not 0 < release.size <= 32 * 1024**3:
            raise ReleaseError("不合法的下載大小")
        if type(release.unpacked_size) is not int or not 0 < release.unpacked_size <= 64 * 1024**3:
            raise ReleaseError("不合法的解壓大小")
        if not re.fullmatch(r"[a-f0-9]{64}", release.sha256):
            raise ReleaseError("不合法的檔案雜湊")
        if release.kind == "component":
            if not re.fullmatch(r"yolo-(cuda|rocm)", release.component):
                raise ReleaseError("不支援的加速元件")
            Version(release.app_version)
        https_url(release.url)
        return release
    except (InvalidSignature, KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise ReleaseError(f"無法驗證發布資訊:{exc}") from exc
