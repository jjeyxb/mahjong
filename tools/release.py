"""Create a signed update archive locally; never uploads or publishes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mia.distribution.release import Release, https_url


def keygen(private: Path, public: Path) -> None:
    if private.exists() or public.exists():
        raise ValueError("金鑰已存在，不會覆寫")
    key = Ed25519PrivateKey.generate()
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open("xb") as fh:
        os.chmod(private, 0o600)
        fh.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(
        base64.b64encode(key.public_key().public_bytes_raw()).decode() + "\n", encoding="utf-8"
    )


def package(source: Path, destination: Path) -> int:
    total = 0
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [source, *sorted(source.rglob("*"))]:
            name = path.relative_to(source.parent).as_posix()
            if path.is_symlink():
                data = os.readlink(path).encode()
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, data)
                total += len(data)
            else:
                archive.write(path, name)
                total += path.stat().st_size if path.is_file() else 0
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("keygen")
    keys.add_argument("--private", type=Path, required=True)
    keys.add_argument("--public", type=Path, required=True)
    release = commands.add_parser("create")
    release.add_argument("--source", type=Path, required=True)
    release.add_argument("--out", type=Path, required=True)
    release.add_argument("--private", type=Path, required=True)
    release.add_argument("--version", required=True)
    release.add_argument("--platform", choices=("windows-x64", "macos-arm64"), required=True)
    release.add_argument("--url", required=True)
    release.add_argument("--notes", default="")
    release.add_argument("--component", choices=("yolo-cuda", "yolo-rocm"))
    release.add_argument("--app-version", default="")
    args = parser.parse_args()
    if args.command == "keygen":
        keygen(args.private, args.public)
        print("金鑰已建立；私鑰請另外備份，勿打包或提交。")
        return 0
    https_url(args.url)
    expected = (
        "component"
        if args.component
        else ("MIA.app" if args.platform == "macos-arm64" else "program")
    )
    if args.source.name != expected:
        parser.error(f"來源應是 {expected} 目錄，不能包含 userdata")
    if args.component and not args.app_version:
        parser.error("加速元件必須指定 --app-version")
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "payload.zip"
    unpacked = package(args.source, archive)
    if archive.stat().st_size >= 2 * 1024**3:
        raise ValueError("此原型尚不支援分片，單一檔案超過 GitHub Releases 限制")
    with archive.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    info = Release(
        args.version,
        args.platform,
        "component" if args.component else "app",
        args.url,
        archive.stat().st_size,
        digest,
        unpacked,
        args.notes,
        args.component or "",
        args.app_version,
    )
    payload = json.dumps(info.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    key = serialization.load_pem_private_key(args.private.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("需要 Ed25519 發布金鑰")
    envelope = dict(
        payload=base64.b64encode(payload).decode(),
        signature=base64.b64encode(key.sign(payload)).decode(),
    )
    (args.out / f"update-{args.platform}.json").write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )
    print(f"已產生簽署資訊與封裝（尚未發布）:{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
