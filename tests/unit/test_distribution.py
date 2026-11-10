from __future__ import annotations

import base64
import hashlib
import io
import json
import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mia.distribution import APP_ID
from mia.distribution.archive import extract
from mia.distribution.components import yolo_worker
from mia.distribution.download import download
from mia.distribution.installer import install
from mia.distribution.layout import Layout
from mia.distribution.release import Release, ReleaseError, verified_release
from mia.distribution.service import atomic_json, check


@pytest.fixture
def keys():
    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(private.public_key().public_bytes_raw()).decode()
    return private, public


def signed(release, private):
    payload = json.dumps(release.to_dict()).encode()
    return json.dumps(
        dict(
            payload=base64.b64encode(payload).decode(),
            signature=base64.b64encode(private.sign(payload)).decode(),
        )
    ).encode()


def release(data=b"archive", **kwargs):
    return Release(
        "0.2.0",
        "macos-arm64",
        "app",
        "https://example.org/update.zip",
        len(data),
        hashlib.sha256(data).hexdigest(),
        1024,
        **kwargs,
    )


def zipped(files):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return data.getvalue()


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.org/update.zip"


def test_signature_and_target(keys):
    private, public = keys
    assert verified_release(signed(release(), private), public, "macos-arm64") == release()
    with pytest.raises(ReleaseError):
        verified_release(signed(release(), private), public, "windows-x64")


def test_tampering_is_rejected(keys):
    private, public = keys
    envelope = json.loads(signed(release(), private))
    envelope["payload"] = base64.b64encode(b'{"version":"999.0"}').decode()
    with pytest.raises(ReleaseError):
        verified_release(json.dumps(envelope).encode(), public, "macos-arm64")


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://example.org/x"},
        {"url": "https://user:pass@example.org/x"},
        {"size": -1},
        {"size": True},
        {"unpacked_size": -1},
        {"sha256": "a"},
        {"version": "../../x"},
        {"kind": "exec"},
    ],
)
def test_invalid_signed_metadata(changes, keys):
    private, public = keys
    with pytest.raises(ReleaseError):
        verified_release(signed(replace(release(), **changes), private), public, "macos-arm64")


def test_owned_paths_survive_relocation(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(os, "environ", os.environ.copy())
    root = tmp_path / "中文 與 spaces/MIA"
    layout = Layout(root / "program/resources", root, True)
    assert layout.config == root / "userdata/config"
    layout.prepare()
    assert layout.logs.is_dir()
    assert layout.cache.is_dir()
    assert Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]).is_relative_to(root)


@pytest.mark.parametrize("name", ["../x", "/x", "program/../../x", "program/a\\b", "program/C:x"])
def test_archive_cannot_escape(tmp_path, name):
    archive = tmp_path / "x.zip"
    archive.write_bytes(zipped({name: b"x"}))
    with pytest.raises(ReleaseError):
        extract(archive, tmp_path / "stage", "program", 1024)
    assert not (tmp_path.parent / "x").exists()


def test_symlink_safe_and_escaping(tmp_path):
    def build(target):
        archive = tmp_path / f"{len(target)}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("MIA.app/real", "good")
            link = zipfile.ZipInfo("MIA.app/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(link, target)
        return archive

    root = extract(build("real"), tmp_path / "good", "MIA.app", 1024)
    assert (root / "link").read_text() == "good"
    with pytest.raises(ReleaseError):
        extract(build("../../outside"), tmp_path / "bad", "MIA.app", 1024)


def test_archive_size_limit(tmp_path):
    archive = tmp_path / "big.zip"
    archive.write_bytes(zipped({"program/x": b"xxxx"}))
    with pytest.raises(ReleaseError):
        extract(archive, tmp_path / "stage", "program", 3)


def test_download_checks_actual_bytes(tmp_path):
    content = b"good"
    target = download(release(content), tmp_path, opener=lambda *_args, **_kw: Response(content))
    assert target.read_bytes() == content
    for broken in (b"bad", b"too long", b""):
        with pytest.raises(ReleaseError):
            download(
                release(content),
                tmp_path,
                opener=lambda *_args, value=broken, **_kw: Response(value),
            )
        assert target.read_bytes() == content
        assert not list(tmp_path.glob("*.part"))


def test_cancel_download(tmp_path):
    with pytest.raises(ReleaseError, match="取消"):
        download(
            release(),
            tmp_path,
            cancelled=lambda: True,
            opener=lambda *_args, **_kw: Response(b"archive"),
        )
    assert not list(tmp_path.glob("*.zip"))


def setup_install(tmp_path, keys):
    private, public = keys
    root = tmp_path / "中文 MIA"
    job = root / "updates/job"
    job.mkdir(parents=True)
    app = root / "MIA.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/MacOS/MIA").write_text("old")
    (root / "userdata").mkdir()
    (root / "userdata/recording").write_bytes(b"keep me")
    atomic_json(root / "install-info.json", {"app_id": APP_ID, "version": "0.1.0"})
    data = zipped({"MIA.app/Contents/MacOS/MIA": b"new"})
    info = release(data)
    (job / f"{info.sha256}.zip").write_bytes(data)
    (job / "release.json").write_bytes(signed(info, private))
    atomic_json(
        job / "plan.json",
        {
            "app_id": APP_ID,
            "root": str(root),
            "archive": f"{info.sha256}.zip",
            "from_version": "0.1.0",
            "parent_pid": 123,
        },
    )
    config = dict(public_key=public, platform="macos-arm64", version="0.1.0")
    return root, job / "plan.json", config


def test_install_preserves_data_and_old_release(tmp_path, keys):
    root, plan, config = setup_install(tmp_path, keys)
    waited = []
    target = install(plan, config, wait=waited.append, health=lambda *_args: None)
    assert waited == [123]
    assert (target / "Contents/MacOS/MIA").read_text() == "new"
    assert (plan.parent / "previous/Contents/MacOS/MIA").read_text() == "old"
    assert (root / "userdata/recording").read_bytes() == b"keep me"
    assert json.loads((root / "install-info.json").read_text())["version"] == "0.2.0"


def test_failed_launch_restores_program_and_version(tmp_path, keys):
    root, plan, config = setup_install(tmp_path, keys)
    state = root / "userdata/ui_state.json"
    state.write_text('{"old":true}')
    settings = root / "userdata/config"
    settings.mkdir()
    (settings / "default.local.yaml").write_text("old: true")

    def fail(*_args):
        state.write_text('{"new-schema":true}')
        (settings / "default.local.yaml").write_text("new: true")
        (settings / "new-file.yaml").write_text("new: true")
        raise ReleaseError("failed health")

    with pytest.raises(ReleaseError, match="health"):
        install(plan, config, wait=lambda _pid: None, health=fail)
    assert (root / "MIA.app/Contents/MacOS/MIA").read_text() == "old"
    assert json.loads((root / "install-info.json").read_text())["version"] == "0.1.0"
    assert (root / "userdata/recording").read_bytes() == b"keep me"
    assert not (root / "updates/install.lock").exists()
    assert state.read_text() == '{"old":true}'
    assert (settings / "default.local.yaml").read_text() == "old: true"
    assert not (settings / "new-file.yaml").exists()


def test_modified_archive_does_not_touch_installed_app(tmp_path, keys):
    root, plan, config = setup_install(tmp_path, keys)
    next(plan.parent.glob("*.zip")).write_bytes(b"tampered")
    with pytest.raises(ReleaseError):
        install(plan, config, wait=lambda _pid: None)
    assert (root / "MIA.app/Contents/MacOS/MIA").read_text() == "old"


def test_unconfigured_update_never_fetches(monkeypatch):
    monkeypatch.setattr(
        "mia.distribution.service.fetch_metadata", lambda _url: pytest.fail("network")
    )
    with pytest.raises(ReleaseError, match="尚未設定"):
        check({"public_key": "", "feed_url": ""})


def test_missing_or_invalid_component_falls_back(tmp_path):
    layout = Layout(tmp_path / "resources", tmp_path, True)
    assert yolo_worker(layout) == layout.worker("yolo")
    atomic_json(layout.components / "active.json", {"component": "../../elsewhere", "version": "1"})
    assert yolo_worker(layout) == layout.worker("yolo")


def test_archive_root_cannot_be_symlink(tmp_path):
    archive = tmp_path / "root.zip"
    with zipfile.ZipFile(archive, "w") as z:
        link = zipfile.ZipInfo("program")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(link, "../../outside")
    with pytest.raises(ReleaseError, match="根目錄"):
        extract(archive, tmp_path / "stage", "program", 100)


def test_downloaded_metadata_cannot_install_other_app_version(tmp_path, keys):
    root, plan, config = setup_install(tmp_path, keys)
    atomic_json(root / "install-info.json", {"app_id": APP_ID, "version": "9.0"})
    with pytest.raises(ReleaseError, match="來源版本"):
        install(plan, config, wait=lambda _pid: pytest.fail("must not wait"))
    assert (root / "MIA.app/Contents/MacOS/MIA").read_text() == "old"


def test_failed_gpu_worker_starts_base_worker(tmp_path, monkeypatch):
    import os
    import sys

    if os.name == "nt":
        pytest.skip("此合成 executable 使用 POSIX shebang")
    from mia.yolo.client import YoloClient

    layout = Layout(tmp_path, tmp_path, True)
    bad = tmp_path / "bad-gpu"
    gpu_environment = tmp_path / "gpu-environment.txt"
    monkeypatch.delenv("MJ_DEVICE", raising=False)
    good = layout.worker("yolo")
    good.parent.mkdir(parents=True)
    bad.write_text(
        f"#!{sys.executable}\nimport os\nfrom pathlib import Path\n"
        f"Path({str(gpu_environment)!r}).write_text(os.environ.get('MJ_DEVICE', ''))\n"
        "raise SystemExit(2)\n"
    )
    good.write_text(
        f"#!{sys.executable}\nimport sys\n"
        'print(\'{"ready":true,"protocol":1,"device":"cpu"}\',flush=True)\n'
        "sys.stdin.buffer.read()\n"
    )
    for path in (bad, good):
        path.chmod(0o755)
    monkeypatch.setattr("mia.yolo.client.LAYOUT", layout)
    monkeypatch.setattr("mia.distribution.components.yolo_worker", lambda _layout: bad)
    client = YoloClient(timeout=3)
    try:
        client.start()
        assert client.device == "cpu"
        assert client.python == good
        assert gpu_environment.read_text() == "cuda"
    finally:
        client.close()
