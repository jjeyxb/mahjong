"""打包版的 GPU 加速元件(mia.yolo.component):清單、啟用、安裝流程、下載驗證。

真的下載與安裝(好幾 GB)不在單元測試裡跑;這裡用假的版面與替身確認「失敗不留痕跡、
成功才換」與檔案驗證的規則。實機流程見 docs/yolo.md「打包版的 GPU 加速」。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import threading
from pathlib import Path

import pytest

from mia.distribution.layout import Layout
from mia.yolo import component

ROOT = Path(__file__).resolve().parents[2]


def _layout(tmp_path: Path, *, lock: str = "torch @ https://x/t.whl --hash=sha256:aa\n") -> Layout:
    """像 Windows 打包版的資料夾:resources 裡有清單與原始碼入口。"""
    resources = tmp_path / "program/_internal/resources"
    gpu = resources / "gpu"
    gpu.mkdir(parents=True)
    (gpu / "windows-x64-rocm.txt").write_text(lock, encoding="ascii")
    manifest = {
        "python": {"url": "https://python.org/e.zip", "sha256": "1" * 64, "size": 10},
        "pip": {"url": "https://pypi.org/pip.whl", "sha256": "2" * 64, "size": 5},
        "backends": {"rocm": {"lock": "windows-x64-rocm.txt", "size": 2000}},
    }
    (gpu / "windows-x64.json").write_text(json.dumps(manifest), encoding="utf-8")
    layout = Layout(resources, tmp_path, True)
    entry = layout.worker("yolo").parent / "_internal" / "yolo.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    return layout


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


def _activate(layout: Layout) -> Path:
    manifest = component._manifest(layout)  # noqa: SLF001
    assert manifest is not None
    ident = component._component_id(layout, manifest, "rocm")  # noqa: SLF001
    folder = layout.components / "gpu" / f"rocm-{ident}"
    folder.mkdir(parents=True)
    (folder / "python.exe").write_text("", encoding="utf-8")
    (layout.components / "gpu" / "active.json").write_text(
        json.dumps({"backend": "rocm", "id": ident}), encoding="utf-8"
    )
    return folder


@pytest.mark.usefixtures("windows")
class TestAvailability:
    def test_only_a_packaged_windows_build_with_the_list_can_download(self, tmp_path) -> None:
        layout = _layout(tmp_path)
        assert component.available(layout)
        assert component.download_size("rocm", layout) == 2015
        assert not component.available(Layout(layout.resources, tmp_path, False))

    def test_the_mac_build_never_downloads(self, tmp_path, monkeypatch) -> None:
        layout = _layout(tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert not component.available(layout)


@pytest.mark.usefixtures("windows")
class TestWorkerCommand:
    def test_nothing_installed_means_the_bundled_cpu_worker(self, tmp_path) -> None:
        assert component.worker_command(_layout(tmp_path)) is None

    def test_the_active_component_runs_the_bundled_source(self, tmp_path) -> None:
        layout = _layout(tmp_path)
        folder = _activate(layout)
        command = component.worker_command(layout)
        assert command is not None
        assert command[0] == str(folder / "python.exe")
        assert command[-1].endswith("yolo.py")
        assert "-I" in command

    def test_a_component_from_an_older_list_is_not_used(self, tmp_path) -> None:
        """MIA 更新後清單換了(torch 版本不同),舊的元件不能拿來跑新的程式。"""
        layout = _layout(tmp_path)
        _activate(layout)
        lock = layout.resources / "gpu/windows-x64-rocm.txt"
        lock.write_text("torch @ https://x/new.whl --hash=sha256:bb\n", encoding="ascii")
        assert component.worker_command(layout) is None

    def test_a_tampered_active_file_falls_back(self, tmp_path) -> None:
        layout = _layout(tmp_path)
        _activate(layout)
        (layout.components / "gpu/active.json").write_text("{", encoding="utf-8")
        assert component.worker_command(layout) is None


@pytest.mark.usefixtures("windows")
class TestClientChoice:
    def test_cpu_uses_the_bundled_worker_and_gpu_uses_the_component(
        self, tmp_path, monkeypatch
    ) -> None:
        """選 CPU 不必載入好幾 GB 的顯示卡函式庫:直接用打包版自帶的 MIA-yolo.exe。"""
        from mia.yolo.client import YoloClient

        layout = _layout(tmp_path)
        folder = _activate(layout)
        layout.worker("yolo").write_text("", encoding="utf-8")
        monkeypatch.setattr("mia.yolo.client.LAYOUT", layout)
        assert YoloClient(device="cpu").python == layout.worker("yolo")
        assert YoloClient(device="gpu").python == folder / "python.exe"


@pytest.mark.usefixtures("windows")
class TestInstall:
    def _fake_steps(self, monkeypatch, *, check_ok: bool = True) -> list[str]:
        steps: list[str] = []

        def python(_manifest, stage, *_args) -> None:
            steps.append("python")
            (stage / "python.exe").write_text("", encoding="utf-8")

        def pip(*_args) -> None:
            steps.append("pip")

        def check(_stage, backend, *_args) -> None:
            steps.append("check")
            if not check_ok:
                raise component._Failed(f"{backend} 跑不動")  # noqa: SLF001

        monkeypatch.setattr(component, "_install_python", python)
        monkeypatch.setattr(component, "_pip_install", pip)
        monkeypatch.setattr(component, "_self_check", check)
        return steps

    def test_success_activates_the_new_component_and_removes_old_ones(
        self, tmp_path, monkeypatch
    ) -> None:
        layout = _layout(tmp_path)
        old = layout.components / "gpu" / "cuda-deadbeef"
        old.mkdir(parents=True)
        steps = self._fake_steps(monkeypatch)
        ok, _ = component.install("rocm", lambda _line: None, threading.Event(), layout)
        assert ok
        assert steps == ["python", "pip", "check"]
        assert component.worker_command(layout) is not None
        assert not old.exists()
        assert not list((layout.components / "gpu").glob(".stage-*"))

    def test_a_failed_check_leaves_everything_as_it_was(
        self, tmp_path, monkeypatch
    ) -> None:
        layout = _layout(tmp_path)
        previous = _activate(layout)
        snapshot = (layout.components / "gpu/active.json").read_text(encoding="utf-8")
        self._fake_steps(monkeypatch, check_ok=False)
        ok, message = component.install("rocm", lambda _line: None, threading.Event(), layout)
        assert not ok
        assert "跑不動" in message
        assert (layout.components / "gpu/active.json").read_text(encoding="utf-8") == snapshot
        assert (previous / "python.exe").is_file()
        assert not list((layout.components / "gpu").glob(".stage-*"))

    def test_an_unknown_backend_is_refused(self, tmp_path) -> None:
        ok, _ = component.install("cuda", lambda _line: None, threading.Event(), _layout(tmp_path))
        assert not ok


class _Response(io.BytesIO):
    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class TestDownload:
    def _item(self, data: bytes) -> dict[str, object]:
        return {"url": "https://example.org/f.zip", "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data)}

    def test_a_matching_file_is_kept(self, tmp_path, monkeypatch) -> None:
        data = b"python" * 100
        monkeypatch.setattr(component.urllib.request, "urlopen",
                            lambda *_a, **_k: _Response(data, "https://example.org/f.zip"))
        target = component._download(self._item(data), tmp_path / "f.zip", "Python",  # noqa: SLF001
                                     lambda _line: None, threading.Event())
        assert target.read_bytes() == data

    def test_a_modified_file_is_refused(self, tmp_path, monkeypatch) -> None:
        item = self._item(b"original")
        monkeypatch.setattr(component.urllib.request, "urlopen",
                            lambda *_a, **_k: _Response(b"tampered", "https://example.org/f.zip"))
        with pytest.raises(component._Failed, match="不符"):  # noqa: SLF001
            component._download(item, tmp_path / "f.zip", "Python", lambda _l: None,  # noqa: SLF001
                                threading.Event())

    def test_a_redirect_to_plain_http_is_refused(self, tmp_path, monkeypatch) -> None:
        data = b"x"
        monkeypatch.setattr(component.urllib.request, "urlopen",
                            lambda *_a, **_k: _Response(data, "http://evil.example/f.zip"))
        with pytest.raises(Exception, match="HTTPS"):
            component._download(self._item(data), tmp_path / "f.zip", "Python",  # noqa: SLF001
                                lambda _l: None, threading.Event())


class TestPipOutput:
    def test_raw_progress_becomes_a_readable_line(self) -> None:
        script = (
            "print('Downloading torch-2.9.1-cp312-win_amd64.whl (2.0 MB)');"
            "print('Progress 2097152 of 2097152')"
        )
        lines: list[str] = []
        component._run([sys.executable, "-c", script], {}, lines.append, threading.Event())  # noqa: SLF001
        assert any("torch-2.9.1-cp312-win_amd64.whl" in line and "2 / 2 MB" in line
                   for line in lines)

    def test_a_failing_pip_reports_its_last_line(self) -> None:
        script = "import sys; print('ERROR: hash mismatch'); sys.exit(1)"
        with pytest.raises(component._Failed, match="hash mismatch"):  # noqa: SLF001
            component._run([sys.executable, "-c", script], {}, lambda _l: None,  # noqa: SLF001
                           threading.Event())


class TestShippedLists:
    """packaging/gpu/ 是 tools/lock_gpu.py 產生、隨安裝檔附上的清單。"""

    def test_every_file_is_pinned_to_an_https_url_and_a_hash(self) -> None:
        gpu = ROOT / "packaging/gpu"
        manifest = json.loads((gpu / "windows-x64.json").read_text(encoding="utf-8"))
        assert set(manifest["backends"]) == {"cuda", "rocm", "xpu"}
        for item in (manifest["python"], manifest["pip"]):
            assert item["url"].startswith("https://")
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for backend, info in manifest["backends"].items():
            lines = [line for line in (gpu / info["lock"]).read_text(encoding="ascii").splitlines()
                     if line and not line.startswith("#")]
            names = {line.split(" @ ", 1)[0] for line in lines}
            assert {"torch", "torchvision", "ultralytics", "setuptools"} <= names, backend
            for line in lines:
                assert re.fullmatch(r"\S+ @ https://\S+ --hash=sha256:[0-9a-f]{64}", line), line
                url = line.split(" @ ", 1)[1].split()[0]
                # 只有 AMD 的 rocm 是原始碼套件;其他都要 wheel,玩家電腦上不能編譯。
                assert url.endswith(".whl") or line.startswith("rocm @ "), line

    def test_the_torch_build_matches_the_developer_setup(self) -> None:
        """打包版與原始碼版(tools/setup_yolo.py --gpu)要裝同一個 torch。"""
        gpu = ROOT / "packaging/gpu"
        for backend in ("cuda", "rocm", "xpu"):
            wanted = (ROOT / f"engines/yolo/requirements-{backend}.txt").read_text(
                encoding="utf-8-sig"
            )
            torch = next(line for line in wanted.splitlines() if line.startswith("torch @ "))
            sha = torch.rsplit("sha256=", 1)[1]
            lock = (gpu / f"windows-x64-{backend}.txt").read_text(encoding="ascii")
            assert f"--hash=sha256:{sha}" in lock, backend
