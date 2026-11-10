from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mia.capture import macos
from mia.capture.base import PermissionDeniedError


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(macos, "_QUARTZ_AVAILABLE", True)
    monkeypatch.setattr(
        macos,
        "Quartz",
        SimpleNamespace(kCGWindowImageBestResolution=1, kCGWindowImageNominalResolution=0),
    )
    return macos.MacOSCaptureBackend()


def test_missing_permission_is_reported_before_hidden_titles_look_like_no_game(
    backend, monkeypatch
):
    requested = []
    monkeypatch.setattr(backend, "has_permission", lambda: False)
    monkeypatch.setattr(backend, "request_permission", lambda: requested.append(True))
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def hidden_titles(**_kwargs):
        pytest.fail("must request permission before trying to match hidden window titles")

    monkeypatch.setattr(macos, "list_windows", hidden_titles)
    for _ in range(2):
        with pytest.raises(PermissionDeniedError, match=r"MIA.app"):
            backend.find_window(["雀魂"])
    assert requested == [True]


def test_existing_authorization_does_not_prompt_and_allows_window_discovery(
    backend, monkeypatch
):
    monkeypatch.setattr(backend, "has_permission", lambda: True)
    monkeypatch.setattr(backend, "request_permission", lambda: pytest.fail("already granted"))
    options = []
    monkeypatch.setattr(macos, "list_windows", lambda **kw: options.append(kw) or [])
    assert backend.list_windows() == []
    assert backend.list_windows_everywhere() == []
    assert options == [{"include_all": False}, {"everywhere": True}]
