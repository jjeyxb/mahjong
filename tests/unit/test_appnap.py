"""App Nap 的宣告。節流有沒有真的解除只能實機量,見 :mod:`mia.utils.appnap`。"""

from __future__ import annotations

import sys

import pytest

from mia.utils import appnap


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appnap, "_activity", None)


def test_other_platforms_do_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appnap.sys, "platform", "win32")
    appnap.hold_off_app_nap()
    assert appnap._activity is None  # noqa: SLF001


@pytest.mark.skipif(sys.platform != "darwin", reason="只有 macOS 有 App Nap")
def test_the_activity_is_kept_alive() -> None:
    """代號被回收就等於活動結束,所以一定要留住。"""
    appnap.hold_off_app_nap()
    first = appnap._activity  # noqa: SLF001
    assert first is not None
    appnap.hold_off_app_nap()
    assert appnap._activity is first  # noqa: SLF001
