"""YOLO 代理的本場:沒有樣板的數字由上一局推(engines/yolo/agent/live_v7/kyoku_info.py)。

錄影只錄到 7 本場,8、9 沒有樣板。這裡借用真的 6、7 字形,把「有樣板的上限」
調到 5,讓它們扮演「讀不出來的 8、9」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "engines" / "yolo" / "agent" / "live_v7"
sys.path[:0] = [str(LIVE), str(LIVE / "ui")]
cv2 = pytest.importorskip("cv2")
kyoku_info = pytest.importorskip("kyoku_info")

TEMPLATES = LIVE / "ui" / "templates_v7"


def _sig(digit: int) -> str:
    image = cv2.imread(str(TEMPLATES / f"honbax__{digit}.png"), cv2.IMREAD_GRAYSCALE)
    return kyoku_info.honba_sig((image > 127).astype("uint8"))


@pytest.fixture
def info(monkeypatch):
    monkeypatch.setattr(kyoku_info, "HONBA_TEMPLATED_MAX", 5)
    k = kyoku_info.KyokuInfo()
    for _ in range(3):  # 上一局:東1局 5 本場,讀得到
        k.feed({"panel": {"kyoku": ["E1", 0.98], "honba": ["5", 1.0]}}, None)
    assert k.value["honba"] == 5
    k.reset()
    return k


def _unread(label: str, sig: str) -> dict:
    return {"kyoku": [label, 0.98], "honba": ["3", 0.55, sig]}


def test_an_unreadable_honba_after_a_known_one_is_the_next_number(info):
    for _ in range(3):  # 舊面板:跟上一局一樣,清掉
        info.feed({"panel": {"kyoku": ["E1", 0.98], "honba": ["5", 1.0]}}, None)
    assert info.value["honba"] is None
    for _ in range(kyoku_info.HONBA_GUESS_VOTES):
        info.feed({"panel": _unread("E1", _sig(6))}, None)
    assert info.value["honba"] == 6
    assert info.start_kyoku_fields()[0]["honba"] == 6


def test_the_old_panel_of_a_guessed_honba_is_not_taken_for_the_next(info):
    for _ in range(kyoku_info.HONBA_GUESS_VOTES):
        info.feed({"panel": _unread("E1", _sig(6))}, None)
    assert info.value["honba"] == 6
    info.reset()
    # 換局後舊面板「東1/6」還留著好幾幀:不可以推成 7
    for _ in range(10):
        info.feed({"panel": _unread("E1", _sig(6))}, None)
    assert info.value["honba"] is None
    # 其實換莊了:東2局 0 本場
    for _ in range(3):
        info.feed({"panel": {"kyoku": ["E2", 0.98], "honba": ["0", 1.0]}}, None)
    assert info.value["kyoku_label"] == "E2"
    assert info.value["honba"] == 0


def test_a_new_glyph_after_a_guessed_honba_is_guessed_again(info):
    for _ in range(kyoku_info.HONBA_GUESS_VOTES):
        info.feed({"panel": _unread("E1", _sig(6))}, None)
    info.reset()
    for _ in range(5):
        info.feed({"panel": _unread("E1", _sig(6))}, None)
    for _ in range(kyoku_info.HONBA_GUESS_VOTES):
        info.feed({"panel": _unread("E1", _sig(7))}, None)
    assert info.value["honba"] == 7


def test_a_weak_match_that_cannot_be_the_honba_does_not_vote(info):
    # 上一局 5,這局只可能是 0 或 6;弱弱地配到「3」是沒有樣板的字被硬配
    for _ in range(5):
        info.feed({"panel": {"kyoku": ["E1", 0.98], "honba": ["3", 0.75, _sig(6)]}}, None)
    assert info.value["honba"] != 3


def test_nothing_is_guessed_without_a_previous_hand():
    k = kyoku_info.KyokuInfo()
    for _ in range(10):
        k.feed({"panel": _unread("E1", _sig(7))}, None)
    assert k.value["honba"] is None
