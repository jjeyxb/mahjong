"""YOLO 代理的點數讀取(engines/yolo/agent/live_v7/ui/scores.py)。

遮罩是 selfplay-20261106 錄影切出來的金色字遮罩(只有數字,已經轉正)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "yolo" / "agent" / "live_v7" / "ui"))
scores = pytest.importorskip("scores")

FIXTURES = ROOT / "tests" / "fixtures"


def _mask(name: str):
    image = cv2.imread(str(FIXTURES / f"score_mask_{name}.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None, name
    return (image > 127).astype("uint8")


@pytest.mark.parametrize(
    ("name", "seat", "digits"),
    [
        ("minus_P2", "P2", 3),  # -300,上下家的負號單獨一塊
        ("minus_P3", "P3", 4),  # -1300,左右家的負號單獨一塊
        ("minus_joined_P1", "P1", 4),  # -4600,負號黏在「4」上
    ],
)
def test_a_minus_sign_is_found_and_removed_before_reading_digits(name, seat, digits):
    negative, mask = scores.strip_minus(_mask(name), seat)
    assert negative
    assert len(scores.split_digits(mask, seat)) == digits


def test_the_left_edge_of_a_leading_four_is_not_a_minus_sign():
    original = _mask("4200_P1")
    negative, mask = scores.strip_minus(original, "P1")
    assert not negative
    assert (mask == original).all()
    assert len(scores.split_digits(mask, "P1")) == 4
