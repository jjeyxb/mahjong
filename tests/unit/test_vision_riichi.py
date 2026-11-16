"""上家的「立直」特效字,見 :mod:`mia.vision.riichi`。

fixture 是特效區那一塊(已縮成牌桌寬 1280 的比例),測試時貼回一張空白牌桌上。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.utils.geometry import Rect
from mia.vision.riichi import _WINDOW, kamicha_riichi, riichi_score

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _table(filename: str, scale: float = 1.0) -> tuple[np.ndarray, Rect]:
    """把特效區貼回一張牌桌,牌桌放大 ``scale`` 倍、往右下挪一點。"""
    patch = cv2.imread(str(FIXTURES / filename))
    assert patch is not None
    x0, y0, x1, y1 = _WINDOW
    canvas = np.zeros((720, 1280, 3), np.uint8)
    canvas[y0:y1, x0:x1] = patch
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image = np.zeros((canvas.shape[0] + 50, canvas.shape[1] + 30, 3), np.uint8)
    image[50:, 30:] = canvas
    return image, Rect(30, 50, canvas.shape[1], canvas.shape[0])


@pytest.mark.parametrize(
    "filename",
    [
        # 真人對局,另一個角色的立繪。
        "riichi_banner_human.png",
        # Mac 的 Steam 版:牌桌比例 1.59,只照寬度縮放就對得上。
        "riichi_banner_steam.png",
    ],
)
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_riichi_banner_is_seen(filename: str, scale: float) -> None:
    image, table = _table(filename, scale)
    assert riichi_score(image, table) > 0.95
    assert kamicha_riichi(image, table)


@pytest.mark.parametrize(
    "filename",
    [
        # 對家碰的特效字蓋在上家這一塊。
        "riichi_banner_toimen_pon.png",
        # 47.7 萬幀裡最像的:友人場設定畫面上的白色大字,0.435。
        "riichi_banner_lobby.png",
    ],
)
def test_other_pictures_are_not_a_riichi_banner(filename: str) -> None:
    image, table = _table(filename)
    assert riichi_score(image, table) < 0.5
    assert not kamicha_riichi(image, table)


def test_a_table_too_small_to_crop_is_not_a_banner() -> None:
    image = np.zeros((20, 20, 3), np.uint8)
    assert riichi_score(image, Rect(0, 0, 20, 20)) == 0.0
