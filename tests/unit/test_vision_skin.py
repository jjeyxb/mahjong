"""角色的手伸過河與摸牌那一格,平滑的手掌會被讀成有信心的白(mia.vision.tiles.skin)。"""

from __future__ import annotations

import numpy as np

from mia.vision.tiles.skin import is_skin


def patch(bgr: tuple[int, int, int], noise: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.full((40, 30, 3), bgr, dtype=np.float32)
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def test_a_palm_is_not_a_tile() -> None:
    """2026-11-08 真人場東四局那一格。"""
    assert is_skin(patch((220, 229, 243)))


def test_a_haku_is_a_tile() -> None:
    assert not is_skin(patch((222, 223, 224)))


def test_ink_is_a_tile() -> None:
    """紅墨多的牌(赤五、中)中位色也偏紅,但牌面有字,不平滑。"""
    tile = patch((200, 210, 250))
    tile[::4] = (40, 40, 180)
    assert not is_skin(tile)


def test_nothing_is_not_skin() -> None:
    assert not is_skin(np.zeros((0, 0, 3), dtype=np.uint8))
