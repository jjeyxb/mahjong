"""牌河網格測試。

重點是**透視**:軸對齊的情況任何寫法都會過,真正要驗的是梯形與旋轉的情況
算出來的格子有沒有跟著變形。所以多數測試用的是刻意歪掉的四邊形。
"""

from __future__ import annotations

import numpy as np
import pytest

from majsoul_copilot.utils.geometry import NormQuad, Rect, Size
from majsoul_copilot.vision.grid import COLS, ROWS, Grid

#: 軸對齊、每格剛好 100x100 的網格,方便直接對數字。
SQUARE = Grid(((0.0, 0.0), (600.0, 0.0), (600.0, 300.0), (0.0, 300.0)))

#: 上窄下寬的梯形 —— 雀魂斜視角下牌河真正的樣子。
TRAPEZOID = Grid(((100.0, 0.0), (500.0, 0.0), (600.0, 300.0), (0.0, 300.0)))


class TestConstruction:
    def test_wrong_number_of_corners_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="4 個角"):
            Grid(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))

    def test_from_norm_applies_the_container_offset(self) -> None:
        """網格座標相對牌桌矩形,牌桌本身在影像中的偏移必須加回去。"""
        quad = NormQuad(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        grid = Grid.from_norm(quad, Rect(100, 200, 600, 300))
        assert grid.corners[0] == (100.0, 200.0)
        assert grid.corners[2] == (700.0, 500.0)

    def test_from_norm_scales_with_resolution(self) -> None:
        quad = NormQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        small = Grid.from_norm(quad, Size(1280, 720))
        large = Grid.from_norm(quad, Size(2560, 1440))
        for a, b in zip(small.corners, large.corners, strict=True):
            assert b == pytest.approx((a[0] * 2, a[1] * 2))


class TestCells:
    def test_cell_grid_covers_the_whole_quad(self) -> None:
        assert SQUARE.cell_quad(0, 0)[0] == pytest.approx((0.0, 0.0))
        assert SQUARE.cell_quad(COLS - 1, ROWS - 1)[2] == pytest.approx((600.0, 300.0))

    def test_cells_are_uniform_when_the_quad_is_a_rectangle(self) -> None:
        for row in range(ROWS):
            for col in range(COLS):
                bounds = SQUARE.cell_bounds(col, row)
                assert bounds.width == 100
                assert bounds.height == 100

    def test_cells_follow_the_perspective_of_a_trapezoid(self) -> None:
        """梯形下,遠端(上方)的格子必須比近端(下方)窄 —— 這正是需要
        四邊形而不是矩形的原因。"""
        top = SQUARE.cell_bounds(0, 0).width  # 對照組:矩形時上下一樣寬
        assert top == 100

        far = TRAPEZOID.cell_bounds(0, 0).width
        near = TRAPEZOID.cell_bounds(0, ROWS - 1).width
        assert far < near, f"遠端 {far} 應該比近端 {near} 窄"

    def test_centres_advance_monotonically_along_both_axes(self) -> None:
        for row in range(ROWS):
            xs = [TRAPEZOID.cell_center(c, row)[0] for c in range(COLS)]
            assert xs == sorted(xs)
        for col in range(COLS):
            ys = [TRAPEZOID.cell_center(col, r)[1] for r in range(ROWS)]
            assert ys == sorted(ys)

    def test_out_of_range_is_an_index_error(self) -> None:
        for col, row in ((COLS, 0), (0, ROWS), (-1, 0), (0, -1)):
            with pytest.raises(IndexError, match="超出"):
                SQUARE.cell_quad(col, row)


class TestWarp:
    def _painted(self) -> np.ndarray:
        """每一格塗上不同灰階,用來驗證 warp 取到的是正確的那一格。"""
        image = np.zeros((300, 600, 3), np.uint8)
        for row in range(ROWS):
            for col in range(COLS):
                image[row * 100 : (row + 1) * 100, col * 100 : (col + 1) * 100] = (
                    10 + row * 60 + col * 8
                )
        return image

    def test_warp_returns_the_requested_size(self) -> None:
        out = SQUARE.warp_cell(self._painted(), 2, 1, (48, 64))
        assert out.shape[:2] == (64, 48)

    def test_warp_picks_the_right_cell(self) -> None:
        image = self._painted()
        for row in range(ROWS):
            for col in range(COLS):
                patch = SQUARE.warp_cell(image, col, row, (32, 32))
                assert patch[16, 16, 0] == 10 + row * 60 + col * 8

    def test_warp_undoes_the_perspective(self) -> None:
        """把一個梯形反扭回去,取出來的每一格都該是同樣大小的正矩形 ——
        這就是四個座位能共用同一套模板的原因。"""
        image = np.zeros((300, 600, 3), np.uint8)
        patches = TRAPEZOID.warp_all(image, (40, 40))
        assert len(patches) == ROWS
        assert all(len(r) == COLS for r in patches)
        assert all(p.shape[:2] == (40, 40) for row in patches for p in row)

    def test_zero_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="須為正"):
            SQUARE.warp_cell(self._painted(), 0, 0, (0, 10))
