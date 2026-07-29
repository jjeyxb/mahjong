"""座標型別與空間轉換的測試。

這些測試看似瑣碎,但座標轉換錯一個像素,後面整條辨識管線都會歪掉,
而且症狀是「準確率莫名偏低」這種很難追的形式。
"""

from __future__ import annotations

import numpy as np
import pytest

from majsoul_copilot.utils.geometry import NormRect, Rect, Size


class TestRect:
    def test_derived_properties(self) -> None:
        rect = Rect(10, 20, 100, 50)
        assert rect.right == 110
        assert rect.bottom == 70
        assert rect.area == 5000
        assert rect.size == Size(100, 50)
        assert rect.center == (60, 45)
        assert rect.aspect == pytest.approx(2.0)

    def test_zero_height_aspect_does_not_raise(self) -> None:
        assert Rect(0, 0, 100, 0).aspect == 0.0

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="不可為負"):
            Rect(0, 0, -1, 10)

    def test_from_bounds(self) -> None:
        assert Rect.from_bounds(10, 20, 110, 70) == Rect(10, 20, 100, 50)

    def test_from_bounds_inverted_clamps_to_zero(self) -> None:
        assert Rect.from_bounds(100, 100, 50, 50) == Rect(100, 100, 0, 0)

    def test_scaled_scales_position_too(self) -> None:
        # Retina 2x:邏輯座標 → 像素座標,位置也必須跟著放大
        assert Rect(10, 20, 100, 50).scaled(2.0) == Rect(20, 40, 200, 100)

    def test_intersect(self) -> None:
        a, b = Rect(0, 0, 100, 100), Rect(50, 50, 100, 100)
        assert a.intersect(b) == Rect(50, 50, 50, 50)

    def test_intersect_disjoint_returns_none(self) -> None:
        assert Rect(0, 0, 10, 10).intersect(Rect(20, 20, 10, 10)) is None

    def test_intersect_touching_edges_is_not_overlap(self) -> None:
        assert Rect(0, 0, 10, 10).intersect(Rect(10, 0, 10, 10)) is None

    def test_inset_clamps_instead_of_going_negative(self) -> None:
        assert Rect(0, 0, 10, 10).inset(20) == Rect(20, 20, 0, 0)

    def test_contains(self) -> None:
        outer = Rect(0, 0, 100, 100)
        assert outer.contains(Rect(10, 10, 50, 50))
        assert not outer.contains(Rect(90, 90, 50, 50))

    def test_as_slice_indexes_numpy_in_yx_order(self) -> None:
        image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        rect = Rect(x=50, y=10, width=30, height=20)
        cropped = image[rect.as_slice()]
        assert cropped.shape == (20, 30, 3)  # (height, width, ch)
        assert np.array_equal(cropped, image[10:30, 50:80])


class TestNormRect:
    def test_to_pixels_with_size_has_no_offset(self) -> None:
        assert NormRect(0.25, 0.5, 0.5, 0.25).to_pixels(Size(400, 200)) == Rect(100, 100, 200, 50)

    def test_to_pixels_with_rect_applies_container_origin(self) -> None:
        # ROI 相對於牌桌矩形,而牌桌矩形本身在整張影像中有偏移
        table = Rect(30, 40, 400, 200)
        assert NormRect(0.25, 0.5, 0.5, 0.25).to_pixels(table) == Rect(130, 140, 200, 50)

    def test_roundtrip_through_pixels(self) -> None:
        table = Rect(11, 23, 1920, 1080)
        original = NormRect(0.1, 0.2, 0.3, 0.4)
        restored = NormRect.from_pixels(original.to_pixels(table), table)
        assert restored.x == pytest.approx(original.x, abs=1e-3)
        assert restored.y == pytest.approx(original.y, abs=1e-3)
        assert restored.width == pytest.approx(original.width, abs=1e-3)
        assert restored.height == pytest.approx(original.height, abs=1e-3)

    def test_resolution_independence(self) -> None:
        """同一份 ROI 在 1080p 與 4K 下必須落在畫面的同一個相對位置。"""
        roi = NormRect(0.4, 0.8, 0.2, 0.15)
        hd = roi.to_pixels(Size(1920, 1080))
        uhd = roi.to_pixels(Size(3840, 2160))
        assert uhd.x == hd.x * 2
        assert uhd.y == hd.y * 2
        assert uhd.width == hd.width * 2
        assert uhd.height == hd.height * 2

    def test_tiny_roi_never_collapses_to_zero(self) -> None:
        # 四捨五入後寬高不可為 0,否則後續切片會拿到空陣列
        tiny = NormRect(0.0, 0.0, 0.0001, 0.0001).to_pixels(Size(100, 100))
        assert tiny.width >= 1 and tiny.height >= 1

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((1.5, 0.0, 0.1, 0.1), "原點須落在"),
            ((0.0, 0.0, 0.0, 0.1), "尺寸須為正"),
            ((0.8, 0.0, 0.5, 0.1), "超出容器邊界"),
        ],
    )
    def test_invalid_values_rejected(self, args: tuple[float, ...], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            NormRect(*args)

    def test_exactly_full_extent_is_allowed(self) -> None:
        assert NormRect(0.0, 0.0, 1.0, 1.0).to_pixels(Size(50, 50)) == Rect(0, 0, 50, 50)
