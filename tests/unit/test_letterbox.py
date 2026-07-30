"""邊框偵測測試。

用合成影像模擬雀魂在各種視窗比例下的呈現方式:
上下黑邊(letterbox)、左右黑邊(pillarbox)、非黑色背景。
"""

from __future__ import annotations

import numpy as np
import pytest

from mia.calibration.letterbox import (
    estimate_background,
    find_content_rect,
    fit_aspect,
)
from mia.utils.geometry import Rect
from tests.conftest import make_letterboxed


class TestEstimateBackground:
    def test_detects_uniform_black_border(self) -> None:
        image = make_letterboxed((800, 600), Rect(100, 100, 600, 400))
        assert np.array_equal(estimate_background(image), np.array([0, 0, 0], dtype=np.uint8))

    def test_detects_non_black_border(self) -> None:
        image = make_letterboxed((800, 600), Rect(100, 100, 600, 400), background=(40, 30, 20))
        assert np.array_equal(estimate_background(image), np.array([40, 30, 20], dtype=np.uint8))

    def test_ignores_transparent_rounded_corners(self) -> None:
        """迴歸測試:macOS 圓角視窗的四角是透明像素,premultiplied 後變純黑。

        實測發現若取樣包含角落,亮色標題列的視窗會被誤判成「背景是黑色」,
        導致整條邊框偵測失效。取樣必須排除角落。
        """
        image = make_letterboxed((800, 600), Rect(100, 100, 600, 400), background=(235, 234, 232))
        for corner in (np.s_[:20, :20], np.s_[:20, -20:], np.s_[-20:, :20], np.s_[-20:, -20:]):
            image[corner] = 0

        background = estimate_background(image)

        assert background.tolist() == [235, 234, 232]


class TestFindContentRect:
    def test_letterbox_top_and_bottom(self) -> None:
        """視窗比畫布高 → 上下黑邊。"""
        expected = Rect(0, 90, 1280, 720)
        found = find_content_rect(make_letterboxed((1280, 900), expected))
        assert found == expected

    def test_pillarbox_left_and_right(self) -> None:
        """視窗比畫布寬 → 左右黑邊。"""
        expected = Rect(160, 0, 1280, 720)
        found = find_content_rect(make_letterboxed((1600, 720), expected))
        assert found == expected

    def test_border_on_all_four_sides(self) -> None:
        expected = Rect(37, 61, 900, 506)
        found = find_content_rect(make_letterboxed((1000, 700), expected))
        assert found == expected

    def test_non_black_background(self) -> None:
        """瀏覽器深色主題:邊框不是純黑。"""
        expected = Rect(50, 120, 1200, 675)
        image = make_letterboxed((1300, 1000), expected, background=(45, 42, 38))
        assert find_content_rect(image) == expected

    def test_content_filling_whole_image(self) -> None:
        image = make_letterboxed((640, 480), Rect(0, 0, 640, 480))
        assert find_content_rect(image) == Rect(0, 0, 640, 480)

    def test_uniform_image_returns_none(self) -> None:
        """整張同色 → 偵測不到內容,必須回 None 而不是硬給一塊。"""
        assert find_content_rect(np.zeros((480, 640, 3), dtype=np.uint8)) is None

    def test_content_too_small_returns_none(self) -> None:
        # 內容區只佔 1%,低於 min_content_ratio → 判定失敗
        image = make_letterboxed((1000, 1000), Rect(450, 450, 100, 100))
        assert find_content_rect(image, min_content_ratio=0.3) is None

    def test_isolated_bright_pixel_does_not_expand_bounds(self) -> None:
        """邊框上的單一亮點(滑鼠殘影、圓角反鋸齒)不該把邊界撐開。

        這是 line_ratio 參數存在的理由 —— 若改用「該列存在任一內容像素」
        的判準,這個測試就會失敗。
        """
        expected = Rect(150, 100, 500, 400)
        image = make_letterboxed((800, 600), expected)
        image[5, 5] = (255, 255, 255)
        image[590, 790] = (255, 255, 255)
        assert find_content_rect(image) == expected

    def test_rejects_non_bgr_input(self) -> None:
        with pytest.raises(ValueError, match="需要 BGR 影像"):
            find_content_rect(np.zeros((10, 10), dtype=np.uint8))


class TestOsWindowChrome:
    """作業系統視窗裝飾的剝除 —— 依實測資料建構。"""

    def _macos_window(self, *, titlebar_bgr: tuple[int, int, int]) -> np.ndarray:
        """模擬 macOS 視窗:純色標題列 + 圓角透明角 + 遊戲畫布。

        尺寸與顏色取自實機量測(Retina 2x 下標題列 56 px,亮色主題 BGR 235/234/232)。
        """
        width, height, titlebar = 3024, 1740, 56
        image = make_letterboxed((width, height), Rect(0, titlebar, width, height - titlebar))
        image[:titlebar] = titlebar_bgr
        # 標題列上的文字與紅綠燈按鈕
        image[20:36, 40:160] = (90, 90, 90)
        image[18:34, 1400:1620] = (70, 70, 70)
        # 圓角:premultiplied alpha 下透明像素為純黑
        for corner in (np.s_[:6, :6], np.s_[:6, -6:]):
            image[corner] = 0
        return image

    def test_light_titlebar_is_peeled_exactly(self) -> None:
        found = find_content_rect(self._macos_window(titlebar_bgr=(235, 234, 232)))
        assert found == Rect(0, 56, 3024, 1684)

    def test_dark_titlebar_is_peeled_exactly(self) -> None:
        """深色主題的標題列顏色完全不同,演算法不可寫死成某個顏色。"""
        found = find_content_rect(self._macos_window(titlebar_bgr=(42, 40, 38)))
        assert found == Rect(0, 56, 3024, 1684)

    def test_titlebar_plus_letterbox_both_peeled(self) -> None:
        """標題列與遊戲黑邊同時存在 —— 兩者顏色不同,必須各自處理。"""
        width, height, titlebar, bar = 1600, 1000, 40, 80
        canvas = Rect(0, titlebar + bar, width, height - titlebar - 2 * bar)
        image = make_letterboxed((width, height), canvas)
        image[:titlebar] = (235, 234, 232)  # 亮色標題列
        image[titlebar : titlebar + bar] = 0  # 黑色 letterbox

        assert find_content_rect(image) == canvas


class TestFitAspect:
    def test_too_tall_crops_vertically(self) -> None:
        # 1000x1000 → 16:9 應裁成 1000x562,且垂直置中
        fitted = fit_aspect(Rect(0, 0, 1000, 1000), 16 / 9)
        assert fitted.width == 1000
        assert fitted.height == 562
        assert fitted.y == (1000 - 562) // 2

    def test_too_wide_crops_horizontally(self) -> None:
        fitted = fit_aspect(Rect(0, 0, 2000, 500), 16 / 9)
        assert fitted.height == 500
        assert fitted.width == round(500 * 16 / 9)
        assert fitted.x == (2000 - fitted.width) // 2

    def test_already_correct_is_unchanged(self) -> None:
        rect = Rect(10, 20, 1920, 1080)
        assert fit_aspect(rect, 16 / 9) == rect

    def test_result_is_always_contained_in_input(self) -> None:
        for rect in (Rect(5, 7, 1000, 300), Rect(5, 7, 300, 1000), Rect(0, 0, 1920, 1080)):
            assert rect.contains(fit_aspect(rect, 16 / 9))

    def test_preserves_offset(self) -> None:
        fitted = fit_aspect(Rect(100, 200, 1000, 1000), 16 / 9)
        assert fitted.x == 100  # 只裁上下,x 不動

    def test_invalid_aspect_rejected(self) -> None:
        with pytest.raises(ValueError, match="須為正數"):
            fit_aspect(Rect(0, 0, 10, 10), 0)
