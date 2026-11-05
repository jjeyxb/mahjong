"""以排序版作參考,保證快速中位數沒有改變亮度與色相門檻的含義。"""

import numpy as np
import pytest

from mia.vision.stats import median_u8


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (7, 9), (64, 84), (120, 90)])
def test_median_matches_numpy_on_every_channel(shape: tuple[int, int]) -> None:
    image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    for channel in range(3):
        assert median_u8(image, channel) == float(np.median(image[:, :, channel]))
        # 實際牌河是從大圖切出來的 view,不保證記憶體連續。
        view = image[::2, ::2]
        assert median_u8(view, channel) == float(np.median(view[:, :, channel]))


def test_even_pixels_average_the_two_middle_values() -> None:
    image = np.array([[0, 110, 111, 255]], dtype=np.uint8)
    assert median_u8(image) == 110.5
