"""影像統計。uint8 只有 256 個值,不必把每個像素拿去排序才能求中位數。"""

from __future__ import annotations

import cv2
import numpy as np


def median_u8(image: np.ndarray, channel: int = 0) -> float:
    """一個 uint8 影像通道的精確中位數,偶數個像素取中間兩值的平均。"""
    count = image.shape[0] * image.shape[1]
    if not count:
        return float("nan")
    histogram = cv2.calcHist([image], [channel], None, [256], [0, 256]).ravel()
    cumulative = histogram.cumsum()
    low = int(cumulative.searchsorted((count - 1) // 2, side="right"))
    high = int(cumulative.searchsorted(count // 2, side="right"))
    return (low + high) / 2
