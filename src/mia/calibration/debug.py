"""校正結果的視覺化標註。

給 ``tools/capture_probe.py`` 與(之後的)ROI 標註工具共用,
把抽象的座標畫成看得見的框,人眼一秒就能判斷校正對不對。
"""

from __future__ import annotations

import cv2
import numpy as np

from mia.calibration.table import Calibration
from mia.utils.geometry import NormRect, Rect

__all__ = ["annotate_calibration"]

_TABLE_COLOR = (0, 220, 0)  # BGR 綠 — 牌桌矩形
_GRID_COLOR = (0, 160, 255)  # BGR 橘 — 三分格線
_TEXT_BG = (0, 0, 0)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """畫帶黑底的文字,避免壓在亮色背景上看不清楚。"""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x, y - th - baseline - 2), (x + tw + 4, y + 2), _TEXT_BG, cv2.FILLED)
    cv2.putText(image, text, (x + 2, y - baseline), font, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)


def annotate_calibration(
    image: np.ndarray,
    calibration: Calibration,
    *,
    rois: dict[str, NormRect] | None = None,
) -> np.ndarray:
    """回傳標註後的影像複本(不修改輸入)。

    畫出:牌桌矩形外框、三分格線(方便目測是否對齊畫布中心)、
    尺寸與寬高比文字,以及選填的 ROI 們。
    """
    canvas = image.copy()
    rect = calibration.table_rect

    cv2.rectangle(canvas, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), _TABLE_COLOR, 2)

    for i in (1, 2):
        gx = rect.x + rect.width * i // 3
        gy = rect.y + rect.height * i // 3
        cv2.line(canvas, (gx, rect.y), (gx, rect.bottom - 1), _GRID_COLOR, 1)
        cv2.line(canvas, (rect.x, gy), (rect.right - 1, gy), _GRID_COLOR, 1)

    _put_label(
        canvas,
        f"table {rect.width}x{rect.height} @({rect.x},{rect.y})  "
        f"aspect={calibration.aspect:.4f}  scale={calibration.scale:.2f}  "
        f"src={calibration.source}",
        (rect.x + 4, max(16, rect.y - 6)),
    )

    for name, roi in (rois or {}).items():
        r: Rect = calibration.roi_to_pixels(roi)
        cv2.rectangle(canvas, (r.x, r.y), (r.right - 1, r.bottom - 1), (255, 120, 0), 1)
        _put_label(canvas, name, (r.x, max(14, r.y - 2)))

    return canvas
