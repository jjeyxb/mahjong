"""牌桌矩形校正。

:class:`Calibration` 是整個視覺管線的座標基準。它回答一個問題:
**「畫面上哪一塊是雀魂的 16:9 遊戲畫布?」**

有了它,所有 ROI 就能以 :class:`~majsoul_copilot.utils.geometry.NormRect`
(0~1 正規化座標)定義並存進 YAML,執行期再換算成當下的實際像素位置。
這樣同一份 ROI 設定可以同時適用於 Retina 與非 Retina、全螢幕與視窗化、
1080p 與 4K —— 不需要為每種情況各存一份座標。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from majsoul_copilot.calibration.letterbox import find_content_rect, fit_aspect
from majsoul_copilot.capture.base import Frame
from majsoul_copilot.config.models import CalibrationConfig
from majsoul_copilot.utils.geometry import NormRect, Rect, Size
from majsoul_copilot.utils.logging import logger

__all__ = ["Calibration", "TableCalibrator"]

CalibrationSource = Literal["auto", "manual", "fallback"]


@dataclass(frozen=True, slots=True)
class Calibration:
    """一次校正的結果。

    Attributes:
        table_rect: 牌桌畫布在**擷取影像**中的像素矩形。
        image_size: 校正當下的擷取影像尺寸。影像尺寸一變就必須重新校正,
            見 :meth:`matches`。
        scale: 來自 :attr:`Frame.scale` 的 pixel/logical 比值。需要把畫面座標
            換算回滑鼠可點擊的邏輯座標時會用到。
        source: ``auto`` 自動偵測成功 / ``manual`` 使用設定檔指定 /
            ``fallback`` 偵測失敗,退而使用整張影像。
        warnings: 校正過程中的疑慮。非空代表結果可能不可靠,上層應顯示給使用者。
    """

    table_rect: Rect
    image_size: Size
    scale: float = 1.0
    source: CalibrationSource = "auto"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def aspect(self) -> float:
        return self.table_rect.aspect

    @property
    def is_reliable(self) -> bool:
        return not self.warnings and self.source != "fallback"

    def matches(self, frame: Frame) -> bool:
        """這份校正是否仍適用於給定的畫面。

        影像尺寸改變(使用者縮放了視窗、切換全螢幕)就必須重新校正。
        """
        return frame.size == self.image_size

    def roi_to_pixels(self, roi: NormRect) -> Rect:
        """ROI 正規化座標 → 擷取影像上的像素矩形。"""
        return roi.to_pixels(self.table_rect)

    def pixels_to_roi(self, rect: Rect) -> NormRect:
        """擷取影像上的像素矩形 → ROI 正規化座標。ROI 標註工具會用到。"""
        return NormRect.from_pixels(rect, self.table_rect)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """從整張擷取影像切出牌桌區域(回傳 view,非複本)。"""
        return image[self.table_rect.as_slice()]

    def __str__(self) -> str:
        flag = "" if self.is_reliable else f" ⚠ {len(self.warnings)} 項警告"
        return (
            f"Calibration(table={self.table_rect} aspect={self.aspect:.4f} "
            f"scale={self.scale:.2f} source={self.source}{flag})"
        )


class TableCalibrator:
    """從一張擷取畫面推導出 :class:`Calibration`。

    流程:

    1. 設定檔若指定了 ``manual_table_rect``,直接採用(逃生門)。
    2. 否則裁掉四周純色邊框,得到內容區。
    3. 檢查內容區的寬高比;偏離 16:9 超過容許值就置中裁成 16:9 並記一筆警告。
    4. 完全偵測不到內容區時,退回使用整張影像並標記為 ``fallback``。
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()

    def calibrate(self, frame: Frame) -> Calibration:
        image = frame.image
        image_size = frame.size
        cfg = self.config

        if cfg.manual_table_rect is not None:
            x, y, w, h = cfg.manual_table_rect
            table_rect = NormRect(x, y, w, h).to_pixels(image_size)
            logger.debug("使用設定檔指定的牌桌矩形: {}", table_rect)
            return Calibration(
                table_rect=table_rect,
                image_size=image_size,
                scale=frame.scale,
                source="manual",
            )

        content = find_content_rect(
            image,
            tolerance=cfg.border_tolerance,
            uniformity=cfg.border_uniformity,
            max_peel_ratio=cfg.max_peel_ratio,
            min_content_ratio=cfg.min_content_ratio,
        )
        if content is None:
            logger.warning("偵測不到內容區,退回使用整張擷取影像作為牌桌矩形")
            return Calibration(
                table_rect=Rect.from_size(image_size),
                image_size=image_size,
                scale=frame.scale,
                source="fallback",
                warnings=(
                    "偵測不到遊戲畫布邊界,已改用整張視窗影像。"
                    "請確認雀魂確實顯示在該視窗中,或在設定檔以 manual_table_rect 手動指定。",
                ),
            )

        warnings: list[str] = []
        deviation = abs(content.aspect - cfg.aspect_ratio) / cfg.aspect_ratio
        if deviation > cfg.aspect_tolerance:
            detected = content.aspect
            if cfg.enforce_aspect:
                content = fit_aspect(content, cfg.aspect_ratio)
                action = f"已依 enforce_aspect 置中裁切成 {content}"
            else:
                action = (
                    "未做裁切(enforce_aspect=false)。若確認目標環境會補黑邊,"
                    "可開啟 enforce_aspect;若畫布位置本來就不對,請用 manual_table_rect 指定"
                )
            warnings.append(
                f"內容區寬高比 {detected:.4f} 偏離預期的 {cfg.aspect_ratio:.4f} "
                f"達 {deviation:.1%}(容許 {cfg.aspect_tolerance:.1%});{action}。"
                f"常見原因:畫面中還有未剝乾淨的視窗裝飾或瀏覽器工具列。"
            )

        calibration = Calibration(
            table_rect=content,
            image_size=image_size,
            scale=frame.scale,
            source="auto",
            warnings=tuple(warnings),
        )
        logger.debug("校正結果: {}", calibration)
        for message in warnings:
            logger.warning(message)
        return calibration
