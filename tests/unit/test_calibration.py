"""牌桌校正測試 —— auto / manual / fallback 三條路徑。"""

from __future__ import annotations

import numpy as np
import pytest

from majsoul_copilot.calibration.table import Calibration, TableCalibrator
from majsoul_copilot.capture.base import Frame, WindowInfo
from majsoul_copilot.config.models import CalibrationConfig
from majsoul_copilot.utils.geometry import NormRect, Rect, Size
from tests.conftest import make_letterboxed


def _frame(image: np.ndarray, window: WindowInfo, scale: float = 1.0) -> Frame:
    return Frame(image=image, window=window, scale=scale)


class TestAutoCalibration:
    def test_detects_letterboxed_canvas_exactly(self, window: WindowInfo) -> None:
        canvas = Rect(0, 90, 1280, 720)  # 正好 16:9
        frame = _frame(make_letterboxed((1280, 900), canvas), window)

        calibration = TableCalibrator().calibrate(frame)

        assert calibration.table_rect == canvas
        assert calibration.source == "auto"
        assert calibration.is_reliable
        assert calibration.warnings == ()

    def test_off_aspect_warns_but_does_not_crop_by_default(self, window: WindowInfo) -> None:
        """預設 enforce_aspect=False:只示警,不動矩形。

        實測 macOS Steam 版雀魂會把畫面撐滿視窗內容區(1.796 而非 16:9),
        此時強制裁切會讓所有 ROI 偏移,所以預設不裁。
        """
        content = Rect(50, 50, 800, 600)  # 4:3,遠離 16:9
        frame = _frame(make_letterboxed((900, 700), content), window)

        calibration = TableCalibrator().calibrate(frame)

        assert calibration.source == "auto"
        assert calibration.warnings, "偏離預期比例時必須留下警告"
        assert not calibration.is_reliable
        assert calibration.table_rect == content, "預設不應改動偵測到的矩形"

    def test_off_aspect_crops_when_enforce_aspect_enabled(self, window: WindowInfo) -> None:
        content = Rect(50, 50, 800, 600)
        frame = _frame(make_letterboxed((900, 700), content), window)

        calibration = TableCalibrator(CalibrationConfig(enforce_aspect=True)).calibrate(frame)

        assert calibration.warnings
        assert calibration.aspect == pytest.approx(16 / 9, rel=0.01)
        assert content.contains(calibration.table_rect)

    def test_within_tolerance_produces_no_warning(self, window: WindowInfo) -> None:
        canvas = Rect(0, 20, 1280, 715)  # aspect 1.790,偏離 16:9 僅 0.7%
        frame = _frame(make_letterboxed((1280, 800), canvas), window)

        calibration = TableCalibrator().calibrate(frame)

        assert calibration.warnings == ()
        assert calibration.is_reliable

    def test_scale_is_carried_through_from_frame(self, window: WindowInfo) -> None:
        frame = _frame(make_letterboxed((1280, 900), Rect(0, 90, 1280, 720)), window, scale=2.0)
        assert TableCalibrator().calibrate(frame).scale == 2.0


class TestFallback:
    def test_uniform_image_falls_back_to_whole_frame(self, window: WindowInfo) -> None:
        frame = _frame(np.zeros((720, 1280, 3), dtype=np.uint8), window)

        calibration = TableCalibrator().calibrate(frame)

        assert calibration.source == "fallback"
        assert calibration.table_rect == Rect(0, 0, 1280, 720)
        assert not calibration.is_reliable
        assert calibration.warnings


class TestManualOverride:
    def test_manual_rect_bypasses_detection(self, window: WindowInfo) -> None:
        # 影像內容刻意與手動指定的矩形不符,驗證確實沒跑自動偵測
        frame = _frame(make_letterboxed((1000, 1000), Rect(0, 0, 1000, 1000)), window)
        config = CalibrationConfig(manual_table_rect=(0.1, 0.2, 0.5, 0.25))

        calibration = TableCalibrator(config).calibrate(frame)

        assert calibration.source == "manual"
        assert calibration.table_rect == Rect(100, 200, 500, 250)
        assert calibration.warnings == ()


class TestCalibrationHelpers:
    @pytest.fixture
    def calibration(self, window: WindowInfo) -> Calibration:
        frame = _frame(make_letterboxed((1280, 900), Rect(0, 90, 1280, 720)), window)
        return TableCalibrator().calibrate(frame)

    def test_roi_to_pixels_is_relative_to_table_not_image(
        self, calibration: Calibration
    ) -> None:
        """ROI 的原點必須是牌桌矩形,不是整張影像 —— 這裡差 90 像素。"""
        roi = NormRect(0.0, 0.0, 0.5, 0.5)
        assert calibration.roi_to_pixels(roi) == Rect(0, 90, 640, 360)

    def test_roi_roundtrip(self, calibration: Calibration) -> None:
        roi = NormRect(0.3, 0.6, 0.2, 0.1)
        restored = calibration.pixels_to_roi(calibration.roi_to_pixels(roi))
        assert restored.x == pytest.approx(roi.x, abs=1e-3)
        assert restored.height == pytest.approx(roi.height, abs=1e-3)

    def test_crop_returns_table_region(self, calibration: Calibration) -> None:
        cropped = calibration.crop(np.zeros((900, 1280, 3), dtype=np.uint8))
        assert cropped.shape == (720, 1280, 3)

    def test_matches_detects_resize(
        self, calibration: Calibration, window: WindowInfo
    ) -> None:
        same = _frame(np.zeros((900, 1280, 3), dtype=np.uint8), window)
        resized = _frame(np.zeros((600, 800, 3), dtype=np.uint8), window)
        assert calibration.matches(same)
        assert not calibration.matches(resized), "影像尺寸改變時必須要求重新校正"

    def test_image_size_recorded(self, calibration: Calibration) -> None:
        assert calibration.image_size == Size(1280, 900)
