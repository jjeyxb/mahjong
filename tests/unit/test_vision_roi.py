"""ROI 解析層測試。"""

from __future__ import annotations

import numpy as np
import pytest

from majsoul_copilot.calibration.table import Calibration
from majsoul_copilot.config.loader import load_config
from majsoul_copilot.config.models import RoiConfig, SeatRoiConfig
from majsoul_copilot.utils.geometry import Rect, Size
from majsoul_copilot.vision.grid import COLS, ROWS
from majsoul_copilot.vision.roi import MissingRoiError, RoiSet, roi_names

#: 刻意讓牌桌矩形有偏移、且不等於整張影像 —— 兩者相同的話,
#: 「換算時忘了加牌桌原點」這個錯誤剛好會被蓋掉,測不出來。
DEFAULT_TABLE = Rect(100, 200, 1000, 500)
DEFAULT_IMAGE = Size(1400, 900)


def make_calibration(table: Rect | None = None, image: Size | None = None) -> Calibration:
    return Calibration(
        table_rect=table or DEFAULT_TABLE, image_size=image or DEFAULT_IMAGE
    )


class TestRoiNames:
    def test_names_are_derived_from_the_config_model(self) -> None:
        names = roi_names()
        assert "own_hand" in names
        assert "own.river" in names
        assert "kamicha.melds" in names
        assert "dora_indicators" in names

    def test_every_seat_has_both_river_and_melds(self) -> None:
        names = set(roi_names())
        for seat in ("own", "kamicha", "toimen", "shimocha"):
            assert f"{seat}.river" in names
            assert f"{seat}.melds" in names

    def test_no_nested_name_leaks_without_its_prefix(self) -> None:
        """巢狀欄位只能以 seat.field 出現,不能有裸的 'river'。"""
        assert "river" not in roi_names()
        assert "melds" not in roi_names()


class TestResolution:
    def test_normalized_coords_become_pixels_relative_to_the_whole_image(self) -> None:
        """ROI 是相對牌桌矩形的,換算後必須加回牌桌本身的偏移。"""
        config = RoiConfig(own_hand=(0.5, 0.5, 0.25, 0.5))
        rois = RoiSet(config, make_calibration())
        # 牌桌 1000x500@(100,200):x = 100 + 0.5*1000,y = 200 + 0.5*500
        assert rois["own_hand"].rect == Rect(600, 450, 250, 250)

    def test_same_roi_maps_to_different_pixels_at_different_resolutions(self) -> None:
        """同一份設定要能同時適用於不同解析度 —— 這就是用正規化座標的理由。"""
        config = RoiConfig(own_hand=(0.1, 0.2, 0.4, 0.3))
        small = RoiSet(config, make_calibration(Rect(0, 0, 1280, 720), Size(1280, 720)))
        large = RoiSet(config, make_calibration(Rect(0, 0, 2560, 1440), Size(2560, 1440)))
        assert large["own_hand"].rect == small["own_hand"].rect.scaled(2.0)

    def test_nested_seat_roi_is_reachable_by_dotted_name(self) -> None:
        config = RoiConfig(toimen=SeatRoiConfig(river=(0.0, 0.0, 1.0, 1.0)))
        rois = RoiSet(config, make_calibration())
        assert rois["toimen.river"].rect == Rect(100, 200, 1000, 500)


class TestMissing:
    def test_unmeasured_roi_raises_instead_of_guessing(self) -> None:
        """沒量測的 ROI 必須炸掉。給預設框硬跑會讓準確率不明不白地變差。"""
        rois = RoiSet(RoiConfig(), make_calibration())
        with pytest.raises(MissingRoiError, match="尚未量測"):
            rois["own_hand"]

    def test_missing_lists_every_unmeasured_roi(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        assert set(roi_names()) <= set(rois.missing)
        assert rois.available == ()

    def test_missing_also_covers_the_river_grids(self) -> None:
        """牌河網格與矩形 ROI 分開存,但「還沒量」要一起報,不能只漏報一半。"""
        rois = RoiSet(RoiConfig(), make_calibration())
        assert "river_grid.own" in rois.missing
        assert rois.grids == {}

    def test_unstaked_grid_raises_with_the_tool_name(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        with pytest.raises(MissingRoiError, match="grid_annotate"):
            rois.grid("own")

    def test_unknown_name_is_a_different_message_than_unmeasured(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        with pytest.raises(MissingRoiError, match="沒有名為"):
            rois["own.hand"]  # 正確的是 own_hand

    def test_get_returns_none_without_raising(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        assert rois.get("own_hand") is None


class TestCrop:
    def test_crop_takes_the_right_pixels(self) -> None:
        calibration = make_calibration(Rect(10, 20, 100, 50), Size(200, 100))
        rois = RoiSet(RoiConfig(own_hand=(0.0, 0.0, 0.5, 1.0)), calibration)
        image = np.zeros((100, 200, 3), np.uint8)
        image[20:70, 10:60] = 255  # 正好是預期的 ROI 範圍

        crop = rois.crop(image, "own_hand")
        assert crop.shape[:2] == (50, 50)
        assert crop.min() == 255  # 全部落在塗白的區域內

    def test_crop_rejects_an_image_of_the_wrong_size(self) -> None:
        """視窗大小變了卻沿用舊的 RoiSet,是會整片歪掉又難查的錯誤。"""
        rois = RoiSet(RoiConfig(own_hand=(0.0, 0.0, 1.0, 1.0)), make_calibration())
        with pytest.raises(ValueError, match="不符"):
            rois.crop(np.zeros((10, 10, 3), np.uint8), "own_hand")

    def test_matches_detects_a_recalibration(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        assert rois.matches(make_calibration())
        assert not rois.matches(make_calibration(Rect(0, 0, 1000, 500)))


class TestAgainstProjectConfig:
    """對真正的 config/default.yaml 做煙霧測試。"""

    def test_all_rectangle_rois_are_measured(self) -> None:
        rois = RoiSet(load_config().roi, make_calibration())
        unmeasured = [n for n in rois.missing if not n.startswith("river_grid.")]
        assert unmeasured == [], f"這些 ROI 還沒量測: {unmeasured}"

    def test_every_roi_stays_inside_the_table_rect(self) -> None:
        table = Rect(0, 0, 2560, 1440)
        rois = RoiSet(load_config().roi, make_calibration(table, Size(2560, 1440)))
        for roi in rois:
            assert table.contains(roi.rect), f"{roi.name} 超出牌桌矩形: {roi.rect}"

    def test_all_four_river_grids_are_calibrated(self) -> None:
        rois = RoiSet(load_config().roi, make_calibration())
        assert set(rois.grids) == {"own", "kamicha", "toimen", "shimocha"}

    def test_every_river_cell_stays_inside_the_table(self) -> None:
        table = Rect(0, 0, 2560, 1440)
        rois = RoiSet(load_config().roi, make_calibration(table, Size(2560, 1440)))
        for seat, grid in rois.grids.items():
            for row in range(ROWS):
                for col in range(COLS):
                    cell = grid.cell_bounds(col, row)
                    assert table.contains(cell), f"{seat} 的格 ({col},{row}) 跑出牌桌: {cell}"

    def test_river_cells_are_roughly_tile_shaped(self) -> None:
        """格子若明顯不是牌的形狀,通常代表四個角的順序標錯或拖歪了。"""
        rois = RoiSet(load_config().roi, make_calibration(Rect(0, 0, 2560, 1440),
                                                          Size(2560, 1440)))
        for seat, grid in rois.grids.items():
            for row in range(ROWS):
                for col in range(COLS):
                    cell = grid.cell_bounds(col, row)
                    assert 40 < cell.width < 140, f"{seat} ({col},{row}) 寬度怪異: {cell}"
                    assert 40 < cell.height < 140, f"{seat} ({col},{row}) 高度怪異: {cell}"

    def test_own_hand_covers_the_drawn_fourteenth_tile(self) -> None:
        """實測摸牌那張的右緣在 0.8218,框沒蓋到就會漏掉剛摸進來的牌。"""
        own_hand = RoiSet(load_config().roi, make_calibration())["own_hand"].norm
        assert own_hand.x + own_hand.width > 0.8218
