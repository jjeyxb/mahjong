"""ROI 解析層測試。"""

from __future__ import annotations

import numpy as np
import pytest

from mia.calibration.table import Calibration
from mia.config.loader import load_config
from mia.config.models import RoiConfig
from mia.utils.geometry import Rect, Size
from mia.vision.roi import MissingRoiError, RoiSet, roi_names

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
        assert roi_names() == ("own_hand",)

    def test_scope_is_own_hand_only(self) -> None:
        """牌河/副露/寶牌/HUD 改由封包提供,不該再出現在 CV 的 ROI 清單裡。

        見 docs/decisions.md。這個測試存在的目的是:如果有人手癢把那些欄位
        加回設定模型,會立刻被擋下來,而不是默默多出沒有消費者的 ROI。
        """
        names = set(roi_names())
        for gone in ("own.river", "own.melds", "kamicha.melds", "toimen.river",
                     "shimocha.river", "dora_indicators", "round_info"):
            assert gone not in names


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

    def test_full_table_roi_reproduces_the_table_rect(self) -> None:
        config = RoiConfig(own_hand=(0.0, 0.0, 1.0, 1.0))
        rois = RoiSet(config, make_calibration())
        assert rois["own_hand"].rect == Rect(100, 200, 1000, 500)


class TestMissing:
    def test_unmeasured_roi_raises_instead_of_guessing(self) -> None:
        """沒量測的 ROI 必須炸掉。給預設框硬跑會讓準確率不明不白地變差。"""
        rois = RoiSet(RoiConfig(), make_calibration())
        with pytest.raises(MissingRoiError, match="尚未量測"):
            rois["own_hand"]

    def test_missing_lists_every_unmeasured_roi(self) -> None:
        rois = RoiSet(RoiConfig(), make_calibration())
        assert set(roi_names()) == set(rois.missing)
        assert rois.available == ()

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

    def test_every_roi_is_measured(self) -> None:
        rois = RoiSet(load_config().roi, make_calibration())
        assert rois.missing == (), f"這些 ROI 還沒量測: {rois.missing}"

    def test_every_roi_stays_inside_the_table_rect(self) -> None:
        table = Rect(0, 0, 2560, 1440)
        rois = RoiSet(load_config().roi, make_calibration(table, Size(2560, 1440)))
        for roi in rois:
            assert table.contains(roi.rect), f"{roi.name} 超出牌桌矩形: {roi.rect}"

    def test_own_hand_covers_the_drawn_fourteenth_tile(self) -> None:
        """實測摸牌那張的右緣在 0.8218,框沒蓋到就會漏掉剛摸進來的牌。"""
        own_hand = RoiSet(load_config().roi, make_calibration())["own_hand"].norm
        assert own_hand.x + own_hand.width > 0.8218


class TestHeadroom:
    """比對牌面時要往 ROI 上方多切一段,給滑鼠抬起來的那張牌。

    第二個回傳值是「原本的 ROI 從第幾列開始」—— 算錯就整個牌框歪掉,
    而歪掉的結果看起來仍然像一副正常的手牌。
    """

    def _set(self, roi: tuple[float, float, float, float]) -> RoiSet:
        return RoiSet(RoiConfig(own_hand=roi), make_calibration())

    def test_it_takes_the_extra_rows_from_above(self) -> None:
        rois = self._set((0.0, 0.5, 1.0, 0.4))
        image = np.zeros((DEFAULT_IMAGE.height, DEFAULT_IMAGE.width, 3), np.uint8)
        plain = rois.crop(image, "own_hand")

        tall, headroom = rois.crop_with_headroom(image, "own_hand", 40)

        assert headroom == 40
        assert tall.shape[0] == plain.shape[0] + 40
        assert tall.shape[1] == plain.shape[1]

    def test_the_original_roi_starts_at_the_reported_offset(self) -> None:
        rois = self._set((0.0, 0.5, 1.0, 0.4))
        image = np.zeros((DEFAULT_IMAGE.height, DEFAULT_IMAGE.width, 3), np.uint8)
        # 讓 ROI 那一塊帶標記,才看得出偏移對不對
        rect = rois["own_hand"].rect
        image[rect.as_slice()] = 200

        tall, headroom = rois.crop_with_headroom(image, "own_hand", 40)

        assert (tall[headroom:] == 200).all()
        assert (tall[:headroom] == 0).all()

    def test_it_stops_at_the_top_of_the_image(self) -> None:
        """ROI 貼近影像上緣時要帶少一點,而且要**說**帶了多少 ——
        假設就是傳進去的那個值會讓牌框整個往上錯位。"""
        rois = self._set((0.0, 0.0, 1.0, 0.4))
        image = np.zeros((DEFAULT_IMAGE.height, DEFAULT_IMAGE.width, 3), np.uint8)
        top = rois["own_hand"].rect.y

        tall, headroom = rois.crop_with_headroom(image, "own_hand", 10_000)

        assert headroom == top
        assert tall.shape[0] == rois["own_hand"].rect.height + top

    def test_zero_headroom_is_just_a_crop(self) -> None:
        rois = self._set((0.0, 0.5, 1.0, 0.4))
        image = np.zeros((DEFAULT_IMAGE.height, DEFAULT_IMAGE.width, 3), np.uint8)

        tall, headroom = rois.crop_with_headroom(image, "own_hand", 0)

        assert headroom == 0
        assert tall.shape == rois.crop(image, "own_hand").shape

    def test_a_wrong_sized_image_is_still_rejected(self) -> None:
        rois = self._set((0.0, 0.5, 1.0, 0.4))
        with pytest.raises(ValueError, match="不適用"):
            rois.crop_with_headroom(np.zeros((10, 10, 3), np.uint8), "own_hand", 40)
