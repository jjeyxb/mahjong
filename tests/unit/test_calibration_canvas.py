"""固定畫布尺寸校正測試。

這一組測試對應的是一個**已經真的發生過**的失敗:2026-08-01 錄的第二段素材,
視窗 900x593、瀏覽器工具列一整條沒被剝掉,91% 的候選因寬高比 1.56 被丟棄,
而通過檢查的 9% 恰恰是剝過頭剝進遊戲裡的幀。中位數鎖進去的矩形位移約 0.7
張牌寬,手牌辨識的每張牌正確率從 96.0% 掉到 29.1% —— 全程沒有任何錯誤訊息。

所以這裡驗的是兩件事:

1. 指定畫布尺寸之後,矩形是**算出來**的,而且算得對(:class:`TestTableRect`)。
2. 自動偵測在那種情境下**不准鎖定**(:class:`TestRefusesToLock`)。
"""

from __future__ import annotations

import numpy as np
import pytest

from mia.calibration.canvas import (
    PRESETS,
    Canvas,
    CanvasChoice,
    CanvasMismatchError,
)
from mia.calibration.stable import StableCalibrator
from mia.calibration.table import TableCalibrator
from mia.capture.base import Frame, WindowInfo
from mia.config.models import CalibrationConfig
from mia.utils.geometry import Rect, Size


def _frame(width: int, height: int, window: WindowInfo, scale: float = 1.0) -> Frame:
    rng = np.random.default_rng(3)
    image = rng.integers(80, 256, (height, width, 3), dtype=np.uint8)
    return Frame(image=image, window=window, scale=scale)


class TestParse:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1920x1080", Canvas(1920, 1080)),
            ("1920X1080", Canvas(1920, 1080)),
            ("1920×1080", Canvas(1920, 1080)),  # 使用者從 UI 標籤複製貼上
            (" 1280 x 720 ", Canvas(1280, 720)),
        ],
    )
    def test_accepts(self, text: str, expected: Canvas) -> None:
        assert Canvas.parse(text) == expected

    @pytest.mark.parametrize("text", [None, "", "1920", "abc", "1920x", "0x720", "-1x9"])
    def test_rejects_without_raising(self, text: str | None) -> None:
        """看不懂就回 None,**不能拋例外**。

        這個字串來自設定檔與 ``data/ui_state.json``,兩者都可能被手改壞,
        而「畫布尺寸看不懂」的正確反應是退回自動偵測,不是讓 UI 開不起來。
        """
        assert Canvas.parse(text) is None

    def test_key_round_trips(self) -> None:
        for preset in PRESETS:
            assert Canvas.parse(preset.key) == preset

    def test_presets_are_all_16_by_9(self) -> None:
        """非 16:9 的畫布雀魂一定補黑邊,那就又要偵測黑邊在哪了。"""
        for preset in PRESETS:
            assert preset.aspect == pytest.approx(16 / 9)


class TestTableRect:
    def test_retina_1280x720(self) -> None:
        """實測值:viewport 1280x720、dpr 2 時擷取到 2560x1614。

        數字不是編的 —— 是真的開一個 Chrome for Testing 調到 1280x720 之後
        量的,工具列 87 個邏輯像素。
        """
        rect = Canvas(1280, 720).table_rect(Size(2560, 1614), 2.0)
        assert rect == Rect(0, 174, 2560, 1440)

    def test_non_retina(self) -> None:
        rect = Canvas(1280, 720).table_rect(Size(1280, 807), 1.0)
        assert rect == Rect(0, 87, 1280, 720)

    def test_rejects_wrong_width(self) -> None:
        """寬度對不上就是對不上,不可以硬算一個出來。

        這正是要擋掉的東西:回一個「看起來很正常」的矩形,後面每一步都會
        照常跑完,而錯誤只表現成準確率變差。
        """
        with pytest.raises(CanvasMismatchError, match="影像寬"):
            Canvas(1920, 1080).table_rect(Size(2560, 1614), 2.0)

    def test_tolerates_rounding_of_fractional_dpr(self) -> None:
        """dpr 1.5 之類的螢幕會有 ±1px 的四捨五入,那不算對不上。"""
        rect = Canvas(1280, 720).table_rect(Size(1921, 1211), 1.5)
        assert rect.width == 1920
        assert rect.height == 1080

    def test_rejects_canvas_taller_than_image(self) -> None:
        """選了一個螢幕放不下的尺寸 —— 瀏覽器會給不了,影像就會比畫布矮。"""
        with pytest.raises(CanvasMismatchError, match="瀏覽器介面"):
            Canvas(2560, 1440).table_rect(Size(2560, 1400), 1.0)

    def test_rejects_implausible_chrome_height(self) -> None:
        """畫布下面還剩一大截 —— 多半是選小了,而那個矩形會整個偏掉。"""
        with pytest.raises(CanvasMismatchError, match="瀏覽器介面"):
            Canvas(1280, 720).table_rect(Size(1280, 1200), 1.0)


class TestChoice:
    def test_set_reports_whether_it_changed(self) -> None:
        """回傳「有沒有變」,呼叫端才知道要不要重啟畫面辨識。"""
        choice = CanvasChoice()
        assert choice.set("1920x1080") is True
        assert choice.set("1920x1080") is False
        assert choice.set(None) is True
        assert choice.key is None

    def test_unparseable_falls_back_to_auto(self) -> None:
        choice = CanvasChoice(Canvas(1280, 720))
        assert choice.set("垃圾") is True
        assert choice.value is None


class TestCalibratorUsesCanvas:
    def test_skips_peeling_entirely(self, window: WindowInfo) -> None:
        """畫面內容完全是雜訊(剝除必然失敗),但指定了畫布就該算得出來。"""
        calibrator = TableCalibrator(canvas=CanvasChoice(Canvas(1280, 720)))
        result = calibrator.calibrate(_frame(2560, 1614, window, scale=2.0))
        assert result.source == "canvas"
        assert result.table_rect == Rect(0, 174, 2560, 1440)
        assert result.is_reliable

    def test_config_string_works_without_an_explicit_choice(
        self, window: WindowInfo
    ) -> None:
        """設定檔也能指定 —— 不是只有 UI 那條路。"""
        calibrator = TableCalibrator(CalibrationConfig(canvas="1280x720"))
        result = calibrator.calibrate(_frame(2560, 1614, window, scale=2.0))
        assert result.source == "canvas"

    def test_mismatch_does_not_silently_fall_back_to_peeling(
        self, window: WindowInfo
    ) -> None:
        """對不上時**不准**偷偷換回自動偵測。

        剝除正是使用者選這個尺寸想避開的東西。悄悄換回去,「我明明選了
        1920x1080」與畫面上的結果就對不起來,而且沒有任何線索。
        """
        calibrator = TableCalibrator(canvas=CanvasChoice(Canvas(1920, 1080)))
        result = calibrator.calibrate(_frame(2560, 1614, window, scale=2.0))
        assert result.source == "fallback"
        assert not result.is_reliable
        assert "1920×1080" in result.warnings[0]

    def test_stable_locks_immediately(self, window: WindowInfo) -> None:
        """算出來的東西沒有「不穩定」可言,不必等 15 幀取中位數。"""
        stable = StableCalibrator(canvas=CanvasChoice(Canvas(1280, 720)))
        assert stable.feed(_frame(2560, 1614, window, scale=2.0)) is not None
        assert stable.locked
        assert stable.failure is None

    def test_stable_reports_mismatch_instead_of_locking(
        self, window: WindowInfo
    ) -> None:
        stable = StableCalibrator(canvas=CanvasChoice(Canvas(1920, 1080)))
        assert stable.feed(_frame(2560, 1614, window, scale=2.0)) is None
        assert not stable.locked
        assert stable.failure is not None
        assert "1920×1080" in stable.failure


class TestRefusesToLock:
    """自動偵測丟太多候選時不准鎖定 —— 2026-08-01 那份壞素材的直接對策。"""

    @staticmethod
    def _garbage(window: WindowInfo) -> Frame:
        """剝除必然失敗的一幀(四邊都不是純色)。"""
        return _frame(1280, 846, window)

    def test_blocks_when_most_candidates_are_rejected(
        self, window: WindowInfo
    ) -> None:
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        for _ in range(5):
            stable.feed(self._garbage(window))
        assert not stable.locked
        assert stable.failure is not None
        assert "認不出牌桌邊界" in stable.failure

    def test_says_nothing_before_there_is_enough_evidence(
        self, window: WindowInfo
    ) -> None:
        """才丟掉兩幀就喊失敗會變成雜訊,使用者學會忽略它就白做了。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        for _ in range(2):
            stable.feed(self._garbage(window))
        assert stable.failure is None

    def test_recovers_once_the_game_actually_appears(
        self, window: WindowInfo
    ) -> None:
        """**必須能復原。**

        遊戲停在登入或載入畫面時剝除本來就一直失敗,那段時間累積的丟棄數
        不該一路跟著使用者到牌桌上 —— 用累計而不是滑動視窗的話,登入花了
        兩分鐘的人就再也鎖不上了。
        """
        from tests.conftest import make_letterboxed
        from tests.unit.test_calibration_stable import IMAGE, TRUE_CANVAS

        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        for _ in range(30):  # 一直卡在載入畫面
            stable.feed(self._garbage(window))
        assert stable.failure is not None

        good = Frame(
            image=make_letterboxed(IMAGE, TRUE_CANVAS), window=window, scale=1.0
        )
        result = None
        for _ in range(30):
            result = stable.feed(good) or result
        assert result is not None
        assert result.table_rect == TRUE_CANVAS
        assert stable.failure is None
