"""準確率報告的統計邏輯。

**這裡測的是統計本身,不是 CV 準確率。** 真正的準確率要有成對的錄影才量得出來
(見 ``tools/evaluate.py``),那份素材目前還沒錄。這裡確保的是:給定已知的
「答案 vs 預測」,報告算出來的數字是對的 —— 否則等素材錄好了,也分不清難看的
數字是 CV 差還是統計寫錯。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import pytest

from majsoul_copilot.eval.align import AlignedFrame, HandState
from majsoul_copilot.eval.report import FrameResult, evaluate_frame, summarize
from majsoul_copilot.vision.tiles.classify import TemplateSet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    """37 張牌面模板。載入一次全檔共用。"""
    return TemplateSet.load()


def result(truth: str, predicted: str, *, index: int = 0) -> FrameResult:
    """``"1m 2m"`` 這種寫法比一串 tuple 好讀,而且不容易數錯張數。"""
    t, p = tuple(sorted(truth.split())), tuple(sorted(predicted.split()))
    return FrameResult(index, t, p, count_ok=len(t) == len(p), low_confidence=0)


class TestFrameResult:
    def test_an_identical_hand_is_exact(self) -> None:
        assert result("1m 2m 3m", "1m 2m 3m").exact

    def test_one_wrong_tile_is_not_exact(self) -> None:
        """整手正確率是實際使用時該看的 —— 向聽計算吃的是整手牌,錯一張就全錯。"""
        frame = result("1m 2m 3m", "1m 2m 4m")
        assert not frame.exact
        assert frame.correct_tiles == 2

    def test_correct_tiles_uses_multiset_intersection(self) -> None:
        """逐位比對會因為一張錯位就整串往後錯,那個數字低估到沒有參考價值。

        這裡少認一張 3m、多認一張 4m,其餘 12 張都對 —— 該算 12 對。
        """
        truth = "1m 1m 2m 2m 3m 3m 4m 4m 5m 5m 6m 6m 7m"
        predicted = "1m 1m 2m 2m 3m 4m 4m 4m 5m 5m 6m 6m 7m"
        assert result(truth, predicted).correct_tiles == 12

    def test_a_wrong_count_is_flagged(self) -> None:
        assert not result("1m 2m 3m", "1m 2m").count_ok


class TestSummary:
    def test_accuracies_are_computed_over_all_frames(self) -> None:
        report = summarize(
            [
                result("1m 2m 3m", "1m 2m 3m"),
                result("1m 2m 3m", "1m 2m 4m"),
            ]
        )
        assert report.exact_accuracy == 0.5
        assert report.tile_accuracy == pytest.approx(5 / 6)
        assert report.count_accuracy == 1.0

    def test_an_empty_report_does_not_divide_by_zero(self) -> None:
        report = summarize([])
        assert report.exact_accuracy == 0.0
        assert "沒有任何可評分的幀" in report.summary()

    def test_confusions_record_the_direction(self) -> None:
        """「9s 被認成 4s」是可以修的,「準確率 93%」不是。"""
        report = summarize([result("1m 2m 9s", "1m 2m 4s")] * 3)
        assert report.confusions[("9s", "4s")] == 3

    def test_confusions_skip_frames_with_a_wrong_count(self) -> None:
        """張數不對時無法確定哪張對應哪張,硬配出來的對照表只是雜訊。"""
        report = summarize([result("1m 2m 3m", "1m 2m")])
        assert report.confusions == {}

    def test_the_summary_lists_the_worst_confusions(self) -> None:
        report = summarize([result("1m 9s", "1m 4s")] * 2)
        assert "9s → 4s" in report.summary()


class TestRedFives:
    def test_a_red_five_read_as_a_normal_five_is_caught(self) -> None:
        """赤五混進普通五的話,總體準確率只掉一點點,很容易被忽略過去。

        赤五與普通五在向聽計算上等價,但打點差一番 —— 必須分開統計。
        """
        report = summarize([result("5mr 1m", "5m 1m")])
        assert report.red_five_recall == 0.0
        assert report.tile_accuracy == 0.5

    def test_a_correct_red_five_counts(self) -> None:
        assert summarize([result("5mr 1m", "5mr 1m")]).red_five_recall == 1.0

    def test_no_red_five_in_the_truth_reports_nan(self) -> None:
        """回 0 會被讀成「認不出來」、回 1 會被讀成「全對」,而實際上是「沒測到」。"""
        assert math.isnan(summarize([result("1m 2m", "1m 2m")]).red_five_recall)


class TestEvaluateFrame:
    """跑完整的辨識管線。素材是 M3 留下的三張參考畫面。

    標準答案取自管線自己的輸出 —— 這裡驗的是**接線**(座標系、記法轉換、
    張數),不是辨識正確率。牌名的正確性由 ``test_vision_classify.py`` 用
    目視確認過的答案負責。
    """

    def _frame(self, truth: tuple[str, ...]) -> AlignedFrame:
        return AlignedFrame(0, 0.0, HandState(0.0, truth, None, 0, True))

    @pytest.mark.parametrize(
        ("name", "tiles"),
        [("hand_13_full", 13), ("hand_14_with_draw", 14), ("hand_4_with_melds", 4)],
    )
    def test_the_tile_count_matches_the_image(
        self, name: str, tiles: int, templates: TemplateSet
    ) -> None:
        image = cv2.imread(str(FIXTURES / f"{name}.png"))
        outcome = evaluate_frame(image, self._frame(("1m",) * tiles), templates)
        assert len(outcome.predicted) == tiles

    def test_predictions_come_back_in_mjai_notation(self, templates: TemplateSet) -> None:
        """classify 回的是雀魂記法(0m/1z),標準答案是 MJAI 記法(5mr/E)。

        少了這層轉換,每一張字牌與赤寶牌都會被算成認錯,而總體準確率仍有
        七成左右 —— 剛好難看到會讓人去查 CV,又不夠難看到會讓人懷疑轉換。
        """
        image = cv2.imread(str(FIXTURES / "hand_4_with_melds.png"))
        outcome = evaluate_frame(image, self._frame(("N", "N", "C", "C")), templates)
        assert all(not t[-1].isdigit() or t[-1] in "0123456789" for t in outcome.predicted)
        assert not any(t.endswith("z") for t in outcome.predicted), "不該留下雀魂的字牌記法"

    def test_a_matching_truth_scores_perfectly(self, templates: TemplateSet) -> None:
        """把管線自己的輸出當答案 —— 應該是 100%。不是的話就是比對邏輯壞了。"""
        image = cv2.imread(str(FIXTURES / "hand_13_full.png"))
        first = evaluate_frame(image, self._frame(("1m",) * 13), templates)
        again = evaluate_frame(image, self._frame(first.predicted), templates)
        assert again.exact
        assert summarize([again]).tile_accuracy == 1.0
