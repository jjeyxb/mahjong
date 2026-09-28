"""遮擋驗證的判定規則。

`tools/occlusion_probe.py` 的量測部分需要一台 Windows、一個瀏覽器、一塊真的
螢幕;**判定部分不需要**,而判定才是會寫錯的地方 —— 它有五個分支,其中三個
是「這次測試不算數」。把它抽成 :func:`judge` 就是為了讓這些分支跑得起測試。

這裡釘住的核心性質只有一個:**「遮擋有沒有成立」要比「結果是什麼」先問。**
遮擋沒成立的時候「沒變黑、沒凍結」兩個觀察都是真的,但什麼也沒證明。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/ 不是套件,測試要自己把它加進 sys.path。工具腳本本身也是這樣
# 自己 import mia 的(見 tools/*.py 開頭),pyproject 的 per-file-ignores
# 就是為此開的例外。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from occlusion_probe import COVERAGE_REQUIRED, Sample, judge

LIVE = Sample(brightness=72.0, motion=27.6, distinct=10, frames=10)
FROZEN = Sample(brightness=72.0, motion=0.0, distinct=1, frames=10)
BLACK = Sample(brightness=0.4, motion=0.0, distinct=1, frames=10)


class TestSample:
    def test_a_single_distinct_frame_is_frozen(self) -> None:
        assert FROZEN.is_frozen
        assert not LIVE.is_frozen

    def test_a_dark_frame_is_black(self) -> None:
        assert BLACK.is_black
        assert not LIVE.is_black


class TestJudge:
    def test_a_live_window_under_full_coverage_passes(self) -> None:
        verdict = judge(baseline=LIVE, occluded=LIVE, restored=LIVE, coverage=1.0)
        assert verdict.passed
        assert verdict.code == 0

    def test_a_frozen_frame_under_occlusion_fails(self) -> None:
        """這是整支工具存在的理由 —— 過期畫面不會報錯,只會給錯建議。"""
        verdict = judge(baseline=LIVE, occluded=FROZEN, restored=LIVE, coverage=1.0)
        assert verdict.code == 1
        assert "凍結" in verdict.headline

    def test_a_black_frame_under_occlusion_fails(self) -> None:
        verdict = judge(baseline=LIVE, occluded=BLACK, restored=LIVE, coverage=1.0)
        assert verdict.code == 1
        assert "全黑" in verdict.headline

    def test_insufficient_coverage_invalidates_the_run(self) -> None:
        verdict = judge(baseline=LIVE, occluded=LIVE, restored=LIVE, coverage=0.4)
        assert verdict.code == 2
        assert "無效" in verdict.headline

    @pytest.mark.parametrize("occluded", [LIVE, FROZEN, BLACK])
    def test_coverage_is_checked_before_anything_else(self, occluded: Sample) -> None:
        """覆蓋率不足時,B 階段量到什麼都不該改變結論。

        少了這條,「遮擋根本沒蓋上」會被報成「✓ 通過」——
        那是這支工具最可能騙到人的方式。
        """
        verdict = judge(
            baseline=LIVE, occluded=occluded, restored=LIVE, coverage=COVERAGE_REQUIRED - 0.01
        )
        assert verdict.code == 2
        assert "無效" in verdict.headline

    def test_a_static_baseline_gives_no_conclusion(self) -> None:
        """遊戲停在選單的時候,「沒凍結」是問不出來的。"""
        verdict = judge(baseline=FROZEN, occluded=FROZEN, restored=FROZEN, coverage=1.0)
        assert verdict.code == 2
        assert "沒有結論" in verdict.headline

    def test_a_target_that_stops_after_the_test_gives_no_conclusion(self) -> None:
        """C 階段才靜止 = 目標自己換場景了,B 的「還活著」是碰巧。"""
        verdict = judge(baseline=LIVE, occluded=LIVE, restored=FROZEN, coverage=1.0)
        assert verdict.code == 2
        assert "沒有結論" in verdict.headline

    def test_a_black_baseline_still_reports_the_black_failure(self) -> None:
        """基準線就全黑的話,錯的是擷取而不是遮擋,要報失敗而不是「沒有結論」。"""
        verdict = judge(baseline=BLACK, occluded=BLACK, restored=BLACK, coverage=1.0)
        assert verdict.code == 1
        assert "全黑" in verdict.headline
