"""牌河評測的統計邏輯。

**這裡測的是統計本身,不是 CV 準確率。** 真正的準確率要有成對的錄影才量得
出來(見 ``tools/eval_rivers.py``)。這裡確保的是:給定已知的「答案 vs 預測」,
報告算出來的數字是對的 —— 否則素材跑出難看的數字時,分不清是 CV 差還是
統計寫錯。
"""

from __future__ import annotations

import pytest

from mia.eval.align import stable_index
from mia.eval.rivers import (
    RiverFrameResult,
    RiverState,
    RiverTimeline,
    summarize_rivers,
)
from mia.vision.tiles.river import SEATS


def result(
    truth: str, predicted: str, *, seat: str = "own", confident: str | None = None
) -> RiverFrameResult:
    """``"1m 2m"`` 這種寫法比一串 tuple 好讀。``confident`` 用 ``"y n y"``。"""
    t, p = tuple(truth.split()), tuple(predicted.split())
    flags = tuple(c == "y" for c in confident.split()) if confident else (True,) * len(p)
    return RiverFrameResult(
        index=0, seat=seat, truth=t, predicted=p, confident=flags, contiguous=True
    )


class TestStableIndex:
    """手牌與牌河共用的「現在可以相信畫面嗎」。"""

    def test_a_settled_state_is_returned(self) -> None:
        assert stable_index([100.0], 101.0, 0.4) == 0

    def test_nothing_before_the_first_state(self) -> None:
        assert stable_index([100.0], 99.0, 0.4) is None

    def test_a_frame_during_the_animation_is_refused(self) -> None:
        """封包已經到了,但畫面還在演動畫 —— 那時畫面與封包必然不一致。"""
        assert stable_index([100.0], 100.2, 0.4) is None

    def test_a_frame_just_before_the_next_change_is_refused(self) -> None:
        """下一個狀態已經在路上,畫面隨時會變。"""
        assert stable_index([100.0, 102.0], 101.8, 0.4) is None


class TestRiverFrameResult:
    def test_position_matters(self) -> None:
        """牌河是有序的 —— 同一堆牌換個順序不算對。"""
        assert result("1m 2m 3m", "1m 3m 2m").correct == 1

    def test_a_short_read_shifts_everything(self) -> None:
        """少讀一格,後面整串往前位移 —— 從「錯一張」變成「幾乎全錯」。

        這就是張數要先對、才輪得到比牌面的原因。
        """
        frame = result("1m 2m 3m 4m", "2m 3m 4m")
        assert not frame.count_ok
        assert not frame.accepted
        assert frame.correct == 0

    def test_silent_errors_only_count_the_confident_ones(self) -> None:
        """沒信心又認錯的,上層本來就會丟掉,不算靜默。"""
        frame = result("1m 2m 3m", "1m 9p 9s", confident="y y n")
        assert frame.silent_errors == 1

    def test_a_hole_makes_the_frame_unusable(self) -> None:
        """牌河中間有洞代表攤平歪了,或畫面上根本不是牌桌。"""
        frame = RiverFrameResult(
            index=0,
            seat="own",
            truth=("1m",),
            predicted=("1m",),
            confident=(True,),
            contiguous=False,
        )
        assert frame.count_ok
        assert not frame.accepted, "張數對,但不連續 —— 仍然不該採用"


class TestRiverReport:
    def test_accuracy_only_counts_accepted_frames(self) -> None:
        """被拒絕的幀不進準確率 —— 即時路徑本來就不會用它們。"""
        good = result("1m 2m", "1m 2m")
        bad = RiverFrameResult(
            index=1,
            seat="own",
            truth=("1m", "2m"),
            predicted=("9p", "9s"),
            confident=(True, True),
            contiguous=False,
        )
        report = summarize_rivers([good, bad])
        assert report.tile_accuracy() == 1.0
        assert report.accept_rate() == 0.5

    def test_a_perfect_run_scores_perfectly(self) -> None:
        report = summarize_rivers([result("1m 2m 3m", "1m 2m 3m")] * 3)
        assert report.tile_accuracy() == 1.0
        assert report.silent_error_rate() == 0.0
        assert report.accept_rate() == 1.0

    def test_seats_are_reported_separately(self) -> None:
        """四家的難度差很多 —— 混在一起看不出是哪一家壞掉。"""
        report = summarize_rivers(
            [result("1m", "1m", seat="own"), result("1m", "9p", seat="toimen")]
        )
        assert report.tile_accuracy("own") == 1.0
        assert report.tile_accuracy("toimen") == 0.0
        assert report.tile_accuracy() == 0.5

    def test_confusions_record_the_direction(self) -> None:
        """「9p 被認成白板」是可以修的,「準確率 93%」不是。"""
        report = summarize_rivers([result("9p 1m", "P 1m")] * 3)
        assert report.confusions()[("9p", "P")] == 3

    def test_an_empty_report_does_not_divide_by_zero(self) -> None:
        report = summarize_rivers([])
        assert report.tile_accuracy() == 0.0
        assert "沒有任何可評分的幀" in report.summary()

    def test_the_summary_lists_every_seat(self) -> None:
        report = summarize_rivers([result("1m", "1m", seat=s) for s in SEATS])
        text = report.summary()
        assert all(seat in text for seat in SEATS)
        assert "採用率" in text


class TestRiverTimeline:
    def _timeline(self, *walls: float) -> RiverTimeline:
        states = [RiverState(w, ((f"{int(w)}m",), (), (), ())) for w in walls]
        return RiverTimeline(states=states).freeze()

    def test_a_settled_state_is_returned(self) -> None:
        assert self._timeline(100.0).stable_at(101.0, 0.4) is not None

    def test_a_boundary_is_never_returned(self) -> None:
        """和了動畫、結算、發牌 —— 那段時間畫面上沒有牌河可以評分。

        少了這個哨兵,那十幾秒會每一幀都被判成 CV 認錯,而原因與 CV 無關。
        """
        timeline = RiverTimeline(states=[RiverState(100.0, (), boundary=True)]).freeze()
        assert timeline.stable_at(101.0, 0.4) is None

    def test_the_span_covers_the_states(self) -> None:
        assert self._timeline(100.0, 105.0).span == (100.0, 105.0)

    def test_an_empty_timeline_has_no_span(self) -> None:
        assert RiverTimeline().freeze().span == (0.0, 0.0)

    @pytest.mark.parametrize("wall", [99.0, 100.2])
    def test_unstable_moments_are_refused(self, wall: float) -> None:
        assert self._timeline(100.0).stable_at(wall, 0.4) is None
