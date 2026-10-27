"""重播器的排程與計分。

這一支測的不是辨識,是**重播的語意對不對** —— 錄影要被當成「一秒醒一次、
手上只有最新那張」來餵,而不是均勻取樣。排程錯了的話整份報告看起來完全
正常,只是量到的不是實際跑起來的那條路。
"""

from __future__ import annotations

from mia.eval.replay import ReplayReport, SeatScore, _grade_grid, _grade_members, schedule
from mia.recorder.session import FrameRecord, SessionManifest, SessionReader


class _Session:
    """只有 manifest 的假 session。排程只看時間戳,不讀像素。"""

    def __init__(self, stamps: list[float], *, first_wall: float = 1000.0) -> None:
        self.manifest = SessionManifest(
            first_frame_wall=first_wall,
            frames=[
                FrameRecord(index=i, timestamp=t, image=f"{i:06d}.jpg", changed=True)
                for i, t in enumerate(stamps)
            ],
        )
        self._wall = first_wall

    def wall_at(self, timestamp: float) -> float:
        return self._wall + timestamp


class TestTheSchedule:
    def test_a_tick_sees_only_the_newest_frame_of_its_second(self) -> None:
        """一秒內來了三張,看得到的只有第三張 —— ``offer_frame`` 只留最新的。"""
        ticks = list(schedule(_Session([0.0, 0.3, 0.6, 0.9]), interval=1.0))
        assert [t.record.index for t in ticks if t.record] == [3]

    def test_it_is_not_every_nth_frame(self) -> None:
        """照時間醒,不是每 n 幀取一張。

        錄影的幀間隔**不均勻** —— 去重規則是「畫面變了才存」,靜止的時候
        同一個檔案被好幾幀共用。均勻取樣會在動畫密集的那幾秒多讀好幾輪,
        而實際跑的迴圈不會。
        """
        # 前兩秒各來一張,第三秒塞了五張 —— 第三秒只讀一次,不是五次。
        stamps = [0.5, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5]
        ticks = list(schedule(_Session(stamps), interval=1.0))
        assert [t.record.index for t in ticks if t.record] == [1, 6]

    def test_a_second_with_no_frame_is_a_tick_with_none(self) -> None:
        """沒有新畫面的那一輪照樣要醒。

        ``TableWorker.tick`` 對 ``None`` 有自己的反應(說「等畫面辨識把牌桌
        送過來」),那也是實際會發生的事 —— 擷取那條路卡住、視窗被遮。
        過濾掉就量不到了。
        """
        ticks = list(schedule(_Session([0.0, 5.5]), interval=1.0))
        assert [t.record is None for t in ticks] == [False, True, True, True, True, False]

    def test_an_empty_recording_schedules_nothing(self) -> None:
        assert list(schedule(_Session([]), interval=1.0)) == []

    def test_the_wall_clock_comes_from_the_session(self) -> None:
        """對分要用牆上時鐘 —— 那是畫面與封包兩個行程之間唯一共通的基準。"""
        ticks = list(schedule(_Session([0.0, 1.5], first_wall=777.0), interval=1.0))
        assert [t.wall for t in ticks] == [778.0, 779.0]


class TestTheGrid:
    """格線逐格比,**位置也算**。少一格會讓後面整串位移。"""

    def test_a_right_tile_counts_as_right(self) -> None:
        score = SeatScore()
        _grade_grid(score, ["1m", "2p"], ["1m", "2p"])
        assert (score.right, score.wrong, score.abstained) == (2, 0, 0)

    def test_an_abstention_is_not_an_error(self) -> None:
        """``?`` 會被放銃分析當成危險 —— 方向保守,不算錯。"""
        score = SeatScore()
        _grade_grid(score, ["1m", "2p"], ["1m", "?"])
        assert (score.right, score.abstained, score.wrong) == (1, 1, 0)

    def test_a_confident_misread_is_the_one_that_matters(self) -> None:
        score = SeatScore()
        _grade_grid(score, ["1m"], ["9s"])
        assert score.wrong == 1

    def test_a_red_five_is_not_a_misread(self) -> None:
        """CV 刻意把赤五併到普通五(``merge_red``)—— 比的時候要先還原,
        不然每張赤五都會被記成一次靜默錯誤。"""
        score = SeatScore()
        _grade_grid(score, ["5mr"], ["5m"])
        assert (score.right, score.wrong) == (1, 0)

    def test_a_missing_cell_is_counted_short(self) -> None:
        score = SeatScore()
        _grade_grid(score, ["1m", "2p", "3s"], ["1m"])
        assert (score.right, score.short) == (1, 2)

    def test_extra_cells_are_counted_long(self) -> None:
        score = SeatScore()
        _grade_grid(score, ["1m"], ["1m", "2p"])
        assert (score.right, score.long) == (1, 1)


class TestTheTrackerRiver:
    """tracker 的河比**集合**,不比位置。

    棄權的牌根本沒送進 tracker(``_read_seat`` 跳過 ``?``),所以它的河比
    畫面上的短 —— 位置本來就對不上。而放銃分析只問「這張是不是現物」。
    """

    def test_a_shorter_river_is_not_an_error(self) -> None:
        score = SeatScore()
        _grade_members(score, ["1m", "2p", "3s"], ["1m", "3s"])
        assert (score.held, score.leaked) == (2, 0)

    def test_a_tile_that_was_never_discarded_is_a_fake_safe_tile(self) -> None:
        """這一欄是**無症狀的失敗**:放銃分析會說那張一定安全。"""
        score = SeatScore()
        _grade_members(score, ["1m"], ["1m", "9s"])
        assert (score.held, score.leaked) == (1, 1)

    def test_counting_is_by_multiset_not_by_set(self) -> None:
        """同一張牌可以被打兩次 —— 第三次就是多出來的。"""
        score = SeatScore()
        _grade_members(score, ["1m", "1m"], ["1m", "1m", "1m"])
        assert (score.held, score.leaked) == (2, 1)


class TestTheReport:
    def test_an_overrun_is_a_tick_that_did_not_keep_up(self) -> None:
        report = ReplayReport(interval=1.0, durations=[0.3, 0.9, 1.4, 2.0])
        assert report.overrun == 2

    def test_it_summarizes_without_any_data(self) -> None:
        """一局都還沒開始就跑完的素材不該讓報告炸掉 —— 除以零在等著。"""
        assert "沒有一輪評得到分" in ReplayReport().summary()


def test_the_real_session_reader_has_what_the_schedule_needs() -> None:
    """假 session 只實作了 ``manifest`` 與 ``wall_at``。真的那個也得有 ——
    不然這一支測的是一個不存在的介面。"""
    assert hasattr(SessionReader, "wall_at")
    assert "manifest" in SessionReader.__init__.__code__.co_names
