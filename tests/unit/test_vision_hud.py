"""HUD 的讀取與它的檢查碼。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.tracker import ReachWatcher, RoundWatcher
from mia.utils.geometry import Rect
from mia.vision.hud import (
    _MINUS,
    Hud,
    HudDigits,
    _glyphs,
    _split_joined,
    read_hud,
    read_turn,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TABLE = Rect(0, 174, 2536, 1430)
#: 東 4 局,余 5,對家立直了一根棒。輪到對家,黃色光條貼在他的點數上方。
SCORES = (39700, 15000, 28400, 15900)


@pytest.fixture(scope="module")
def hud() -> Hud:
    image = cv2.imread(str(FIXTURES / "table_full.jpg"))
    assert image is not None
    return read_hud(image, TABLE, HudDigits.load())


class TestOnARealFrame:
    def test_the_round(self, hud: Hud) -> None:
        assert (hud.bakaze, hud.kyoku, hud.honba, hud.kyotaku) == ("E", 4, 0, 1)

    def test_the_tiles_left(self, hud: Hud) -> None:
        """畫面印 ``05``,前面的零不影響。"""
        assert hud.wall == 5

    def test_all_four_scores_each_turned_upright(self, hud: Hud) -> None:
        assert hud.scores == SCORES

    def test_the_dealer_is_the_red_corner(self, hud: Hud) -> None:
        assert hud.dealer == 1

    def test_the_dealer_gives_away_our_seat(self, hud: Hud) -> None:
        """第 4 局莊家是 3 號座位,他坐我下家 —— 所以我是 2 號。"""
        assert hud.own_seat == 2

    def test_the_scores_add_up(self, hud: Hud) -> None:
        assert hud.scores_agree


class TestTheTurnLight:
    def test_it_is_toimens_turn(self) -> None:
        image = cv2.imread(str(FIXTURES / "table_full.jpg"))
        assert read_turn(image, TABLE) == 2

    def test_a_dark_panel_reads_nothing(self) -> None:
        """燈閃到暗的那一下、結算畫面:沒有一邊亮著就棄權,讓上層沿用上一次的。"""
        blank = np.full((1430, 2536, 3), 30, dtype=np.uint8)
        assert read_turn(blank, Rect(0, 0, 2536, 1430)) is None

    def test_two_lights_read_nothing(self) -> None:
        image = cv2.imread(str(FIXTURES / "table_full.jpg"))
        lit = image.copy()
        # 把對家那條燈複製一份到自己那邊:兩邊都亮,不知道是誰。
        top = (slice(TABLE.y + 448, TABLE.y + 466), slice(1219, 1339))
        bottom = (slice(TABLE.y + 666, TABLE.y + 684), slice(1219, 1339))
        lit[bottom] = image[top]
        assert read_turn(lit, TABLE) is None


class TestItAbstains:
    def test_a_blank_screen_reads_nothing(self) -> None:
        """大廳、讀取畫面:每個欄位各自棄權,而不是讀出一個數字。"""
        blank = np.full((1430, 2536, 3), 30, dtype=np.uint8)
        hud = read_hud(blank, Rect(0, 0, 2536, 1430), HudDigits.load())
        assert hud == Hud()


class TestJoinedDigits:
    def test_a_wide_span_is_two_digits(self) -> None:
        """透視讓相鄰的字在欄位投影上重疊:34600 只切得出四段。"""
        spans = [(0, 30), (32, 48), (50, 66), (68, 84)]
        assert _split_joined(spans) == [(0, 15), (15, 30), (32, 48), (50, 66), (68, 84)]

    def test_a_lone_digit_is_left_alone(self) -> None:
        """只有一段時沒有東西可以比。點數 0 就是這樣。"""
        assert _split_joined([(0, 30)]) == [(0, 30)]


class TestTheChecksum:
    def test_a_misread_digit_does_not_add_up(self) -> None:
        """實測讀掉一位:15900 讀成 5900。"""
        assert not Hud(kyotaku=1, scores=(39700, 5000, 28400, 15900)).scores_agree

    def test_sticks_on_the_table_count(self) -> None:
        assert Hud(kyotaku=0, scores=(25000, 25000, 25000, 25000)).scores_agree
        assert Hud(kyotaku=1, scores=(24000, 25000, 25000, 25000)).scores_agree

    def test_a_missing_field_is_not_a_pass(self) -> None:
        assert not Hud(kyotaku=0, scores=(25000, 25000, 25000, None)).scores_agree
        assert not Hud(scores=(25000, 25000, 25000, 25000)).scores_agree


def _round(**changes: object) -> Hud:
    fields: dict[str, object] = {
        "bakaze": "E",
        "kyoku": 2,
        "wall": 60,
        "honba": 1,
        "kyotaku": 0,
        "scores": (25000, 25000, 25000, 25000),
        "dealer": 3,
    }
    return Hud(**{**fields, **changes})  # type: ignore[arg-type]


class TestNegativeScores:
    """關掉擊飛的友人場,點數可以是負的。實測兩家負分時一局從頭到尾沒有事件流。"""

    @staticmethod
    def _patch(text: str) -> np.ndarray:
        """畫一排字:負號是一條矮的橫線,數字是高的直條,模板不管它們像不像,只量高度。"""
        patch = np.zeros((40, 12 * len(text) + 8, 3), dtype=np.uint8)
        for index, char in enumerate(text):
            left = 4 + index * 12
            if char == "-":
                patch[18:22, left : left + 8] = 255
            else:
                patch[6:34, left : left + 8] = 255
        return patch

    def test_a_minus_is_much_shorter_than_a_digit(self) -> None:
        glyphs = _glyphs(self._patch("-1700"))
        assert len(glyphs) == 5
        assert glyphs[0].shape[0] < _MINUS * glyphs[1].shape[0]

    def test_the_total_still_adds_up(self) -> None:
        negative = _round(kyotaku=0, scores=(79400, 32800, -10500, -1700))
        assert negative.scores_agree

    def test_a_negative_seat_can_be_filled_in(self) -> None:
        hud_read = _round(kyotaku=0, scores=(79400, 32800, None, -1700))
        assert hud_read.filled().scores == (79400, 32800, -10500, -1700)


class TestSticksFromScores:
    """桌面總點數固定,少掉的就是立直棒。模板只到 5。"""

    def test_it_counts_them(self) -> None:
        hud = _round(kyotaku=None, scores=(23000, 23000, 23000, 23000))
        assert hud.with_sticks().kyotaku == 8

    def test_a_score_mid_animation_is_not_a_stick(self) -> None:
        hud = _round(kyotaku=None, scores=(23500, 23000, 23000, 23000))
        assert hud.with_sticks().kyotaku is None

    def test_an_unread_score_means_it_cannot_tell(self) -> None:
        hud = _round(kyotaku=None, scores=(23000, 23000, 23000, None))
        assert hud.with_sticks().kyotaku is None

    def test_a_read_number_is_left_alone(self) -> None:
        assert _round(kyotaku=0).with_sticks().kyotaku == 0

    def test_a_reach_past_the_templates_is_still_seen(self) -> None:
        """第六根棒子讀不到,立直不能因此看不到。"""
        watcher = ReachWatcher()
        watcher.start(_round(kyotaku=5, scores=(24000, 24000, 24000, 23000)))
        moved = _round(kyotaku=None, scores=(23000, 24000, 24000, 23000))
        assert watcher.observe(moved) == []
        assert watcher.observe(moved) == [0]


class TestRoundWatcher:
    def test_it_needs_two_matching_reads(self) -> None:
        watcher = RoundWatcher()
        assert watcher.observe(_round()) is None
        assert watcher.observe(_round()) is not None

    def test_the_tiles_left_may_change_between_reads(self) -> None:
        """余牌每一巡都在變。"""
        watcher = RoundWatcher()
        watcher.observe(_round(wall=60))
        assert watcher.observe(_round(wall=59)) is not None

    def test_a_different_read_starts_over(self) -> None:
        """結算面板收起來的那一瞬間,點數還在跳。"""
        watcher = RoundWatcher()
        watcher.observe(_round(kyoku=1))
        assert watcher.observe(_round(kyoku=2)) is None
        assert watcher.observe(_round(kyoku=2)) is not None

    def test_scores_that_do_not_add_up_never_settle(self) -> None:
        watcher = RoundWatcher()
        bad = _round(scores=(25000, 25000, 25000, 5000))
        assert watcher.observe(bad) is None
        assert watcher.observe(bad) is None

    def test_an_unread_honba_only_takes_longer(self) -> None:
        """模板沒有的數字讀不到,實測 5 本場那一局從頭到尾沒有事件流。"""
        watcher = RoundWatcher()
        reads = [watcher.observe(_round(honba=None)) for _ in range(4)]
        assert reads[:3] == [None, None, None]
        assert reads[3] is not None

    def test_an_unknown_wind_does_not_block_the_dealer(self) -> None:
        """西入認不得,但莊家不能因此整局不知道。"""
        watcher = RoundWatcher()
        watcher.observe(_round(bakaze=None))
        settled = watcher.observe(_round(bakaze=None))
        assert settled is not None
        assert settled.dealer == 3

    def test_the_first_kyoku_falls_back_to_east(self) -> None:
        """看不到上一局。場風錯只是少算役牌,事件流沒有場風就整局沒有建議。"""
        watcher = RoundWatcher()
        watcher.observe(_round(bakaze=None))
        settled = watcher.observe(_round(bakaze=None))
        assert settled is not None
        assert settled.bakaze == "E"

    @staticmethod
    def _next(watcher: RoundWatcher, **changes: object) -> Hud:
        """同一個 watcher 上再開一局。"""
        watcher.reset()
        settled = None
        for _ in range(4):
            settled = watcher.observe(_round(**changes)) or settled
        assert settled is not None
        return settled

    @pytest.mark.parametrize(
        ("wind", "after"), [("S", "W"), ("W", "N"), ("E", "S")], ids=["西入", "北入", "南入"]
    )
    def test_the_wind_moves_on_when_the_kyoku_number_drops(self, wind: str, after: str) -> None:
        watcher = RoundWatcher()
        self._next(watcher, bakaze=wind, kyoku=4, honba=0)
        assert self._next(watcher, bakaze=None, kyoku=1, honba=0).bakaze == after

    def test_the_wind_stays_while_the_number_goes_up(self) -> None:
        watcher = RoundWatcher()
        self._next(watcher, bakaze="W", kyoku=1, honba=0)
        assert self._next(watcher, bakaze=None, kyoku=2, honba=0).bakaze == "W"

    def test_an_unread_honba_is_one_more_than_the_last(self) -> None:
        """模板只到 5。真人打到過 9 本場。"""
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=5)
        for honba in range(6, 12):
            assert self._next(watcher, kyoku=1, honba=None).honba == honba

    def test_honba_can_carry_into_the_next_kyoku(self) -> None:
        """流局莊家沒聽牌:局數往前,本場還是加一。"""
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=7)
        assert self._next(watcher, kyoku=2, honba=None).honba == 8

    def test_a_misread_honba_is_corrected(self) -> None:
        """本場只可能歸零或加一,8 被看成 3 不可能。"""
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=7)
        assert self._next(watcher, kyoku=2, honba=3).honba == 8

    def test_honba_cannot_reset_on_the_same_kyoku(self) -> None:
        """莊家沒換就沒有人歸零過。"""
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=7)
        assert self._next(watcher, kyoku=1, honba=0).honba == 8

    def test_a_win_by_someone_else_resets_honba(self) -> None:
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=7)
        assert self._next(watcher, kyoku=2, honba=0).honba == 0

    def test_reading_the_same_kyoku_again_does_not_add_one(self) -> None:
        """局被誤判清空、重讀一次時,HUD 還是同一個數字。"""
        watcher = RoundWatcher()
        self._next(watcher, kyoku=1, honba=2)
        assert self._next(watcher, kyoku=1, honba=2).honba == 2

    def test_once_per_kyoku(self) -> None:
        """立直之後點數會變,不要因此再採信一次。"""
        watcher = RoundWatcher()
        watcher.observe(_round())
        watcher.observe(_round())
        assert watcher.observe(_round(kyotaku=1)) is None
        watcher.reset()
        watcher.observe(_round())
        assert watcher.observe(_round()) is not None


class TestAnotherStartingScore:
    """友人場可以改起始點數。寫死 100000 的話檢查碼永遠過不了,整場沒有事件流。"""

    OPENING = Hud(bakaze="E", kyoku=1, honba=0, kyotaku=0, scores=(35000,) * 4, dealer=0)

    def test_the_total_comes_from_the_first_kyoku(self) -> None:
        watcher = RoundWatcher()
        watcher.observe(self.OPENING)
        settled = watcher.observe(self.OPENING)
        assert settled is not None
        assert settled.scores_agree

    def test_it_lasts_for_the_whole_game(self) -> None:
        watcher = RoundWatcher()
        watcher.observe(self.OPENING)
        watcher.observe(self.OPENING)
        watcher.reset()
        later = Hud(bakaze="E", kyoku=2, honba=0, kyotaku=1, scores=(47000, 34000, 34000, 24000),
                    dealer=3)
        watcher.observe(later)
        settled = watcher.observe(later)
        assert settled is not None
        assert settled.total == 140000

    def test_sticks_use_it_too(self) -> None:
        hud = Hud(scores=(35000, 34000, 35000, 35000), total=140000)
        assert hud.with_sticks().kyotaku == 1

    def test_a_misread_digit_is_not_learned(self) -> None:
        """四個數字不一樣就不算開局,檢查碼的本事不會打折。"""
        watcher = RoundWatcher()
        wrong = Hud(bakaze="E", kyoku=1, honba=0, kyotaku=0, scores=(25000, 25000, 26000, 25000),
                    dealer=0)
        watcher.observe(wrong)
        assert watcher.observe(wrong) is None

    def test_a_reach_is_still_seen(self) -> None:
        watcher = ReachWatcher()
        start = replace(self.OPENING, total=140000)
        watcher.start(start)
        after = Hud(kyotaku=1, scores=(34000, 35000, 35000, 35000))
        watcher.observe(after)
        assert watcher.observe(after) == [0]


class TestReachWatcher:
    def _started(self) -> ReachWatcher:
        watcher = ReachWatcher()
        watcher.start(_round())
        return watcher

    def test_a_thousand_gone_and_a_stick_on_the_table(self) -> None:
        watcher = self._started()
        hud = _round(kyotaku=1, scores=(25000, 24000, 25000, 25000))
        assert watcher.observe(hud) == []
        assert watcher.observe(hud) == [1]

    def test_nothing_changed_means_nothing(self) -> None:
        watcher = self._started()
        assert watcher.observe(_round()) == []
        assert watcher.observe(_round()) == []

    def test_two_players_can_reach_in_turn(self) -> None:
        watcher = self._started()
        first = _round(kyotaku=1, scores=(25000, 24000, 25000, 25000))
        watcher.observe(first)
        watcher.observe(first)
        second = _round(kyotaku=2, scores=(24000, 24000, 25000, 25000))
        watcher.observe(second)
        assert watcher.observe(second) == [0]

    def test_two_at_once_between_reads(self) -> None:
        """每 0.5 秒一個動作的時候,兩次讀取之間可能有兩個人立直。"""
        watcher = self._started()
        hud = _round(kyotaku=2, scores=(24000, 24000, 25000, 25000))
        watcher.observe(hud)
        assert watcher.observe(hud) == [0, 1]

    def test_settling_up_at_the_end_is_not_a_reach(self) -> None:
        """局尾的點數移動:有人剛好付 1000,但供託沒有多。"""
        watcher = self._started()
        hud = _round(scores=(26000, 24000, 25000, 25000))
        watcher.observe(hud)
        assert watcher.observe(hud) == []

    def test_a_flicker_is_not_a_reach(self) -> None:
        watcher = self._started()
        assert watcher.observe(_round(kyotaku=1, scores=(25000, 24000, 25000, 25000))) == []
        assert watcher.observe(_round()) == []
        assert watcher.observe(_round()) == []

    def test_nothing_before_the_round_is_known(self) -> None:
        hud = _round(kyotaku=1, scores=(25000, 24000, 25000, 25000))
        watcher = ReachWatcher()
        assert watcher.observe(hud) == []
        assert watcher.observe(hud) == []


class TestOneScoreCannotBeRead:
    """實測一家 34400 有一位數認不出來,那一局一個事件都沒有。"""

    @staticmethod
    def _hud(scores, kyotaku=0):
        return Hud(kyotaku=kyotaku, scores=scores)

    def test_it_comes_from_the_total(self) -> None:
        hud = self._hud((None, 19900, 18900, 26800)).filled()
        assert hud.scores == (34400, 19900, 18900, 26800)
        assert hud.scores_agree

    def test_reach_sticks_are_part_of_the_total(self) -> None:
        assert self._hud((None, 19900, 18900, 26800), 2).filled().scores[0] == 32400

    def test_two_missing_cannot_be_worked_out(self) -> None:
        scores = (None, None, 18900, 26800)
        assert self._hud(scores).filled().scores == scores

    def test_without_the_sticks_it_cannot_be_worked_out(self) -> None:
        scores = (None, 19900, 18900, 26800)
        assert self._hud(scores, None).filled().scores == scores

    def test_a_number_that_cannot_be_a_score_is_left_alone(self) -> None:
        """另外三家有一家讀錯的話算出來不會是整百。"""
        scores = (None, 19950, 18900, 26800)
        assert self._hud(scores).filled().scores == scores


class TestReachWithAnUnreadableScore:
    """補算的點數不能拿去判斷誰立直。實測棒子先多一根,1000 點算到讀不到的那一家頭上。"""

    OPENING = (34400, 19900, 18900, 26800)

    def _watcher(self) -> ReachWatcher:
        watcher = ReachWatcher()
        watcher.start(Hud(kyotaku=0, scores=self.OPENING))
        return watcher

    def test_the_sticks_moving_first_is_not_a_reach(self) -> None:
        watcher = self._watcher()
        moving = Hud(kyotaku=1, scores=(None, 19900, 18900, 26800))
        assert watcher.observe(moving) == []
        assert watcher.observe(moving) == []

    def test_someone_elses_reach_still_counts(self) -> None:
        watcher = self._watcher()
        done = Hud(kyotaku=1, scores=(None, 18900, 18900, 26800))
        assert watcher.observe(done) == []
        assert watcher.observe(done) == [1]

    def test_the_round_is_settled_with_the_filled_score(self) -> None:
        rounds = RoundWatcher()
        hud = Hud(kyoku=2, honba=0, kyotaku=0, dealer=0, scores=(None, 19900, 18900, 26800))
        assert rounds.observe(hud) is None
        settled = rounds.observe(hud)
        assert settled is not None
        assert settled.scores[0] == 34400
