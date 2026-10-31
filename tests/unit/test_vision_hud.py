"""HUD 的讀取與它的檢查碼。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.tracker import RoundWatcher
from mia.utils.geometry import Rect
from mia.vision.hud import Hud, HudDigits, read_hud

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


class TestItAbstains:
    def test_a_blank_screen_reads_nothing(self) -> None:
        """大廳、讀取畫面:每個欄位各自棄權,而不是讀出一個數字。"""
        blank = np.full((1430, 2536, 3), 30, dtype=np.uint8)
        hud = read_hud(blank, Rect(0, 0, 2536, 1430), HudDigits.load())
        assert hud == Hud()


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

    def test_an_unknown_wind_does_not_block_the_dealer(self) -> None:
        """南場還認不得,但莊家不能因此整局不知道。"""
        watcher = RoundWatcher()
        watcher.observe(_round(bakaze=None))
        settled = watcher.observe(_round(bakaze=None))
        assert settled is not None
        assert settled.dealer == 3

    def test_once_per_kyoku(self) -> None:
        """立直之後點數會變,不要因此再採信一次。"""
        watcher = RoundWatcher()
        watcher.observe(_round())
        watcher.observe(_round())
        assert watcher.observe(_round(kyotaku=1)) is None
        watcher.reset()
        watcher.observe(_round())
        assert watcher.observe(_round()) is not None
