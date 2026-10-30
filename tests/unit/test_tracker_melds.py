"""副露長度 → 新的組數。"""

from __future__ import annotations

from mia.tracker.melds import MeldWatcher, guess_meld


def watcher() -> MeldWatcher:
    return MeldWatcher(units=(3.0, 3.0))


def feed(w: MeldWatcher, *lengths: float | None) -> list[int]:
    return [w.observe(0, length) for length in lengths]


def test_a_group_needs_two_matching_reads() -> None:
    """鳴牌動畫中間牌在飛,長度是過渡值。"""
    assert feed(watcher(), 0, 3.0, 3.0) == [0, 0, 1]


def test_a_moving_length_is_not_trusted() -> None:
    assert feed(watcher(), 1.5, 2.4, 3.0, 3.0) == [0, 0, 0, 1]


def test_groups_of_different_widths_each_count_once() -> None:
    """碰 1.0、暗槓 1.2、大明槓 1.3 —— 只看增量,寬度差不會累積。"""
    w = watcher()
    assert sum(feed(w, 3.0, 3.0, 6.6, 6.6, 10.5, 10.5, 13.5, 13.5)) == 4
    assert w.groups[0] == 4


def test_four_pons_are_not_three_daiminkan() -> None:
    """總長 4.0 與 3.9 組長幾乎一樣,從總長反推會錯;增量不會。"""
    w = watcher()
    feed(w, 3, 3, 6, 6, 9, 9, 12, 12)
    assert w.groups[0] == 4


def test_it_never_shrinks() -> None:
    """立繪或結算面板蓋住時讀成 0 或讀不到,不是有人把副露收回去。"""
    w = watcher()
    assert feed(w, 3, 3, 0, None, 0, 3, 3) == [0, 1, 0, 0, 0, 0, 0]
    assert w.groups[0] == 1


def test_two_groups_between_reads_count_as_two() -> None:
    w = watcher()
    assert feed(w, 6.6, 6.6) == [0, 2]


def test_positions_are_independent_and_reset_clears() -> None:
    w = watcher()
    feed(w, 3, 3)
    assert w.observe(1, 0) == 0
    assert w.groups == [1, 0]
    w.reset()
    assert w.groups == [0, 0]


def test_the_length_moves_only_when_a_group_is_confirmed() -> None:
    """新的一組從這個長度之後開始 —— 認牌面只認那一段。"""
    w = watcher()
    feed(w, 3.0)
    assert w.length(0) == 0
    feed(w, 3.0)
    assert w.length(0) == 3.0


class TestGuessMeld:
    def test_one_honour_is_the_whole_group(self) -> None:
        """字牌不能吃,認出一張就知道是碰(或槓)。"""
        assert guess_meld(["C"]) == ("C", "C", "C")

    def test_one_number_could_be_either(self) -> None:
        assert guess_meld(["6s"]) == ("6s", "?", "?")

    def test_a_run_is_kept_in_order(self) -> None:
        assert guess_meld(["8p", "6p", "7p"]) == ("6p", "7p", "8p")
        assert guess_meld(["7p", "6p"]) == ("6p", "7p", "?")

    def test_two_of_a_run_can_pin_down_the_third(self) -> None:
        """差 2 就知道中間;差 1 而碰到端點也只有一種順子。"""
        assert guess_meld(["8p", "6p"]) == ("6p", "7p", "8p")
        assert guess_meld(["1m", "2m"]) == ("1m", "2m", "3m")
        assert guess_meld(["9s", "8s"]) == ("7s", "8s", "9s")

    def test_the_same_number_in_two_places_is_a_pon(self) -> None:
        """吃不會有兩張一樣的牌。"""
        assert guess_meld(["6m"], repeated=["6m"]) == ("6m", "6m", "6m")

    def test_a_run_with_a_repeat_is_wrong_somewhere(self) -> None:
        assert guess_meld(["6m", "7m"], repeated=["6m"]) == ("?", "?", "?")

    def test_nothing_read_is_unknown(self) -> None:
        assert guess_meld([]) == ("?", "?", "?")

    def test_what_cannot_be_one_group_is_dropped(self) -> None:
        """有一張認錯了,不知道是哪一張:整組不採信。"""
        assert guess_meld(["C", "N"]) == ("?", "?", "?")
        assert guess_meld(["C", "6s"]) == ("?", "?", "?")
        assert guess_meld(["6s", "6m"]) == ("?", "?", "?")
        assert guess_meld(["4s", "7s"]) == ("?", "?", "?")
        assert guess_meld(["4s", "6s", "7s", "8s"]) == ("?", "?", "?")

