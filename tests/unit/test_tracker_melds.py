"""副露長度 → 新的組數。"""

from __future__ import annotations

from mia.tracker.melds import MeldWatcher


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
