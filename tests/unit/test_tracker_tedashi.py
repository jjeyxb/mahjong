"""手牌形狀 → 摸切還是手切。"""

from __future__ import annotations

from mia.tracker.tedashi import TsumogiriWatcher
from mia.vision.tiles.backs import BackShape

REST = BackShape(330.0, False, 1, True)
DRAWN = BackShape(330.0, True, 1, True)
HOLED = BackShape(300.0, True, 2, True)
ARM = BackShape(150.0, False, 4, False)


def feed(watcher: TsumogiriWatcher, position: int, *shapes: BackShape, start: float = 0.0) -> float:
    now = start
    for shape in shapes:
        watcher.observe(position, now, shape)
        now += 0.15
    return now


class TestAlongTheLine:
    def test_the_drawn_tile_just_leaves(self) -> None:
        """摸切:尾端那張不見,本體一直是完整的。"""
        watcher = TsumogiriWatcher()
        now = feed(watcher, 3, REST, DRAWN, DRAWN, REST, REST)
        assert watcher.tsumogiri(3, now) is True

    def test_a_hole_while_the_drawn_tile_waits(self) -> None:
        watcher = TsumogiriWatcher()
        now = feed(watcher, 3, REST, DRAWN, HOLED, HOLED, REST)
        assert watcher.tsumogiri(3, now) is False

    def test_the_arm_is_not_a_hole(self) -> None:
        """角色的手把手牌切成好幾段:那不是乾淨的形狀,不算。"""
        watcher = TsumogiriWatcher()
        now = feed(watcher, 3, REST, DRAWN, ARM, ARM, REST)
        assert watcher.tsumogiri(3, now) is True

    def test_toimen_needs_two_looks(self) -> None:
        watcher = TsumogiriWatcher()
        now = feed(watcher, 2, REST, DRAWN, HOLED, REST)
        assert not watcher.tegiri_seen(2, now)
        now = feed(watcher, 2, HOLED, start=now)
        assert watcher.tegiri_seen(2, now)

    def test_an_old_hole_is_forgotten(self) -> None:
        """上一手的手切不能算到這一手。"""
        watcher = TsumogiriWatcher()
        now = feed(watcher, 3, REST, DRAWN, HOLED, HOLED, REST)
        later = feed(watcher, 3, *([REST] * 12), DRAWN, REST, start=now)
        assert watcher.tsumogiri(3, later) is True


class TestInPieces:
    def test_two_whole_pieces_are_a_hole(self) -> None:
        watcher = TsumogiriWatcher()
        rest, drawn = BackShape(50400.0, False, 1, True), BackShape(53800.0, False, 1, True)
        split = BackShape(50700.0, False, 2, True)
        now = feed(watcher, 1, rest, drawn, split)
        assert watcher.tsumogiri(1, now) is False

    def test_pieces_with_much_missing_are_the_arm(self) -> None:
        watcher = TsumogiriWatcher()
        rest, drawn = BackShape(50400.0, False, 1, True), BackShape(53800.0, False, 1, True)
        covered = BackShape(40800.0, False, 2, True)
        now = feed(watcher, 1, rest, drawn, covered, rest)
        assert watcher.tsumogiri(1, now) is True


class TestNoVerdict:
    def test_an_initial_joined_hand_does_not_prove_tsumogiri(self) -> None:
        watcher = TsumogiriWatcher()
        joined = BackShape(360.0, False, 1, True)
        now = feed(watcher, 2, joined, joined, REST, REST)
        assert watcher.tsumogiri(2, now) is None

    def test_a_separate_draw_establishes_the_reference_for_later_turns(self) -> None:
        watcher = TsumogiriWatcher()
        now = feed(watcher, 2, REST, DRAWN, REST)
        later = feed(watcher, 2, *([REST] * 24), start=now)
        assert watcher.tsumogiri(2, later) is True

    def test_a_meld_requires_a_new_reference_for_the_shorter_hand(self) -> None:
        watcher = TsumogiriWatcher()
        now = feed(watcher, 2, REST, DRAWN, REST)
        watcher.forget(2)
        short = BackShape(250.0, False, 1, True)
        now = feed(watcher, 2, short, short, start=now)
        assert watcher.tsumogiri(2, now) is None

    def test_nothing_seen(self) -> None:
        assert TsumogiriWatcher().tsumogiri(2, 1.0) is None

    def test_no_hand_found(self) -> None:
        """版面不同、量不到手牌:不下判斷,照舊填手切。"""
        watcher = TsumogiriWatcher()
        now = feed(watcher, 2, BackShape(0.0, False, 0, False), BackShape(5.0, False, 1, True))
        assert watcher.tsumogiri(2, now) is None

    def test_a_call_forgets_the_longer_hand(self) -> None:
        """鳴牌之後手牌短了一截。拿鳴牌前的長度比,之後每一手都像少了一張。"""
        watcher = TsumogiriWatcher()
        short, short_drawn = BackShape(250.0, False, 1, True), BackShape(250.0, True, 1, True)
        now = feed(watcher, 3, REST, REST)
        watcher.forget(3)
        now = feed(watcher, 3, short, short_drawn, short_drawn, short, start=now)
        assert watcher.tsumogiri(3, now) is True

    def test_reset(self) -> None:
        watcher = TsumogiriWatcher()
        now = feed(watcher, 3, REST, DRAWN, HOLED, HOLED)
        watcher.reset()
        assert watcher.tsumogiri(3, now) is None
