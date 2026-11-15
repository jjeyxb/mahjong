from __future__ import annotations

import pytest

from mia.mjai.events import Ankan, Chi, Daiminkan, Kakan, Pon
from mia.mjai.melds import valid_meld
from mia.yolo.stream import validate_inventory


@pytest.mark.parametrize("event", [
    Chi(0, 2, "6s", ["4s", "5s"]),
    Chi(1, 0, "9s", ["8s", "9s"]),
    Chi(1, 0, "9s", ["7p", "8s"]),
    Pon(3, 0, "P", ["P", "E"]),
    Pon(1, 0, "C", ["C", "E"]),
    Pon(1, 1, "N", ["N", "N"]),
    Daiminkan(1, 0, "5pr", ["5pr", "5p", "5p"]),
    Ankan(2, ["E", "E", "E", "S"]),
    Kakan(2, "3s", ["3s", "3s"]),
])
def test_rejects_recorded_invalid_call_shapes_and_sources(event):
    assert not valid_meld(event)


@pytest.mark.parametrize("event", [
    Chi(0, 3, "6s", ["5s", "7s"]),
    Chi(2, 1, "5pr", ["4p", "6p"]),
    Pon(0, 1, "N", ["N", "N"]),
    Pon(0, 2, "5p", ["5pr", "5p"]),
    Daiminkan(3, 1, "5p", ["5p", "5pr", "5p"]),
    Ankan(1, ["2s"] * 4),
    Kakan(1, "2s", ["2s"] * 3),
])
def test_preserves_legal_calls_and_red_fives(event):
    assert valid_meld(event)


def test_yolo_does_not_feed_a_mismatched_triplet_to_the_engine():
    from tests.unit.test_hybrid_merge import start

    reason = validate_inventory([start(), Pon(1, 0, "P", ["P", "E"])])
    assert reason is not None and "副露" in reason
