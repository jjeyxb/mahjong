"""Fusion policies protect temporal identity, original scores, and red fives."""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.hybrid.fusion import Evidence, Fusion, TemplateReader, annotate, guard_stream
from mia.tracker import Stream
from mia.vision.tiles.classify import Match


class Reader:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = self.hits = 0

    def read(self, _frame, _det, _kind, _seat, *, allow_new=True):
        if not allow_new:
            return None
        self.calls += 1
        return self.evidence


def observation(index=1, pai="2m", conf=0.4, center=(40, 80), kind="hand"):
    tile = dict(pai=pai, conf=conf, box=[10, 20, 70, 140], center=list(center))
    return dict(
        frame_index=index,
        t_ms=index * 100,
        p0_hand=dict(tiles=[tile]) if kind == "hand" else None,
        river={"P0": dict(tiles=[tile])} if kind == "river" else None,
    )


FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_low_yolo_confidence_requires_two_strong_template_observations():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)), budget_ms=float("inf"))
    original = observation()
    keep = deepcopy(original)
    first = fusion.apply(FRAME, original)
    assert first["p0_hand"]["tiles"][0]["pai"] == "2m"
    second = fusion.apply(FRAME, observation(2))
    changed = second["p0_hand"]["tiles"][0]
    assert changed["pai"] == "1m"
    assert changed["raw_yolo"] == dict(pai="2m", conf=0.4)
    assert changed["template"]["score"] == 0.95
    assert original == keep


def test_two_high_confidence_disagreements_pause_own_hand_not_choose_larger_number():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)))
    fusion.apply(FRAME, observation(conf=0.99))
    obs = fusion.apply(FRAME, observation(2, conf=0.99))
    tile = obs["p0_hand"]["tiles"][0]
    assert tile["pai"] == "2m"
    assert obs["fusion"]["pending"]
    payload = annotate(dict(hand=dict(stable=True, confident=True)), obs["fusion"])
    assert not payload["hand"]["stable"] and not payload["hand"]["confident"]
    stream = guard_stream(Stream(), payload)
    assert stream.stalled and stream.doubt


@pytest.mark.parametrize(
    "score,margin,brightness", [(0.74, 0.30, 1), (0.95, 0.09, 1), (0.95, 0.3, 0.4)]
)
def test_weak_template_evidence_never_overrides_yolo(score, margin, brightness):
    fusion = Fusion(Reader(Evidence("1m", score, margin, brightness)))
    for i in range(1, 4):
        obs = fusion.apply(FRAME, observation(i, conf=0.99))
        assert obs["p0_hand"]["tiles"][0]["conf"] == 0.99
        assert not obs["fusion"]["pending"]


@pytest.mark.parametrize("old,new", [("5p", "5pr"), ("5pr", "5p")])
def test_red_fives_are_not_changed_by_merge_red_template_margins(old, new):
    fusion = Fusion(Reader(Evidence(new, 0.99, 0.3)))
    fusion.apply(FRAME, observation(pai=old))
    obs = fusion.apply(FRAME, observation(2, pai=old))
    assert obs["p0_hand"]["tiles"][0]["pai"] == old
    assert "raw_yolo" not in obs["p0_hand"]["tiles"][0]


def test_same_frame_does_not_count_as_two_temporal_votes():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)))
    fusion.apply(FRAME, observation())
    assert fusion.apply(FRAME, observation())["p0_hand"]["tiles"][0]["pai"] == "2m"


def test_spatial_identity_jitter_and_round_reset():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)))
    fusion.apply(FRAME, observation(center=(40, 80)))
    same = fusion.apply(FRAME, observation(2, center=(43, 82)))
    assert same["p0_hand"]["tiles"][0]["pai"] == "1m"
    fusion.reset()
    first = fusion.apply(FRAME, observation(3, center=(43, 82)))
    assert first["p0_hand"]["tiles"][0]["pai"] == "2m"


def test_old_votes_do_not_cross_a_long_capture_gap():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)))
    fusion.apply(FRAME, observation())
    assert fusion.apply(FRAME, observation(30))["p0_hand"]["tiles"][0]["pai"] == "2m"


def test_budget_still_keeps_unverified_geometry_and_labels():
    reader = Reader(Evidence("1m", 0.95, 0.2))
    fusion = Fusion(reader, max_new=0)
    original = observation()
    result = fusion.apply(FRAME, original)
    assert result["p0_hand"] == original["p0_hand"]
    assert reader.calls == 0


def test_strong_yolo_river_conflict_is_audited_without_creating_a_second_tile():
    fusion = Fusion(Reader(Evidence("1m", 0.95, 0.2)))
    fusion.apply(FRAME, observation(kind="river", conf=0.99))
    obs = fusion.apply(FRAME, observation(2, kind="river", conf=0.99))
    assert len(obs["river"]["P0"]["tiles"]) == 1
    assert obs["river"]["P0"]["tiles"][0]["pai"] == "2m"
    assert obs["fusion"]["counts"]["disagree"] == 1
    assert not obs["fusion"]["pending"]


def test_pixel_cache_invalidates_on_changed_pixels_but_not_unrelated_background(monkeypatch):
    import mia.hybrid.fusion as module

    monkeypatch.setattr(module, "classify", lambda *_a, **_k: Match("1m", 0.99, "2m", 0.3))
    reader = TemplateReader("mjpface_default", capacity=2)
    det = observation()["p0_hand"]["tiles"][0]
    frame = FRAME.copy()
    reader.read(frame, det, "hand", 0)
    frame[1000, 1800] = 200
    reader.read(frame, det, "hand", 0)
    assert reader.calls == 1 and reader.hits == 1
    frame[40, 40] = 200
    reader.read(frame, det, "hand", 0)
    assert reader.calls == 2


def test_detection_presence_alone_cannot_keep_old_template_votes_alive():
    reader = Reader(Evidence("1m", 0.95, 0.2))
    fusion = Fusion(reader)
    fusion.apply(FRAME, observation())
    # The detector keeps reporting a box, but the template budget is exhausted.
    fusion.max_new = 0
    for i in range(2, 25):
        fusion.apply(FRAME, observation(i))
    fusion.max_new = 24
    fresh = fusion.apply(FRAME, observation(25))
    assert fresh["p0_hand"]["tiles"][0]["pai"] == "2m"
    confirmed = fusion.apply(FRAME, observation(26))
    assert confirmed["p0_hand"]["tiles"][0]["pai"] == "1m"


@pytest.mark.parametrize(
    "filename,old,expected,confidence",
    [
        ("hybrid_river_north.png", "9m", "N", 0.5449),
        ("hybrid_river_east.png", "F", "E", 0.1704),
    ],
)
def test_real_recording_patch_corrects_a_low_confidence_yolo_label(
    monkeypatch, filename, old, expected, confidence
):
    patch = cv2.imread(str(Path(__file__).parents[1] / "fixtures" / filename))
    assert patch is not None
    reader = TemplateReader("mjpface_default")
    monkeypatch.setattr(reader, "patch", lambda *_a: (patch, False))
    fusion = Fusion(reader)
    fusion.apply(FRAME, observation(pai=old, conf=confidence, kind="river"))
    obs = fusion.apply(FRAME, observation(2, pai=old, conf=confidence, kind="river"))
    tile = obs["river"]["P0"]["tiles"][0]
    assert tile["pai"] == expected
    assert tile["raw_yolo"]["pai"] == old
    assert not obs["fusion"]["pending"]
