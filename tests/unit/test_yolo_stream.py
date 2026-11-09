"""Causal identity and rollback regressions for the optional visual backend."""

from copy import deepcopy

from mia.yolo.stream import LedgerStream

HAND = [f"{i}m" for i in range(1, 10)] + ["1p", "2p", "3p", "P"]


def start(hand=None, dealer=0):
    return dict(
        event_id="s",
        type="start_kyoku",
        export_eligible=True,
        status="CONFIRMED",
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=(dealer + 3) % 4,
        dora_marker="9s",
        scores=[10000, 20000, 30000, 40000],
        tehais=[["?"] * 13, ["?"] * 13, ["?"] * 13, list(hand or HAND)],
        seat_map={"P0": 3, "P1": 0, "P2": 1, "P3": 2},
    )


def event(eid, typ, actor, **kw):
    return dict(
        event_id=eid,
        type=typ,
        actor=f"P{actor}",
        status="CONFIRMED",
        export_eligible=True,
        frame_first=1,
        **kw,
    )


def snapshot(events, **kw):
    return dict(hand_id="H001", events=events, **kw)


def test_duplicate_discard_and_its_inferred_pon_are_not_replayed():
    events = [
        start(),
        event("t", "tsumo", 0, pai="P"),
        event("d", "dahai", 0, pai="P", tsumogiri=False),
        event("p", "pon", 1, target="P0", pai="P", consumed=["P", "P"], source_discard="d"),
        event("fake-d", "dahai", 0, pai="P", tsumogiri=False),
        event(
            "fake-p", "pon", 1, target="P0", pai="P", consumed=["P", "P"], source_discard="fake-d"
        ),
        event("d1", "dahai", 1, pai="1s", tsumogiri=False),
    ]
    adapter = LedgerStream()
    stream = adapter.update(snapshot(events))
    assert stream.doubt is None
    assert [e.TYPE for e in stream.events] == ["start_kyoku", "tsumo", "dahai", "pon", "dahai"]
    assert adapter.rejected == ("fake-d", "fake-p")
    assert stream.events[0].oya == 0
    assert stream.events[0].scores == [40000, 10000, 20000, 30000]


def test_same_chi_with_distinct_discard_sources_is_a_legitimate_second_call():
    hand = ["6s", "6s"] + [f"{i}m" for i in range(1, 10)] + ["1p", "P"]
    events = [start(hand)]
    for n in range(2):
        events.extend(
            [
                event(f"t{n}", "tsumo", 0, pai="4p"),
                event(f"d{n}", "dahai", 0, pai="6s", tsumogiri=False),
                event(
                    f"c{n}",
                    "chi",
                    1,
                    target="P0",
                    pai="6s",
                    consumed=["5s", "7s"],
                    source_discard=f"d{n}",
                ),
                event(f"x{n}", "dahai", 1, pai="1s", tsumogiri=False),
                event(f"y{n}", "dahai", 2, pai="2s", tsumogiri=False),
                event(f"z{n}", "dahai", 3, pai="3s", tsumogiri=False),
            ]
        )
    stream = LedgerStream().update(snapshot(events))
    assert stream.doubt is None
    assert len([e for e in stream.events if e.TYPE == "chi"]) == 2


def test_missing_dealer_draw_can_only_be_inferred_for_an_opponent():
    stream = LedgerStream().update(
        snapshot([start(dealer=1), event("d", "dahai", 1, pai="1s", tsumogiri=False)])
    )
    assert [e.TYPE for e in stream.events] == ["start_kyoku", "tsumo", "dahai"]
    own = LedgerStream().update(
        snapshot([start(), event("d", "dahai", 0, pai="1m", tsumogiri=False)])
    )
    assert len(own.events) == 1


def test_corrected_tile_replaces_an_old_snapshot_without_mutating_the_input():
    events = [
        start(),
        event("t", "tsumo", 0, pai="4p"),
        event("d", "dahai", 0, pai="1m", tsumogiri=False),
    ]
    adapter = LedgerStream()
    before = adapter.update(snapshot(events))
    changed = deepcopy(events)
    changed[-1]["pai"] = "2m"
    after = adapter.update(snapshot(changed))
    assert before.events[-1].pai == "1m"
    assert after.events[-1].pai == "2m"
    assert events[-1]["pai"] == "1m"
    assert adapter.kyoku == 1


def test_stable_physical_hand_repairs_initial_inventory_and_survives_skipped_frames():
    wrong = list(HAND)
    wrong[-1] = "1m"
    events = [
        start(wrong),
        event("t", "tsumo", 0, pai="4p"),
        event("d", "dahai", 0, pai="P", tsumogiri=False),
    ]
    adapter = LedgerStream()
    assert adapter.update(snapshot(events)).doubt is not None
    physical = [t for t in HAND if t != "P"] + ["4p"]
    repaired = adapter.update(snapshot(events, hand=dict(tiles=physical, stable=True, frame=10)))
    assert repaired.doubt is None
    assert sorted(repaired.events[0].tehais[0]) == sorted(HAND)
    assert adapter.update(snapshot(events)).doubt is None
    assert adapter.update(dict(hand_id="H002", events=events)).doubt is not None


def test_the_hand_after_an_unconfirmed_draw_and_discard_does_not_rewrite_the_initial_hand():
    # 錄影 selfplay-20261102-053001 等五場:第二巡摸 5p 打 1m 都還沒確認,
    # 畫面已經是打完的手牌
    events = [
        start(),
        event("t", "tsumo", 0, pai="4p"),
        event("d", "dahai", 0, pai="P", tsumogiri=False),
        dict(event("t2", "tsumo", 0, pai="5p"), export_eligible=False),
        dict(event("d2", "dahai", 0, pai="1m", tsumogiri=False), export_eligible=False),
    ]
    after = [t for t in HAND if t not in ("P", "1m")] + ["4p", "5p"]
    adapter = LedgerStream()
    stream = adapter.update(snapshot(events, hand=dict(tiles=after, stable=True, frame=10)))
    assert sorted(stream.events[0].tehais[0]) == sorted(HAND)


def test_red_fives_and_cancellations_remain_distinct():
    events = [
        start(),
        event("t", "tsumo", 0, pai="5pr"),
        event("d", "dahai", 0, pai="5pr", tsumogiri=True),
    ]
    stream = LedgerStream().update(snapshot(events))
    assert stream.doubt is None
    assert stream.events[-1].pai == "5pr"
    events[-1]["status"] = "CANCELLED"
    assert LedgerStream().update(snapshot(events)).events[-1].TYPE == "tsumo"


def test_ron_cancels_reach_acceptance_but_preserves_the_declaration():
    events = [
        start(),
        event("t", "tsumo", 0, pai="4p"),
        event("r", "reach", 0),
        event("d", "dahai", 0, pai="4p", tsumogiri=True),
    ]
    events[2].update(status="CANCELLED", export_eligible=False)
    stream = LedgerStream().update(snapshot(events))
    assert [e.TYPE for e in stream.events] == ["start_kyoku", "tsumo", "reach", "dahai"]


def test_pending_own_draw_pauses_advice_until_its_face_is_known():
    draw = event("t", "tsumo", 0, pai=None)
    draw.update(status="PENDING", export_eligible=False)
    adapter = LedgerStream()
    pending = adapter.update(snapshot([start(), draw]))
    assert pending.stalled and pending.doubt is not None
    draw.update(pai="4p", status="CONFIRMED", export_eligible=True)
    confirmed = adapter.update(snapshot([start(), draw]))
    assert not confirmed.stalled and confirmed.doubt is None


def test_a_draw_its_discard_points_to_holds_the_stream_until_its_face_is_known():
    """錄影 20261024 東2局:摸三索沒讀到,跳過的話打東被丟掉,下一張摸牌頂替它。"""
    draw = event("t1", "tsumo", 0, pai=None)
    draw.update(status="PENDING", export_eligible=False)
    events = [
        start(),
        draw,
        event("d1", "dahai", 0, pai="P", tsumogiri=False, p0_tsumo="t1"),
        event("t2", "tsumo", 0, pai="4p"),
        event("d2", "dahai", 0, pai="4p", tsumogiri=True, p0_tsumo="t2"),
    ]
    adapter = LedgerStream()
    held = adapter.update(snapshot(events))
    assert held.stalled
    assert [e.TYPE for e in held.events] == ["start_kyoku"]
    draw.update(pai="3s", status="CONFIRMED", export_eligible=True)
    stream = adapter.update(snapshot(events))
    assert not stream.stalled
    assert [e.to_dict().get("pai") for e in stream.events[1:3]] == ["3s", "P"]


def test_a_pending_draw_right_after_our_own_call_is_not_waited_for():
    """錄影 170335 東1局1本場:碰牌後冒出一筆待定摸牌,打牌還指著它。碰牌後不摸牌。"""
    hand = ["3p", "3p"] + HAND[:11]
    fake = event("t2", "tsumo", 0, pai=None)
    fake.update(status="PENDING", export_eligible=False)
    events = [
        start(hand, dealer=3),
        event("t3", "tsumo", 3, pai="?"),
        event("d3", "dahai", 3, pai="3p", tsumogiri=False),
        event("p", "pon", 0, target="P3", pai="3p", consumed=["3p", "3p"], source_discard="d3"),
        fake,
        event("d0", "dahai", 0, pai="P", tsumogiri=False, p0_tsumo="t2"),
    ]
    stream = LedgerStream().update(snapshot(events))
    assert [e.TYPE for e in stream.events][-2:] == ["pon", "dahai"]


def test_a_call_seen_before_the_discard_it_claims_still_follows_it():
    """錄影 043327 東3局3本場:吃的橫幅比被吃的立直宣言牌早幾幀,事件帳裡排在前面。"""
    events = [
        start(),
        event("t", "tsumo", 0, pai="P"),
        event("c", "chi", 1, target="P0", pai="3m", consumed=["2m", "4m"], source_discard="d"),
        event("d", "dahai", 0, pai="3m", tsumogiri=False),
        event("d1", "dahai", 1, pai="1s", tsumogiri=False),
    ]
    adapter = LedgerStream()
    stream = adapter.update(snapshot(events))
    assert [e.TYPE for e in stream.events] == ["start_kyoku", "tsumo", "dahai", "chi", "dahai"]
    assert adapter.rejected == ()


def test_waiting_says_why_the_start_is_not_exported_yet():
    """實打時 YOLO 整局等不到開局,重播卻讀得到;要記下當時卡在哪一步。"""
    adapter = LedgerStream()
    adapter.update(snapshot([]))
    assert "沒有 start_kyoku" in adapter.waiting
    pending = start() | dict(export_eligible=False, status="PENDING", missing=["tehais"])
    adapter.update(snapshot([pending]))
    assert "PENDING" in adapter.waiting
    assert "tehais" in adapter.waiting
    adapter.update(snapshot([start() | dict(seat_map={})]))
    assert "seat_map" in adapter.waiting
    assert adapter.update(snapshot([start()])).events
    assert adapter.waiting == ""
