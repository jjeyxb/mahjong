"""目前知道的一切 → MJAI 事件流。理由見 ``mia/tracker/stream.py``。"""

from __future__ import annotations

from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    Kakan,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    StartKyoku,
    Tsumo,
)
from mia.tracker import Action, KyokuLog, Move, OwnMove, Step, compose
from mia.vision.hud import Hud

T = Action.TSUMO
C = Action.CALL
D = Action.DAHAI
TEHAI = ("1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "S")
HUD = Hud(
    bakaze="S",
    kyoku=2,
    wall=69,
    honba=1,
    kyotaku=0,
    scores=(30000, 25000, 25000, 20000),
    dealer=1,
)


def log(**fields: object) -> KyokuLog:
    return KyokuLog(hud=HUD, tehai=TEHAI, doras=[(0, "9m")], **fields)  # type: ignore[arg-type]


def body(events: tuple[MjaiEvent, ...]) -> list[MjaiEvent]:
    return list(events[1:])


class TestStart:
    def test_the_header_comes_from_the_hud(self) -> None:
        start = compose(log()).events[0]
        assert isinstance(start, StartKyoku)
        assert (start.bakaze, start.kyoku, start.honba, start.oya) == ("S", 2, 1, 1)
        assert start.dora_marker == "9m"
        assert start.tehais[0] == list(TEHAI)
        assert start.tehais[1] == ["?"] * 13
        assert start.scores == [30000, 25000, 25000, 20000]

    def test_nothing_without_the_deal(self) -> None:
        assert compose(KyokuLog(hud=HUD, doras=[(0, "9m")])).events == ()

    def test_nothing_without_the_dora(self) -> None:
        assert compose(KyokuLog(hud=HUD, tehai=TEHAI)).events == ()

    def test_nothing_when_the_scores_do_not_add_up(self) -> None:
        wrong = Hud(bakaze="S", kyoku=2, kyotaku=0, scores=(1, 2, 3, 4), dealer=1)
        assert compose(KyokuLog(hud=wrong, tehai=TEHAI, doras=[(0, "9m")])).events == ()


class TestOthers:
    def test_a_draw_is_hidden_and_the_discard_is_shown(self) -> None:
        stream = compose(log(steps=[Step(T, 1), Step(D, 1, "2p")]))
        assert body(stream.events) == [
            Tsumo(actor=1, pai="?"),
            Dahai(actor=1, pai="2p", tsumogiri=False),
        ]

    def test_an_unread_tile_uses_the_guess(self) -> None:
        stream = compose(log(steps=[Step(T, 1), Step(D, 1, None, "7p")]))
        assert body(stream.events)[-1] == Dahai(actor=1, pai="7p", tsumogiri=False)

    def test_a_guess_can_never_be_the_fifth_copy(self) -> None:
        """自己手上有兩張南、寶牌沒有,別家打過兩張:猜南就是第五張。"""
        steps = [
            Step(T, 1),
            Step(D, 1, "S"),
            Step(T, 2),
            Step(D, 2, "S"),
            Step(T, 3),
            Step(D, 3, None, "S"),
        ]
        last = body(compose(log(steps=steps)).events)[-1]
        assert isinstance(last, Dahai)
        assert last.pai != "S"

    def test_certain_tiles_later_keep_their_place(self) -> None:
        """猜測排在有信心的牌前面,還是不能佔掉它的名額。"""
        steps = [
            Step(T, 1),
            Step(D, 1, None, "S"),
            Step(T, 2),
            Step(D, 2, "S"),
            Step(T, 3),
            Step(D, 3, "S"),
        ]
        events = body(compose(log(steps=steps)).events)
        assert [e.pai for e in events if isinstance(e, Dahai)][1:] == ["S", "S"]
        assert events[1] != Dahai(actor=1, pai="S", tsumogiri=False)

    def test_a_confident_fifth_copy_is_replaced_and_counted(self) -> None:
        steps = [Step(D, 1, "S"), Step(D, 2, "S"), Step(D, 3, "S")]
        stream = compose(log(steps=steps))
        assert stream.forced == 1
        assert sum(isinstance(e, Dahai) and e.pai == "S" for e in stream.events) == 2

    def test_a_hidden_tile_still_gets_a_tile(self) -> None:
        last = body(compose(log(steps=[Step(T, 1), Step(D, 1)])).events)[-1]
        assert isinstance(last, Dahai)
        assert last.pai != "?"


class TestOwn:
    def test_own_moves_come_from_the_hand(self) -> None:
        moves = [OwnMove(Move.DRAW, ("9p",)), OwnMove(Move.DISCARD, ("9p",), tsumogiri=True)]
        stream = compose(log(steps=[Step(T, 0), Step(D, 0, None, "1p")], moves=moves))
        assert body(stream.events) == [
            Tsumo(actor=0, pai="9p"),
            Dahai(actor=0, pai="9p", tsumogiri=True),
        ]

    def test_the_draw_goes_out_before_turn_tracking_sees_it(self) -> None:
        """Mortal 要在打之前回答,而 TurnTracker 要打出去才排得出這一手。"""
        stream = compose(log(moves=[OwnMove(Move.DRAW, ("9p",))]))
        assert body(stream.events) == [Tsumo(actor=0, pai="9p")]

    def test_the_stream_stops_rather_than_invents_own_tiles(self) -> None:
        stream = compose(log(steps=[Step(T, 0), Step(C, 0, target=3)]))
        assert stream.stalled
        assert body(stream.events) == []

    def test_a_turn_the_hand_never_saw_is_tsumogiri(self) -> None:
        """立直之後摸到就打,手牌可能一次都沒讀到 14 張。"""
        stream = compose(log(steps=[Step(T, 0), Step(D, 0, "9p")]))
        assert body(stream.events) == [
            Tsumo(actor=0, pai="9p"),
            Dahai(actor=0, pai="9p", tsumogiri=True),
        ]

    def test_a_tsumogiri_too_fast_for_the_hand(self) -> None:
        """河裡打的是北,手牌接下來打的是一萬:這一手手牌根本沒看到。"""
        moves = [OwnMove(Move.DRAW, ("9p",)), OwnMove(Move.DISCARD, ("1m",))]
        steps = [Step(T, 0), Step(D, 0, "N"), Step(T, 0), Step(D, 0, "1m")]
        assert body(compose(log(steps=steps, moves=moves)).events) == [
            Tsumo(actor=0, pai="N"),
            Dahai(actor=0, pai="N", tsumogiri=True),
            Tsumo(actor=0, pai="9p"),
            Dahai(actor=0, pai="1m", tsumogiri=False),
        ]

    def test_the_next_river_tile_tells_a_misread_from_a_missed_turn(self) -> None:
        """河裡下一張對得上手牌要打的第二張:第一張只是認錯了,沒有漏掉任何一手。"""
        moves = [
            OwnMove(Move.DRAW, ("9p",)),
            OwnMove(Move.DISCARD, ("1m",)),
            OwnMove(Move.DRAW, ("7p",)),
            OwnMove(Move.DISCARD, ("2m",)),
        ]
        steps = [Step(T, 0), Step(D, 0, "N"), Step(T, 0), Step(D, 0, "2m")]
        events = body(compose(log(steps=steps, moves=moves)).events)
        assert [e.pai for e in events if isinstance(e, Dahai)] == ["1m", "2m"]

    def test_a_misread_own_river_tile_is_not_a_missed_turn(self) -> None:
        """摸牌數對得上:河裡的牌只是認錯了,不要把它當成摸到的牌。"""
        moves = [OwnMove(Move.DRAW, ("9p",)), OwnMove(Move.DISCARD, ("1m",))]
        events = body(compose(log(steps=[Step(T, 0), Step(D, 0, "N")], moves=moves)).events)
        assert events == [
            Tsumo(actor=0, pai="9p"),
            Dahai(actor=0, pai="1m", tsumogiri=False),
        ]

    def test_own_pon_uses_the_last_discard(self) -> None:
        moves = [OwnMove(Move.CALL, ("E", "E"))]
        stream = compose(log(steps=[Step(T, 2), Step(D, 2, "E")], moves=moves))
        assert body(stream.events)[-1] == Pon(actor=0, target=2, pai="E", consumed=["E", "E"])

    def test_own_chi(self) -> None:
        moves = [OwnMove(Move.CALL, ("1m", "2m"))]
        stream = compose(log(steps=[Step(T, 3), Step(D, 3, "3m")], moves=moves))
        assert body(stream.events)[-1] == Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"])


class TestLegality:
    """引擎報錯一次就整場被移除,所以不合法的事件不能送出去。"""

    def test_a_fifth_copy_in_own_tiles_stops_the_stream(self) -> None:
        """自己摸到的牌是確定的,換不掉 —— 只能停在這裡,等手牌修正。"""
        moves = [OwnMove(Move.DRAW, ("9m",))]
        hud_log = KyokuLog(
            hud=HUD,
            tehai=("9m", "9m", "9m", *TEHAI[3:]),
            doras=[(0, "9m")],
            moves=moves,
        )
        stream = compose(hud_log)
        assert stream.stalled
        assert not any(isinstance(event, Tsumo) for event in stream.events)

    def test_a_discard_from_void_stops_the_stream(self) -> None:
        moves = [OwnMove(Move.DISCARD, ("9p",), tsumogiri=True)]
        stream = compose(log(steps=[Step(D, 0, "9p")], moves=moves))
        assert stream.stalled
        assert body(stream.events) == []


class TestOwnKans:
    def test_a_closed_kan_is_four_tiles_with_no_target(self) -> None:
        moves = [OwnMove(Move.DRAW, ("E",)), OwnMove(Move.CALL, ("E", "E", "E", "E"))]
        hand = ("E", "E", "E", *TEHAI[:9], "S")
        kyoku = KyokuLog(
            hud=HUD, tehai=hand, doras=[(0, "9m")], steps=[Step(T, 0), Step(C, 0)], moves=moves
        )
        assert body(compose(kyoku).events)[-1] == Ankan(actor=0, consumed=["E"] * 4)

    def test_an_added_kan_is_one_tile(self) -> None:
        moves = [OwnMove(Move.DRAW, ("E",)), OwnMove(Move.CALL, ("E",))]
        events = body(compose(log(steps=[Step(T, 0)], moves=moves)).events)
        assert events[-1] == Kakan(actor=0, pai="E", consumed=["E", "E", "E"])


class TestOtherMelds:
    def _called(
        self, faces: tuple[str, ...], *, actor: int = 2, pai: str | None = "7p"
    ) -> MjaiEvent:
        steps = [Step(T, 1), Step(D, 1, pai), Step(C, actor, target=1), Step(D, actor, "1s")]
        melds: list[list[tuple[str, ...]]] = [[], [], [], []]
        melds[actor].append(faces)
        return body(compose(log(steps=steps, melds=melds)).events)[2]

    def test_a_skipped_seat_can_only_pon(self) -> None:
        called = self._called(("?", "?", "?"), actor=3)
        assert called == Pon(actor=3, target=1, pai="7p", consumed=["7p", "7p"])

    def test_the_next_seat_with_a_read_run_is_chi(self) -> None:
        called = self._called(("6p", "7p", "8p"))
        assert called == Chi(actor=2, target=1, pai="7p", consumed=["6p", "8p"])

    def test_the_next_seat_with_a_triple_is_pon(self) -> None:
        called = self._called(("7p", "7p", "7p"))
        assert isinstance(called, Pon)

    def test_faces_correct_a_wrong_guess(self) -> None:
        """河裡猜成 9 筒,副露認到三張 7 筒:打出來的當然是 7 筒。"""
        steps = [Step(T, 1), Step(D, 1, None, "9p"), Step(C, 2, target=1), Step(D, 2, "1s")]
        melds: list[list[tuple[str, ...]]] = [[], [], [("7p", "7p", "7p")], []]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[1] == Dahai(actor=1, pai="7p", tsumogiri=False)
        assert events[2] == Pon(actor=2, target=1, pai="7p", consumed=["7p", "7p"])

    def test_an_unread_chi_still_makes_a_run(self) -> None:
        called = self._called(("?", "?", "?"), pai="8p")
        assert isinstance(called, Chi)
        numbers = sorted(int(t[0]) for t in (called.pai, *called.consumed))
        assert numbers[2] - numbers[0] == 2

    def test_an_open_kan_takes_three_tiles(self) -> None:
        steps = [Step(T, 1), Step(D, 1, "7p"), Step(C, 3, target=1, kan=True), Step(T, 3)]
        melds: list[list[tuple[str, ...]]] = [[], [], [], [("?", "?", "?")]]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[2] == Daiminkan(actor=3, target=1, pai="7p", consumed=["7p", "7p", "7p"])

    def test_a_closed_kan_has_no_target(self) -> None:
        steps = [Step(T, 1), Step(C, 1), Step(T, 1), Step(D, 1, "1s")]
        melds: list[list[tuple[str, ...]]] = [[], [("C", "C", "C")], [], []]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[1] == Ankan(actor=1, consumed=["C"] * 4)


class TestMarks:
    def test_a_reach_wraps_its_discard(self) -> None:
        steps = [Step(T, 1), Step(D, 1, "2p", serial=0), Step(T, 1), Step(D, 1, "3p", serial=1)]
        events = body(compose(log(steps=steps, reaches=[(1, 1)])).events)
        assert events[3:] == [
            Reach(actor=1),
            Dahai(actor=1, pai="3p", tsumogiri=False),
            ReachAccepted(actor=1),
        ]

    def test_a_hidden_discard_does_not_shift_the_reach(self) -> None:
        """立直的位置是河裡第幾張,一打出就被鳴走的牌沒進過河。"""
        steps = [Step(T, 1), Step(D, 1), Step(T, 1), Step(D, 1, "3p", serial=0)]
        events = body(compose(log(steps=steps, reaches=[(1, 0)])).events)
        assert events[3] == Reach(actor=1)

    def test_own_reach_counts_own_discards(self) -> None:
        """宣言牌一打出就被鳴走時河裡沒有它,所以自己的立直照手牌數。"""
        moves = [
            OwnMove(Move.DRAW, ("9p",)),
            OwnMove(Move.DISCARD, ("9p",), tsumogiri=True),
            OwnMove(Move.DRAW, ("1p",)),
            OwnMove(Move.DISCARD, ("1p",), tsumogiri=True),
        ]
        steps = [Step(T, 0), Step(D, 0, "9p"), Step(T, 0), Step(D, 0)]
        events = body(compose(log(steps=steps, moves=moves, own_reach=1)).events)
        assert events[3:] == [
            Reach(actor=0),
            Dahai(actor=0, pai="1p", tsumogiri=True),
            ReachAccepted(actor=0),
        ]

    def test_a_later_reading_replaces_the_tile(self) -> None:
        """立直那一列要等橫放牌找到才對得準,之前讀到的是歪的。"""
        steps = [Step(T, 1), Step(D, 1, None, "E", serial=0)]
        readings = {(1, 0): ("7p", "7p")}
        events = body(compose(log(steps=steps, readings=readings)).events)
        assert events[-1] == Dahai(actor=1, pai="7p", tsumogiri=False)

    def test_a_new_dora_lands_where_it_was_seen(self) -> None:
        log_ = log(steps=[Step(T, 1), Step(D, 1, "2p"), Step(T, 2), Step(D, 2, "3p")])
        log_.doras.append((2, "4m"))
        assert body(compose(log_).events)[2] == Dora(dora_marker="4m")

    def test_more_knowledge_only_changes_what_it_touches(self) -> None:
        """牌面晚到:前綴不變,接不上的只有鳴牌那一步之後。"""
        steps = [Step(T, 1), Step(D, 1, "7p"), Step(C, 2, target=1), Step(D, 2, "1s")]
        early = compose(log(steps=steps, melds=[[], [], [("?", "?", "?")], []])).events
        late = compose(log(steps=steps, melds=[[], [], [("6p", "7p", "8p")], []])).events
        assert early[:3] == late[:3]
