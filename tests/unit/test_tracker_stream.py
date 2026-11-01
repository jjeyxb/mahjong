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
    fields.setdefault("doras", [(0, "9m")])
    return KyokuLog(hud=HUD, tehai=TEHAI, **fields)  # type: ignore[arg-type]


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


    def test_an_added_kan_the_turns_never_saw(self) -> None:
        """排序器只看到摸打,手牌看到摸、加槓、嶺上、打。實測這裡整局卡死。"""
        hand = ("E", "E", "E", *TEHAI[:9], "S")
        moves = [
            OwnMove(Move.DRAW, ("E",)),
            OwnMove(Move.CALL, ("E",)),
            OwnMove(Move.DRAW, ("9p",)),
            OwnMove(Move.DISCARD, ("9p",), tsumogiri=True),
        ]
        kyoku = KyokuLog(
            hud=HUD,
            tehai=hand,
            doras=[(0, "9m"), (2, "C")],
            steps=[Step(T, 0), Step(D, 0, "9p")],
            moves=moves,
        )
        stream = compose(kyoku)
        assert not stream.stalled
        assert [type(e) for e in body(stream.events)] == [Tsumo, Kakan, Tsumo, Dahai, Dora]


class TestNextUp:
    def test_it_goes_on_the_tail(self) -> None:
        steps = [Step(T, 1), Step(D, 1, "2p")]
        events = body(compose(log(steps=steps, next_up=Step(D, 2, "3p", serial=0))).events)
        assert events[-2:] == [
            Tsumo(actor=2, pai="?"),
            Dahai(actor=2, pai="3p", tsumogiri=False),
        ]

    def test_it_uses_the_latest_reading(self) -> None:
        kyoku = log(
            steps=[Step(T, 1), Step(D, 1, "2p")],
            next_up=Step(D, 2, "3p", serial=0),
            readings={(2, 0): ("4p", None)},
        )
        assert body(compose(kyoku).events)[-1] == Dahai(actor=2, pai="4p", tsumogiri=False)

    def test_own_moves_come_first(self) -> None:
        """手牌看到自己摸牌了:輪到自己,不是別人。"""
        moves = [OwnMove(Move.DRAW, ("9p",))]
        steps = [Step(T, 3), Step(D, 3, "2p")]
        events = body(compose(log(steps=steps, moves=moves, next_up=Step(D, 1, "3p"))).events)
        assert events[-1] == Tsumo(actor=0, pai="9p")

    def test_the_wall_has_seventy_tiles(self) -> None:
        """第 71 次摸牌引擎會報錯,然後整場再也沒有建議。"""
        steps = [Step(T, 1 + turn % 3) for turn in range(70)]
        steps = [s for step in steps for s in (step, Step(D, step.actor))]
        stream = compose(log(steps=steps, next_up=Step(D, 1, "3p")))
        assert sum(isinstance(event, Tsumo) for event in stream.events) == 70
        assert stream.stalled

    def test_a_fifth_copy_is_cut_off(self) -> None:
        """讀錯成第五張:留著引擎會報錯,切掉就好,下一輪會重算。"""
        steps = [Step(T, 1), Step(D, 1, "E")]
        events = body(compose(log(steps=steps, next_up=Step(D, 2, "E"), doras=[(0, "E")])).events)
        assert not any(isinstance(e, Dahai) and e.actor == 2 for e in events)


class TestOwnCallsCorrectTheRiver:
    """手上拿出去的牌來自手牌,河裡認錯的那張要跟著改,否則別家的捨牌就記錯了。"""

    @staticmethod
    def _called(river: str, *tiles: str) -> Chi | Pon:
        steps = [Step(T, 3), Step(D, 3, river), Step(C, 0, target=3)]
        moves = [OwnMove(Move.CALL, tiles)]
        stream = compose(log(steps=steps, moves=moves))
        event = body(stream.events)[-1]
        assert isinstance(event, Chi | Pon)
        assert body(stream.events)[1].pai == event.pai  # type: ignore[attr-defined]
        return event

    def test_a_pon_names_the_tile(self) -> None:
        assert self._called("C", "E", "E").pai == "E"

    def test_a_run_with_a_hole_names_the_tile(self) -> None:
        assert self._called("6p", "4p", "6p").pai == "5p"

    def test_a_river_reading_that_fits_is_kept(self) -> None:
        assert self._called("3p", "4p", "5p").pai == "3p"
        assert self._called("6p", "4p", "5p").pai == "6p"

    def test_a_wrong_reading_picks_a_tile_that_fits(self) -> None:
        assert self._called("C", "4p", "5p").pai in ("3p", "6p")


class TestCallsTheRiverNeverSaw:
    """一打出來就被鳴走的牌,河裡沒有。Steam 實測的那一局。"""

    def test_own_discard_is_never_rewritten_by_meld_faces(self) -> None:
        steps = [
            Step(T, 3),
            Step(D, 3, "2p"),
            Step(T, 0),
            Step(D, 0, "1s"),
            Step(C, 1, target=0),
            Step(D, 1, "9m"),
        ]
        moves = [OwnMove(Move.DRAW, ("9s",)), OwnMove(Move.DISCARD, ("1m",))]
        melds: list[list[tuple[str, ...]]] = [[], [("P", "P", "P")], [], []]
        stream = compose(log(steps=steps, melds=melds, moves=moves))
        events = body(stream.events)
        assert Dahai(actor=0, pai="1m", tsumogiri=False) in events
        assert Dahai(actor=3, pai="P", tsumogiri=False) in events
        assert Pon(actor=1, target=3, pai="P", consumed=["P", "P"]) in events
        assert not stream.stalled

    def test_an_own_turn_the_sorter_invented_is_dropped(self) -> None:
        steps = [
            Step(T, 0),
            Step(D, 0, "1m", serial=0),
            Step(T, 1),
            Step(D, 1, "1s"),
            Step(T, 0),
            Step(D, 0),
            Step(T, 1),
            Step(D, 1, "2s"),
            Step(T, 0),
            Step(D, 0, "2m", serial=1),
        ]
        moves = [
            OwnMove(Move.DRAW, ("9s",)),
            OwnMove(Move.DISCARD, ("1m",)),
            OwnMove(Move.DRAW, ("9p",)),
            OwnMove(Move.DISCARD, ("2m",)),
        ]
        stream = compose(log(steps=steps, moves=moves))
        own = [e.pai for e in stream.events if isinstance(e, Tsumo) and e.actor == 0]
        assert own == ["9s", "9p"]
        assert not stream.stalled

    def test_an_own_discard_called_at_once_is_a_real_turn(self) -> None:
        steps = [Step(T, 0), Step(D, 0), Step(C, 1, target=0), Step(D, 1, "9m")]
        moves = [OwnMove(Move.DRAW, ("9s",)), OwnMove(Move.DISCARD, ("1m",))]
        melds: list[list[tuple[str, ...]]] = [[], [("1m", "1m", "1m")], [], []]
        events = body(compose(log(steps=steps, melds=melds, moves=moves)).events)
        assert events[1] == Dahai(actor=0, pai="1m", tsumogiri=False)
        assert events[2] == Pon(actor=1, target=0, pai="1m", consumed=["1m", "1m"])


class TestRedInMelds:
    def _stream(
        self,
        pai: str,
        faces: tuple[str, ...],
        suits: str = "mps",
        *,
        tehai: tuple[str, ...] = TEHAI,
    ) -> list[MjaiEvent]:
        steps = [Step(T, 1), Step(D, 1, pai), Step(C, 2, target=1), Step(D, 2, "1s")]
        melds: list[list[tuple[str, ...]]] = [[], [], [faces], []]
        reds = {(2, 0): frozenset(suits)}
        kept = KyokuLog(
            hud=HUD, tehai=tehai, doras=[(0, "9m")], steps=steps, melds=melds, meld_reds=reds
        )
        return body(compose(kept).events)

    def test_the_five_in_the_hand_turns_red(self) -> None:
        called = self._stream("6s", ("4s", "5s", "6s"))[2]
        assert called == Chi(actor=2, target=1, pai="6s", consumed=["4s", "5sr"])

    def test_the_called_five_turns_red_in_the_river_too(self) -> None:
        events = self._stream("5s", ("6s", "7s", "?"))
        assert events[1] == Dahai(actor=1, pai="5sr", tsumogiri=False)
        assert events[2] == Chi(actor=2, target=1, pai="5sr", consumed=["6s", "7s"])

    def test_a_pon_of_fives(self) -> None:
        called = self._stream("5p", ("5p", "5p", "5p"))[2]
        assert called == Pon(actor=2, target=1, pai="5p", consumed=["5pr", "5p"])

    def test_a_run_without_a_five_stays_as_it_is(self) -> None:
        called = self._stream("7m", ("6m", "7m", "8m"))[2]
        assert called == Chi(actor=2, target=1, pai="7m", consumed=["6m", "8m"])

    def test_the_threshold_of_another_suit_does_not_count(self) -> None:
        called = self._stream("6m", ("4m", "5m", "6m"), "ps")[2]
        assert called == Chi(actor=2, target=1, pai="6m", consumed=["4m", "5m"])

    def test_there_is_only_one_red_five_of_a_suit(self) -> None:
        tehai = ("5sr", *TEHAI[1:])
        called = self._stream("6s", ("4s", "5s", "6s"), tehai=tehai)[2]
        assert called == Chi(actor=2, target=1, pai="6s", consumed=["4s", "5s"])


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

    def test_a_lone_face_with_no_copies_left_is_a_chi(self) -> None:
        """只認到被鳴的 5 筒,而另外三張都看得到了:碰不了,只可能是吃。"""
        steps = [Step(T, 2), Step(D, 2, "5p"), Step(C, 3, target=2), Step(D, 3, "E")]
        melds: list[list[tuple[str, ...]]] = [[], [], [], [("5p", "?", "?")]]
        kyoku = log(steps=steps, melds=melds, doras=[(0, "5p")])
        called = body(compose(kyoku).events)[2]
        assert isinstance(called, Chi)
        assert called.pai == "5p"

    def _added_kan(self, *doras: tuple[int, str]) -> list[MjaiEvent]:
        """對家碰了下家的發,下一輪摸牌。``doras`` 是開局之後翻開的寶牌。"""
        steps = [
            Step(T, 1),
            Step(D, 1, "F"),
            Step(C, 2, target=1),
            Step(D, 2, "1p"),
            Step(T, 3),
            Step(D, 3, "2p"),
            Step(T, 1),
            Step(D, 1, "3p"),
            Step(T, 2),
            Step(D, 2, "4p"),
        ]
        melds: list[list[tuple[str, ...]]] = [[], [], [("F", "F", "F")], []]
        return body(compose(log(steps=steps, melds=melds, doras=[(0, "9m"), *doras])).events)

    def test_a_dora_with_no_kan_is_an_added_kan(self) -> None:
        events = self._added_kan((6, "C"))
        kan = events.index(Kakan(actor=2, pai="F", consumed=["F", "F", "F"]))
        assert events[kan - 1] == Tsumo(actor=2, pai="?")
        assert events[kan + 1] == Tsumo(actor=2, pai="?")
        assert events[kan + 2] == Dora(dora_marker="C")
        assert events[kan + 3] == Dahai(actor=2, pai="4p", tsumogiri=False)

    def test_the_kan_goes_to_whoever_has_a_pon(self) -> None:
        """寶牌翻開時排序還落後:下家先摸,但他沒有碰,不是他。"""
        events = self._added_kan((4, "C"))
        assert Kakan(actor=2, pai="F", consumed=["F", "F", "F"]) in events

    def test_no_new_dora_no_kan(self) -> None:
        assert not any(isinstance(event, Kakan) for event in self._added_kan())

    def test_the_fourth_copy_has_to_exist(self) -> None:
        """第四張已經翻成寶牌指示牌了:不可能加槓,送出去還會是第五張。"""
        events = self._added_kan((6, "F"))
        assert not any(isinstance(event, Kakan) for event in events)
        assert Dora(dora_marker="F") in events

    def test_a_kan_we_can_see_keeps_its_dora(self) -> None:
        """大明槓看得到:寶牌是它的,不要多補一次加槓。"""
        steps = [
            Step(T, 1),
            Step(D, 1, "F"),
            Step(C, 2, target=1),
            Step(D, 2, "1p"),
            Step(T, 3),
            Step(D, 3, "E"),
            Step(C, 1, target=3, kan=True),
            Step(T, 1),
            Step(D, 1, "3p"),
            Step(T, 2),
            Step(D, 2, "4p"),
        ]
        melds: list[list[tuple[str, ...]]] = [[], [], [("F", "F", "F")], []]
        kyoku = log(steps=steps, melds=melds, doras=[(0, "9m"), (6, "C")])
        events = body(compose(kyoku).events)
        assert not any(isinstance(event, Kakan) for event in events)
        kan = next(i for i, e in enumerate(events) if isinstance(e, Daiminkan))
        assert events[kan + 2] == Dora(dora_marker="C")

    def test_a_closed_kan_of_fives_holds_the_red(self) -> None:
        steps = [Step(T, 1), Step(C, 1), Step(T, 1), Step(D, 1, "1s")]
        melds: list[list[tuple[str, ...]]] = [[], [("5s", "5s", "5s")], [], []]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[1] == Ankan(actor=1, consumed=["5sr", "5s", "5s", "5s"])

    def test_an_open_kan_of_fives_holds_the_red(self) -> None:
        steps = [Step(T, 1), Step(D, 1, "5m"), Step(C, 3, target=1, kan=True), Step(T, 3)]
        melds: list[list[tuple[str, ...]]] = [[], [], [], [("5m", "?", "?")]]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[2] == Daiminkan(actor=3, target=1, pai="5m", consumed=["5mr", "5m", "5m"])

    def test_an_open_kan_on_the_red_itself(self) -> None:
        steps = [Step(T, 1), Step(D, 1, "5mr"), Step(C, 3, target=1, kan=True), Step(T, 3)]
        melds: list[list[tuple[str, ...]]] = [[], [], [], [("5m", "?", "?")]]
        events = body(compose(log(steps=steps, melds=melds)).events)
        assert events[2] == Daiminkan(actor=3, target=1, pai="5mr", consumed=["5m", "5m", "5m"])

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
