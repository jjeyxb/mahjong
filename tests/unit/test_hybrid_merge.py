"""混合辨識在事件層合併兩條事件流(mia.hybrid.merge)。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from mia.hybrid.merge import (
    LAG_OK,
    SETTLE,
    Side,
    StreamMerger,
    TileChoice,
    canonical,
    merge,
    plausible,
)
from mia.mjai.events import (
    Ankan,
    Dahai,
    Dora,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    StartKyoku,
    Tsumo,
)
from mia.tracker import Stream

HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "W"]


def start(kyoku: int = 1) -> StartKyoku:
    return StartKyoku(
        bakaze="E", kyoku=kyoku, honba=0, kyotaku=0, oya=0, dora_marker="1p",
        tehais=[HAND, ["?"] * 13, ["?"] * 13, ["?"] * 13], scores=[25000] * 4,
    )


def turn(actor: int, pai: str, drawn: str = "?") -> list[MjaiEvent]:
    return [Tsumo(actor=actor, pai=drawn if actor == 0 else "?"),
            Dahai(actor=actor, pai=pai, tsumogiri=False)]


def side(events: Sequence[MjaiEvent], wall: float = 0.0) -> Side:
    s = Side()
    s.update(tuple(events), wall)
    return s


# 自己打 W,下家打 9m,對家打 N,上家打 1s,自己再摸。
OPENING = [start(), *turn(0, "W", "C"), *turn(1, "9m"), *turn(2, "N"), *turn(3, "1s")]
NEXT = [*turn(0, "C", "2s"), *turn(1, "7m")]


def test_identical_streams_pass_through():
    events = [*OPENING, *NEXT]
    assert merge(side(events), side(events)) == tuple(events)


def test_a_misread_opponent_tile_takes_the_yolo_face():
    misread = [*OPENING[:4], *turn(1, "8m"), *OPENING[6:], *NEXT]
    assert merge(side(misread), side([*OPENING, *NEXT])) == (*OPENING, *NEXT)


def test_a_phantom_call_gives_way_to_the_normal_turn_yolo_saw():
    """樣板讀出對家碰下家的 9m,YOLO 在同一個位置看到對家照常摸打。"""
    phantom = [*OPENING[:5], Pon(actor=2, target=1, pai="9m", consumed=["9m", "9m"]),
               Dahai(actor=2, pai="N", tsumogiri=False), *OPENING[7:], *NEXT]
    assert merge(side(phantom), side([*OPENING, *NEXT])) == (*OPENING, *NEXT)


def test_a_call_yolo_never_saw_is_kept():
    """被立刻鳴走的牌,YOLO 連那張捨牌帶碰整段都沒有。"""
    called = [*OPENING[:5], Pon(actor=2, target=1, pai="9m", consumed=["9m", "9m"]),
              Dahai(actor=2, pai="N", tsumogiri=False), *OPENING[7:], *NEXT]
    missing = [*OPENING[:3], *OPENING[7:], *NEXT]
    assert merge(side(called), side(missing)) == tuple(called)


def test_own_moves_stay_with_the_template():
    yolo_misread = [*OPENING, *turn(0, "C", "3s"), *turn(1, "7m")]
    assert merge(side([*OPENING, *NEXT]), side(yolo_misread)) == (*OPENING, *NEXT)


def test_a_call_of_an_own_discard_is_an_opponent_move():
    """別家碰自己打的牌不改變自己的手牌,照樣聽 YOLO。"""
    phantom = [*OPENING[:3], Pon(actor=1, target=0, pai="W", consumed=["W", "W"]),
               Dahai(actor=1, pai="9m", tsumogiri=False), *OPENING[6:], *NEXT]
    assert merge(side(phantom), side([*OPENING, *NEXT])) == (*OPENING, *NEXT)


def test_a_late_yolo_is_not_trusted():
    misread = [*OPENING[:4], *turn(1, "8m"), *OPENING[6:], *NEXT]
    template = side(misread, wall=10.0)
    yolo = side([*OPENING, *NEXT], wall=10.0 + LAG_OK + 1)
    assert merge(template, yolo) == tuple(misread)


def test_the_template_runs_ahead_of_yolo():
    ahead = [*OPENING, *NEXT, *turn(2, "P")]
    assert merge(side(ahead), side(OPENING)) == tuple(ahead)


def test_a_different_round_is_ignored():
    assert merge(side([start(1), *OPENING[1:]]), side([start(2), *OPENING[1:]]))[0] == start(1)


def test_an_ankan_without_its_dora_is_not_plausible():
    kan = [Tsumo(actor=3, pai="?"), Ankan(actor=3, consumed=["P", "P", "P", "P"])]
    rinshan, dora = Tsumo(actor=3, pai="?"), Dora(dora_marker="9m")
    discard = Dahai(actor=3, pai="2s", tsumogiri=False)
    assert not plausible([*OPENING, *kan, rinshan, discard])
    # 樣板把翻寶牌排在嶺上摸牌後面,YOLO 排在前面,兩種都對。
    assert plausible([*OPENING, *kan, rinshan, dora, discard])
    assert plausible([*OPENING, *kan, dora, rinshan, discard])
    assert plausible([*OPENING, *kan, rinshan])


def test_five_of_a_tile_is_not_plausible():
    fives = [start(), *turn(1, "N"), *turn(2, "N"), *turn(3, "N"), *turn(1, "N"), *turn(2, "N")]
    assert not plausible(fives)
    assert plausible(fives[:-2])


def test_only_the_implausible_correction_is_dropped():
    """YOLO 修對一張,另一張讀錯成第五張北:只退讀錯的那段。"""
    # 北已經看得見四張:對家、上家、自己摸到、下家。
    before = [start(), *turn(0, "W", "C")]
    middle = [*turn(2, "N"), *turn(3, "N"), *turn(0, "N", "N"), *turn(1, "N"), *turn(2, "1s"),
              *turn(3, "2s"), *turn(0, "C", "3s")]
    after = [*turn(2, "P"), *turn(3, "S")]
    truth = [*before, *turn(1, "9m"), *middle, *turn(1, "7m"), *after]
    template = [*before, *turn(1, "8m"), *middle, *turn(1, "7m"), *after]
    yolo = [*before, *turn(1, "9m"), *middle, *turn(1, "N"), *after]
    assert not plausible(yolo)
    assert merge(side(template), side(yolo)) == tuple(truth)


def test_a_kan_yolo_missed_is_kept():
    """2026-11-08 真人場:YOLO 漏了對家暗槓 9p,只讀到翻寶牌,而且排在那家摸牌之前。"""
    head = [*OPENING[:5]]
    template = [*head, Tsumo(actor=2, pai="?"), Ankan(actor=2, consumed=["9p"] * 4),
                Tsumo(actor=2, pai="?"), Dora(dora_marker="3s"),
                Dahai(actor=2, pai="6p", tsumogiri=True), *OPENING[7:], *NEXT]
    yolo = [*head, Dora(dora_marker="3s"), Tsumo(actor=2, pai="?"),
            Dahai(actor=2, pai="6p", tsumogiri=True), *OPENING[7:], *NEXT]
    assert merge(side(template), side(yolo)) == tuple(template)


def test_a_kan_the_template_read_late_is_not_doubled():
    """錄影 053001 西二局三本場:樣板把暗槓晚讀一巡,YOLO 在對的位置讀到。槓只能有一個。"""
    kan = [Tsumo(actor=2, pai="?"), Ankan(actor=2, consumed=["9p"] * 4),
           Tsumo(actor=2, pai="?"), Dora(dora_marker="3s")]
    tail = [*turn(3, "2s")]
    template = [*OPENING, *NEXT, *kan, Dahai(actor=2, pai="S", tsumogiri=False), *tail]
    yolo = [*OPENING[:5], *kan, Dahai(actor=2, pai="N", tsumogiri=False), *OPENING[7:], *NEXT,
            *turn(2, "S"), *tail]
    assert merge(side(template), side(yolo)) == tuple(yolo)


def cut(events: Sequence[MjaiEvent], actor: int, pai: str) -> list[MjaiEvent]:
    """把某家打的那張改成摸切。"""
    return [
        Dahai(actor=e.actor, pai=e.pai, tsumogiri=True)
        if isinstance(e, Dahai) and e.actor == actor and e.pai == pai else e
        for e in events
    ]


def test_opponent_tsumogiri_follows_yolo():
    """樣板看手牌形狀估摸切,YOLO 看摸進來的牌還在不在,準得多。兩邊不一致時聽 YOLO 的。"""
    template = cut([*OPENING, *NEXT], 1, "9m")
    yolo = cut([*OPENING, *NEXT], 3, "1s")
    assert merge(side(template), side(yolo)) == tuple(yolo)


def test_own_and_ruled_cuts_stay_with_the_template():
    """自己的摸切樣板清楚;鳴牌後打的一定是手切,規則定的不必聽 YOLO。"""
    called = [*OPENING[:5], Pon(actor=2, target=1, pai="9m", consumed=["9m", "9m"]),
              Dahai(actor=2, pai="N", tsumogiri=False), *OPENING[7:], *NEXT]
    yolo = cut(cut(called, 2, "N"), 0, "C")
    assert merge(side(called), side(yolo)) == tuple(called)


def test_reach_accepted_goes_back_behind_the_declaration():
    """YOLO 把立直成立排在下家摸牌之後:排回去,兩邊就對得齊,照樣聽樣板的順序。"""
    declare = [Tsumo(actor=3, pai="?"), Reach(actor=3), Dahai(actor=3, pai="1s", tsumogiri=False)]
    packet = [*OPENING[:7], *declare, ReachAccepted(actor=3), *NEXT]
    yolo = [*OPENING[:7], *declare, NEXT[0], ReachAccepted(actor=3), *NEXT[1:]]
    assert canonical(yolo) == tuple(packet)
    assert merge(side(packet), side(yolo)) == tuple(packet)


def test_a_reach_accepted_yolo_has_not_seen_yet_is_kept():
    declare = [Tsumo(actor=3, pai="?"), Reach(actor=3), Dahai(actor=3, pai="1s", tsumogiri=False)]
    template = [*OPENING[:7], *declare, ReachAccepted(actor=3), *NEXT]
    yolo = [*OPENING[:7], *declare, *NEXT]
    assert merge(side(template), side(yolo)) == tuple(template)


def test_reach_accepted_before_its_declaration_is_dropped_for_now():
    """立直棒先讀到、宣言牌還沒有:不插到那家上一巡的捨牌後面,也不能留著當一致點。"""
    declare = [Tsumo(actor=3, pai="?"), Reach(actor=3)]
    early = [*OPENING[:7], *declare, ReachAccepted(actor=3)]
    assert canonical(early) == (*OPENING[:7], *declare)
    template = [*OPENING[:7], *declare, Dahai(actor=3, pai="1s", tsumogiri=False),
                ReachAccepted(actor=3), *NEXT]
    assert merge(side(template), side(early)) == tuple(template)


class TestStreamMerger:
    def setup_method(self):
        self.now = 0.0
        self.out: list[tuple[int, Stream]] = []
        self.merger = StreamMerger(lambda k, s: self.out.append((k, s)), clock=lambda: self.now)

    def offer(self, source: str, kyoku: int, events: Sequence[MjaiEvent], **flags) -> None:
        self.merger.offer(source, kyoku, Stream(events=tuple(events), **flags))

    def test_the_template_skeleton_with_yolo_corrections(self):
        misread = [*OPENING[:4], *turn(1, "8m"), *OPENING[6:]]
        self.offer(StreamMerger.TEMPLATE, 1, misread)
        assert self.out[-1][1].events == tuple(misread)
        self.now = 0.5
        self.offer(StreamMerger.YOLO, 7, [*OPENING, *NEXT[:1]])
        assert self.out[-1][1].events == tuple(OPENING)

    def test_unchanged_results_are_not_published_again(self):
        self.offer(StreamMerger.TEMPLATE, 1, OPENING)
        self.offer(StreamMerger.YOLO, 1, OPENING)
        self.offer(StreamMerger.TEMPLATE, 1, OPENING)
        assert len(self.out) == 1

    def test_flags_follow_the_template(self):
        self.offer(StreamMerger.TEMPLATE, 1, OPENING, stalled=True, doubt="等手牌")
        assert self.out[-1][1].doubt == "等手牌"
        assert self.out[-1][1].stalled

    def test_yolo_fills_in_until_the_template_reads_the_new_round(self):
        self.offer(StreamMerger.TEMPLATE, 1, [start(1), *OPENING[1:]])
        self.offer(StreamMerger.YOLO, 1, [start(1), *OPENING[1:]])
        first = self.out[-1][0]
        self.offer(StreamMerger.TEMPLATE, 2, [])
        # YOLO 還在上一局:不能拿來當新的一局。
        assert self.out[-1][1].events == ()
        second = replace(start(2), dora_marker="5s")
        opening = [second, *turn(0, "S", "4m"), *turn(1, "2p")]
        self.offer(StreamMerger.YOLO, 2, opening[:3])
        kyoku, stream = self.out[-1]
        assert stream.events[0] == second
        assert kyoku == first + 1
        self.offer(StreamMerger.TEMPLATE, 2, opening)
        assert self.out[-1][0] == kyoku
        assert self.out[-1][1].events == tuple(opening)

    def test_a_renchan_that_is_still_the_last_round_is_not_used(self):
        """2026-11-09 CPU 場:連莊時 YOLO 只把本場數加一,配牌、寶牌與整局事件還是上一局的。"""
        self.offer(StreamMerger.TEMPLATE, 1, [start(1), *OPENING[1:], *NEXT])
        self.offer(StreamMerger.TEMPLATE, 2, [])
        stale = replace(start(1), honba=1)
        self.offer(StreamMerger.YOLO, 2, [stale, *OPENING[1:], *NEXT])
        assert self.out[-1][1].events == ()

    def test_a_new_round_that_repeats_what_yolo_saw_last_round_is_not_used(self):
        """同一場:YOLO 連局數都改了,開局之後還是上一局的事件。樣板的配牌也跟 YOLO 差一張。"""
        self.offer(StreamMerger.YOLO, 1, [start(1), *OPENING[1:], *NEXT])
        self.offer(StreamMerger.TEMPLATE, 1, [replace(start(1), dora_marker="9s"), *OPENING[1:]])
        self.offer(StreamMerger.TEMPLATE, 2, [])
        self.offer(StreamMerger.YOLO, 2, [replace(start(2), dora_marker="5s"), *OPENING[1:], *NEXT])
        assert self.out[-1][1].events == ()


class TestTileChoices:
    """兩邊讀成不同牌的別家捨牌,穩定了才問玩家,玩家選的優先。"""

    # 下家第一張:樣板讀 8m,YOLO 讀 9m。
    MISREAD = (*OPENING[:4], Dahai(actor=1, pai="8m", tsumogiri=False), *OPENING[5:], *NEXT)
    RIGHT = (*OPENING, *NEXT)

    def setup_method(self):
        self.now = 0.0
        self.out: list[tuple[int, Stream]] = []
        self.asked: list[tuple[int, tuple[TileChoice, ...]]] = []
        self.merger = StreamMerger(
            lambda k, s: self.out.append((k, s)),
            on_choices=lambda k, c: self.asked.append((k, c)),
            clock=lambda: self.now,
        )

    def offer_both(self, template: Sequence[MjaiEvent], yolo: Sequence[MjaiEvent]) -> None:
        self.merger.offer(StreamMerger.TEMPLATE, 1, Stream(events=tuple(template)))
        self.merger.offer(StreamMerger.YOLO, 1, Stream(events=tuple(yolo)))

    def test_a_fresh_disagreement_is_not_asked_yet(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        assert self.asked == []

    def test_a_settled_disagreement_is_asked(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        self.now = SETTLE
        self.offer_both(self.MISREAD, self.RIGHT)
        kyoku, choices = self.asked[-1]
        assert choices == (TileChoice(kyoku, 1, 0, template="8m", yolo="9m", chosen="9m"),)

    def test_the_player_pick_wins_and_is_not_asked_again(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        self.now = SETTLE
        self.offer_both(self.MISREAD, self.RIGHT)
        kyoku, _ = self.asked[-1]
        self.merger.pick(kyoku, 1, 0, "8m")
        assert self.out[-1][1].events == tuple(self.MISREAD)
        assert self.asked[-1] == (kyoku, ())
        self.now = SETTLE * 2
        self.offer_both(self.MISREAD, self.RIGHT)
        assert self.out[-1][1].events == tuple(self.MISREAD)

    def test_a_pick_outside_the_two_readings_is_ignored(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        self.now = SETTLE
        self.offer_both(self.MISREAD, self.RIGHT)
        kyoku, _ = self.asked[-1]
        self.merger.pick(kyoku, 1, 0, "C")
        assert self.out[-1][1].events == tuple(self.RIGHT)

    def test_agreeing_later_drops_the_question(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        self.now = SETTLE
        self.offer_both(self.MISREAD, self.RIGHT)
        self.offer_both(self.RIGHT, self.RIGHT)
        assert self.asked[-1][1] == ()

    def test_a_pick_for_the_last_round_is_ignored(self):
        self.offer_both(self.MISREAD, self.RIGHT)
        self.now = SETTLE
        self.offer_both(self.MISREAD, self.RIGHT)
        kyoku, _ = self.asked[-1]
        self.merger.pick(kyoku - 1, 1, 0, "8m")
        assert self.out[-1][1].events == tuple(self.RIGHT)

    def test_an_extra_discard_is_not_a_choice(self):
        """多一張、少一張不是「哪張牌」的問題,不問。"""
        extra = [*OPENING, *turn(1, "8m"), *NEXT]
        self.offer_both([*OPENING, *NEXT], extra)
        self.now = SETTLE
        self.offer_both([*OPENING, *NEXT], extra)
        assert all(choices == () for _, choices in self.asked)
