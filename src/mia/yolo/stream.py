"""Convert revisable visual observations into MIA's complete-round snapshots.

The upstream MortalGuard intentionally ignores some corrections. This adapter
instead rebuilds from the ledger each time and lets StreamFeeder replay changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from mia.mjai.events import MjaiEvent, StartKyoku, parse_event
from mia.mjai.tiles import normalize_red
from mia.tracker import Stream

CALLS = frozenset(("chi", "pon", "daiminkan", "ankan", "kakan"))
_TYPES = CALLS | {"tsumo", "dahai", "reach", "reach_accepted", "dora"}


def _seat(value: object) -> int | None:
    if isinstance(value, str) and value in ("P0", "P1", "P2", "P3"):
        return int(value[1])
    return None


def _initial_from_hand(events: Sequence[MjaiEvent], hand: Sequence[str]) -> list[str] | None:
    """Undo accepted own moves from a contemporaneous, stable physical hand."""
    tiles = Counter(hand)
    for event in reversed(events[1:]):
        if getattr(event, "actor", None) != 0:
            continue
        body = event.to_dict()
        if event.TYPE == "tsumo":
            tiles[body["pai"]] -= 1
        elif event.TYPE == "dahai":
            tiles[body["pai"]] += 1
        elif event.TYPE in ("chi", "pon", "daiminkan", "ankan"):
            tiles.update(body["consumed"])
        elif event.TYPE == "kakan":
            tiles[body["pai"]] += 1
    if sum(tiles.values()) != 13 or any(n < 0 for n in tiles.values()):
        return None
    result = list(tiles.elements())
    return sorted(result) if _counts_ok(result) else None


def _counts_ok(tiles: Sequence[str]) -> bool:
    return (
        all(t != "?" for t in tiles)
        and all(n <= 4 for n in Counter(normalize_red(t) for t in tiles).values())
        and all(n <= 1 for t, n in Counter(tiles).items() if t.endswith("r"))
    )


class LedgerStream:
    """Re-evaluate causal order and own inventory; never deduplicate by tile alone."""

    def __init__(self) -> None:
        self.hand_id: str | None = None
        self.kyoku = 0
        self.rejected: tuple[str, ...] = ()
        self._initial: list[str] | None = None

    def update(self, payload: Mapping[str, Any]) -> Stream:
        hand_id = str(payload.get("hand_id", ""))
        if hand_id != self.hand_id:
            self.hand_id = hand_id
            self.kyoku += 1
            self._initial = None
        ledger = payload.get("events", [])
        start = next((e for e in ledger if e["type"] == "start_kyoku"), None)
        if not start or not start.get("export_eligible"):
            return Stream(stalled=True)
        sm = start.get("seat_map") or {}
        me = sm.get("P0")
        if not isinstance(me, int) or me not in range(4):
            return Stream(stalled=True)
        try:
            msg = {
                k: start[k]
                for k in (
                    "bakaze",
                    "kyoku",
                    "honba",
                    "kyotaku",
                    "oya",
                    "scores",
                    "tehais",
                    "dora_marker",
                )
            }
            msg["type"] = "start_kyoku"
            msg["oya"] = (msg["oya"] - me) % 4
            for key in ("scores", "tehais"):
                msg[key] = [msg[key][(me + i) % 4] for i in range(4)]
            events: list[MjaiEvent] = [parse_event(msg)]
        except KeyError, TypeError, ValueError, IndexError:
            return Stream(stalled=True)

        turn, can_discard = msg["oya"], False
        last_discard: Mapping[str, Any] | None = None
        accepted_discards: set[str] = set()
        claimed: set[str] = set()
        rejected: list[str] = []
        own_last: Mapping[str, Any] | None = None
        waiting_own = False
        for e in ledger:
            typ = e["type"]
            cancelled_reach = typ == "reach" and e.get("status") == "CANCELLED"
            if (e.get("status") == "CANCELLED" and not cancelled_reach) or typ not in _TYPES:
                continue
            if not e.get("export_eligible") and not cancelled_reach:
                if e.get("actor") == "P0" and typ in CALLS | {"tsumo", "dahai"}:
                    waiting_own = True
                continue
            actor, target = _seat(e.get("actor")), _seat(e.get("target"))
            eid = e["event_id"]
            body: dict[str, Any] = {"type": typ}
            if actor is not None:
                body["actor"] = actor
            if target is not None:
                body["target"] = target
            if typ == "tsumo":
                if actor != turn or can_discard:
                    rejected.append(eid)
                    continue
                if actor == 0 and e.get("pai") in (None, "?"):
                    waiting_own = True
                    break
                body["pai"] = e["pai"]
                can_discard = True
            elif typ == "dahai":
                if actor == turn and not can_discard and actor != 0:
                    # Opponent draws have no visible face, including the dealer's
                    # fourteenth starting tile. Infer only on the expected turn.
                    events.append(parse_event({"type": "tsumo", "actor": actor, "pai": "?"}))
                    can_discard = True
                if actor != turn or not can_discard:
                    rejected.append(eid)
                    continue
                assert actor is not None
                body.update(pai=e["pai"], tsumogiri=bool(e["tsumogiri"]))
                last_discard = e
                accepted_discards.add(eid)
                can_discard, turn = False, (actor + 1) % 4
            elif typ in ("chi", "pon", "daiminkan"):
                source = e.get("source_discard")
                if (
                    last_discard is None
                    or actor is None
                    or target is None
                    or source not in accepted_discards
                    or source in claimed
                    or source != last_discard["event_id"]
                    or target != _seat(last_discard.get("actor"))
                    or e.get("pai") != last_discard.get("pai")
                    or (typ == "chi" and actor != (target + 1) % 4)
                ):
                    rejected.append(eid)
                    continue
                # A provisional next-player draw can precede a late call banner.
                if events[-1].TYPE == "tsumo":
                    events.pop()
                body.update(pai=e["pai"], consumed=list(e["consumed"]))
                claimed.add(source)
                turn, can_discard = actor, typ != "daiminkan"
            elif typ in ("ankan", "kakan"):
                if actor != turn or not can_discard:
                    rejected.append(eid)
                    continue
                body["consumed"] = list(e["consumed"])
                if typ == "kakan":
                    body["pai"] = e["pai"]
                can_discard = False
            elif typ == "dora":
                body["dora_marker"] = e["dora_marker"]
            try:
                event = parse_event(body)
            except KeyError, TypeError, ValueError:
                break
            events.append(event)
            if actor == 0 and typ in CALLS | {"tsumo", "dahai"}:
                own_last = e
        self.rejected = tuple(rejected)

        physical = payload.get("hand") or {}
        full = physical.get("tiles") or []
        if (
            physical.get("stable")
            and own_last is not None
            and int(physical.get("frame", -1)) >= int(own_last.get("frame_first", 0))
        ):
            repaired = _initial_from_hand(events, full)
            if repaired is not None:
                self._initial = repaired
        if self._initial is not None:
            initial = events[0].to_dict()
            initial["tehais"][0] = list(self._initial)
            events[0] = parse_event(initial)

        reason = validate_inventory(events)
        if waiting_own and reason is None:
            reason = "等待自己的摸打辨識"
        return Stream(tuple(events), stalled=waiting_own, doubt=reason)


def validate_inventory(events: Sequence[MjaiEvent]) -> str | None:
    """Check own hand and visible physical copies before feeding an engine."""
    if not events or not isinstance(events[0], StartKyoku):
        return "等待開局資訊"
    hand = Counter(events[0].tehais[0])
    visible = Counter(hand)
    visible[events[0].dora_marker] += 1
    if sum(hand.values()) != 13 or not _counts_ok(list(visible.elements())):
        return "起手牌或寶牌張數對不上"
    for event in events[1:]:
        body = event.to_dict()
        actor = body.get("actor")
        typ = event.TYPE
        newly_seen: list[str] = []
        spent: list[str] = []
        if typ == "dora":
            newly_seen = [body["dora_marker"]]
        elif actor == 0:
            if typ == "tsumo":
                hand[body["pai"]] += 1
                newly_seen = [body["pai"]]
            elif typ == "dahai":
                spent = [body["pai"]]
            elif typ in ("chi", "pon", "daiminkan", "ankan"):
                spent = body["consumed"]
            elif typ == "kakan":
                spent = [body["pai"]]
        elif typ in ("dahai", "kakan"):
            newly_seen = [body["pai"]]
        elif typ in ("chi", "pon", "daiminkan", "ankan"):
            newly_seen = body["consumed"]
        hand.subtract(spent)
        if any(n < 0 for n in hand.values()):
            return "自己的摸打或鳴牌與手牌對不上，等待畫面更正"
        visible.update(newly_seen)
        if not _counts_ok(list(visible.elements())):
            return "可見牌超過四張，等待事件更正"
    return None
