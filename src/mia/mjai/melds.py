"""Structural checks for a visually reconstructed four-player meld."""

from __future__ import annotations

from collections import Counter

from mia.mjai.events import Ankan, Chi, Daiminkan, Kakan, MjaiEvent, Pon
from mia.mjai.tiles import HONOR_ORDER, normalize_red


def valid_meld(event: MjaiEvent) -> bool:
    """Non-melds pass; calls must have legal faces, sizes, and source seats.

    This checks the meld itself, not whether it was actually observed or whether
    its consumed tiles are in a player's hand. Those require stream evidence.
    """
    if not isinstance(event, (Chi, Pon, Daiminkan, Ankan, Kakan)):
        return True
    if event.actor not in range(4):
        return False
    if isinstance(event, (Chi, Pon, Daiminkan)):
        if event.target not in range(4) or event.target == event.actor:
            return False
        if isinstance(event, Chi) and event.actor != (event.target + 1) % 4:
            return False
    tiles = [*event.consumed]
    if not isinstance(event, Ankan):
        tiles.append(event.pai)
    expected = 3 if isinstance(event, (Chi, Pon)) else 4
    if len(tiles) != expected:
        return False
    plain = [normalize_red(tile) for tile in tiles]
    if any(
        tile not in HONOR_ORDER
        and not (len(tile) == 2 and tile[0] in "123456789" and tile[1] in "mps")
        for tile in plain
    ):
        return False
    if any(n > 1 for tile, n in Counter(tiles).items() if tile.endswith("r")):
        return False
    if not isinstance(event, Chi):
        return len(set(plain)) == 1
    if any(tile in HONOR_ORDER for tile in plain) or len({tile[1] for tile in plain}) != 1:
        return False
    ranks = sorted(int(tile[0]) for tile in plain)
    return ranks == list(range(ranks[0], ranks[0] + 3))
