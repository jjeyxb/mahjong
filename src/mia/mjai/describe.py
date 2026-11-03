"""MJAI 事件 → 給人看的中文。Log 視窗用。

牌名這裡寫成「三萬」,不用 :func:`~mia.mjai.tiles.tile_name` 的 ``3m``:那個是要跟
畫面上的牌對照,這裡是一句話。

座位一律講相對位置。封包的 ``actor`` 是絕對座位,畫面那條路的已經是依視角,所以
``seat`` 要給自己坐哪:封包給實際座位,畫面給 0。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Hora,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)
from mia.mjai.tiles import HONOR_NAMES, HONOR_ORDER, UNKNOWN, normalize_red

__all__ = ["describe", "tile_text"]

_SEATS = ("自家", "下家", "對家", "上家")
_NUMBERS = "一二三四五六七八九"
_SUITS = {"m": "萬", "p": "筒", "s": "索"}
_WINDS = dict(zip(HONOR_ORDER, HONOR_NAMES, strict=True))


def describe(
    event: MjaiEvent,
    *,
    seat: int | None = None,
    exact_discards: bool = True,
    tile: Callable[[str], str] | None = None,
) -> str:
    """一個事件的中文說明。

    Args:
        event: 事件。
        seat: 自己坐哪。不知道時(對局中途接上)改用絕對座位講。
        exact_discards: 別家的 ``tsumogiri`` 是真的嗎。不是的話只寫「打」,不寫摸切或手切。
        tile: 牌怎麼寫,收 MJAI 記法。預設是中文牌名(「三萬」);Log 視窗用它換成牌面圖。
    """
    name = tile or _tile
    who = _who(getattr(event, "actor", None), seat)
    match event:
        case StartGame():
            return "對局開始"
        case EndGame():
            return "對局結束"
        case EndKyoku():
            return "本局結束"
        case StartKyoku():
            return _start_kyoku(event, seat, name)
        case Tsumo():
            return f"{who} 摸牌" if event.pai == UNKNOWN else f"{who} 摸 {name(event.pai)}"
        case Dahai():
            return f"{who} {_how(event, seat, exact_discards)} {name(event.pai)}"
        case Chi() | Pon() | Daiminkan():
            verb = {Chi: "吃", Pon: "碰", Daiminkan: "大明槓"}[type(event)]
            hand = "、".join(name(tile) for tile in event.consumed)
            return f"{who} {verb} {_who(event.target, seat)}的 {name(event.pai)},用 {hand}"
        case Ankan():
            return f"{who} 暗槓 {_tiles(event.consumed, name)}"
        case Kakan():
            return f"{who} 加槓 {name(event.pai)}"
        case Dora():
            return f"翻開新的寶牌指示牌 {name(event.dora_marker)}"
        case Reach():
            return f"{who} 宣告立直"
        case ReachAccepted():
            return f"{who} 立直成立"
        case Hora():
            way = "自摸" if event.actor == event.target else f"榮和 {_who(event.target, seat)}"
            return f"{who} {way}{_deltas(event.deltas, seat)}"
        case Ryukyoku():
            return f"流局{_deltas(event.deltas, seat)}"
        case Kita():
            return f"{who} 拔北"
    return event.TYPE


def _start_kyoku(event: StartKyoku, seat: int | None, name: Callable[[str], str]) -> str:
    parts = [
        f"{_WINDS.get(event.bakaze, event.bakaze)}{_NUMBERS[event.kyoku - 1]}局 "
        f"{event.honba} 本場",
        f"供託 {event.kyotaku}",
        f"莊家是{_who(event.oya, seat)}",
        f"寶牌指示牌 {name(event.dora_marker)}",
        "點數 " + _scores(event.scores, seat),
    ]
    if seat is not None:
        parts.append("配牌 " + _tiles(event.tehais[seat], name))
    return ",".join(parts)


def _how(event: Dahai, seat: int | None, exact: bool) -> str:
    if not exact and event.actor != seat:
        return "打"
    return "摸切" if event.tsumogiri else "手切"


def _who(actor: int | None, seat: int | None) -> str:
    if actor is None:
        return ""
    return f"{actor} 家" if seat is None else _SEATS[(actor - seat) % len(_SEATS)]


def _scores(scores: Sequence[int], seat: int | None) -> str:
    return "、".join(f"{_who(actor, seat)} {score}" for actor, score in _ordered(scores, seat))


def _deltas(deltas: Sequence[int] | None, seat: int | None) -> str:
    """點數變動,沒動的不列。"""
    if not deltas:
        return ""
    moved = [f"{_who(a, seat)} {d:+d}" for a, d in _ordered(deltas, seat) if d]
    return f",{'、'.join(moved)}" if moved else ""


def _ordered(values: Sequence[int], seat: int | None) -> list[tuple[int, int]]:
    """從自家開始排:封包的順序是絕對座位,照那個排每個人看到的順序都不一樣。"""
    actors = [((seat or 0) + offset) % len(values) for offset in range(len(values))]
    return [(actor, values[actor]) for actor in actors]


def _tiles(tiles: Sequence[str], name: Callable[[str], str]) -> str:
    return " ".join(name(tile) for tile in tiles)


def tile_text(tile: str) -> str:
    """牌的中文名(「三萬」「赤五筒」「東」)。:func:`describe` 預設就是這樣寫牌。"""
    return _tile(tile)


def _tile(tile: str) -> str:
    """牌的中文名。認不得的原樣回傳:顯示用的,不該因為多了一種牌就崩掉。"""
    if tile == UNKNOWN:
        return "未知"
    if tile in _WINDS:
        return _WINDS[tile]
    plain = normalize_red(tile)
    if len(plain) != 2 or plain[1] not in _SUITS or not plain[0].isdigit():
        return tile
    name = f"{_NUMBERS[int(plain[0]) - 1]}{_SUITS[plain[1]]}"
    return f"赤{name}" if plain != tile else name
