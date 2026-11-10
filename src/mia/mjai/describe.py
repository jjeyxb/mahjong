"""MJAI 事件 → 給人看的中文。Log 視窗用。

牌名這裡寫成「三萬」,不用 :func:`~mia.mjai.tiles.tile_name` 的 ``3m``:那個是要跟
畫面上的牌對照,這裡是一句話。

座位一律講相對位置。封包的 ``actor`` 是絕對座位,畫面那條路的已經是依視角,所以
``seat`` 要給自己坐哪:封包給實際座位,畫面給 0。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mia.i18n import language, tr
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

#: 座位名。用到時才翻(:func:`_who`),語言換了也跟得上。
_SEATS = ("自家", "下家", "對家", "上家")
_NUMBERS = "一二三四五六七八九"
_SUITS = {"m": "萬", "p": "筒", "s": "索"}
_WINDS = dict(zip(HONOR_ORDER, HONOR_NAMES, strict=True))

#: 英文的字牌名。數牌寫成 ``3m``、``red 5p``,跟牌譜網站與 MJAI 的寫法一致。
_HONORS_EN = {
    "E": "East", "S": "South", "W": "West", "N": "North", "P": "White", "F": "Green", "C": "Red",
}


def _chinese() -> bool:
    """牌名、局名用中文寫嗎。英文介面改用 ``3m``、``East 1``。"""
    return language() != "en"


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
            return tr("對局開始")
        case EndGame():
            return tr("對局結束")
        case EndKyoku():
            return tr("本局結束")
        case StartKyoku():
            return _start_kyoku(event, seat, name)
        case Tsumo():
            if event.pai == UNKNOWN:
                return tr("{who} 摸牌", who=who)
            return tr("{who} 摸 {name}", who=who, name=name(event.pai))
        case Dahai():
            return f"{who} {_how(event, seat, exact_discards)} {name(event.pai)}"
        case Chi() | Pon() | Daiminkan():
            verb = {Chi: tr("吃"), Pon: tr("碰"), Daiminkan: tr("大明槓")}[type(event)]
            hand = tr("、").join(name(tile) for tile in event.consumed)
            return tr(
                "{who} {verb} {target}的 {name},用 {hand}",
                who=who,
                verb=verb,
                target=_who(event.target, seat),
                name=name(event.pai),
                hand=hand,
            )
        case Ankan():
            return tr("{who} 暗槓 {tiles}", who=who, tiles=_tiles(event.consumed, name))
        case Kakan():
            return tr("{who} 加槓 {name}", who=who, name=name(event.pai))
        case Dora():
            return tr("翻開新的寶牌指示牌 {name}", name=name(event.dora_marker))
        case Reach():
            return tr("{who} 宣告立直", who=who)
        case ReachAccepted():
            return tr("{who} 立直成立", who=who)
        case Hora():
            way = (
                tr("自摸")
                if event.actor == event.target
                else tr("榮和 {who}", who=_who(event.target, seat))
            )
            return f"{who} {way}{_deltas(event.deltas, seat)}"
        case Ryukyoku():
            return tr("流局{deltas}", deltas=_deltas(event.deltas, seat))
        case Kita():
            return tr("{who} 拔北", who=who)
    return event.TYPE


def _start_kyoku(event: StartKyoku, seat: int | None, name: Callable[[str], str]) -> str:
    parts = [
        tr(
            "{wind}{number}局 {honba} 本場",
            wind=_wind(event.bakaze),
            number=_NUMBERS[event.kyoku - 1] if _chinese() else event.kyoku,
            honba=event.honba,
        ),
        tr("供託 {kyotaku}", kyotaku=event.kyotaku),
        tr("莊家是{who}", who=_who(event.oya, seat)),
        tr("寶牌指示牌 {name}", name=name(event.dora_marker)),
        tr("點數 ") + _scores(event.scores, seat),
    ]
    if seat is not None:
        parts.append(tr("配牌 ") + _tiles(event.tehais[seat], name))
    return tr(",").join(parts)


def _how(event: Dahai, seat: int | None, exact: bool) -> str:
    if not exact and event.actor != seat:
        return tr("打")
    return tr("摸切") if event.tsumogiri else tr("手切")


def _who(actor: int | None, seat: int | None) -> str:
    if actor is None:
        return ""
    if seat is None:
        return tr("{actor} 家", actor=actor)
    return tr(_SEATS[(actor - seat) % len(_SEATS)])


def _scores(scores: Sequence[int], seat: int | None) -> str:
    return tr("、").join(f"{_who(actor, seat)} {score}" for actor, score in _ordered(scores, seat))


def _deltas(deltas: Sequence[int] | None, seat: int | None) -> str:
    """點數變動,沒動的不列。"""
    if not deltas:
        return ""
    moved = [f"{_who(a, seat)} {d:+d}" for a, d in _ordered(deltas, seat) if d]
    return tr(",") + tr("、").join(moved) if moved else ""


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
        return tr("未知")
    if tile in _WINDS:
        return _wind(tile)
    plain = normalize_red(tile)
    if len(plain) != 2 or plain[1] not in _SUITS or not plain[0].isdigit():
        return tile
    name = f"{_NUMBERS[int(plain[0]) - 1]}{_SUITS[plain[1]]}" if _chinese() else plain
    return tr("赤{name}", name=name) if plain != tile else name


def _wind(tile: str) -> str:
    """字牌名,場風也用這個。"""
    if _chinese():
        return _WINDS.get(tile, tile)
    return _HONORS_EN.get(tile, tile)
