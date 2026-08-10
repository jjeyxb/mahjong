"""牌的表示法轉換:雀魂 ↔ MJAI。

兩套表示法
----------
=========  =====================  ==========================
種類       雀魂 (liqi)            MJAI
=========  =====================  ==========================
數牌       ``1m``~``9m`` 等       同左
赤五       ``0m`` / ``0p`` / ``0s``  ``5mr`` / ``5pr`` / ``5sr``
字牌       ``1z``~``7z``          ``E S W N P F C``
未知牌     ——                     ``?``
=========  =====================  ==========================

字牌的對應順序是東南西北白發中(``1z`` = 東 … ``7z`` = 中)。

轉換失敗一律拋 :class:`TileError` 而不是回傳預設值。餵給 AI 的是一條事件流,
一張牌錯掉會讓後續整局的狀態全歪;寧可當場中止該事件,也不要靜靜地送出錯的牌。
"""

from __future__ import annotations

__all__ = [
    "HONOR_NAMES",
    "HONOR_ORDER",
    "UNKNOWN",
    "TileError",
    "is_red_five",
    "mjai_to_ms",
    "ms_to_mjai",
    "normalize_red",
    "sort_key",
    "tile_name",
]

UNKNOWN = "?"

#: ``1z``~``7z`` 依序對應的 MJAI 字牌代號:東 南 西 北 白 發 中
HONOR_ORDER = ("E", "S", "W", "N", "P", "F", "C")

_SUITS = ("m", "p", "s")

_RED_TO_MJAI = {"0m": "5mr", "0p": "5pr", "0s": "5sr"}
_MJAI_TO_RED = {v: k for k, v in _RED_TO_MJAI.items()}
_HONOR_TO_MJAI = {f"{i + 1}z": name for i, name in enumerate(HONOR_ORDER)}
_MJAI_TO_HONOR = {v: k for k, v in _HONOR_TO_MJAI.items()}


class TileError(ValueError):
    """無法辨識的牌字串。"""


def ms_to_mjai(tile: str) -> str:
    """雀魂表示法 → MJAI 表示法。

    >>> ms_to_mjai("0p")
    '5pr'
    >>> ms_to_mjai("1z")
    'E'
    >>> ms_to_mjai("7s")
    '7s'
    """
    if tile == UNKNOWN:
        return UNKNOWN
    if tile in _RED_TO_MJAI:
        return _RED_TO_MJAI[tile]
    if tile in _HONOR_TO_MJAI:
        return _HONOR_TO_MJAI[tile]
    if len(tile) == 2 and tile[1] in _SUITS and tile[0].isdigit() and tile[0] != "0":
        return tile
    raise TileError(f"無法辨識的雀魂牌字串: {tile!r}")


def mjai_to_ms(tile: str) -> str:
    """MJAI 表示法 → 雀魂表示法。建立自動打牌(M9 之後)時會用到。"""
    if tile == UNKNOWN:
        return UNKNOWN
    if tile in _MJAI_TO_RED:
        return _MJAI_TO_RED[tile]
    if tile in _MJAI_TO_HONOR:
        return _MJAI_TO_HONOR[tile]
    if len(tile) == 2 and tile[1] in _SUITS and tile[0].isdigit() and tile[0] != "0":
        return tile
    raise TileError(f"無法辨識的 MJAI 牌字串: {tile!r}")


def is_red_five(tile: str) -> bool:
    """判斷是否為赤五。接受兩種表示法。"""
    return tile in _RED_TO_MJAI or tile in _MJAI_TO_RED


#: 字牌給人看的名字,順序同 :data:`HONOR_ORDER`。
HONOR_NAMES = ("東", "南", "西", "北", "白", "發", "中")

_TO_NAME = {
    **{f"{i + 1}z": name for i, name in enumerate(HONOR_NAMES)},
    **{code: name for code, name in zip(HONOR_ORDER, HONOR_NAMES, strict=True)},
}


def tile_name(tile: str) -> str:
    """給人看的牌名。兩種表示法都收。

    只有字牌會被換掉:``1z`` 與 ``E`` 都是「東」。數牌維持 ``3m`` ——
    點數本來就是阿拉伯數字,換成「三萬」反而比對不上畫面上的牌。

    認不得的字串原樣回傳,不拋例外:這是顯示用的,不該讓 UI 因為多了一種
    牌就崩掉。

    >>> tile_name("1z"), tile_name("E"), tile_name("3m")
    ('東', '東', '3m')
    """
    return _TO_NAME.get(tile, tile)


#: 顯示用的花色順序:萬 → 筒 → 索 → 字牌。
_SUIT_ORDER = {"m": 0, "p": 1, "s": 2}
_HONOR_GROUP = 3
_UNKNOWN_GROUP = 9


def sort_key(tile: str) -> tuple[int, int, int]:
    """理牌用的排序鍵(MJAI 記法)。

    ``sorted()`` 直接排字串會把筒子插進萬子中間 —— ``1m`` < ``1p`` < ``2m``,
    所以 13 張手牌會排成 1m 1p 1p 2m 2p 3m…,畫面上看起來像壞掉。麻將的順序是
    **先花色再點數**,字牌固定在最後、依東南西北白發中。

    赤五排在同款普通五**之前**(``5mr`` 在 ``5m`` 前)。兩者點數相同但不能互換,
    固定順序才不會每次更新都跳動。

    >>> sorted(["1p", "2m", "5mr", "5m", "E"], key=sort_key)
    ['2m', '5mr', '5m', '1p', 'E']
    """
    if tile == UNKNOWN:
        return (_UNKNOWN_GROUP, 0, 0)
    if tile in _MJAI_TO_HONOR:
        return (_HONOR_GROUP, HONOR_ORDER.index(tile), 0)
    if tile in _MJAI_TO_RED:
        return (_SUIT_ORDER[tile[1]], 5, 0)
    if len(tile) == 2 and tile[1] in _SUIT_ORDER and tile[0].isdigit():
        return (_SUIT_ORDER[tile[1]], int(tile[0]), 1)
    # 認不得的牌排到最後,而不是拋例外 —— 排序是顯示用的,不該讓 UI 崩掉
    return (_UNKNOWN_GROUP, 0, 0)


def normalize_red(tile: str) -> str:
    """把赤五還原成普通五(MJAI 表示法)。

    >>> normalize_red("5mr")
    '5m'
    >>> normalize_red("3p")
    '3p'
    """
    return tile[:2] if tile in _MJAI_TO_RED else tile


def ms_list_to_mjai(tiles: list[str]) -> list[str]:
    """整串轉換。任何一張失敗就整串失敗。"""
    return [ms_to_mjai(t) for t in tiles]
