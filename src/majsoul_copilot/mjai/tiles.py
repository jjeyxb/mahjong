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
    "HONOR_ORDER",
    "UNKNOWN",
    "TileError",
    "is_red_five",
    "mjai_to_ms",
    "ms_to_mjai",
    "normalize_red",
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
