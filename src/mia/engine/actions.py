"""把 Mortal 的 Q 值對應回具體的動作。

Mortal 在 ``meta`` 裡回兩個東西:``mask_bits`` 是這一手哪些動作合法的位元遮罩,
``q_values`` 是**只含合法動作**的分數,依動作索引升冪排列。兩者 zip 起來才知道
「+1.28 分那個是切東還是立直」。

沒有這一層,UI 上就只有一串沒有標籤的數字 —— 而「為什麼這樣打」是本專題
相對於單純顯示答案的全部加值。

動作空間
--------
libriichi 的 ``ACTION_SPACE`` 是 46:

===========  ==========================================
索引         動作
===========  ==========================================
0 – 8        1m – 9m
9 – 17       1p – 9p
18 – 26      1s – 9s
27 – 33      東 南 西 北 白 發 中
34 – 36      赤五 5mr 5pr 5sr
37           立直
38 – 40      吃(三種)
41           碰
42           槓
43           和了
44           流局(九種九牌)
45           不動作
===========  ==========================================

這張表是**實測出來的**,不是抄文件。用三個構造好的局面餵真的引擎,比對
``mask_bits`` 的位元與手上實際有哪些牌:

* 0–8、9–17、18–26、27–33、37:一手 123456789m 123p 5p 摸 9s,合法動作
  恰為那 14 張牌 + 立直,位元完全對應。
* 34:手上放一張 ``5mr``,索引 34 出現在遮罩裡。
* 38–41、45:上家打 3m,手上 1m2m/2m4m/4m5m 三種吃法加一對 3m3m,
  遮罩恰為 38、39、40、41、45。
* 44:么九九種的手,遮罩含 44。

42(槓)與 43(和了)沒有直接驗到,但前後都已夾死,位置沒有別的可能。

吃的三種為什麼不細分
--------------------
38/39/40 確定是吃的三個變體,但**無法從遮罩分辨哪個對應被吃那張在順子裡的
位置** —— 那次測試三種吃法同時合法,四個索引對三種吃法沒有唯一解。所以這裡
一律標成「吃」。實際吃了哪三張要看動作本身的 ``consumed``,那是明確的。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mia.mjai.events import MjaiEvent
from mia.mjai.tiles import HONOR_ORDER

__all__ = ["ACTION_SPACE", "TILE_ACTIONS", "Candidate", "action_label", "decode_candidates"]

#: libriichi 的動作空間大小。
ACTION_SPACE = 46

#: 前這麼多個索引是「切某一張牌」。
TILE_ACTIONS = 37

#: 索引 → 標籤。牌用 MJAI 記法,與專案其他地方一致。
_LABELS: tuple[str, ...] = (
    *(f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)),
    *HONOR_ORDER,
    "5mr",
    "5pr",
    "5sr",
    "reach",
    "chi",
    "chi",
    "chi",
    "pon",
    "kan",
    "hora",
    "ryukyoku",
    "none",
)

assert len(_LABELS) == ACTION_SPACE, "標籤表與動作空間大小不符"

#: 非切牌動作的中文顯示名。
_DISPLAY = {
    "reach": "立直",
    "chi": "吃",
    "pon": "碰",
    "kan": "槓",
    "hora": "和了",
    "ryukyoku": "流局",
    "none": "不動作",
}


def action_label(event: MjaiEvent) -> str:
    """一個動作的簡短說法,給 UI 直接顯示。

    ``str(event)`` 是 debug 用的(``reach(actor=0)``),把它擺到畫面上會讓
    使用者看到 actor 編號這種與決策無關的東西。

    Returns:
        像「切 1z」、「立直」、「碰 5m」這樣。認不得的型別回 ``type`` 本身,
        不會拋例外 —— UI 在畫的時候不該因為多了一種動作就崩掉。

    Note:
        回傳值也用來比較各引擎有沒有分歧,所以**不同動作必須給出不同字串**。
        鳴牌帶上被鳴的那張就是為了這個:兩個引擎都建議「碰」但碰的不是同一張,
        那是分歧。
    """
    kind = event.TYPE
    pai = getattr(event, "pai", None)
    if kind == "dahai":
        return f"切 {pai}"
    if kind in {"chi", "pon", "daiminkan", "kakan"}:
        return f"{_DISPLAY.get(_ACTION_ALIAS.get(kind, kind), kind)} {pai}"
    if kind == "ankan":
        consumed = getattr(event, "consumed", ())
        return f"槓 {consumed[0]}" if consumed else "槓"
    return _DISPLAY.get(kind, kind)


#: MJAI 的事件型別 → 顯示名的鍵。三種槓在畫面上都只說「槓」。
_ACTION_ALIAS = {"daiminkan": "kan", "kakan": "kan", "ankan": "kan"}


@dataclass(frozen=True, slots=True)
class Candidate:
    """一個合法動作與它的分數。

    Attributes:
        index: 動作空間裡的索引。
        label: 切牌動作是牌名(MJAI 記法),其餘是 ``reach`` / ``pon`` 這種。
        q: Q 值。**是相對的** —— 只有同一手之內互相比較才有意義,
            不同局面之間的絕對值不可比。
        chosen: 是不是引擎最後選的那個(該手的最高分)。
    """

    index: int
    label: str
    q: float
    chosen: bool = False

    @property
    def is_tile(self) -> bool:
        """是不是「切某一張牌」。"""
        return self.index < TILE_ACTIONS

    @property
    def display(self) -> str:
        """給人看的名稱。切牌回牌名本身,由 UI 決定要不要畫成牌面圖。"""
        return self.label if self.is_tile else _DISPLAY.get(self.label, self.label)

    def __str__(self) -> str:
        mark = " ←" if self.chosen else ""
        return f"{self.display} {self.q:+.3f}{mark}"


def decode_candidates(meta: Mapping[str, Any] | None) -> tuple[Candidate, ...]:
    """把 ``Advice.meta`` 解成有標籤的候選動作,依 Q 值由高到低。

    Args:
        meta: 引擎回覆裡的 ``meta``。缺 ``q_values`` 或 ``mask_bits`` 時回空的
            —— 不是所有引擎都提供這些(規則式 baseline 就沒有)。

    Returns:
        候選動作,最高分在前。長度不符時回空的。

    Note:
        ``q_values`` 的長度必須等於 ``mask_bits`` 的位元數。不相等代表對這份
        引擎輸出的理解有誤,這時**回空的而不是勉強 zip** —— 錯開一格的標籤
        會讓 UI 顯示「建議切 3m」而引擎其實說的是 4m,那種錯誤看不出來。
    """
    if not meta:
        return ()
    values = meta.get("q_values")
    mask = meta.get("mask_bits")
    if not isinstance(values, list) or not isinstance(mask, int):
        return ()

    legal = [i for i in range(min(mask.bit_length(), ACTION_SPACE)) if mask >> i & 1]
    if len(legal) != len(values):
        return ()

    best = max(range(len(values)), key=lambda i: values[i]) if values else -1
    candidates = [
        Candidate(index, _LABELS[index], float(q), chosen=position == best)
        for position, (index, q) in enumerate(zip(legal, values, strict=True))
    ]
    candidates.sort(key=lambda c: -c.q)
    return tuple(candidates)
