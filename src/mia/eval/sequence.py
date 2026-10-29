"""沒有時間基準時,靠「牌河只會一張一張變」把畫面與狀態序列對起來。

為什麼需要
----------
即時對局的封包是一則一則跟著畫面來的,每一幀用 ``wall`` 就配得到那一刻的
標準答案(:func:`~mia.eval.rivers.align_rivers`)。牌譜不是:整場紀錄在
``fetchGameRecord`` **一開始一次**送完,畫面卻要幾分鐘才演完。時間線上每個
狀態的 ``wall`` 都一樣,``stable_at`` 回答不了任何問題。

但順序還在。牌譜演的就是那一串動作,而且**不會倒退** —— 畫面上看到的狀態
序號只增不減。所以問題變成:給一串畫面觀察、一串狀態,找一條不倒退的配法,
讓兩邊最不衝突。這是序列比對(動態規劃),與 DTW 同一類。

代價怎麼算
----------
每一家分開比,而且**只用這一幀自己就判得出可信的資訊**:

* 那一家的河不連續(:attr:`~mia.vision.tiles.river.River.is_plausible`
  不成立)—— 攤平歪了或被立繪蓋住,這一家這一幀不提供任何資訊,代價 0。
* 張數不同 —— 每差一張 :data:`COUNT_WEIGHT`。
* 有信心的格子與狀態不同 —— 每格 1。沒信心的格子(``?``)不算。

張數佔大頭是刻意的:一局之內每一個 ``dahai`` 正好讓某一家多一張,所以
四家的張數組合幾乎就唯一決定了狀態;牌面只在張數打平時幫忙分辨。

⚠ 這個對齊會偏袒 CV
--------------------
對齊是用 CV 自己的讀取決定的。狀態交界處,讀短一張的那一幀會被配到
**前一個**狀態 —— 而在那個狀態下它看起來是對的。即時素材的 ``stable_at``
用「演完了沒」跳過交界,這裡沒有時間可以用,所以改成**丟掉每一段頭尾
幾幀**(:func:`interior`)。

這個偏差有多大不是用猜的:即時素材兩種對齊都做得出來,拿同一批讀取結果
各評一次,差多少就是偏多少。見 ``docs/cv_mjai.md`` §3.4。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mia.eval.rivers import RiverFrameResult, RiverState
from mia.tracker import UNKNOWN
from mia.vision.tiles.river import CONFIDENT, SEATS

__all__ = [
    "COUNT_WEIGHT",
    "Observation",
    "align_monotone",
    "cost_matrix",
    "interior",
    "observe",
]

#: 張數每差一張的代價,以「一格有信心卻不符」為 1 計。
#:
#: 大於 1 是因為張數錯比牌面錯更不可能是 CV 的問題:採用的幀張數正確率
#: 是九成以上,而有信心卻認錯是 0.19%。真的出現張數不符,幾乎都是配錯了
#: 狀態。
COUNT_WEIGHT = 2.0


@dataclass(frozen=True, slots=True)
class Observation:
    """一幀畫面能提供的資訊。

    Attributes:
        rivers: 依視角座位排好的四家。每一家是讀出來的牌(沒信心的是
            :data:`~mia.tracker.UNKNOWN`),或 ``None`` —— 這一家這一幀
            不可信,**不是**「讀到零張」。兩者的差別就是整個對齊能不能用:
            空桌是很強的證據(新局開始),被蓋住則什麼都不是。
    """

    rivers: tuple[tuple[str, ...] | None, ...]

    @property
    def informative(self) -> bool:
        return any(r is not None for r in self.rivers)


def observe(
    results: Sequence[RiverFrameResult],
    *,
    min_score: float = CONFIDENT[0],
    min_margin: float = CONFIDENT[1],
) -> Observation:
    """把 :func:`~mia.eval.rivers.read_rivers` 的結果收成一個觀察。"""
    by_seat = {r.seat: r for r in results}
    rivers: list[tuple[str, ...] | None] = []
    for seat in SEATS:
        result = by_seat.get(seat)
        if result is None or not result.contiguous:
            rivers.append(None)
            continue
        rivers.append(
            tuple(
                label if result.confident_at(i, min_score, min_margin) else UNKNOWN
                for i, label in enumerate(result.predicted)
            )
        )
    return Observation(tuple(rivers))


def _seat_cost(seen: tuple[str, ...], truth: tuple[str, ...]) -> float:
    cost = COUNT_WEIGHT * abs(len(seen) - len(truth))
    cost += sum(1 for a, b in zip(seen, truth, strict=False) if a != UNKNOWN and a != b)
    return cost


def cost_matrix(observations: Sequence[Observation], states: Sequence[RiverState]) -> np.ndarray:
    """``(幀數, 狀態數)`` 的代價表。

    逐格算是幀數 × 狀態數 × 四家,幾百萬次比對。但相鄰狀態之間只有一家
    在變,所以每一家**不同的河**其實很少 —— 先對每一家各自去重,算一張
    小表再查,快兩個數量級。
    """
    total = np.zeros((len(observations), len(states)), dtype=np.float64)
    for position in range(len(SEATS)):
        truths: dict[tuple[str, ...], int] = {}
        state_ids = np.array(
            [truths.setdefault(s.rivers[position], len(truths)) for s in states], dtype=np.intp
        )
        seens: dict[tuple[str, ...], int] = {}
        frame_ids = np.full(len(observations), -1, dtype=np.intp)
        for t, observation in enumerate(observations):
            seen = observation.rivers[position]
            if seen is not None:
                frame_ids[t] = seens.setdefault(seen, len(seens))
        if not seens:
            continue
        table = np.array(
            [[_seat_cost(seen, truth) for truth in truths] for seen in seens], dtype=np.float64
        )
        rows = frame_ids >= 0
        total[rows] += table[frame_ids[rows]][:, state_ids]
    return total


def align_monotone(cost: np.ndarray) -> np.ndarray:
    """找一條不倒退的配法,讓總代價最小。回傳每一幀配到的狀態索引。

    平手時**往後配**。這只影響沒有資訊的幀(代價整列相同),而那些幀
    本來就不評分;對有資訊的幀,平手代表兩個狀態在畫面上真的分不出來
    —— 最常見的是「打出去又立刻被鳴走」,前後兩個狀態看得見的河一模一樣。
    往後配在那種情況下猜「已經鳴了」,往前配猜「還沒打」;兩者都可能錯,
    所以報告要把 ``full`` 的一致率分開列。
    """
    frames, states = cost.shape
    if frames == 0 or states == 0:
        return np.zeros(frames, dtype=np.intp)
    best = np.empty_like(cost)
    best[0] = cost[0]
    for t in range(1, frames):
        best[t] = cost[t] + np.minimum.accumulate(best[t - 1])

    path = np.empty(frames, dtype=np.intp)
    k = _last_argmin(best[-1])
    path[-1] = k
    for t in range(frames - 2, -1, -1):
        k = _last_argmin(best[t, : k + 1])
        path[t] = k
    return path


def _last_argmin(row: np.ndarray) -> int:
    return len(row) - 1 - int(np.argmin(row[::-1]))


def interior(path: np.ndarray, edge: int) -> np.ndarray:
    """每一段相同狀態的頭尾各丟 ``edge`` 幀,回傳留下來的遮罩。

    這是 :meth:`~mia.eval.rivers.RiverTimeline.stable_at` 的替代品:沒有時間
    可以問「動畫演完了沒」,只好問「這一幀離交界夠不夠遠」。交界附近正是
    對齊會偏袒 CV 的地方(見模組說明)。
    """
    keep = np.ones(len(path), dtype=bool)
    if edge <= 0 or len(path) == 0:
        return keep
    starts = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
    ends = np.r_[starts[1:], len(path)]
    for start, end in zip(starts, ends, strict=True):
        keep[start : min(end, start + edge)] = False
        keep[max(start, end - edge) : end] = False
    return keep
