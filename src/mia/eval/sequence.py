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

局與局之間
----------
封包裡的局邊界(和了、流局、結算)也是序列裡的一步,而那段時間畫面上
**沒有可以評分的牌河** —— 結算面板蓋住、立繪蓋住。配到那裡的幀不給答案。

第一版沒有這一步,結果結算畫面被配到**下一局的空桌**:面板蓋住的那幾家
讀成 0 張而且連續,與空桌完全吻合。tracker 這時候正確地還握著上一局的河,
於是每一張都被算成假現物(一場 555 張)。

分得開靠的是 :class:`~mia.tracker.kyoku.KyokuWatcher` 早就在用的那條規則
(``docs/cv_mjai.md`` §4.1):**新局的空桌是四家都 0 張而且都讀得到**。
所以:

* 邊界的代價等於「四家都空」,但讀不到的家不扣分 —— 面板蓋住什麼都可以。
* 一局開頭那個四家都空的狀態,每有一家讀不到就加罰 :data:`UNREAD_WEIGHT`。

只讀到一部分是空的,邊界比較便宜;四家都讀成空的,兩者打平,平手往後配
進新局 —— 與 ``KyokuWatcher`` 宣告新局的時機一致。

會不會偏袒 CV
-------------
擔心過:對齊是用 CV 自己的讀取決定的,交界處讀短一張的幀可能被配到前一個
狀態而「看起來是對的」。所以寫了 :func:`interior`(丟掉每段頭尾幾幀)。

實測沒有發生。即時素材兩種對齊都做得出來,拿同一批讀取結果比:真時間
有答案的 1160 幀,這裡配到的狀態**全部相同**(含鳴走的牌)。張數佔代價
大頭,單一幀讀錯推不動整條路徑。``interior`` 留著,但不是主要的防線。

它反而看得**更多**:``stable_at`` 會跳過狀態不到 1 秒的幀(出牌快、局尾),
這裡不會 —— 而那裡有舊評分量不到的真錯誤。見 ``docs/cv_mjai.md`` §3.4。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mia.eval.rivers import RiverFrameResult, RiverState, RiverTimeline
from mia.recorder.session import SessionReader
from mia.tracker import UNKNOWN
from mia.utils.geometry import Rect
from mia.vision.tiles import river as _river
from mia.vision.tiles.river import CONFIDENT, SEATS

__all__ = [
    "COUNT_WEIGHT",
    "UNREAD_WEIGHT",
    "Observation",
    "align_monotone",
    "align_session",
    "cost_matrix",
    "fingerprint",
    "interior",
    "observe",
]

#: 張數每差一張的代價,以「一格有信心卻不符」為 1 計。
#:
#: 大於 1 是因為張數錯比牌面錯更不可能是 CV 的問題:採用的幀張數正確率
#: 是九成以上,而有信心卻認錯是 0.19%。真的出現張數不符,幾乎都是配錯了
#: 狀態。
COUNT_WEIGHT = 2.0

#: 一局開頭的空桌,每有一家讀不到的加罰。見模組說明的「局與局之間」。
#:
#: 等於 :data:`COUNT_WEIGHT`:讀不到的那一家可能藏著任何張數,當成「至少一張」
#: 來罰,就足以讓只讀到部分空桌的幀留在邊界。
UNREAD_WEIGHT = COUNT_WEIGHT

#: 邊界狀態在代價計算裡長的樣子:四家都空。
_EMPTY = ((), (), (), ())


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

    ``states`` 可以含邊界(``boundary=True``),見模組說明的「局與局之間」。

    逐格算是幀數 × 狀態數 × 四家,幾百萬次比對。但相鄰狀態之間只有一家
    在變,所以每一家**不同的河**其實很少 —— 先對每一家各自去重,算一張
    小表再查,快兩個數量級。
    """
    total = np.zeros((len(observations), len(states)), dtype=np.float64)
    rivers = [_EMPTY if s.boundary else s.rivers for s in states]
    for position in range(len(SEATS)):
        truths: dict[tuple[str, ...], int] = {}
        state_ids = np.array(
            [truths.setdefault(r[position], len(truths)) for r in rivers], dtype=np.intp
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

    # 一局開頭的空桌要四家都讀得到才算數。見模組說明的「局與局之間」。
    opening = np.array(
        [not s.boundary and all(len(r) == 0 for r in s.rivers) for s in states], dtype=bool
    )
    if opening.any():
        unread = np.array(
            [sum(r is None for r in o.rivers) for o in observations], dtype=np.float64
        )
        total[:, opening] += UNREAD_WEIGHT * unread[:, None]
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
    best = np.empty(cost.shape, dtype=np.float32)
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


def align_session(
    session: SessionReader,
    timeline: RiverTimeline,
    reads: Mapping[int, Sequence[RiverFrameResult]],
) -> dict[int, RiverState]:
    """整份錄影的每一幀配一個狀態。取代 ``stable_at`` 的那一份答案。

    Args:
        session: 錄影。
        timeline: 封包建出來的狀態序列。**只用順序,不用 wall** —— 牌譜素材
            的 wall 全部一樣。
        reads: 每一張**有寫檔的**幀(``changed``)的讀取結果,以幀序號為鍵。
            沒有變化的幀畫面與前一張相同,沿用寫出那張影像的幀。

    Returns:
        幀序號 → 狀態。兩種幀不在裡面:

        * 四家都讀不了的 —— 那一幀沒有任何證據,路徑落在哪裡只是平手時
          往後配的結果,拿它當答案等於自己編一個。
        * 配到局邊界的 —— 那段時間畫面上沒有牌河。

    Note:
        對齊在**全部幀**上做,不只有變化的那些。有變化的幀幾乎全是動畫
        中途(錄影只在畫面變化時存檔),只拿它們對齊等於只用最不可信的
        那一段;靜止的幀在這裡就是「這個狀態持續了多久」的權重。
    """
    source: dict[str, int] = {}
    for record in session.manifest.frames:
        if record.changed:
            source[record.image] = record.index
    records = [r for r in session.manifest.frames if source.get(r.image) in reads]
    observations = [observe(reads[source[r.image]]) for r in records]
    states = _steps(timeline)
    path = align_monotone(cost_matrix(observations, states))
    return {
        record.index: states[k]
        for record, k, observation in zip(records, path, observations, strict=True)
        if observation.informative and not states[k].boundary
    }


def _steps(timeline: RiverTimeline) -> list[RiverState]:
    """對齊用的狀態序列:開頭補一個邊界(開局前的大廳),連續的邊界併成一個。

    和了之後封包連著送 ``hora``、``end_kyoku``,對畫面來說是同一段「沒有牌河」,
    拆成兩步只會讓平手時多一個往後配的落點。
    """
    steps = [RiverState(0.0, (), boundary=True)]
    for state in timeline.states:
        if state.boundary and steps[-1].boundary:
            continue
        steps.append(state)
    return steps


def fingerprint(table: Rect, skin: str) -> str:
    """讀取結果的指紋。快取的讀取要跟它一起存,對不上就重讀。

    牌河怎麼讀由這些決定:四家角點(含橫放牌的尺度)、格子大小、牌面比例、搜尋窗上下延伸多少、
    覆核的第二個高度、信心門檻、皮膚、牌桌矩形。任何一個改了,舊的讀取結果就**不是現在這套
    辨識的結果** —— 而拿它評分的報告看起來完全正常。角點正是會被調的那一個。
    """
    parts = (
        repr(sorted((k, v.corners, v.sideways_scale) for k, v in _river.DEFAULT_CORNERS.items())),
        repr(_river.CELL),
        repr(_river.FACE),
        repr((_river.SEARCH_ABOVE, _river.SEARCH_BELOW)),
        repr(_river.CHECK_FACE_HEIGHT),
        repr(_river.CONFIDENT),
        skin,
        repr((table.x, table.y, table.width, table.height)),
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
