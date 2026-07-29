"""把錄下來的畫面與封包推出來的手牌對起來。

錄製時兩個行程各寫各的檔:``tools/record.py`` 寫畫面,``tools/gt.py`` 寫封包。
它們唯一的共同基準是**牆上時鐘** —— session manifest 存了第一幀的 ``time.time()``
(:meth:`SessionReader.wall_at`),錄影檔每一行也有 ``wall`` 欄位。

畫面比封包慢
------------
封包在動作**發生時**就到了,畫面要等動畫演完才變。摸一張牌從封包抵達到牌真的
出現在手上,中間隔著發牌動畫;打一張牌也一樣。所以「這一幀的手牌」不能直接取
最接近的那個狀態 —— 在轉場的那半秒裡,畫面與封包必然不一致,那不是 CV 認錯。

處理方式是**只取穩定區間**:一個狀態前後 :data:`SETTLE` 秒內都沒有別的狀態變化,
落在中間的幀才拿來評分。轉場期間的幀直接跳過,不算對也不算錯。

這個取捨會讓可用的幀變少(實測約三到四成),但剩下的每一幀都有明確的標準答案。
反過來做 —— 把轉場幀算進去、用一個容忍度去湊 —— 會讓準確率變成一個由容忍度
決定的數字,那就沒有意義了。
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from majsoul_copilot.groundtruth.dump import parse_dump
from majsoul_copilot.groundtruth.liqi import LiqiParser
from majsoul_copilot.groundtruth.schema import LiqiSchema
from majsoul_copilot.groundtruth.to_mjai import MajsoulToMjai
from majsoul_copilot.mjai.handstate import HandTracker, UnknownSeatError
from majsoul_copilot.recorder.session import SessionReader
from majsoul_copilot.utils.logging import logger

__all__ = ["SETTLE", "AlignedFrame", "HandTimeline", "align", "build_timeline"]

#: 一個狀態的前後各要安靜這麼久,落在中間的幀才算「穩定」。
#:
#: 0.4 秒是估計值,不是量出來的 —— 雀魂的發牌與打牌動畫大約 0.3 秒,留一點餘裕。
#: 錄到真實素材之後應該用 :func:`~majsoul_copilot.eval.report.evaluate` 掃幾個值
#: 看準確率曲線在哪裡平掉,那個轉折點才是動畫的真實長度。
SETTLE = 0.4


@dataclass(frozen=True, slots=True)
class HandState:
    """某個時刻的標準答案手牌。

    Attributes:
        wall: 這個狀態成立的牆上時鐘(封包抵達的時間)。
        tiles: 暗手牌,含剛摸進來那張(MJAI 表示法)。
        drawn: 這一巡摸到的牌。
        melds: 副露組數。
        in_sync: 產生這個狀態時事件流有沒有對得上。False 的不該拿來評分。
    """

    wall: float
    tiles: tuple[str, ...]
    drawn: str | None
    melds: int
    in_sync: bool


@dataclass(slots=True)
class HandTimeline:
    """一整場的手牌變化,依時間排序。

    Attributes:
        states: 每次手牌改變後的狀態。
        seat: 自己的座位。
    """

    states: list[HandState] = field(default_factory=list)
    seat: int | None = None

    def __len__(self) -> int:
        return len(self.states)

    @property
    def span(self) -> tuple[float, float]:
        """第一個與最後一個狀態的牆上時鐘。空的話回 ``(0, 0)``。"""
        if not self.states:
            return (0.0, 0.0)
        return (self.states[0].wall, self.states[-1].wall)

    def at(self, wall: float) -> HandState | None:
        """這個時刻的手牌 —— 最後一個「已經發生」的狀態。

        不做穩定性檢查。要評分請用 :meth:`stable_at`。
        """
        index = bisect.bisect_right(self._walls, wall) - 1
        return self.states[index] if index >= 0 else None

    def stable_at(self, wall: float, settle: float = SETTLE) -> HandState | None:
        """這個時刻的手牌,但只在畫面「應該已經演完」時才回答。

        Args:
            wall: 幀的牆上時鐘。
            settle: 前後各要安靜多久。

        Returns:
            穩定的手牌狀態;落在轉場期間、或事件流已經對不上時回 ``None``。
        """
        index = bisect.bisect_right(self._walls, wall) - 1
        if index < 0:
            return None
        state = self.states[index]
        if not state.in_sync:
            return None
        # 動畫還沒演完
        if wall - state.wall < settle:
            return None
        # 下一個狀態已經在路上,畫面隨時會變
        if index + 1 < len(self.states) and self._walls[index + 1] - wall < settle:
            return None
        return state

    # bisect 需要一個純浮點數的平行陣列;每次重算太慢,狀態建好之後就不再變
    _walls: list[float] = field(default_factory=list, repr=False)

    def freeze(self) -> HandTimeline:
        self.states.sort(key=lambda s: s.wall)
        self._walls = [s.wall for s in self.states]
        return self


def build_timeline(dump: Path | str) -> HandTimeline:
    """從 WebSocket 錄影檔建出手牌時間軸。

    Args:
        dump: ``tools/gt.py`` 產生的 ``.jsonl``。

    Returns:
        已排序、可查詢的時間軸。

    Raises:
        ValueError: 錄影檔沒有牆上時鐘(``wall`` 欄位),無法與畫面對齊。

    Note:
        ``parse_dump`` 給的 ``frame.timestamp`` 是相對於**錄製開始**的秒數,
        兩個行程各自從自己的起點算,湊不到一起。對齊要用 ``wall_clock``,
        那是兩個獨立行程之間唯一共通的東西。
    """
    schema = LiqiSchema.load()
    converters: dict[str, MajsoulToMjai] = {}
    tracker = HandTracker(_name="ground-truth")
    timeline = HandTimeline()
    seen_wall = False

    for frame, message in parse_dump(Path(dump), schema):
        seen_wall = seen_wall or frame.wall_clock > 0
        converter = converters.get(frame.flow)
        if converter is None:
            converter = converters[frame.flow] = MajsoulToMjai(LiqiParser(schema))

        for event in converter.handle(message):
            before = (tuple(tracker.tiles), tracker.drawn)
            try:
                tracker.feed(event)
            except UnknownSeatError:
                logger.warning("錄影檔在 start_game 之前就出現 start_kyoku,略過")
                continue
            if (tuple(tracker.tiles), tracker.drawn) == before:
                continue  # 手牌沒變的事件(別家的動作)不必記一個狀態
            timeline.states.append(
                HandState(
                    wall=frame.wall_clock,
                    tiles=tuple(tracker.tiles),
                    drawn=tracker.drawn,
                    melds=tracker.melds,
                    in_sync=tracker.in_sync,
                )
            )

    if timeline.states and not seen_wall:
        # 全是 0 的話,時間軸看起來完全正常,但每一幀都會配到第一個狀態 ——
        # 準確率報告會出來一個很難看的數字,而原因與 CV 無關。當場擋掉。
        raise ValueError(
            f"{dump} 沒有 wall 欄位(2026-07-27 之前的舊格式),無法與畫面對齊。"
        )

    timeline.seat = tracker.seat
    return timeline.freeze()


@dataclass(frozen=True, slots=True)
class AlignedFrame:
    """一幀畫面,加上它對應的標準答案。

    Attributes:
        index: 在 session 裡的幀序號。
        wall: 牆上時鐘。
        truth: 這一幀應該看到的手牌。
    """

    index: int
    wall: float
    truth: HandState


def align(
    session: SessionReader,
    timeline: HandTimeline,
    *,
    settle: float = SETTLE,
    indices: Sequence[int] | None = None,
) -> Iterator[AlignedFrame]:
    """把 session 的每一幀配上標準答案,配不到的直接略過。

    Args:
        session: 已錄製的畫面。
        timeline: :func:`build_timeline` 的結果。
        settle: 穩定區間的長度。
        indices: 只處理這些幀;``None`` 表示全部。

    Yields:
        有標準答案的幀。**數量通常遠少於總幀數** —— 轉場、載入畫面、
        對局之外的時間都沒有答案可配。

    Raises:
        ValueError: session 缺少牆上時鐘錨點(2026-07-27 之前的舊格式)。
    """
    if not session.can_align:
        raise ValueError(
            f"session {session.manifest.session_id} 沒有 first_frame_wall,無法自動對齊。"
            "請用 2026-07-27 之後的格式重錄。"
        )

    wanted = set(indices) if indices is not None else None
    for record in session.manifest.frames:
        if wanted is not None and record.index not in wanted:
            continue
        wall = session.wall_at(record.timestamp)
        truth = timeline.stable_at(wall, settle)
        if truth is not None:
            yield AlignedFrame(record.index, wall, truth)
