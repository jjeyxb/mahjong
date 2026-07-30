"""封包執行緒:錄影檔 → MJAI 事件 → 引擎建議。功能 2 的即時版。

為什麼是「跟著檔案走」而不是把擷取器塞進來
------------------------------------------
擷取本身仍然由 ``tools/gt.py`` 的子程序做,這一層只是跟著它寫出來的錄影檔走。
理由與 ``gt.py`` 當初決定用子程序跑 mitmdump 完全相同:**Playwright 與
mitmproxy 各自帶著自己的事件迴圈**,塞進主程式只會讓兩邊的生命週期互相糾纏
—— 而主程式這邊還有一個 Qt 事件迴圈要顧。

換來三件實質的好處:

* **三種擷取方式都能用。** cdp / proxy / local 寫出的檔案格式相同,這一層
  一行都不用改就同時支援網頁版與 Steam 版。
* **邊玩邊留下 GT 錄影。** 即時建議與離線評測用的是同一份檔案,不必為了
  產生資料集再打一場。
* **瀏覽器崩掉不會拖垮 UI。** 反之亦然。

代價是多一次落地與讀回,實測延遲遠低於一次推論的時間,而人打牌的節奏是秒級。

從頭讀而不是從尾巴讀
--------------------
預設從檔案開頭讀。座位、寶牌、誰立直了全都只在**先前**的事件裡,從尾巴接上
的引擎會拿著一副空的局面給出看起來正常但其實錯的建議。從頭讀一遍 550 個事件
實測約 1.7 秒,而郵箱會把中途的建議合併掉,所以使用者只會看到最後那個。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from mia.engine.base import AIEngine, EngineError
from mia.engine.multiplex import EngineGroup
from mia.groundtruth.dump import DumpFrame
from mia.groundtruth.stream import MjaiDecoder
from mia.live.bus import Advices, PacketHand, UpdateBus, WorkerStatus
from mia.mjai import MjaiEvent
from mia.mjai.handstate import HandTracker, UnknownSeatError
from mia.utils.logging import logger

__all__ = ["DumpTail", "PacketWorker"]

#: 檔案沒有新內容時的等待間隔。人打牌是秒級的節奏,這個值不需要更小。
_POLL = 0.15

_READ_SIZE = 1 << 16


class DumpTail:
    """跟著一個正在被寫入的錄影檔走,逐個吐出 frame。

    Args:
        path: 錄影檔路徑。**還不存在也可以** —— 會等它出現,因為擷取子程序
            通常比這邊晚幾秒才建檔。
        from_start: 從檔案開頭讀(預設),或只讀之後才寫進來的。

    Note:
        用二進位模式讀。``DumpWriter`` 每寫一行就 flush,但一次 ``write``
        在底層仍可能被切成多個 syscall,所以讀到的最後一行**可能是半截的**
        —— 這裡把不完整的尾巴留在緩衝區,等下一次讀到換行才處理。
        以位元組而非文字處理則是為了不讓多位元組字元被切在中間。
    """

    def __init__(self, path: Path | str, *, from_start: bool = True, poll: float = _POLL) -> None:
        self.path = Path(path)
        self.from_start = from_start
        self.poll = poll
        self.lines = 0
        self.bad_lines = 0

    def frames(self, stop: threading.Event) -> Iterator[DumpFrame]:
        """一直吐 frame,直到 ``stop`` 被設起來。"""
        while not self.path.exists():
            if stop.wait(self.poll):
                return

        with self.path.open("rb") as handle:
            if not self.from_start:
                handle.seek(0, os.SEEK_END)
            buffer = b""
            while not stop.is_set():
                chunk = handle.read(_READ_SIZE)
                if not chunk:
                    if stop.wait(self.poll):
                        return
                    continue
                buffer += chunk
                *complete, buffer = buffer.split(b"\n")
                for raw in complete:
                    frame = self._parse(raw)
                    if frame is not None:
                        yield frame

    def _parse(self, raw: bytes) -> DumpFrame | None:
        if not raw.strip():
            return None
        self.lines += 1
        try:
            return DumpFrame.from_json(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, ValueError, KeyError) as exc:
            self.bad_lines += 1
            logger.warning("錄影檔第 {} 行讀不出來,略過: {}", self.lines, exc)
            return None

    def __repr__(self) -> str:
        return f"<DumpTail {self.path.name} lines={self.lines} bad={self.bad_lines}>"


class PacketWorker(threading.Thread):
    """讀封包錄影、轉成 MJAI 事件、餵給引擎,把建議投進郵箱。

    Args:
        bus: 結果往哪投。
        dump: 要跟著走的錄影檔。
        engines: 要用的引擎。**座位不需要事先知道** —— ``bot.py`` 會照
            ``start_game`` 的 ``id`` 換座位,所以引擎可以在對局開始前就先起好
            (載權重要 10 秒,等到發牌才載就來不及了)。
        from_start: 見 :class:`DumpTail`。
    """

    def __init__(
        self,
        bus: UpdateBus,
        *,
        dump: Path | str,
        engines: Sequence[AIEngine],
        from_start: bool = True,
    ) -> None:
        super().__init__(name="packets", daemon=True)
        self._bus = bus
        self._tail = DumpTail(dump, from_start=from_start)
        self._group = EngineGroup(engines)
        self._decoder = MjaiDecoder()
        self._tracker = HandTracker(_name="live")
        self._stop = threading.Event()
        self.status = WorkerStatus("封包")

        #: 見過 ``start_game`` 了嗎。MJAI 引擎要靠它才知道自己是誰,在那之前
        #: 送任何事件進去都會讓引擎報錯、然後被整組移出這一場。
        self._armed = False
        #: 統計:處理過幾個事件、幾個決策點。
        self.events = 0
        self.decisions = 0

    # ------------------------------------------------------------------ 生命週期

    def stop(self) -> None:
        self._stop.set()

    @property
    def seat(self) -> int | None:
        return self._tracker.seat

    def run(self) -> None:
        self.status.alive = True
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - 執行緒裡漏出來的例外只會被吞掉
            logger.exception("封包執行緒異常結束")
            self.status.say(f"AI 建議已停止:{exc}")
        finally:
            self.status.alive = False
            self._group.close()

    def _run(self) -> None:
        self.status.say("啟動引擎…")
        try:
            self._group.start()
        except EngineError as exc:
            self.status.say(f"引擎啟動失敗:{exc}")
            return

        self.status.say(f"等待對局開始({self._tail.path.name})")
        for frame in self._tail.frames(self._stop):
            decoded = self._decoder.feed(frame)
            if decoded is None:
                continue
            for event in decoded.events:
                self._handle(event)

    # ------------------------------------------------------------------ 一個事件

    def _handle(self, event: MjaiEvent) -> None:
        if event.TYPE == "start_game":
            # 新的一場:座位可能變了,手牌追蹤要從頭來
            self._tracker = HandTracker(_name="live")
            self._armed = True

        if not self._armed:
            # 中途接上一場已經在進行的對局。引擎沒有 start_game 就不知道自己是
            # 誰,硬餵下去只會讓它報錯然後被移出整場 —— 寧可什麼都不做,等下
            # 一場。實務上不會走到這裡:擷取子程序是我們自己開的,瀏覽器從
            # 登入開始跑,authGame 一定看得到。
            self.status.say("在對局中途接上,要等下一場才有建議")
            return

        self.events += 1
        before = (tuple(self._tracker.tiles), self._tracker.drawn)
        try:
            self._tracker.feed(event)
        except UnknownSeatError as exc:
            logger.warning("手牌追蹤:{}", exc)

        after = (tuple(self._tracker.tiles), self._tracker.drawn)
        if after != before:
            # 只在真的變了才投。別家打牌的事件佔多數,那些不會動到自己的手牌,
            # 重投只會讓 UI 白算一次向聽。
            self._bus.post(PacketHand(after[0], after[1]))

        result = self._group.react(event)
        for name, reason in result.failures.items():
            self.status.say(f"{name} 掉隊 — {reason.splitlines()[0]}")
        if not self._group.failed:
            self.status.say("")

        if result.actions:
            self.decisions += 1
            self._bus.post(Advices(result.advices))

    def __repr__(self) -> str:
        return (
            f"<PacketWorker alive={self.status.alive} seat={self.seat} "
            f"events={self.events} decisions={self.decisions}>"
        )
