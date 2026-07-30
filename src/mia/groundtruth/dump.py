"""WebSocket 原始錄影的格式定義:寫入、讀取、解析成 liqi 訊息。

**擷取與解析刻意分離。** 原始位元組先落地成 JSON Lines,liqi 的解析全在離線階段。
這樣解析程式有 bug 或協定改版時,重跑一次就好 —— 不必重打一場牌。

這個檔案格式是兩種擷取後端的交會點:

* :mod:`mia.groundtruth.capture_addon` —— mitmproxy(MITM 攔截)
* :mod:`mia.groundtruth.cdp` —— Chrome DevTools Protocol(讀瀏覽器已解密的 frame)

兩者寫出的檔案在位元組層級完全相同,所以下游(解析、轉 MJAI、評測)不需要
知道資料是怎麼來的。要換擷取方式時,下游一行都不用改。

檔案格式(每行一個 frame)::

    {"t": 12.345, "wall": 1785…, "flow": "a1b2…", "dir": "s2c", "len": 87, "b64": "AQpN…"}

``t`` 是相對於錄製開始的秒數(單調時鐘),用來跟畫面錄影對齊。
``flow`` 是 WebSocket 連線識別 —— **不能省**,每條連線有自己獨立的 ``msg_id``
序列,混在同一個解析器裡會讓請求與回應互相配錯。
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from mia.groundtruth.liqi import LiqiMessage, LiqiParseError, LiqiParser
from mia.groundtruth.schema import LiqiSchema
from mia.utils.logging import logger

__all__ = ["DumpFrame", "DumpStats", "DumpWriter", "iter_frames", "parse_dump"]


@dataclass(frozen=True, slots=True)
class DumpFrame:
    """錄影檔中的一個原始 frame。"""

    timestamp: float  # 相對於錄製開始的秒數
    wall_clock: float
    from_client: bool
    payload: bytes
    flow: str = ""  # WebSocket 連線識別;舊格式的錄影沒有這個欄位

    @classmethod
    def from_json(cls, data: dict) -> DumpFrame:
        return cls(
            timestamp=float(data["t"]),
            wall_clock=float(data.get("wall", 0.0)),
            from_client=data["dir"] == "c2s",
            payload=base64.b64decode(data["b64"]),
            flow=str(data.get("flow", "")),
        )


@dataclass
class DumpStats:
    """一次解析的統計。用來判斷錄影品質。"""

    total: int = 0
    parsed: int = 0
    failed: int = 0
    actions: int = 0
    obfuscated_actions: int = 0
    flows: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def record_failure(self, reason: str) -> None:
        self.failed += 1
        # 只保留錯誤訊息的第一段,避免每個 msg_id 都算成不同原因
        key = reason.split("(")[0].strip()[:80]
        self.failures[key] = self.failures.get(key, 0) + 1

    def summary(self) -> str:
        rate = self.parsed / self.total if self.total else 0.0
        lines = [
            f"frame {self.total} 個(來自 {self.flows} 條連線),"
            f"解析成功 {self.parsed}({rate:.1%}),失敗 {self.failed}",
            f"其中牌局動作 {self.actions} 個,需 XOR 反混淆 {self.obfuscated_actions} 個",
        ]
        lines.extend(f"  失敗原因 ×{count}: {reason}" for reason, count in
                     sorted(self.failures.items(), key=lambda kv: -kv[1]))
        return "\n".join(lines)


class DumpWriter:
    """把 WebSocket frame 寫成錄影檔。

    由兩種擷取後端共用,確保寫出的格式完全一致 —— 這是「換擷取方式不影響下游」
    這個性質的實作基礎。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a", encoding="utf-8")
        self._start = time.monotonic()
        self.frames = 0

    def write(self, payload: bytes, *, from_client: bool, flow: str) -> None:
        """記錄一個 binary frame。

        Args:
            payload: frame 的完整位元組(未經任何處理)。
            from_client: True 表示客戶端送出。
            flow: WebSocket 連線識別。同一條連線必須給同一個值。
        """
        if self._file is None:
            raise RuntimeError("錄影檔已關閉")
        self._file.write(
            json.dumps(
                {
                    "t": round(time.monotonic() - self._start, 4),
                    "wall": round(time.time(), 3),
                    "flow": flow,
                    "dir": "c2s" if from_client else "s2c",
                    "len": len(payload),
                    "b64": base64.b64encode(payload).decode("ascii"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.frames += 1
        # 邊錄邊 flush:錄製幾乎都是被 Ctrl-C 中斷的,不能指望正常關檔來保資料。
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> DumpWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_frames(path: Path | str) -> Iterator[DumpFrame]:
    """逐行讀出原始 frame。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield DumpFrame.from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("錄影檔第 {} 行格式錯誤,略過: {}", line_no, exc)


def parse_dump(
    path: Path | str,
    schema: LiqiSchema | None = None,
    *,
    stats: DumpStats | None = None,
) -> Iterator[tuple[DumpFrame, LiqiMessage]]:
    """讀取錄影並產出 (原始 frame, 解析結果)。

    **每條 WebSocket 連線各自維護一個解析器。** 每條連線有獨立的 ``msg_id``
    序列,共用解析器會讓請求與回應互相配錯,回應就會用錯誤的型別去解。
    這裡依 frame 的 ``flow`` 欄位分流(舊格式沒有該欄位,會全部歸到同一個
    解析器,行為與過去相同)。

    Args:
        schema: 共用的協定 schema。載入一次約需數百毫秒,傳入可避免重複載入。
        stats: 傳入即會就地累計統計,供呼叫端在迭代結束後檢視。
    """
    shared_schema = schema or LiqiSchema.load()
    parsers: dict[str, LiqiParser] = {}
    counters = stats if stats is not None else DumpStats()

    for frame in iter_frames(path):
        counters.total += 1
        parser = parsers.get(frame.flow)
        if parser is None:
            parser = parsers[frame.flow] = LiqiParser(shared_schema)
        try:
            message = parser.parse(frame.payload, from_client=frame.from_client)
        except LiqiParseError as exc:
            counters.record_failure(str(exc))
            continue
        counters.parsed += 1
        if message.is_action:
            counters.actions += 1
            if message.deobfuscated:
                counters.obfuscated_actions += 1
        yield frame, message

    counters.flows = len(parsers)
