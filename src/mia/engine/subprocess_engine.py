"""以子程序執行的引擎:JSON Lines over stdio。

為什麼是子程序
--------------
``libriichi`` 不在 PyPI(要從 Mortal repo 自行編譯)、``mjai`` 在 PyPI 上只有
cp312 wheel —— 兩者都裝不進主程式的 Python 3.14 環境。這個技術限制本身就決定了
架構。附帶的兩個好處:torch 載入慢且崩潰時不該拖垮 UI,以及多模型比較可以直接
起 N 個子程序。授權隔離**不是**理由,本專案自己就是 AGPL-3.0。

協定
----
一行進、一行出,嚴格一對一::

    父 → 子   {"type": "tsumo", "actor": 0, "pai": "3s"}
    子 → 父   {"type": "dahai", "actor": 0, "pai": "1z", "tsumogiri": false}
              或 {"type": "none"}          ← 這一手不需要動作

啟動時子程序要先送一行 ``{"type": "hello", ...}``,父程序收到才算就緒。這一步
不是客套:torch 載入權重要好幾秒,沒有握手就分不出「還在載入」與「已經死了」。

一對一是刻意的。MJAI 原本的形式是「有動作才輸出」,但那樣父程序無法區分
「引擎不打算動作」與「引擎還在算」,只能靠逾時猜 —— 每一手都要等滿逾時,
一場 250 手就是好幾分鐘的純等待。多送一個 ``none`` 就完全消掉這個問題。

崩潰之後
--------
MJAI 引擎是**有狀態**的,它自己從事件流重建局面。所以重啟一個引擎不等於復原
—— 新的行程對這一局一無所知。:attr:`SubprocessEngine.replay_on_restart` 預設
會把先前送過的事件全部重送一次,讓它追上進度。不重送的話,引擎會用一個空白的
局面繼續給建議,而且**不會報錯** —— 那種錯誤最難察覺。
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from mia.engine.base import Advice, EngineError, EngineTimeout
from mia.mjai import NONE_ACTION, MjaiEvent, MjaiFormatError, parse_event
from mia.utils.logging import logger

__all__ = ["SubprocessEngine", "SubprocessSpec"]

#: 讀取執行緒遇到 EOF 時推進佇列的哨兵。用物件而非 None —— None 是合法的
#: JSON 值,拿它當哨兵日後一定會撞在一起。
_EOF = object()

#: stderr 只留最後這麼多行。子程序可能刷出大量 torch 警告,全留會吃光記憶體;
#: 但完全不留,崩潰時就只剩一個 returncode,查不出原因。
_STDERR_LINES = 60

_HELLO = "hello"


@dataclass(frozen=True, slots=True)
class SubprocessSpec:
    """怎麼把子程序叫起來。

    Attributes:
        name: 引擎名稱,會出現在 :attr:`~mia.engine.base.Advice.engine`。
        argv: 完整命令列,第一個元素是直譯器路徑。
        cwd: 工作目錄,``None`` 表示沿用父程序的。子程序端應自己處理
            ``sys.path``,不要靠 cwd —— 那樣父程序從哪裡啟動都不影響結果。
        env: 追加的環境變數,會疊在目前環境之上而非取代。
        startup_timeout: 等 ``hello`` 的秒數。預設放得寬 —— 載入 130 MB 的
            權重在冷啟動時要好幾秒。
        react_timeout: 每一手的秒數。實測 Mortal 單次推論約 14 ms,5 秒是
            兩個數量級的餘裕,會超過就是真的出事了。
    """

    name: str
    argv: Sequence[str]
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    startup_timeout: float = 60.0
    react_timeout: float = 5.0


class SubprocessEngine:
    """在子程序裡跑的 :class:`~mia.engine.base.AIEngine`。

    Args:
        spec: 子程序的啟動方式。
        replay_on_restart: 崩潰重啟後是否重送歷史事件。關掉只有在「你確定
            接下來會從 ``start_game`` 重新開始」時才正確。
    """

    def __init__(self, spec: SubprocessSpec, *, replay_on_restart: bool = True) -> None:
        self.spec = spec
        self.replay_on_restart = replay_on_restart
        self._process: subprocess.Popen[str] | None = None
        self._lines: Queue[str | object] = Queue()
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        self._readers: list[threading.Thread] = []
        self._history: list[str] = []
        self._closed = False

    # ------------------------------------------------------------------ 介面

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._closed:
            raise EngineError(f"{self.name}: 已經 close(),不能再啟動")
        if self.is_running:
            return
        self._spawn()

    def react(self, event: MjaiEvent) -> Advice:
        if not self.is_running:
            raise EngineError(f"{self.name}: 尚未啟動 —— 先呼叫 start()")

        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        reply = self._exchange(line)
        elapsed = (time.perf_counter() - started) * 1000

        self._history.append(line)
        return self._to_advice(reply, elapsed)

    def peek(self, event: MjaiEvent) -> Advice:
        """問一個假設性的後續,然後把引擎復原。

        復原的方式是**砍掉重練 + 重播真實歷史**。子程序裡的 libriichi ``Bot``
        沒有 rollback,所以沒有更便宜的做法 —— 一旦事件送進去,它的內部狀態就
        變了。這裡刻意不去記帳「等一下如果使用者真的立直了就不用復原」:那個
        最佳化要靠猜使用者接下來做什麼,而猜錯的代價是整場都拿著錯的局面給建議,
        還**不會報錯**。慢一點換確定正確。

        代價實測:重啟 + 重播約一秒(視局面進行到多後面)。而引擎會回 ``reach``
        的時機一場大概兩次,而且那正是使用者停下來讀建議的那一刻。
        """
        if not self.is_running:
            raise EngineError(f"{self.name}: 尚未啟動 —— 先呼叫 start()")

        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        reply = self._exchange(line)
        elapsed = (time.perf_counter() - started) * 1000
        # **不**寫進 _history —— 這件事沒有真的發生
        advice = self._to_advice(reply, elapsed)
        self.restart()
        return advice

    def close(self) -> None:
        self._closed = True
        self._terminate()

    def restart(self) -> None:
        """砍掉重練。``replay_on_restart`` 為真時會重送歷史事件追上進度。

        Raises:
            EngineError: 重啟或重播失敗。
        """
        self._terminate()
        self._spawn()
        if not self.replay_on_restart or not self._history:
            return

        history = list(self._history)
        logger.info(f"{self.name}: 重播 {len(history)} 個事件以追上局面")
        for line in history:
            self._exchange(line)  # 回覆丟掉 —— 重播的是過去,不是現在要的建議

    def __enter__(self) -> SubprocessEngine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"<SubprocessEngine {self.name} {state}>"

    # ------------------------------------------------------------------ 內部

    def _spawn(self) -> None:
        spec = self.spec
        env = {**os.environ, **spec.env}
        # 子程序的 stdout 若被緩衝,一問一答會卡死 —— 父程序等回覆,子程序等
        # 緩衝區滿。子程序端自己也 flush,這裡是第二道保險。
        env.setdefault("PYTHONUNBUFFERED", "1")

        self._lines = Queue()
        self._stderr.clear()
        try:
            self._process = subprocess.Popen(
                list(spec.argv),
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise EngineError(f"{self.name}: 啟動失敗 —— {exc}") from exc

        self._readers = [
            self._start_reader(self._process.stdout, self._lines.put, eof=True),
            self._start_reader(self._process.stderr, self._stderr.append, eof=False),
        ]
        self._handshake()

    def _start_reader(self, stream: Any, sink: Any, *, eof: bool) -> threading.Thread:
        """把一條輸出管線抽到背景執行緒。

        ``eof`` 決定收尾時要不要推 :data:`_EOF`。stdout 需要 —— 那是父程序
        察覺子程序死掉的唯一途徑,沒有它只會等到逾時,錯誤訊息會說「太慢」
        而不是「它死了」。
        """

        def pump() -> None:
            with stream:
                for raw in stream:
                    sink(raw.rstrip("\n"))
            if eof:
                sink(_EOF)

        thread = threading.Thread(target=pump, daemon=True, name=f"{self.name}-reader")
        thread.start()
        return thread

    def _handshake(self) -> None:
        reply = self._read_line(self.spec.startup_timeout, phase="啟動")
        kind = reply.get("type")
        if kind != _HELLO:
            raise EngineError(
                f"{self.name}: 啟動時應先送 {_HELLO},卻收到 {kind!r} —— "
                f"子程序可能把其他輸出寫進了 stdout。{self._stderr_hint()}"
            )
        logger.info(f"{self.name}: 就緒 {reply.get('detail', '')}".rstrip())

    def _exchange(self, line: str) -> dict[str, Any]:
        process = self._process
        assert process is not None and process.stdin is not None
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise EngineError(
                f"{self.name}: 寫入失敗,子程序已經不在了。{self._stderr_hint()}"
            ) from exc
        return self._read_line(self.spec.react_timeout, phase="回覆")

    def _read_line(self, timeout: float, *, phase: str) -> dict[str, Any]:
        """讀一行 JSON。非 JSON 的雜訊會被跳過,不計入逾時預算之外。

        跳過而不是報錯,是因為子程序載入的第三方套件不保證只往 stderr 寫
        (torch 與 CUDA 相關的提示就有前科)。但雜訊會**留在日誌裡** ——
        靜靜吞掉的話,日後真的有人往 stdout print 出東西,只會看到莫名其妙的逾時。
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout(
                    f"{self.name}: 等待{phase}超過 {timeout:g} 秒。{self._stderr_hint()}"
                )
            try:
                item = self._lines.get(timeout=remaining)
            except Empty:
                continue

            if item is _EOF:
                code = self._exit_code_after_eof()
                raise EngineError(
                    f"{self.name}: 子程序在{phase}階段結束了(returncode={code})。"
                    f"{self._stderr_hint()}"
                )

            text = str(item).strip()
            if not text:
                continue
            if not text.startswith("{"):
                logger.warning(f"{self.name}: 忽略 stdout 上的非 JSON 輸出 — {text[:200]}")
                continue
            try:
                return dict(json.loads(text))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise EngineError(f"{self.name}: 回覆不是合法的 JSON 物件 — {text[:200]}") from exc

    def _exit_code_after_eof(self) -> int | None:
        """stdout 見到 EOF 之後查 returncode。

        管線關閉與行程真正結束在 Windows 上不是同一個時間點 ——
        ``poll()`` 不會等,常常還來得及看到 ``STILL_ACTIVE``(對應到
        ``None``)。EOF 已經代表子程序在收尾了,短暫 ``wait()`` 換一個
        準確的 returncode,不會真的卡住。
        """
        if self._process is None:
            return None
        try:
            return self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return self._process.poll()

    def _to_advice(self, reply: dict[str, Any], elapsed_ms: float) -> Advice:
        if reply.get("type") == "error":
            raise EngineError(f"{self.name}: 子程序回報錯誤 — {reply.get('message', reply)}")

        # meta 不屬於 MJAI 協定,parse_event 會忽略它 —— 先取走才不會弄丟
        meta = reply.get("meta")
        if reply.get("type") == NONE_ACTION:
            return Advice(self.name, None, meta, elapsed_ms)
        try:
            action = parse_event(reply)
        except MjaiFormatError as exc:
            raise EngineError(f"{self.name}: 回覆不是合法的 MJAI 動作 — {exc}") from exc
        return Advice(self.name, action, meta, elapsed_ms)

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            # 先關 stdin —— 子程序的迴圈讀到 EOF 會自己收尾,比 SIGTERM 乾淨
            if process.stdin is not None:
                with contextlib.suppress(OSError, ValueError):
                    process.stdin.close()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{self.name}: 不理會 terminate,改用 kill")
                    process.kill()
        for thread in self._readers:
            thread.join(timeout=1.0)
        self._readers = []

    def _stderr_hint(self) -> str:
        if not self._stderr:
            return "子程序的 stderr 沒有任何輸出。"
        tail = "\n".join(self._stderr)
        return f"子程序 stderr 最後 {len(self._stderr)} 行:\n{tail}"
