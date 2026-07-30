"""封包擷取子程序的管理。

這一層只做三件事:組命令列、開子程序、看它有沒有還活著。擷取本身完全是
``tools/gt.py`` 的事 —— 見 :mod:`mia.live.packets` 開頭關於「為什麼是子程序」
的說明。

**src/ 引用 tools/ 是刻意的。** 一般來說相依方向該反過來,但這裡的職責就是
「協調那個工具」:把 argv 寫死在 ``tools/ui.py`` 裡的話,它就成了唯一測不到
的一段;放在這裡至少 :func:`capture_command` 是可測的,而它正是最容易寫錯的
那一部分(旗標名稱、路徑、子命令)。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from mia.live.bus import WorkerStatus
from mia.utils.logging import logger
from mia.utils.paths import PROJECT_ROOT

__all__ = ["CaptureProcess", "capture_command"]

#: 擷取工具。三種後端(cdp / proxy / local)都由它的子命令提供。
GT_TOOL = PROJECT_ROOT / "tools" / "gt.py"

#: 判定「起來了」之前要撐過的秒數。比這短就結束一定是啟動失敗
#: (Playwright 沒裝、埠被佔用),不是使用者關掉瀏覽器。
_STARTUP_GRACE = 5.0


def capture_command(
    dump: Path | str,
    *,
    mode: str = "cdp",
    url: str | None = None,
    user_data_dir: str | None = None,
    url_filter: str | None = None,
    connect: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """組出擷取子程序的命令列。

    Args:
        dump: 錄影檔輸出路徑。封包執行緒會跟著這個檔案走。
        mode: ``cdp`` / ``proxy`` / ``local``,對應 ``tools/gt.py`` 的子命令。
        url: 要開啟的網址(僅 ``cdp``)。
        user_data_dir: 持久化的瀏覽器設定檔目錄(僅 ``cdp``)。保留登入狀態,
            連打幾場時差別很大。
        url_filter: 網址關鍵字過濾。
        connect: 連到已在跑的瀏覽器而不是自己開一個(僅 ``cdp``)。
        extra: 直接附加的參數。
    """
    if mode not in ("cdp", "proxy", "local"):
        raise ValueError(f"未知的擷取模式 {mode!r},可用 cdp / proxy / local")

    command = [sys.executable, str(GT_TOOL), mode, "--out", str(dump)]
    if url_filter:
        command += ["--filter", url_filter]
    if mode == "cdp":
        if connect:
            command += ["--connect", connect]
        if url:
            command += ["--url", url]
        if user_data_dir:
            command += ["--user-data-dir", user_data_dir]
    command += extra or []
    return command


class CaptureProcess:
    """擷取子程序的生命週期。

    不讀它的 stdout —— 資料走的是錄影檔,stdout 只有給人看的進度訊息。
    子程序的 stderr 也不轉接,讓它直接印到終端機,使用者才看得到
    「請在瀏覽器中登入」這類指示。
    """

    def __init__(self, command: list[str], *, name: str = "封包擷取") -> None:
        self.command = command
        self.status = WorkerStatus(name)
        self._process: subprocess.Popen[bytes] | None = None
        self._started_at = 0.0

    def start(self) -> None:
        logger.info("啟動擷取子程序: {}", " ".join(self.command))
        try:
            self._process = subprocess.Popen(self.command)
        except OSError as exc:
            self.status.say(f"啟動擷取子程序失敗:{exc}")
            return
        self._started_at = time.monotonic()
        self.status.alive = True
        self.status.say("擷取子程序啟動中 —— 請在瀏覽器裡登入並開始對局")

    def poll(self) -> None:
        """檢查子程序是否還活著,並把結果反映到 :attr:`status`。

        由 UI 執行緒定期呼叫。分開「啟動就失敗」與「跑了一陣子才結束」兩種
        —— 前者是設定問題(該去修),後者通常是使用者自己關掉瀏覽器(正常)。
        """
        if self._process is None or not self.status.alive:
            return
        code = self._process.poll()
        if code is None:
            return

        self.status.alive = False
        elapsed = time.monotonic() - self._started_at
        if elapsed < _STARTUP_GRACE:
            self.status.say(
                f"擷取子程序啟動 {elapsed:.1f} 秒就結束了(exit {code})—— "
                f"看終端機的訊息,通常是 Playwright 沒裝或瀏覽器沒下載"
            )
        else:
            self.status.say(f"封包擷取已結束(exit {code}),不會再有新的建議")

    def stop(self, *, timeout: float = 5.0) -> None:
        """先請它自己收尾,不從就殺掉。

        用 SIGTERM 而不是直接 kill:``tools/gt.py`` 把 SIGTERM 轉成
        KeyboardInterrupt,那條路會讓 Playwright 正常關閉瀏覽器、把統計印出來。
        直接 kill 會留下一個孤兒 Chromium。
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("擷取子程序沒有在 {} 秒內結束,強制終止", timeout)
            process.kill()
            process.wait()

    def __repr__(self) -> str:
        code = self._process.poll() if self._process is not None else "未啟動"
        return f"<CaptureProcess alive={self.status.alive} exit={code}>"
