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
from collections.abc import Callable
from pathlib import Path

from mia.live.bus import WorkerStatus
from mia.live.control import ControlFile
from mia.utils.logging import logger
from mia.utils.paths import PROJECT_ROOT

__all__ = ["CaptureLauncher", "CaptureProcess", "capture_command"]

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
    canvas: str | None = None,
    control: Path | str | None = None,
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
        canvas: 要把瀏覽器 viewport 調成的尺寸,例如 ``1920x1080``(僅 ``cdp``)。
            ``connect`` 時**刻意不送** —— 那個視窗不是我們開的,不該去動它。
        control: 控制檔路徑(僅 ``cdp``)。開下去之後主程式還想改設定就只能
            靠它,見 :mod:`mia.live.control`。與 ``canvas`` 同樣不送給 ``connect``。
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
        elif canvas or control:
            if canvas:
                command += ["--canvas", canvas]
            if control:
                command += ["--control", str(control)]
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

    def __init__(
        self, command: list[str], *, name: str = "封包擷取", status: WorkerStatus | None = None
    ) -> None:
        self.command = command
        #: 可以由外面傳進來 —— :class:`CaptureLauncher` 每開一場就換一個
        #: ``CaptureProcess``,但狀態列讀的必須是同一個物件,否則換場之後
        #: UI 還盯著上一場那個已經不會再更新的狀態。
        self.status = status if status is not None else WorkerStatus(name)
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


class CaptureLauncher:
    """「開始遊戲」按下去之後發生的事:開一場新的擷取。

    **每一場一個新的錄影檔。** :class:`~mia.groundtruth.dump.DumpWriter` 是用
    append 模式開檔的,同一個檔案接兩場的話,從檔頭讀的封包執行緒會先把上一場
    整個重播一遍,然後拿著一個已經結束的牌局給建議 —— 而那**不會報錯**。

    路徑是**懶決定**的:第一次有人問(封包執行緒建構,或按下開始遊戲)才配一個
    時間戳目錄,配了就固定到這一場結束。所以「先打開 AI 建議、再按開始遊戲」
    也成立 —— :class:`~mia.live.packets.DumpTail` 本來就會等檔案出現。

    Args:
        build_command: 給一個錄影檔路徑,回傳要執行的命令列。通常是
            :func:`capture_command` 綁上使用者的旗標。
        root: 錄影檔要放在哪個目錄底下,每場一個子目錄。
    """

    def __init__(
        self,
        build_command: Callable[[Path], list[str]],
        *,
        root: Path | str,
        name: str = "封包擷取",
    ) -> None:
        self._build = build_command
        self.root = Path(root)
        self.status = WorkerStatus(name)
        self._dump: Path | None = None
        self._process: CaptureProcess | None = None
        self._control: ControlFile | None = None
        self._used: set[Path] = set()

    @property
    def dump(self) -> Path:
        """這一場要寫到哪。第一次問的時候才決定,之後固定到換場為止。"""
        if self._dump is None:
            self._dump = self._allocate()
        return self._dump

    @property
    def control(self) -> ControlFile:
        """跟這一場的擷取子程序講話的通道。

        放在錄影檔旁邊,所以**一場一個** —— 上一場的指令不會殘留到下一場,
        而且事後翻錄影目錄就看得到主程式當時送了什麼。
        """
        dump = self.dump
        if self._control is None or self._control.path.parent != dump.parent:
            self._control = ControlFile(dump.parent / "control.json")
        return self._control

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.status.alive

    def launch(self) -> bool:
        """開一場。回傳有沒有真的開起來 —— 已經在跑的話什麼都不做。

        **已經在跑就不動它**,而不是先關再開:關掉瀏覽器等於把那一場丟掉,
        而使用者按這個按鈕的意思從來不是「把現在這場砍了」。
        """
        if self.running:
            return False
        if self._process is not None:
            # 上一場已經結束了 —— 換一個新的錄影檔,理由見類別的 docstring
            self._dump = None
        dump = self.dump
        logger.info("封包錄影會寫到 {}", dump)
        self._process = CaptureProcess(self._build(dump), status=self.status)
        self._process.start()
        return True

    def poll(self) -> None:
        if self._process is not None:
            self._process.poll()

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._process is not None:
            self._process.stop(timeout=timeout)

    def _allocate(self) -> Path:
        """配一個還沒被用過的錄影檔路徑。

        時間戳只到秒 —— 連按兩次「開始遊戲」會撞在同一秒,而撞到就等於兩場寫進
        同一個檔案,正是這個類別要避免的事。所以撞了就往後找。

        自己記得配過哪些,不能只看目錄存不存在:目錄是**擷取子程序**收到第一個
        frame 時才建的,在那之前上一場的名字看起來還是空的。反過來也要看磁碟
        —— 上一次執行留下來的目錄不該被重用。

        刻意**不在這裡建目錄**:光是問路徑(打開 AI 建議就會問)不該在
        ``data/live/`` 留下一個空資料夾。真正要建的是 ``DumpWriter``,它本來
        就會建。
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        parent = self.root / stamp
        suffix = 2
        while parent in self._used or parent.exists():
            parent = self.root / f"{stamp}-{suffix}"
            suffix += 1
        self._used.add(parent)
        return parent / "ws.jsonl"

    def __repr__(self) -> str:
        return f"<CaptureLauncher running={self.running} dump={self._dump}>"
