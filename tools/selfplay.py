#!/usr/bin/env python3
"""自己打一場給自己看:瀏覽器打電腦,同時跑畫面辨識,打完比對錯在哪。**不是 app 的功能。**

為什麼要有這個
--------------
畫面辨識的素材一直是「先錄下來、之後離線重播」。那量不到兩件事:

* **真的跑起來有多晚。** 重播時認副露牌面是等它認完才往下走,牌河、手牌、引擎都是一條線,
  沒有互搶核心、沒有動畫追著跑。
* **當下給了什麼。** 事件流會改寫,局末對不代表當下對。

這裡讓 app 裡那幾條執行緒原封不動跑在真的對局上,每個版本的事件流都記下時間,封包同時錄。
兩邊用同一個牆上時鐘,打完直接算得出每張牌晚多少、哪一張讀錯,再看同一個引擎的答案差多少。

一條指令做完
------------
::

    .venv/bin/python tools/selfplay.py

登入過就不用碰:開瀏覽器、開一間四人半莊、加三台電腦、打完、比對。要看到問題的畫面,
報告裡的問題依序存在 ``issues/``。``--games N`` 連打 N 場,每場一個 session。

修完怎麼驗
----------
::

    .venv/bin/python tools/selfplay.py --replay data/recordings/selfplay-x --kyoku 3

把錄好的畫面**照原本的時間**播給同一組執行緒,事件流寫到 ``vision-replay.jsonl`` 再比一次。
不是一次跑完的離線重算:延遲與當下讀到什麼都跟著時間走,加速就不是同一件事了。所以一場
要跑一場的時間,``--kyoku`` 只播某一局,不給就全部。

規則
----
* 自動打牌只在全電腦的桌上動手,見 :func:`autoplay.cpu_table`。這支工具與
  ``tools/autoplay.py`` 一樣留在 ``tools/``,不進發佈版。
* 打牌的 Mortal 與給建議的 Mortal 是兩個:前者看封包,後者只看畫面。

產出(``data/recordings/selfplay-<時間>/``):

* ``ws.jsonl``:封包,答案從這裡算。
* ``frames/``、``manifest.json``:畫面,與 ``tools/record.py`` 同一種格式,之後可以重播。
* ``vision.jsonl``:每個版本的事件流、手牌動作、排序步驟、每次建議、每句狀態,都帶牆上時鐘。
* ``report.txt``:比對結果。
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from capture import GameWatcher
from mia.calibration.canvas import Canvas, CanvasChoice
from mia.calibration.table import Calibration
from mia.capture.base import CaptureBackend, Frame, WindowInfo
from mia.capture.factory import create_backend
from mia.config.loader import load_config
from mia.config.models import AppConfig
from mia.engine.mortal import mortal_engine
from mia.eval.streamdiff import (
    VISION_LOG,
    Issue,
    TruthKyoku,
    agreement,
    compare,
    load_run,
    load_truth,
    summary,
)
from mia.live.advice import VisionAdviceWorker
from mia.live.bus import Advices, Update, UpdateBus, WorkerStatus
from mia.live.table import TableLink
from mia.live.vision import VisionWorker
from mia.recorder import SessionReader, SessionWriter
from mia.tracker import Stream
from mia.utils.geometry import Rect
from mia.utils.logging import logger, setup_logging
from mia.utils.paths import DATA_DIR

#: 打牌與給建議用的權重。
DEFAULT_WEIGHTS = "models/mortal_298k.pth"

#: 與 ``tools/gt.py`` 的自動打牌同一個畫布。
CANVAS = "1280x720"

#: 一場最久幾分鐘。半莊實測二十幾分鐘,卡住的話不要一直掛著。
_MAX_MINUTES = 90.0

#: 封包說打完之後再錄幾秒,讓最後一局的畫面演完。
_TAIL = 8.0

#: 多久看一次封包有沒有打完。
_POLL = 2.0

#: 開打之後這麼久沒有新的牌局事件,就當作牌局已經不在了。
#:
#: 對局被中斷時不會有 ``end_game``,瀏覽器回到大廳之後大廳的心跳還在,只看封包有沒有在動是
#: 看不出來的。一手最多 25 秒,局間結算也才幾十秒。
_IDLE = 180.0

#: 寫畫面的佇列最多幾張。寫不完就丟,不能拖慢辨識,那是這裡要量的東西。
_FRAME_QUEUE = 30

#: 判斷全黑:每隔幾個像素取一點,平均亮度低於多少。
_BLACK_STEP = 16
_BLACK_LEVEL = 2.0

#: 全黑多久要出聲。開局轉場也會黑一下。
_BLACK_WARN = 20.0

#: 重播的事件流寫到這裡,不蓋掉實打那一份。
REPLAY_LOG = "vision-replay.jsonl"

#: 只存畫面有變的幀,這是變化的門檻(縮圖的平均灰階差)。
#:
#: ``tools/record.py`` 的 1.5 是為了省磁碟,而一張牌落到河裡在縮圖上幾乎看不出來:那一幀不存
#: 的話,重播看到的畫面就與實打不同。這裡要拿來重播,所以取低。
_CHANGE_THRESHOLD = 0.3

#: 重播單獨一局時,從開局前幾秒播起。校正要蒐集好幾幀才鎖定。
_REPLAY_LEAD = 5.0

#: 問題的畫面取封包之後幾秒。封包比畫面早,動畫演完才看得到。
_ISSUE_DELAY = 2.0

#: Playwright 開的瀏覽器。
_BROWSER_OWNERS = ("Chrome for Testing", "Chromium")

#: 等瀏覽器視窗出現的上限,秒。
_WINDOW_TIMEOUT = 60.0


class RunLog:
    """``vision.jsonl`` 的寫入端。好幾條執行緒會同時寫。"""

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._last: tuple[int, tuple[Any, ...], str] | None = None

    def now(self) -> float:
        return time.time()

    def write(self, **row: Any) -> None:
        line = json.dumps({"wall": self.now(), **row}, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def stream(self, kyoku: int, stream: Stream, **trace: Any) -> None:
        """記一個版本。``trace`` 是給事後查的:實打才出現的錯,重播不一定重現得了。"""
        note = stream.doubt or ("等手牌" if stream.stalled else "")
        current = (kyoku, stream.events, note)
        if current == self._last:
            return
        self._last = current
        self.write(
            kind="stream",
            kyoku=kyoku,
            note=note,
            events=[e.to_dict() for e in stream.events],
            **trace,
        )

    def close(self) -> None:
        with self._lock:
            self._file.close()


class LoggedStatus(WorkerStatus):
    """狀態列的每一句話也記下來:「引擎掉隊」只會出現在這裡。"""

    def __init__(self, name: str, log: RunLog) -> None:
        super().__init__(name)
        self._log = log

    def say(self, message: str) -> None:
        if message != self.read():
            self._log.write(kind="status", name=self.name, message=message)
        super().say(message)


class TapBus(UpdateBus):
    """郵箱只留最新的一封,所以每次建議要在投進去的時候就記下來。"""

    def __init__(self, log: RunLog) -> None:
        super().__init__()
        self._log = log

    def post(self, update: Update) -> None:
        if isinstance(update, Advices):
            for advice in update.advices:
                if advice.is_decision:
                    action = advice.action.to_dict() if advice.action else None
                    self._log.write(kind="advice", engine=advice.engine, action=action)
        super().post(update)


class TappedAdvice(VisionAdviceWorker):
    """app 裡的畫面建議,多記每個版本的事件流。

    記在 ``_offer`` 而不是 ``_feed``:後者只拿得到最新的那份,引擎忙的時候會跳過版本,
    而被跳過的版本也是當下給出去的局面。
    """

    def __init__(self, *args: Any, log: RunLog, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._run_log = log

    def _offer(self, kyoku: int, stream: Stream) -> None:
        table = self._table
        self._run_log.stream(
            kyoku,
            stream,
            moves=[(move.move.value, *move.tiles) for move in table.own_moves],
            steps=[
                (step.action.value, step.actor, step.pai or step.guess, step.serial)
                for step in table.steps
            ],
        )
        super()._offer(kyoku, stream)


def is_black(image: Any) -> bool:
    """整張畫面是黑的。隔幾個像素取樣就夠,這要每一幀算。"""
    return float(image[::_BLACK_STEP, ::_BLACK_STEP].mean()) < _BLACK_LEVEL


class FrameRecorder(threading.Thread):
    """把畫面寫成 session。寫檔不能卡住辨識,所以另開一條,來不及就丟。

    **順便看畫面是不是黑的。** 視窗沒被畫出來時擷取拿到的是全黑的圖,而且不會報錯:實測一場
    前 23 分鐘全黑,打牌照常,辨識一局都沒看到,打完才從報告知道。
    """

    def __init__(self, writer: SessionWriter) -> None:
        super().__init__(name="frame-recorder", daemon=True)
        self._writer = writer
        self._frames: queue.Queue[Frame | None] = queue.Queue(_FRAME_QUEUE)
        self.table_rect: Rect | None = None
        self.dropped = 0
        self.stored = 0
        #: 從什麼時候開始連續全黑。``None`` 表示現在看得到。
        self.black_since: float | None = None
        #: 全黑的總秒數。
        self.black = 0.0
        self._seen = 0.0

    def offer(self, frame: Frame) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def finish(self) -> None:
        self._frames.put(None)
        self.join(timeout=30)

    def run(self) -> None:
        while (frame := self._frames.get()) is not None:
            self._watch_black(frame)
            self.stored += self._writer.add_frame(frame, table_rect=self.table_rect).changed

    def _watch_black(self, frame: Frame) -> None:
        now = frame.captured_at
        if not is_black(frame.image):
            self.black_since = None
            return
        if self.black_since is None:
            self.black_since = self._seen = now
        self.black += now - self._seen
        self._seen = now


class TeeBackend:
    """擷取後端多一個出口:每一幀也交給 :class:`FrameRecorder`。"""

    def __init__(self, inner: CaptureBackend, recorder: FrameRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def capture(self, window: WindowInfo) -> Frame:
        frame = self._inner.capture(window)
        self._recorder.offer(frame)
        return frame


class RecordingLink(TableLink):
    """牌桌矩形要寫進 manifest,它只在校正那一邊才有。"""

    def __init__(self, recorder: FrameRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def offer_frame(self, image: Any, calibration: Calibration) -> None:
        self._recorder.table_rect = calibration.table_rect
        super().offer_frame(image, calibration)


class ReplayBackend:
    """把錄好的畫面照原本的時間播回去,對 :class:`VisionWorker` 來說就是一個擷取後端。"""

    name = "replay"

    def __init__(self, session: SessionReader, start: float, end: float) -> None:
        self._session = session
        self._records = session.manifest.frames
        self._stamps = [record.timestamp for record in self._records]
        self._first = session.manifest.first_frame_wall or 0.0
        self._start = start - self._first
        self._end = end - self._first
        self._began: float | None = None
        self._image: tuple[str, Any] | None = None

    @property
    def wall(self) -> float:
        """播到錄影裡的哪個時刻,牆上時鐘。"""
        return self._first + self._position()

    @property
    def finished(self) -> bool:
        return self._position() >= self._end

    def _position(self) -> float:
        if self._began is None:
            return self._start
        return self._start + time.monotonic() - self._began

    def capture(self, window: WindowInfo) -> Frame:
        if self._began is None:
            self._began = time.monotonic()
        at = bisect.bisect_right(self._stamps, self._position()) - 1
        record = self._records[max(at, 0)]
        if self._image is None or self._image[0] != record.image:
            self._image = (record.image, self._session.load_image(record))
        return Frame(self._image[1], window, record.scale)

    def close(self) -> None:
        """沒有東西要收。"""


class ReplayLog(RunLog):
    """時間要記錄影裡的時刻,才對得上封包。"""

    def __init__(self, path: Path, backend: ReplayBackend) -> None:
        super().__init__(path)
        self._backend = backend

    def now(self) -> float:
        return self._backend.wall


def _game_window(backend: CaptureBackend, config: AppConfig) -> WindowInfo | None:
    """等 ``tools/gt.py`` 開的那個瀏覽器視窗。

    **一定要指定,不能讓辨識自己找。** 辨識比瀏覽器先起來的話,找到的會是別的視窗:實測鎖到
    Safari 開著這個專案的 GitHub 頁面,標題也有「雀魂」,整場沒看到牌桌。
    """
    deadline = time.monotonic() + _WINDOW_TIMEOUT
    while time.monotonic() < deadline:
        for window in backend.list_windows_everywhere():
            if window.matches(_BROWSER_OWNERS, fields="owner") and window.matches(
                config.capture.window.title_patterns, fields="title"
            ):
                return window
        time.sleep(0.5)
    return None


def play(args: argparse.Namespace, config: AppConfig) -> Path | None:
    """打一場。回傳 session 目錄,開房間失敗回 ``None``。"""
    name = args.name or time.strftime("selfplay-%Y%m%d-%H%M%S")
    backend = create_backend(config.capture)
    writer = SessionWriter(
        args.root,
        session_id=name,
        backend=backend.name,
        change_threshold=_CHANGE_THRESHOLD,
        notes="selfplay.py:對電腦的四人半莊",
    )
    session = writer.path
    log = RunLog(session / VISION_LOG)
    recorder = FrameRecorder(writer)
    bus = TapBus(log)
    link = RecordingLink(recorder)
    advice = TappedAdvice(
        bus,
        config=config.roi,
        engines=[mortal_engine(args.mortal, seat=0, name="mortal")],
        link=link,
        log=log,
    )
    advice.status = LoggedStatus("AI 建議", log)

    dump = session / "ws.jsonl"
    command = [
        sys.executable, str(ROOT / "tools/gt.py"), "cdp", "--out", str(dump),
        "--autoplay", "mortal", "--mortal", args.mortal, "--cpu-room",
        *(("--overtime",) if args.overtime else ()),
    ]
    print(f"session:{session}", flush=True)
    with (session / "gt.out").open("w", encoding="utf-8") as output:
        packets = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    window = _game_window(backend, config)
    if window is None:
        print("等不到瀏覽器視窗", file=sys.stderr)
        packets.terminate()
        packets.wait(timeout=20)
        writer.close()
        log.close()
        return None
    logger.info("擷取目標:{}", window)
    vision = VisionWorker(
        bus,
        config=config,
        backend=TeeBackend(backend, recorder),  # type: ignore[arg-type]
        window=window,
        canvas=CanvasChoice(Canvas.parse(CANVAS)),
        sink=link,
    )
    vision.status = LoggedStatus("畫面", log)
    recorder.start()
    advice.start()
    vision.start()

    watcher = GameWatcher(dump)
    deadline = time.monotonic() + args.max_minutes * 60
    finished = False
    seen, moved = 0, time.monotonic()
    warned = False
    try:
        while time.monotonic() < deadline:
            if packets.poll() is not None:
                print(f"瀏覽器那邊先結束了(代碼 {packets.returncode})", flush=True)
                break
            if watcher.poll():
                print(f"打完了,共 {watcher.kyoku} 局", flush=True)
                finished = True
                time.sleep(_TAIL)
                break
            since = recorder.black_since
            if since is not None and time.monotonic() - since > _BLACK_WARN and not warned:
                print(
                    f"⚠ 畫面全黑超過 {_BLACK_WARN:.0f} 秒,辨識看不到牌桌。"
                    "瀏覽器視窗要看得到:螢幕不要鎖定,也不要留在別的桌面。",
                    flush=True,
                )
                warned = True
            if watcher.events != seen:
                seen, moved = watcher.events, time.monotonic()
            elif seen and time.monotonic() - moved > _IDLE:
                print(f"{_IDLE:.0f} 秒沒有牌局事件,對局大概被中斷了,收工", flush=True)
                break
            time.sleep(_POLL)
        else:
            print(f"超過 {args.max_minutes:.0f} 分鐘,收工", flush=True)
    except KeyboardInterrupt:
        print("收到中斷,收工", flush=True)
    finally:
        vision.stop()
        advice.stop()
        vision.join(timeout=5)
        advice.join(timeout=5)
        recorder.finish()
        writer.close()
        log.close()
        if packets.poll() is None:
            # 用 SIGTERM:背景執行時子行程會繼承「忽略 SIGINT」,實測送了沒反應。
            packets.terminate()
            try:
                packets.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - 保險
                packets.kill()

    print(f"畫面存了 {recorder.stored} 張,來不及寫丟掉 {recorder.dropped} 張", flush=True)
    if recorder.black:
        print(f"⚠ 有 {recorder.black:.0f} 秒畫面全黑,那段時間的比對沒有意義", flush=True)
    if not finished and watcher.kyoku == 0:
        print(f"沒有打到牌局,看 {session / 'gt.out'}", file=sys.stderr)
        return None
    return session


def replay(args: argparse.Namespace, config: AppConfig) -> None:
    """把一個 session 重播一次,見模組說明的「修完怎麼驗」。"""
    session = SessionReader(args.replay)
    truth = load_truth(args.replay / "ws.jsonl")
    start, end = _span(truth, args.kyoku, session)
    backend = ReplayBackend(session, start, end)
    log = ReplayLog(args.replay / REPLAY_LOG, backend)
    bus = TapBus(log)
    link = TableLink()
    advice = TappedAdvice(
        bus,
        config=config.roi,
        engines=[mortal_engine(args.mortal, seat=0, name="mortal")],
        link=link,
        log=log,
    )
    advice.status = LoggedStatus("AI 建議", log)
    vision = VisionWorker(
        bus,
        config=config,
        backend=backend,  # type: ignore[arg-type]
        window=session.manifest.window,
        canvas=CanvasChoice(Canvas.parse(CANVAS)),
        sink=link,
    )
    vision.status = LoggedStatus("畫面", log)
    advice.start()
    while not link.attached:
        time.sleep(0.05)
    print(f"重播 {end - start:.0f} 秒", flush=True)
    vision.start()
    try:
        while not backend.finished:
            time.sleep(0.5)
    finally:
        vision.stop()
        advice.stop()
        vision.join(timeout=5)
        advice.join(timeout=5)
        log.close()


def _span(
    truth: Sequence[TruthKyoku], kyoku: int | None, session: SessionReader
) -> tuple[float, float]:
    """要播的時間範圍。``kyoku`` 從 1 算,前面多留幾秒讓校正鎖定。"""
    first = session.manifest.first_frame_wall or 0.0
    if kyoku is None:
        return first, first + session.manifest.duration
    if not 1 <= kyoku <= len(truth):
        raise SystemExit(f"這一場只有 {len(truth)} 局")
    start = truth[kyoku - 1].start - _REPLAY_LEAD
    if kyoku < len(truth):
        return start, truth[kyoku].start
    return start, first + session.manifest.duration


def _keep_awake() -> None:
    """打牌期間不讓螢幕睡著,睡著的話擷取只拿得到全黑的圖。

    ``-w`` 跟著這個行程走,結束就失效。已經鎖定的螢幕它救不了。
    """
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])


def _shots(session: Path, issues: Sequence[Issue]) -> None:
    """每個問題存一張封包之後的畫面,檔名是問題的順序。"""
    reader = SessionReader(session)
    records = reader.manifest.frames
    stamps = [record.timestamp for record in records]
    origin = reader.manifest.first_frame_wall
    folder = session / "issues"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    for index, issue in enumerate(issues):
        if issue.wall is None or not records or origin is None:
            continue
        at = bisect.bisect_left(stamps, issue.wall + _ISSUE_DELAY - origin)
        record = records[min(at, len(records) - 1)]
        source = session / "frames" / record.image
        if source.exists():
            shutil.copy(source, folder / f"{index:03d}-{issue.kind}{source.suffix}")


def report(
    session: Path, weights: str | None, *, log: str = VISION_LOG, kyoku: int | None = None
) -> str:
    """比對一個 session,寫報告,回傳報告。``kyoku`` 從 1 算,只比那一局。"""
    truth = load_truth(session / "ws.jsonl")
    if kyoku is not None:
        truth = truth[kyoku - 1 : kyoku]
    run = load_run(session / log)
    results = compare(truth, run.versions)

    agreed = None
    if weights:

        def make() -> Any:
            engine = mortal_engine(weights, seat=0, name="mortal")
            engine.start()
            return engine

        agreed = agreement(truth, run.versions, make)

    text = summary(results, agreed, run, truth)
    if log != VISION_LOG:
        (session / "report-replay.txt").write_text(text + "\n", encoding="utf-8")
        return text
    (session / "report.txt").write_text(text + "\n", encoding="utf-8")
    _shots(session, [issue for result in results for issue in result.issues])
    return text


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:  # noqa: ARG001
    """SIGTERM 也走乾淨收尾,否則 manifest 寫不出來,畫面檔就成了沒有索引的孤兒。"""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--games", type=int, default=1, help="連打幾場")
    parser.add_argument("--name", help="session 名稱,預設用時間。--games 大於 1 時不能給")
    parser.add_argument("--root", type=Path, default=DATA_DIR / "recordings")
    parser.add_argument("--mortal", default=DEFAULT_WEIGHTS, help="打牌與給建議用的權重")
    parser.add_argument(
        "--overtime", action="store_true", help="1 位必要點數設最大、關掉擊飛,一定打進西入"
    )
    parser.add_argument("--max-minutes", type=float, default=_MAX_MINUTES)
    parser.add_argument("--report", type=Path, metavar="DIR", help="不打,只重新比對這個 session")
    parser.add_argument("--replay", type=Path, metavar="DIR", help="把這個 session 重播一次再比對")
    parser.add_argument("--kyoku", type=int, metavar="N", help="--replay 只播第 N 局,從 1 算")
    parser.add_argument("--no-engine", action="store_true", help="比對時不跑引擎,快很多")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    weights = None if args.no_engine else args.mortal
    if args.report:
        print(report(args.report, weights))
        return 0
    if args.replay:
        replay(args, load_config())
        print(report(args.replay, weights, log=REPLAY_LOG, kyoku=args.kyoku))
        return 0
    if args.name and args.games > 1:
        parser.error("--name 只能配一場")

    config = load_config()
    _keep_awake()
    for game in range(args.games):
        print(f"第 {game + 1} 場,共 {args.games} 場", flush=True)
        session = play(args, config)
        if session is None:
            return 1
        print(report(session, weights), flush=True)
        logger.info("報告在 {}", session / "report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
