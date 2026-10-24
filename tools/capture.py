#!/usr/bin/env python3
"""一個指令錄完一局:畫面與封包一起開始、一起結束,當場驗素材好不好。

為什麼要有這個
--------------
產生「可以量 CV 準確率」的素材需要**兩份檔案時間對得上**:畫面錄影
(``tools/record.py``)與封包錄影(``tools/gt.py cdp``)。原本那是兩個指令、
兩個終端機、兩次手動停止,而少做一步素材就報廢。

實測報廢過一次:``data/recordings/20260801-110639`` 的畫面只錄了 848 秒,
封包錄了 1433 秒。那一局有 7 次立直分散在四家,而畫面只涵蓋到**一家**的 ——
量「立直前 vs 立直後」的時候只剩一家可用,幅度就釘不下來。

而那件事是**幾個月後**才被發現的。所以這支工具做兩件事:

1. 兩邊一起開、一起停。結束的時機由**封包**決定(看到 ``end_game``),
   不是由人決定 —— 人會停早。
2. 停完**當場**跑一次對齊檢查,把「這份素材能不能用」當場講出來。
   錄的時候不知道,就會像上面那次一樣浪費掉整局。

用法::

    .venv/bin/python tools/capture.py

    # 不想等整局(例如只想錄幾分鐘試試)
    .venv/bin/python tools/capture.py --max-minutes 5

按 Ctrl-C 也會乾淨地收掉兩邊。
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mia.calibration.stable import StableCalibrator
from mia.capture.base import CaptureBackend, CaptureError, WindowInfo
from mia.capture.factory import create_backend
from mia.config.loader import load_config
from mia.config.models import AppConfig
from mia.groundtruth.dump import iter_frames
from mia.groundtruth.stream import MjaiDecoder
from mia.recorder import SessionWriter, record_session
from mia.utils.logging import logger, setup_logging
from mia.utils.paths import DATA_DIR

#: 多久看一次封包檔有沒有「對局結束」。
#:
#: 5 秒不是為了反應快 —— 錄影本來就該錄到結束畫面演完。它只是不要讓
#: 「停不下來」變成另一個失敗模式。
_POLL = 5.0

#: 等瀏覽器開起來、使用者登入、找得到遊戲視窗的上限。
#:
#: 給得很寬(10 分鐘):登入可能要掃 QR code、可能要挑房間。等不到就放棄,
#: 但**不要**在這裡失敗得太快 —— 使用者剛按下指令就被罵是最惱人的。
_WINDOW_TIMEOUT = 600.0


class GameWatcher:
    """盯著封包檔,看到「對局結束」就說該停了。

    只讀**新增的行**。每次重讀整個檔會隨著對局變長越來越慢,而它跑在
    錄影的同一條路上 —— 那會變成掉幀的原因。
    """

    def __init__(self, dump: Path) -> None:
        self.dump = dump
        self._decoder = MjaiDecoder()
        self._seen = 0
        self.ended = False
        self.kyoku = 0

    def poll(self) -> bool:
        """回傳「對局已經結束」。檔案還沒出現也算沒結束。"""
        if self.ended or not self.dump.exists():
            return self.ended
        for index, frame in enumerate(iter_frames(self.dump)):
            if index < self._seen:
                continue
            self._seen = index + 1
            decoded = self._decoder.feed(frame)
            if decoded is None:
                continue
            for event in decoded.events:
                if event.TYPE == "start_kyoku":
                    self.kyoku += 1
                elif event.TYPE == "end_game":
                    self.ended = True
        return self.ended


def _find_window(backend: CaptureBackend, config: AppConfig) -> WindowInfo | None:
    match = config.capture.window
    try:
        return backend.find_window(
            match.title_patterns,
            match.owner_patterns,
            match.host_patterns,
            min_width=match.min_width,
            min_height=match.min_height,
        )
    except CaptureError:
        return None


def _wait_for_window(backend: CaptureBackend, config: AppConfig) -> WindowInfo | None:
    print("等遊戲視窗出現…(登入、進房都可以慢慢來,按 Ctrl-C 放棄)", flush=True)
    deadline = time.monotonic() + _WINDOW_TIMEOUT
    while time.monotonic() < deadline:
        window = _find_window(backend, config)
        if window is not None:
            print(f"找到視窗:{window.title}({window.bounds.width}x{window.bounds.height})")
            return window
        time.sleep(1.0)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DATA_DIR / "recordings")
    parser.add_argument("--name", help="session 名稱,預設用時間")
    parser.add_argument("--fps", type=float, help="畫面錄影的目標 fps")
    parser.add_argument(
        "--max-minutes", type=float, metavar="MIN",
        help="最多錄幾分鐘。**省略才是正確用法** —— 由封包決定何時停,"
             "人訂的時限正是上次把素材錄壞的原因",
    )
    parser.add_argument(
        "--autoplay", choices=("tsumogiri", "mortal"), metavar="MODE",
        help="讓程式替你打完整局 —— **只在對電腦的房間用**。見 tools/autoplay.py",
    )
    parser.add_argument("--mortal", metavar="WEIGHTS", help="--autoplay mortal 用的權重檔")
    parser.add_argument(
        "--autoplay-dry-run", action="store_true",
        help="自動打牌只印「會點哪裡」不真的點。按鈕座標還沒量過,第一次請用這個",
    )
    parser.add_argument("--notes", default="capture.py:畫面與封包同時錄")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    config = load_config()
    try:
        backend = create_backend(config.capture)
    except CaptureError as exc:
        print(f"擷取後端起不來:\n{exc}", file=sys.stderr)
        return 1

    with backend:
        writer = SessionWriter(
            args.root,
            session_id=args.name,
            backend=backend.name,
            notes=args.notes,
        )
        dump = writer.path / "ws.jsonl"
        writer.path.mkdir(parents=True, exist_ok=True)

        print(f"session:{writer.path}")
        print("開瀏覽器並開始錄封包…", flush=True)
        command = [sys.executable, str(ROOT / "tools/gt.py"), "cdp", "--out", str(dump)]
        if args.autoplay:
            command += ["--autoplay", args.autoplay]
            if args.mortal:
                command += ["--mortal", args.mortal]
            if args.autoplay_dry_run:
                command.append("--autoplay-dry-run")
        packets = subprocess.Popen(command, cwd=ROOT)

        try:
            window = _wait_for_window(backend, config)
            if window is None:
                print("等不到遊戲視窗,放棄。封包那邊也收掉。", file=sys.stderr)
                return 1

            watcher = GameWatcher(dump)
            limit = args.max_minutes * 60 if args.max_minutes else None
            started = time.monotonic()
            last = started

            def should_stop() -> bool:
                nonlocal last
                now = time.monotonic()
                if limit is not None and now - started >= limit:
                    print("\n到了 --max-minutes 的時限,停。", flush=True)
                    return True
                if now - last < _POLL:
                    return False
                last = now
                if watcher.poll():
                    print(f"\n封包說對局結束了(共 {watcher.kyoku} 局),停。", flush=True)
                    return True
                return False

            print("開始錄。打完整局就會自己停 —— 不用回來按。", flush=True)
            with writer:
                stats = record_session(
                    backend,
                    window,
                    writer,
                    target_fps=args.fps or config.capture.target_fps,
                    calibrator=StableCalibrator(config.calibration),
                    stop_check=should_stop,
                )
        except KeyboardInterrupt:
            print("\n收到 Ctrl-C,兩邊一起收。", flush=True)
            return 1
        finally:
            packets.send_signal(signal.SIGINT)
            try:
                packets.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - 保險
                logger.warning("封包錄製沒有在 20 秒內收掉,強制結束")
                packets.kill()

    print()
    print(f"目錄  : {writer.path}")
    print(f"畫面  : {stats.frames} 幀(存檔 {stats.stored}),{stats.duration:.0f} 秒")
    print(f"封包  : {dump}")
    print()
    print("素材好不好 —— 當場驗:")
    subprocess.run(
        [sys.executable, str(ROOT / "tools/eval_rivers.py"),
         str(writer.path), str(dump), "--dry-run"],
        cwd=ROOT,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
