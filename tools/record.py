#!/usr/bin/env python3
"""錄製雀魂對局畫面,產生離線開發用的資料集。

用法::

    # 錄 60 秒(自動找雀魂視窗)
    .venv/bin/python tools/record.py --duration 60

    # 一直錄到 Ctrl-C
    .venv/bin/python tools/record.py

    # 指定視窗、加註記
    .venv/bin/python tools/record.py --window "#4914" --notes "王座間 東風戰 測試皮膚"

    # 檢視已錄製的 session
    .venv/bin/python tools/record.py --inspect data/recordings/20260726-190312

    # 把 session 的關鍵幀輸出成圖檔(給人工標註用)
    .venv/bin/python tools/record.py --export data/recordings/20260726-190312 --export-dir out/
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from mia.calibration import StableCalibrator
from mia.capture import CaptureError, WindowInfo, create_backend
from mia.capture.base import CaptureBackend
from mia.config.loader import load_config
from mia.config.models import AppConfig
from mia.recorder import SessionReader, SessionWriter, record_session
from mia.utils.logging import setup_logging
from mia.utils.paths import DATA_DIR

DEFAULT_ROOT = DATA_DIR / "recordings"


def _resolve_window(backend: CaptureBackend, spec: str | None, config: AppConfig) -> WindowInfo:
    if spec is None:
        w = config.capture.window
        return backend.find_window(
            w.title_patterns,
            w.owner_patterns,
            w.host_patterns,
            min_width=w.min_width,
            min_height=w.min_height,
        )
    if spec.startswith("#"):
        handle = int(spec[1:])
        for window in backend.list_windows():
            if window.handle == handle:
                return window
        raise SystemExit(f"找不到 handle 為 {handle} 的視窗")
    matched = [w for w in backend.list_windows() if w.matches([spec])]
    if not matched:
        raise SystemExit(f"沒有視窗符合 {spec!r}")
    return max(matched, key=lambda w: w.bounds.area)


def cmd_inspect(path: Path) -> int:
    reader = SessionReader(path)
    m = reader.manifest
    print(f"session   : {m.session_id}")
    print(f"開始時間  : {m.started_at}")
    print(f"視窗      : {m.window_owner} — {m.window_title}")
    print(f"影像尺寸  : {m.image_size[0]}x{m.image_size[1]}  格式 {m.image_format}")
    print(f"牌桌矩形  : {reader.table_rect}")
    print(f"時長      : {m.duration:.1f} 秒")
    print(f"幀數      : {m.frame_count}  (實際存檔 {m.stored_frames} 張,"
          f"{100 * m.stored_frames / m.frame_count if m.frame_count else 0:.0f}%)")
    if m.notes:
        print(f"註記      : {m.notes}")

    events = list(reader.events())
    print(f"GT 事件   : {len(events)} 筆")

    total = sum(p.stat().st_size for p in (path / "frames").glob("*") if p.is_file())
    print(f"磁碟用量  : {total / 1e6:.1f} MB")
    if m.stored_frames:
        print(f"            平均每張 {total / m.stored_frames / 1e6:.2f} MB")
    return 0


def cmd_export(path: Path, out_dir: Path, limit: int | None) -> int:
    reader = SessionReader(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in reader.frames(changed_only=True):
        if limit is not None and count >= limit:
            break
        cv2.imwrite(str(out_dir / f"{count:05d}_{frame.captured_at:08.2f}s.png"), frame.image)
        count += 1
    print(f"已輸出 {count} 張關鍵幀到 {out_dir}")
    return 0


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:  # noqa: ARG001
    """把 SIGTERM 轉成 KeyboardInterrupt。

    record_session() 只認得 KeyboardInterrupt 才會結束迴圈、讓 ``with writer:``
    正常收尾寫出 manifest.json。背景執行(nohup、systemd,或被其他工具用非
    SIGINT 的方式終止)送的通常是 SIGTERM —— 沒有這個轉換,Python 會直接
    終止行程,已寫入的畫面檔會變成沒有索引的孤兒資料,整個 session 報廢。
    """
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="錄製雀魂對局畫面")
    parser.add_argument("--duration", type=float, metavar="SEC", help="錄製秒數;省略則錄到 Ctrl-C")
    parser.add_argument("--fps", type=float, help="目標幀率,預設取自設定檔")
    parser.add_argument("--window", metavar="SPEC", help="標題/程式名子字串,或 #視窗ID")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="session 存放的上層目錄")
    parser.add_argument("--name", help="session 目錄名,預設用時間戳")
    parser.add_argument("--notes", default="", help="寫進 manifest 的註記")
    parser.add_argument(
        "--format", default="jpg", choices=["jpg", "png"],
        help="jpg 預設(1MB/張、9ms 寫入,對模板比對的影響在小數第四位);"
             "png 無損但 4MB/張、127ms 寫入,10 fps 下會掉幀",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 品質,不建議低於 90")
    parser.add_argument(
        "--change-threshold", type=float, default=1.5,
        help="縮圖平均絕對差超過此值才存檔。設 0 表示每幀都存",
    )
    parser.add_argument("--inspect", type=Path, metavar="DIR", help="檢視 session 後結束")
    parser.add_argument("--export", type=Path, metavar="DIR", help="輸出關鍵幀後結束")
    parser.add_argument("--export-dir", type=Path, default=Path("out/frames"))
    parser.add_argument("--export-limit", type=int, help="最多輸出幾張")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    if args.inspect:
        return cmd_inspect(args.inspect)
    if args.export:
        return cmd_export(args.export, args.export_dir, args.export_limit)

    config = load_config()
    try:
        with create_backend(config.capture) as backend:
            window = _resolve_window(backend, args.window, config)
            writer = SessionWriter(
                args.root,
                session_id=args.name,
                image_format=args.format,
                jpeg_quality=args.jpeg_quality,
                change_threshold=args.change_threshold,
                backend=backend.name,
                notes=args.notes,
            )
            with writer:
                stats = record_session(
                    backend,
                    window,
                    writer,
                    target_fps=args.fps or config.capture.target_fps,
                    duration=args.duration,
                    calibrator=StableCalibrator(config.calibration),
                )
    except CaptureError as exc:
        print(f"\n擷取失敗:\n{exc}", file=sys.stderr)
        return 1

    print()
    print(f"目錄      : {writer.path}")
    print(f"幀數      : {stats.frames}  (存檔 {stats.stored},{stats.store_ratio:.0%})")
    print(f"時長      : {stats.duration:.1f} 秒  實際 {stats.actual_fps:.1f} fps")
    if stats.errors:
        print(f"擷取失敗  : {stats.errors} 次")
    if stats.dropped:
        print(f"跟不上幀率: {stats.dropped} 次 — 可考慮調低 --fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
