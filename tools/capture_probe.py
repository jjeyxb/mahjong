#!/usr/bin/env python3
"""擷取層的開發用 CLI。

M1 的驗收工具,也是之後每次換機器 / 換平台時的第一件事。

用法::

    # 看得到哪些視窗、目前用哪個後端
    python tools/capture_probe.py --list
    python tools/capture_probe.py --list --all      # 不過濾系統浮層

    # 抓一張圖存檔(--window 可給子字串或 #視窗ID)
    python tools/capture_probe.py --window Chrome --out shot.png

    # 用設定檔裡的 pattern 找雀魂
    python tools/capture_probe.py --out shot.png

    # 測速
    python tools/capture_probe.py --window Chrome --bench 60

    # 跑牌桌校正並輸出標註圖
    python tools/capture_probe.py --window Chrome --calibrate --out calib.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from majsoul_copilot.calibration import StableCalibrator
from majsoul_copilot.calibration.debug import annotate_calibration
from majsoul_copilot.capture import (
    CaptureBackend,
    CaptureError,
    WindowInfo,
    available_backends,
    create_backend,
)
from majsoul_copilot.config.loader import load_config
from majsoul_copilot.config.models import AppConfig
from majsoul_copilot.utils.logging import setup_logging


def _resolve_window(
    backend: CaptureBackend, spec: str | None, config: AppConfig
) -> WindowInfo:
    """把命令列的 --window 參數解析成一個 WindowInfo。"""
    windows = backend.list_windows()

    if spec is None:
        win_cfg = config.capture.window
        return backend.find_window(
            win_cfg.title_patterns,
            win_cfg.owner_patterns,
            win_cfg.host_patterns,
            min_width=win_cfg.min_width,
            min_height=win_cfg.min_height,
        )

    if spec.startswith("#"):
        handle = int(spec[1:])
        for window in windows:
            if window.handle == handle:
                return window
        raise SystemExit(f"找不到 handle 為 {handle} 的視窗")

    matched = [w for w in windows if w.matches([spec])]
    if not matched:
        listing = "\n".join(f"    {w}" for w in windows)
        raise SystemExit(f"沒有視窗符合 {spec!r}。目前看得到:\n{listing}")
    return max(matched, key=lambda w: w.bounds.area)


def cmd_list(backend: CaptureBackend, *, include_all: bool) -> None:
    windows = backend.list_windows(include_all=include_all)
    print(f"後端: {backend.name}    可用後端: {', '.join(available_backends())}")
    print(f"視窗數: {len(windows)}\n")
    for window in windows:
        print(f"  {window}")
    if not windows:
        print("  (沒有視窗。macOS 上若標題全是空的,代表缺少螢幕錄製權限)")


def cmd_capture(backend: CaptureBackend, window: WindowInfo, out: Path | None) -> None:
    frame = backend.capture(window)
    print(f"視窗    : {window}")
    print(f"影像    : {frame.size} (像素)")
    print(f"邏輯尺寸: {window.bounds.size}")
    print(f"scale   : {frame.scale:.2f}  (pixel / logical)")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame.image)
        print(f"已存檔  : {out}")


def cmd_bench(backend: CaptureBackend, window: WindowInfo, count: int) -> None:
    backend.capture(window)  # warmup
    durations: list[float] = []
    for _ in range(count):
        start = time.perf_counter()
        frame = backend.capture(window)
        durations.append(time.perf_counter() - start)

    durations.sort()
    mean = sum(durations) / len(durations)
    print(f"後端 {backend.name} — {frame.size},{count} 幀")
    print(f"  平均   : {mean * 1000:6.2f} ms  ({1 / mean:5.1f} fps)")
    print(f"  中位數 : {durations[len(durations) // 2] * 1000:6.2f} ms")
    print(f"  最快   : {durations[0] * 1000:6.2f} ms")
    print(f"  p95    : {durations[int(len(durations) * 0.95) - 1] * 1000:6.2f} ms")
    print(f"  最慢   : {durations[-1] * 1000:6.2f} ms")


def cmd_calibrate(
    backend: CaptureBackend,
    window: WindowInfo,
    config: AppConfig,
    out: Path | None,
    *,
    canvas_out: Path | None = None,
) -> None:
    # 刻意取多幀:單幀校正是啟發式,會被動畫與立繪干擾。實測同一場錄影的
    # 216 幀產生了 21 種不同的 table_rect,其中 16% 明顯錯誤 —— 探測工具若
    # 只看一幀,回報的「可靠」就是抽籤抽到的結果,不是這台機器真正的狀況。
    stable = StableCalibrator(config.calibration)
    calibration = None
    frame = backend.capture(window)
    for _ in range(config.calibration.stabilize_frames * 4):
        frame = backend.capture(window)
        calibration = stable.feed(frame)
        if calibration is not None:
            break
        time.sleep(1.0 / config.capture.target_fps)

    if calibration is None:
        accepted, needed = stable.progress
        print(f"校正失敗  : 只蒐集到 {accepted}/{needed} 個可用的候選")
        print("  多數畫面都沒被判定成牌桌。請確認遊戲確實在畫面上(不是載入畫面或選單),")
        print("  或在 config/default.local.yaml 用 manual_table_rect 手動指定。")
        return

    print(f"影像      : {frame.size}")
    print(f"採用候選  : {stable.progress[0]} 幀")
    print(f"牌桌矩形  : {calibration.table_rect}")
    print(f"寬高比    : {calibration.aspect:.4f}  (預期 {config.calibration.aspect_ratio:.4f})")
    print(f"來源      : {calibration.source}")
    print(f"可靠      : {'是' if calibration.is_reliable else '否'}")
    for message in calibration.warnings:
        print(f"  ⚠ {message}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotate_calibration(frame.image, calibration))
        print(f"標註圖    : {out}")
        print("  綠框 = 偵測到的牌桌矩形,橘線 = 三分格線。請目視確認綠框有沒有貼齊遊戲畫布。")

    if canvas_out:
        canvas_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(canvas_out), calibration.crop(frame.image))
        tr = calibration.table_rect
        print(f"純畫布圖  : {canvas_out}  ({tr.width}x{tr.height})")
        print("  已裁掉瀏覽器分頁列/網址列等視窗裝飾,座標可直接當 ROI 量測的參考影像使用。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="擷取層探測工具", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="列出視窗後結束")
    parser.add_argument("--all", action="store_true", help="--list 時不過濾系統浮層")
    parser.add_argument(
        "--window",
        metavar="SPEC",
        help="目標視窗:標題/程式名子字串,或 #視窗ID。省略則用設定檔的 pattern",
    )
    parser.add_argument("--out", type=Path, metavar="PATH", help="輸出 PNG 路徑")
    parser.add_argument("--bench", type=int, metavar="N", help="連續擷取 N 幀並統計耗時")
    parser.add_argument("--calibrate", action="store_true", help="執行牌桌校正")
    parser.add_argument(
        "--canvas-out", type=Path, metavar="PATH",
        help="連同 --calibrate 使用:輸出裁掉瀏覽器視窗裝飾後的純遊戲畫布,供 ROI 量測用",
    )
    parser.add_argument(
        "--backend", choices=["auto", "macos", "windows", "mss"], help="覆寫設定檔的後端選擇"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    overrides = {"capture": {"backend": args.backend}} if args.backend else None
    config = load_config(overrides=overrides)

    try:
        with create_backend(config.capture) as backend:
            if args.list:
                cmd_list(backend, include_all=args.all)
                return 0

            window = _resolve_window(backend, args.window, config)

            if args.bench:
                cmd_bench(backend, window, args.bench)
            elif args.calibrate:
                cmd_calibrate(backend, window, config, args.out, canvas_out=args.canvas_out)
            else:
                cmd_capture(backend, window, args.out)
    except CaptureError as exc:
        print(f"\n擷取失敗:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
