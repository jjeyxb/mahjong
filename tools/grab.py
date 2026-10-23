#!/usr/bin/env python3
"""抓一張現在的遊戲畫面存成檔案。

診斷用。框選對話框、牌河辨識出問題時,**最花時間的一步是重現**:截圖裡看得到
症狀,卻量不到像素。這個工具抓的是與辨識同一條路的同一張影像
(:func:`~mia.live.snapshot.grab_game_frame`),所以存下來的檔案可以直接餵給
``tools/frame_table.py`` 或任何分析腳本::

    python tools/grab.py --out /tmp/now.png

遊戲要開著,而且版本要與設定一致 —— 它找的是與畫面辨識同一個視窗。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.capture.base import CaptureError, WindowNotFoundError
from mia.config.loader import load_config
from mia.live.snapshot import grab_game_frame
from mia.platforms import WEB
from mia.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("frame.png"), help="存到哪")
    parser.add_argument("--platform", default=WEB, help="web 或 steam")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level)

    try:
        frame = grab_game_frame(load_config(), args.platform)
    except WindowNotFoundError:
        raise SystemExit("找不到遊戲視窗 —— 雀魂開著嗎?--platform 選對了嗎?") from None
    except CaptureError as exc:
        raise SystemExit(f"抓不到畫面:{exc}") from None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), frame.image):
        raise SystemExit(f"寫不出 {args.out}")
    height, width = frame.image.shape[:2]
    print(f"{args.out}:{width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
