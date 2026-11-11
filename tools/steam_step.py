#!/usr/bin/env python3
"""Steam 版看牌譜時,每隔幾秒點一下牌桌空白處,讓牌譜往下走一步 —— 錄素材用,**不是 app 的功能**。

    .venv\\Scripts\\python.exe tools\\steam_step.py                # 每 1.5 秒一步
    .venv\\Scripts\\python.exe tools\\steam_step.py --interval 2   # 慢一點

為什麼要有這個
--------------
Steam 版沒有封包,驗證辨識要靠牌譜:在客戶端播牌譜、``tools/record.py --platform steam`` 錄
畫面,答案從同一份牌譜取。但牌譜的自動播放太快 —— 動畫還沒演完就跳下一步,錄到的多半是
半透明的中間畫面。點牌桌空白處一次走一步,間隔由這裡決定。

網頁版有 ``tools/paipu.py``(透過瀏覽器按、每步截一張);Steam 是 Unity 客戶端,沒有瀏覽器
可以接,只能真的移動滑鼠去點。

兩條安全規則
------------
* **只在雀魂是最前面的視窗時點。** 不是的話這一步跳過、等下一輪 —— 使用者切去別的視窗時,
  點下去的會是那個視窗。
* **連續幾步畫面都沒變就停**,當成牌譜播完了(``--still``)。

點的位置是牌桌左側、上家牌河外面那塊桌布(相對整個 16:9 畫面的比例,``--at`` 可改),
不在任何牌、頭像、按鈕上。點完滑鼠留在原地:擷取用的是 PrintWindow,拍不到游標。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mia.capture.base import CaptureError
from mia.config.loader import load_config
from mia.live.snapshot import grab_game_frame
from mia.platforms import STEAM

#: 預設點這裡:相對畫面的 (x, y)。2560x1440 實測:上家牌河左外側的空桌布。
DEFAULT_AT = (0.22, 0.54)

#: 一步前後縮圖的平均差小於這個,就算「畫面沒變」。
_STILL_DIFF = 0.5


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "win32":
        raise SystemExit("這支工具只給 Windows 的 Steam 版用。")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--interval", type=float, default=1.5, help="每一步間隔幾秒(預設 1.5)")
    parser.add_argument(
        "--at",
        type=_point,
        default=DEFAULT_AT,
        metavar="X,Y",
        help="點哪裡,相對畫面的比例(預設 0.22,0.54)",
    )
    parser.add_argument(
        "--still", type=int, default=6, help="連續幾步畫面都沒變就停(預設 6)"
    )
    parser.add_argument("--max-steps", type=int, default=0, help="最多走幾步,0 表示不限")
    args = parser.parse_args(argv)

    import win32api
    import win32con
    import win32gui

    from mia.capture.windows import enable_dpi_awareness

    enable_dpi_awareness()
    config = load_config()
    try:
        frame = grab_game_frame(config, STEAM)
    except CaptureError as exc:
        raise SystemExit(f"找不到 Steam 版的雀魂視窗:{exc}") from None
    hwnd = frame.window.handle
    previous = _thumb(frame.image)
    print(f"目標:{frame.window}")
    print("請先點一下雀魂視窗讓它在最前面;要停就按 Ctrl-C。", flush=True)

    steps = still = skipped = 0
    try:
        while not args.max_steps or steps < args.max_steps:
            time.sleep(args.interval)
            if win32gui.GetForegroundWindow() != hwnd:
                skipped += 1
                if skipped % 5 == 1:
                    print("雀魂不在最前面,先不點(點一下雀魂視窗就會繼續)", flush=True)
                continue
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            x = round(left + (right - left) * args.at[0])
            y = round(top + (bottom - top) * args.at[1])
            screen_x, screen_y = win32gui.ClientToScreen(hwnd, (x, y))
            win32api.SetCursorPos((screen_x, screen_y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            steps += 1

            time.sleep(min(1.0, args.interval * 0.6))
            try:
                now = _thumb(grab_game_frame(config, STEAM).image)
            except CaptureError:
                continue
            changed = float(np.abs(now - previous).mean()) >= _STILL_DIFF
            previous = now
            still = 0 if changed else still + 1
            if steps % 20 == 0:
                print(f"已走 {steps} 步", flush=True)
            if still >= args.still:
                print(f"連續 {still} 步畫面都沒變,當成播完了,停止。")
                break
    except KeyboardInterrupt:
        print()
    print(f"共走 {steps} 步。記得回錄影那邊按 Ctrl-C。")
    return 0


def _thumb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)


def _point(text: str) -> tuple[float, float]:
    x, y = (float(v) for v in text.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("要是 0~1 之間的比例,例如 0.22,0.54")
    return x, y


if __name__ == "__main__":
    sys.exit(main())
