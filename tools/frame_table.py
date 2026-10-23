#!/usr/bin/env python3
"""手動框出牌桌矩形 —— 自動校正走不通時的逃生門。

為什麼需要手動
--------------
自動校正是從四邊往中間剝除整條顏色一致的帶狀區域。這招在大部分視窗尺寸下
可以,但它是啟發式,而且**錯的時候很安靜**:

* 實測 ``20260801-112437``(1800x1186):剝除停在瀏覽器工具列裡,剝出來的
  長寬比 1.56,整份錄影沒有一幀校得準。
* 同一份錄影開頭遊戲正在重連,畫面全黑 —— 黑畫面剝得特別順,給出一個
  長寬比 1.7645(離 16:9 只差 0.7%)的矩形,於是被鎖進整場。

牌桌矩形是**所有東西的共同原點**:ROI 與四家牌河的角點全部相對它正規化。
框歪 3%,手牌的槽位會一路累加偏移,到第 13 張差了半張牌。

手動框一次就好,而且它**不會錯得無聲無息** —— 框歪了自己看得到。

用法
----
從錄影裡挑一幀來框(要挑**牌桌已經出現**的那種,不要載入畫面)::

    python tools/frame_table.py data/recordings/20260801-112437 --frame 1857

或直接框一張圖::

    python tools/frame_table.py --image canvas.png

拖曳框出遊戲畫布(不含瀏覽器分頁列與網址列),Enter 確認、c 取消重框。
輸出是**相對整張影像**的比例,可以直接貼進 ``config/default.yaml`` 的
``calibration.manual_table_rect``,或用 ``--table-rect`` 傳給評測工具。

為什麼正規化到影像而不是別的
----------------------------
與 ``calibration.manual_table_rect`` 同一個格式,所以同一組數字兩邊都能用。
遊戲畫面是跟著視窗等比變的,存成比例之後視窗拉大也不必重框 —— 除非視窗的
**長寬比**變了,那時框會歪,而那是看得見的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.recorder import SessionReader

#: 顯示用的最大寬度。4K 擷取直接開會超出螢幕,框不到右半邊。
MAX_WIDTH = 1400

#: 16:9。印出來對照用 —— 框出來的比例離它太遠通常是把工具列也框進去了。
EXPECTED_ASPECT = 16 / 9


def _load(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """把要框的那張圖讀出來,連同它的來歷。"""
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"讀不到 {args.image}")
        return image, str(args.image)

    session = SessionReader(args.session)
    records = session.manifest.frames
    if not records:
        raise SystemExit(f"{args.session} 裡沒有任何幀")
    by_index = {r.index: r for r in records}
    if args.frame is None:
        # 預設取中段而不是第一幀:錄製開頭常常停在載入畫面或選單,
        # 而那正是最不該拿來框的東西 —— 那時螢幕上根本沒有牌桌。
        record = records[len(records) // 2]
    elif args.frame in by_index:
        record = by_index[args.frame]
    else:
        raise SystemExit(f"session 裡沒有第 {args.frame} 幀(有 {len(records)} 幀)")
    return session.load_image(record), f"{args.session} 第 {record.index} 幀"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path, nargs="?", help="tools/record.py 錄的 session 目錄")
    parser.add_argument("--frame", type=int, default=None, help="要框哪一幀,預設取中段")
    parser.add_argument("--image", type=Path, default=None, help="改成直接框一張圖")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH, help="顯示用的最大寬度")
    args = parser.parse_args(argv)

    if args.session is None and args.image is None:
        parser.error("要給一個 session 目錄,或用 --image 指定一張圖")

    image, origin = _load(args)
    height, width = image.shape[:2]
    print(f"來源:{origin}({width}x{height})")
    print("拖曳框出遊戲畫布,不要含瀏覽器分頁列與網址列。Enter 確認,c 取消重框。")

    scale = min(1.0, args.max_width / width)
    shown = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image
    box = cv2.selectROI("框出牌桌(Enter 確認 / c 取消)", shown, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = (round(v / scale) for v in box)
    if w <= 0 or h <= 0:
        print("\n沒有框任何東西。")
        return 1

    aspect = w / h
    deviation = abs(aspect - EXPECTED_ASPECT) / EXPECTED_ASPECT
    print(f"\n像素    : {w}x{h}@({x},{y})")
    print(f"長寬比  : {aspect:.4f}(16:9 是 {EXPECTED_ASPECT:.4f},差 {deviation:.1%})")
    if deviation > 0.03:
        # 不擋 —— 有些環境本來就不是 16:9(實測 Steam 版客戶區 1.7915)。
        # 但差這麼多最常見的原因是把工具列也框進去了,那值得當場講一聲。
        print("  ⚠ 差得有點多。最常見的原因是把瀏覽器工具列也框進去了。")

    norm = (x / width, y / height, w / width, h / height)
    print("\n貼進 config/default.yaml:")
    print(f"  manual_table_rect: [{norm[0]:.4f}, {norm[1]:.4f}, {norm[2]:.4f}, {norm[3]:.4f}]")
    print("\n或直接傳給評測工具:")
    print(f"  --table-rect {norm[0]:.4f},{norm[1]:.4f},{norm[2]:.4f},{norm[3]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
