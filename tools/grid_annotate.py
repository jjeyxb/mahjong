#!/usr/bin/env python3
"""牌河網格標定:拖四個角,即時看 6x3 網格對齊得如何。

為什麼是拖四個角而不是自動偵測
------------------------------
試過從「非桌面色」的剪影自動擬合(凸包多邊形逼近、四邊包絡線加穩健迴歸都試
了),四家牌河沒有一個對得準。原因不在演算法:**參考畫面裡沒有任何一個牌河是
滿的 18 格**,右下角那格幾乎不會被填到,於是「右邊界的下段」與「下邊界的右段」
兩條資料同時缺席,任何方法都只是在對不存在的資料外推。

人眼反而毫無困難 —— 網格的走向從已經填了的格子一眼就看得出來,延伸到空格是
本能。所以這裡只要求你把四個角拖到位,同時把 6x3 網格即時疊上去當回饋:格線
壓在牌與牌的縫上就對了。

用法::

    python tools/grid_annotate.py --image data/roi_ref/extent.png \\
        --out data/roi_ref/grids.yaml

參考圖要是**純畫布圖**(``capture_probe.py --canvas-out`` 或 ``roi_union.py``
的輸出),座標才會與 ``config/default.yaml`` 的 ROI 同一個基準。

操作方式
--------
===========  ==========================================
拖曳角落      移動最近的那個角(黃色圓點)
Enter        確認,下一家
s            跳過(保留原值)
b            上一家
r            重設成這一家 river 矩形的四個角
n            換下一張參考圖(角不動,只換底圖)
+ / -        放大 / 縮小顯示
q / Esc      結束並輸出
===========  ==========================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import yaml

from majsoul_copilot.calibration.table import Calibration
from majsoul_copilot.config.loader import load_config
from majsoul_copilot.utils.geometry import Rect, Size
from majsoul_copilot.vision import RoiSet
from majsoul_copilot.vision.grid import COLS, ROWS, Grid

WINDOW = "River grid"
HUD_HEIGHT = 58
NAV_KEYS = frozenset({13, 10, 32, ord("s"), ord("b")})
NAV_DEBOUNCE = 0.25

#: 每家牌河,以及它在畫面上的方位。``vertical`` 代表「一列 6 張」的方向在畫面上
#: 是垂直的 —— 上家與下家坐在左右兩側,他們的牌河跟著轉了 90 度。
SEATS: tuple[tuple[str, str, bool], ...] = (
    ("own", "自家,畫面下方", False),
    ("shimocha", "下家,畫面右側", True),
    ("toimen", "對面,畫面上方", False),
    ("kamicha", "上家,畫面左側", True),
)

#: 顯示時在 river 矩形外多留這個比例,免得角落被切在畫面邊緣不好拖。
MARGIN = 0.18

HANDLE_RADIUS = 9
GRAB_RADIUS = 22

COLOR_QUAD = (0, 255, 255)
COLOR_GRID = (60, 90, 255)
COLOR_HANDLE = (0, 220, 255)
COLOR_ACTIVE = (80, 255, 80)


def _seat_grid(corners: list[tuple[float, float]], vertical: bool) -> Grid:
    """依方位把四個角排成 Grid 要的順序。

    :class:`Grid` 固定把第一條邊當成「一列 6 張」的方向。左右兩側的牌河轉了
    90 度,所以要把四個角整體轉一格,長邊才會對到 6 那一軸。
    """
    ordered = corners[1:] + corners[:1] if vertical else corners
    return Grid(tuple(ordered))


class Annotator:
    def __init__(
        self,
        images: list[np.ndarray],
        names: list[str],
        rois: RoiSet,
        values: dict[str, Any],
        max_width: int,
        max_height: int,
        on_save: Any,
    ) -> None:
        self.images = images
        self.names = names
        self.rois = rois
        self.values = values
        self.max_width = max_width
        self.max_height = max_height
        self.on_save = on_save

        self.image_index = 0
        self.index = 0
        self.dragging: int | None = None
        self._last_nav = 0.0
        self._base: np.ndarray | None = None
        self._base_key: tuple[Any, ...] = ()

        self.corners: dict[str, list[tuple[float, float]]] = {}
        for seat, _, _ in SEATS:
            stored = values.get(seat)
            self.corners[seat] = (
                [(float(x), float(y)) for x, y in stored] if stored else self._seed(seat)
            )

    # ---------------------------------------------------------------- 幾何

    def _seed(self, seat: str) -> list[tuple[float, float]]:
        """起始位置用該家 river 矩形的四個角 —— 已經很接近,拖動距離短。"""
        rect = self.rois[f"{seat}.river"].rect
        return [
            (float(rect.x), float(rect.y)),
            (float(rect.right), float(rect.y)),
            (float(rect.right), float(rect.bottom)),
            (float(rect.x), float(rect.bottom)),
        ]

    @property
    def seat(self) -> str:
        return SEATS[self.index][0]

    @property
    def view(self) -> Rect:
        """目前這一家的顯示範圍(river 矩形外加一圈邊界)。"""
        rect = self.rois[f"{self.seat}.river"].rect
        mx, my = int(rect.width * MARGIN), int(rect.height * MARGIN)
        height, width = self.images[0].shape[:2]
        return Rect.from_bounds(
            max(0, rect.x - mx), max(0, rect.y - my),
            min(width, rect.right + mx), min(height, rect.bottom + my),
        )

    @property
    def scale(self) -> float:
        """縮放倍率。

        寬與高都要受限:上家與下家的牌河是直的,只看寬度的話會算出 1200x1280
        這種超出螢幕的視窗,而視窗一旦被系統縮放,滑鼠座標就跟畫面對不上了。
        """
        view = self.view
        return min(4.0, self.max_width / view.width,
                   self.max_height / max(1, view.height - HUD_HEIGHT))

    def _to_display(self, point: tuple[float, float]) -> tuple[int, int]:
        view, s = self.view, self.scale
        return int((point[0] - view.x) * s), int((point[1] - view.y) * s)

    def _to_image(self, x: int, y: int) -> tuple[float, float]:
        view, s = self.view, self.scale
        return view.x + x / s, view.y + y / s

    # ---------------------------------------------------------------- 滑鼠

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:  # noqa: ARG002
        y -= HUD_HEIGHT
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [self._to_display(p) for p in self.corners[self.seat]]
            distances = [np.hypot(px - x, py - y) for px, py in pts]
            nearest = int(np.argmin(distances))
            self.dragging = nearest if distances[nearest] <= GRAB_RADIUS else None
            return
        if self.dragging is None:
            return
        self.corners[self.seat][self.dragging] = self._to_image(x, y)
        # 在視窗外放開左鍵收不到 LBUTTONUP,所以也認「左鍵已經不是按著的狀態」
        if event == cv2.EVENT_LBUTTONUP or not flags & cv2.EVENT_FLAG_LBUTTON:
            self.dragging = None

    # ---------------------------------------------------------------- 繪製

    def _scaled(self) -> np.ndarray:
        key = (self.image_index, self.index, self.max_width, self.max_height)
        if self._base is None or self._base_key != key:
            view = self.view
            crop = self.images[self.image_index][view.as_slice()]
            s = self.scale
            self._base = cv2.resize(
                crop, (int(view.width * s), int(view.height * s)),
                interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA,
            )
            self._base_key = key
        return self._base

    def render(self) -> np.ndarray:
        canvas = self._scaled().copy()
        corners = self.corners[self.seat]
        vertical = SEATS[self.index][2]
        grid = _seat_grid(corners, vertical)

        for row in range(ROWS):
            for col in range(COLS):
                pts = np.array(
                    [self._to_display(p) for p in grid.cell_quad(col, row)], np.int32
                )
                cv2.polylines(canvas, [pts], True, COLOR_GRID, 1, cv2.LINE_AA)

        quad = np.array([self._to_display(p) for p in corners], np.int32)
        cv2.polylines(canvas, [quad], True, COLOR_QUAD, 2, cv2.LINE_AA)
        for i, point in enumerate(corners):
            colour = COLOR_ACTIVE if i == self.dragging else COLOR_HANDLE
            cv2.circle(canvas, self._to_display(point), HANDLE_RADIUS, colour, -1)
            cv2.circle(canvas, self._to_display(point), HANDLE_RADIUS, (0, 0, 0), 1)

        return self._hud(canvas)

    def _hud(self, canvas: np.ndarray) -> np.ndarray:
        bar = np.zeros((HUD_HEIGHT, canvas.shape[1], 3), np.uint8)
        seat, _, vertical = SEATS[self.index]
        axis = "6-axis vertical" if vertical else "6-axis horizontal"
        cv2.putText(bar, f"[{self.index + 1}/{len(SEATS)}]  {seat}  ({axis})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, "drag corners  Enter=ok  r=reset  s=skip  b=back  n=nextimg  +/- q",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        if len(self.images) > 1:
            cv2.putText(bar, self.names[self.image_index],
                        (canvas.shape[1] - 260, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (140, 255, 140), 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    # ---------------------------------------------------------------- 迴圈

    def _announce(self) -> None:
        seat, hint, _ = SEATS[self.index]
        print(f"\n[{self.index + 1}/{len(SEATS)}] {seat} —— {hint}")
        print("  把四個角拖到牌河那一整塊的外緣,格線壓在牌縫上就對了。")

    def _advance(self, step: int) -> None:
        self.dragging = None
        wrapped = not 0 <= self.index + step < len(SEATS)
        self.index = (self.index + step) % len(SEATS)
        if wrapped:
            print("\n--- 繞回開頭。已標的都留著,要結束請按 q ---")
        self._announce()

    def run(self) -> dict[str, Any]:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self._announce()
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n視窗已關閉,輸出目前結果。")
                    break
                if key in NAV_KEYS:
                    now = time.monotonic()
                    if now - self._last_nav < NAV_DEBOUNCE:
                        continue
                    self._last_nav = now
                if key in (ord("q"), 27):
                    print("\n中止,輸出目前結果。")
                    break
                if key in (13, 10, 32):
                    self.values[self.seat] = [list(p) for p in self.corners[self.seat]]
                    self.on_save(self.values)
                    self._advance(1)
                elif key == ord("s"):
                    self._advance(1)
                elif key == ord("b"):
                    self._advance(-1)
                elif key == ord("r"):
                    self.corners[self.seat] = self._seed(self.seat)
                    print("  已重設成 river 矩形的四個角")
                elif key == ord("n"):
                    self.image_index = (self.image_index + 1) % len(self.images)
                    print(f"  參考圖: {self.names[self.image_index]}")
                elif key in (ord("+"), ord("=")):
                    self.max_width = int(self.max_width * 1.15)
                    self.max_height = int(self.max_height * 1.15)
                elif key in (ord("-"), ord("_")):
                    self.max_width = max(300, int(self.max_width / 1.15))
                    self.max_height = max(200, int(self.max_height / 1.15))
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return self.values


def _normalise(values: dict[str, Any], table: Rect) -> dict[str, Any]:
    """像素座標 → 相對牌桌矩形的正規化座標。"""
    out: dict[str, Any] = {}
    for seat, _, _ in SEATS:
        pts = values.get(seat)
        out[seat] = (
            [[round((x - table.x) / table.width, 5), round((y - table.y) / table.height, 5)]
             for x, y in pts]
            if pts else None
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="牌河網格四角標定",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, action="append", default=[], required=True,
                        metavar="PATH", help="純畫布參考圖,可重複給多張(標註時按 n 切換)")
    parser.add_argument("--out", type=Path, metavar="PATH", help="輸出 YAML 片段")
    parser.add_argument("--max-width", type=int, default=1200, help="顯示寬度上限")
    parser.add_argument("--max-height", type=int, default=740,
                        help="顯示高度上限。側面兩家的牌河是直的,不限高會超出螢幕")
    args = parser.parse_args(argv)

    images: list[np.ndarray] = []
    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"讀不到影像: {path}")
        if images and image.shape[:2] != images[0].shape[:2]:
            raise SystemExit(
                f"{path} 的尺寸與第一張不同。請先用 roi_union.py --dump-aligned 對位。"
            )
        images.append(image)
    names = [p.name for p in args.image]

    height, width = images[0].shape[:2]
    config = load_config()
    rois = RoiSet(config.roi, Calibration(
        table_rect=Rect(0, 0, width, height), image_size=Size(width, height)))
    missing = [f"{s}.river" for s, _, _ in SEATS if f"{s}.river" not in rois]
    if missing:
        raise SystemExit(f"這些 river 矩形還沒量測,無法決定起始位置: {', '.join(missing)}")

    print(f"參考影像: {names[0]}  ({width}x{height})")

    values: dict[str, Any] = {}
    if args.out and args.out.exists():
        loaded = yaml.safe_load(args.out.read_text(encoding="utf-8")) or {}
        stored = loaded.get("river_grid", loaded)
        table = Rect(0, 0, width, height)
        for seat, _, _ in SEATS:
            pts = stored.get(seat) if isinstance(stored, dict) else None
            if pts:
                values[seat] = [
                    (table.x + x * table.width, table.y + y * table.height) for x, y in pts
                ]

    table_rect = Rect(0, 0, width, height)

    def save(current: dict[str, Any]) -> None:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                yaml.safe_dump({"river_grid": _normalise(current, table_rect)},
                               allow_unicode=True, sort_keys=False, default_flow_style=None),
                encoding="utf-8",
            )

    print("\n提示:走完最後一家會繞回開頭繼續,要結束請按 q。")
    result = Annotator(images, names, rois, values, args.max_width, args.max_height, save).run()

    block = {"river_grid": _normalise(result, table_rect)}
    text = yaml.safe_dump(block, allow_unicode=True, sort_keys=False, default_flow_style=None)
    done = [s for s, _, _ in SEATS if result.get(s)]
    print("\n" + "=" * 60)
    print(text.rstrip())
    print("=" * 60)
    print(f"已標定 {len(done)}/{len(SEATS)}")
    if args.out:
        save(result)
        print(f"\n已寫入 {args.out}")
        print("  片段檔 —— 確認無誤後併進 config/default.yaml 的 roi.river_grid 區塊。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
