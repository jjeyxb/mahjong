#!/usr/bin/env python3
"""ROI 標註器 —— 在參考畫面上手動拉出各區域的外框。

輸入必須是**純畫布圖**(已裁掉瀏覽器分頁列/網址列的遊戲畫面),因為
`RoiConfig` 的座標是相對 `table_rect` 正規化的。用這個指令產生::

    .venv/bin/python tools/capture_probe.py --window "Chrome for Testing" \\
        --calibrate --canvas-out canvas.png

然後標註::

    .venv/bin/python tools/roi_annotate.py --image canvas.png

一次標不完、或某些區域要換一張參考圖(例如自己的副露要挑有吃碰槓的那局),
就分次跑,用 --only 指定欄位、--out 指定同一個檔案累積結果::

    .venv/bin/python tools/roi_annotate.py --image a.png --out roi.yaml
    .venv/bin/python tools/roi_annotate.py --image b.png --out roi.yaml --only own_hand

--out 檔已存在時會先載入當作起始值,只覆寫這次真的重畫的欄位。

操作方式
--------
滑鼠左鍵拖曳畫框,然後:

===========  ==========================================
Enter/Space  確認,前往下一個欄位
r            清掉目前這個框重畫
s            跳過(保留原值,沒有原值就維持 null)
b            回上一個欄位
d            刪除這個欄位已存的值(設回 null)
n            換下一張參考圖(框不動,只換底圖)
+ / -        放大 / 縮小顯示
q / Esc      結束並輸出目前結果
===========  ==========================================
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import yaml

from mia.config.loader import load_config

WINDOW = "ROI annotate"

#: 顯示用的最大寬度。原圖通常是 2560 寬(Retina),直接開會超出螢幕。
#: 太接近螢幕寬度時 macOS 會自己縮放視窗,一縮放滑鼠座標就跟畫面對不上,
#: 所以預設值留了餘裕。螢幕夠大可以用 --max-width 調高。
DEFAULT_MAX_WIDTH = 1200

#: HUD 橫幅的高度。它是疊在畫布**上方**的,所以滑鼠的 y 要先扣掉這個值
#: 才會是畫布座標 —— 忘了扣的話框會固定畫在游標上方 58px。
HUD_HEIGHT = 58

#: 會切換欄位的按鍵,需要去彈跳,否則按久一點會一次跳掉好幾個欄位。
NAV_KEYS = frozenset({13, 10, 32, ord("s"), ord("b")})
NAV_DEBOUNCE = 0.25  # 秒

# BGR
COLOR_CURRENT = (0, 255, 255)  # 正在畫的框:亮黃
COLOR_DONE = (120, 220, 120)   # 已確認的框:淡綠
COLOR_HUD = (255, 255, 255)
COLOR_CROSSHAIR = (80, 80, 255)  # 準星:偏紅,牌桌上不會撞色


@dataclass(frozen=True)
class RoiField:
    """一個要標註的 ROI 欄位。

    ``label`` 只用 ASCII —— OpenCV 的 putText 畫不出中文,中文說明走終端機。
    """

    path: str
    label: str
    hint: str


#: 目前 CV 只辨識自己的手牌 —— 其餘狀態由封包提供,見 docs/decisions.md。
#: 這個清單刻意跟 config.models.RoiConfig 的欄位一一對應,加欄位時兩邊要一起改。
FIELDS: tuple[RoiField, ...] = (
    RoiField("own_hand", "own_hand  [BOTTOM edge]", "自己的手牌,畫面最下緣那一排(含摸進來那張)"),
)

FIELD_BY_PATH = {f.path: f for f in FIELDS}


# --------------------------------------------------------------- 巢狀取值

def _get(values: dict[str, Any], path: str) -> tuple[float, float, float, float] | None:
    node: Any = values
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if node is None:
        return None
    return tuple(float(v) for v in node)  # type: ignore[return-value]


def _set(values: dict[str, Any], path: str, rect: tuple[float, ...] | None) -> None:
    parts = path.split(".")
    node = values
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = list(rect) if rect is not None else None


# --------------------------------------------------------------- 載入起始值

def _load_existing(out: Path | None, *, fresh: bool) -> dict[str, Any]:
    """起始值來源:--out 既有檔優先,其次是專案設定檔裡已量好的 roi。"""
    if fresh:
        return {}

    if out and out.exists():
        loaded = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"{out} 的內容不是 mapping,無法當起始值")
        return loaded.get("roi", loaded)

    roi = load_config().roi
    return roi.model_dump(exclude_none=False)


# --------------------------------------------------------------- 標註主迴圈

class Annotator:
    def __init__(self, images: list[np.ndarray], names: list[str], fields: list[RoiField],
                 values: dict[str, Any], max_width: int,
                 on_save: Callable[[dict[str, Any]], None] = lambda _: None) -> None:
        self.on_save = on_save  # 每確認一個框就存檔,視窗被關掉也不會白做
        self.images = images
        self.names = names
        self.image_index = 0
        self.fields = fields
        self.values = values
        self.height, self.width = images[0].shape[:2]
        self.scale = min(1.0, max_width / self.width)
        self.index = 0

        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self.cursor: tuple[int, int] = (0, 0)
        self.pending: tuple[float, float, float, float] | None = None

        self._last_nav = 0.0

        # 縮圖每幀都重算的話,2560x1440 的 resize 會讓拖曳明顯跟不上手
        self._base: np.ndarray | None = None
        self._base_key: tuple[float, int] = (-1.0, -1)

    def _scaled(self) -> np.ndarray:
        key = (self.scale, self.image_index)
        if self._base is None or self._base_key != key:
            self._base = cv2.resize(
                self.images[self.image_index],
                (int(self.width * self.scale), int(self.height * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
            self._base_key = key
        return self._base

    # ---------------------------------------------------------- 座標換算

    def _to_norm(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float, float, float]:
        """兩個顯示座標的點 → 相對整張畫布的正規化 (x, y, w, h)。"""
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        nx0 = min(max(x0 / self.scale / self.width, 0.0), 1.0)
        nx1 = min(max(x1 / self.scale / self.width, 0.0), 1.0)
        ny0 = min(max(y0 / self.scale / self.height, 0.0), 1.0)
        ny1 = min(max(y1 / self.scale / self.height, 0.0), 1.0)
        return (nx0, ny0, nx1 - nx0, ny1 - ny0)

    def _to_display(self, rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x, y, w, h = rect
        return (
            int(x * self.width * self.scale),
            int(y * self.height * self.scale),
            int((x + w) * self.width * self.scale),
            int((y + h) * self.height * self.scale),
        )

    # ---------------------------------------------------------- 滑鼠

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:  # noqa: ARG002
        y -= HUD_HEIGHT  # 視窗最上面那條是 HUD,不是畫布
        self.cursor = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 不管前一次拖曳有沒有正常收尾,按下去就是重新開始
            self.drag_start = (x, y)
            self.drag_now = (x, y)
            return

        if self.drag_start is None:
            return
        self.drag_now = (x, y)

        # 在視窗**外**放開左鍵的話收不到 LBUTTONUP,拖曳就會永遠卡住、
        # 橡皮筋框黏著游標跑 —— 框最上/最下緣的區域時很容易發生。
        # 所以除了 UP 事件,也認「左鍵已經不是按著的狀態」這個條件。
        if event == cv2.EVENT_LBUTTONUP or not flags & cv2.EVENT_FLAG_LBUTTON:
            rect = self._to_norm(self.drag_start, self.drag_now)
            # 只是點一下、沒有真的拖出面積的話當作取消,避免誤觸把框清成 0
            self.pending = rect if rect[2] > 0.002 and rect[3] > 0.002 else None
            self.drag_start = None

    # ---------------------------------------------------------- 繪製

    def render(self) -> np.ndarray:
        canvas = self._scaled().copy()
        field = self.fields[self.index]

        # 其他欄位已確認的框:畫淡一點當參考,才知道有沒有重疊或漏掉
        for other in self.fields:
            if other.path == field.path:
                continue
            rect = _get(self.values, other.path)
            if rect is None:
                continue
            x0, y0, x1, y1 = self._to_display(rect)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_DONE, 1)
            cv2.putText(canvas, other.path, (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DONE, 1, cv2.LINE_AA)

        # 目前欄位:優先顯示正在拖曳的,其次是這次畫好的,再其次是既有值
        active = None
        if self.drag_start is not None and self.drag_now is not None:
            active = self._to_norm(self.drag_start, self.drag_now)
        elif self.pending is not None:
            active = self.pending
        else:
            active = _get(self.values, field.path)

        if active is not None:
            x0, y0, x1, y1 = self._to_display(active)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_CURRENT, 2)

        self._draw_crosshair(canvas)
        return self._draw_hud(canvas, field, active)

    def _draw_crosshair(self, canvas: np.ndarray) -> None:
        """畫十字準星。

        除了對齊好用之外,它也是滑鼠座標有沒有對上的自我診斷 —— 準星要是跟
        實際游標位置差一截,就代表視窗被系統縮放了,調小 --max-width 即可。
        """
        cx, cy = self.cursor
        height, width = canvas.shape[:2]
        if not (0 <= cx < width and 0 <= cy < height):
            return
        cv2.line(canvas, (cx, 0), (cx, height), COLOR_CROSSHAIR, 1)
        cv2.line(canvas, (0, cy), (width, cy), COLOR_CROSSHAIR, 1)

    def _draw_hud(self, canvas: np.ndarray, field: RoiField,
                  active: tuple[float, float, float, float] | None) -> np.ndarray:
        bar = np.zeros((HUD_HEIGHT, canvas.shape[1], 3), dtype=np.uint8)
        head = f"[{self.index + 1}/{len(self.fields)}]  {field.label}"
        if len(self.images) > 1:
            source = (f"n> [{self.image_index + 1}/{len(self.images)}] "
                      f"{self.names[self.image_index]}")
            cv2.putText(bar, source, (canvas.shape[1] - 640, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (140, 255, 140), 1, cv2.LINE_AA)
        cv2.putText(bar, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1, cv2.LINE_AA)

        cx = self.cursor[0] / self.scale / self.width
        cy = self.cursor[1] / self.scale / self.height
        if active is None:
            detail = f"(unset)          cursor {cx:.3f},{cy:.3f}"
        else:
            detail = (
                f"x={active[0]:.4f} y={active[1]:.4f} w={active[2]:.4f} h={active[3]:.4f}"
                f"    cursor {cx:.3f},{cy:.3f}"
            )
        cv2.putText(bar, detail, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (180, 220, 255), 1, cv2.LINE_AA)
        if self.drag_start is not None:
            cv2.putText(bar, "DRAGGING", (canvas.shape[1] - 110, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CURRENT, 1, cv2.LINE_AA)
        cv2.putText(bar, "drag  Enter=ok r=redo s=skip b=back d=clear n=nextimg +/- q=quit",
                    (canvas.shape[1] - 640, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (160, 160, 160), 1, cv2.LINE_AA)
        return np.vstack([bar, canvas])

    # ---------------------------------------------------------- 迴圈

    def _announce(self) -> None:
        field = self.fields[self.index]
        current = _get(self.values, field.path)
        state = "已有值,不重畫就按 s 保留" if current else "尚未量測"
        print(f"\n[{self.index + 1}/{len(self.fields)}] {field.path} —— {field.hint}  ({state})")

    def _advance(self, step: int) -> None:
        """換到下一個欄位。

        走到底會繞回第一個,**不會自動結束** —— 每個欄位在不同參考圖上看起來
        差很多(滿手 13 張 vs 只剩 1 張),第一輪量完通常還要回頭修,結束的時機
        只能由使用者用 q 決定。
        """
        self.pending = None
        wrapped = not 0 <= self.index + step < len(self.fields)
        self.index = (self.index + step) % len(self.fields)
        if wrapped:
            print("\n--- 繞回開頭。已量的值都留著,要結束請按 q ---")
        self._announce()

    def run(self) -> dict[str, Any]:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self._announce()
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKey(20) & 0xFF

                # 視窗被使用者關掉 —— 當成 q,把已標的結果吐出來
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n視窗已關閉,輸出目前結果。")
                    break

                # 換欄位的鍵做去彈跳:Enter 按久一點會被 20ms 的輪詢連續讀到,
                # 一路跳過好幾個欄位,而畫面上只看得出「怎麼突然不能畫了」。
                if key in NAV_KEYS:
                    now = time.monotonic()
                    if now - self._last_nav < NAV_DEBOUNCE:
                        continue
                    self._last_nav = now

                if key in (ord("q"), 27):
                    print("\n中止,輸出目前結果。")
                    break
                if key in (13, 10, 32):  # Enter / Return / Space
                    if self.pending is not None:
                        _set(self.values, self.fields[self.index].path, self.pending)
                        self.on_save(self.values)
                    self._advance(1)
                elif key == ord("r"):
                    self.pending = None
                    self.drag_start = None  # 卡住的拖曳也一併清掉
                elif key == ord("s"):
                    self._advance(1)
                elif key == ord("b"):
                    self._advance(-1)
                elif key == ord("d"):
                    _set(self.values, self.fields[self.index].path, None)
                    self.on_save(self.values)
                    self.pending = None
                    print("  已清空這個欄位")
                elif key == ord("n"):
                    # 換底圖但框不動 —— 同一個框可以立刻對照別的局面驗證
                    self.image_index = (self.image_index + 1) % len(self.images)
                    print(f"  參考圖: {self.names[self.image_index]}")
                elif key in (ord("+"), ord("=")):
                    self.scale = min(self.scale * 1.15, 2.0)
                elif key in (ord("-"), ord("_")):
                    self.scale = max(self.scale / 1.15, 0.15)
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # macOS 需要多轉一圈事件迴圈視窗才真的消失
        return self.values


# --------------------------------------------------------------- 輸出

def _round(values: Any) -> Any:
    if isinstance(values, dict):
        return {k: _round(v) for k, v in values.items()}
    # 從 pydantic model_dump 來的是 tuple,safe_dump 不認得,一律轉成 list
    if isinstance(values, list | tuple):
        return [round(float(v), 4) for v in values]
    return values


def _dump(values: dict[str, Any]) -> str:
    block = {"roi": _round(values)}
    return yaml.safe_dump(block, allow_unicode=True, sort_keys=False, default_flow_style=None)


def _write(values: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump(values), encoding="utf-8")


def _emit(values: dict[str, Any], out: Path | None) -> None:
    text = _dump(values)

    done = [f.path for f in FIELDS if _get(values, f.path) is not None]
    missing = [f.path for f in FIELDS if _get(values, f.path) is None]

    print("\n" + "=" * 60)
    print(text.rstrip())
    print("=" * 60)
    print(f"已量測 {len(done)}/{len(FIELDS)}")
    if missing:
        print("尚缺: " + ", ".join(missing))

    if out:
        _write(values, out)
        print(f"\n已寫入 {out}")
        print("  這是片段檔,不是完整設定 —— 確認無誤後再併進 config/default.yaml 的 roi: 區塊。")


# --------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="手動標註牌桌各區域 ROI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, action="append", default=[], metavar="PATH",
                        help="參考畫面,必須是純畫布圖。可重複給多張,標註時按 n 切換 "
                             "—— 框不動、只換底圖,同一個框能立刻對照不同局面驗證")
    parser.add_argument("--out", type=Path, metavar="PATH",
                        help="輸出 YAML 片段。檔案已存在時會先載入當起始值")
    parser.add_argument("--only", metavar="A,B",
                        help="只標註這幾個欄位(逗號分隔),例如 own_hand")
    parser.add_argument("--fresh", action="store_true",
                        help="不要載入任何既有值,從全空開始")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH,
                        help=f"顯示寬度上限,預設 {DEFAULT_MAX_WIDTH}")
    parser.add_argument("--list-fields", action="store_true", help="列出所有欄位後結束")
    args = parser.parse_args(argv)

    if args.list_fields:
        for field in FIELDS:
            print(f"  {field.path:<18} {field.hint}")
        return 0

    if not args.image:
        parser.error("需要 --image(或用 --list-fields 只看欄位列表)")

    images: list[np.ndarray] = []
    names: list[str] = []
    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"讀不到影像: {path}")
        images.append(image)
        names.append(path.name)

    height, width = images[0].shape[:2]
    aspect = width / height
    print(f"參考影像: {names[0]}  ({width}x{height},比例 {aspect:.4f})")
    for extra, path in zip(images[1:], args.image[1:], strict=True):
        # 尺寸不同就代表座標系不同,疊在同一組框上量必定錯位。對位是
        # roi_union.py 的工作(它有整圖比對),這裡只負責擋下來。
        if extra.shape[:2] != (height, width):
            raise SystemExit(
                f"{path} 是 {extra.shape[1]}x{extra.shape[0]},與第一張的 {width}x{height} 不同。\n"
                "請先用 roi_union.py --dump-aligned 把所有參考圖對位到同一個尺寸。"
            )
        print(f"          {path.name}")

    if abs(aspect - 16 / 9) / (16 / 9) > 0.03:
        print("  ⚠ 比例偏離 16:9 超過 3%。這看起來不是裁乾淨的純畫布圖,")
        print("    量出來的座標會有系統性偏移。建議先用 capture_probe.py --canvas-out 重出一張。")

    if args.only:
        wanted = [p.strip() for p in args.only.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in FIELD_BY_PATH]
        if unknown:
            raise SystemExit(f"未知欄位: {', '.join(unknown)}(用 --list-fields 看有哪些)")
        fields = [FIELD_BY_PATH[p] for p in wanted]
    else:
        fields = list(FIELDS)

    values = _load_existing(args.out, fresh=args.fresh)
    out = args.out
    autosave: Callable[[dict[str, Any]], None] = (
        (lambda v: _write(v, out)) if out else (lambda _: None)
    )
    annotator = Annotator(images, names, fields, values, args.max_width, on_save=autosave)
    print("\n提示:走完最後一個欄位會繞回開頭繼續,要結束請按 q。")
    _emit(annotator.run(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
