"""幾何基礎型別與座標空間轉換。

座標空間約定
------------
本專案存在三種座標空間。任何處理座標的函式都必須清楚知道自己吃的是哪一種,
混用是這類專案最容易產生詭異偏移的來源:

1. **邏輯座標 (logical / points)**
   作業系統回報的視窗位置與大小。macOS Retina 下 1 point = 2 pixels;
   Windows 高 DPI 下亦有 1.25x / 1.5x / 2x 等縮放。

2. **像素座標 (pixel / device)**
   實際擷取到的影像像素。**所有 CV 運算一律在此空間進行。**

3. **正規化座標 (normalized)**
   相對於「牌桌矩形」的 0.0 ~ 1.0 比例,見 :class:`NormRect`。
   **所有 ROI 定義一律以此空間儲存於 YAML**,如此才能在不同螢幕解析度、
   不同視窗大小、Retina 與非 Retina 之間重複使用同一份設定。

:attr:`majsoul_copilot.capture.base.Frame.scale` 即 pixel / logical 的比值。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NormRect", "Rect", "Size"]


@dataclass(frozen=True, slots=True)
class Size:
    """寬高。單位由使用端決定(邏輯或像素)。"""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Size 不可為負: {self.width}x{self.height}")

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        """寬高比。高度為 0 時回傳 0.0 而非拋例外,方便在檢查流程中使用。"""
        return self.width / self.height if self.height else 0.0

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class Rect:
    """整數矩形,左上角原點。單位由使用端決定(邏輯或像素)。"""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Rect 尺寸不可為負: {self.width}x{self.height}")

    # --- 衍生屬性 ---

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    # --- 建構 ---

    @classmethod
    def from_bounds(cls, left: int, top: int, right: int, bottom: int) -> Rect:
        """由左上/右下邊界建構(right 與 bottom 為 exclusive)。"""
        return cls(left, top, max(0, right - left), max(0, bottom - top))

    @classmethod
    def from_size(cls, size: Size) -> Rect:
        """以原點為左上角、涵蓋整個 size 的矩形。"""
        return cls(0, 0, size.width, size.height)

    # --- 運算 ---

    def scaled(self, factor: float) -> Rect:
        """整體縮放(含位置)。用於邏輯座標 → 像素座標的轉換。"""
        return Rect(
            round(self.x * factor),
            round(self.y * factor),
            round(self.width * factor),
            round(self.height * factor),
        )

    def offset(self, dx: int, dy: int) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def inset(self, amount: int) -> Rect:
        """四邊各內縮 amount 像素;內縮過度時回傳零尺寸矩形而非負值。"""
        return Rect(
            self.x + amount,
            self.y + amount,
            max(0, self.width - 2 * amount),
            max(0, self.height - 2 * amount),
        )

    def intersect(self, other: Rect) -> Rect | None:
        """交集;無重疊時回傳 None。"""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect.from_bounds(left, top, right, bottom)

    def contains(self, other: Rect) -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def as_slice(self) -> tuple[slice, slice]:
        """回傳可直接用於 numpy 影像切片的索引:``image[rect.as_slice()]``。

        注意 numpy 的軸順序是 (y, x),這個方法就是為了避免每次都手寫錯。
        """
        return slice(self.y, self.bottom), slice(self.x, self.right)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}@({self.x},{self.y})"


@dataclass(frozen=True, slots=True)
class NormRect:
    """正規化矩形,座標為相對於某個容器的 0.0 ~ 1.0 比例。

    ROI 定義一律使用這個型別儲存,執行期再透過 :meth:`to_pixels` 換算成
    當下畫面的實際像素位置。
    """

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError(f"NormRect 原點須落在 0~1: ({self.x}, {self.y})")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(f"NormRect 尺寸須為正: {self.width}x{self.height}")
        if self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9:
            raise ValueError(
                f"NormRect 超出容器邊界: "
                f"right={self.x + self.width:.4f} bottom={self.y + self.height:.4f}"
            )

    def to_pixels(self, container: Rect | Size) -> Rect:
        """換算成實際像素矩形。

        container 傳 :class:`Rect` 時會加上容器原點偏移(常見情境:ROI 相對於
        牌桌矩形,而牌桌矩形本身在整張擷取影像中有偏移);傳 :class:`Size`
        時視原點為 (0, 0)。

        寬高至少為 1 像素,避免極小 ROI 被四捨五入成空矩形而在後續切片時炸掉。
        """
        if isinstance(container, Rect):
            ox, oy, cw, ch = container.x, container.y, container.width, container.height
        else:
            ox, oy, cw, ch = 0, 0, container.width, container.height
        return Rect(
            ox + round(self.x * cw),
            oy + round(self.y * ch),
            max(1, round(self.width * cw)),
            max(1, round(self.height * ch)),
        )

    @classmethod
    def from_pixels(cls, rect: Rect, container: Rect | Size) -> NormRect:
        """由像素矩形反推正規化座標。ROI 標註工具會用到。"""
        if isinstance(container, Rect):
            ox, oy, cw, ch = container.x, container.y, container.width, container.height
        else:
            ox, oy, cw, ch = 0, 0, container.width, container.height
        if cw <= 0 or ch <= 0:
            raise ValueError(f"容器尺寸須為正: {cw}x{ch}")
        return cls(
            (rect.x - ox) / cw,
            (rect.y - oy) / ch,
            rect.width / cw,
            rect.height / ch,
        )

    def __str__(self) -> str:
        return (
            f"({self.x:.4f}, {self.y:.4f}, {self.width:.4f}, {self.height:.4f})"
        )
