"""牌河網格:四個角 → 每一格的位置,以及把每一格反扭回正矩形。

為什麼牌河不能用切牌那套
------------------------
:mod:`majsoul_copilot.vision.tiles.segment` 靠牌的橘色斜面切牌,對自家手牌與
副露很準,但**對牌河完全無效**:牌河是 6x3 的網格,前排會把後排的斜面遮掉。
實測自家牌河三列的斜面覆蓋率分別只有 0.05~0.11、0.03~0.07,只有最前排是 0.77。
牌面彼此相連,改用亮度或連通元件也切不開(整片牌河會連成單一元件)。

改用固定幾何,而不是逐幀偵測
----------------------------
牌河的位置**每一局都一樣** —— 它是牌桌平面上一塊固定的矩形區域。平面上的矩形
經過透視投影後必定還是四邊形,所以只要量一次四個角,單應變換就能算出全部
18 格分別在哪裡,不需要每一幀重新找。這比逐幀偵測穩健得多,也快得多。

同一個變換反過來用,還能把每一格**反扭回正矩形**(:meth:`Grid.warp_cell`),
模板比對就不必應付形變 —— 上家與下家的牌河在畫面上是斜的,不校正的話同一張牌
在不同座位看起來差很多,得為每個座位各備一套模板。

四個角要自己量,不要自動偵測
----------------------------
試過從「非桌面色」的剪影自動擬合四角(凸包多邊形逼近、四邊包絡線 + 穩健迴歸
都試了),四個牌河沒有一個對得準。原因不在演算法:**參考畫面裡沒有任何一個
牌河是滿的 18 格**,右下角那格幾乎不會被填到,於是「右邊界的下段」與「下邊界
的右段」兩條資料同時缺席,任何方法都只是在對不存在的資料外推。自動擬合只能
當標註工具的起始位置用。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from majsoul_copilot.utils.geometry import NormQuad, Rect, Size

__all__ = ["COLS", "ROWS", "Grid"]

#: 牌河的格數。每列 6 張,最多 3 列 —— 每家的打牌數上限就是這樣,不會有第四列。
COLS = 6
ROWS = 3


@dataclass(frozen=True, slots=True)
class Grid:
    """一個牌河的網格幾何。

    Attributes:
        corners: 網格四個角的**像素**座標,浮點數。順序必須是
            ``(col=0,row=0) → (col=COLS,row=0) → (col=COLS,row=ROWS) →
            (col=0,row=ROWS)``,也就是沿著「一列 6 張」的方向先走。

            這個順序決定了 (col, row) 對應到畫面的哪個方向,所以會隨座位不同
            而在畫面上呈現不同的旋轉:自家的第 0 列在畫面上方,對面的第 0 列
            在畫面下方(他的視角是顛倒的),上家與下家則是左右兩側。用「玩家
            自己的視角」定義而不是「畫面方向」,四個座位才能共用同一套索引 ——
            ``cell(0, 0)`` 永遠是那一家的第一張捨牌。
    """

    corners: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError(f"網格需要剛好 4 個角,拿到 {len(self.corners)} 個")

    @classmethod
    def from_norm(cls, quad: NormQuad, container: Rect | Size) -> Grid:
        """由設定檔的正規化四邊形建立。container 通常是牌桌矩形。"""
        return cls(quad.to_pixels(container))

    # ------------------------------------------------------------------ 幾何

    @property
    def _to_image(self) -> np.ndarray:
        """網格座標 (col, row) → 影像像素的單應矩陣。"""
        source = np.array([(0, 0), (COLS, 0), (COLS, ROWS), (0, ROWS)], np.float32)
        return cv2.getPerspectiveTransform(source, np.array(self.corners, np.float32))

    def cell_quad(self, col: int, row: int) -> tuple[tuple[float, float], ...]:
        """某一格的四個角(像素座標)。"""
        self._check(col, row)
        unit = np.array(
            [[(col, row), (col + 1, row), (col + 1, row + 1), (col, row + 1)]], np.float32
        )
        mapped = cv2.perspectiveTransform(unit, self._to_image)[0]
        return tuple((float(x), float(y)) for x, y in mapped)

    def cell_center(self, col: int, row: int) -> tuple[float, float]:
        quad = self.cell_quad(col, row)
        return (
            sum(p[0] for p in quad) / 4.0,
            sum(p[1] for p in quad) / 4.0,
        )

    def cell_bounds(self, col: int, row: int) -> Rect:
        """某一格的軸對齊外接矩形。

        只適合拿來做「這一格有沒有牌」這類粗略判斷。要餵給模板比對請用
        :meth:`warp_cell` —— 外接矩形會把鄰格的邊角一起框進來。
        """
        quad = self.cell_quad(col, row)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return Rect.from_bounds(
            int(np.floor(min(xs))), int(np.floor(min(ys))),
            int(np.ceil(max(xs))), int(np.ceil(max(ys))),
        )

    # ------------------------------------------------------------------ 取像

    def warp_cell(
        self, image: np.ndarray, col: int, row: int, size: tuple[int, int]
    ) -> np.ndarray:
        """把某一格反扭成正矩形取出來。

        這是給模板比對用的入口:輸出的每一格都是同樣尺寸的正矩形,透視形變
        已經消掉,四個座位可以共用同一套牌面模板。

        Args:
            image: 整張擷取影像(corners 是這張影像上的像素座標)。
            size: 輸出的 (寬, 高)。

        Returns:
            ``size`` 大小的 BGR 影像。
        """
        self._check(col, row)
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError(f"輸出尺寸須為正: {width}x{height}")
        destination = np.array(
            [(0, 0), (width, 0), (width, height), (0, height)], np.float32
        )
        matrix = cv2.getPerspectiveTransform(
            np.array(self.cell_quad(col, row), np.float32), destination
        )
        return cv2.warpPerspective(image, matrix, (width, height))

    def warp_all(
        self, image: np.ndarray, size: tuple[int, int]
    ) -> list[list[np.ndarray]]:
        """全部 18 格,``[row][col]`` 索引。"""
        return [[self.warp_cell(image, c, r, size) for c in range(COLS)] for r in range(ROWS)]

    # ------------------------------------------------------------------ 其他

    @staticmethod
    def _check(col: int, row: int) -> None:
        if not (0 <= col < COLS and 0 <= row < ROWS):
            raise IndexError(f"格子 (col={col}, row={row}) 超出 {COLS}x{ROWS} 的範圍")

    def __str__(self) -> str:
        pts = " ".join(f"({x:.0f},{y:.0f})" for x, y in self.corners)
        return f"Grid({COLS}x{ROWS} {pts})"
