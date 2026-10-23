"""讀出牌河:哪一格有牌、那一格在畫面上的哪裡。

與 :mod:`~mia.vision.tiles.hand` 是同一類的東西 —— 位置是**固定的**,不需要
偵測,查表就好。差別在牌河是二維的,而且**每一排的幾何都不一樣**。

為什麼每一排要各一組常數
------------------------
牌河平躺在桌上,而鏡頭是斜的。離鏡頭越遠的那一排看起來越窄:實測 own 的
三排間距是 ``82.0 / 84.5 / 85.6``(ROI 寬 514),遠排比近排窄 4%。

4% 聽起來很小,但一排有六格,累積到第六格就是 20 px —— 比牌縫還寬。用同一組
常數切三排,最遠那排會整排偏掉,而偏掉的格子切到的是隔壁那張牌的一半。
實測症狀是「只有一排讀得出來」,而那一排是碰巧對上的那排。

**只有縮放,沒有梯形變形。** 一度以為近排的牌被壓得比遠排扁(那樣就得上
homography),實測牌面長寬比遠排 1.17、近排 1.18 —— 一樣。透視只體現在整體
大小上,不在單張牌的內部,所以每排一組縮放就夠了。

為什麼不量上下緣,而是給一條夠寬的搜尋帶
----------------------------------------
``cv2.matchTemplate`` 本來就會在搜尋區裡平移找最佳位置,所以垂直方向只要給
一條**比牌高一些**的帶子,它自己會對上去。反過來把帶子切得剛好貼著牌,等於
把平移餘裕拿掉 —— 實測同一張「北」在寬鬆格子裡是 0.845,貼著切之後掉到
0.403。切得準不會比較好,切得準只是比較難。

牌河最多 18 格,但不是 18 張
---------------------------
6 格 x 3 排。被別家鳴走的牌會從河裡**消失**,後面的往前遞補,所以第 k 格裝的
不一定是第 k 張捨牌。拿封包對答案時要用
:attr:`~mia.mjai.table.Player.visible_river`,不是 ``river``。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect

__all__ = [
    "COLUMNS",
    "DEFAULT_RIVER",
    "ROTATION",
    "ROWS",
    "SEATS",
    "RiverLayout",
    "RowLayout",
    "read_river",
    "upright",
]

#: 四家,與 ``data/roi_ref/roi.yaml`` 的鍵一致。順序是**出牌順序** ——
#: 自己、下家、對家、上家。
SEATS = ("own", "shimocha", "toimen", "kamicha")

#: 把每一家的牌河轉成與自己同向所需的旋轉。``None`` 表示本來就是正的。
#:
#: 轉**影像**而不是轉模板:一家只要轉一次,轉模板要轉 37 次。而且轉完之後
#: 四家共用同一套「由遠而近、由左而右」的索引,上層不必各家各寫一遍。
#:
#: 四個方向都目視確認過 —— 轉完之後牌面上的字是正的。另外用模板分數交叉
#: 驗過:每一家在正確的方向上分數最高,其餘三個方向明顯較差。
ROTATION: dict[str, int | None] = {
    "own": None,
    "shimocha": cv2.ROTATE_90_CLOCKWISE,
    "toimen": cv2.ROTATE_180,
    "kamicha": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

#: 牌河的格數。雀魂固定 6 格一排、三排。
COLUMNS = 6
ROWS = 3

#: 判定一格有牌的中位亮度門檻。牌面是象牙色、桌布是深藍,實測真牌 177~223、
#: 桌布 50~62 —— 中間空得很開,門檻取 110 離兩邊都遠。
#:
#: 用中位數而不是平均:一格裡若有一半是牌、一半是桌布(切得略寬時會發生),
#: 平均會落在中間變得不好判,中位數則跟著佔多數的那一邊走。
_OCCUPIED_MEDIAN = 110.0


@dataclass(frozen=True, slots=True)
class RowLayout:
    """一排在 ROI 內的幾何,全部是**相對 ROI 尺寸**的比例。

    Attributes:
        origin: 第 0 格的左緣,相對 ROI 寬。
        pitch: 相鄰兩格左緣的距離,相對 ROI 寬。
        top: 搜尋帶的上緣,相對 ROI 高。
        height: 搜尋帶的高度,相對 ROI 高。比一張牌高 —— 見模組說明。
    """

    origin: float
    pitch: float
    top: float
    height: float

    def cell(self, size: tuple[int, int], column: int) -> Rect:
        """第 ``column`` 格的搜尋區。``size`` 是 ``(ROI 寬, ROI 高)``。"""
        width, height = size
        x = round((self.origin + column * self.pitch) * width)
        return Rect(
            x,
            round(self.top * height),
            round(self.pitch * width),
            round(self.height * height),
        )


@dataclass(frozen=True, slots=True)
class RiverLayout:
    """一家牌河的三排幾何。

    Attributes:
        rows: 由遠而近三排。順序與畫面上由上而下一致,也與捨牌順序一致 ——
            牌河是從最遠那排開始往自己這邊填的。
        face: 牌面佔一格的比例 ``(寬, 高)``。模板會縮到這個大小再比對。
            寬實測三排都是 0.73~0.74,所以共用一個值;高是相對**搜尋帶**的,
            而搜尋帶比牌高,因此這個數字看起來偏小是正常的。
    """

    rows: tuple[RowLayout, RowLayout, RowLayout]
    face: tuple[float, float]

    def cell(self, size: tuple[int, int], index: int) -> Rect:
        """第 ``index`` 格(0~17,由遠而近、由左而右)的搜尋區。"""
        if not 0 <= index < COLUMNS * ROWS:
            raise ValueError(f"牌河只有 {COLUMNS * ROWS} 格,拿到 index={index}")
        return self.rows[index // COLUMNS].cell(size, index % COLUMNS)

    def face_size(self, size: tuple[int, int], index: int) -> tuple[int, int]:
        """第 ``index`` 格該用多大的模板。``(寬, 高)``,給 ``cv2.resize`` 用。"""
        cell = self.cell(size, index)
        return round(self.face[0] * cell.width), round(self.face[1] * cell.height)


#: own 牌河的量測值。
#:
#: 量測條件:``data/recordings/20260801-110639``(2536x1608 擷取、牌桌矩形
#: 2536x1430),``own.river`` ROI 裁出 514x288。起點與間距由牌縫的中心位置
#: 量出(實測一排六格的間距差值 87/84/83/85/84/84,殘差 < 2px);搜尋帶的
#: 上緣與牌面比例以「整排同時變得有信心」為目標擬合。
DEFAULT_RIVER = RiverLayout(
    rows=(
        RowLayout(origin=0.0156, pitch=0.1595, top=0.000, height=0.389),
        RowLayout(origin=0.0078, pitch=0.1644, top=0.243, height=0.389),
        RowLayout(origin=0.0019, pitch=0.1665, top=0.594, height=0.389),
    ),
    face=(0.73, 0.63),
)


@dataclass(frozen=True, slots=True)
class River:
    """一家牌河讀出來的結果。

    Attributes:
        cells: 每一格的搜尋區,由遠而近、由左而右。長度固定 18 —— 空格也留著,
            因為「第幾格」要對得上捨牌順序。
        occupied: 每一格有沒有牌。
    """

    cells: tuple[Rect, ...]
    occupied: tuple[bool, ...]

    @property
    def count(self) -> int:
        """看得見幾張牌。"""
        return sum(self.occupied)

    @property
    def filled(self) -> tuple[int, ...]:
        """有牌的那幾格的索引。"""
        return tuple(i for i, on in enumerate(self.occupied) if on)

    def is_plausible(self) -> bool:
        """有牌的格子是不是從頭連續排到某一格。

        牌河是**從第 0 格依序往後填**的,中間不會有洞 —— 被鳴走的牌後面會
        往前遞補。所以出現「有、沒有、有」就代表切歪了或是抓在動畫中間,
        整家丟掉比較安全。

        這與手牌的張數檢查是同一個用意,而那個檢查實測攔下了八成的靜默錯誤。
        """
        return self.filled == tuple(range(self.count))


def upright(image: np.ndarray, seat: str) -> np.ndarray:
    """把一家的牌河轉成與自己同向。

    Args:
        image: 該家 ``river`` ROI 裁出來的 BGR 影像。
        seat: :data:`SEATS` 之一。

    Returns:
        轉正後的影像。``own`` 回傳的是**原物件**,不是複本 —— 不轉就沒有
        複製的理由,而呼叫端本來就不該改它。
    """
    if seat not in ROTATION:
        raise ValueError(f"不認得的座位 {seat!r},要是 {SEATS} 之一")
    rotation = ROTATION[seat]
    return image if rotation is None else cv2.rotate(image, rotation)


def read_river(image: np.ndarray, layout: RiverLayout = DEFAULT_RIVER) -> River:
    """切出一家牌河的 18 格,並判斷每一格有沒有牌。

    Args:
        image: 該家 ``river`` ROI 裁出來的 BGR 影像。**已經轉正** —— 側面與
            對家的牌河要先旋轉成與自己同向,見 :mod:`mia.vision.tiles.seats`。
        layout: 幾何。預設是 own 的量測值。

    Returns:
        每一格的位置與有無。**不判斷牌面是什麼** —— 那是
        :func:`~mia.vision.tiles.classify.classify` 的工作。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={image.shape}")

    height, width = image.shape[:2]
    size = (width, height)
    cells: list[Rect] = []
    occupied: list[bool] = []
    for index in range(COLUMNS * ROWS):
        cell = layout.cell(size, index)
        cells.append(cell)
        occupied.append(_occupied(image, cell))
    return River(cells=tuple(cells), occupied=tuple(occupied))


def _occupied(image: np.ndarray, cell: Rect) -> bool:
    """這一格是牌還是桌布。見 :data:`_OCCUPIED_MEDIAN`。"""
    patch = image[cell.as_slice()]
    if patch.size == 0:
        return False
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return bool(np.median(grey) >= _OCCUPIED_MEDIAN)
