"""讀出四家的牌河:攤平成俯視圖,再切成 6x3 的網格。

與 :mod:`~mia.vision.tiles.hand` 的差別在於**牌是躺著的**。手牌立在自己面前,
正對鏡頭,所以一組固定比例的槽位就夠;牌河平躺在桌上,而鏡頭是斜的,四家
各在桌子的一邊 —— 同一張牌在上家與下家的河裡,在畫面上是兩個不同的形狀。

為什麼用 homography
-------------------
牌河的牌全部平躺,而且每張一樣厚,所以**四家的牌面落在同一個平面上**。
一個平面在針孔相機下的成像,正是 homography 的定義 —— 它不是「夠用的近似」,
它就是正確的模型。攤平之後每一格都是正面、直立、等大的牌面,
:func:`~mia.vision.tiles.classify.classify` 的模板一張都不用改。

旋轉也一併吃掉了:四個角點的**順序**決定攤平後的方向,所以不需要再對影像
轉 90/180/270 一次。角點依「遠左、遠右、近右、近左」排列,「遠近」是相對
那一家自己 —— 他的第一張捨牌落在遠排左邊。

走過的冤枉路
------------
一度以為近排的牌有梯形變形(單張牌內部的上緣被壓得比下緣扁),那會需要
逐格校正。實測牌面長寬比遠排 1.17、近排 1.18 —— 一樣,**那個變形不存在**。
真正的問題是整個網格的軸與影像的軸對不齊:own 的三排剛好是水平的,所以
「每排一組起點加間距」能走;側面兩家的排是斜的,同樣的模型套上去整排偏掉。

校正是搜出來的,不是量出來的
----------------------------
角點用座標下降搜,目標是「攤平之後有牌的格數 + 有信心的格數」。不用標準答案
—— 正確的攤平會讓整片**同時**變得有信心,錯的攤平是整片都勉強。

兩個目標函數的陷阱都踩過,都寫在 :data:`DEFAULT_CORNERS` 的說明裡。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect

__all__ = [
    "CELL",
    "COLUMNS",
    "CONFIDENT",
    "DEFAULT_CORNERS",
    "FACE",
    "ROWS",
    "SEATS",
    "River",
    "RiverView",
    "face_size",
    "read_river",
]

#: 牌河的格數。雀魂固定 6 格一排、三排。
COLUMNS = 6
ROWS = 3

#: 四家,與 ``data/roi_ref/roi.yaml`` 的鍵一致。順序是**出牌順序** ——
#: 自己、下家、對家、上家。
SEATS = ("own", "shimocha", "toimen", "kamicha")

#: 攤平後一格的像素大小 ``(寬, 高)``。
#:
#: 這個比例不是牌的真實比例,是**量出來最好認的那個**。攤平的目標是讓模板
#: 對得上,不是還原物理尺寸 —— 模板本身就是斜著畫的貼圖,不是正投影。
CELL = (90, 120)

#: 牌面佔一格的比例 ``(寬, 高)``。手牌是 0.84,牌河小得多 —— 一格裡除了牌面
#: 還含著牌縫與相鄰牌的邊緣。套錯會讓模板大一圈,分數整片掉下來,
#: 而症狀看起來像幾何沒校好。
FACE = (0.72, 0.70)

#: 牌河自己的信心門檻 ``(最低分數, 最小差距)``。
#:
#: 手牌用的是 ``classify`` 裡的 0.55 / 0.04,而牌河**不該共用那一組**,
#: 因為兩邊的風險完全不同:
#:
#: * 手牌認錯了使用者一眼就發現 —— 他正在看自己的手牌。實測靜默錯誤 0.018%。
#: * 牌河沒人會去核對。放銃分析憑它算危險度,算錯了沒有任何症狀。
#:
#: 1921 幀的掃描(見 ``tools/eval_rivers.py`` 的門檻掃描):
#:
#: ====================  ============  ==========
#: 分數 / 差距           靜默錯誤      棄權成本
#: ====================  ============  ==========
#: 0.55 / 0.04(手牌)     3.01%         7.40%
#: 0.55 / 0.08           2.50%         10.25%
#: **0.60 / 0.10**       **0.81%**     **27.07%**
#: 0.65 / 0.12           0.72%         39.02%
#: 0.70 / 0.15           0.19%         52.70%
#: ====================  ============  ==========
#:
#: 取 0.60 / 0.10:靜默錯誤降到約四分之一,代價是四分之一的牌變成「不確定」。
#: 這個交換划得來,因為**棄權是可以往安全方向處理的** —— 讀不到的牌當成
#: 危險就好,而誤判成安全會放銃。成本不對稱,門檻就該不對稱。
#:
#: ⚠ 只在一份錄影上掃過。換一份素材要重掃一次再決定要不要改。
CONFIDENT = (0.60, 0.10)

#: 判定一格有牌的中位亮度門檻。牌面是象牙色、桌布是深藍,實測真牌 177~223、
#: 桌布 50~62。用中位數而不是平均:取樣區若有一半是牌、一半是桌布,平均會
#: 落在中間變得不好判,中位數則跟著佔多數的那一邊走。
_OCCUPIED_MEDIAN = 110.0


@dataclass(frozen=True, slots=True)
class RiverView:
    """一家牌河的四個角,**相對牌桌矩形**的正規化座標。

    用正規化座標而不是像素,理由與 ROI 相同:同一組數字要能跟著視窗大小走。

    Attributes:
        corners: 四個角 ``((x, y), ...)``,順序是**遠左、遠右、近右、近左**。
            順序就是方向 —— 換順序等於轉 90 度,所以它不只是「四個點」,
            是「四個有編號的點」。
    """

    corners: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]

    def quad(self, table: Rect) -> np.ndarray:
        """換算成擷取影像上的像素四邊形。"""
        return np.array(
            [[table.x + x * table.width, table.y + y * table.height] for x, y in self.corners],
            dtype=np.float32,
        )

    def rectify(
        self, image: np.ndarray, table: Rect, *, cell: tuple[int, int] = CELL
    ) -> np.ndarray:
        """把這一家的牌河攤平成 ``6 x 3`` 格的俯視圖。

        Args:
            image: 整張擷取影像(不是 ROI 裁過的)。
            table: 牌桌矩形,見 :class:`~mia.calibration.table.Calibration`。
            cell: 攤平後一格的大小。

        Returns:
            ``(3 * cell高, 6 * cell寬)`` 的 BGR 影像,牌面是正的。
        """
        width, height = cell
        target = np.array(
            [[0, 0], [COLUMNS * width, 0], [COLUMNS * width, ROWS * height], [0, ROWS * height]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(self.quad(table), target)
        return cv2.warpPerspective(image, matrix, (COLUMNS * width, ROWS * height))


#: 四家的角點。
#:
#: 量測條件:``data/recordings/20260801-110639`` 幀 12141(2536x1608 擷取、
#: 牌桌矩形 2536x1430),四家牌河 15/17/17/16 張。以座標下降搜出,目標是
#: 「有牌的格數 + 0.9 x 有信心的格數」。
#:
#: 目標函數踩過兩個坑,都留在這裡當紀錄:
#:
#: 1. **只看平均分會讓方框縮小。** 把難認的格子排除在外,平均分就上升了 ——
#:    於是優化器主動漏掉一格,而漏掉的那格讓後面整串索引往前位移一格,
#:    表現出來是「整家幾乎全錯」。所以格數要進目標。
#: 2. **座標下降會卡在偏半格的局部解。** 要挪一整格才會變好,中間那段是變差
#:    的,把步長調小救不了。所以從「原位 + 沿兩個格軸各平移 ±1 格」共九個
#:    起點各跑一次 —— 上家正是靠 ``(-1, 0)`` 那個起點才對上的。
#:
#: ⚠ **只在一個畫布尺寸上驗過。** 角點是相對牌桌矩形的比例,理論上跟著縮放,
#: 但沒有第二個尺寸的素材可以證實。
DEFAULT_CORNERS: dict[str, RiverView] = {
    "own": RiverView(((0.3998, 0.4839), (0.5907, 0.4839), (0.5978, 0.7021), (0.3927, 0.6895))),
    "shimocha": RiverView(((0.5883, 0.4944), (0.5777, 0.2678), (0.7125, 0.2657), (0.7386, 0.4902))),
    "toimen": RiverView(((0.5836, 0.2720), (0.4073, 0.2804), (0.4144, 0.1350), (0.5777, 0.1392))),
    "kamicha": RiverView(((0.4160, 0.2667), (0.4077, 0.4960), (0.2468, 0.5002), (0.2847, 0.2667))),
}


def face_size(cell: tuple[int, int] = CELL) -> tuple[int, int]:
    """模板該縮到多大。傳給 :func:`~mia.vision.tiles.classify.classify` 的 ``size``。"""
    return round(FACE[0] * cell[0]), round(FACE[1] * cell[1])


@dataclass(frozen=True, slots=True)
class River:
    """一家牌河讀出來的結果。

    Attributes:
        cells: 每一格在**攤平後影像**上的矩形,由遠而近、由左而右。長度固定
            18 —— 空格也留著,因為「第幾格」要對得上捨牌順序。
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

        牌河是**從第 0 格依序往後填**的,中間不會有洞 —— 被別家鳴走的牌後面
        會往前遞補。所以出現「有、沒有、有」就代表攤平歪了或是抓在動畫中間,
        整家丟掉比較安全。

        與手牌的張數檢查是同一個用意,而那個檢查實測攔下了八成的靜默錯誤。
        """
        return self.filled == tuple(range(self.count))


def read_river(flat: np.ndarray, *, cell: tuple[int, int] = CELL) -> River:
    """在**攤平後**的影像上切 18 格,並判斷每一格有沒有牌。

    Args:
        flat: :meth:`RiverView.rectify` 的輸出。
        cell: 攤平時用的一格大小。

    Returns:
        每一格的位置與有無。**不判斷牌面是什麼** —— 那是
        :func:`~mia.vision.tiles.classify.classify` 的工作。
    """
    if flat.ndim != 3 or flat.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={flat.shape}")

    width, height = cell
    cells: list[Rect] = []
    occupied: list[bool] = []
    for index in range(COLUMNS * ROWS):
        rect = Rect((index % COLUMNS) * width, (index // COLUMNS) * height, width, height)
        cells.append(rect)
        occupied.append(_occupied(flat, _face_of(rect, cell)))
    return River(cells=tuple(cells), occupied=tuple(occupied))


def _face_of(cell: Rect, size: tuple[int, int]) -> Rect:
    """格子正中央、牌面該在的那一塊。

    **有無判斷只能看這裡,不能看整格。** 一格比一張牌大一圈(:data:`FACE`
    是 0.72 x 0.70,也就是牌面只佔一格的一半),所以一格裡裝著一張牌時,
    另外一半是桌布 —— 整格取中位數會落在牌與桌布之間。

    實測:一局開頭河裡只有一張牌時,那一格整格的中位亮度是 97,而門檻是
    110 —— 判成空的。症狀是「每局開頭的幀全部被拒絕」,而 own 的採用率
    因此只有一半。
    """
    face_width, face_height = face_size(size)
    return Rect(
        cell.x + (cell.width - face_width) // 2,
        cell.y + (cell.height - face_height) // 2,
        face_width,
        face_height,
    )


def _occupied(image: np.ndarray, cell: Rect) -> bool:
    """這一塊是牌還是桌布。見 :data:`_OCCUPIED_MEDIAN`。"""
    patch = image[cell.as_slice()]
    if patch.size == 0:
        return False
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return bool(np.median(grey) >= _OCCUPIED_MEDIAN)
