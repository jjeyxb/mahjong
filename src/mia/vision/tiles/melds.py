"""讀副露區:攤平成一條帶子,只問每一格**是不是翻開的牌**。

這是副露的第一階段。第二階段(認出每一張是什麼,用來算「壁」)還沒做 ——
見 ``docs/cv_mjai.md`` §3.1 為什麼要分兩階段。

第一階段不必認牌面,所以**格子不必擺正**
---------------------------------------
牌河那邊攤平是為了讓模板比對得到正面、直立、等大的牌面。這裡只問
:func:`~mia.vision.tiles.river.is_face_up` —— 桌布、牌背、翻開的牌,
三樣東西各由一個顏色維度分開,**跟方向無關**。

所以這裡的四個角只要框住那條帶子、而且沿長度方向切得開就夠,
不必像 :data:`~mia.vision.tiles.river.DEFAULT_CORNERS` 那樣把牌轉正。

只數張數換得到什麼
------------------
* **鳴牌的直接證據。** ``RiverWatcher`` 現在是靠「同一個變短的河連兩次
  才採信」猜的(見 ``mia/tracker/rivers.py``),而它之所以要猜,就是因為
  看不到副露區。副露多了三張 = 有人鳴了。
* **副露組數**與「3副露」威脅標籤(現在完全沒算,方向是低估威脅)。

牌山就在旁邊,而牌山也是牌
--------------------------
牌河四周是空桌布,副露不是 —— 它緊貼著牌山。但**牌山是背面朝上**,而
``is_face_up`` 擋掉的正是牌背。所以第一階段躲得過這件事;第二階段要認
牌面的時候躲不過,那時候才需要把邊界切準。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.river import is_face_up

__all__ = [
    "CELL",
    "COLUMNS",
    "DEFAULT_VIEWS",
    "MeldStrip",
    "MeldView",
    "read_melds",
]

#: 帶子切成幾格。
#:
#: 上限是**四組槓** = 16 張。但槓的第四張是疊在第三張**上面**的,不是往旁邊
#: 排,所以畫面上一組槓仍然只佔三張的寬度 —— 四組槓最多 12 格。
#:
#: 取 13:多留一格,寧可末端多一格空的,也不要在真的排滿時切掉最後一組。
COLUMNS = 13

#: 攤平後一格的大小(寬, 高)。只拿來判有無,不比對模板,所以可以比牌河小 ——
#: 小一點整條帶子才便宜。
CELL = (48, 64)


@dataclass(frozen=True, slots=True)
class MeldView:
    """一家副露區的四個角,**相對牌桌矩形**的正規化座標。

    與 :class:`~mia.vision.tiles.river.RiverView` 同一個模型,但角點順序的
    意義不同:這裡是**沿著副露長出去的方向**排的。

    Attributes:
        corners: 四個角 ``((x, y), ...)``,順序是
            「最舊那組的外側、最舊那組的內側、最新那端的內側、最新那端的外側」。

            方向很重要:第 0 格必須是**最先鳴的那一組**。雀魂的副露從自己
            右手邊的桌角往外長,所以上家是由下往上、對家是由左往右 ——
            四家各不相同,只能一家一家量。
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
        """把這一家的副露區攤平成 ``1 x COLUMNS`` 格的帶子。"""
        width, height = cell
        target = np.array(
            [[0, 0], [0, height], [COLUMNS * width, height], [COLUMNS * width, 0]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(self.quad(table), target)
        return cv2.warpPerspective(image, matrix, (COLUMNS * width, height))


@dataclass(frozen=True, slots=True)
class MeldStrip:
    """一家副露區的讀數。

    Attributes:
        occupied: 每一格是不是翻開的牌,依序(第 0 格是最先鳴的那一組)。
    """

    occupied: tuple[bool, ...]

    @property
    def filled(self) -> tuple[int, ...]:
        return tuple(i for i, v in enumerate(self.occupied) if v)

    @property
    def count(self) -> int:
        return len(self.filled)

    def is_plausible(self) -> bool:
        """有牌的格子是不是從頭連續排到某一格。

        副露**從桌角往外長,中間不會有洞** —— 與牌河同一個檢查、同一個理由。
        出現「有、沒有、有」就代表攤平歪了,或是抓在動畫中間、立繪蓋住了
        一半。整家丟掉比較安全。

        這個檢查在副露上比在牌河上更重要:牌河有十幾張,少看到一張只是少
        一張現物;副露只有 0~4 組,少看到一組就是整組不見。
        """
        return self.filled == tuple(range(self.count))


#: 量好的副露區角點。**四家只有兩家。**
#:
#: 量測條件:``data/recordings/20261024-123746``(2560x1614 擷取、牌桌矩形
#: 2560x1440)。先目測一組副露的兩端,依「一張 = 幾分之一」外插到 13 格,
#: 再拿封包真值用座標下降搜。
#:
#: 整份錄影逐幀對分(穩定秒數 1.0):
#:
#: ==========  ========  ========  ==========  ==========
#: 家          可評分    張數對    整家棄權    **讀錯**
#: ==========  ========  ========  ==========  ==========
#: 上家        3705      95.92%    3.83%       0.24%
#: 對家        3732      95.26%    4.42%       0.32%
#: ==========  ========  ========  ==========  ==========
#:
#: **讀錯的那幾次全部是「讀太多」,兩家都是 0.00% 讀太少。** 多算副露會
#: 高估威脅,方向符合 ``docs/cv_mjai.md`` 第七節的要求。
#:
#: 搜角點值得記的兩件事
#: --------------------
#: 1. **框的是「可能的全長」,不是「現在有多長」。** 第一次框住現有那 9 張
#:    的兩端,13 個格子全部落在那 9 張牌裡,讀出來 13/13。
#: 2. **目標函數要分開看「0 張」與「有副露」。** 不分的話退化解會贏 ——
#:    一個框到桌布上的四邊形永遠讀 0,而大部分幀的真值就是 0。
#:
#: 這裡比牌河好搜,因為**有真值**:封包逐秒都說得出每一家有幾張副露牌。
#: ``river.DEFAULT_CORNERS`` 當初是在沒有真值的情況下搜的。搜出來的結果
#: 只比目測好 0.2 個百分點(上家 95.73% → 95.92%),所以目測的起點本來
#: 就不差 —— 搜的價值在於**不必再目測另外幾家**。
#:
#: 缺的兩家
#: --------
#: * **下家**:這份素材 0 次鳴牌,一張樣本都沒有。**搜不了,只能補素材。**
#: * **自己**:副露緊貼著自己的手牌,而手牌也是**正面朝上**的 ——
#:   ``is_face_up`` 分不開,搜出來的分數接近 0(連「沒有副露」的幀都讀錯,
#:   因為 13 格的帶子一路穿過手牌)。牌山那個麻煩躲得過(牌山是背面),
#:   這個躲不過。
#:
#:   **但自己這一家正好是最不需要的。** 放銃分析從不對自己算危險,所以
#:   自己的副露只餵「壁」,而壁用的 ``TableTracker.visible`` 本來就含自己的
#:   手牌。進了副露的牌是從手牌出去的,少算會讓 ``remaining`` 偏高,於是
#:   「四張全見」的排除不觸發、那個待牌仍然算危險 —— **方向是保守的**。
DEFAULT_VIEWS: dict[str, MeldView] = {
    "toimen": MeldView(
        ((0.2035, 0.0278), (0.2035, 0.0868), (0.5645, 0.0868), (0.5645, 0.0278))
    ),
    "kamicha": MeldView(
        ((0.0371, 0.8917), (0.0734, 0.8868), (0.1829, 0.2142), (0.1499, 0.2400))
    ),
}


def read_melds(
    image: np.ndarray, table: Rect, view: MeldView, *, cell: tuple[int, int] = CELL
) -> MeldStrip:
    """讀一家的副露區。

    Args:
        image: 整張擷取影像(不是 ROI 裁過的)。
        table: 牌桌矩形。
        view: 這一家的四個角。
        cell: 攤平後一格的大小。
    """
    flat = view.rectify(image, table, cell=cell)
    width, height = cell
    return MeldStrip(
        tuple(
            is_face_up(flat[0:height, i * width : (i + 1) * width])
            for i in range(COLUMNS)
        )
    )
