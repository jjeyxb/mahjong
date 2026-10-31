"""讀副露區:先量長度(第一階段),鳴牌確認時再認牌面(第二階段)。

第一階段只問每一欄**是不是牌**,用來數組數;第二階段 :func:`read_meld_faces`
認出新長出來那一組是哪幾張,用來判役牌、染手 —— 見 ``docs/cv_mjai.md``
§3.1 為什麼要分兩階段、§3.8 第二階段怎麼做。

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
``is_face_up`` 擋掉的正是牌背。所以第一階段躲得過這件事。第二階段躲不過
(框得到牌山、自己的手牌、名牌),所以它**只掃第一階段量到的那一段** ——
掃整條帶子的雛型,自家視角把手牌裡的白認成了副露。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet, ink_masks
from mia.vision.tiles.river import CELL as RIVER_CELL
from mia.vision.tiles.river import COLUMNS as RIVER_COLUMNS
from mia.vision.tiles.river import (
    CONFIDENT,
    DEFAULT_CORNERS,
    ROWS,
    classify_river_tile,
    face_coverage,
    is_face_up,
    tile_masks,
)

__all__ = [
    "CELL",
    "COLUMNS",
    "DEFAULT_VIEWS",
    "EXTENT",
    "GROUP_LENGTH",
    "MeldFaces",
    "MeldStrip",
    "MeldView",
    "occupied_length",
    "read_meld_faces",
    "read_melds",
]

#: 帶子切成幾格。
#:
#: 上限是**四組槓** = 16 張。原本以為槓的第四張疊在上面、一組槓只佔三張寬,
#: **牌譜實測只有加槓是這樣**(疊在橫放那張上面);暗槓是四張並排,大明槓是
#: 三直一橫。四組大明槓約 17 張直牌寬,13 格切不完。見 ``docs/cv_mjai.md`` §3.7。
#:
#: 暫時維持 13:現有的兩家角點是照 13 格搜的,改格數要連角點一起重搜。
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
        self,
        image: np.ndarray,
        table: Rect,
        *,
        cell: tuple[int, int] = CELL,
        columns: int = COLUMNS,
    ) -> np.ndarray:
        """把這一家的副露區攤平成 ``1 x columns`` 格的帶子。

        ``columns`` 大於 :data:`COLUMNS` 時,沿長度方向把最新那一端**外插**
        出去。角點是照 13 格量的,外插是線性的,遠端會因為透視而略為偏短 ——
        量長度只要單調、穩定,不要求每一格等於一張牌。
        """
        width, height = cell
        matrix = self.matrix(table, cell=cell, columns=columns)
        return cv2.warpPerspective(image, matrix, (columns * width, height))

    def matrix(
        self, table: Rect, *, cell: tuple[int, int] = CELL, columns: int = COLUMNS
    ) -> np.ndarray:
        """原圖 → 帶子的透視變換。:meth:`rectify` 用它,認牌面時要反過來用。"""
        width, height = cell
        quad = self.quad(table)
        if columns != COLUMNS:
            scale = columns / COLUMNS
            quad = quad.copy()
            quad[2] = quad[1] + (quad[2] - quad[1]) * scale
            quad[3] = quad[0] + (quad[3] - quad[0]) * scale
        target = np.array(
            [[0, 0], [0, height], [columns * width, height], [columns * width, 0]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(quad, target)


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


#: 量好的副露區角點。**四家都有了**(2026-10-30,見最後一段)。
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
#:
#: 補齊的兩家(2026-10-30)
#: ----------------------
#: 牌譜逐步截圖(``tools/paipu.py``)換視角補上了素材,``docs/cv_mjai.md`` §3.7。
#: 這兩家只用 :func:`occupied_length` 驗過,沒有走 :func:`read_melds` 的格子:
#:
#: * **下家**:目測起點、以格子張數搜過一輪,再用長度驗。
#: * **自己**:「跟手牌分不開」是錯的 —— 那是當初的帶子太長、一路穿進手牌。
#:   從最舊那組(右下角)量起、量到第一個空隙,手牌在空隙另一邊。
DEFAULT_VIEWS: dict[str, MeldView] = {
    "own": MeldView(
        ((0.9570, 0.9440), (0.9570, 0.8610), (0.4490, 0.8610), (0.4490, 0.9440))
    ),
    "shimocha": MeldView(
        ((0.8250, 0.0550), (0.7750, 0.0450), (0.8350, 0.4700), (0.9000, 0.4800))
    ),
    "toimen": MeldView(
        ((0.2035, 0.0278), (0.2035, 0.0868), (0.5645, 0.0868), (0.5645, 0.0278))
    ),
    "kamicha": MeldView(
        ((0.0371, 0.8917), (0.0734, 0.8868), (0.1829, 0.2142), (0.1499, 0.2400))
    ),
}

#: 每一家「一組吃 / 碰」在 :func:`occupied_length` 裡有多長。
#:
#: 各家不同,是因為角點框的比例不同(一格不等於一張牌)。量法:牌譜素材每一張
#: 有副露的圖,長度 ÷ 理論組數(吃 / 碰 / 加槓 1、暗槓 1.2、大明槓 1.3)取中位數。
#: 用同一個組長,每一張都落在理論值的 ±0.3 組長以內:
#:
#: ======  ======  ==========  ==========
#: 家      組長    0 張        有副露
#: ======  ======  ==========  ==========
#: 自己    3.33    254/254     62/62
#: 下家    3.80    349/349     133/133
#: 對家    3.01    326/326     156/156
#: 上家    2.60    385/385     97/97
#: ======  ======  ==========  ==========
#:
#: 兩份牌譜(四槓子、大明槓),共七個視角。下家的大明槓那一份錄到一半中斷,
#: 還沒驗;大明槓只在對家驗過(增量 1.29 組長,理論 1.3)。
GROUP_LENGTH: dict[str, float] = {
    "own": 3.33,
    "shimocha": 3.80,
    "toimen": 3.01,
    "kamicha": 2.60,
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


#: 量長度時帶子延伸到幾格。
#:
#: 四組大明槓約 4 x (3 + 1.33) = 17.3 張寬(``docs/cv_mjai.md`` §3.7),
#: 13 格裝不下。取 20 多留一點:長度是從起點量到**第一個空隙**為止,
#: 帶子長出去碰到桌布不會多算。
EXTENT = 20

#: 一欄裡「是牌」(正面或背面)的像素超過這個比例,這一欄就算有牌。
#:
#: 不取 0.5:橫放的牌比直的矮,在帶子裡只佔上半部;加槓疊上去的那張在
#: 帶子外面。實測三個位置用 0.35 量到的長度,同一個狀態裡完全不跳。
_COLUMN_OCCUPIED = 0.35

#: 允許的空隙(帶子上的像素,一格寬 :data:`CELL` [0] = 48)。組與組之間有一條
#: 細縫,不能把它當成結尾;但真正的結尾之後是一大片桌布。
_GAP = 10


def occupied_length(
    image: np.ndarray,
    table: Rect,
    view: MeldView,
    *,
    cell: tuple[int, int] = CELL,
    extent: int = EXTENT,
) -> float:
    """副露區從起點連續有牌多長,單位是**這個視角的一格**。

    為什麼不數格子
    --------------
    :func:`read_melds` 把帶子切成等寬的格子,每格問一次 ``is_face_up``。
    副露一多就撐不住(``docs/cv_mjai.md`` §3.7,四槓子實測):

    * 橫放的牌比直的寬,一組碰約 3.33 張寬,格子越切越偏,某一格落在縫上
      讀成空的 → 不連續 → 整家棄權。
    * 暗槓兩端是**背面**,``is_face_up`` 擋的正是背面 → 中間一定有洞。

    這裡改成逐欄看「正面或背面」的比例,從起點量到第一個空隙。三個位置在
    四槓子那場的每一個狀態裡長度都**完全不跳**,而每多一組加固定的量
    (暗槓 ÷ 碰 = 1.13~1.21,理論 4 / 3.33 = 1.2)。

    單位是攤平後的一格,不是一張牌 —— 各視角的角點框的比例不同,換算成
    「幾組」要各用各的組長。

    Returns:
        長度,以格為單位。0 表示沒有副露。
    """
    flat = view.rectify(image, table, cell=cell, columns=extent)
    face, back = tile_masks(flat)
    occupied = (face | back).mean(axis=0) > _COLUMN_OCCUPIED
    end = miss = 0
    for x, filled in enumerate(occupied):
        if filled:
            end, miss = x + 1, 0
            continue
        if end == 0 and x < cell[0]:
            # 還沒碰到第一張牌。角點不會剛好壓在第一張牌的邊上(自己那家實測
            # 起點外多了 14 px 桌布),開頭讓一格,否則一開始就當成結尾。
            continue
        miss += 1
        if miss > _GAP:
            break
    return end / cell[0]


# ---------------------------------------------------------------------- 牌面

#: 掃描用的框(寬, 高),在**牌河平面**上的像素。比模板大一圈,對齊交給
#: :func:`cv2.matchTemplate`。
_WINDOW = (100, 140)

#: 框的步距。框比最大的模板寬 36、高 48,步距小於這個就不會漏掉位置;
#: 實測 15 x 20 與 25 x 40 認出來的完全一樣,後者快三倍以上。
_STEP = (25, 40)

#: 模板尺寸。副露比牌河離鏡頭遠一點、角度也不同,上家又更小一號:四槓子
#: 那場上家的字牌要 0.82 ~ 0.9 倍才認得到。
_SIZES = ((64, 92), (58, 83), (52, 75))

#: 白的分數門檻,依座位。其他牌照牌河的 :data:`CONFIDENT`。
#:
#: 白沒有圖案,框到牌的側面、頂面也湊得出分數:下家 0.6 ~ 0.65,對家(看得到
#: 牌的頂面)到 0.76。真的白在自家、對家、下家都在 0.85 以上,但 Steam 實戰
#: 的上家(最小最糊)只有 0.80 —— 跟對家的假白分不開,只能各用各的。
_HAKU_SCORE = {"own": 0.85, "shimocha": 0.85, "toimen": 0.85, "kamicha": 0.75}

#: 框中央至少要有這麼多是牌面,才拿去比模板。擋掉框在桌布上的。
_FACE_MIN = 0.6

#: 同一個位置只有一張牌:兩個命中的框距離小於這個(x, y),只留分數高的。
#: 擋的是跨在兩張牌之間的框(實測 6 索旁邊多出 4 索,0.62 對 0.69)。
_SAME_PLACE = (45, 60)

#: 同一種牌的兩個命中相距多遠(牌河平面的 x)才算**兩張**。橫放的牌比直的
#: 寬,同一張會被相鄰兩框各認一次:兩場牌譜裡這種重複最遠 50;碰的兩張
#: 不同的牌最近 100(直的一張約 90 寬)。
_APART = 100

#: 帶子在長度以外、內外兩側各放寬幾格(副露那一格)。框中心落在範圍內就掃。
_PAD = 0.5

#: 找赤五的框(寬, 高),牌河平面的像素。一張直放的牌面大小;橫放的把兩邊對調。
_RED_WINDOW = (60, 84)

#: 找赤五的框步距。
_RED_STEP = 4

#: 框裡至少這麼多是紙,才算框在牌面上。
_RED_PAPER = 0.45

#: 框裡至少這麼多像素是墨水。擋掉只框到牌邊一角的。
_RED_INK_PIXELS = 300

#: 一組副露的 :func:`_red_score` 到多少算有赤五,依花色。
#:
#: 兩份素材、七個視角共 78 組副露:真的赤五 4 組(5 索 3 組、5 萬 1 組)0.79 ~ 0.88,
#: 其他的萬子最高 0.75、筒子 0.51、索子 0.46。5 筒與上家、對家沒有真的樣本:把真實
#: 畫面裡的普通五改成紅色,最低 0.79。``docs/cv_mjai.md`` §4.4。
_RED_MELD = {"m": 0.77, "p": 0.65, "s": 0.62}


def read_meld_faces(
    image: np.ndarray,
    table: Rect,
    seat: str,
    templates: TemplateSet,
    *,
    start: float,
    end: float,
    open_end: bool = True,
) -> MeldFaces:
    """認出副露區 ``start`` 到 ``end`` 格之間的牌面(雀魂記法,赤五當普通五)。

    ``start`` / ``end`` 是 :func:`occupied_length` 的單位。只認**新長出來的那
    一段**:舊的組已經認過,而且掃的範圍越短越快。``open_end`` 表示 ``end``
    後面沒有別的牌,可以往外放寬半格;後面緊接著另一組時要關掉。

    為什麼借用牌河的攤平
    --------------------
    第一階段的帶子只框長度方向,牌是斜的、歪的,不能比模板。而**同一家的
    牌河與副露在同一個桌面上**,牌河的透視變換本來就是為了把牌轉正、
    放大到模板的大小 —— 拿它攤平副露區,牌面一樣是正的,只差一點遠近。
    直接攤平帶子則要每家另外找縮放,上家還有剪切,試過不行。

    橫放的那張(吃 / 碰進來的)轉 90 度再比。

    Returns:
        認得出來的牌。認不出的就不在裡面 —— 上層要把它當成未知,不要猜。
    """
    if end <= start:
        return MeldFaces(frozenset())
    plane, polygon = _meld_plane(image, table, seat, start, end, open_end=open_end)
    hits = _scan(plane, polygon, seat, templates)
    repeated = {
        a.label
        for a in hits
        for b in hits
        if a.label == b.label and abs(a.x - b.x) >= _APART
    }
    score = _red_score(plane, polygon)
    return MeldFaces(
        frozenset(hit.label for hit in hits),
        frozenset(repeated),
        frozenset(suit for suit, floor in _RED_MELD.items() if score >= floor),
    )


@dataclass(frozen=True, slots=True)
class MeldFaces:
    """一組副露認出來的牌面。

    Attributes:
        tiles: 認得出來的牌(雀魂記法)。
        repeated: 其中在**兩個不同位置**都認到的。吃不會有兩張一樣的牌,
            所以這代表碰或槓 —— 只認出一種牌時,靠它分得出「碰 6 萬」與
            「吃,其中一張是 6 萬」。
        red_suits: 這組的牌如果是這幾種花色,裡面的五是赤五。這裡只看顏色,不知道
            這組是什麼花色;牌面常常認不全,花色要上層從牌河與規則推。
    """

    tiles: frozenset[str]
    repeated: frozenset[str] = frozenset()
    red_suits: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Hit:
    """一個認得出來的位置(牌河平面上的框左上角)。"""

    score: float
    x: int
    y: int
    label: str


def _meld_plane(
    image: np.ndarray,
    table: Rect,
    seat: str,
    start: float,
    end: float,
    *,
    open_end: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """副露那一段攤到牌河平面上。回傳影像,與要掃的範圍(影像上的四邊形)。"""
    width, height = CELL
    river_width, river_height = RIVER_CELL
    target = np.array(
        [
            [0, 0],
            [RIVER_COLUMNS * river_width, 0],
            [RIVER_COLUMNS * river_width, ROWS * river_height],
            [0, ROWS * river_height],
        ],
        dtype=np.float32,
    )
    to_plane = cv2.getPerspectiveTransform(DEFAULT_CORNERS[seat].quad(table), target)
    from_strip = np.linalg.inv(DEFAULT_VIEWS[seat].matrix(table, columns=EXTENT))
    # 起點不往前放寬:前一組最後那張牌的中心就在 start 前半格,放寬會把它
    # 認進這一組。第一組前面沒有東西,照放。
    head = start if start > 0 else -_PAD
    tail = end + _PAD if open_end else end
    strip = np.array(
        [
            [head * width, -_PAD * height],
            [head * width, (1 + _PAD) * height],
            [tail * width, (1 + _PAD) * height],
            [tail * width, -_PAD * height],
        ],
        dtype=np.float32,
    )
    polygon = cv2.perspectiveTransform(strip[None], to_plane @ from_strip)[0]
    margin = np.array(_WINDOW, dtype=np.float32)
    left, top = polygon.min(axis=0) - margin
    right, bottom = polygon.max(axis=0) + margin
    shift = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
    size = (int(right - left), int(bottom - top))
    plane = cv2.warpPerspective(image, shift @ to_plane, size)
    return plane, (polygon - (left, top)).astype(np.float32)


def _scan(
    plane: np.ndarray, polygon: np.ndarray, seat: str, templates: TemplateSet
) -> list[_Hit]:
    """:func:`read_meld_faces` 的本體:每個位置一個命中,分數高的在前。"""
    window_width, window_height = _WINDOW
    hits: list[_Hit] = []
    for x in range(0, plane.shape[1] - window_width, _STEP[0]):
        for y in range(0, plane.shape[0] - window_height, _STEP[1]):
            centre = (x + window_width / 2, y + window_height / 2)
            if cv2.pointPolygonTest(polygon, centre, False) < 0:
                continue
            for window in _windows(plane, x, y):
                label, score = _classify(window, templates, haku=_HAKU_SCORE[seat])
                if label is not None:
                    hits.append(_Hit(score, x, y, label))
    kept: list[_Hit] = []
    near_x, near_y = _SAME_PLACE
    for hit in sorted(hits, key=lambda h: h.score, reverse=True):
        if all(abs(hit.x - k.x) > near_x or abs(hit.y - k.y) > near_y for k in kept):
            kept.append(hit)
    return kept


def _red_score(plane: np.ndarray, polygon: np.ndarray) -> float:
    """這組裡最像一張整張紅的牌的那一框,0~1。

    赤五很少被模板認到(牌面只有這幾像素,紅與普通又只差顏色),但顏色不用認牌:
    掃一張牌大的框,問墨水有多少是紅的。

    **框的兩半都要有紅。** 萬字本來就是紅的,框往下滑到只蓋住一張牌的下半部時,
    一萬到九萬的紅墨都能到 0.87。赤五上下都紅,普通萬子上半是黑的。
    """
    ink, red = ink_masks(plane)
    width, height = _RED_WINDOW
    best = 0.0
    for wide, tall in ((width, height), (height, width)):
        for x in range(0, plane.shape[1] - wide, _RED_STEP):
            for y in range(0, plane.shape[0] - tall, _RED_STEP):
                centre = (x + wide / 2, y + tall / 2)
                if cv2.pointPolygonTest(polygon, centre, False) < 0:
                    continue
                inked = int(np.count_nonzero(ink[y : y + tall, x : x + wide]))
                if inked < _RED_INK_PIXELS or 1 - inked / (wide * tall) < _RED_PAPER:
                    continue
                reds = red[y : y + tall, x : x + wide]
                total = int(np.count_nonzero(reds))
                half = reds[: tall // 2] if tall > wide else reds[:, : wide // 2]
                first = int(np.count_nonzero(half))
                if total:
                    balance = 2 * min(first, total - first) / total
                    best = max(best, total / inked * balance)
    return best


def _windows(plane: np.ndarray, x: int, y: int) -> list[np.ndarray]:
    """``(x, y)`` 那一框:直的一張,加上同一個中心、轉 90 度的兩張。"""
    width, height = _WINDOW
    out = []
    upright = plane[y : y + height, x : x + width]
    if upright.shape[:2] == (height, width):
        out.append(upright)
    pad = (height - width) // 2
    wide = plane[y + pad : y + height - pad, x - pad : x + width + pad] if x >= pad else plane[:0]
    if wide.shape[:2] == (width, height):
        out.append(cv2.rotate(wide, cv2.ROTATE_90_CLOCKWISE))
        out.append(cv2.rotate(wide, cv2.ROTATE_90_COUNTERCLOCKWISE))
    return out


def _classify(
    window: np.ndarray, templates: TemplateSet, *, haku: float
) -> tuple[str | None, float]:
    """這一框是哪張牌;沒把握回 ``(None, 0)``。"""
    width, height = _WINDOW
    centre = window[height // 4 : height - height // 4, width // 4 : width - width // 4]
    if face_coverage(centre) < _FACE_MIN:
        return None, 0.0
    best: tuple[str | None, float] = (None, 0.0)
    for size in _SIZES:
        match = classify_river_tile(window, templates, size=size)
        floor = haku if match.label == "5z" else CONFIDENT[0]
        if match.score >= max(floor, best[1]) and match.margin >= CONFIDENT[1]:
            best = (match.label, match.score)
    return best
