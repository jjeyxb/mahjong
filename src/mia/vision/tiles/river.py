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

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.classify import Match, TemplateSet, classify

__all__ = [
    "CELL",
    "COLUMNS",
    "CONFIDENT",
    "DEFAULT_CORNERS",
    "FACE",
    "ROWS",
    "SEARCH_ABOVE",
    "SEARCH_BELOW",
    "SEATS",
    "SIDEWAYS_FACE",
    "SIDEWAYS_SHIFT",
    "River",
    "RiverView",
    "classify_river_tile",
    "classify_sideways",
    "face_coverage",
    "face_size",
    "find_sideways",
    "is_face_up",
    "read_river",
    "search_window",
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

#: 認牌用的模板,佔一格的比例 ``(寬, 高)``。即 64 x 92 像素。
#:
#: 原本是 0.72 x 0.70(65 x 84),那是「整體最佳」掃出來的 —— 但掃描時搜尋
#: 範圍就是格子本身,而牌面在格子裡**比格子低**:第一列的牌面頂端在格子
#: 下方約 30 像素處,真正的高度約 92,加起來超出格子。高的模板塞不下,
#: 掃描只好選一個塞得下的矮模板。被壓扁 9% 的模板讓「二」的兩橫間距對不上,
#: 一萬那一橫反而落在中間 —— 二萬在一份素材上**一次都沒認對過**,13.5%
#: 有信心地讀成一萬(``docs/cv_mjai.md`` §3.5)。
#:
#: 改成正確的高度,同時讓搜尋往下延伸(:data:`SEARCH_BELOW`)。兩份素材上
#: 量到的最佳尺寸一致(寬 64,高 92;對家 97~102),而**正確率與靜默錯誤
#: 同時改善** —— 不是門檻上的交換,是原本的幾何設定就錯了。
#:
#: ⚠ 只決定**模板**的大小。「這一格有沒有牌」看的是 :data:`_OCCUPANCY_FACE`,
#: 刻意不跟著動:張數判斷沒有問題,改它只會冒險。
FACE = (0.71, 0.77)

#: 認牌時搜尋窗往格子**下方**多看幾像素(以 :data:`CELL` 的高度為準)。
#:
#: 見 :data:`FACE`。牌面在格子裡的落點每一列不同(第一列約 30、第二列約 12、
#: 第三列 0),所以不能把格線整個往下挪 —— 那會讓第三列反過來被切掉。
#: 窗往下長,讓模板自己找落點,那正是 :func:`cv2.matchTemplate` 的工作。
#:
#: 攤平圖因此比三列高一截(:meth:`RiverView.rectify`)。平面往下自然延伸,
#: 不需要另外處理邊界。
SEARCH_BELOW = 30

#: 認牌時搜尋窗往格子**上方**多看幾像素。
#:
#: 第三列的牌面頂端貼著格子上緣(落點 0),差個幾像素就跑到格子上面去。窗只
#: 往下長的話,模板滑不上去,牌面頂端被切掉,而「二」被切掉上面那一橫就是
#: 「一」—— 校正素材下家第三列有一張二萬,在 49 張不重複影像裡**每一次**都
#: 有信心地讀成一萬(對手的河,所以是假現物)。往上看 8 像素之後同一格
#: 讀成二萬、分數 0.84。
#:
#: 曾經以為是第三列的牌面比較矮,把第三列的模板壓到 80 —— 錯誤是不見了,
#: 但那只是讓它從「有信心的一萬」變成「沒信心的一萬」,棄權而已。
#:
#: 它也讓第三列不再對角點敏感:角點稍微一動,第三列的落點就會變成負的。
#: 上一列的牌在一整格(120 像素)之外,模板對不到它。
SEARCH_ABOVE = 8

#: 「這一格有沒有牌」看的那一塊,佔一格的比例。與 :data:`FACE` 分開的理由
#: 見那裡。數值就是 :data:`FACE` 改之前的值,有無判斷的所有量測都建立在它上面。
_OCCUPANCY_FACE = (0.72, 0.70)

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
#: 0.55 / 0.04(手牌)     0.74%         9.74%
#: 0.55 / 0.08           0.71%         12.97%
#: **0.60 / 0.10**       **0.18%**     **25.38%**
#: 0.65 / 0.12           0.14%         36.49%
#: 0.70 / 0.15           0.14%         50.96%
#: ====================  ============  ==========
#:
#: (這張表是加上 :data:`CHECK_FACE_HEIGHT` 的覆核之後重掃的。覆核之前手牌那組
#: 門檻的靜默錯誤是 3.01%,現行這組是 0.81%。)
#:
#: 取 0.60 / 0.10:靜默錯誤降到四分之一以下,代價是四分之一的牌變成「不確定」。
#: 這個交換划得來,因為**棄權是可以往安全方向處理的** —— 讀不到的牌當成
#: 危險就好,而誤判成安全會放銃。成本不對稱,門檻就該不對稱。
#:
#: ⚠ 只在一份錄影上掃過。換一份素材要重掃一次再決定要不要改。
CONFIDENT = (0.60, 0.10)

#: 橫放的立直宣言牌比直放寬多少(攤平後的像素)。
#:
#: 宣言牌躺著放,量出來佔 110~115px 而直放是 85~88px(一格 90)。多出來的
#: 寬度把**同一列**後面的每一張都往右推這麼多 —— 跨列不會推,因為下一列
#: 從頭開始排。
SIDEWAYS_SHIFT = 27

#: 旋轉模板要贏直放模板多少,才算是候選。
#:
#: 這個數字單獨用是**沒用**的:實測正例最高 +0.122,而正常牌的尾巴衝到
#: +0.270 —— 逐格設門檻一定夾不住。它只負責把候選縮到個位數,真正拍板的是
#: :func:`find_sideways` 最後那一步「挪了之後有沒有變好」。
#:
#: 兩份錄影、4 張宣言牌、313 個候選的量測:
#:
#: ====================  ==========  ==========
#: 候選規則              召回        假陽
#: ====================  ==========  ==========
#: 只看分數差 >= 0       100%        16%
#: 加上「直放沒信心」    100%        6%
#: **再加上挪了變好**    **94.6%**   **0%**
#: ====================  ==========  ==========
_SIDEWAYS_MARGIN = 0.06

#: 比對橫放牌時取的窗寬,以一格為單位。
#:
#: 橫放的牌躺著,寬度超過一格 —— 只取一格會把牌面切掉一截,而被切掉的
#: 那一截正好是旋轉模板要對上的部分。
_SIDEWAYS_WINDOW = 1.4

#: 橫放牌面佔一格的比例 ``(寬, 高)``。對應 :data:`FACE` 的直放版本。
#:
#: **這不是把 :data:`FACE` 的寬高交換。** 第一版就是那樣寫的(0.70 x 0.72),
#: 而掃出來的最佳值是 1.22 x 0.49 —— 寬得多、矮得多。原因是躺下來之後透視
#: 壓縮的軸也跟著換了,所以長寬比不是交換的關係(交換會給 1.29,實際是 2.49)。
#:
#: 代價差很大:四張宣言牌在交換尺寸下的分數是 0.53 附近,**全部過不了
#: :data:`CONFIDENT` 的 0.60**;量出來的尺寸是 0.65~0.81,全部過得了。
#:
#: 掃描結果(兩份錄影、4 張宣言牌,各取三幀;單位是像素,一格 90x120):
#:
#: ================  ==========  ========
#: 牌                最佳尺寸    分數
#: ================  ==========  ========
#: 自己 5m           114x55      0.807
#: 下家 5p           110x59      0.733
#: 對家 5m           106x59      0.794
#: 對家 2m           102x59      0.731(認錯,見下)
#: ================  ==========  ========
#:
#: ⚠ **最佳值跟著座位走** —— 離鏡頭越近越大(自己 114、對家 102)。這裡取的是
#: 中間值 110x59,四家都堪用。要分家給的話得先有更多樣本:現在只有 4 張牌,
#: 分家調就是對單一樣本過擬合。
#:
#: ⚠ 對家那張 2m 在**任何**尺寸下都認成 1m。那是已知的 2m→1m 混淆
#: (靜默錯誤裡最大的一項),與旋轉無關。
SIDEWAYS_FACE = (1.22, 0.49)

#: 判定一格有牌的中位亮度門檻。牌面是象牙色、桌布是深藍,實測真牌 177~223、
#: 桌布 50~62。用中位數而不是平均:取樣區若有一半是牌、一半是桌布,平均會
#: 落在中間變得不好判,中位數則跟著佔多數的那一邊走。
_OCCUPIED_MEDIAN = 110.0

#: 牌背的飽和度下限。這麼飽和的一塊,不會是牌面。
#:
#: 牌背是橘色的,亮度跟牌面一樣高,所以只看亮度分不出來。而河的末端常常會
#: 出現一張牌背 —— 實測上家有 43% 的幀多讀一張,一律在結尾,放大看全是橘色、
#: 沒有任何圖案的牌背。多出來那一張會被硬認成某張牌(分數低、過不了信心門檻),
#: 但**張數已經錯了**,而張數錯會讓那一家整幀被拒絕。所以它傷的不是準確率,
#: 是採用率:上家的採用率因此只有 56.9%。
#:
#: 實測(400 幀、四家):牌面中位飽和度 4(95 百分位 37),牌背中位 126
#: (5 百分位 80)。
_BACK_SATURATION = 70.0

#: 牌背的色相範圍(OpenCV 的 0~179,也就是色環的一半)。**光靠飽和度不夠。**
#:
#: 為什麼要加色相
#: --------------
#: 白板在牌河裡是**淺藍色**的,不是白的 —— 它平躺著,而牌桌的藍光打在一張
#: 完全空白的牌面上就是這個結果。實測(使用者的 2560x1614 畫面,三家各一張):
#:
#: ==============  ======  ========  ======
#: 東西            亮度    飽和度    色相
#: ==============  ======  ========  ======
#: 白板            176     **79**    104
#: 一般牌面        178     1~28      30
#: 牌背            ~140    93~174    5~20
#: 桌布            84      139       112
#: ==============  ======  ========  ======
#:
#: 白板的飽和度 79 已經高過原本的門檻 70,所以它被判成「這格沒有牌」——
#: 牌河中間破一個洞,那一家整幀被丟掉。而牌背的 5 百分位是 80.5,
#: **兩者在飽和度上已經重疊**,再怎麼調那一個數字都分不開。
#:
#: 色相就分得很開:白板 104(藍),牌背 5~20(橘)。所以改成「夠飽和**而且**
#: 是橘的才算牌背」。桌布也是藍的(112),但它暗(84),由 :data:`_OCCUPIED_MEDIAN`
#: 擋掉 —— 三者各由一個維度分開,沒有一個維度要同時做兩件事。
#:
#: 範圍不能只框橘色那一小段。牌背在四家的光照下偏得比想像中多 —— 實測
#: 「結尾多一張」的那一格(上家,35 幀),色相中位 55、5~95 百分位 42~82,
#: 已經是黃綠色了,而不是橘。只框到 40 的話那些牌背全部漏過去,上家的
#: 採用率從 96.7% 掉回 88.5%(連續率仍有 99.3% —— 又是結尾多一張)。
#:
#: 取 90:牌背的 95 百分位是 82,白板是 104,中間還有 22 的空隙。
#:
#: 色環是繞的,所以紅紫兩端都要含進來(接近 0 與接近 179 是同一個顏色)。
_BACK_HUE = (90.0, 170.0)


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
            ``(3 * cell高 + 下方延伸, 6 * cell寬)`` 的 BGR 影像,牌面是正的。
            比三列高一截,是給認牌的搜尋窗往下看用的(:data:`SEARCH_BELOW`)。
            格子的位置不變,第三列以下只是平面的延伸。
        """
        width, height = cell
        target = np.array(
            [[0, 0], [COLUMNS * width, 0], [COLUMNS * width, ROWS * height], [0, ROWS * height]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(self.quad(table), target)
        below = _below(cell)
        return cv2.warpPerspective(image, matrix, (COLUMNS * width, ROWS * height + below))


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
#: 2026-10-30 重搜(對家除外):改用**兩份素材的多幀真值**(各約 60 幀,
#: 加一批河排到第三列的幀),目標是兩份的「有信心且對 − 20 x 有信心卻錯 −
#: 5 x 張數不對」總和。原本的角點是在 110639 一幀上搜的,而 110639 的畫布是
#: 1.7685 不是 16:9,新素材上上家第六欄因此整場 0 次有信心(503 次棄權)。
#: 重搜之後兩份素材都進步,見 ``docs/cv_mjai.md`` §3.6。
#:
#: **對家刻意不動。** 對家搜出來的角點讓第一、二列從 72% / 56% 升到 91% / 65%,
#: 但它改變了攤平後的水平比例,橫放的宣言牌對不上旋轉模板 —— 對封包真值,
#: 對家的立直從 268 次命中變成 0 次。漏掉立直等於把危險牌當安全,正確率
#: 換不回來。要動對家,得先讓宣言牌的比對不依賴比例。
#:
#: ⚠ **只在一種長寬比上驗過。** 角點是相對牌桌矩形的比例,理論上跟著縮放;
#: 但雀魂在不同長寬比下排版不同(``20260801-112437`` 是 1.83,整份讀不出來)。
DEFAULT_CORNERS: dict[str, RiverView] = {
    "own": RiverView(((0.3978, 0.4839), (0.5907, 0.4859), (0.5998, 0.7021), (0.3887, 0.6975))),
    "shimocha": RiverView(((0.5883, 0.4964), (0.5777, 0.2678), (0.7125, 0.2657), (0.7346, 0.4922))),
    "toimen": RiverView(((0.5836, 0.2720), (0.4073, 0.2804), (0.4144, 0.1350), (0.5777, 0.1392))),
    "kamicha": RiverView(((0.4160, 0.2667), (0.4077, 0.4920), (0.2488, 0.5002), (0.2847, 0.2667))),
}


def face_size(cell: tuple[int, int] = CELL) -> tuple[int, int]:
    """模板該縮到多大。傳給 :func:`~mia.vision.tiles.classify.classify` 的 ``size``。"""
    return round(FACE[0] * cell[0]), round(FACE[1] * cell[1])


def _below(cell: tuple[int, int]) -> int:
    return round(SEARCH_BELOW * cell[1] / CELL[1])


def search_window(flat: np.ndarray, rect: Rect, cell: tuple[int, int] = CELL) -> np.ndarray:
    """認一格牌要看的那一塊:格子本身,往下延伸 :data:`SEARCH_BELOW`、
    往上延伸 :data:`SEARCH_ABOVE`(到影像頂端為止)。

    **認牌一律從這裡取,不要自己切 ``flat[rect.as_slice()]``。** 切格子本身的話
    第一列的牌面底部在窗外,模板滑不到正確的位置 —— 那正是二萬讀成一萬的原因。
    """
    top = max(rect.y - round(SEARCH_ABOVE * cell[1] / CELL[1]), 0)
    return flat[top : rect.y + rect.height + _below(cell), rect.x : rect.x + rect.width]


#: 覆核用的第二個模板高度,以一格的高度為單位。即 96 像素。
#:
#: 2026-10-29 改成 0.803:主模板從 0.70 改成 0.77(見 :data:`FACE`),覆核維持
#: 原本 0.73 / 0.70 的**比例**(92 → 96)。維持比例而不是取整數 0.80,是因為
#: 覆核高度由 :func:`classify_river_tile` 依呼叫端給的 ``size`` 按比例推 ——
#: 校正的探針(``ui/panel/tablerect.py``)刻意用舊的 84 高,比例不變它才會
#: 跟以前一模一樣(84 → 88)。實測覆核 96 與單一尺度在兩份素材上結果
#: 相同 —— 主模板的高度對了之後,第二個尺度不再擋下任何東西。留著是因為
#: 它不花正確率,而它當初擋的那一類錯(尺度給錯人)換一份素材可能又出現。
#:
#: 以下是改之前的紀錄。
#:
#: 為什麼要第二個尺度
#: ------------------
#: :data:`FACE` 是掃出來的最佳值,但它是**整體**最佳,不是每一張都最佳。
#: 實測 ``2m`` 在 0.70 時有三分之二被讀成 ``1m``,而且錯的那個分數還更高
#: (0.692 vs 0.670)—— 門檻調不動它,因為它不是「分數不夠」,是「分數給錯人」。
#:
#: 原因是牌面在格子裡的位置比假設的更低,模板的最佳落點常常頂在格子下緣
#: (實測四家有 12~57% 的最佳位置剛好貼在搜尋範圍的底邊)。模板拉高到 0.73
#: 時 ``2m`` 的正確率跳到 55~65%,但**其他牌開始變差**(整體 94.7% → 92.7%),
#: 所以沒有一個高度兩邊都好。
#:
#: 那就不要選邊 —— **兩個高度都算一次,答案不一樣就棄權**。尺度是這裡的
#: 干擾參數,而「換個尺度答案就變了」本身就是不可信的證據。
#:
#: 481 幀的量測(第二個高度掃 0.72 / 0.73 / 0.75):
#:
#: ====================  ==========  ==========  ==========
#: 策略                  每張牌      靜默錯誤    棄權
#: ====================  ==========  ==========  ==========
#: 單一尺度 0.70          94.7%       0.65%       24.47%
#: **一致 0.70 + 0.73**  **94.7%**   **0.19%**   **25.40%**
#: 一致 0.70 + 0.75      94.7%       0.19%       27.18%
#: 取高分 0.70 / 0.75    95.5%       0.50%       18.62%
#: ====================  ==========  ==========  ==========
#:
#: 取「一致 0.73」:靜默錯誤降到不到三分之一,只多花 0.93 個百分點的棄權。
#: 「取高分」那一列每張牌最漂亮、棄權也最低,但**靜默錯誤只降一半** ——
#: 牌河沒人會去核對,那一欄才是該看的。見 :data:`CONFIDENT` 的同一個理由。
CHECK_FACE_HEIGHT = 0.77 * 0.73 / 0.70


def classify_river_tile(
    cell: np.ndarray, templates: TemplateSet, *, size: tuple[int, int] | None = None
) -> Match:
    """認一格牌河的牌,並用第二個模板尺度覆核。

    Args:
        cell: 攤平後**一整格**的影像(不要自己先裁成牌面 ——
            對齊是 :func:`cv2.matchTemplate` 的工作)。
        templates: 牌面模板集。
        size: 主要的模板尺寸,預設 :func:`face_size`。

    Returns:
        與 :func:`~mia.vision.tiles.classify.classify` 同樣的 ``Match``。
        兩個尺度答案不一致時,回傳的 ``Match`` 把另一個答案當成**同分的次高**
        —— 差距因此是 0,過不了任何信心門檻。

        這樣寫而不是多回一個布林值,是為了讓診斷輸出直接說得出原因:
        ``1m 0.70(次 2m 0.70)⚠`` 一眼就看得出是兩個候選打平,
        不是分數不夠。
    """
    width, height = size if size is not None else face_size()
    # merge_red:牌河裡赤五與普通五完全等價(振聽、筋都看普通五),所以
    # 「次高是自己的紅版本」不構成歧義。見 classify 的 merge_red 說明。
    best = classify(cell, templates, size=(width, height), merge_red=True)
    # 用比例換算而不是直接乘 CELL —— 呼叫端給的 size 可能來自別的格子大小,
    # 那時候硬套 CELL 會讓兩個尺度的相對關係跑掉。
    check_height = round(height * CHECK_FACE_HEIGHT / FACE[1])
    if check_height == height:
        return best
    other = classify(cell, templates, size=(width, check_height), merge_red=True)
    if other.label == best.label:
        return best
    return Match(
        label=best.label,
        score=best.score,
        runner_up=other.label,
        runner_up_score=best.score,
        brightness=best.brightness,
    )


@dataclass(frozen=True, slots=True)
class River:
    """一家牌河讀出來的結果。

    Attributes:
        cells: 每一格在**攤平後影像**上的矩形,由遠而近、由左而右。長度固定
            18 —— 空格也留著,因為「第幾格」要對得上捨牌順序。
        occupied: 每一格有沒有牌。
        sideways: 哪一格是橫放的立直宣言牌,沒有就是 ``None``。格子的位置
            已經把它算進去了 —— 這個欄位留著是為了讓上層說得出
            「這一家立直了」,那是 ``Reach`` 事件的來源。
        clipped: **知道有牌但讀不到**的格子。一列裡有橫放牌的時候,那一列的
            物理寬度超過六格,於是最後一格落在攤平用的四邊形外面。這種格子
            的 ``occupied`` 仍然是 ``True``(張數與連續性都要對),但
            ``cells`` 裡的矩形是切掉的 —— 上層**必須棄權**,不要拿它去認牌。
    """

    cells: tuple[Rect, ...]
    occupied: tuple[bool, ...]
    sideways: int | None = None
    clipped: tuple[int, ...] = ()

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


def read_river(
    flat: np.ndarray, *, cell: tuple[int, int] = CELL, sideways: int | None = None
) -> River:
    """在**攤平後**的影像上切 18 格,並判斷每一格有沒有牌。

    Args:
        flat: :meth:`RiverView.rectify` 的輸出。
        cell: 攤平時用的一格大小。
        sideways: 哪一格是橫放的立直宣言牌(:func:`find_sideways` 的輸出)。
            給了的話,**同一列**後面的格子整排往右挪 :data:`SIDEWAYS_SHIFT`。
            不確定的時候給 ``None`` —— 挪錯比不挪糟,因為不挪只錯那一張,
            挪錯會讓那一列後面全錯。

    Returns:
        每一格的位置與有無。**不判斷牌面是什麼** —— 那是
        :func:`~mia.vision.tiles.classify.classify` 的工作。
    """
    if flat.ndim != 3 or flat.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={flat.shape}")

    width, height = cell
    cells: list[Rect] = []
    occupied: list[bool] = []
    clipped: list[int] = []
    for index in range(COLUMNS * ROWS):
        column, row = index % COLUMNS, index // COLUMNS
        shift = _shift_of(index, sideways)
        rect = Rect(column * width + shift, row * height, width, height)
        cells.append(rect)
        if rect.x + rect.width <= flat.shape[1]:
            occupied.append(_occupied(flat, _face_of(rect, cell)))
            continue
        # 挪出去了。**有無要用沒挪的位置判,不能判成空的。**
        #
        # 判成空的會讓那一列破一個洞,而洞會讓 :meth:`River.is_plausible`
        # 把整家這一幀丟掉 —— 實測舊素材 10 次、新素材 5 次,全是立直之後。
        # 那等於用「扔掉整家」換「宣言牌之後的正確率變好看」,而被扔掉的
        # 其他五張本來是讀得到的。
        here = Rect(column * width, row * height, width, height)
        there = _occupied(flat, _face_of(here, cell))
        occupied.append(there)
        if there:
            clipped.append(index)
    return River(
        cells=tuple(cells),
        occupied=tuple(occupied),
        sideways=sideways,
        clipped=tuple(clipped),
    )


def _shift_of(index: int, sideways: int | None) -> int:
    """這一格要不要讓開橫放牌多出來的寬度。

    只有**同一列、在它後面**的格子要讓。下一列從頭排,不受影響 —— 這是
    實測的:宣言牌把後面推了 27px,而推的範圍在換列的地方就斷了。
    """
    if sideways is None or index <= sideways:
        return 0
    return SIDEWAYS_SHIFT if index // COLUMNS == sideways // COLUMNS else 0


def find_sideways(
    flat: np.ndarray,
    templates: TemplateSet,
    *,
    river: River | None = None,
    cell: tuple[int, int] = CELL,
    candidates: Sequence[int] | None = None,
) -> int | None:
    """哪一格是橫放的立直宣言牌 —— 沒有就回 ``None``。

    為什麼需要它
    ------------
    宣言牌橫放,佔的寬度多 27px,而格線是固定等寬的 —— 所以它**本身**認不出
    來,而且同一列後面的每一張都偏。實測(兩份錄影):宣言牌那一格的正確率是
    **0.0%**、有信心的張數是 **0**;宣言牌之後那幾張從 96% 掉到 33%。

    怎麼判
    ------
    三道關卡,每一道都是因為前一道單獨不夠才加上去的:

    1. **直放認不出來。** 宣言牌 221/221 格都過不了 :data:`CONFIDENT` ——
       它真的沒有直放,所以直放模板本來就認不出它。而正常格子有六成多是
       認得出來的,這一刀砍掉大半。靈敏度 100%,零代價。
    2. **旋轉模板贏得比較多**(:data:`_SIDEWAYS_MARGIN`)。只縮候選,
       不拍板 —— 單獨用的話假陽 6%。
    3. **挪了之後認得出來的牌變多。** 這一關才是拍板的。前兩關的假陽裡
       51/92 是**九筒** —— 九筒是 3x3 的圓點,轉九十度跟原來幾乎一樣,
       所以「像不像轉過的」對它原則上無解。那就別問了,直接問我們真正在意
       的事:假設它橫放、把後面挪開,那幾張有沒有變得認得出來。
       真的橫放牌後面本來全偏著,挪完會對回來;九筒誤判後面本來是對的,
       挪完會壞掉。實測 0/92 假陽通過這一關。

    順便得到一個很乾淨的性質:宣言牌是當列最後一張的時候沒有東西可以驗,
    於是回 ``None``。**而那時候它也還沒害到任何人** —— 它只錯自己那一張,
    後面還沒有牌。這個偵測器只需要在有影響的時候正確,而剛好就在那時候
    驗得出來。

    Args:
        flat: 攤平後的影像。
        templates: 牌面模板集(直放的;旋轉版由 :meth:`TemplateSet.rotated` 來)。
        river: 已經讀好的牌河,省一次重讀。**必須是沒挪過的**那一份。
        cell: 攤平時用的一格大小。
        candidates: 只檢查這幾格,``None`` 代表全部有牌的格子。

            即時那條路用得到:每一格都問一次要約 460 毫秒(四家 1830),
            而呼叫端通常已經知道哪幾格可疑 —— 宣言牌一定是**認不出來的**
            那幾格之一(實測 221/221 都過不了 :data:`CONFIDENT`),而且
            一定**不是當列最後一張**(最後一張沒有後續可驗,見第三關)。
            把候選縮到那幾格,大部分的幀會直接回 ``None`` 而不花錢。

    Returns:
        橫放那一格的索引,或 ``None``。拿它餵
        :func:`read_river` 的 ``sideways``,並記得那一列最後一格可能因此
        落在四邊形外面(見 :attr:`River.clipped`)。
    """
    if river is None:
        river = read_river(flat, cell=cell)
    size = face_size(cell)
    rotated_size = sideways_face_size(cell)

    best: tuple[float, int] | None = None
    wanted = (
        river.filled if candidates is None else tuple(k for k in candidates if k in river.filled)
    )
    for index in wanted:
        upright = classify_river_tile(
            search_window(flat, river.cells[index], cell), templates, size=size
        )
        if upright.score >= CONFIDENT[0] and upright.margin >= CONFIDENT[1]:
            continue
        patch = _sideways_patch(flat, index, cell)
        upright_score = _top_score(patch, templates, size)
        rotated_score = _top_score(patch, templates.rotated(), rotated_size)
        if upright_score is None or rotated_score is None:
            continue
        lead = rotated_score - upright_score
        if lead >= _SIDEWAYS_MARGIN and (best is None or lead > best[0]):
            best = (lead, index)

    if best is None:
        return None
    return best[1] if _shift_helps(flat, river, best[1], templates, cell) else None


def _sideways_patch(flat: np.ndarray, index: int, cell: tuple[int, int]) -> np.ndarray:
    """比對橫放牌的那一塊:往右寬到 :data:`_SIDEWAYS_WINDOW` 格,上下與
    :func:`search_window` 一樣延伸。

    理由與 :data:`SEARCH_ABOVE` 相同:第三列的宣言牌一樣會被切頂。兩份素材
    對封包真值逐幀比過,換成這個窗之後立直偵測的命中、漏判、誤判**完全不變**。
    """
    width, height = cell
    column, row = index % COLUMNS, index // COLUMNS
    x = column * width
    rect = Rect(x, row * height, min(round(width * _SIDEWAYS_WINDOW), flat.shape[1] - x), height)
    return search_window(flat, rect, cell)


def sideways_face_size(cell: tuple[int, int] = CELL) -> tuple[int, int]:
    """橫放牌面在攤平圖上的像素大小。見 :data:`SIDEWAYS_FACE`。"""
    return (round(cell[0] * SIDEWAYS_FACE[0]), round(cell[1] * SIDEWAYS_FACE[1]))


def classify_sideways(
    flat: np.ndarray, index: int, templates: TemplateSet, *, cell: tuple[int, int] = CELL
) -> Match:
    """認橫放的立直宣言牌是哪一張。

    用的是逆時針轉過的模板(:meth:`TemplateSet.rotated`)與一個比一格更寬的
    窗 —— 躺著的牌比一格寬,只取一格會把牌面切掉一截,而被切掉的正好是
    要對上的部分。

    為什麼要單獨一支:直放的 :func:`classify_river_tile` 對這一格的正確率是
    **0.0%**、有信心的張數是 **0**(兩份錄影、28 格)。那不是門檻的問題 ——
    它真的沒有直放,用直放模板問錯了問題。

    Args:
        flat: 攤平後的影像。
        index: 橫放那一格的索引(:func:`find_sideways` 的輸出)。
        templates: **直放的**模板集,旋轉在裡面做。
        cell: 攤平時用的一格大小。

    Returns:
        與 :func:`classify_river_tile` 同樣的 ``Match``,但**永遠不會有信心**
        —— 差距被歸零。見下。

    Note:
        **為什麼刻意讓它過不了門檻。**

        :data:`SIDEWAYS_FACE` 量對之後,分數從 0.53 升到 0.65~0.81,於是它開始
        **有信心地講錯話**:舊素材那張 2m 被認成 1m,22 格裡 10 格有信心、
        靜默錯誤 100%。正確率從 0.0% 升到 54.5%,但換來的是有信心的錯誤 ——
        那個交換在放銃分析裡是賠的(讀不到的牌當危險處理,成本遠低於把
        危險牌說成別的牌)。

        而且證據量差了三個數量級:直放那條路的門檻是在 1921 幀上掃出來的,
        橫放這條路**總共只有 4 張不同的宣言牌**,其中一張在任何尺寸下都認錯。
        四張牌不足以讓一個讀數有資格宣稱把握。

        寫法沿用 :func:`classify_river_tile` 兩個尺度不一致時的做法:把次高分
        設成與最高分相同,差距因此是 0。標籤與分數照樣留著 —— 診斷輸出看得到
        「它以為是什麼」,而上層拿不到「可以採信」。
    """
    patch = _sideways_patch(flat, index, cell)
    best = classify(patch, templates.rotated(), size=sideways_face_size(cell))
    return Match(
        label=best.label,
        score=best.score,
        runner_up=best.runner_up,
        runner_up_score=best.score,
        brightness=best.brightness,
    )


def _top_score(patch: np.ndarray, templates: TemplateSet, size: tuple[int, int]) -> float | None:
    """整套模板裡最高的那個相關係數。模板比窗還大就回 ``None``。"""
    top: float | None = None
    for _, face in templates.scaled(size):
        if face.shape[0] >= patch.shape[0] or face.shape[1] >= patch.shape[1]:
            return None
        _, score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(patch, face, cv2.TM_CCOEFF_NORMED))
        top = score if top is None else max(top, score)
    return top


def _confident_count(
    flat: np.ndarray,
    columns: Sequence[int],
    row: int,
    shift: int,
    templates: TemplateSet,
    cell: tuple[int, int],
) -> int:
    """這幾格裡有幾格認得出來(整排往右挪 ``shift``)。"""
    width, height = cell
    size = face_size(cell)
    found = 0
    for column in columns:
        x = column * width + shift
        if x < 0 or x + width > flat.shape[1]:
            continue
        match = classify_river_tile(
            search_window(flat, Rect(x, row * height, width, height), cell), templates, size=size
        )
        if match.score >= CONFIDENT[0] and match.margin >= CONFIDENT[1]:
            found += 1
    return found


def _shift_helps(
    flat: np.ndarray,
    river: River,
    index: int,
    templates: TemplateSet,
    cell: tuple[int, int],
) -> bool:
    """把同一列後面的格子挪開之後,認得出來的牌有變多嗎。

    見 :func:`find_sideways` 第三關。沒有後續可驗的時候回 ``False`` ——
    寧可漏掉,因為那種情況下漏掉不會錯任何一張牌。
    """
    column, row = index % COLUMNS, index // COLUMNS
    after = [
        c for c in range(column + 1, COLUMNS) if river.occupied[row * COLUMNS + c]
    ]
    if not after:
        return False
    plain = _confident_count(flat, after, row, 0, templates, cell)
    moved = _confident_count(flat, after, row, SIDEWAYS_SHIFT, templates, cell)
    return moved > plain


def _face_of(cell: Rect, size: tuple[int, int]) -> Rect:
    """格子正中央、牌面該在的那一塊。

    **有無判斷只能看這裡,不能看整格。** 一格比一張牌大一圈(:data:`_OCCUPANCY_FACE`
    是 0.72 x 0.70,也就是牌面只佔一格的一半),所以一格裡裝著一張牌時,
    另外一半是桌布 —— 整格取中位數會落在牌與桌布之間。

    實測:一局開頭河裡只有一張牌時,那一格整格的中位亮度是 97,而門檻是
    110 —— 判成空的。症狀是「每局開頭的幀全部被拒絕」,而 own 的採用率
    因此只有一半。
    """
    face_width = round(_OCCUPANCY_FACE[0] * size[0])
    face_height = round(_OCCUPANCY_FACE[1] * size[1])
    return Rect(
        cell.x + (cell.width - face_width) // 2,
        cell.y + (cell.height - face_height) // 2,
        face_width,
        face_height,
    )


def _occupied(image: np.ndarray, cell: Rect) -> bool:
    """這一塊是一張**攤開的捨牌**嗎。

    兩道都要過:

    * 夠亮 —— 不是桌布。見 :data:`_OCCUPIED_MEDIAN`。
    * 不是橘的 —— 不是牌背。見 :data:`_BACK_SATURATION` 與 :data:`_BACK_HUE`。

    只看亮度的話牌背會過關,因為橘色跟象牙色一樣亮;而只看飽和度的話**白板**
    會被擋掉,因為它平躺著會染上桌子的藍。三樣東西要三個維度才分得開。
    """
    return is_face_up(image[cell.as_slice()])


def is_face_up(patch: np.ndarray) -> bool:
    """這一塊是一張**翻開的牌**嗎 —— 不是桌布,也不是牌背。

    三樣東西要三個維度才分得開:只看亮度的話牌背會過關(橘色跟象牙色一樣亮),
    只看飽和度的話**白板**會被擋掉(它平躺著會染上桌子的藍)。

    牌河與寶牌面板共用這一支。兩邊問的是同一個問題 —— 寶牌槽裡不是翻開的
    指示牌就是橘色的牌背,而那正是這裡擋掉的東西。門檻是在牌河上量的
    (見 :data:`_OCCUPIED_MEDIAN`、:data:`_BACK_SATURATION`、:data:`_BACK_HUE`),
    寶牌面板實測翻開的槽飽和度 37、牌背 169 —— 分得比牌河還開。
    """
    if patch.size == 0:
        return False
    # 亮度用灰階而不是 HSV 的 V。V 是三通道的最大值,所以藍色桌布的 V
    # 會被藍通道拉高到門檻以上 —— 那會讓空格子變成「有牌」。
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if float(np.median(grey)) < _OCCUPIED_MEDIAN:
        return False
    return not _is_back(float(np.median(hsv[:, :, 0])), float(np.median(hsv[:, :, 1])))


def face_coverage(patch: np.ndarray) -> float:
    """這一塊裡「是攤開的牌面」的像素佔多少,0~1。

    :func:`_occupied` 的逐像素版本,問的是同一件事,只是不先取中位數就表態。
    有無判斷要的是一個是非題,而**對齊**要的是程度:格線壓到隔壁的牌時,
    那一格仍然「有牌」,但牌面只佔得到一半。中位數看不出這件事,佔比看得出。

    門檻故意與 :func:`_occupied` 共用。兩邊各寫一份的話,哪天調了其中一個,
    另一個會安靜地跟著不準 —— 而那種不準不會有任何錯誤訊息。
    """
    if patch.size == 0:
        return 0.0
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    back = ((hue <= _BACK_HUE[0]) | (hue >= _BACK_HUE[1])) & (saturation >= _BACK_SATURATION)
    return float(((grey > _OCCUPIED_MEDIAN) & ~back).mean())


def _is_back(hue: float, saturation: float) -> bool:
    """這一塊是橘色的牌背嗎。

    飽和度低的時候色相是雜訊(一塊近乎無彩的象牙色,色相是哪個值都不奇怪),
    所以**兩個條件要一起看**:夠飽和,而且落在橘色那一段。
    """
    if saturation < _BACK_SATURATION:
        return False
    return hue <= _BACK_HUE[0] or hue >= _BACK_HUE[1]
