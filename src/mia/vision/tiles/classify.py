"""判斷一張牌是什麼:模板比對。

輸入是 :func:`~mia.vision.tiles.hand.read_hand` 給的牌框影像,
輸出是雀魂記法的牌名(``1m``~``9m``、``0m`` 赤五、``1z``~``7z``)。
要轉成 MJAI 用 :func:`~mia.mjai.tiles.ms_to_mjai`。

模板從哪來
----------
從**雀魂官方資源**的牌面圖集切出來(``tools/fetch_tiles.py``),不是從遊戲畫面
裁的。官方資源直接給出已經標好的 37 張,不需要人工標註,也不需要一份畫面與
封包配對的錄影 —— 而一張標錯的模板是永久且靜默的錯誤源。

為什麼是「比例 + 平移搜尋」而不是固定裁切
------------------------------------------
第一版把模板與查詢各自用固定比例裁一次再算相關係數,結果**萬子與字牌全錯**
(三萬被判成一萬、中被判成四萬)。原因不是貼圖與 3D 渲染的差距 —— 兩者其實
長得幾乎一樣 —— 而是兩邊的裁切沒有對齊:牌框含上緣斜面,圖集不含。

改用 :func:`cv2.matchTemplate` 讓它自己找對齊位置就好了。但**尺度不能一起搜尋**:
放寬到 ±20% 之後,6 筒被拉伸成 9 筒、9 索被縮成 4 索,筒子與索子開始出錯。
尺度其實是已知的 —— 槽位模型給了精確的牌框大小,牌面固定佔牌框
:data:`FACE_RATIO`,只要搜尋平移。

那個比例是用**無標註**的方式定出來的:掃過 0.60~0.94,取「平均最高分」的極大值
(0.84),之後才拿已知答案量準確率,不是拿答案去調參。

為什麼用彩色
------------
赤五(``0m``/``0p``/``0s``)與普通五只差顏色和一個小紅點。灰階比對雖然也全對,
但差距只有 0.021~0.031,太薄;改成彩色後差距拉開到 0.063~0.113。

已知的限制
----------
* **只驗過自家手牌。** 牌河與副露有透視形變,不在範圍內。
* **模板要對得上牌面皮膚。** 雀魂的 ``mjpface_*`` 是可換的牌面皮膚,換了就要
  重跑 ``fetch_tiles.py --skin``。這與牌**背**皮膚無關 —— 定位那一層已經不看
  顏色了(見 :mod:`~mia.vision.tiles.hand`)。
* 目前的驗證集只有 24 張已知答案的牌、單一解析度。真正的準確率要等一份畫面與
  封包配對的錄影才能量(見 ``docs/decisions.md``)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mia.utils.paths import ASSETS_DIR
from mia.vision.tiles.hand import Hand

__all__ = [
    "FACE_RATIO",
    "HOVER_HEADROOM",
    "MIN_BRIGHTNESS",
    "RED_INK",
    "WHOLE_RED",
    "Match",
    "TemplateSet",
    "classify",
    "classify_hand",
    "ink_masks",
    "red_ink",
    "settle_five",
]

#: 牌面佔牌框的比例。掃描 0.60~0.94 後取平均最高分的極大值。
#: 0.78~0.92 都是 100% 正確的高原,所以這個值不是刀鋒上的巧合。
FACE_RATIO = 0.84

#: 牌框上方額外的搜尋空間,以牌框高度為單位。
#:
#: 滑鼠停在待選牌上時雀魂會把那張牌抬起來,實測抬 **32% 的牌高**
#: (session 20260801-110639:65px / 205px,88 個樣本全部落在 65±5px)。
#: 0.5 是那個值的 1.5 倍餘裕 —— 抬起來的那張整張都要留在搜尋範圍內,
#: 少一點點就等於認不得。理由見 :func:`classify_hand`。
HOVER_HEADROOM = 0.5

#: 判定為可信所需的最低分數。實測 24 張正確答案的最低分是 0.631
#: (那張所在的參考畫面牌底被切掉一截),取 0.55 留餘裕。
MIN_SCORE = 0.55

#: 與次高分的最小差距。實測最薄的是普通五 vs 赤五的 0.084,取 0.04。
#: 差距太小代表「像兩張牌」,那種時候寧可回報不確定,交給上層丟掉這一幀。
MIN_MARGIN = 0.04

#: 比對區與模板的亮度比下限。
#:
#: ``TM_CCOEFF_NORMED`` 對亮度與對比是**不變**的 —— 它先減掉各自的平均再做
#: 正規化,所以它問的是「形狀像不像」,不是「看起來像不像」。那個不變性正是
#: 它在這裡的盲點:一塊深藍色桌布與象牙色的白板,相關係數可以到 0.69,
#: 因為兩者都是「平的」。白板的像素標準差只有 8.3,其餘 36 個模板最低 44 ——
#: 它幾乎沒有結構可比,於是什麼平滑的東西都能跟它對上。
#:
#: 實測(4327 幀的評測素材):桌布 0.25~0.26,最暗的真牌 0.79,一般真牌 1.00。
#: 取 0.5 落在中間,離兩邊都很遠。
#:
#: 這條檢查擋的不是「認錯牌」,是「**那一格根本不是牌**」—— 牌框落在桌布上
#: (摸牌那格的位置算歪、動畫中間整排牌在滑)的時候,它是唯一攔得住的東西。
MIN_BRIGHTNESS = 0.5

DEFAULT_SKIN = "mjpface_default"

#: 模板縮放後每邊至少要這麼多像素。再小的話相關係數只是在比雜訊 ——
#: 與其回報一個看似合理的分數,不如當場拒絕。
MIN_TEMPLATE_PX = 16


@dataclass(frozen=True, slots=True)
class Match:
    """一次比對的結果。

    Attributes:
        label: 最像的牌(雀魂記法)。
        score: 該牌的相關係數,1.0 為完全相同。
        runner_up: 次像的牌 —— 診斷用。判錯時看這個通常就知道為什麼。
        runner_up_score: 次高分。
        brightness: 比對區與模板的亮度比。見 :data:`MIN_BRIGHTNESS`。
            預設 1.0 代表「沒量過,當作沒問題」—— 手工建構的 ``Match``
            (測試、診斷)不該因為少填一個欄位就被判成不可信。
    """

    label: str
    score: float
    runner_up: str
    runner_up_score: float
    brightness: float = 1.0

    @property
    def margin(self) -> float:
        """第一名領先第二名多少。比絕對分數更能反映「認得有多確定」。"""
        return self.score - self.runner_up_score

    @property
    def is_confident(self) -> bool:
        """分數、差距、亮度三個都過關才算數。

        三個問的是不同的事:分數是「像不像」,差距是「有沒有別的也一樣像」,
        亮度是「**這裡到底是不是一張牌**」。少了第三個,落在桌布上的牌框會
        以 0.69 的分數與 0.35 的差距自信地回報白板。
        """
        return (
            self.score >= MIN_SCORE
            and self.margin >= MIN_MARGIN
            and self.brightness >= MIN_BRIGHTNESS
        )

    def __str__(self) -> str:
        flag = "" if self.is_confident else " ⚠"
        dim = "" if self.brightness >= MIN_BRIGHTNESS else f" 暗 {self.brightness:.2f}"
        return (
            f"{self.label} {self.score:.3f}"
            f"(次 {self.runner_up} {self.runner_up_score:.3f}){dim}{flag}"
        )


class TemplateSet:
    """一套牌面模板,對應一種牌面皮膚。

    模板影像的尺寸不重要 —— 比對前會依查詢影像的牌框大小重新縮放,
    所以同一套模板可以用在任何視窗尺寸上。
    """

    def __init__(self, faces: dict[str, np.ndarray]) -> None:
        if not faces:
            raise ValueError("模板集是空的")
        self._faces = dict(faces)
        self._cache: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
        self._brightness: dict[str, float] = {}
        self._rotated: TemplateSet | None = None

    @classmethod
    def load(cls, skin: str = DEFAULT_SKIN, *, root: Path | None = None) -> TemplateSet:
        """從 ``assets/tiles/<皮膚>/`` 載入。

        目錄不存在或空的時候直接拋錯,不退回某個「內建的」模板集 ——
        拿錯皮膚的模板硬跑只會讓準確率不明不白地變差。
        """
        base = (root or ASSETS_DIR) / "tiles" / skin
        files = sorted(base.glob("*.png")) if base.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"找不到牌面模板:{base}。請先執行 "
                f"`python tools/fetch_tiles.py --skin {skin}` 取得。"
            )
        faces = {}
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"讀不到模板 {path}")
            faces[path.stem] = image
        return cls(faces)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._faces))

    def __len__(self) -> int:
        return len(self._faces)

    def __contains__(self, label: str) -> bool:
        return label in self._faces

    def scaled(self, size: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
        """縮放到指定尺寸的模板。牌框大小在一場對局裡不變,所以快取很值得。"""
        cached = self._cache.get(size)
        if cached is None:
            cached = [
                (label, cv2.resize(face, size, interpolation=cv2.INTER_AREA))
                for label, face in self._faces.items()
            ]
            self._cache[size] = cached
        return cached

    def only(self, *labels: str) -> TemplateSet:
        """只留這幾張的模板集。"""
        return TemplateSet({label: self._faces[label] for label in labels})

    def rotated(self) -> TemplateSet:
        """整套模板逆時針轉九十度 —— 認橫放的立直宣言牌用。

        只轉**逆時針**一個方向。實測雀魂的宣言牌一律朝左躺,所以另一個方向
        的模板只會貢獻假陽 —— 而且曾經害我把量測看反過:第一次比對只試了
        順時針,得到的分數全是雜訊,還被我當成「訊號方向相反」。

        快取在實例上:這是 37 張影像的旋轉,而呼叫端是每一幀的每一格。
        """
        if self._rotated is None:
            self._rotated = TemplateSet(
                {
                    label: cv2.rotate(face, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    for label, face in self._faces.items()
                }
            )
        return self._rotated

    def brightness(self, label: str) -> float:
        """模板的中位灰階值。見 :data:`MIN_BRIGHTNESS`。

        用中位數而不是平均:牌面上的圖案(八筒那一堆黑圈)會把平均拉低,
        而我們要的是「牌底色有多亮」。實測 37 個模板的中位值落在 178~223,
        最低的是八筒 —— 圖案佔的面積最大的那張。

        在**未縮放**的原圖上算。中位數對縮放不敏感,所以不必為每個牌框尺寸
        各算一次。
        """
        cached = self._brightness.get(label)
        if cached is None:
            grey = cv2.cvtColor(self._faces[label], cv2.COLOR_BGR2GRAY)
            cached = float(np.median(grey))
            self._brightness[label] = cached
        return cached

    def __str__(self) -> str:
        return f"TemplateSet({len(self._faces)} 類)"


def classify(
    tile: np.ndarray,
    templates: TemplateSet,
    *,
    tile_height: int | None = None,
    size: tuple[int, int] | None = None,
    merge_red: bool = False,
) -> Match:
    """判斷一張牌是什麼。

    Args:
        tile: 要搜尋的 BGR 影像。通常就是一張牌的牌框(**含上緣斜面**,不要
            自己先裁掉 —— 對齊是 :func:`cv2.matchTemplate` 的工作),但也可以
            是一條更高的直條,見 ``tile_height``。
        templates: 對應目前牌面皮膚的模板集。
        tile_height: 牌框本身的高度。``None`` 表示 ``tile`` 就是牌框。
        size: 模板要縮到的 ``(寬, 高)``,單位像素。給了就**蓋過**
            :data:`FACE_RATIO` 推出來的尺寸。

            牌河用得到:那裡的牌是攤平之後的俯視圖,牌面佔一格的比例與手牌
            不同(手牌 0.84,牌河約 0.71),而且橫豎兩個方向也不一樣。硬套
            手牌的比例會讓模板大一圈,分數整片掉下來 —— 症狀看起來像幾何
            沒校好,但其實幾何是對的,而往幾何去修會越修越糟。
            給值時代表 ``tile`` 是一條含額外搜尋空間的直條,模板仍然依牌框
            大小縮放 —— 拿整條的高度去縮模板會把牌面拉長,那比搜不到還糟。
        merge_red: 挑次高分時**跳過贏家自己的赤五版本**。

            :attr:`Match.margin` 問的是「有沒有**別的牌**也一樣像」,而
            ``0m`` 不是別的牌 —— 它跟 ``5m`` 在振聽與筋裡完全等價。實測
            普通五與赤五的差距只有 0.084,所以牌河的門檻(0.10)會把每一張
            五都判成「像兩張牌」而棄權。手牌那邊的 :data:`MIN_MARGIN` 取
            0.04 正是為了容納這一對 —— 這個參數是同一件事的另一種做法,
            而且不必為它放寬門檻。

            **只在赤五與普通五無差別的地方開。** 和牌算符要知道紅不紅,
            那裡不要開。

    Returns:
        最像的牌與診斷資訊。**永遠會回傳一個結果**,不會拋錯 —— 用
        :attr:`Match.is_confident` 判斷要不要採信。即時串流裡認不出來是常態
        (抓在動畫中間),那是上層該丟掉這一幀,不是例外狀況。
    """
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={tile.shape}")

    height, width = tile.shape[:2]
    face_height = tile_height if tile_height is not None else height
    if size is None:
        size = (int(width * FACE_RATIO), int(face_height * FACE_RATIO))
    if min(size) < MIN_TEMPLATE_PX or size[0] >= width or size[1] >= height:
        raise ValueError(
            f"牌框 {width}x{face_height} 太小 —— 模板會縮成 {size[0]}x{size[1]},"
            f"每邊至少要 {MIN_TEMPLATE_PX} 像素,且要留得下平移搜尋的餘裕"
        )

    scored: list[tuple[str, float]] = []
    best = ("", -2.0)
    best_at = (0, 0)
    for label, face in templates.scaled(size):
        # minMaxLoc 要的是位置,不只是分數 —— 亮度檢查得知道贏家對在哪裡,
        # 才切得出真正拿去比的那一塊。成本相對 matchTemplate 可以忽略。
        _, score, _, at = cv2.minMaxLoc(cv2.matchTemplate(tile, face, cv2.TM_CCOEFF_NORMED))
        scored.append((label, score))
        if score > best[1]:
            best, best_at = (label, score), (int(at[0]), int(at[1]))

    second = ("", -2.0)
    for label, score in scored:
        if label == best[0] or score <= second[1]:
            continue
        if merge_red and _same_tile(label, best[0]):
            continue
        second = (label, score)

    window = tile[best_at[1] : best_at[1] + size[1], best_at[0] : best_at[0] + size[0]]
    reference = templates.brightness(best[0])
    brightness = _median_grey(window) / reference if reference else 1.0
    return Match(best[0], best[1], second[0], second[1], brightness)


#: 紅色的墨水(OpenCV HSV):色相、飽和度下限、亮度下限。
_RED_HUE = (10, 170)
_RED_SATURATION = 70
_RED_VALUE = 70

#: 紙:飽和度低而且亮。這以外都算墨水。
_PAPER_SATURATION = 45
_PAPER_VALUE = 150

#: 五的墨水有多少是紅的才算赤五,依花色。普通五有黑字、黑圈、綠條,赤五整張紅。
#:
#: 取兩邊的中間。半莊 13 局的牌河逐格量(``docs/cv_mjai.md`` §4.4):
#:
#: ====  ==========  ==========
#: 花色  普通五最高  赤五最低
#: ====  ==========  ==========
#: 萬    0.60        0.86
#: 筒    0.16        0.86
#: 索    0.33        0.59
#: ====  ==========  ==========
RED_INK = {"m": 0.73, "p": 0.50, "s": 0.46}

#: 不管是哪張牌,紅到這個程度就只可能是赤五或中。同一份素材其他牌最高 0.72。
WHOLE_RED = 0.8


def ink_masks(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐像素的「是墨水」與「是紅色」,兩張布林圖。"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ink = ~((saturation < _PAPER_SATURATION) & (value > _PAPER_VALUE))
    red = (
        ((hue <= _RED_HUE[0]) | (hue >= _RED_HUE[1]))
        & (saturation >= _RED_SATURATION)
        & (value >= _RED_VALUE)
    )
    return ink, red


def red_ink(patch: np.ndarray) -> float:
    """牌面的墨水裡紅色佔多少,0~1。

    赤五與普通五形狀一樣,差別只在顏色,而模板比對要看整張牌的圖案,差距很薄。
    這裡直接問顏色:牌框要裁到牌面上,外面的桌布會被算成墨水。
    """
    ink, red = ink_masks(patch)
    total = int(ink.sum())
    return float(red.sum()) / total if total else 0.0


def settle_five(label: str, patch: np.ndarray) -> str:
    """``label`` 是五的話,由 ``patch`` 的顏色決定赤不赤。其他牌原樣回傳。"""
    suit = label[-1]
    if label[0] not in "05" or suit not in RED_INK:
        return label
    return f"{0 if red_ink(patch) >= RED_INK[suit] else 5}{suit}"


def _same_tile(a: str, b: str) -> bool:
    """兩個標籤是同一張牌嗎(只差在赤不赤)。

    雀魂記法裡赤五是 ``0m``/``0p``/``0s``。這裡用純字串判斷而不是叫
    :func:`~mia.mjai.tiles.normalize_red`,是為了不讓 vision 這層相依 mjai ——
    兩層的方向是「畫面 → 牌名」,反過來相依會讓模板比對綁上規則模型。
    """
    return a.replace("0", "5") == b.replace("0", "5")


def _median_grey(image: np.ndarray) -> float:
    return float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))


def classify_hand(
    roi: np.ndarray, hand: Hand, templates: TemplateSet, *, headroom: int = 0
) -> tuple[tuple[Match, ...], Match | None]:
    """一次判斷整手牌。

    Args:
        roi: ``own_hand`` ROI 的 BGR 影像。``headroom`` 不為 0 時,是上緣往上
            多切了 ``headroom`` 個像素的版本 —— 見下。
        hand: :func:`~mia.vision.tiles.hand.read_hand` 的結果。牌框座標相對
            **原本的 ROI**,這裡會自己加上 ``headroom`` 的偏移。
        templates: 模板集。
        headroom: 牌框上方額外的搜尋高度,單位是像素。

    Returns:
        ``(暗手牌的比對結果, 摸牌那張的比對結果或 None)``。
        摸牌那張刻意分開回傳 —— 對應 MJAI 的 ``tsumo``,「手上有這張牌」與
        「這一巡摸到這張牌」是兩件事。

    Note:
        **為什麼需要 headroom。** 滑鼠停在待選牌上時,雀魂會把那張牌往上抬 ——
        實測(session 20260801-110639)抬 **65 px / 牌高 205 px,約 32%**。
        ROI 是貼著手牌量的,抬起來的那張有三分之一跑到 ROI 外面,留在框裡的
        是牌的下半截加一片桌布,於是那一張比對不出來或比對錯,而**整手牌的
        向聽數就跟著錯**。使用者滑到某張牌的時機,正是他在看建議的時機。

        給了 headroom 之後,同一張牌在抬起狀態下的相關係數實測回到 0.99。
    """
    concealed = tuple(
        classify(
            roi[: headroom + box.y + box.height, box.x : box.x + box.width],
            templates,
            tile_height=box.height,
        )
        for box in hand.concealed
    )
    drawn = None
    if hand.drawn is not None:
        box = hand.drawn
        drawn = classify(
            roi[: headroom + box.y + box.height, box.x : box.x + box.width],
            templates,
            tile_height=box.height,
        )
    return concealed, drawn
