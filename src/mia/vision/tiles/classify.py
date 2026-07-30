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
    "Match",
    "TemplateSet",
    "classify",
    "classify_hand",
]

#: 牌面佔牌框的比例。掃描 0.60~0.94 後取平均最高分的極大值。
#: 0.78~0.92 都是 100% 正確的高原,所以這個值不是刀鋒上的巧合。
FACE_RATIO = 0.84

#: 判定為可信所需的最低分數。實測 24 張正確答案的最低分是 0.631
#: (那張所在的參考畫面牌底被切掉一截),取 0.55 留餘裕。
MIN_SCORE = 0.55

#: 與次高分的最小差距。實測最薄的是普通五 vs 赤五的 0.084,取 0.04。
#: 差距太小代表「像兩張牌」,那種時候寧可回報不確定,交給上層丟掉這一幀。
MIN_MARGIN = 0.04

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
    """

    label: str
    score: float
    runner_up: str
    runner_up_score: float

    @property
    def margin(self) -> float:
        """第一名領先第二名多少。比絕對分數更能反映「認得有多確定」。"""
        return self.score - self.runner_up_score

    @property
    def is_confident(self) -> bool:
        """分數與差距都過關才算數。"""
        return self.score >= MIN_SCORE and self.margin >= MIN_MARGIN

    def __str__(self) -> str:
        flag = "" if self.is_confident else " ⚠"
        return (
            f"{self.label} {self.score:.3f}"
            f"(次 {self.runner_up} {self.runner_up_score:.3f}){flag}"
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

    def __str__(self) -> str:
        return f"TemplateSet({len(self._faces)} 類)"


def classify(tile: np.ndarray, templates: TemplateSet) -> Match:
    """判斷一張牌是什麼。

    Args:
        tile: 一張牌的 BGR 影像,就是 :class:`~mia.vision.tiles.hand.Hand`
            給的牌框裁出來的那塊(**含上緣斜面**,不要自己先裁掉 —— 對齊是
            :func:`cv2.matchTemplate` 的工作)。
        templates: 對應目前牌面皮膚的模板集。

    Returns:
        最像的牌與診斷資訊。**永遠會回傳一個結果**,不會拋錯 —— 用
        :attr:`Match.is_confident` 判斷要不要採信。即時串流裡認不出來是常態
        (抓在動畫中間),那是上層該丟掉這一幀,不是例外狀況。
    """
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道影像,拿到 shape={tile.shape}")

    height, width = tile.shape[:2]
    size = (int(width * FACE_RATIO), int(height * FACE_RATIO))
    if min(size) < MIN_TEMPLATE_PX or size[0] >= width or size[1] >= height:
        raise ValueError(
            f"牌框 {width}x{height} 太小 —— 模板會縮成 {size[0]}x{size[1]},"
            f"每邊至少要 {MIN_TEMPLATE_PX} 像素,且要留得下平移搜尋的餘裕"
        )

    best = ("", -2.0)
    second = ("", -2.0)
    for label, face in templates.scaled(size):
        score = float(cv2.matchTemplate(tile, face, cv2.TM_CCOEFF_NORMED).max())
        if score > best[1]:
            best, second = (label, score), best
        elif score > second[1]:
            second = (label, score)
    return Match(best[0], best[1], second[0], second[1])


def classify_hand(
    roi: np.ndarray, hand: Hand, templates: TemplateSet
) -> tuple[tuple[Match, ...], Match | None]:
    """一次判斷整手牌。

    Args:
        roi: ``own_hand`` ROI 的 BGR 影像。
        hand: :func:`~mia.vision.tiles.hand.read_hand` 的結果。
        templates: 模板集。

    Returns:
        ``(暗手牌的比對結果, 摸牌那張的比對結果或 None)``。
        摸牌那張刻意分開回傳 —— 對應 MJAI 的 ``tsumo``,「手上有這張牌」與
        「這一巡摸到這張牌」是兩件事。
    """
    concealed = tuple(
        classify(roi[box.as_slice()], templates) for box in hand.concealed
    )
    drawn = classify(roi[hand.drawn.as_slice()], templates) if hand.drawn else None
    return concealed, drawn
