"""讀牌桌上的 HUD:局數、剩餘牌數、四家點數、莊家、本場與供託。

``StartKyoku`` 要的欄位幾乎都印在畫面上,只是分散在兩塊地方:

* **中央面板。** 「東 3 局」「余 69」、四邊的點數、四個角的風牌 —— **紅色的
  東所在的那個角就是莊家**。
* **左上角寶牌面板的下緣。** 一根紅點棒 x 供託、八點棒 x 本場。

為什麼不用 OCR
--------------
字型固定、位置固定、只有十個數字。切開之後比模板就好,而模板可以直接從錄影
裡取、用封包的點數當標籤(``assets/hud/``),不必多一個幾百 MB 的相依套件。

它讀錯的時候看得出來
--------------------
點數帶著兩層天然的檢查碼:最小單位是 100,而**四家總和 + 場上立直棒 x 1000
= 開局總點數**(四人場 100000)。這一層比任何信心門檻都強 —— 見
:attr:`Hud.scores_agree`。莊家也一樣:局數決定莊家是幾號座位,紅東的位置說他
坐在自己的哪一邊,兩個合起來就是**自己的絕對座位**。

讀不到就回 ``None``,不猜。立繪、結算面板蓋住時,每個欄位各自棄權。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mia.utils.geometry import Rect
from mia.utils.paths import ASSETS_DIR

__all__ = ["DEALER_SEATS", "Hud", "HudDigits", "read_hud", "read_wall"]

#: 中央面板在牌桌矩形裡的位置。下面的欄位都用「面板座標」量,較好對照截圖。
_PANEL = (0.4142, 0.2948, 0.175, 0.2)
_PANEL_SIZE = (450, 290)

#: 比對前把一個數字縮成多大(寬, 高)。透視讓四邊的字粗細不同,縮成一樣大才
#: 比得起來,所以四個方向共用一套模板:分方向的話每個方向分到的樣本太少,
#: 在另一份錄影上反而差(49 錯 vs 3 錯,1113 個數字)。
_DIGIT = (16, 24)

#: 數字比桌面亮多少才算筆畫。金色、青色、白色三種字都過得了。
_INK = 150

#: 一個筆畫至少幾欄。擋掉壓縮雜點。
_MIN_WIDTH = 2

#: 認得出來的門檻:最像的相關係數、領先第二名多少。
_SCORE = 0.60
_MARGIN = 0.05

#: 四人場開局的總點數。見 :attr:`Hud.scores_agree`。
_TOTAL = 100000

#: 四個方向的點數要轉幾次 90 度(順時針)才會是正的,依視角座位。
_TURNS = (0, 1, 2, 3)

#: 視角座位:自己、下家、對家、上家。與牌河的 ``SEATS`` 同序。
DEALER_SEATS = ("own", "shimocha", "toimen", "kamicha")


def _on_panel(x0: int, y0: int, x1: int, y1: int) -> tuple[float, float, float, float]:
    """面板座標 → 相對牌桌矩形的 ``(x, y, 寬, 高)``。"""
    x, y, width, height = _PANEL
    across, down = _PANEL_SIZE
    return (
        x + x0 / across * width,
        y + y0 / down * height,
        (x1 - x0) / across * width,
        (y1 - y0) / down * height,
    )


_SCORES = (
    _on_panel(155, 200, 285, 240),
    _on_panel(312, 90, 362, 180),
    _on_panel(155, 50, 285, 80),
    _on_panel(84, 95, 134, 185),
)
_WIND = _on_panel(168, 96, 206, 140)
_ROUND = _on_panel(205, 96, 236, 140)
_WALL = _on_panel(208, 148, 272, 186)
#: 四個角的風牌,依視角座位。
_CORNERS = (
    _on_panel(0, 235, 75, 290),
    _on_panel(360, 235, 435, 290),
    _on_panel(350, 0, 430, 50),
    _on_panel(5, 0, 60, 45),
)
#: 左上角面板下緣,緊貼著 ``x`` 後面的那一位數。
_KYOTAKU = (0.058, 0.128, 0.040, 0.038)
_HONBA = (0.128, 0.128, 0.024, 0.038)

#: 紅東的判定:紅色像素佔這一角的比例。另外三角是深藍底白字,實測接近 0。
_RED = 0.15


class HudDigits:
    """HUD 用的模板。一套數字一個子目錄,``assets/hud/<set>/<字>.png``。

    * ``score`` 四邊的金色點數。
    * ``wall`` 余幾張的青色數字。
    * ``round`` 第幾局的青色斜體。
    * ``stick`` 本場與供託的白色斜體。**只有 0 ~ 4**:素材裡沒見過更大的。
      5 以上會比不過門檻而回 ``None``,不會被說成別的數字。
    * ``wind`` 場風。**只有東**:現有素材都是東場。
    """

    def __init__(self, sets: dict[str, dict[str, np.ndarray]]) -> None:
        self._sets = sets

    @classmethod
    def load(cls, *, root: Path | None = None) -> HudDigits:
        """從 ``assets/hud/`` 載入。找不到就拋錯,不要安靜地什麼都讀不到。"""
        base = (root or ASSETS_DIR) / "hud"
        sets: dict[str, dict[str, np.ndarray]] = {}
        for folder in sorted(base.glob("*")):
            if not folder.is_dir():
                continue
            sets[folder.name] = {}
            for path in sorted(folder.glob("*.png")):
                face = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if face is None:
                    raise ValueError(f"讀不到模板 {path}")
                sets[folder.name][path.stem] = face.astype(np.float32)
        if not sets:
            raise FileNotFoundError(f"找不到 HUD 模板:{base}")
        return cls(sets)

    def best(self, name: str, patch: np.ndarray) -> str | None:
        """``patch`` 是這一套裡的哪個字;沒把握回 ``None``。"""
        ranked = sorted(
            ((_correlation(patch, face), label) for label, face in self._sets[name].items()),
            reverse=True,
        )
        score, label = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        return label if score >= _SCORE and score - runner_up >= _MARGIN else None


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float32).ravel()
    y = b.ravel()
    x = x - x.mean()
    y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6))


@dataclass(frozen=True, slots=True)
class Hud:
    """一張畫面讀到的 HUD。每個欄位讀不到就是 ``None``。

    Attributes:
        bakaze: 場風。現在只認得 ``E``。
        kyoku: 第幾局,1 起算。
        wall: 余幾張。
        honba: 本場數。
        kyotaku: 場上的立直棒數。
        scores: 四家點數,**依視角**:自己、下家、對家、上家。
        dealer: 莊家坐哪一邊,依視角(0 是自己)。
    """

    bakaze: str | None = None
    kyoku: int | None = None
    wall: int | None = None
    honba: int | None = None
    kyotaku: int | None = None
    scores: tuple[int | None, ...] = (None, None, None, None)
    dealer: int | None = None

    @property
    def scores_agree(self) -> bool:
        """點數對不對得起來:四家總和 + 立直棒 x 1000 = 開局總點數。

        任何一位數讀錯都會過不了,所以這個比信心門檻可靠得多。

        Note:
            前提是開局 25000 的四人場。友人場改過起始點數的話會永遠過不了 ——
            方向是「不敢用」,不是用錯。
        """
        if self.kyotaku is None or any(score is None for score in self.scores):
            return False
        total = sum(score for score in self.scores if score is not None)
        return total + self.kyotaku * 1000 == _TOTAL

    @property
    def own_seat(self) -> int | None:
        """自己的絕對座位。

        第 k 局的莊家是 k - 1 號座位,而 :attr:`dealer` 說他坐在自己的哪一邊。
        """
        if self.kyoku is None or self.dealer is None:
            return None
        return (self.kyoku - 1 - self.dealer) % 4


def _crop(image: np.ndarray, table: Rect, field: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = field
    return image[
        int(table.y + y * table.height) : int(table.y + (y + height) * table.height),
        int(table.x + x * table.width) : int(table.x + (x + width) * table.width),
    ]


def _digits(patch: np.ndarray) -> list[np.ndarray]:
    """切成一個一個數字,每個縮成 :data:`_DIGIT`。由左而右。"""
    if patch.size == 0:
        return []
    ink = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2] > _INK
    columns = [*ink.any(axis=0).tolist(), False]
    out = []
    start = None
    for x, filled in enumerate(columns):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            if x - start >= _MIN_WIDTH:
                rows = np.flatnonzero(ink[:, start:x].any(axis=1))
                glyph = patch[rows[0] : rows[-1] + 1, start:x]
                grey = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
                out.append(cv2.resize(grey, _DIGIT, interpolation=cv2.INTER_AREA))
            start = None
    return out


def _number(patch: np.ndarray, digits: HudDigits, name: str) -> int | None:
    """讀一個數字欄位。有一位認不出來就整個不算。"""
    found = [digits.best(name, glyph) for glyph in _digits(patch)]
    if not found or None in found:
        return None
    return int("".join(ch for ch in found if ch is not None))


def _score(image: np.ndarray, table: Rect, position: int, digits: HudDigits) -> int | None:
    patch = _crop(image, table, _SCORES[position])
    value = _number(np.rot90(patch, -_TURNS[position]), digits, "score")
    # 點數的最小單位是 100。尾數不對就是讀錯了(或那裡根本不是點數)。
    return value if value is not None and value % 100 == 0 else None


def _dealer(image: np.ndarray, table: Rect) -> int | None:
    """紅色的東在哪一角。不是剛好一角就棄權 —— 立繪也可能是紅的。"""
    red = []
    for corner in _CORNERS:
        hsv = cv2.cvtColor(_crop(image, table, corner), cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = ((hue <= 8) | (hue >= 172)) & (saturation >= 120) & (value >= 100)
        red.append(float(mask.mean()))
    lit = [position for position, share in enumerate(red) if share >= _RED]
    return lit[0] if len(lit) == 1 else None


def read_wall(image: np.ndarray, table: Rect, digits: HudDigits) -> int | None:
    """只讀余牌。一次約 0.13 毫秒,每一輪都讀也無所謂。"""
    return _number(_crop(image, table, _WALL), digits, "wall")


def read_hud(image: np.ndarray, table: Rect, digits: HudDigits) -> Hud:
    """讀一張畫面的 HUD。讀不到的欄位是 ``None``。"""
    wind = cv2.resize(
        cv2.cvtColor(_crop(image, table, _WIND), cv2.COLOR_BGR2GRAY),
        _DIGIT,
        interpolation=cv2.INTER_AREA,
    )
    return Hud(
        bakaze=digits.best("wind", wind),
        kyoku=_number(_crop(image, table, _ROUND), digits, "round"),
        wall=_number(_crop(image, table, _WALL), digits, "wall"),
        honba=_number(_crop(image, table, _HONBA), digits, "stick"),
        kyotaku=_number(_crop(image, table, _KYOTAKU), digits, "stick"),
        scores=tuple(_score(image, table, position, digits) for position in range(4)),
        dealer=_dealer(image, table),
    )
