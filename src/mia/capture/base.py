"""擷取層的共用型別與抽象基底。

設計要點:**以「視窗」為擷取單位,而不是螢幕矩形。**

原因是本專案要在遊戲畫面上疊一層 HUD。若以螢幕矩形擷取,自己的 Overlay 會被
一起拍進去餵給 OpenCV,造成辨識污染。指定視窗擷取則是取該視窗自身的繪圖緩衝區,
上層蓋了什麼都與擷取結果無關。

(實測:macOS 26.5.2 上,被完全遮擋的背景視窗仍可正確擷取到完整內容。)

:class:`MSSCaptureBackend` 是唯一的例外 —— 它只能抓螢幕區域,因此在該模式下
Overlay 必須避開辨識區域。它的定位純粹是 fallback。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from mia.utils.geometry import Rect, Size
from mia.utils.logging import logger

__all__ = [
    "BackendUnavailableError",
    "CaptureBackend",
    "CaptureError",
    "CaptureFailedError",
    "Frame",
    "PermissionDeniedError",
    "WindowInfo",
    "WindowNotFoundError",
]


# --------------------------------------------------------------------------- 例外


class CaptureError(Exception):
    """擷取層錯誤的共同基底。"""


class BackendUnavailableError(CaptureError):
    """後端在此平台不可用(缺少相依套件或作業系統不支援)。"""


class PermissionDeniedError(CaptureError):
    """缺少螢幕錄製權限。"""


class WindowNotFoundError(CaptureError):
    """找不到符合條件的視窗。"""


class CaptureFailedError(CaptureError):
    """視窗存在,但擷取當下失敗(例如視窗剛被關閉或最小化)。"""


# --------------------------------------------------------------------------- 資料型別


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """一個可擷取視窗的描述。

    Attributes:
        handle: 平台原生識別碼。macOS 為 ``CGWindowID``,Windows 為 ``HWND``。
        title: 視窗標題。macOS 上取得標題需要螢幕錄製權限,未授權時會是空字串。
        owner: 擁有此視窗的程式名稱。
        bounds: 視窗在螢幕上的位置與大小,**邏輯座標**(見 utils.geometry 的說明)。
        pid: 擁有此視窗的行程 ID。
    """

    handle: int
    title: str
    owner: str
    bounds: Rect
    pid: int = 0

    def matches(self, patterns: Iterable[str], *, fields: str = "both") -> bool:
        """大小寫不敏感的子字串比對,任一 pattern 命中即為 True。

        Args:
            patterns: 要比對的字串們。空的可迭代物件一律回傳 False。
            fields: ``"title"`` / ``"owner"`` / ``"both"``。
        """
        haystacks: list[str] = []
        if fields in ("title", "both"):
            haystacks.append(self.title.casefold())
        if fields in ("owner", "both"):
            haystacks.append(self.owner.casefold())
        return any(p.casefold() in h for p in patterns if p for h in haystacks)

    def __str__(self) -> str:
        return f"[{self.handle}] {self.owner or '?'} — {self.title or '(無標題)'} {self.bounds}"


@dataclass(frozen=True, slots=True)
class Frame:
    """一次擷取的結果。

    Attributes:
        image: BGR uint8 影像,shape 為 (H, W, 3)。**像素座標空間。**
        window: 擷取當下的視窗資訊,其 ``bounds`` 為邏輯座標。
        scale: pixel / logical 的比值。macOS Retina 為 2.0,一般螢幕為 1.0。
            把邏輯座標換算成影像上的像素位置時要乘上這個值。
        captured_at: :func:`time.monotonic` 時間戳。用單調時鐘而非 wall clock,
            因為它的用途是計算幀間隔,不可被系統校時影響。
    """

    image: np.ndarray
    window: WindowInfo
    scale: float = 1.0
    captured_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        img = self.image
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Frame.image 必須是 (H, W, 3) 的 BGR 影像,收到 {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"Frame.image 必須是 uint8,收到 {img.dtype}")

    @property
    def size(self) -> Size:
        h, w = self.image.shape[:2]
        return Size(w, h)

    @property
    def rect(self) -> Rect:
        """整張影像的矩形,原點為 (0, 0)。"""
        return Rect.from_size(self.size)

    def crop(self, rect: Rect) -> np.ndarray:
        """依像素矩形裁切。超出邊界的部分會先被夾到影像範圍內。

        回傳的是 view 而非複本 —— 呼叫端若要修改請自行 ``.copy()``。
        """
        clipped = rect.intersect(self.rect)
        if clipped is None:
            raise ValueError(f"裁切矩形 {rect} 完全落在影像 {self.size} 之外")
        return self.image[clipped.as_slice()]


# --------------------------------------------------------------------------- 後端


class CaptureBackend(ABC):
    """擷取後端的抽象基底。

    子類別只需實作 :meth:`list_windows` 與 :meth:`capture`;
    :meth:`find_window` 的挑選邏輯在此共用。
    """

    name: ClassVar[str] = "base"

    @staticmethod
    def is_available() -> bool:
        """此後端在當前平台是否可用。不應拋出例外。"""
        return False

    @abstractmethod
    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:
        """列出可擷取的視窗。

        Args:
            include_all: 預設只回傳一般應用程式視窗(排除選單列、Dock、
                控制中心那類系統浮層)。設為 True 則不過濾,除錯時用。
        """

    @abstractmethod
    def capture(self, window: WindowInfo) -> Frame:
        """擷取指定視窗。

        Raises:
            CaptureFailedError: 視窗已消失或系統回傳空影像。
        """

    # 比對強度權重。程式名稱比視窗標題可靠得多 —— 標題是使用者內容,
    # 會出現「在編輯器裡打開一個檔名含『雀魂』的檔案」這種假陽性
    # (實測就撞過:VS Code 的標題含遊戲名,且視窗比遊戲還大)。
    _SCORE_OWNER_MATCH = 8  # 程式名就是遊戲本身,最強的訊號
    _SCORE_TITLE_EXACT = 6
    _SCORE_TITLE_DOMINANT = 4  # pattern 佔了標題的一大半 —— 那是「標題就是遊戲名」
    _SCORE_HOST_MATCH = 2  # 瀏覽器承載,且標題已命中
    _SCORE_TITLE_SUBSTRING = 1

    #: 命中的 pattern 至少要佔標題這個比例,才算 :data:`_SCORE_TITLE_DOMINANT`。
    #:
    #: 實測的兩個對手:遊戲視窗標題是「雀魂麻將」(pattern「雀魂」佔 50%),
    #: 而 Safari 開著本專案的 GitHub 頁時標題有 80 幾個字(佔 2%)。
    #: 門檻取 0.25 —— 遊戲那邊還有一倍餘裕,而「標題裡順帶提到遊戲名」的
    #: 那類視窗(檔案路徑、網頁標題、聊天室訊息)幾乎不可能這麼短。
    _TITLE_DOMINANT_RATIO = 0.25

    @classmethod
    def score_window(
        cls,
        window: WindowInfo,
        title_patterns: Sequence[str],
        owner_patterns: Sequence[str],
        host_patterns: Sequence[str] = (),
    ) -> int:
        """替候選視窗評分,分數越高越可能是目標。0 表示完全不符。

        Args:
            host_patterns: 可能承載網頁版雀魂的瀏覽器程式名。**只有在標題也命中時
                才加分**;否則任何瀏覽器視窗都會拿到分數而被誤選。

        單靠標題比對不夠力:實測時 VS Code 開著檔名含「雀魂」的檔案,與真正跑著
        網頁版雀魂的 Chromium **拿到完全相同的分數**(兩者 owner 都不符,都只命中
        標題子字串),然後同分時比面積,較大的 VS Code 就贏了。瀏覽器加分正是為了
        拆開這種平手。

        **但那招在對手也是瀏覽器時就沒用了。** 2026-08-04 實測撞到:Safari 開著
        本專案的 GitHub 頁,而 repo 描述裡有「雀魂」兩個字 —— 它與真正的遊戲視窗
        同樣拿到「標題子字串 + 瀏覽器」共 3 分,然後面積比遊戲大,於是贏了。
        整個功能 1 對著一個網頁跑 CV,而畫面上只是「沒反應」。

        所以標題分數改成**看覆蓋率**:命中的 pattern 佔標題越大一塊,越可能
        「這個標題就是遊戲名」而不是「這個標題順帶提到遊戲名」。這個判據不必
        列舉任何特定程式,對編輯器、瀏覽器、聊天軟體一視同仁。
        """
        score = 0
        if window.matches(owner_patterns, fields="owner"):
            score += cls._SCORE_OWNER_MATCH

        score += cls._title_score(window.title, title_patterns)
        if score and window.matches(host_patterns, fields="owner"):
            score += cls._SCORE_HOST_MATCH
        return score

    @classmethod
    def _title_score(cls, raw_title: str, title_patterns: Sequence[str]) -> int:
        """標題像不像「這個視窗就是遊戲」。

        取**最長的**命中 pattern 來算覆蓋率:多個 pattern 都命中時,長的那個
        說明得更多(「maj-soul」比「雀魂」更難是巧合)。
        """
        title = raw_title.casefold()
        if not title:
            return 0
        hits = [p.casefold() for p in title_patterns if p and p.casefold() in title]
        if not hits:
            return 0
        longest = max(hits, key=len)
        if title == longest:
            return cls._SCORE_TITLE_EXACT
        if len(longest) >= len(title) * cls._TITLE_DOMINANT_RATIO:
            return cls._SCORE_TITLE_DOMINANT
        return cls._SCORE_TITLE_SUBSTRING

    def find_window(
        self,
        title_patterns: Sequence[str] = (),
        owner_patterns: Sequence[str] = (),
        host_patterns: Sequence[str] = (),
        *,
        min_width: int = 1,
        min_height: int = 1,
    ) -> WindowInfo:
        """找出最可能是目標的視窗。

        依 :meth:`score_window` 評分排序,同分時取面積最大的
        —— 遊戲視窗通常是同類候選中最大的那個。

        Raises:
            WindowNotFoundError: 沒有候選。錯誤訊息會附上實際看到的視窗清單,
                方便直接判斷是 pattern 寫錯還是遊戲根本沒開。
        """
        windows = self.list_windows()
        scored = [
            (self.score_window(w, title_patterns, owner_patterns, host_patterns), w)
            for w in windows
            if w.bounds.width >= min_width and w.bounds.height >= min_height
        ]
        candidates = [(s, w) for s, w in scored if s > 0]
        if not candidates:
            seen = "\n".join(f"    {w}" for w in windows) or "    (沒有任何可擷取的視窗)"
            raise WindowNotFoundError(
                f"找不到符合條件的視窗。\n"
                f"  標題 patterns: {list(title_patterns)}\n"
                f"  程式 patterns: {list(owner_patterns)}\n"
                f"  最小尺寸: {min_width}x{min_height}\n"
                f"  目前看得到的視窗:\n{seen}"
            )
        score, chosen = max(candidates, key=lambda pair: (pair[0], pair[1].bounds.area))
        if score <= self._SCORE_TITLE_SUBSTRING:
            # 只靠標題子字串命中,程式名毫無旁證 —— 編輯器開著檔名含「雀魂」的檔案
            # 就會落在這一類。仍然回傳(可能真的是目標),但要讓使用者知道。
            logger.warning(
                "只憑標題子字串選中 {} —— 程式名並不像雀魂或瀏覽器。"
                "若不是想要的視窗,請用 --window 明確指定。",
                chosen,
            )
        return chosen

    def close(self) -> None:  # noqa: B027 - 刻意的非抽象空實作
        """釋放資源。

        預設不做事 —— 多數後端沒有需要釋放的東西,不該強迫每個子類別都寫一個
        空方法。有資源要收的(如 :class:`~mia.capture.mss_fallback.MSSCaptureBackend`
        持有的 mss session)再覆寫。
        """

    def __enter__(self) -> CaptureBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
