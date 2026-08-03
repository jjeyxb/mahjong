"""固定畫布尺寸:把校正從「猜」換成「先指定再驗證」。

為什麼需要這個
--------------
:mod:`mia.calibration.letterbox` 的剝除是啟發式的,而它會在某些視窗尺寸下
**安靜地給出錯的答案**。實測(2026-08-01,同一天錄的兩段素材、同一台機器、
同一個瀏覽器):

===================  =============  ==============================
視窗(邏輯像素)      單幀校正通過率  最常見的候選
===================  =============  ==============================
1268x804                   98 / 100  ``2536x1430@(0,174)``  ← 正確
900x593                     9 / 100  ``1798x1152@(1,27)``   ← 工具列整條沒剝掉
===================  =============  ==============================

視窗變窄時,分頁標題與紅綠燈按鈕這些**固定寬度**的東西在一整列裡佔的比例
變大,那一列就過不了 ``border_uniformity`` 的門檻,剝除停在瀏覽器工具列裡面。
於是 91% 的候選因為寬高比 1.56 被丟掉 —— 而**能通過寬高比檢查的,恰恰就是
那些剝過頭、剝進遊戲畫布裡的幀**。寬高比檢查因此反過來在挑錯的答案,
:class:`~mia.calibration.stable.StableCalibrator` 取的中位數是一組有系統性
偏差的離群值。

那次錄下來的 ``table_rect`` 是 ``1712x968@(69,218)``,真值是
``1800x1012@(0,174)`` —— 位移約 0.7 張牌寬。手牌辨識的每張牌正確率
從 96.0% 掉到 29.1%,而過程中沒有任何錯誤訊息。

掃過 ``border_uniformity`` 0.5~0.85 × ``border_tolerance`` 20/40 共八組,
1268x804 有四組誤差在 16px 內,900x593 **沒有任何一組小於 131px**。
這不是門檻沒調好:瀏覽器的深色工具列與雀魂的深色牌桌之間本來就沒有對比,
剝除在那個尺寸下結構性地找不到邊界。

作法
----
瀏覽器是 MIA 自己開的(見 :meth:`~mia.live.runtime.LiveRuntime.start_game`),
所以尺寸不必用猜的 —— **直接下命令**。使用者選一個畫布尺寸,
``tools/gt.py cdp --canvas`` 把瀏覽器視窗調到讓**頁面 viewport 剛好等於**
那個尺寸,``table_rect`` 就直接算得出來,完全不跑剝除。

要選的是**畫布**而不是視窗:視窗 1920x1080 扣掉瀏覽器工具列之後 viewport
是 1920x(1080-工具列高),那不是 16:9,雀魂會自己補黑邊 —— 又回到要偵測邊界。
所以命令的對象是 viewport,視窗高度由程式加上工具列高算出來。

推導出來的矩形仍然要驗
----------------------
指定了不代表就對:使用者可能事後手動拖過視窗、瀏覽器可能因為螢幕放不下而
拒絕那個尺寸。:meth:`Canvas.table_rect` 會回報這些不一致,寧可說「對不上」
也不要給一個看起來很正常的錯矩形 —— 那正是這個模組要消滅的失敗模式。
"""

from __future__ import annotations

from dataclasses import dataclass

from mia.utils.geometry import Rect, Size

__all__ = ["PRESETS", "Canvas", "CanvasChoice", "CanvasMismatchError"]

#: 擷取影像寬度與畫布寬度容許差這麼多像素。
#:
#: 理想是 0 —— 桌面版 Chrome 的視窗左右沒有邊框,viewport 寬度就是視窗寬度。
#: 留 2px 是給 devicePixelRatio 不是整數時的四捨五入(縮放 150% 的螢幕)。
SIDE_TOLERANCE = 2

#: 畫布上方的瀏覽器介面高度(邏輯像素)的合理範圍。
#:
#: 實測 macOS Chrome for Testing 是 87(分頁列 + 網址列,沒有書籤列)。
#: 上限放寬到 240 是為了容納書籤列、擴充套件列、以及 Windows 的視窗標題列。
#: 落在範圍外代表「這張影像的高度跟選的畫布配不起來」,通常是選錯尺寸。
CHROME_RANGE = (0, 240)


class CanvasMismatchError(ValueError):
    """擷取到的影像與指定的畫布尺寸對不上。"""


@dataclass(frozen=True, slots=True)
class Canvas:
    """一個指定的遊戲畫布尺寸,單位是**邏輯(CSS)像素**。

    Attributes:
        width: 畫布寬。
        height: 畫布高。
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"畫布尺寸必須為正:{self.width}x{self.height}")

    @property
    def key(self) -> str:
        """存進設定檔與 UI 狀態的字串形式,例如 ``1920x1080``。"""
        return f"{self.width}x{self.height}"

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def label(self) -> str:
        """給人看的。用 ``×`` 而不是 ``x``,與設定檔裡的鍵區分開。"""
        return f"{self.width}×{self.height}"

    @classmethod
    def parse(cls, text: str | None) -> Canvas | None:
        """從 ``"1920x1080"`` 解析。``None``、空字串、格式不對都回 ``None``。

        **不拋例外**:這個字串來自設定檔或使用者狀態檔,兩者都可能被手改壞,
        而「畫布尺寸看不懂」的正確反應是退回自動偵測,不是讓程式起不來。
        """
        if not text:
            return None
        parts = text.lower().replace("×", "x").split("x")
        if len(parts) != 2:
            return None
        try:
            width, height = (int(p.strip()) for p in parts)
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        return cls(width, height)

    def table_rect(self, image_size: Size, scale: float = 1.0) -> Rect:
        """畫布在擷取影像中的像素矩形。

        Args:
            image_size: 擷取到的整張視窗影像尺寸(像素)。
            scale: :attr:`~mia.capture.base.Frame.scale`,pixel / logical。
                Retina 是 2.0。

        Returns:
            畫布矩形。

        Raises:
            CanvasMismatchError: 影像與這個畫布尺寸配不起來。

        Note:
            **畫布貼齊影像底部、佔滿整個寬度。** 桌面版瀏覽器的介面全在上方,
            左右與下方沒有東西,所以 viewport 的下緣就是視窗的下緣。這個假設
            只有在 viewport 真的是 16:9 時才成立 —— 否則雀魂會補黑邊,畫布就
            不再貼齊。而 viewport 是我們自己下命令調出來的,所以它成立。
        """
        width = round(self.width * scale)
        height = round(self.height * scale)
        side = image_size.width - width
        chrome = (image_size.height - height) / scale

        if abs(side) > SIDE_TOLERANCE:
            raise CanvasMismatchError(
                f"影像寬 {image_size.width}px 與畫布 {self.label} @{scale:g}x "
                f"(= {width}px)差了 {side} px。"
                "視窗大小可能被手動改過,或這一場不是用這個畫布尺寸開的。"
            )
        if not CHROME_RANGE[0] <= chrome <= CHROME_RANGE[1]:
            raise CanvasMismatchError(
                f"畫布 {self.label} 上方剩 {chrome:.0f} 個邏輯像素給瀏覽器介面,"
                f"不在合理範圍 {CHROME_RANGE[0]}~{CHROME_RANGE[1]}。"
                "通常是畫布尺寸選得比視窗還大(瀏覽器裝不下就不會照做)。"
            )
        # 左右若差 1~2px 就置中,不要讓誤差全堆到某一邊
        return Rect(max(0, side // 2), image_size.height - height, width, height)


#: 選單裡提供的畫布尺寸。全部是 16:9 —— 雀魂的設計比例,非 16:9 一定補黑邊。
#:
#: 由小到大排:1280x720 幾乎任何螢幕都放得下,2560x1440 需要 4K 或 Retina。
#: 選一個放不下的尺寸不會壞掉,瀏覽器會拒絕、:meth:`Canvas.table_rect` 會
#: 當場說對不上 —— 那正是這整個模組存在的意義。
PRESETS: tuple[Canvas, ...] = (
    Canvas(1280, 720),
    Canvas(1600, 900),
    Canvas(1920, 1080),
    Canvas(2560, 1440),
)


class CanvasChoice:
    """使用者現在選的畫布尺寸,**一個可變的共用格子**。

    為什麼是一個可變物件而不是傳值:這個選擇同時被兩個子系統讀 ——
    擷取子程序的命令列(開瀏覽器時要下的尺寸)與 :class:`VisionWorker`
    的校正設定 —— 而兩者都是**延遲建構**的(功能打開才建)。傳值的話
    UI 改了之後,還沒建起來的那一邊會拿到舊值,而那不會有任何症狀。

    ``None`` 表示自動偵測,也就是原本的剝除啟發式。
    """

    def __init__(self, canvas: Canvas | None = None) -> None:
        self.value = canvas

    @property
    def key(self) -> str | None:
        """字串形式,自動偵測時為 ``None``。可直接存進設定檔或狀態檔。"""
        return self.value.key if self.value else None

    def set(self, key: str | None) -> bool:
        """依字串設定。回傳**有沒有真的變**,讓呼叫端決定要不要重啟。"""
        canvas = Canvas.parse(key)
        if canvas == self.value:
            return False
        self.value = canvas
        return True

    def __str__(self) -> str:
        return self.value.label if self.value else "自動偵測"
