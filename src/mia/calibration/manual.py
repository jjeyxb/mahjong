"""使用者自己框的手牌範圍 —— 不靠校正的那條路。

為什麼需要它
------------
牌桌校正有兩條路,兩條在 Steam 版上都不好走:

* **邊框剝除**是啟發式,實測會安靜地鎖進偏掉的矩形
  (見 :mod:`mia.calibration.canvas` 開頭的那張表)。
* **固定畫布**要求「畫布貼齊影像底部、佔滿整個寬度」,而那建立在
  「viewport 是我們自己下命令調出來的」之上。Steam 的 Unity 客戶端不經過
  瀏覽器,沒有人下得了那個命令 —— 實測視窗 1512x872、扣掉 28pt 標題列之後
  客戶區是 1512x844,比例 1.7915,**不是 16:9**,選單裡的 preset 沒有一個
  對得上。

框一次就好,而且它**不會錯得無聲無息** —— 框歪了使用者自己看得到。
這正是前兩條路最大的問題:它們錯的時候畫面上只是「沒反應」。

為什麼正規化到**擷取影像**而不是牌桌矩形
----------------------------------------
因為走這條路的時候根本沒有牌桌矩形 —— 不跑校正正是重點。存成影像的比例還
順便解決了縮放:遊戲畫面是跟著視窗等比變的,使用者把視窗拉大之後同一組比例
仍然框在手牌上,不必重框。

**不是萬無一失**:視窗比例變了(不是等比縮放)的話框就會歪,那時要重框一次。
但那是看得見的,而看得見的錯誤遠好過看不見的。

為什麼是可變物件
----------------
與 :class:`~mia.calibration.canvas.CanvasChoice` 同一個理由:讀它的是
擷取執行緒,而它是**延遲建構**的。傳值的話使用者重框之後,還沒建起來的那一邊
會拿到舊的框 —— 畫面照常更新,只是框在錯的地方。

這個格子**每一幀都被讀一次**,所以重框是立刻生效的,不必重啟執行緒。
"""

from __future__ import annotations

from mia.utils.geometry import NormRect, Rect, Size

__all__ = ["HandRoiChoice"]


class HandRoiChoice:
    """使用者框的手牌範圍,**一個可變的共用格子**。``None`` 代表沒框過。

    Args:
        rect: 初始值,``(x, y, 寬, 高)`` 四個 0~1 的比例。格式不對或超出邊界時
            當成沒框過 —— 那通常是從 ``ui_state.json`` 讀回來的舊值,
            而為了一個記住的框讓程式開不起來是不划算的交易。
    """

    def __init__(self, rect: tuple[float, float, float, float] | None = None) -> None:
        self._value: NormRect | None = None
        if rect is not None:
            self.set(rect)

    @property
    def value(self) -> NormRect | None:
        """現在的框。``None`` 代表沒框過,呼叫端該走校正那條路。"""
        return self._value

    def set(self, rect: tuple[float, float, float, float] | NormRect) -> bool:
        """換一個框。回傳**有沒有真的變**。

        收得下不合法的值(負寬高、超出邊界)並回 ``False`` —— 來源有兩個:
        磁碟上的舊設定,以及使用者在對話框裡點一下沒拖(寬高是 0)。
        兩個都不該當成錯誤,前者是正常的版本演進,後者是正常的誤觸。
        """
        try:
            norm = rect if isinstance(rect, NormRect) else NormRect(*rect)
        except (TypeError, ValueError):
            return False
        if norm == self._value:
            return False
        self._value = norm
        return True

    def clear(self) -> bool:
        """不要這個框了,回去走校正。回傳**有沒有真的變**。"""
        if self._value is None:
            return False
        self._value = None
        return True

    def pixels(self, image: Size) -> Rect | None:
        """這個框在某張影像上的像素位置。沒框過時是 ``None``。"""
        return None if self._value is None else self._value.to_pixels(image)

    def as_tuple(self) -> tuple[float, float, float, float] | None:
        """存進 ``ui_state.json`` 的形式。"""
        v = self._value
        return None if v is None else (v.x, v.y, v.width, v.height)

    def __str__(self) -> str:
        v = self._value
        if v is None:
            return "(沒框過)"
        return f"{v.width:.3f}x{v.height:.3f}@({v.x:.3f},{v.y:.3f})"

    def __repr__(self) -> str:
        return f"<HandRoiChoice {self}>"
