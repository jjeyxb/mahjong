"""遊戲是哪一版:網頁版還是 Steam 版。

與 :mod:`mia.features` 同樣是**沒有任何相依**的識別字模組,理由也一樣:
:mod:`mia.ui`(呈現)與 :mod:`mia.live`(執行)都要用同一組名字,而那兩邊
誰都不該 import 對方。

為什麼這是一個「版本」而不是一個設定項
--------------------------------------
兩版的差別不是偏好,是**資料來源少了一條**:

======== ================== ==========================================
版本     畫面               封包
======== ================== ==========================================
網頁版   擷取瀏覽器視窗     CDP 攔 WebSocket
Steam    擷取遊戲視窗       **沒有** —— Unity 客戶端不經過瀏覽器
======== ================== ==========================================

:data:`SUPPORTED` 寫的是每一版撐得起什麼。沒有來源的功能,UI 照它把開關畫成停用
並說明原因 —— 可按而按了沒反應,看起來就是壞掉。

**2026-10-25:Steam 版拿到放銃分析了。** 這一段原本寫著「Steam 版的事件流
之後會有,接上之後這張表就會改」—— 現在改了。

:data:`~mia.features.DANGER` 有兩條路,而**純畫面那條不碰封包**:牌河、
寶牌、立直、局的邊界全部從螢幕讀(見 ``docs/cv_mjai.md``)。所以它在兩版上
都能跑,只是 Steam 版上只有那一條。走哪條由
:class:`~mia.features.DangerSourceChoice` 決定,使用者看到的是一個開關。

**2026-10-31:Steam 版也有 AI 建議了。** 它要完整的事件流,而事件流現在可以只靠
畫面組出來(見 :mod:`mia.tracker.stream` 與 :mod:`mia.live.advice`)。所以這張表
目前沒有任何版本缺功能,但機制留著:新的功能未必兩版都有來源。

為什麼不用 enum
---------------
與 :mod:`mia.features` 一致:這些字串會寫進 ``data/ui_state.json``,
而那個檔案要跨版本讀得回來。字串本身因此**不要改**。
"""

from __future__ import annotations

from mia.features import ADVICE, DANGER, VISION

__all__ = [
    "NAMES",
    "PLATFORMS",
    "STEAM",
    "SUPPORTED",
    "WEB",
    "PlatformChoice",
    "supports",
    "why_unsupported",
]

#: 網頁版雀魂,跑在 MIA 自己開的 Chromium 裡。
WEB = "web"

#: Steam 上的 Unity 客戶端。
STEAM = "steam"

#: 選單的順序。網頁版排前面 —— 它是功能完整的那一個。
PLATFORMS = (WEB, STEAM)

#: 給人看的名字。
NAMES = {WEB: "網頁版", STEAM: "Steam 版"}

#: 每一版撐得起哪些功能。見模組說明。
SUPPORTED = {
    WEB: (VISION, ADVICE, DANGER),
    STEAM: (VISION, ADVICE, DANGER),
}

#: 功能在某一版上不能用的理由。**要寫給使用者看**,所以講的是「為什麼」
#: 而不是「不支援」—— 後者會讓人以為是 MIA 還沒做,而去找設定裡的開關。
_REASONS: dict[str, str] = {}


def supports(platform: str, feature: str) -> bool:
    """這一版有沒有這個功能的資料來源。

    認不得的版本一律回 ``True``:這個函式的作用是**擋掉確定沒有來源的組合**,
    而不是當白名單。設定檔或 ``ui_state.json`` 裡出現了沒見過的名字時,
    放行比整個功能消失好 —— 後者沒有任何錯誤訊息。
    """
    allowed = SUPPORTED.get(platform)
    return True if allowed is None else feature in allowed


def why_unsupported(platform: str, feature: str) -> str | None:
    """為什麼這一版沒有這個功能。支援的話回 ``None``。"""
    if supports(platform, feature):
        return None
    return _REASONS.get(platform, f"{NAMES.get(platform, platform)}沒有這個功能的資料來源")


class PlatformChoice:
    """使用者現在選的版本,**一個可變的共用格子**。

    與 :class:`~mia.calibration.canvas.CanvasChoice`、
    :class:`~mia.engine.styles.StyleChoice` 同一個模式,理由也一樣:讀它的是
    **延遲建構**的工作執行緒(功能打開才建)。傳值的話使用者改了版本之後,
    還沒建起來的那一邊會拿到舊值,而那不會有任何症狀 —— 畫面照常更新,
    只是對著另一個視窗在認牌。

    Args:
        value: 初始版本。認不得的名字(``ui_state.json`` 裡是舊版本留下的)
            退回 :data:`WEB`,不報錯 —— 為了一個記住的選擇讓程式開不起來
            是不划算的交易。
    """

    def __init__(self, value: str = WEB) -> None:
        self._value = value if value in PLATFORMS else WEB

    @property
    def value(self) -> str:
        return self._value

    @property
    def name(self) -> str:
        """給人看的名字。"""
        return NAMES.get(self._value, self._value)

    def set(self, value: str) -> bool:
        """換版本。回傳**有沒有真的變**,讓呼叫端決定要不要重啟工作執行緒。

        換版本要重找視窗、重新校正,成本不低,而使用者在選單上重選同一項
        是很常見的動作。
        """
        if value not in PLATFORMS or value == self._value:
            return False
        self._value = value
        return True

    def supports(self, feature: str) -> bool:
        """這一版有沒有這個功能的資料來源。"""
        return supports(self._value, feature)

    def why_unsupported(self, feature: str) -> str | None:
        return why_unsupported(self._value, feature)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<PlatformChoice {self._value}>"
