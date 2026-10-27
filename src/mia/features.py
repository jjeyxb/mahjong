"""三個功能的識別字。

放在一個沒有任何相依的模組裡,是為了讓 :mod:`mia.ui`(呈現)與
:mod:`mia.live`(執行)都能用同一組名字,而不必互相 import ——
UI 不該知道 live 的存在(它也要服務錄影重播),live 不該知道 Qt 的存在。

字串本身會出現在設定與日誌裡,所以**不要改**。
"""

from __future__ import annotations

__all__ = ["ADVICE", "DANGER", "NAMES", "VISION", "DangerSourceChoice"]

#: 功能 1:螢幕擷取 → 手牌 → 向聽 / 進張。
VISION = "vision"

#: 功能 2:封包 → MJAI → 引擎建議。
ADVICE = "advice"

#: 功能 3:桌面狀態 → 每張牌切出去的危險度。
#:
#: **有兩條路**:封包那條(準,只有網頁版有)與純畫面那條(兩版都能跑)。
#: 使用者看到的是一個開關 —— 走哪條由 :class:`DangerSourceChoice` 決定。
#:
#: 與 :data:`ADVICE` 的封包那條同源,但**不載引擎** —— 它要的是事件流本身。
#: 分開兩個識別字是為了讓使用者能只開這個:載 130MB 權重與看放銃危險度
#: 是兩件不同代價的事。
DANGER = "danger"

#: 給人看的名字。狀態列與開關的標籤都取這裡,兩處不會寫得不一樣。
NAMES = {VISION: "畫面辨識", ADVICE: "AI 建議", DANGER: "放銃分析"}


class DangerSourceChoice:
    """放銃分析要走哪一條路 —— **一個可變的共用格子**。

    :data:`DANGER` 有兩種實作,算的是同一件事、用同一支 ``assess``,
    差別只在資訊從哪來:

    ============  ====================  ==========================
    路            資料                  代價
    ============  ====================  ==========================
    封包          CDP 攔 WebSocket      準,但只有網頁版有
    純畫面        只看螢幕              約四成的牌讀不出來,兩版都能跑
    ============  ====================  ==========================

    **這個選擇預設不該丟給使用者。** 他要的是「告訴我哪張牌危險」,不是
    「選一個資料來源」—— 有封包就用封包,沒有就用畫面,程式自己知道。
    所以這不是兩個開關,是一個開關加一個**強制**選項(給開發與比較用)。

    曾經做成兩個開關,而兩個同時打開會互相覆蓋(它們投到同一個格子),
    畫面上的危險度在兩種答案之間跳,使用者完全看不出為什麼。

    與 :class:`~mia.platforms.PlatformChoice` 同一個模式,理由也一樣:
    讀它的是**延遲建構**的工作執行緒(功能打開才建)。
    """

    def __init__(self, vision_only: bool = False) -> None:
        self._vision_only = vision_only

    @property
    def vision_only(self) -> bool:
        """使用者有沒有強制走純畫面那條。"""
        return self._vision_only

    def set(self, on: bool) -> bool:
        """改選。回傳**有沒有真的變**,讓呼叫端決定要不要重啟工作執行緒。"""
        if on == self._vision_only:
            return False
        self._vision_only = on
        return True
