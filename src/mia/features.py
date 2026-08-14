"""三個功能的識別字。

放在一個沒有任何相依的模組裡,是為了讓 :mod:`mia.ui`(呈現)與
:mod:`mia.live`(執行)都能用同一組名字,而不必互相 import ——
UI 不該知道 live 的存在(它也要服務錄影重播),live 不該知道 Qt 的存在。

字串本身會出現在設定與日誌裡,所以**不要改**。
"""

from __future__ import annotations

__all__ = ["ADVICE", "DANGER", "NAMES", "VISION"]

#: 功能 1:螢幕擷取 → 手牌 → 向聽 / 進張。
VISION = "vision"

#: 功能 2:封包 → MJAI → 引擎建議。
ADVICE = "advice"

#: 功能 3:封包 → 桌面狀態 → 每張牌切出去的危險度。
#:
#: 與 :data:`ADVICE` 同樣吃封包,但**不載引擎** —— 它要的是事件流本身。
#: 分開兩個識別字是為了讓使用者能只開這個:載 130MB 權重與看放銃危險度
#: 是兩件不同代價的事。
DANGER = "danger"

#: 給人看的名字。狀態列與開關的標籤都取這裡,兩處不會寫得不一樣。
NAMES = {VISION: "畫面辨識", ADVICE: "AI 建議", DANGER: "放銃分析"}
