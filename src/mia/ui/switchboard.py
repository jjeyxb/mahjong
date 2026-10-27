"""「能開關功能的東西」—— UI 這一層對 :mod:`mia.live` 的全部要求。

宣告成 Protocol 而不是直接吃一個 :class:`~mia.live.runtime.LiveRuntime`:
UI 也要服務錄影重播與示範資料,那些模式裡根本沒有工作執行緒可以開關。

放在單獨一個模組,是因為側邊視窗與 Overlay **都**要用它。留在
``ui/panel/window.py`` 的話,Overlay 就得去 import 側邊視窗才拿得到型別 ——
那兩個是平行的呈現方式,誰都不該相依誰。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mia.capture.base import Frame
    from mia.utils.geometry import NormRect

__all__ = [
    "CanvasPicker",
    "GameLauncher",
    "HandRoiPicker",
    "PlatformPicker",
    "StylePicker",
    "Switchboard",
    "TableRectPicker",
    "is_on",
]


class Switchboard(Protocol):
    """只宣告 UI 真正用到的三個方法。"""

    def available(self, key: str) -> bool: ...
    def is_enabled(self, key: str) -> bool: ...
    def set_enabled(self, key: str, on: bool) -> None: ...


class GameLauncher(Protocol):
    """能開一場遊戲的東西。

    與 :class:`Switchboard` 分開,因為那是**兩個層級的事**:開關管的是「這個
    功能要不要跑」,這個管的是「有沒有一場遊戲可以看」。
    :class:`~mia.live.runtime.LiveRuntime` 兩個都實作,但重播與示範模式兩個
    都沒有,而 ``--tail``(跟著別人正在錄的檔案走)只有前者。
    """

    #: 這個模式下 MIA 到底負不負責開遊戲。與 :meth:`Switchboard.available`
    #: 同一個用意:「不存在」與「存在但沒在跑」是兩件事,UI 要分得出來才能把
    #: 按鈕畫成停用而不是可按 —— 可按而按了沒事發生,看起來就是壞掉。
    def can_start_game(self) -> bool: ...
    def game_running(self) -> bool: ...
    def start_game(self) -> None: ...


class PlatformPicker(Protocol):
    """能決定玩的是哪一版的東西(網頁版 / Steam 版)。

    又是一個獨立的 Protocol,理由與 :class:`CanvasPicker` 相同:錄影重播與
    示範資料裡沒有「現在玩的是哪一版」這回事 —— 錄的時候就定了。

    ⚠ **換版本會關掉新版本撐不起的功能,並重啟畫面辨識。** Steam 版沒有封包
    來源,所以 AI 建議與放銃分析整個沒有東西可吃(見 :mod:`mia.platforms`)。
    所以 :meth:`set_platform` 在「選的跟現在一樣」時什麼都不該做。
    """

    def can_pick_platform(self) -> bool: ...
    def platform(self) -> str: ...
    def set_platform(self, value: str) -> None: ...


class HandRoiPicker(Protocol):
    """能讓使用者自己框手牌範圍的東西。

    為什麼需要手動框,見 :mod:`mia.ui.panel.handroi`:Steam 版的畫布不是
    16:9,選單裡的 preset 沒有一個對得上,而自動剝除會**安靜地**鎖錯。

    介面切在這裡的理由:**對話框是 Qt 的,抓畫面不是。** 側邊視窗負責開對話框
    (它本來就在 UI 執行緒上),而「去哪抓一張畫面」「框完存到哪」是接線層的事
    —— 與 :class:`CanvasPicker` 把持久化留在 ``tools/ui.py`` 是同一個分法。

    :meth:`grab_frame` 是**同步**的,實測 10~30 ms,直接在 UI 執行緒上做。
    它會丟 :class:`~mia.capture.base.WindowNotFoundError` 或
    :class:`~mia.capture.base.CaptureError` —— 兩個都是使用者救得回來的狀況
    (遊戲沒開、沒給螢幕錄製權限),呼叫端要接住並說人話。
    """

    def can_pick_hand_roi(self) -> bool: ...
    def grab_frame(self) -> Frame: ...
    def hand_roi(self) -> NormRect | None: ...
    def set_hand_roi(self, rect: NormRect | None) -> None: ...


class TableRectPicker(Protocol):
    """能讓使用者自己框**整個遊戲畫面**的東西。

    與 :class:`HandRoiPicker` 是兩件事,不要混:

    * 手牌範圍**取代**校正 —— 框了之後那條路上根本不算牌桌矩形。
    * 牌桌矩形**就是**校正的結果 —— ROI 與四家牌河的角點仍然照常相對它推導,
      只是那個矩形改成人給的。

    所以它影響的東西多得多(見 :mod:`mia.ui.panel.tablerect`),而這也是
    它值得一個獨立 Protocol 而不是塞進 :class:`HandRoiPicker` 的理由 ——
    兩個能不能用的條件不一樣:手牌範圍在 Steam 版上是主要那條路,牌桌矩形
    則是自動校正走不通時才動的。

    :meth:`grab_frame` 與 :class:`HandRoiPicker` 的同名方法是同一件事,
    同一個接線類別通常兩個都實作。
    """

    def can_pick_table_rect(self) -> bool: ...
    def grab_frame(self) -> Frame: ...
    def table_rect(self) -> NormRect | None: ...
    def set_table_rect(self, rect: NormRect | None) -> None: ...


class CanvasPicker(Protocol):
    """能決定遊戲畫布尺寸的東西。

    又是一個獨立的 Protocol 而不是塞進 :class:`Switchboard`:選畫布只對
    「MIA 自己開瀏覽器 + 畫面辨識有跑」的組合有意義。錄影重播的畫布尺寸
    是錄的時候就定死的,``--no-vision`` 則根本沒有人要用這個矩形。
    """

    def can_pick_canvas(self) -> bool: ...
    def canvas(self) -> str | None: ...
    def set_canvas(self, key: str | None) -> None: ...


class StylePicker(Protocol):
    """能換打法風格的東西。

    又是一個獨立的 Protocol,理由與 :class:`CanvasPicker` 相同:換風格只在
    「AI 建議這條路存在」時有意義。錄影重播與示範資料裡引擎是命令列決定的,
    ``--no-packets`` 則根本沒有引擎。

    ⚠ **換風格會把引擎整個重啟**(關掉再開,重載權重)。這不是可以順手多按
    幾次的操作 —— 每份權重 130MB,實測 0.5~0.6 秒,而且重啟期間那一巡的建議
    會斷掉。所以 :meth:`set_style` 在「選的跟現在一樣」時什麼都不該做。

    **比較對象是第二個格子**,不是另一種風格。:meth:`set_compare` 接
    ``None`` 代表不比較;挑到與主要風格同一個時會被擋掉(跟自己比永遠沒有
    分歧,而那看起來像個結論)。開著比較時推論成本與記憶體都是兩倍。
    """

    def can_pick_style(self) -> bool: ...
    def styles(self) -> tuple[str, ...]: ...
    def style(self) -> str | None: ...
    def set_style(self, name: str) -> None: ...
    def compare(self) -> str | None: ...
    def set_compare(self, name: str | None) -> None: ...


def is_on(switchboard: Switchboard | None, key: str) -> bool:
    """這個功能現在該不該顯示內容。

    **沒有 switchboard 時一律視為開著。** 重播與示範模式是命令列決定跑哪一條
    路,而它確實在跑 —— 那時候顯示「未開啟」會讓人去找一個不存在的開關。
    """
    if switchboard is None:
        return True
    return switchboard.is_enabled(key)
