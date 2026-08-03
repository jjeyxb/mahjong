"""「能開關功能的東西」—— UI 這一層對 :mod:`mia.live` 的全部要求。

宣告成 Protocol 而不是直接吃一個 :class:`~mia.live.runtime.LiveRuntime`:
UI 也要服務錄影重播與示範資料,那些模式裡根本沒有工作執行緒可以開關。

放在單獨一個模組,是因為側邊視窗與 Overlay **都**要用它。留在
``ui/panel/window.py`` 的話,Overlay 就得去 import 側邊視窗才拿得到型別 ——
那兩個是平行的呈現方式,誰都不該相依誰。
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["CanvasPicker", "GameLauncher", "Switchboard", "is_on"]


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


class CanvasPicker(Protocol):
    """能決定遊戲畫布尺寸的東西。

    又是一個獨立的 Protocol 而不是塞進 :class:`Switchboard`:選畫布只對
    「MIA 自己開瀏覽器 + 畫面辨識有跑」的組合有意義。錄影重播的畫布尺寸
    是錄的時候就定死的,``--no-vision`` 則根本沒有人要用這個矩形。
    """

    def can_pick_canvas(self) -> bool: ...
    def canvas(self) -> str | None: ...
    def set_canvas(self, key: str | None) -> None: ...


def is_on(switchboard: Switchboard | None, key: str) -> bool:
    """這個功能現在該不該顯示內容。

    **沒有 switchboard 時一律視為開著。** 重播與示範模式是命令列決定跑哪一條
    路,而它確實在跑 —— 那時候顯示「未開啟」會讓人去找一個不存在的開關。
    """
    if switchboard is None:
        return True
    return switchboard.is_enabled(key)
