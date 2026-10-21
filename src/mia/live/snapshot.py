"""抓一張遊戲畫面 —— 給 UI 用的同步版本。

為什麼不重用 :class:`~mia.live.vision.VisionWorker`
---------------------------------------------------
那是一條**執行緒**,它的產出走郵箱,而且只在功能打開時才存在。手動框選是
設定動作:使用者按下按鈕,當場要看到一張圖。為了一張圖去啟動一條工作執行緒、
等它投一筆進郵箱、再從郵箱撈出來,中間每一步都可能失敗,而失敗的理由還要
再傳回來。

同步抓一張就好。成本實測 10~30 ms,在 UI 執行緒上直接做不會讓畫面卡住。
"""

from __future__ import annotations

from mia.capture.base import HOST_EXCLUDE, HOST_REQUIRE, CaptureBackend, Frame
from mia.config.models import AppConfig
from mia.platforms import WEB

__all__ = ["grab_game_frame"]


def grab_game_frame(
    config: AppConfig, platform: str = WEB, backend: CaptureBackend | None = None
) -> Frame:
    """找到遊戲視窗並抓一張。

    Args:
        config: 視窗比對條件從這裡來,與 :class:`~mia.live.vision.VisionWorker`
            讀的是同一份 —— 兩邊找的必須是同一個視窗,否則使用者框的是 A、
            辨識跑的是 B。
        platform: 玩的是哪一版。決定要找瀏覽器還是原生客戶端。
        backend: 擷取後端。省略則依平台自動建立,**用完就關** ——
            這是一次性的動作,留著一個後端沒有人會記得關掉。

    Raises:
        WindowNotFoundError: 找不到遊戲視窗。
        CaptureError: 後端不可用(macOS 沒給螢幕錄製權限是最常見的)或擷取失敗。
    """
    own = backend is None
    if backend is None:
        from mia.capture.factory import create_backend

        backend = create_backend(config.capture)
    try:
        match = config.capture.window
        window = backend.find_window(
            match.title_patterns,
            match.owner_patterns,
            match.host_patterns,
            host=HOST_REQUIRE if platform == WEB else HOST_EXCLUDE,
            min_width=match.min_width,
            min_height=match.min_height,
        )
        return backend.capture(window)
    finally:
        if own:
            backend.close()
