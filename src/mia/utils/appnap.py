"""告訴 macOS 不要節流本程式(App Nap)。

玩的時候遊戲在前景,MIA 就是背景 App。macOS 會在幾十秒內把背景 App 的計時器放慢、
CPU 優先權降低,而 MIA 一路靠計時器與 CPU 運算走完,每一段慢幾倍,加起來就是十秒。

實測(2026-11-01,背景 45 秒之後):

* 100 ms 的計時器變成 300 ms,QTimer 與 ``Event.wait`` 都一樣
* 認一副手牌從 0.80 秒變成 3.29 秒

宣告活動之後,整段都維持 100 ms 與 0.80 秒。
"""

from __future__ import annotations

import sys

from mia.utils.logging import logger

__all__ = ["hold_off_app_nap"]

#: 活動的代號。系統看的是它還活著,被回收就等於活動結束,所以一直留著。
_activity: object | None = None


def hold_off_app_nap() -> None:
    """從現在到程式結束都不要節流。重複呼叫沒有影響。

    用 AllowingIdleSystemSleep:只求不被節流,不該因為開著 MIA 就讓電腦不能睡。
    """
    global _activity
    if sys.platform != "darwin" or _activity is not None:
        return

    import Foundation

    _activity = Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        Foundation.NSActivityUserInitiatedAllowingIdleSystemSleep
        | Foundation.NSActivityLatencyCritical,
        "即時辨識遊戲畫面",
    )
    logger.debug("已宣告活動,系統不會節流本程式")
