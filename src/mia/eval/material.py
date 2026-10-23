"""素材本身夠不夠格拿來量準確率。

與 :mod:`~mia.eval.align` 的分工:那裡問的是「這一幀有沒有標準答案」,
這裡問的是「**整份素材能不能信**」。兩者都是在跑 CV 之前就該擋下來的事,
因為它們壞掉的時候,報告仍然會給出一個看起來完全正常的數字。
"""

from __future__ import annotations

from mia.recorder.session import SessionReader
from mia.utils.logging import logger

__all__ = ["require_one_calibration"]


def require_one_calibration(session: SessionReader, *, anyway: bool = False) -> None:
    """校正中途換過的素材不能拿來量準確率,除非呼叫端明講要看。

    ROI 與牌河角點全部**正規化到 table_rect**,所以換過校正等於前半段與後半段
    用了兩套座標。那時候跑出來的數字量的是校正,不是辨識 —— 而它看起來跟一份
    正常的報告一模一樣。

    實測 ``20260801-112437``:錄製開頭遊戲正在重連,校正鎖在黑色載入畫面上,
    整場的牌河每張牌 20.2%、手牌 26.3%。換上正確的矩形之後,同一幀的手牌
    一張不差。**那兩個數字跟 CV 沒有任何關係。**

    Args:
        session: 要檢查的素材。
        anyway: 使用者明確要求照跑。只警告,不擋。

    Raises:
        SystemExit: 校正中途換過,而且沒有指定 ``anyway``。
    """
    at = session.manifest.table_rect_from
    if at <= 0:
        return
    message = (
        f"這份素材的校正在第 {at} 幀換過(錄製時鎖錯了,複查之後重鎖)。"
        f"第 {at} 幀之前的畫面是用另一個牌桌矩形錄的,而 ROI 與牌河角點全部"
        "相對它正規化 —— 混在一起量出來的數字量的是校正,不是辨識。"
        "請重錄,或加 --anyway 明確表示你知道這件事。"
    )
    if anyway:
        logger.warning(message)
        return
    raise SystemExit(message)
