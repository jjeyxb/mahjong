"""量化評測:拿封包當標準答案,量 CV 認得對不對。

純 CV 沒辦法宣稱 100% 準確,但可以讓錯誤變成一個**可量測、可追蹤**的數字,
並指出錯在哪幾種牌上 —— 這是本專題最強的論述點。

兩步:

1. :mod:`~majsoul_copilot.eval.align` 把錄下的畫面與封包推出來的手牌配對。
   兩個錄製行程唯一的共同基準是牆上時鐘。
2. :mod:`~majsoul_copilot.eval.report` 對配好的幀跑一次辨識,產出準確率與
   混淆對照表。

CLI 在 ``tools/evaluate.py``。
"""

from majsoul_copilot.eval.align import AlignedFrame, HandTimeline, align, build_timeline
from majsoul_copilot.eval.report import FrameResult, TileReport, evaluate_frame, summarize

__all__ = [
    "AlignedFrame",
    "FrameResult",
    "HandTimeline",
    "TileReport",
    "align",
    "build_timeline",
    "evaluate_frame",
    "summarize",
]
