"""視覺層:純 CV,無狀態。

這一層只回答「**這一幀看到什麼**」,絕不推論「發生了什麼事」。

範圍**只有自己的手牌與副露**。牌河與對手的手牌/副露改由封包(``groundtruth``)
提供 —— 這是專案中期的一次方向調整,原因與量測成果見 ``docs/decisions.md``。
"""

from mia.vision.roi import MissingRoiError, Roi, RoiSet, roi_names

__all__ = ["MissingRoiError", "Roi", "RoiSet", "roi_names"]
