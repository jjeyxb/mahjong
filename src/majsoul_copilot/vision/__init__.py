"""視覺層:純 CV,無狀態。

這一層只回答「**這一幀看到什麼**」,絕不推論「發生了什麼事」。跨幀的差分、
事件產生、合法性校驗都是 tracker 的工作。這條界線是整個專案最重要的設計
約束 —— 混在一起是這類專案最常見的失敗原因:一旦某幀辨識錯誤又被寫進狀態,
錯誤就會一路累積下去,而且事後無從分辨是「看錯了」還是「推論錯了」。
"""

from majsoul_copilot.vision.grid import COLS, ROWS, Grid
from majsoul_copilot.vision.roi import MissingRoiError, Roi, RoiSet, roi_names

__all__ = ["COLS", "ROWS", "Grid", "MissingRoiError", "Roi", "RoiSet", "roi_names"]
