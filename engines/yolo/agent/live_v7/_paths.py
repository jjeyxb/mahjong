r"""專案路徑(2026-10-31 整理成單一專案資料夾 Project_1031)。

    Project_1031\
        *.py              上一層共用的純邏輯模組與模型載入程式
                          (river_track / river / drawn / hand / event_zones /
                           tiles / capture / 1_capture / *_model.py)
        live_v7\          主程式(錄製、重播、狀態機、Mortal 守門器)
        Model\            所有辨識模型權重 + Mortal 的 mortal.pth
        assets\           場況樣板(templates、templates_wind)、測試用截圖
        Mortal\           Mortal 原始碼(重建 docker 映像用)
        runs\             每次錄製/重播的輸出

★ 所有路徑都從這支檔案的位置往上推,不會指到專案資料夾外面。
  整個 Project_1031 資料夾搬到別的地方也能直接跑。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MJAI_DIR = os.path.dirname(HERE)            # = 專案根目錄(沿用舊名稱)
PROJECT_DIR = MJAI_DIR
UI_DIR = os.path.join(HERE, "ui")
MODEL_DIR = os.path.join(PROJECT_DIR, "Model")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

for p in (MJAI_DIR, UI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# 場況樣板的位置(原本在舊專案 mjai_live,已複製到 assets\)
DEFAULT_OLD_DIR = ASSETS_DIR
