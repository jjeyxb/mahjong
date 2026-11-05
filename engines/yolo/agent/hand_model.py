"""P0 手牌區專用模型(hand model)的推論包裝。

從舊專案 mjai_live/hand_model.py 搬過來。改動只有兩處:
    ・`from tile_model import ...` -> `from tiles import ...`
      (MJAI 這邊沒有整張畫面的 tile_model,那三個純函式住在 tiles.py)
    ・拿掉 merge_hand()
      那個函式是為了「整張畫面模型 + 手牌模型兩份偵測要合成一份」而存在的。
      MJAI 這邊根本沒跑整張畫面模型 —— 牌河走 river_model、手牌走這個 ——
      沒有要合的東西。要是哪天把 tile_model 搬過來了再把它一起搬。
裁切邏輯、固定範圍、塗黑規則、門檻全部**照抄**,一個字都不要改,理由在下面。

=== 為什麼裁切邏輯不能隨便寫 ===
這顆權重是用「裁一個固定範圍、把範圍裡的鳴牌區塗黑」這種圖訓練的。推論時的
前處理如果跟訓練時不一樣,就不是「尺度要跟訓練時一樣」那條鐵則的問題,而是
「連輸入內容本身都不一樣」—— 這比尺度不對還嚴重。

固定範圍(1920x1080 基準):
    (220, 905, 1610, 170)
這塊比 main_hand 的偵測框大很多。資料集本來就是拿這塊固定範圍去掃,不是拿
main_hand 框直接裁 —— 固定範圍才能穩定框住整條手牌,不會因為某一幀 main_hand
偵測框忽大忽小而裁出不一樣大小的圖。

範圍裡的鳴牌區(meld_area,由 region_model 偵測)要塗黑:手牌模型的訓練資料
從來沒看過鳴牌的牌面,混進去不是「多一點雜訊」,是「餵它沒看過的內容」。
P0 的固定範圍在畫面下緣,不會碰到 P3 那個寶牌指示 UI 的問題,所以不需要
像舊專案的 batch_crop_meld_hand.py 那樣另外登記排除區。

=== ★ 已知缺陷(舊專案量到的,第五步要驗的就是它) ===
    9p -> 6p  21 筆
    6p -> 5p  16 筆
    4p -> 2p   6 筆
**全部出在手牌**(不是牌河)。記錄的原因是 1610 寬的條帶餵進 imgsz=640 會被
縮到 **0.40**,筒子的圓點在那個尺度下糊成一團。記錄的解法是切成兩半
845x146(重疊 80px),縮放變 **0.76**。

這一步刻意**先照原樣用、先量**(使用者 2026-10-24 決定)。先知道這顆權重在
這台機器、這個版面上的實際錯誤率,再決定要不要動前處理 —— 動了前處理就等於
換了輸入內容,等於要重訓,代價很大,不能憑舊紀錄就動手。

=== 信心度門檻 ===
DEFAULT_CONF 沿用 0.25。不要用訓練時 val 集的信心度分布去猜門檻 ——
舊專案的教訓:牌河舊模型在 val 集看起來沒事,實機卻只有 0.1~0.4。
5_hand.py 的報告第 6 節會印實機的逐牌種信心度分布,看那個再調。
"""
import os

from hand import (HAND_CONF, MAX_ZONE_P0, REF_H, REF_W,  # noqa: F401
                  in_hand_zone, scaled_zone)
from region_model import RegionModel
from tiles import base_pai, infer_kwargs, merge_overlaps

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Model", "hand_model_best.pt")

# 固定範圍、縮放、門檻都住在 hand.py(零相依)。這個檔案模組層級就要
# ultralytics,座標放在這裡的話連 --selftest 都要先等 torch 載入十幾秒。
DEFAULT_CONF = HAND_CONF


class HandModel:
    """裁固定範圍、塗黑範圍內的鳴牌區、對裁切圖推論。

    可以傳入外面已經建立好的 RegionModel 共用。5_hand.py 每一幀本來就會跑
    一次局勢模型(要 meld_area),把那一份結果用 region_dets= 傳進來可以省掉
    重複推論 —— 舊專案的 run_step13.py 踩過這個坑,一幀跑兩次局勢模型,
    fps 直接砍半。
    """

    def __init__(self, model_path=None, conf=DEFAULT_CONF, region=None,
                 region_conf=0.25):
        from ultralytics import YOLO
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到手牌模型:{path}")
        print(f"⏳ 載入手牌模型:{path}")
        self.model = YOLO(path)
        self.pred_kw = infer_kwargs(path)
        self.names = dict(self.model.names)
        self.conf = conf
        self.region = region or RegionModel(default_conf=region_conf)
        self.last_masked = None       # 給 --save-crop 看「模型實際吃到什麼」
        print(f"✅ 載入完成,共 {len(self.names)} 類,信心度門檻 {conf}")

    def detect(self, frame, conf=None, region_dets=None, right=None):
        """回傳 P0 手牌的偵測,座標已經換回整張畫面。

        frame 必須是 1920x1080(capture.py 已經縮放好了)。

        region_dets 傳這一幀已經推論好的局勢偵測結果(裡面要含 meld_area;
        不含也不會壞,只是塗黑會漏掉)。不傳的話自己再推論一次。

        right 給了就當成這一幀裁切的右邊界(整張畫面座標)。
        ★ 這會讓裁切寬度逐幀變動,也就讓餵進 imgsz 的縮放逐幀變動 ——
          代價跟理由寫在 hand.py 的「動態右邊界」那一段,動它之前先讀。
        """
        h, w = frame.shape[:2]
        zx, zy, zw, zh = scaled_zone(w, h, right)
        crop = frame[zy:zy + zh, zx:zx + zw]
        if crop.size == 0:
            return []

        melds = (region_dets if region_dets is not None
                 else self.region.detect(frame, only=("meld_area",)))

        # 範圍內的鳴牌區塗黑(見模組說明)。用**中心點**判斷落在範圍內,
        # 跟建資料集時的邏輯一致。
        masked = crop.copy()
        for d in melds:
            if d["cls"] != "meld_area":
                continue
            if not in_hand_zone(d["center"], w, h):
                continue
            bx1, by1, bx2, by2 = d["box"]
            lx1 = max(0, int(bx1) - zx)
            ly1 = max(0, int(by1) - zy)
            lx2 = min(zw, int(bx2) - zx)
            ly2 = min(zh, int(by2) - zy)
            if lx2 > lx1 and ly2 > ly1:
                masked[ly1:ly2, lx1:lx2] = 0
        self.last_masked = masked

        result = self.model.predict(masked, conf=conf or self.conf,
                                    **self.pred_kw,
                                    verbose=False)[0]
        out = []
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                pai = self.names[int(result.boxes.cls[i])]
                x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
                x1, y1 = int(x1) + zx, int(y1) + zy
                x2, y2 = int(x2) + zx, int(y2) + zy
                out.append({
                    "pai": pai,
                    "base": base_pai(pai),
                    "conf": float(result.boxes.conf[i]),
                    "box": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "seat": "P0",
                    "model": "hand",
                })
        return merge_overlaps(out)
