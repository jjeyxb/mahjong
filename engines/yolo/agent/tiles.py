"""牌面偵測的純工具。零相依 —— 不 import ultralytics、不碰模型檔。

為什麼單獨開一個檔:這三個函式原本住在 tile_model.py 裡,而那個檔案模組層級
就 `from ultralytics import YOLO`,管的又是手牌/鳴牌區那個完全不同的模型。
river_model.py 只需要這三個純函式,卻會因此拖著一個用不到的模型相依。

(2026-10-22 的教訓:第二步把宣告判斷的邏輯全部寫在 2_events.py 裡,而模組名
 不能以數字開頭,結果任何別的程式要用那段邏輯都得走 importlib。可重用的東西
 不要卡在不該依賴的地方。)
"""

# 中心點距離小於這個值就視為「同一張牌的重複偵測」。
MERGE_DIST = 20

# 赤寶牌跟一般的五在規則上是同一種牌,只是額外算一飜。「打出去的牌跟摸進來
# 的牌是不是同一張」這種比對要用基礎牌型,不然模型把 5s 認成 5sr(或反過來)
# 就會讓比對失準。
RED_TO_BASE = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}


def base_pai(pai):
    """赤寶牌轉成對應的一般牌;其他原樣回傳。"""
    return RED_TO_BASE.get(pai, pai)


def is_red(pai):
    return pai in RED_TO_BASE


def merge_overlaps(dets, dist=MERGE_DIST, keep_order=False):
    """把「同一張牌的多個候選」收斂成一個,取信心度最高的。

    ★ 刻意**不分類別**比較。低門檻下同一張牌常常同時冒出好幾個不同類別的
      猜測疊在一起(7p@0.12 / 1p@0.07 / S@0.05),那本來就是同一張牌,只是
      模型拿不定主意。照類別分開去重的話這幾個猜測會全部留下來,變成憑空
      多出好幾張牌。
    keep_order=True:照呼叫者給的順序挑(river_model 用:先挑沒被窗邊切到的)。
    """
    kept = []
    order = dets if keep_order else sorted(dets, key=lambda z: -z["conf"])
    for d in order:
        cx, cy = d["center"]
        if any(abs(k["center"][0] - cx) < dist
               and abs(k["center"][1] - cy) < dist for k in kept):
            continue
        kept.append(d)
    return kept


def in_region(dets, region, padding=0):
    """篩出中心點落在 region=(x, y, w, h) 裡的偵測。"""
    x, y, w, h = region
    x1, y1 = x - padding, y - padding
    x2, y2 = x + w + padding, y + h + padding
    return [d for d in dets
            if x1 <= d["center"][0] <= x2 and y1 <= d["center"][1] <= y2]


# ---------------------------------------------------------------------------
# 推論參數 sidecar:<權重名>.infer.json
# ---------------------------------------------------------------------------
def infer_kwargs(weight_path):
    """讀權重旁邊的 <權重名>.infer.json,回傳要多傳給 predict() 的參數。

    ★ 2026-11-03 為什麼要有這個:YOLO26 有兩個輸出頭 ——
        one2one(end2end,不做 NMS)和 one2many(要 NMS)。
      ultralytics 8.4.127(使用者電腦上的版本)預設用 one2one;較新的版本
      (8.4.197)預設用 one2many。2026-11-03 重訓的牌河 / 手牌模型 one2one 頭
      很弱:同一張 P3 的 7s,one2one 0.12、one2many 0.94;手牌一幀 13 張只
      讀到 11 張。所以這幾顆權重要明確指定 end2end=false。
      不寫死在程式裡,是因為舊的鳴牌 / 局勢模型(也是 YOLO26)一直用 one2one
      跑、也驗證過 —— 換頭等於換了推論方式,要重新量過才能改。
    沒有這個檔 = 什麼都不改(照 ultralytics 預設)。

    格式:{"end2end": false}
    """
    import json
    import os
    p = os.path.splitext(weight_path)[0] + ".infer.json"
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    kw = {}
    if "end2end" in d and d["end2end"] is not None:
        kw["end2end"] = bool(d["end2end"])
    if kw:
        print(f"   推論參數({os.path.basename(p)}):{kw}")
    return kw
