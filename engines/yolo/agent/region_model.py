"""局勢判斷模型(區域模型)的推論包裝。

類別名稱一律從 .pt 檔本身讀,不吃外部的 yaml——models/ 底下那份
region_model_classes.yaml 已經跟實際模型對不上了(它寫 10 類、少了
ryukyoku,連 tsumo 的索引都不一樣)。模型重訓之後 yaml 沒跟著更新是
很容易發生的事,而索引對錯一格就會把 pon 讀成 kan,所以直接信模型。
"""
import os

from ultralytics import YOLO

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Model", "region_model_best.pt")

# 會改變「當前行動者」或代表一局結束的宣告類別。摸打流程用的
# arm/main_hand/drawn_tile/meld_area 不在這裡,那是後面的步驟要用的。
EVENT_CLASSES = ("chi", "pon", "kan", "reach", "ron", "tsumo")
RYUKYOKU_CLASS = "ryukyoku"

# 偵測框碰到畫面邊緣就標記起來。真正的宣告橫幅一定完整落在畫面內
# (實測所有真樣本離邊界最近的也還有 28px),被邊界切掉的框幾乎都不是宣告。
#
# 實例:和了結算畫面右緣有一片背景裝飾的櫻花花瓣,被模型認成 chi
#(conf 0.41,框貼著畫面右緣)。注意這不是「把自摸認成吃」那種兩種橫幅
# 之間的混淆——它是對「純背景美術」誤觸發,代表模型可能對任意裝飾元素
# 產生宣告類別的假偵測,而結算畫面整片都是這種動畫裝飾。
#
# 這種假偵測如果落進某一家的區域,會變成一個不存在的鳴牌事件被送進 mjai
# 事件流,而且不會有任何錯誤訊息。邊緣檢查擋得掉貼邊的那些;真正治本的
# 是「結算/動畫畫面期間根本不該偵測事件」,那要等狀態機那一層才做得到。
EDGE_MARGIN = 2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def dedupe(detections, iou_threshold=0.6):
    """同一個物件被輸出成兩個幾乎重疊的框時,只留信心度最高的那個。

    實測 log 裡確實會出現同一幀、同類別、框座標幾乎一模一樣、但信心度
    差很多的兩筆(例如 pon 0.817 跟 0.432)。不去重的話,同一次鳴牌會被
    當成兩個事件,而且那個低信心度的複本會讓「這一類的信心度分佈」看起來
    比實際差,調門檻時會被誤導。
    """
    kept = []
    for det in sorted(detections, key=lambda d: -d["conf"]):
        if any(d["cls"] == det["cls"] and _iou(d["box"], det["box"]) >= iou_threshold
               for d in kept):
            continue
        kept.append(det)
    return kept


class RegionModel:
    """跑一次推論,回傳結構化的偵測結果。

    conf 門檻刻意可以逐類別設定:實測 arm 的信心度天生比 main_hand 這種
    大區塊低很多,用同一個門檻不是太鬆就是太緊。目前這一步只用得到宣告
    類別,其他類別留著讓之後的步驟共用同一個包裝。
    """

    def __init__(self, model_path=None, default_conf=0.25, conf_overrides=None,
                 imgsz=None):
        path = model_path or DEFAULT_MODEL_PATH
        self.imgsz = imgsz      # None = ultralytics 預設(640);要跟訓練時一致
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到模型檔:{path}")
        print(f"⏳ 載入局勢判斷模型:{path}")
        self.model = YOLO(path)
        from tiles import infer_kwargs
        self.pred_kw = infer_kwargs(path)
        self.names = dict(self.model.names)
        self.default_conf = default_conf
        self.conf_overrides = conf_overrides or {}
        print(f"✅ 載入完成,共 {len(self.names)} 類:"
              f"{', '.join(self.names[i] for i in sorted(self.names))}")

    def _threshold_for(self, class_name):
        return self.conf_overrides.get(class_name, self.default_conf)

    def detect(self, frame, only=None):
        """回傳 [{cls, conf, box=(x1,y1,x2,y2), center=(cx,cy)}, ...]。

        only 給一組類別名稱時只回傳那幾類。推論本身用所有類別裡最低的
        門檻去跑,再逐類別套用各自的門檻——一次推論就好,不必為了不同
        門檻跑很多次。
        """
        wanted = set(only) if only else None
        floor = min([self.default_conf] + list(self.conf_overrides.values()))
        kw = {"imgsz": self.imgsz} if self.imgsz else {}
        kw.update(self.pred_kw)
        result = self.model.predict(frame, conf=floor, verbose=False, **kw)[0]

        fh, fw = frame.shape[:2]
        out = []
        for box in result.boxes:
            name = self.names[int(box.cls[0])]
            if wanted is not None and name not in wanted:
                continue
            conf = float(box.conf[0])
            if conf < self._threshold_for(name):
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append({
                "cls": name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "clipped": (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN
                            or x2 >= fw - EDGE_MARGIN or y2 >= fh - EDGE_MARGIN),
            })
        return dedupe(out)
