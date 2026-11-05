"""牌河專用的牌面模型(river model)。

資料集:Mahjong_AI/dataset_new_train(260 張全景截圖,每張依 RIVER_ZONES 裁出
四家牌河,960 train + 80 valid)。訓練結果(yolov8n,--rect,72 epochs 早停):

    best epoch 42   precision 0.973   recall 0.994   mAP50 0.985   mAP50-95 0.891

⚠️ 這個數字偏樂觀:Roboflow 匯出時同一張原始畫面有 2~3 個增強版本,valid 裡
   的畫面很可能在 train 裡有「兄弟」。真正的泛化要看沒見過的新截圖 ——
   3_river.py 就是做這件事的,而且它量的是**實機連續畫面**,比單張更嚴格
  (單張看不出閃爍,而閃爍是這個任務最直接的錯誤訊號)。

=== ★ 這個模型要「裁切後」推論,跟整張畫面的模型相反 ===
tile_model(手牌、鳴牌區、寶牌、P0 摸到的那張)的規則是「一定要對整張畫面
推論,不可以先裁切」,因為它是用整張畫面訓練的,裁切會讓尺度差將近 6 倍。
**這個模型剛好相反**:它是用裁切圖訓練的,所以推論也必須裁切。

兩條規則不衝突,是同一個原則:**推論時的尺度要跟訓練時一樣**。

=== 輸出 ===
detect() 回傳的座標已經換回整張 1920x1080 畫面:
    {pai, base, conf, box, center, seat, model, clipped, crop_box}
    seat       哪一家的牌河
    model      "river"(除錯時分得出來這張是哪個模型讀的)
    clipped    框碰到裁切邊界 —— ★ 這個旗標是 2026-10-23 加的,見下
    crop_box   裁切圖內的座標,存疊圖時用

=== 為什麼要 clipped ===
RIVER_ZONES 畫小了的時候,邊界上的牌會被裁掉一半,模型於是讀錯或讀不到。
那個症狀看起來只像「這張牌讀不到」,完全不像座標問題 —— 會往模型品質的
方向查,查很久。有這個旗標就一眼看得出來。

跟第二步宣告偵測的「貼邊」關卡是同一個道理,但方向相反:宣告的貼邊是
**正常的**(橫幅從側邊滑進來,剛出現時本來就被切到),牌河的貼邊是
**不正常的**(牌是靜止的,會貼邊就是框不夠大)。
"""
import os

from river import RIVER_ZONES, SEATS, assign_seats
from tiles import base_pai, infer_kwargs, merge_overlaps

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Model",
    "river_model_best.pt")

# 這個模型的信心度分布跟整張畫面模型完全不同(舊的 0.1~0.4,這個通常 0.8
# 以上 —— 實測一張真截圖四家 44 張,最低 0.74、平均 0.93),所以門檻不沿用
# tile_model 的 0.1。0.25 是 ultralytics 的預設值。
#
# ⚠️ 要往上調之前先看報告第 3 節的「逐牌種中位信心度」跟「答案穩定但
#    模型沒把握的格子」—— 門檻一旦高過某一種牌的中位數,那種牌就會整個
#    讀不到,而且連「漏讀」都看不到(框根本不會被產生出來)。
# 2026-10-23 改成 0.10(原本 0.25)。實測的理由:
#   ・正確的答案有 0.15~0.47 的 —— P2 的 6p @0.15、兩格 3s @0.46/0.47
#     (後兩格人工確認過是對的)。0.25 會把它們整段丟掉。
#   ・答案是**幾百幀的多數決**,不是單幀判定。所以單幀門檻低不會變成假
#     事件 —— 保護來自投票、跨家去重、以及「答案穩定但信心度低」的旗標,
#     不是來自門檻。
#   ・門檻設高的代價很具體:那一格整個消失,連「漏讀」都看不到。
DEFAULT_CONF = 0.10

# 框的邊緣離裁切邊界幾 px 以內算貼邊。2px 是為了容忍取整誤差。
CLIP_TOL = 2

# 跨家去重的距離。見 dedupe_across_seats。
CROSS_SEAT_DIST = 20


class RiverModel:
    """牌河模型。每幀對四塊牌河各推論一次(一次 batch)。"""

    def __init__(self, model_path=None, conf=DEFAULT_CONF, zones=None):
        from ultralytics import YOLO      # 延後 import:測試不需要 torch
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到牌河模型:{path}")
        print(f"⏳ 載入牌河模型:{path}")
        self.model = YOLO(path)
        self.pred_kw = infer_kwargs(path)
        self.names = dict(self.model.names)
        self.conf = conf
        self.zones = dict(zones or RIVER_ZONES)
        self.last_moved, self.last_dropped = [], []
        print(f"✅ 載入完成,共 {len(self.names)} 類,信心度門檻 {conf}")

    def crops(self, frame, zones=None):
        """回傳 [(seat, x, y, w, h, 裁切圖), ...]。存疊圖也用這個,所以
        裁切這件事只有一份實作 —— 兩份遲早會不一致。

        zones 給了就用它(牌河會長,框要跟著動;見 3_river.py 的自動擴張),
        而且會夾在畫面範圍內 —— 擴出畫面外的話 numpy 切片會安靜地給出
        比較小的圖,座標就全錯了。

        ★ zones 只給一部分的家也可以,那就只裁那幾塊 —— 第五步只要 P0,
          裁 1 塊而不是 4 塊,推論時間直接省下來。
          (原本寫死跑 SEATS,只傳 {"P0": ...} 會 KeyError: 'P1',
           2026-10-24 實機踩到。下游的 river.seat_of_center 本來就有
           `s in zones` 的防護,沒給的那幾家會被當成「不在任何框裡」而
           忽略 —— 正是要的行為,不需要另外處理。)
        """
        zones = zones or self.zones
        out = []
        for seat, (x, y, w, h) in self._clamped(frame, zones).items():
            for wx, wy, ww, wh in self.windows(seat, (x, y, w, h)):
                out.append((seat, wx, wy, ww, wh,
                            frame[wy:wy + wh, wx:wx + ww]))
        return out

    @staticmethod
    def _clamped(frame, zones):
        h0, w0 = frame.shape[:2]
        out = {}
        for seat in SEATS:
            if seat not in zones:
                continue
            x, y, w, h = zones[seat]
            x = max(0, min(x, w0 - 1))
            y = max(0, min(y, h0 - 1))
            w = max(1, min(w, w0 - x))
            h = max(1, min(h, h0 - y))
            out[seat] = (x, y, w, h)
        return out

    @staticmethod
    def _inner_clip(d, union, tol=4):
        if union is None:
            return False
        ux, uy, uw, uh = union
        wx, wy, ww, wh = d["zone"]
        if (wx, wy, ww, wh) == (ux, uy, uw, uh):
            return False                       # 沒分窗
        x1, y1, x2, y2 = d["box"]
        return ((x1 <= wx + tol and wx > ux + tol)
                or (x2 >= wx + ww - tol and wx + ww < ux + uw - tol)
                or (y1 <= wy + tol and wy > uy + tol)
                or (y2 >= wy + wh - tol and wy + wh < uy + uh - tol))

    def _keep_ext_on_last_row(self, dets, used):
        """擴張出去的那一塊:只收「跟這一家最後一排對齊」的牌。

        ★ 2026-11-05 實測:P2 的框往左擴 200px,左下角會蓋到 P3 牌河最上面那排
          -> 25 幀讀出 70 筆 0.2~0.5 的假牌算在 P2 頭上。溢出的牌一定是接在
          最後一排後面(同一排),別人的東西不會剛好對齊那一排。
          (第一版用「被框邊切到就丟」,但 P0 第 3 排本來就貼著框的下緣,
           真的溢出牌也會被當成切到 —— 不能用。)"""
        from river import RIVER_ORDER, rows_of
        out = []
        for seat in SEATS:
            mine = [d for d in dets if d["seat"] == seat]
            base, union = self.zones.get(seat), used.get(seat)
            if not mine or base is None or union is None or \
                    tuple(union) == tuple(base):
                out += mine
                continue
            bx, by, bw, bh = base

            def inside(d):
                cx, cy = d["center"]
                return bx <= cx <= bx + bw and by <= cy <= by + bh
            core = [d for d in mine if inside(d) and d["conf"] >= 0.5]
            ext = [d for d in mine if not inside(d)]
            out += [d for d in mine if inside(d)]
            if not core or not ext:
                continue
            horiz = RIVER_ORDER[seat]["axis"] == "y"      # 排是橫的
            k = 1 if horiz else 0                         # 分排用的座標
            rows = rows_of(seat, core)
            last = [d for d, r in zip(core, rows) if r == max(rows)]
            band = sorted(d["center"][k] for d in last)[len(last) // 2]
            size = sorted((d["box"][3] - d["box"][1]) if horiz
                          else (d["box"][2] - d["box"][0]) for d in last)
            size = size[len(size) // 2]
            out += [d for d in ext if abs(d["center"][k] - band) <= 0.6 * size]
        return out

    def windows(self, seat, zone):
        """實際送進模型的裁切窗。框沒擴張 -> 就是它自己。

        ★ 2026-11-05 牌河超過 18 張(第 3 排往外長)時框會擴張(record_obs 的
          自動擴張,P0 往右最多 +200px)。整塊擴大的框裁下來縮到模型輸入,
          牌會比訓練時小約 1/3 -> 實測 82 幀 1222 張:跟原尺寸比 19 張讀成
          別的牌(7p<->6p 最多)、漏 3 張。把框「整個往外平移、大小不變」
          則 347 張 0 差異(使用者的提議)。
          但只平移會讓第 1、2 排最前面幾張掉出框 -> 追蹤器以為牌被拿走。
          所以擴張後改成兩個原尺寸的窗:一個在原位、一個貼齊擴張後的外緣,
          合起來蓋住整個擴張框;兩窗重疊處的同一張牌在 detect 裡合併。"""
        base = self.zones.get(seat)
        x, y, w, h = zone
        if base is None:
            return [zone]
        bx, by, bw, bh = base
        if w <= bw + 4 and h <= bh + 4:
            return [zone]
        # 窗 A = 原本的框(在擴張框裡的原位);窗 B = 同尺寸、貼齊往外長的那個角
        # (P3 往下又往左 -> 左下角;P0 往右 -> 右邊;P2 往左;P1 往上)
        a = (max(x, min(bx, x + w - bw)), max(y, min(by, y + h - bh)), bw, bh)
        bxx = x if bx > x + 2 else x + w - bw
        byy = y if by > y + 2 else y + h - bh
        return [a, (bxx, byy, bw, bh)]

    def detect(self, frame, conf=None, zones=None):
        """回傳牌河的偵測,座標已換回整張畫面。

        zones 只給一部分的家就只看那幾家(見 crops)。

        frame 必須是 1920x1080(capture.py 已經縮放好了)—— RIVER_ZONES 是
        那個尺寸的座標。
        """
        parts = self.crops(frame, zones)
        results = self.model.predict([p[5] for p in parts], **self.pred_kw,
                                     conf=conf or self.conf, verbose=False)
        out = []
        for (seat, x, y, w, h, _img), r in zip(parts, results):
            out.extend(boxes_to_dets(r.boxes, self.names, seat, x, y, w, h,
                                     zone=(x, y, w, h)))
        # ★★ 座位歸屬:判準是「中心點落在誰的框裡」,不是「哪一塊裁圖讀到
        #    它」。詳細理由見 river.assign_seats。改過家的 / 丟掉的留在
        #    last_moved / last_dropped,呼叫的人要印出來 —— 目前的架構下這
        #    兩個應該永遠是空的,不是空的就代表真的有東西壞了。
        # 歸屬用「整個(擴張後的)框」,不是個別的窗 —— 一家可能有兩個窗
        used = self._clamped(frame, zones or self.zones)
        # 兩個窗的「內側」邊(在擴張框裡面的那條)切到的半張牌:另一個窗一定
        # 看得到完整的那張 -> 丟掉半張的(實測會讀出 0.15~0.4 的假牌)
        out = [d for d in out if not self._inner_clip(d, used.get(d["seat"]))]
        out = self._keep_ext_on_last_row(out, used)
        out, self.last_moved, self.last_dropped = assign_seats(out, used)
        merged = []
        for seat in SEATS:
            mine = [d for d in out if d["seat"] == seat]
            # 兩個窗的重疊處同一張牌會讀兩次:先留「沒被窗邊切到」的,再比信心度
            mine.sort(key=lambda d: (bool(d.get("clipped")), -d["conf"]))
            merged += merge_overlaps(mine, keep_order=True)
        return dedupe_across_seats(merged)


def dedupe_across_seats(dets, dist=CROSS_SEAT_DIST):
    """★ 落在兩家框的重疊處、被兩家各讀一次的那張牌,只留信心度高的。

    為什麼需要:四家的框**本來就有重疊**(2026-10-23 算出來的,P0 跟 P1
    的角落重疊 32x9px、P0 跟 P3 重疊 20x19px),而且框一擴大重疊只會更多。
    重疊處的牌會被兩塊裁圖各偵測一次,變成兩筆不同 seat 的偵測:
      ・張數憑空多一張
      ・「同一種牌四家加起來超過 4 張」的規則檢查會亂噴
      ・那張牌會被指派給錯的玩家
    而且**從辨識結果上完全看不出來** —— 兩筆的牌名還是對的。

    為什麼這樣去重是安全的:一張牌只屬於一家的牌河,而不同家的牌河之間
    隔著桌面,距離遠得多。兩筆相距 20px 以內卻掛在不同家,只可能是同一
    張牌被讀了兩次。

    (合併本身還是逐家先做 —— 同一家裡的多候選收斂跟跨家去重是兩件事。)

    ⚠️ 2026-10-24 之後這個函式應該永遠不會真的刪到東西:assign_seats 先按
       中心點歸屬過了,同一個位置的兩筆偵測會拿到**同一個** seat,於是在
       上面逐家的 merge_overlaps 就收斂掉了,到這裡不可能還有「位置一樣但
       seat 不一樣」的兩筆。留著是後盾 —— 歸屬那一步哪天改壞了,這裡還能
       擋住「張數憑空多一張」。
    """
    kept = []
    for d in sorted(dets, key=lambda z: -z["conf"]):
        cx, cy = d["center"]
        if any(k["seat"] != d["seat"]
               and abs(k["center"][0] - cx) < dist
               and abs(k["center"][1] - cy) < dist for k in kept):
            continue
        kept.append(d)
    return kept


def boxes_to_dets(boxes, names, seat, ox, oy, cw=None, ch=None,
                  tol=CLIP_TOL, zone=None):
    """ultralytics 的 boxes(裁切圖座標)-> 整張畫面座標的偵測清單。

    拆成獨立函式是為了測試:位移算錯的話,每一張牌都會落到錯的位置,而且
    in_region 會安靜地把它們全部篩掉 —— 看起來只是「牌河讀不到牌」。
    """
    out = []
    if boxes is None:
        return out
    for i in range(len(boxes)):
        pai = names[int(boxes.cls[i])]
        x1, y1, x2, y2 = (int(float(v)) for v in boxes.xyxy[i])
        clipped = False
        if cw is not None and ch is not None:
            clipped = (x1 <= tol or y1 <= tol
                       or x2 >= cw - tol or y2 >= ch - tol)
        out.append({
            "pai": pai,
            "base": base_pai(pai),
            "conf": float(boxes.conf[i]),
            "box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            "center": ((x1 + x2) // 2 + ox, (y1 + y2) // 2 + oy),
            "crop_box": (x1, y1, x2, y2),
            "seat": seat,
            "model": "river",
            "clipped": clipped,
            # ★ 這一筆是用哪個框裁出來的。框會動(自動擴張 / --grow),不記
            #   下來的話離線分析拿固定的 RIVER_ZONES 去比,貼邊跟餘裕全部
            #   會算錯 —— 而且錯得完全看不出來。
            "zone": zone or (ox, oy, cw or 0, ch or 0),
        })
    return out
