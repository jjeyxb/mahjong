"""鳴牌區專用模型(meld model)v3.2 —— 推論端包裝。

從舊專案 mjai_live/meld_model.py 搬過來。改動只有兩處:
    ・`from crop_zones import ...`  -> `from meld import ...`
    ・`from tile_model import base_pai` -> `from tiles import base_pai`
      (MJAI 沒有整張畫面的 tile_model,那個純函式住在 tiles.py)
門檻、裁切、去重、三道載入關卡、imgsz 核對全部**照抄**,理由在下面。

**整條推論路徑上不跑任何別的模型。** 裁四塊固定座標 -> 推論 -> 座標換回全
畫面 -> 去重。前處理在訓練與推論兩端是同一份程式碼(meld.py),而且載入前會
核對權重旁邊那份 crop_meta.json。


=== 權重:meld_model_v32_1280.pt ===
★ 不要改成 meld_model_best.pt。那個檔名下面放的是 v2 時代的權重(用「手牌+
  鳴牌」的大框 + 塗黑訓練的)。檔名一樣、前處理不一樣,載起來**不會報錯**,
  只會讀不到牌。v3.2 的權重一律叫 meld_model_v32*.pt。
★ 也不要換成 meld_model_v32.pt(640 訓練那顆)。1280 那顆把類別讀錯從
  12 筆降到 1 筆,見下面的對照表。


=== 信心度門檻 0.15 是量出來的 ===
valid 的 333 個框跑過四組對照(IoU 0.5 配對,ultralytics 8.4.127):

    conf │  配對成功      │ 類別讀錯 │   漏讀    │ 重複框
         │  640    1280   │ 640 1280 │ 640  1280 │ 640 1280
    0.15 │  330    330    │  12    1 │   3    3  │  74   60
    0.25 │  323    325    │  10    1 │  10    8  │  43   45
    0.40 │  307    319    │  10    1 │  26   14  │  20   25
    0.50 │  299    310    │   7    1 │  34   23  │  15   15

★ 1280 的類別讀錯在四個門檻**都是 1**(同一筆 7s->9s)。640 是 12/10/10/7、
  會隨門檻變 —— 那是「模型拿不定主意」;不隨門檻變表示剩下那一筆跟信心度
  無關,是單一個案。

門檻維持 0.15,三個理由:
  1. 配對成功(330)與漏讀(3)兩欄都是四個門檻裡最好的。
  2. 代價只有多 15 個重複框,而重複框是免費的 —— dedup() 實測把 74 個收到
     0 個、漏讀一個都沒增加。背景誤報四組都是 0。
  3. ★ **赤五只有在 0.15 讀得到**:5mr 在 0.15 是 3/3 全對,0.25 以上漏 2/3。
     赤五正是規則層幫不上的那一類(只有槓例外),漏讀直接等於少算一飜。
     字牌 P/C/F 也一樣,0.15 乾淨、0.25 以上開始漏。

★ 這個選擇跟 **ultralytics 8.4.127** 和 **imgsz=1280** 綁在一起。YOLO26 是
  NMS-free 的,重複框的數量完全由後處理決定,而那一段在版本之間改過(同一份
  權重、同一批資料、同一個 conf:8.4.157 只報 18 個重複、8.4.127 報 74 個)。
  升級之前不要動,升級之後要重跑上面四組再定一次。載入時會比對版本並警告。


=== ★★ 還沒解決的已知缺陷:一局結束時手牌會被讀成鳴牌 ★★ ===
舊專案 2026-10-12 第一次實機測試(690 幀)量到的,**資料集永遠查不出來**
(那 520 張全是局中畫面):

    座位   f<639 平均張數   f>=639(結束時)   讀到的牌
    P0        0.00            2.15            2p 5sr 4p 5s 3s 4p 3s
    P3        1.89            0.92            1p 3m 4m 2m 1p

P0 那 7 張花色混雜、有重複、還含赤五 —— 不是任何合法的鳴牌組合,**那是手牌**。
原因:一局結束時手牌會**攤開翻正**,攤平的牌是躺著的,框很矮,
所以 MAX_BOX_H 那條高度門檻擋不住(它是拿「立著的手牌 138~159px」量的,
而那是**局中**的樣子)。

⚠️ **不要改 MAX_BOX_H 去擋它。** 攤平的手牌跟躺平的鳴牌一樣矮,高度這一維
   在局末沒有資訊。那不是門檻調得不好,是那一維分不開。
   正確的界線是「局中 / 局末」,不是框高 —— 要擋就用局勢模型的
   ron/tsumo/ryukyoku 當訊號,在局末停止鳴牌定案。
   ★ 而且舊專案查過,「結算畫面蓋住桌面 -> meld_area 消失」**不成立**:
     結算期 meld_area 還是被偵測得到。


=== 還是沒有 merge_meld() ===
P0 沒鳴牌的時候,14 張手牌會向右攤到 x 1532 附近,正好攤在這個框裡
(框是 x 657~1833)。所以「框裡只信鳴牌模型」會吃掉 x 657~1532 那一段的手牌。
解法是量過的:**用框的高度分流,不是用範圍** ——
    範圍外                      -> 只信手牌模型
    範圍內、框高 >= MAX_BOX_H   -> 是手牌,留給手牌那一邊
    範圍內、框高 <  MAX_BOX_H   -> 是鳴牌,用這個模型
為什麼還不寫:這條規則一寫下去就跟 hand_model 的接法綁在一起(誰先跑、
P0 那一段誰讓誰),而「一個半對的 merge」會被直接拿去用。等真的要把鳴牌接進
手牌那條路時,連 merge 一起設計。
"""
import os

from meld import (REF_H, REF_W, SEATS, dedup, in_excluded, load_meta,
                  scaled_zone, too_tall)
from tiles import base_pai, infer_kwargs

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Model",
    "meld_model_v32_1280.pt")

DEFAULT_CONF = 0.15                     # 量出來的,見模組說明

# 訓練與所有驗證數字都是在這個版本上跑的。NMS-free 的後處理行為跟版本綁死,
# 所以版本本身就是前處理規格的一部分。
# ★ 這個常數一開始被抄錯成 "8.4.21",害載入時的比對**每次都誤報** ——
#   而每次都亮的警告等於沒有警告。教訓:版本號要用程式讀、不要靠人抄。
TRAINED_ULTRALYTICS = "8.4.127"

# ================================================================
# ★ 訓練時的 imgsz —— 這是「前處理規格」的一部分,不是調優參數
# ================================================================
# ⚠️ 這個常數存在的理由,是 ultralytics 把推論尺度**藏在 .pt 裡**:
#       Model._reset_ckpt_args():  include = {"imgsz", "data", "task", ...}
#       Model.predict():           args = {**self.overrides, **custom, **kwargs}
#    所以 1280 訓練的權重,predict() 不傳 imgsz 時「剛好」會用 1280。
#    那是對的行為,但它是**看不見的**:權重被重新匯出一次、或哪天 ultralytics
#    改了那個 include 集合,尺度就會安靜地退回 640 —— 症狀只是「讀到的牌變少」,
#    跟裁切範圍寫錯的症狀一模一樣。
# 所以這裡做兩件事:
#   1. 載入後把權重裡記的 imgsz 讀出來核對,對不上就**拒絕執行**
#   2. predict() 明確傳 imgsz,不靠繼承
TRAIN_IMGSZ = 1280


def meta_path_for(weight_path):
    """這顆權重配的裁切規格檔。
    Model/meld_model_v32_1280.pt -> Model/meld_model_v32_1280.crop_meta.json

    ★ 為什麼要「一顆權重一份」:這樣拿 v2 時代的 meld_model_best.pt 來載,
      會因為「找不到它的規格檔」而停下來,不會配上 v3.2 的座標硬跑、然後吐出
      一份看起來很合理的垃圾。
    ★ 為什麼不去讀資料集那一份:資料集不會跟著搬機器。規格要跟**權重**一起走。
    """
    stem, _ = os.path.splitext(weight_path)
    return stem + ".crop_meta.json"


def _check_ultralytics():
    """版本對不上就警告(**不擋**)。

    不擋的理由:版本不同不等於「餵模型沒看過的輸入」(那種才該直接拒絕執行),
    而是「重複框的數量會不一樣、0.15 這個門檻要重新量」。擋下來只會逼人去改
    常數,那更糟。
    """
    try:
        from importlib.metadata import version
        got = version("ultralytics")
    except Exception:
        return None
    if got != TRAINED_ULTRALYTICS:
        print(f"⚠️ ultralytics {got},但這顆權重的所有數字(conf "
              f"{DEFAULT_CONF}、DUP_DIST、重複框計數)都是在 "
              f"{TRAINED_ULTRALYTICS} 上量的。")
        print("   YOLO26 是 NMS-free 的,重複框完全由後處理決定,那一段在版本"
              "之間改過\n"
              "   (同一份權重同一批資料:8.4.157 報 18 個重複、8.4.127 報 74"
              " 個)。")
        print(f"   -> 要嘛裝回 {TRAINED_ULTRALYTICS},要嘛重跑 conf "
              f"0.15/0.25/0.40/0.50 四組對照再定一次門檻。")
    return got


def _check_imgsz(model, where=""):
    """核對權重裡記的 imgsz 跟 TRAIN_IMGSZ 是否一致。

    讀不到就只警告(不同版本的 ultralytics 放的位置可能不一樣);讀到了而且
    不一樣就直接停 —— 那表示這顆權重不是用 TRAIN_IMGSZ 訓練的。
    """
    got = None
    try:
        got = (model.overrides or {}).get("imgsz")
    except Exception:
        pass
    if got is None:
        print(f"⚠️ 讀不到權重裡記的 imgsz(ultralytics 版本差異)。"
              f"推論會明確用 {TRAIN_IMGSZ},但無法確認權重就是這個尺度訓練的。")
        return None
    if isinstance(got, (list, tuple)):
        got = max(got)
    if int(got) != TRAIN_IMGSZ:
        raise SystemExit(
            f"❌ imgsz 對不上{(' (' + where + ')') if where else ''}:"
            f"權重記的是 {got},程式要的是 {TRAIN_IMGSZ}。\n"
            f"   推論的尺度必須跟訓練時一樣,不然就是餵它沒看過的東西 ——\n"
            f"   而且不會報錯,只會讀到的牌變少。\n"
            f"   要嘛換回 imgsz={TRAIN_IMGSZ} 訓練的權重,要嘛把 TRAIN_IMGSZ\n"
            f"   改成 {got} 並**重跑 conf 對照**(0.15 綁在特定權重上)。")
    print(f"✅ imgsz 核對通過:{TRAIN_IMGSZ}")
    return int(got)


class MeldModel:
    """鳴牌區模型:四家各裁一塊固定座標、一次 batch 推論、座標換回全畫面。

    載入時三道關卡,順序有意義 —— 規格對不上就根本不該載權重:
      1. 權重存在嗎
      2. 這顆權重的裁切規格檔存在嗎(Model/<權重名>.crop_meta.json)
      3. 規格內容一致嗎(meld.load_meta -> verify_meta,對不上 SystemExit)
    然後才 import ultralytics、比對版本、載權重。
    ★ ultralytics 的 import 刻意放在規格核對**之後**,所以沒裝 torch 的環境
      也測得到前兩關。
    """

    def __init__(self, model_path=None, conf=DEFAULT_CONF, meta_path=None):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到鳴牌區模型:{path}")

        mp = meta_path or meta_path_for(path)
        if not os.path.isfile(mp):
            raise FileNotFoundError(
                f"找不到這顆權重的裁切規格檔:{mp}\n"
                f"   權重 {os.path.basename(path)} 是用某一組裁切範圍訓練的,\n"
                f"   沒有那份紀錄就無法確認推論端裁的是同一塊 —— 裁錯不會報錯,"
                f"\n   只會讀不到牌(舊專案為這件事壞過一次)。\n"
                f"   如果這是 v2 時代的權重(meld_model_best.pt),那它配的是\n"
                f"   「手牌+鳴牌」的大框 + 塗黑,**不能**用這一版的程式跑。")
        meta = load_meta(mp, f"meld_model({os.path.basename(path)})")
        print(f"✅ 裁切規格核對通過:{meta['crop_spec']}")

        from ultralytics import YOLO          # 延後 import:離線測試不需要 torch
        _check_ultralytics()
        print(f"⏳ 載入鳴牌區模型:{path}")
        self.model = YOLO(path)
        self.pred_kw = infer_kwargs(path)
        _check_imgsz(self.model, os.path.basename(path))
        self.names = dict(self.model.names)
        self.conf = conf
        self.meta = meta
        print(f"✅ 載入完成,共 {len(self.names)} 類,信心度門檻 {conf}")

    # -- 前處理:這一段一定要跟建資料集時一樣 ---------------------------
    @staticmethod
    def _check_frame(frame):
        """畫面尺寸不是 1920x1080 就直接吵。

        ⚠️ 這個檢查比它看起來重要。scaled_zone() 會把框**跟著畫面等比例縮小**,
           所以餵一張 1280x720 進來不會有任何「超出邊界」的症狀 —— 它會乖乖裁
           出一塊 784x97 的圖,然後 YOLO 放大到 imgsz,牌在網路眼裡的大小就跟
           訓練時完全不一樣。結果只表現成「讀不到牌」,查起來很痛。
           等比例縮放那個功能是給**建資料集**用的,推論端不需要。
        """
        h, w = frame.shape[:2]
        if (w, h) != (REF_W, REF_H):
            raise ValueError(
                f"鳴牌模型收到 {w}x{h} 的畫面,必須是 {REF_W}x{REF_H}。\n"
                f"   裁切範圍會跟著縮放,所以不會報錯也不會裁壞,只會讓牌在"
                f"網路眼裡的大小\n"
                f"   跟訓練時不一樣、然後讀不到牌。capture.py 負責縮放成 "
                f"{REF_W}x{REF_H},先去看那裡。")

    def _crop(self, frame, seat):
        """裁一塊固定座標。不塗黑、不依賴任何模型。"""
        h, w = frame.shape[:2]
        zx, zy, zw, zh = scaled_zone(seat, w, h)
        crop = frame[zy:zy + zh, zx:zx + zw]
        if crop.shape[0] != zh or crop.shape[1] != zw:
            # numpy 切片超出邊界不會報錯,會安靜地給你一塊比較小的圖。
            raise ValueError(
                f"{seat} 裁出來是 {crop.shape[1]}x{crop.shape[0]},"
                f"應該是 {zw}x{zh}(畫面 {w}x{h})。")
        return crop, (zx, zy, zw, zh)

    def crops_for_debug(self, frame, seats=SEATS):
        """回傳 [(seat, 餵給模型的那張圖, zone)] —— 讀不到牌的時候先看這個。

        「連框都沒框到」最常見的原因不是模型差,是**餵進去的圖跟訓練時不一樣**。
        與其猜,不如把模型實際看到的那張圖存下來比對。v3.2 沒有塗黑,
        所以黑色比例應該接近 0 —— 看到幾十個百分比就是又在塗黑了。
        """
        self._check_frame(frame)
        out = []
        for seat in seats:
            crop, zone = self._crop(frame, seat)
            out.append((seat, crop, zone))
        return out

    def detect(self, frame, conf=None, seats=SEATS, dedup_dets=True):
        """回傳四家鳴牌區的偵測,座標已經換回整張畫面。

        frame 必須是 1920x1080。
        dedup_dets=False 回傳沒去重的原始結果 —— 只有「要量重複框有多少」
        的時候才需要,正常推論一律去重。
        """
        self._check_frame(frame)
        h, w = frame.shape[:2]
        crops, metas = [], []
        for seat in seats:
            crop, zone = self._crop(frame, seat)
            crops.append(crop)
            metas.append((seat, zone))
        if not crops:
            return []

        # ★ imgsz 明確傳,不靠從 .pt 繼承(理由見 TRAIN_IMGSZ 的說明)
        results = self.model.predict(crops, conf=conf or self.conf,
                                     **self.pred_kw,
                                     imgsz=TRAIN_IMGSZ, verbose=False)
        out = []
        for (seat, (zx, zy, _zw, _zh)), res in zip(metas, results):
            if res.boxes is None:
                continue
            for i in range(len(res.boxes)):
                pai = self.names[int(res.boxes.cls[i])]
                x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i])

                # 高度門檻:P0 的裁切圖裡有手牌的右半段(版面決定的,調座標
                # 無解),立著的手牌比躺平的鳴牌高四成。完整的數字與餘裕計算
                # 見 meld.MAX_BOX_H。
                # ⚠️ 這條目前是**備援**不是主力 —— 驗證時 P0 手牌誤判是 0。
                #    留著是因為它只是一行判斷,而手牌被讀成鳴牌會一路錯到
                #    mjai 事件去。
                # ⚠️ 它**擋不住局末攤開的手牌**(見模組說明的已知缺陷)。
                if too_tall(y2 - y1, seat, h):
                    continue

                x1, y1 = int(x1) + zx, int(y1) + zy
                x2, y2 = int(x2) + zx, int(y2) + zy
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 排除區(目前只有 P3 的寶牌指示面板,v3.2 已經在框外,備援)
                if in_excluded(cx, cy, seat, w, h):
                    continue

                out.append({
                    "pai": pai,
                    "base": base_pai(pai),
                    "conf": float(res.boxes.conf[i]),
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "seat": seat,
                    "model": "meld",
                })
        return dedup(out) if dedup_dets else out
