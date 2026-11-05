"""寶牌指示區:裁固定的 5 格面板,用手牌模型讀牌面(取代整張畫面跑 tile_model)。

=== 為什麼換 ===
舊做法是整張 1920x1080 丟 tile_model 再篩左上角。寶牌在整張畫面裡只有 52x77,
縮到 640 只剩 17 像素寬,信心常常只有 0.1~0.3(2026-10-30 實戰:槓寶牌 2m
讀成 1m,信心 0.14~0.16,Mortal 因此當掉)。

寶牌面板是正面、不變形的 UI,牌面跟手牌長得幾乎一樣。把面板裁下來放大 2 倍、
貼到跟手牌模型訓練時一樣大的 1610x170 畫布上,手牌模型就看得很清楚。
2026-11-01 在 20261030_183152 的 7 個面板上:全部讀對,大多 0.8~0.97,
最低是 4m 0.5~0.67(萬子的紅字比較吃虧)。

=== 面板幾何(1920x1080,2026-11-01 從錄影量)===
5 格,第 k 格 x = 29 + 56k、寬 52;y = 47~124。
沒翻開的格子是橘色牌背 —— 手牌模型會把它讀成 P(白),所以先用顏色判斷
是不是牌背,是牌背就不收那一格的偵測。

★ 2026-11-01 六場錄影(dump_dora):亮的面板 85 次讀值全部讀對(最低 2m 0.45、
  3s 0.53、S 0.66)。錯的全是「面板變暗」的時候 —— 局末結算、換場動畫會蓋一層
  黑,牌背變成咖啡色,舊的「橘色像素比例」判不出牌背,手牌模型就把 4 個暗牌背
  讀成 P 0.9 以上 -> 會投出假的槓寶牌「白」。
  改成:
    牌背   = 格子的色相中位數在橘色範圍、飽和度中位數高(不看亮度,暗了也認得)
    變暗   = 5 格裡最亮的一格,亮度中位數都 < DIM_V -> 整個面板這一幀不讀
           (正常亮度:牌背 V≈242、牌面 V≈200~222;變暗時 ≤131)

=== 輸出 ===
detect(frame) 回傳的格式跟舊 tile_model.detect 一樣(整張畫面座標的
{pai, conf, box, center}),center 直接給那一格的中心,下游 KyokuInfo 的
DORA_SLOT_X / DORA_SLOT_Y 對格子邏輯完全不用改。
"""
import os

import numpy as np

SLOT_X0 = 29
SLOT_PITCH = 56
SLOT_W = 52
SLOT_Y1, SLOT_Y2 = 47, 124
N_SLOTS = 5

CROP = (25, 44, 309, 128)       # x1, y1, x2, y2:5 格再外擴一點
SCALE = 2.0                     # 放大後牌大約 104x154,跟手牌模型的牌差不多
CANVAS = (170, 1610)            # 手牌模型訓練時的輸入大小(固定範圍 1610x170)
PASTE_X, PASTE_Y = 100, 1
CANVAS_GRAY = 40

BACK_HUE = (10, 24)             # 牌背:色相中位數(OpenCV 0~180)
BACK_SAT = 150                  #       飽和度中位數 >= 這個
DIM_V = 180                     # 5 格最亮的一格亮度中位數 < 這個 = 面板變暗
DEFAULT_CONF = 0.25

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Model", "hand_model_best.pt")


def slot_box(k):
    x1 = SLOT_X0 + SLOT_PITCH * k
    return (x1, SLOT_Y1, x1 + SLOT_W, SLOT_Y2)


def slot_center(k):
    x1, y1, x2, y2 = slot_box(k)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def slot_hsv(frame, k):
    """格子的 (色相, 飽和度, 亮度) 中位數。內縮 6 像素避開邊框。"""
    import cv2
    x1, y1, x2, y2 = slot_box(k)
    roi = frame[y1 + 6:y2 - 6, x1 + 6:x2 - 6]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return tuple(int(v) for v in np.median(hsv, axis=0))


def is_back(hsv):
    h, s, _ = hsv
    return BACK_HUE[0] <= h <= BACK_HUE[1] and s >= BACK_SAT


def make_canvas(frame):
    import cv2
    x1, y1, x2, y2 = CROP
    pan = frame[y1:y2, x1:x2]
    big = cv2.resize(pan, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_CUBIC)
    can = np.full((CANVAS[0], CANVAS[1], 3), CANVAS_GRAY, np.uint8)
    h = min(CANVAS[0] - PASTE_Y, big.shape[0])
    can[PASTE_Y:PASTE_Y + h, PASTE_X:PASTE_X + big.shape[1]] = big[:h]
    return can


def to_frame_x(cx_canvas):
    return (cx_canvas - PASTE_X) / SCALE + CROP[0]


class DoraReader:
    def __init__(self, model_path=None, conf=DEFAULT_CONF, model=None):
        if model is None:
            from ultralytics import YOLO
            path = model_path or DEFAULT_MODEL_PATH
            if not os.path.isfile(path):
                raise FileNotFoundError(f"找不到手牌模型:{path}")
            print(f"⏳ 寶牌用手牌模型讀面板:{path}")
            model = YOLO(path)
            from tiles import infer_kwargs
            self.pred_kw = infer_kwargs(path)
        else:
            self.pred_kw = {}
        self.model = model
        self.names = dict(model.names)
        self.conf = conf
        self.last = None            # 每格的讀值,給除錯 / dump_dora 用

    def read_slots(self, frame):
        """回傳 5 格:{"state": back/tile/empty/dim, "pai", "conf", "hsv"}"""
        import cv2
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080))
        slots = []
        for k in range(N_SLOTS):
            hsv = slot_hsv(frame, k)
            slots.append({"state": "back" if is_back(hsv) else "empty",
                          "pai": None, "conf": 0.0, "hsv": hsv})
        if max(sl["hsv"][2] for sl in slots) < DIM_V:
            for sl in slots:
                sl["state"] = "dim"
            self.last = slots
            return slots
        r = self.model.predict(make_canvas(frame), conf=self.conf,
                               **self.pred_kw,
                               verbose=False)[0]
        for b in r.boxes:
            x1, _, x2, _ = (float(v) for v in b.xyxy[0])
            cx = to_frame_x((x1 + x2) / 2)
            k = int(round((cx - SLOT_X0 - SLOT_W / 2) / SLOT_PITCH))
            if not 0 <= k < N_SLOTS:
                continue
            if abs(cx - slot_center(k)[0]) > SLOT_W / 2:
                continue
            s = slots[k]
            if s["state"] == "back":
                continue
            c = float(b.conf[0])
            if c > s["conf"]:
                s.update(state="tile", pai=self.names[int(b.cls[0])],
                         conf=round(c, 4))
        self.last = slots
        return slots

    def detect(self, frame):
        """跟舊 tile_model.detect 同格式(給 RawUiReader 直接替換)。"""
        out = []
        for k, s in enumerate(self.read_slots(frame)):
            if s["state"] != "tile":
                continue
            cx, cy = slot_center(k)
            out.append({"pai": s["pai"], "conf": s["conf"],
                        "box": slot_box(k), "center": (cx, cy)})
        return out
