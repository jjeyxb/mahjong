"""面板文字辨識:局數、本場、供託。

=== 為什麼用樣板比對而不是 OCR ===
這些欄位不是「任意文字」,是**固定字型、固定位置、可能值很少**的圖案:
    局數  東1局 ~ 南4局,最多八種
    本場  通常 0~4,偶爾更多
    供託  通常 0~4
八種固定圖案不需要 OCR。樣板比對沒有外部依賴、結果是確定性的,而且相似度
分數可以直接當信心度用。

OCR 在這個專案已經有過不良紀錄:舊版 vision_bridge.py 因為 pytesseract /
tessdata 抓不到而會整個掛掉,得另外加一層降級保護。為了讀八種固定圖案而引入
一個會讓程式死掉的外部依賴,不划算。

=== 顏色把字從背景挑出來,但分不開兩個欄位 ===
局數是青字(H≈94),面板背景是深色,用色相做遮罩比調灰階門檻穩——色相不隨
明暗變。計數器是白字配深色底,用亮度即可。

⚠️ 局數跟「余N(牌山剩餘)」的顏色**完全一樣**(兩者 H 都是 94)。一開始以為
一個是綠一個是青,實際量過才發現是同一個顏色——分開它們的是位置,不是顏色。
所以 ROI 的上下界要夾緊:局數的字在 y 392~416,余N 在 y 432~453,中間只差
16px。ROI 訂太鬆就會把兩行讀成一團。

(這些 ROI 都是從實機截圖量出來的,而且確認過內容四邊都沒有貼到 ROI 邊界
 ——貼邊代表字被切掉了,樣板就再也對不上。)

=== 正規化保留長寬比 ===
遮罩先裁到外接框,再等比例放進固定畫布(letterbox),不是直接拉伸。
理由:本場的「1」跟「0」寬度差很多,直接拉伸會把兩者變得幾乎一樣。局數那三個
字寬度一致,拉不拉伸都行,但同一套流程共用比較不會出錯。

=== 局數要拆字比對,不能整塊比 ===
「東1局」跟「東2局」有三分之二的內容一模一樣,只有中間那個數字不同,而數字
只佔整體寬度的四分之一。實測收來的八張樣板(東1~4、南1~4):

    整塊比對   錯誤答案就能拿到 0.87,而正確答案在不同局的不同繪製之下大約
               0.90 —— 兩者幾乎重疊,挑錯時分數看起來還是很高,完全看不出來。
    拆字比對   正確答案最低 0.82,錯誤答案最高 0.66 —— 間距 0.16。

拆字之後兩段各自的間距其實更大(風 1.00 vs 0.66,數字 0.82~0.88 vs 0.25~0.46),
但整體分數取「最弱的那一段」,所以實際間距由風那一段決定。這是刻意的:某一段
對得很好、另一段很差,整體就不該算可信。

所以局數一律拆成「風 / 數字 / 局」三段,只比對前兩段(第三個字固定是「局」,
不帶資訊)。

拆字還有一個副作用是好的:樣板可以組合。收到 東1~東4 跟 南1 就等於有了
南1~南4,不必真的看到每一種組合。

拆段用逐欄投影,字與字之間有 4~5px 的全空欄。切不出預期的段數就退回整塊
比對——寧可信心度低一點,也不要因為切壞了就完全讀不出來。

=== 一局之內這些值不會變:確認之後就停止辨識 ===
⚠️ 但供託是例外,見下一節。局數與本場在一局之內是固定的,而樣板比對有明確
的正確/錯誤間距(0.82 vs 0.66)。所以流程是:

    讀不到的幀(被結算面板蓋住、閃光、動畫)跳過,不影響任何事
    認得的讀數累積到 MIN_VOTES 次就確認
    三個欄位都確認之後 done = True,後面每一幀直接跳過,完全不再做辨識
    換局呼叫 reset() 清掉,重新開始讀

這代表遮擋或閃光完全無害:那些幀不投票,等畫面恢復再讀就好,整局有幾百幀
可以用。也代表大部分時間根本不花計算——一局只在開頭讀個幾幀。

⚠️ 換局瞬間面板還顯示上一局的值,單看相似度會把舊值鎖進來。這個用規則擋:
每一局至少會改變局數或本場其中一個(閒家和 -> 局數+1、莊家和或流局 ->
本場+1、流局莊家未聽 -> 兩個都變),所以新讀到的組合不能跟上一局完全相同。
一樣就是還在讀舊面板,繼續等。

=== ⚠️ 供託會在局中改變,所以它不能凍結 ===
有人立直,立直棒就放上桌:那一家的點數立刻少 1000,供託立刻加 1。這件事
發生在**局中**,不是換局的時候。所以「讀到就停」那一套對供託是錯的——
實機症狀是供託永遠停在開局時的值(通常是 0),連帶讓 scores.py 的總和檢查
算錯(這個 bug 真的發生了)。

所以欄位分成兩種:
    固定欄位(kyoku、honba)  累計投票 -> 確認 -> 之後整局不再辨識
    活的欄位(kyotaku)       每一幀都讀,連續 LIVE_CONFIRM 幀讀到同一個值
                              就改成那個值,允許改變

活的欄位不能用「整局累計投票」:累計票是為了「值固定、只有雜訊會變」的情況
設計的,值真的改變時舊值的票會壓住新值。連續 N 幀則相反——它要求新值穩定
才接受,而且舊值一旦不再出現就不再有影響力。N 只要 3:面板是靜止的文字,
真的改了就會一直是新的值;讀不到的幀不算數(不清 streak 也不推進它)。

done 只看固定欄位有沒有確認(外加活的欄位至少讀到過一次),所以「這一局的
面板讀好了沒」這個問題的答案不會因為供託變動而反覆。

=== 門檻是量出來的 ===
MATCH_MIN = 0.70。依據:拆字之後正確答案的分數 ≥0.82(同一個值在不同局的
不同繪製),錯誤答案 ≤0.66。0.70 落在兩者中間。整塊比對的話這個門檻毫無意義
——錯誤答案就有 0.87。
用 run_step8.py --report 重新量過之後,如果兩群的界線移動了,就跟著改。
"""
import os
from collections import Counter

import cv2
import numpy as np

# (x, y, w, h) —— 從 1920x1080 的實機截圖量出來的
ROIS = {
    # 內容 x 921~999 y 392~416,四邊各留 5~7px
    "kyoku": (915, 386, 92, 36),     # 東2局
    # 內容 x 261~275 y 140~172。計數器要用「緊的」ROI:放寬會把資訊框的
    # 外框邊也框進來(實測放到 90x70 就四邊全部貼邊)。
    "honba": (250, 132, 56, 44),     # 本場(左上角資訊框右邊那個計數器)
    "kyotaku": (108, 132, 56, 44),   # 供託/立直棒(左邊那個計數器)
}

CANVAS = (96, 48)     # 正規化後的畫布 (w, h)
MIN_PIXELS = 30       # 遮罩至少要這麼多像素,才算「這個欄位現在看得見」
# 一局之內累計這麼多次「認得」才確認(不必連續)。
#
# 為什麼只要 2 次:量出來的間距是正確 ≥0.82 / 錯誤 ≤0.66,樣板本身就是完整
# 證據,不需要靠票數堆信心。設 2 而不是 1,擋的是「動畫播到一半剛好被讀到」
# 那種一閃而過的畫面——多等一幀不花任何成本。
MIN_VOTES = 2
LIVE_FIELDS = ("kyotaku",)   # 會在局中改變的欄位(見模組說明)
LIVE_CONFIRM = 3             # 活的欄位連續幾幀讀到同一個值才改
VOTE_MARGIN = 2       # 而且最高票要是第二名的這麼多倍
MATCH_MIN = 0.70      # 相似度低於這個值就算不認得(量出來的,見模組說明)

# 要拆字比對的欄位。expect = 預期切出幾段,use = 用哪幾段(對應標籤的第幾個字)。
# 局數「東2局」切三段,只用前兩段:第三個字固定是「局」,不帶資訊。
COMPOSITE = {"kyoku": {"expect": 3, "use": (0, 1)}}

# 共用同一批樣板的欄位。本場與供託的數字是同一套字型——實測同一個數字在兩個
# 計數器之間相似度 0.99~1.00,不同數字 0.30~0.62。共用之後兩邊的樣板互通,
# 其中一邊沒收到的值可以用另一邊的補(實測 kyotaku 缺 3,honba 有)。
SHARED_POOL = {"honba": "digit", "kyotaku": "digit",
               # 點數(scores.py):水平的兩家一組、垂直的兩家一組。
               # 為什麼不是四家共用一個池:實測同一個數字在水平家與垂直家
               # 之間只有 0.11~0.36,那是兩種長寬比完全不同的字。
               "scoreP0": "score_flat", "scoreP2": "score_flat",
               "scoreP1": "score_side", "scoreP3": "score_side"}

# 池裡的鍵是「標籤 + 來源欄位」,因為同一個標籤會有好幾份樣板(P0 的 0 跟
# P2 的 0 都要留著,比對時取最像的那一份)。舊寫法是 setdefault,同一個標籤
# 只留最先讀到的那一份——本場/供託那種 0.99 相似的情況無所謂,點數不行:
# 同一個數字在 P0 與 P2 之間只有 0.70~0.80,丟掉一份就等於把間距讓掉一半。
POOL_SEP = "\t"
SEG_MIN_GAP = 2       # 兩段之間至少要空這麼多欄才算分開(字內部的小縫不算)


_SHAPE_WARNED = set()


def as_mask(a, where="", key=None):
    """把任何一份遮罩/灰階影像壓成二維,壓不了就直接說清楚。

    為什麼需要這個:實機上遇過 cv2 回傳三維 (H, W, 1) 的灰階影像(同一份
    PNG 在別台機器讀出來是二維,所以這是環境差異,不是資料問題)。糟糕的是
    多一個軸不會馬上壞——sum(axis=0) 照樣算、切片照樣切,一路傳到
    np.nonzero() 才爆 "too many values to unpack",離真正的原因隔了三層呼叫。
    所以在「讀進來」跟「開始算」這兩個入口就壓平,而且第一次遇到會印出形狀
    跟版本,這樣下次有人踩到看得出原因。

    key 是「同一類只講一次」的分組(十七張樣板都三維的時候,講十七遍只是
    洗版)。不給就用 where 本身。
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 1:
        k = key or where
        if k not in _SHAPE_WARNED:
            _SHAPE_WARNED.add(k)
            print(f"⚠️ {where} 讀出來是 {a.shape},預期二維;已壓成二維繼續。"
                  f"(cv2 {cv2.__version__} / numpy {np.__version__})")
        return a[:, :, 0]
    raise ValueError(f"{where} 需要二維遮罩,但拿到 {a.shape}")


def _cyan(roi):
    """局數的青字(H≈94)。余N 也是同一個顏色,靠 ROI 的位置分開。"""
    h = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return ((h[:, :, 0] >= 80) & (h[:, :, 0] <= 105)
            & (h[:, :, 1] > 60) & (h[:, :, 2] > 90))


def _bright(roi):
    """計數器的白字。面板底色是深色,亮度分得開。"""
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return g > 150


MASK_OF = {"kyoku": _cyan, "honba": _bright, "kyotaku": _bright}


def mask_of(frame, field):
    """取出某個欄位的二值遮罩(ROI 大小)。"""
    x, y, w, h = ROIS[field]
    m = MASK_OF[field](frame[y:y + h, x:x + w])
    return as_mask(m, f"{field} 的遮罩").astype(np.uint8)


def normalize(mask):
    """裁到外接框、等比例放進固定畫布。看不到東西回傳 None。

    等比例(letterbox)而不是直接拉伸:本場的「1」跟「0」寬度差很多,拉伸會
    讓兩者變得幾乎一樣,比對就分不出來了。
    """
    mask = as_mask(mask, "normalize() 的輸入")
    if mask.sum() < MIN_PIXELS:
        return None
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    tw, th = CANVAS
    scale = min(tw / cw, th / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    small = cv2.resize(crop * 255, (nw, nh), interpolation=cv2.INTER_AREA)
    small = as_mask(small, "normalize() 縮放後", key="normalize_resize")  # cv2 5.x
    out = np.zeros((th, tw), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = small
    return (out > 127).astype(np.uint8)


def similarity(a, b):
    """兩個正規化遮罩的相似度(交集/聯集)。0~1,越大越像。

    用 IoU 而不是逐點相同率:背景佔畫布大部分面積,逐點相同率會因為「背景都
    對上了」而全部逼近 1,分不出好壞。

    ★ 兩邊都先壓成二維。這是最後一道防線:不管樣板是從哪條路讀進來的
      (面板、點數、自風),cv2 5.x 給的 (H, W, 1) 都會在這裡被擋住,不會
      變成 broadcast 錯誤。第一道防線還是在「讀檔」那裡(會印警告)。
    """
    a = as_mask(a, "similarity() 的輸入", key="similarity")
    b = as_mask(b, "similarity() 的輸入", key="similarity")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def split_parts(norm, min_gap=SEG_MIN_GAP):
    """用逐欄投影把正規化後的遮罩切成幾段,每段各自再正規化一次。

    再正規化一次是必要的:切出來的「東」跟「1」寬高差很多,不各自放進統一的
    畫布就沒辦法互相比對。
    """
    cols = norm.sum(axis=0)
    spans, start = [], None
    for i, c in enumerate(cols):
        if c > 0 and start is None:
            start = i
        elif c == 0 and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(cols) - 1))
    merged = []
    for sp in spans:
        if merged and sp[0] - merged[-1][1] - 1 < min_gap:
            merged[-1] = (merged[-1][0], sp[1])
        else:
            merged.append(sp)
    return [normalize(norm[:, a:b + 1]) for a, b in merged]


class Templates:
    """一疊已知圖案。檔名格式 <欄位>__<標籤>.png。"""

    def __init__(self, folder):
        self.folder = folder
        self.items = {}       # field -> {label: normalized mask}
        # 拆字比對用的零件庫:field -> [ {字元: 遮罩}, ... ] 每個使用到的段一個
        # 字典。這是從 items 自動拆出來的,不需要另外收檔案——收了 kyoku__E1
        # 就同時貢獻了「東」跟「1」兩個零件。
        self.parts = {}
        self.pools = {}       # 池名 -> {label: 遮罩}(本場/供託共用數字)
        self.load()

    def load(self):
        self.items = {}
        if not os.path.isdir(self.folder):
            return
        for name in sorted(os.listdir(self.folder)):
            if not name.endswith(".png") or "__" not in name:
                continue
            field, label = name[:-4].split("__", 1)
            img = cv2.imread(os.path.join(self.folder, name),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = as_mask(img, f"樣板 {name}", key="樣板檔")
            self.items.setdefault(field, {})[label] = (img > 127).astype(np.uint8)
        self._build_parts()
        self._build_pools()

    def _build_pools(self):
        """把共用同一套字型的欄位合併成一個池。"""
        self.pools = {}
        for field, pool in SHARED_POOL.items():
            for label, norm in self.items.get(field, {}).items():
                if label.startswith("unknown"):
                    continue
                key = f"{label}{POOL_SEP}{field}"
                self.pools.setdefault(pool, {})[key] = norm

    def _look_in(self, field):
        """這個欄位要在哪一批樣板裡找。"""
        pool = SHARED_POOL.get(field)
        if pool and self.pools.get(pool):
            return self.pools[pool]
        return {k: v for k, v in self.items.get(field, {}).items()
                if not k.startswith("unknown")}

    def _build_parts(self):
        """把 COMPOSITE 欄位的整塊樣板拆成零件庫。"""
        self.parts = {}
        for field, spec in COMPOSITE.items():
            libs = [{} for _ in spec["use"]]
            for label, norm in self.items.get(field, {}).items():
                if label.startswith("unknown") or len(label) < len(spec["use"]):
                    continue
                segs = split_parts(norm)
                if len(segs) != spec["expect"]:
                    continue          # 切不出預期段數的樣板不進零件庫
                for k, si in enumerate(spec["use"]):
                    libs[k].setdefault(label[k], segs[si])
            if all(libs):
                self.parts[field] = libs

    def save(self, field, label, norm):
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, f"{field}__{label}.png")
        cv2.imwrite(path, norm * 255)
        self.items.setdefault(field, {})[label] = norm
        self._build_parts()
        self._build_pools()
        return path

    def count(self, field):
        return len(self.items.get(field, {}))

    def match(self, field, norm):
        """辨識:這是哪一個值。回傳 (標籤, 分數),沒有樣板回傳 (None, 0.0)。

        COMPOSITE 的欄位優先用拆字比對(間距大得多,見模組說明),切不出預期
        段數才退回整塊比對。

        不會回傳 unknownN 那種還沒命名的樣板——回報一個叫 unknown3 的答案
        沒有任何意義。那些檔案只在去重時有用(見 duplicate_of)。

        注意這裡「不」套門檻——門檻由呼叫端決定。工具要看得到未達門檻的分數
        才有辦法把門檻訂在對的位置。
        """
        if field in COMPOSITE and field in self.parts:
            label, score = self._match_parts(field, norm)
            if label is not None:
                return label, score
        best, score = None, 0.0
        for label, tpl in self._look_in(field).items():
            s = similarity(norm, tpl)
            if s > score:
                best, score = label, s
        # 池裡的鍵帶著來源欄位(見 POOL_SEP),回傳的答案只要標籤那一半
        return (best.split(POOL_SEP)[0] if best is not None else None), score

    def duplicate_of(self, field, norm):
        """去重:這張圖是不是已經存過了。回傳 (標籤, 分數)。

        跟 match() 是兩個不同的問題,所以是兩個方法:
          match        「這是哪一個值」——會拆字,而且不看未命名的樣板
          duplicate_of 「這張圖存過沒」——整張比,而且要看未命名的樣板
        用 match 去做去重會出錯:拆字比對回傳的是「零件的分數」,就算整張圖
        一模一樣也可能只有 0.83,於是同一張圖被重複存進去。
        """
        best, score = None, 0.0
        for label, tpl in self.items.get(field, {}).items():
            s = similarity(norm, tpl)
            if s > score:
                best, score = label, s
        return best, score

    def _match_parts(self, field, norm):
        """拆字比對。切不出預期段數回傳 (None, 0.0) 讓呼叫端退回整塊比對。"""
        spec = COMPOSITE[field]
        segs = split_parts(norm)
        if len(segs) != spec["expect"]:
            return None, 0.0
        chars, scores = [], []
        for k, si in enumerate(spec["use"]):
            seg = segs[si]
            if seg is None:
                return None, 0.0
            ch, sc = None, 0.0
            for c, tpl in self.parts[field][k].items():
                v = similarity(seg, tpl)
                if v > sc:
                    ch, sc = c, v
            if ch is None:
                return None, 0.0
            chars.append(ch)
            scores.append(sc)
        # 分數取最低的那一段:某一段對得很好、另一段很差,整體就不該算可信。
        return "".join(chars), min(scores)


class PanelReader:
    """把面板讀成穩定的值。

    一局之內這些值不會變,所以用「整局累計投票」而不是「連續 N 幀相同」:
    讀不到的幀不投票,偶爾的誤讀被大量正確讀數淹沒。換局呼叫 reset() 清票
    ——那是這些值唯一會改變的時機。
    """

    def __init__(self, templates, fields=("kyoku", "honba"),
                 match_min=MATCH_MIN, min_votes=MIN_VOTES,
                 margin=VOTE_MARGIN, stale_check=("kyoku", "honba"),
                 live=LIVE_FIELDS, live_confirm=LIVE_CONFIRM):
        self.templates = templates
        self.fields = tuple(fields)
        # 會在局中改變的欄位:每一幀都讀,連續幾幀一致才改(見模組說明)
        self.live = tuple(f for f in live if f in self.fields)
        self.live_confirm = live_confirm
        self._streak = {f: (None, 0) for f in self.fields}
        self.match_min = match_min
        self.min_votes = min_votes
        self.margin = margin
        # 這幾個欄位的組合每一局都必須改變(見模組說明)。用來擋「換局瞬間
        # 讀到上一局的面板」。
        self.stale_check = tuple(f for f in stale_check if f in self.fields)
        self.value = {f: None for f in self.fields}    # 已確認的值
        self.score = {f: 0.0 for f in self.fields}
        self.votes = {f: Counter() for f in self.fields}
        self.last = {f: (None, 0.0) for f in self.fields}  # 這一幀的原始讀值
        self.previous = None     # 上一局確認的組合(換局時由 reset 帶過來)
        self.conflicts = []      # 確認之後多數票又變了 —— 誤讀的證據
        self.unread = {f: 0 for f in self.fields}      # 讀不到的幀數
        self.reads = {f: 0 for f in self.fields}       # 真的做了辨識的次數
        self.frames_used = 0     # 至少辨識了一個欄位的幀數
        self.stale_frames = 0    # 讀到的組合跟上一局一樣的幀數

    @property
    def done(self):
        """這一局的面板讀好了沒。

        「好了」= 每個欄位都有值、而且不是上一局的殘影。之後固定欄位不再
        辨識,活的欄位(供託)照樣每一幀讀——done 說的是「讀好了」,不是
        「不會再變了」。
        """
        if any(self.value[f] is None for f in self.fields):
            return False
        return not self._is_stale()

    @property
    def frozen(self):
        """固定欄位全部確認,可以不必再辨識它們了。"""
        return all(self.value[f] is not None
                   for f in self.fields if f not in self.live)

    def _is_stale(self):
        """讀到的組合跟上一局完全相同 -> 還在看上一局的面板。"""
        if self.previous is None or not self.stale_check:
            return False
        now = tuple(self.value[f] for f in self.stale_check)
        return None not in now and now == self.previous

    def update(self, frame):
        """吃一幀。回傳這一幀「剛確認而且跟之前不同」的欄位 {field: value}。

        固定欄位確認之後就不再辨識(一局之內不會變);活的欄位(供託)每一幀
        都讀,因為有人立直它就會加 1。
        """
        if self.done and not self.live:
            return {}
        self.frames_used += 1
        changed = {}
        for field in self.fields:
            live = field in self.live
            if (not live and self.value[field] is not None
                    and not self._is_stale()):
                continue          # 這個固定欄位確認過了,不用再讀
            self.reads[field] += 1
            norm = normalize(mask_of(frame, field))
            if norm is None:
                # 讀不到(面板被結算畫面蓋住、閃光、動畫)。不投票、不動已確認
                # 的值——「看不到」跟「看到別的」是兩件事。
                # 活的欄位也一樣:不推進 streak,也不清掉它。
                self.last[field] = (None, 0.0)
                self.unread[field] += 1
                continue
            label, score = self.templates.match(field, norm)
            self.last[field] = (label, score)
            if label is None or score < self.match_min:
                continue          # 認不得也不投票,不要把雜訊算進去

            if live:
                # 連續 N 幀讀到同一個值才改。用連續而不是累計:累計票是為了
                # 「值固定、只有雜訊會變」設計的,值真的改變時舊票會壓住新值。
                was, streak = self._streak[field]
                streak = streak + 1 if was == label else 1
                self._streak[field] = (label, streak)
                self.score[field] = score
                if streak >= self.live_confirm and self.value[field] != label:
                    self.value[field] = label
                    changed[field] = label
                continue

            self.votes[field][label] += 1
            self.score[field] = score

            top = self.votes[field].most_common(2)
            best, n = top[0]
            second = top[1][1] if len(top) > 1 else 0
            if n < self.min_votes or n < self.margin * max(second, 1):
                continue
            if self.value[field] == best:
                continue
            if self.value[field] is not None:
                # 固定欄位在一局之內不該改變。改變了就是先前那個是誤讀。
                # (活的欄位走上面 live 那一段,不會走到這裡。)
                self.conflicts.append({
                    "field": field, "was": self.value[field], "now": best,
                    "votes": dict(self.votes[field])})
            self.value[field] = best
            changed[field] = best

        if self._is_stale():
            # 讀到的是上一局的面板。清掉重讀,不要把舊值當成這一局的答案。
            self.stale_frames += 1
            for f in self.stale_check:
                self.value[f] = None
                self.votes[f].clear()
            return {}
        return changed

    def state(self):
        """給畫面顯示的一行(ASCII)。"""
        if self.done:
            return "PANEL DONE  " + "  ".join(
                f"{f}={self.value[f]}" for f in self.fields)
        bits = []
        for f in self.fields:
            label, score = self.last[f]
            shown = self.value[f] if self.value[f] is not None else "?"
            tally = ",".join(f"{k}:{v}" for k, v in
                             self.votes[f].most_common(3))
            bits.append(f"{f}={shown}({label or '-'}:{score:.2f} [{tally}])")
        return "  ".join(bits)

    def report(self):
        lines = [f"  實際做辨識的幀數 {self.frames_used}"
                 + f",各欄位辨識次數 {self.reads}"
                 + (f",讀到上一局殘影 {self.stale_frames} 幀"
                    if self.stale_frames else "")]
        for f in self.fields:
            total = sum(self.votes[f].values())
            lines.append(f"  {f}: 確認 {self.value[f] or '(讀不出來)'}  "
                         f"有效票 {total}  讀不到 {self.unread[f]} 幀  "
                         f"票數 {dict(self.votes[f].most_common())}")
        for c in self.conflicts:
            lines.append(f"  ⚠️ {c['field']} 在同一局內從 {c['was']} 變成 "
                         f"{c['now']} —— 一局之內不該改變,其中一個是誤讀。"
                         f"票數 {c['votes']}")
        return lines

    def reset(self):
        """換局:清票,並且把剛才那一局的組合記下來當作「殘影」的比對基準。

        ⚠️ 活的欄位(供託)也一起清掉。供託會跨局延續(沒人和就一直留在桌上),
        但那不表示可以沿用上一局讀到的值——重讀幾幀就有了,沿用反而會在
        「上一局結束時有人立直、這一局開始前被收走」的情況下讀到舊值。
        """
        if self.stale_check and all(self.value[f] is not None
                                    for f in self.stale_check):
            self.previous = tuple(self.value[f] for f in self.stale_check)
        self.value = {f: None for f in self.fields}
        self.votes = {f: Counter() for f in self.fields}
        self.unread = {f: 0 for f in self.fields}
        self.reads = {f: 0 for f in self.fields}
        self._streak = {f: (None, 0) for f in self.fields}
        self.frames_used = 0
        self.stale_frames = 0
