"""自風(牌桌中央儀表板四個角的東南西北),用來算 mjai 的絕對座位編號。

=== 為什麼需要 ===
mjai 的 actor 是「這一場對局的絕對座位編號」,起家是 0,整場不變。而畫面上的
P0..P3 是「相對於鏡頭」的位置,P0 永遠是自己。兩者只有在「自己剛好是起家」
的時候才一致——session.SEAT_INDEX 原本寫死 {P0:0,P1:1,P2:2,P3:3} 就是這個
假設,錯了 Mortal 會收到別人的牌,而且不會有任何錯誤訊息。

=== 怎麼算 ===
    莊家的 actor  = kyoku - 1          (mjai 的規定:東1局莊家是 0,東2局是 1…)
    某一家的 actor = (莊家 + 那一家的自風index) mod 4
                     自風 index:東=0 南=1 西=2 北=3

自風只告訴我們「離莊家幾個座位」,絕對編號要靠面板的局數。兩個都要。

實例(meld_difference_data/P0(1).jpg,面板是「東2局」):
    莊家 actor = 1
    P0 的自風是西(2) -> actor(P0) = (1 + 2) mod 4 = 3
    畫面上紅底的東在 P2  -> actor(P2) = (1 + 0) mod 4 = 1  ✓ 跟莊家對上

=== 四個角是哪一家 ===
儀表板四個角各一個風牌,每一個都朝向它自己那一家,所以旋轉方向都不同:

    左下 = P0(自己,正放)      右下 = P1
    左上 = P3                  右上 = P2(倒著)

⚠️ 因為旋轉加上桌面的透視,同一個字在四個角長得不一樣——樣板要「每個角各一
   套」,不能共用。四個角 × 四個字 = 16 張。

=== 東是紅底 ===
東的牌面是紅底深色筆畫,其他三個是深藍底淺色筆畫。所以:
  ・取遮罩不能用亮度門檻(東會整個反過來),要用「跟底色的距離」。
    底色取 ROI 內的中位數——牌面佔多數,筆畫是少數。
  ・紅底本身就是「這一家是莊家」的獨立訊號,拿來跟風的判讀交叉檢查。

=== ★ 以 P0 為準,其餘用推的 ===
四家的風是東南西北各一個、而且照座位順序遞增,所以**知道一家就唯一決定其他
三家**。程式只把 P0(左下角、自己的位置、正放)當真相來源:

    P0 讀不出來 -> 這一幀作廢(不會拿其他角來頂)
    其他三家讀得到 -> 當交叉檢查,跟推出來的不一樣就作廢
    其他三家讀不到或沒樣板 -> 單純用推的,不影響結果

好處是只要 P0 那一個角的四張樣板就能運作,不必湊滿 16 張;而多的樣板只會
讓它更安全,不會讓它更容易壞。

⚠️ 旋轉不影響辨識——樣板是每個角各一套,比對永遠是同一個角對同一個角,
   旋轉早就烤進樣板裡了。實測四個角的分隔度(對的最低 vs 錯的最高):
       P0 正放 +0.545   P2 倒轉 +0.452   P3 側轉 +0.479   P1 側轉 +0.309
   倒轉的那個角一點都不比正放的差。以 P0 為準是為了「少依賴」,不是因為
   其他角讀不準。

=== 檢查(都是規則,不是門檻)===
  1. 其他角讀到的風,必須跟 P0 推出來的一致
  2. 紅底的那一家必須是東
  3. 算出來的四個 actor 照座位順序必須 +1 遞增
  再加上 session 那邊的第四重:莊家必須是這一局第一個打牌的人。
  任何一項不過就不確認——寧可回報「讀不出來」,也不要給 Mortal 一個錯的編號。
"""
import os

import cv2
import numpy as np

from panel import as_mask, normalize, similarity

# 四個角的 ROI(牌面內部,不含外框)。量自 meld_difference_data 的 12 張截圖:
# 先用寬鬆的框讓程式找出字的外接框,取 12 張的聯集再往外放 5px。
# ⚠️ 框要收在牌面裡面。放到牌面外會讓「底色中位數」取到桌面的亮藍,整塊牌面
#    就被當成筆畫了(第一版就是這樣,三張圖的字框莫名其妙變成兩倍大)。
ROIS = {
    "P0": (797, 496, 53, 36),      # 左下,正放
    "P1": (1072, 494, 55, 44),     # 右下
    "P2": (1058, 319, 54, 34),     # 右上,倒著
    "P3": (814, 318, 48, 34),      # 左上
}

SEAT_ORDER = ("P0", "P1", "P2", "P3")   # 座位順序 = 風往前推進的方向
WINDS = ("E", "S", "W", "N")            # mjai 記法,index 就是自風 index
WIND_TEXT = {"E": "東", "S": "南", "W": "西", "N": "北"}

STROKE_DIST = 130     # 跟底色差多少算筆畫(三通道絕對差相加)
RED_MARGIN = 25       # 底色的 R 比 B 大這麼多就算紅底
MIN_PIXELS = 60       # 遮罩至少要這麼多像素才算「這個角現在看得見」
# 小於這個面積的連通塊當雜訊丟掉。
# ★ 這不是美觀問題,是正確性問題:normalize() 會裁到「所有非零像素」的外接框,
#   所以角落一顆孤立的點就把框撐大一圈,字被縮小、位移,IoU 直接崩掉。
#   實測兩張肉眼完全一樣的截圖,只差角落一顆雜訊,分數就從 1.00 掉到 0.37。
#   筆畫的連通塊實測都在 30px 以上(字大約 45x26),12 離兩邊都很遠。
MIN_BLOB = 12
MATCH_MIN = 0.70      # 跟 panel 一樣的門檻
MIN_VOTES = 2         # 累計幾次通過全部檢查才確認


def mask_of(frame, seat):
    """取出一個角的筆畫遮罩與底色。回傳 (mask, bg)。

    底色取 ROI 內的中位數,不是外圈:牌面佔 ROI 的大部分,筆畫是少數,所以
    中位數就是牌面底色。這樣紅底(深色筆畫)跟藍底(淺色筆畫)可以用同一套
    程式處理——「離底色遠」在兩種情況下都是筆畫。
    """
    x, y, w, h = ROIS[seat]
    c = frame[y:y + h, x:x + w].astype(np.int16)
    if c.size == 0:
        return None, None
    bg = np.median(c.reshape(-1, c.shape[-1]), axis=0)
    m = (np.abs(c - bg).sum(axis=2) > STROKE_DIST)
    m = as_mask(m, f"{seat} 的自風遮罩").astype(np.uint8)
    return drop_specks(m), bg


def drop_specks(mask, min_blob=MIN_BLOB):
    """丟掉太小的連通塊。理由見 MIN_BLOB ——這一步關係到正確性,不是美觀。"""
    n, lab, st, _c = cv2.connectedComponentsWithStats(mask, 8)
    lab = as_mask(lab, "連通塊標籤", key="cc_labels")   # cv2 5.x 可能多一軸
    out = np.zeros_like(mask)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_blob:
            out[lab == k] = 1
    return out


def is_red(bg):
    """紅底 = 東 = 莊家。bg 是 BGR。"""
    return bg is not None and float(bg[2]) > float(bg[0]) + RED_MARGIN


def next_wind(w):
    return WINDS[(WINDS.index(w) + 1) % 4]


def seat_after(seat):
    return SEAT_ORDER[(SEAT_ORDER.index(seat) + 1) % 4]


def consistent(winds):
    """四家的風照座位順序是不是 東→南→西→北 的循環。

    winds 是 {seat: 風};有 None 的位置代表沒讀到,會被跳過。
    回傳 (通過嗎, 說明)。
    """
    known = {s: w for s, w in winds.items() if w}
    if len(known) < 2:
        return True, ""          # 資訊不足,不算失敗
    for s, w in known.items():
        nxt = seat_after(s)
        if nxt in known and known[nxt] != next_wind(w):
            return False, (f"{s}={WIND_TEXT[w]} 的下一家 {nxt} 應該是"
                           f"{WIND_TEXT[next_wind(w)]},卻讀到"
                           f"{WIND_TEXT[known[nxt]]}")
    if len(known) == 4 and len(set(known.values())) != 4:
        return False, f"四家的風有重複:{ {s: WIND_TEXT[w] for s, w in known.items()} }"
    return True, ""


def derive_from(seat, wind):
    """知道一家的自風,其餘三家照座位順序推出來。

    這不是猜:四家的風是東南西北各一個,而且照座位順序遞增(P0 的下一家 +1)。
    知道任何一家就唯一決定其他三家。
    """
    base = WINDS.index(wind)
    k0 = SEAT_ORDER.index(seat)
    return {s: WINDS[(base + (SEAT_ORDER.index(s) - k0)) % 4]
            for s in SEAT_ORDER}


def fill_missing(winds):
    """剛好缺一家的時候,用座位順序把它推出來。回傳 (補好的 dict, 是推的嗎)。

    ⚠️ 只在「其他三家都讀到而且互相一致」時才推。
    (WindReader 現在改成以 P0 為準、其餘用 derive_from 推,所以這個函式主要
     給 run_step11 收樣板時用——那時候要盡量把四個角都補齊。)
    """
    miss = [s for s, w in winds.items() if not w]
    if len(miss) != 1:
        return dict(winds), False
    ok, _why = consistent(winds)
    if not ok:
        return dict(winds), False
    s = miss[0]
    prev = SEAT_ORDER[(SEAT_ORDER.index(s) - 1) % 4]
    out = dict(winds)
    out[s] = next_wind(winds[prev])
    return out, True


def dealer_of(winds):
    """哪一家是莊家(風是東的那一家)。讀不出來回傳 None。"""
    for s, w in winds.items():
        if w == "E":
            return s
    return None


def actors_of(winds, kyoku):
    """{seat: mjai 的 actor 編號}。kyoku 是這一局的局數(東2局的 2)。

    莊家的 actor = kyoku - 1(mjai 的規定),其餘 = (莊家 + 自風index) mod 4。
    kyoku 或風缺一個就回傳 None ——絕對編號一定要兩個都有,不能只靠一個猜。
    """
    if not kyoku or any(not w for w in winds.values()):
        return None
    oya = (int(kyoku) - 1) % 4
    return {s: (oya + WINDS.index(w)) % 4 for s, w in winds.items()}


def actors_increment(actors):
    """算出來的 actor 照座位順序必須 +1 遞增。這是第三重檢查。"""
    if not actors:
        return False, "還沒算出 actor"
    for s in SEAT_ORDER:
        nxt = seat_after(s)
        if (actors[s] + 1) % 4 != actors[nxt]:
            return False, (f"{s} 的 actor={actors[s]},下一家 {nxt} 應該是"
                           f"{(actors[s] + 1) % 4},卻是 {actors[nxt]}")
    return True, ""


class WindTemplates:
    """每個角各一套樣板。檔名 seat__wind.png,例如 P0__W.png。"""

    def __init__(self, folder):
        self.folder = folder
        self.items = {s: {} for s in ROIS}
        if folder and os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if not fn.endswith(".png") or "__" not in fn:
                    continue
                seat, rest = fn[:-4].split("__", 1)
                if seat in self.items and rest in WINDS:
                    img = cv2.imread(os.path.join(folder, fn),
                                     cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # ★ 一定要過 as_mask:cv2 5.x 的 IMREAD_GRAYSCALE 會回傳
                        #   (H, W, 1) 三維陣列(4.x 是二維)。沒壓平的話比對時
                        #   (48,96) 對 (48,96,1) 會 broadcast 失敗,程式當掉。
                        #   面板樣板早就這樣處理,自風樣板當初漏了——實際在
                        #   cv2 5.0.0 的環境跑 run_live 第一幀就爆。
                        img = as_mask(img, f"自風樣板 {fn}", key="自風樣板檔")
                        self.items[seat][rest] = (img > 127).astype(np.uint8)

    def save(self, seat, wind, norm):
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, f"{seat}__{wind}.png")
        cv2.imwrite(path, norm * 255)
        self.items[seat][wind] = norm
        return path

    def match(self, seat, norm):
        """回傳 (風, 分數)。沒有樣板回傳 (None, 0.0)。不套門檻——門檻由呼叫端決定。"""
        best, score = None, 0.0
        for wind, tpl in self.items.get(seat, {}).items():
            s = similarity(norm, tpl)
            if s > score:
                best, score = wind, s
        return best, score

    def count(self, seat=None):
        if seat:
            return len(self.items.get(seat, {}))
        return sum(len(v) for v in self.items.values())


class WindReader:
    """讀四個角的自風,通過三重檢查才確認。

    跟 PanelReader 一樣是「讀到就好,不用一直讀」:確認之後就停,直到 reset()。
    自風每一局都會變,所以換局一定要 reset()。
    """

    def __init__(self, templates, min_votes=MIN_VOTES, match_min=MATCH_MIN):
        self.tpl = templates
        self.min_votes = min_votes
        self.match_min = match_min
        self.reset()

    def reset(self):
        self.winds = {s: None for s in ROIS}
        self.confirmed_by = []    # 直接讀到的那幾家
        self.inferred = []        # 用座位順序推出來的那幾家
        self.red = None           # 紅底在哪一家(獨立訊號)
        self.done = False
        self.votes = 0
        self.rejected = []        # [(幀, 為什麼不算數)]
        self._last = None

    def read(self, frame):
        """讀一次,不改變狀態。回傳 (winds, red, 每家的分數)。"""
        winds, scores = {}, {}
        red = None
        for seat in ROIS:
            m, bg = mask_of(frame, seat)
            if m is None or m.sum() < MIN_PIXELS:
                winds[seat], scores[seat] = None, 0.0
                continue
            if is_red(bg):
                red = seat
            norm = normalize(m)
            if norm is None:
                winds[seat], scores[seat] = None, 0.0
                continue
            w, s = self.tpl.match(seat, norm)
            winds[seat] = w if s >= self.match_min else None
            scores[seat] = s
        return winds, red, scores

    def update(self, frame, frame_no=0):
        """吃一幀。通過全部檢查並累計夠票數就確認。回傳這一幀有沒有更新。

        ★ 以 P0 為準,其餘三家用座位順序推。P0 是自己的位置、正放、不旋轉,
          而且是四個角裡分隔度最好的(實測對 0.990~1.000 / 錯最高 0.445)。
          這樣只要 P0 那一個角的四張樣板就能運作,不必湊滿 16 張。

        ★ 其他三家讀得到的話不是拿來當答案,是拿來當**交叉檢查**:讀到而且
          跟推出來的不一樣 -> 這一幀作廢。讀不到或沒樣板就單純用推的。
          這樣「多的樣板只會讓它更安全,不會讓它更容易壞」。
        """
        if self.done:
            return False
        got, red, scores = self.read(frame)
        why = self._check(got, red)
        if why:
            self.rejected.append((frame_no, why))
            self._last = None
            self.votes = 0
            return False
        full = derive_from("P0", got["P0"])
        key = tuple(full[s] for s in SEAT_ORDER)
        if key != self._last:
            self._last, self.votes = key, 1
        else:
            self.votes += 1
        self.winds = full
        self.confirmed_by = [s for s in SEAT_ORDER if got[s]]
        self.inferred = [s for s in SEAT_ORDER if not got[s]]
        self.red = red
        if self.votes >= self.min_votes:
            self.done = True
            return True
        return False

    def _check(self, got, red):
        """檢查這一幀。全部通過回傳空字串,否則回傳不通過的理由。

        got 是「直接讀到的」(沒讀到的是 None)。以 P0 為準往外推,再拿其他
        三家讀到的結果來對。
        """
        if not got.get("P0"):
            return "P0(左下角,自己的自風)讀不出來——那是唯一的真相來源"
        full = derive_from("P0", got["P0"])
        for s in SEAT_ORDER:
            if got[s] and got[s] != full[s]:
                return (f"P0={WIND_TEXT[got['P0']]} 推出 {s} 應該是"
                        f"{WIND_TEXT[full[s]]},但那一角讀到的是"
                        f"{WIND_TEXT[got[s]]}——兩個來源對不上")
        if red is not None and full[red] != "E":
            return (f"紅底在 {red},但推出來那一家是 {WIND_TEXT[full[red]]}"
                    f"——紅底一定是東")
        return ""

    def dealer(self):
        return dealer_of(self.winds) if self.done else None

    def actors(self, kyoku):
        """{seat: actor};算不出來回傳 None。算出來一定通過遞增檢查。"""
        if not self.done:
            return None
        a = actors_of(self.winds, kyoku)
        if a is None:
            return None
        ok, _why = actors_increment(a)
        return a if ok else None

    def report(self):
        if not self.done:
            miss = [s for s, w in self.winds.items() if not w]
            return [f"自風:還沒讀出來(缺 {miss or '全部'},"
                    f"擋下 {len(self.rejected)} 次)"]
        line = "  ".join(f"{s}={WIND_TEXT[w]}" for s, w in self.winds.items())
        out = [f"自風:{line}   莊家={self.dealer()}"]
        out.append(f"  直接讀到 {self.confirmed_by};"
                   f"{self.inferred or '無'} 是用座位順序推的")
        if self.rejected:
            out.append(f"  擋下 {len(self.rejected)} 次不通過檢查的讀取")
        return out
