"""鳴牌區的固定裁切範圍 + 純工具。零相依 —— 不 import ultralytics、不碰 cv2。

從舊專案 mjai_live/crop_zones.py 搬過來(只留鳴牌用得到的部分)。
裁切範圍只定義在這裡一次:建資料集那邊 import 它、推論這邊 import 它、
而且推論載入權重**之前**會先核對權重旁邊那份 crop_meta.json,對不上就拒絕
執行(verify_meta / load_meta)。


★★ 這支檔案存在的理由,是舊專案真的踩過的坑 ★★
  第一版的鳴牌/手牌資料集是用一支後來消失的程式裁的,推論端把座標**抄了一份
  寫死在自己檔案裡**。鳴牌端要接的時候已經無從抄起,只能從資料集反推。
  反推的數字是對的,但「兩邊各寫一份、靠人記得同步」這個結構本身就是錯的:
  只要一邊改了另一邊沒跟上,模型就會收到它沒看過的輸入,**而且不會報錯**,
  只會莫名其妙讀不到牌。


=== v3.2:只框鳴牌區,固定座標 ===
每一家一塊**固定**的框,大小取「這一家鳴滿四組時會長到多大」。不管當下鳴了
幾組,裁出來都是同一個尺寸,鳴不滿的部分是空桌面。
    固定尺寸 = **零依賴**。推論時裁同一塊座標,訓練與推論完全一致,
    不需要先跑任何別的模型。
    (用局勢模型每幀現場偵測 meld_area 可以裁得更緊、解析度更好,但那會讓
     鳴牌辨識的成敗綁在 region_model 上。舊專案 2026-10-05 權衡過,選零依賴。)

空白不是隨機的:每一家的鳴牌都從固定的一邊開始長 ——
    P0 靠右(右邊界固定往左長)   P1 靠上   P2 靠左   P3 靠下
位置一致,對模型是線索不是雜訊。

座標的三個來源(舊專案 crop_zones.py 有完整記載):
  (a) 四張四暗槓截圖用顏色分割自動量到的牌串 + 10px
  (b) 整份資料集 520 張、4278 個鳴牌框實際佔到的範圍
  (c) 使用者 2026-10-04 指定:每家有一個邊改用舊的「手牌+鳴牌」框邊界,
      就是鳴牌往自己手牌方向長的那一側(加槓疊上去的第四張會往那邊多一截)
  ★ (a) 單獨不夠:四暗槓的四張是平排的,加槓疊上去那種畫面截圖裡沒有。
  ★ (b) 單獨不夠:資料集裡沒有人鳴滿四組,P0 左邊界只到 1097,
        而四組會一路長到 667。別看到資料集只用到 1097 就往右縮。


=== ⚠️ 跟 hand.MAX_ZONE_P0 是**兩回事**,不要混 ===
    meld.MELD_ZONES["P0"]  = (657, 905, 1176, 146)   只有鳴牌區
    hand.MAX_ZONE_P0       = (220, 905, 1610, 170)   手牌+鳴牌的長條
兩個都對,因為它們配的是**不同的權重**:手牌模型是用後者訓練的。
★ 絕對不要把 hand_model 改成吃這裡的座標 —— 那會餵舊權重一塊它沒看過的
  範圍,而且不會報錯,只會讀不到牌。要換就得連權重一起重訓。
"""
import json

REF_W, REF_H = 1920, 1080

# (x, y, w, h),1920x1080 基準。只框鳴牌區,固定尺寸。
MELD_ZONES = {
    "P0": (657, 905, 1176, 146),
    "P1": (1474, 32, 286, 608),
    "P2": (383, 25, 783, 88),
    "P3": (55, 221, 323, 753),
}

SEATS = ("P0", "P1", "P2", "P3")

# ================================================================
# P0 的例外:框得再準,手牌還是會進來
# ================================================================
# 這不是調座標能解決的,是版面本身決定的:
#     P0 手牌  中心 x  456~1532   中心 y  993~1001   框高 138~159
#     P0 鳴牌  中心 x 1149~1779   中心 y  994~1005   框高  77~101
# 兩群的中心 y **完全重疊**,x 也有 1149~1532 重疊 —— 沒鳴牌時 14 張手牌向右
# 攤開,正好攤在「四組鳴牌會出現」的位置上。兩種狀態不會同時發生,但固定框
# 得各自蓋到,所以鳴牌區的框必然涵蓋手牌的右半段。
#
# 唯一分得開的那一維是**框的高度**:躺平/側放的鳴牌比立著的手牌矮四成,
# 兩群之間空了 37px(101 對 138),中間一個框都沒有。
#
# ★ 建資料集時丟的是**標籤**,不是像素。裁切圖裡手牌照樣看得見,讓模型自己
#   學會「立著的牌不是鳴牌」—— 跟 v1 把手牌塗黑是相反的做法。塗黑會製造出
#   實機不存在的畫面(v1 就是這樣壞的),而且推論時得靠 hand_model 逐幀重做
#   同樣的塗黑,等於把鳴牌辨識綁在另一個模型上。
#   驗證成立:訓練後在 P0 的 2080 張裁切圖上,框高 >= 120 的偵測是 **0 個**。
#
# ⚠️ 推論端比的是**夾過**的高度,跟建資料集時(原始高度)不一樣,所以門檻
#    要各自算一次站不站得住:
#      P0 的框 y 905~1051。手牌中心 y 993~1001、高 138~159
#        -> 上緣 913.5~932,沒有被夾到;下緣 1062.5~1080.5,一定被夾到 1051
#        -> 夾完的高度 = 119 ~ 137.5
#      鳴牌 77~101px,整張都在框內,完全不會被夾。
#    101 跟 119 之間還有 18px,120 落在裡面:對真鳴牌有 19px 餘裕(絕不誤殺),
#    對手牌只有 1px。真的出現漏掉的手牌就把**推論端**降到 110
#    ((101+119)/2,兩邊各留 9px),不要動訓練端那個 120。
MAX_BOX_H = {
    "P0": 120,      # 鳴牌最高 101,手牌最低 138,取中間
    "P1": None,
    "P2": None,
    "P3": None,
}

# 裁切框裡面要**排除**的區域(x, y, w, h)。中心點落在這裡面的偵測丟掉。
# P3 的左上角是寶牌指示面板。v3.2 的 P3 框從 y=221 開始、面板已經在框外,
# 所以這條現在是**備援**。留著的理由:哪天把 P3 的框往上拉,陷阱會立刻回來,
# 而那是安靜發生的(舊專案實測過:面板那一格會被當成 P3 的鳴牌,類別還每次
# 都不一樣 —— P / 4m / 2m / 9m / F …)。
EXCLUDE_ZONES = {
    "P3": [(0, 0, 400, 205)],
}

# 這一版的前處理規格。改任何一項都要改這個字串 —— 它會跟權重旁邊那份
# crop_meta.json 核對,對不上就拒絕執行。
CROP_SPEC = "v3.2-meld-only-h120"

# 同一張牌被框兩次的判定距離(中心點,chebyshev,1920x1080 基準)。
# 量法:44 張標註圖、140 個框,算每張牌到最近鄰居的距離,取每家最小值 x0.66
#     座位   最近鄰距離(最小/中位)   DUP_DIST
#     P0        71.7 / 78.2            47
#     P1        27.5 / 32.6            18
#     P2        46.1 / 54.4            30
#     P3        51.0 / 63.0            34
# P1 特別小是因為那一家的牌沿著「往鏡頭方向」排,透視讓牌變窄。
# ★ 門檻一定要比「正常鄰牌的距離」小,不然會把隔壁那張真牌當成重複吃掉。
#   調它的時候唯一要盯的數字是**漏讀**:設太大就會從那裡冒出來。
#   實測 74 個重複框全部收掉、漏讀一個都沒增加,代表這組有餘裕。
# ★ 為什麼不能用 tiles.merge_overlaps():它是固定 20px,對這裡太小 ——
#   P0 正常鄰牌距離 71.7px,重複框卻常常差 30~40px,20px 收不掉。
DUP_DIST = {"P0": 47, "P1": 18, "P2": 30, "P3": 34}
DUP_DIST_DEFAULT = 18


def scaled_zone(seat, img_w=REF_W, img_h=REF_H):
    """把 MELD_ZONES 依影像尺寸等比例縮放 -> (x, y, w, h)。

    ⚠️ 等比例縮放是給**建資料集**用的(Roboflow 匯出混著 1920x1080 與
       2048x1152)。推論端不該用到 —— capture.py 保證給 1920x1080,
       而且 MeldModel 會擋住別的尺寸(見那邊 _check_frame 的說明)。
    """
    x, y, w, h = MELD_ZONES[seat]
    sx, sy = img_w / REF_W, img_h / REF_H
    return round(x * sx), round(y * sy), round(w * sx), round(h * sy)


def center_in_zone(cx, cy, zone):
    x, y, w, h = zone
    return x <= cx <= x + w and y <= cy <= y + h


def in_meld_zone(center, seat, img_w=REF_W, img_h=REF_H):
    """這個中心點是不是落在某一家的鳴牌裁切範圍裡。"""
    return center_in_zone(center[0], center[1],
                          scaled_zone(seat, img_w, img_h))


def seat_of_meld(center, img_w=REF_W, img_h=REF_H):
    """這個中心點落在哪一家的鳴牌區。不在任何一家回 None。

    ★ 四家的框**互不重疊**(v3.2 的性質;v2 的 P0 與 P3 在左下角有交集,
      v3.2 沒有),所以不會有一個點同時屬於兩家的情形。
    """
    for seat in SEATS:
        if in_meld_zone(center, seat, img_w, img_h):
            return seat
    return None


def zones_overlap(img_w=REF_W, img_h=REF_H):
    """四家的框有沒有互相重疊。回傳重疊的配對 list(正常應該是空的)。

    dedup() 只比同一家,前提就是四家互不重疊。哪天改座標改到重疊,
    同一張牌會出現在兩家結果裡而去重收不掉 —— 所以這件事要測得到。
    """
    bad = []
    zs = {s: scaled_zone(s, img_w, img_h) for s in SEATS}
    for i, a in enumerate(SEATS):
        for b in SEATS[i + 1:]:
            ax, ay, aw, ah = zs[a]
            bx, by, bw, bh = zs[b]
            if (ax < bx + bw and bx < ax + aw
                    and ay < by + bh and by < ay + ah):
                bad.append((a, b))
    return bad


def in_excluded(cx, cy, seat, img_w=REF_W, img_h=REF_H):
    """中心點是不是落在這一家要排除的 UI 區域裡(目前只有 P3 的寶牌面板)。"""
    sx, sy = img_w / REF_W, img_h / REF_H
    for x, y, w, h in EXCLUDE_ZONES.get(seat, ()):
        if (x * sx <= cx <= (x + w) * sx) and (y * sy <= cy <= (y + h) * sy):
            return True
    return False


def too_tall(box_h_px, seat, img_h=REF_H):
    """這個框是不是高到不可能是鳴牌(也就是:它其實是手牌)。

    門檻是 None(P1/P2/P3)就一律回 False —— 那三家只有鳴牌,不過濾。
    硬加一個門檻只會多一條會誤殺的規則。
    """
    limit = MAX_BOX_H.get(seat)
    if limit is None:
        return False
    return box_h_px >= limit * (img_h / REF_H)


def dedup(dets, dist=None):
    """同一張牌被框兩次 -> 留信心度高的那一個。

    ★ 依**座位**取門檻(見 DUP_DIST),而且**不分類別**比較 —— 同一張牌被
      框兩次的時候兩個框的類別常常不一樣(那正是要收掉的情況);照類別分開
      去重會把它們全部留下來,變成憑空多出一張牌。
    ★ 只比同一家(四家的框互不重疊,見 zones_overlap)。
    """
    kept = []
    for d in sorted(dets, key=lambda z: -z["conf"]):
        seat = d.get("seat")
        lim = dist if dist is not None else DUP_DIST.get(seat, DUP_DIST_DEFAULT)
        cx, cy = d["center"]
        if any(k.get("seat") == seat
               and abs(k["center"][0] - cx) < lim
               and abs(k["center"][1] - cy) < lim
               for k in kept):
            continue
        kept.append(d)
    return kept


def meta_dict():
    """建資料集時寫出去、推論時核對用的那幾個 key。"""
    return {
        "crop_spec": CROP_SPEC,
        "ref": [REF_W, REF_H],
        "max_zones": {s: list(MELD_ZONES[s]) for s in SEATS},
        "exclude_zones": {s: [list(z) for z in v]
                          for s, v in sorted(EXCLUDE_ZONES.items())},
        "max_box_h": {s: MAX_BOX_H.get(s) for s in SEATS},
        "mask": "none",
    }


def verify_meta(meta, where=""):
    """核對一份 crop_meta.json 跟這裡的定義是否一致。不一致就 raise。

    ★ 只比 meta_dict() 裡那幾個 key。crop_meta.json 還會多記 summary、seats
      這些「這份資料集裁出來長什麼樣」的統計,跟前處理規格無關,每跑一次都
      不一樣,不能拿來比。
      (舊專案的坑:原本寫 `if meta == mine: return True`,對真正的
       crop_meta.json **永遠不相等**,然後下面的 diff 迴圈又找不到任何差異
       -> 丟出一個**沒有內容**的錯誤訊息。呼叫端只好自己先濾 key 才呼叫得動,
       那等於把「哪些 key 算規格」抄到第二個地方,抄漏一個就安靜地驗不到。)
    """
    mine = meta_dict()
    diff = []
    for k in mine:
        if meta.get(k) != mine[k]:
            diff.append(f"    {k}:\n      規格檔 {meta.get(k)!r}\n"
                        f"      程式   {mine[k]!r}")
    if not diff:
        return True
    raise SystemExit(
        f"❌ 裁切規格對不上{(' (' + where + ')') if where else ''}。\n"
        f"   權重是用規格檔那一組裁切範圍訓練的,推論端用另一組等於餵它\n"
        f"   沒看過的輸入,而且不會報錯、只會讀不到牌。先弄清楚哪一邊才對:\n"
        + "\n".join(diff))


def load_meta(path, where=""):
    """讀一份 crop_meta.json 並核對 -> 回傳整個 dict(含 summary 等額外資訊)。

    核對只該有一個入口 —— 有兩個以上的呼叫端時,走同一個函式才不會漏驗。
    """
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    verify_meta(meta, where or path)
    return meta
