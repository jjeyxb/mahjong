"""P0 手牌區的版面常數 + 純工具。零相依 —— 不 import ultralytics、不碰 cv2。

★ 為什麼座標住在這裡而不是 hand_model.py:
  hand_model.py 模組層級就要 `from ultralytics import YOLO`,座標放那邊的話
  連 --selftest 都要先等 torch 載入十幾秒。probe_region.py 的開頭記過同一個坑。


=== 版面:實測值,不是猜的 ===
    牌距          **94.0 px**(2026-10-24 實測 3097 幀,最小=中位=最大,
                  完全不動 —— 版面是固定的,本來就該這樣)
    摸入牌框寬    約 105 px
    P0 摸入牌     cx 265~1532、cy 943~997、**x2 最大 1586**
                  (2195 筆。cx 群聚剛好每組鳴牌差 3x94px:
                   0組~1500 / 1組~1200 / 2組~960 / 3組~660 / 4組~240)

★ 從最後那行可以導出一件事:手牌固定範圍是 x 220..1830,所以 P0 的摸入牌
  **2195/2195 筆本來就在範圍內**,餘裕 244px。而且鳴牌越多手牌越短、摸入牌
  只會往左移,所以無鳴牌的 1586 就是最右極限 —— 右邊界永遠不是瓶頸。

★★ 教訓(踩過):不要用猜的數字當版面常數。
   舊版把牌距的保底值寫成 75(猜的),於是「最右一張離前一張超過牌距 1.35 倍
   就是摸入牌」這個判準的門檻變成 127,而實際間距湊不到 —— 結果摸入牌
   **0/3097 幀**被切出來。摸入牌一律用局勢模型給的座標認,不要靠間距猜。


=== ⚠️ 裁切寬度不要動(除非你讀過這段)===
MAX_ZONE_P0 跟訓練資料集的 MAX_ZONES["P0"] **完全一致**。手牌模型是用
「裁這塊固定範圍、把範圍裡的鳴牌區塗黑」的圖訓練的,所以推論時的前處理
如果不一樣,不是「尺度跟訓練時不同」那種程度的問題,而是**連輸入內容本身
都不一樣**。

scaled_zone() 留了 right= 可以改右邊界,但**預設不要用**。踩過的坑:
把右邊界動態設成摸入牌的 x2 會讓裁切寬逐幀變動 ->
    沒摸入牌         裁 1610 -> 餵 imgsz=640 縮放 0.40
    有摸入牌(1鳴牌)  裁 1093 -> 縮放 0.59
同一副手牌在這兩個尺度下讀出來的張數本來就不一樣。實機:
    f456  10張 5p,5p,7p,7p,7p,5sr,5s,7s,E,E   <- 0.40 讀的(漏了三張)
    f532  同一副手牌讀到 13 張                 <- 0.59 讀的
拿這兩份去做差分,結果沒有任何意義 —— 而且那種錯**不會以異常的形式浮出來**,
它會偽裝成合理的變化。要用動態右邊界的話,必須保證「要互相比較的那幾幀」
用的是同一個值。
"""

from collections import Counter

from tiles import base_pai, is_red

REF_W, REF_H = 1920, 1080

# 跟訓練資料集的 MAX_ZONES["P0"] 完全一致。見上面「裁切寬度不要動」。
MAX_ZONE_P0 = (220, 905, 1610, 170)
P0_RIGHT_FIXED = MAX_ZONE_P0[0] + MAX_ZONE_P0[2]      # 1830

# 手牌模型的信心度門檻。沿用 ultralytics 預設 0.25。
# ★ 不要用訓練時 val 集的信心度分布去猜門檻 —— 舊專案的教訓:牌河舊模型在
#   val 集看起來沒事,實機卻只有 0.1~0.4。要調就看實機的分布。
HAND_CONF = 0.25

# 相鄰兩張牌中心點的距離(實測)。
PITCH = 94.0

# 暗牌張數的合法值。n 組鳴牌 -> 13-3n = 13/10/7/4/1;吃碰之後還沒打出那一
# 瞬間會多兩張 -> 11/8/5/2。**不在這個集合裡就一定讀錯了**。
# ★ 2026-10-26 實機抓到的例子:
#       f1051  9張 3p, 3p, 3p, 8p, N, N, N, P, F  (牌河有新牌 8幀)
#   9 不是合法張數。而且那副裡還有三張 3p —— 那是剛剛**槓掉**的牌,
#   四張應該全部搬進鳴牌區並且被塗黑(見 hand_model.py 的前處理)。
#   張數不合法是最便宜的警報:不需要標準答案就知道這份讀數壞了。
LEGAL_CONCEALED = frozenset((1, 2, 4, 5, 7, 8, 10, 11, 13))

# --- 多幀投票 ---------------------------------------------------------------
# 視窗裡要湊滿幾張票才看得出結果,以及最高票要佔多少才算定案。
# 沿用第三步牌河的量級(CONFIRM_FRAMES=8 / SETTLED_SHARE=0.6)。
CONFIRM_VOTES = 8
SETTLED_SHARE = 0.6
# 一直湊不到門檻的保險絲:收滿這麼多張票就用最高票定案,並標記 shaky。
# ★ 一定要有 —— 不然動畫一直不停就永遠不定案,畫面上整局都沒有手牌。
MAX_VOTES = 30

SUITS = "mps"
HONOURS = ("E", "S", "W", "N", "P", "F", "C")


def sort_key(pai):
    """m -> p -> s -> 風牌/三元牌;同一個數字時赤寶牌排前面。

    認不出來的名字一律排到最後面,並且用字串當第二鍵 —— 排序絕對不可以
    因為模型多了一個沒見過的類別就整個爆掉。
    """
    b = base_pai(pai)
    red = 0 if is_red(pai) else 1
    if b in HONOURS:
        return (3, HONOURS.index(b), red, b)
    if len(b) == 2 and b[1] in SUITS and b[0].isdigit():
        return (SUITS.index(b[1]), int(b[0]), red, b)
    return (9, 0, 0, str(pai))


def sort_hand(pais):
    return sorted(pais, key=sort_key)


def hand_str(pais):
    """`6m, 6m, 8m, 5p, ...`"""
    return ", ".join(sort_hand(pais))


def scaled_zone(img_w=REF_W, img_h=REF_H, right=None):
    """MAX_ZONE_P0 依畫面尺寸等比例縮放,回傳 (x, y, w, h)。

    right 給了就當成新的右邊界(整張畫面座標、縮放前)。
    ⚠️ 預設不要給 —— 理由見模組說明最後一段。
    """
    x, y, w, h = MAX_ZONE_P0
    if right is not None:
        w = max(1, min(int(right), P0_RIGHT_FIXED) - x)
    sx, sy = img_w / REF_W, img_h / REF_H
    return round(x * sx), round(y * sy), round(w * sx), round(h * sy)


def in_hand_zone(center, img_w=REF_W, img_h=REF_H):
    """這個中心點是不是落在 P0 手牌的固定範圍裡。"""
    zx, zy, zw, zh = scaled_zone(img_w, img_h)
    cx, cy = center
    return zx <= cx <= zx + zw and zy <= cy <= zy + zh


def pick_p0_drawn(dets, img_w=REF_W, img_h=REF_H):
    """從局勢模型的偵測裡挑出 P0 的摸入牌。沒有就回 None。

    判準是**中心點落在 P0 手牌固定範圍裡** —— 跟第三步的牌河座位歸屬同一個
    原則:純幾何,不看是哪塊裁圖讀到它、也不先看信心度。同一幀有多個符合的
    才取信心度最高的那個。
    (P1~P3 的摸入牌在畫面上離這塊很遠,不會誤收:實測 2195 筆 P0 摸入牌的
     cy 全部落在 943~997,而這塊範圍的 y 是 905~1075。)
    """
    best = None
    for d in dets:
        if d.get("cls") != "drawn_tile":
            continue
        if not in_hand_zone(d["center"], img_w, img_h):
            continue
        if best is None or d["conf"] > best["conf"]:
            best = d
    return best


def match_drawn(hand_dets, drawn, max_dist=PITCH / 2):
    """手牌模型的哪一筆偵測就是那張摸入牌。對不上回 None。

    門檻用半個牌距(47px):比這更遠的一定是隔壁那張,寧可回 None 也不要
    對錯 —— 對錯會同時害到牌名跟底下的「暗牌有哪些」。
    """
    if drawn is None or not hand_dets:
        return None
    dcx, dcy = drawn["center"]
    best, bd = None, None
    for d in hand_dets:
        cx, cy = d["center"]
        gap = abs(cx - dcx) + abs(cy - dcy)
        if bd is None or gap < bd:
            best, bd = d, gap
    return best if (best is not None and bd <= max_dist) else None


def concealed_of(hand_dets, drawn, max_dist=PITCH / 2):
    """手牌模型的偵測扣掉摸入牌那一張 = 暗牌。

    ★ 為什麼要扣:手牌模型是對整條手牌區推論的,摸入牌就在那一條裡面,
      所以它會**一起被讀到**。拿沒扣過的去投票,「手牌」會時多時少一張,
      取決於那一幀摸入牌在不在 —— 同一副手牌投出兩種答案。
      扣掉之後,不管什麼時候投票,得到的都是同一個東西:暗牌。
    """
    hit = match_drawn(hand_dets, drawn, max_dist)
    if hit is None:
        return list(hand_dets)
    return [d for d in hand_dets if d is not hit]


def name_of_drawn(hand_dets, drawn, max_dist=PITCH / 2):
    """摸入牌是哪一張牌。沒把握就回 None。

    ★★ 為什麼要有這個函式 ★★
    兩顆模型的偵測**鍵不一樣**,這件事踩過一次(2026-10-26 實機 KeyError):
        局勢模型(region_model)  cls / conf / box / center / clipped
        手牌模型(hand_model)    pai / base / conf / box / center / seat / model
    「哪裡有一張摸入牌」是局勢模型回答的,它**沒有 pai** —— 它只知道那裡有
    一張剛摸進來的牌,不知道是什麼牌。牌名只有手牌模型有。
    所以要牌名就得把兩邊對起來:拿手牌模型裡**中心點最接近**那張摸入牌的,
    就是它。

    門檻用半個牌距(47px):比這更遠的一定是隔壁那張,寧可回 None 也不要
    標錯 —— 畫面上標錯牌名比不標更糟。
    """
    hit = match_drawn(hand_dets, drawn, max_dist)
    return None if hit is None else hit.get("pai")


TSUMOGIRI = "摸切"
TEDASHI = "手切"


def diff(a, b):
    """多重集合 a - b,回傳排序好的 list。"""
    c = Counter(a)
    c.subtract(Counter(b))
    return sort_hand([p for p, k in c.items() if k > 0 for _ in range(k)])


def judge_dahai(prev, cur, naki=None, drawn=None, river=None):
    """打牌之後比對前後兩份定案的暗牌 -> (結論, 打出的是哪一張, 說明)。

    判不出來回 (None, None, 原因)。

    使用者 2026-10-26 定的規則:
        ・P0 發起鳴牌(吃/碰/槓)之後,手牌總數有變化 -> 手切
        ・沒鳴牌:這次打牌後的手牌跟上次不一樣 -> 手切,一樣 -> 摸切
        ・★ 摸切判斷完成了但沒辨識到摸入牌 -> 看牌河那張新的牌是什麼

    === ★ 為什麼主判準用「內容」而不是「張數」 ===
    「內容有沒有變」這一條**五種情況全對**,張數那一條只對一半:

        情況                暗牌張數      內容      正確答案   張數判   內容判
        沒鳴牌,摸切        13 -> 13      一樣      摸切       ✓       ✓
        沒鳴牌,手切        13 -> 13      換一張    手切       ✗       ✓
        吃/碰後那一打      11 -> 10      變了      手切       ✓       ✓
        槓後摸切           10 -> 10      一樣      摸切       ✓       ✓
        槓後手切           10 -> 10      換一張    手切       ✗       ✓

    ★★ 槓跟吃/碰**不一樣**,不可以混在一起:吃/碰那一巡沒摸牌,所以那一打
       必然手切;**槓會補摸嶺上牌**,之後那一打摸切手切都可能。所以這裡只
       對吃/碰套用「必然手切」的捷徑,槓一律走內容比對。
       (舊專案第四步記過同一件事:「槓相反(補摸嶺上牌)」。)

    === ★★ 打出的是哪一張:牌河才是權威 ★★ ===
    牌河那一格**就是**被打出去的那張牌 —— 這是定義,不是推論。相對地:
        摸切時用「摸入牌」當答案,是**推論**(假設打的就是剛摸的那張)
        手切時用「手上少掉的那張」當答案,也是**推論**(假設手牌讀得對)
    所以優先用牌河。牌河讀不到才退回推論值。

    兩邊都有而且**不一樣**的時候一定要講出來:那代表兩顆**獨立**的模型
    (牌河模型 vs 手牌模型)有一顆讀錯了。它們看的是畫面上完全不同的兩塊
    區域,同時錯成一樣的機率極低,所以「兩邊一致」是很強的佐證,
    「兩邊不一致」則是免費抓到的錯 —— 不需要標準答案。
    """
    if prev is None or cur is None:
        return None, None, "還沒有可以比的前一份手牌"
    out = diff(prev, cur)       # 少掉的
    inn = diff(cur, prev)       # 多出來的
    notes = []

    # ★★ 一次打牌不可能動這麼多張 -> 這兩份手牌不是同一局的 ★★
    # 單一次動作最多動幾張:
    #     手切        少 1、多 1
    #     吃/碰       少 2(鳴牌那次定案沒抓到的話,連打牌一起 = 少 3)
    #     槓          少 3~4(連打牌 = 少 5)
    # 所以 5 張以內都還可能是「漏抓了中間一次定案」。超過就一定是換局 ——
    # 配牌會整手換掉(十三出、十三進)。
    # ★ 為什麼需要這條:程式**沒有**換局偵測(牌河讀到空只代表被擋住,
    #   不是換局,見 5_hand.py 那段說明)。所以新局第一次打牌時,拿來比的
    #   會是**上一局**的手牌。沒有這條的話那次會吐出一個看起來很正常、
    #   實際上毫無意義的「手切」。寧可明講判不出來。
    if len(out) >= 6 or len(inn) >= 6:
        return None, None, (f"整手換掉({len(out)}出/{len(inn)}進)—— "
                            f"這兩份手牌不是同一局的,不判")

    # ★★ 打牌只會讓手牌變少或不變,**不可能變多** ★★
    # 2026-10-26 實機:
    #     f1051  P0 打出 3s 手切  (7張 -> 9張,少了 -、多了 8p, P)
    # 打完牌手上多兩張是物理上不可能的,代表前後兩份讀數至少有一份是錯的
    # (很可能是前一份漏讀了那兩張,或是這一份把鳴牌區的牌讀進來了)。
    # 這種時候**不可以**硬吐一個結論 —— 那會變成一筆看起來正常、實際上
    # 毫無根據的手切。寧可明講判不出來,而且把兩邊的張數寫出來方便查。
    # ★ 這一條也順便蓋掉「手牌變了卻沒有任何一張離開」:out 是空的就代表
    #   prev 的每一張都還在 cur 裡,那 len(cur) >= len(prev);再加上
    #   prev != cur,就一定是 len(cur) > len(prev)。所以不需要另外寫一條。
    if len(cur) > len(prev):
        return None, None, (f"打牌之後手牌反而變多({len(prev)}張 -> "
                            f"{len(cur)}張,多了 {', '.join(inn)}) —— "
                            f"不可能,兩份讀數至少有一份是錯的,不判")

    bad = [n for n in (len(prev), len(cur)) if n not in LEGAL_CONCEALED]
    if bad:
        notes_bad = (f"★{'、'.join(str(n) for n in bad)} 張不是合法的暗牌"
                     f"張數(只能是 13/11/10/8/7/5/4/2/1)")
    else:
        notes_bad = None


    # --- 先定結論 ---------------------------------------------------
    if naki in ("chi", "pon"):
        zh = "吃" if naki == "chi" else "碰"
        verdict = TEDASHI
        notes.append(f"{zh}之後那一巡沒有摸牌,必然手切")
        if len(prev) == len(cur):
            notes.append(f"★但張數沒變({len(prev)}張)—— 鳴牌應該會讓手牌變少")
        guess = out[0] if len(out) == 1 else None
    elif prev == cur:
        verdict = TSUMOGIRI
        notes.append(f"{len(cur)}張沒變、內容也一樣")
        guess = drawn if drawn and drawn != "?" else None
    else:
        verdict = TEDASHI
        if len(out) == 1 and not inn:
            notes.append(f"{len(prev)}張 -> {len(cur)}張,少了 {out[0]}")
        elif len(out) == 1 and len(inn) == 1:
            n = f"少了 {out[0]}、多了 {inn[0]}"
            if base_pai(out[0]) == base_pai(inn[0]):
                # 5s/5sr 互相讀錯只差一飜,跟「9p 讀成 6p」完全不同等級。
                n += "(★只差赤寶牌,很可能是同一張讀成兩種)"
            notes.append(f"{len(prev)}張 -> {len(cur)}張,{n}")
        else:
            notes.append(f"{len(prev)}張 -> {len(cur)}張,"
                         f"少了 {', '.join(out) or '-'}、"
                         f"多了 {', '.join(inn) or '-'}")
        guess = out[0] if len(out) == 1 else None

    # --- 再決定「打出的是哪一張」-------------------------------------
    if notes_bad:
        notes.append(notes_bad)

    if river:
        pai = river
        if guess and base_pai(guess) != base_pai(river):
            src = "摸入" if verdict == TSUMOGIRI else "手上少的"
            notes.append(f"★牌河說 {river}、{src}說 {guess}"
                         f" —— 兩顆模型有一顆讀錯了")
        elif not guess:
            # ★ 措辭要分開。手切分支的牌名來自「手上少掉哪一張」,
            #   根本不會用到摸入牌 —— 在那裡講「沒辨識到摸入牌」是誤導的
            #   (2026-10-26 使用者就是被這句話誤導去查摸入牌偵測)。
            notes.append("(沒辨識到摸入牌,牌名取自牌河)"
                         if verdict == TSUMOGIRI
                         else "(手上沒有剛好少一張,牌名取自牌河)")
    else:
        pai = guess
        if pai:
            notes.append("(牌河還沒定案,牌名是推論的)")
        else:
            notes.append("★牌河沒定案、也沒辨識到摸入牌 —— 不知道打了哪一張")
    return verdict, pai, ";".join(notes)


class HandSlots:
    """手牌**逐格**投票:每一個位置各自投自己的票,互不影響。

    使用者 2026-10-26:「投票判斷是對手牌中的每一張單獨進行,避免牌面辨識
    錯誤。」

    === ★★ 為什麼非得逐格,不能對整副投 ★★ ===
    整副投票是拿「十三張排成的那個 tuple」當一張選票。只要**任何一張**讀錯,
    整張選票就跟別人不一樣 —— 逐張的錯誤率被放大十三倍:
        逐張 99% 正確 -> 整副一次就對的機率只有 0.99^13 = 88%
        逐張 95%      -> 0.95^13 = 51%,一半的票都是廢票
    票一散,最高票就湊不到門檻,於是要嘛拖很久才定案、要嘛靠保險絲硬定一個
    可能是錯的。而且**錯一張就整副作廢**,那十二張讀對的資訊全部被丟掉。

    逐格投票把它拆開:位置 7 讀錯不影響位置 1~6、8~13。每一格各自用自己的
    多數決,一張讀錯只會讓那一格的票數難看,不會害到別人。

    === 位置怎麼配對 ===
    跟第三步牌河的格子同一套:中心點距離小於半個牌距(47px)就算同一格,
    位置用移動平均(偵測框每幀會抖幾 px,不平均的話會越飄越遠)。

    ★ 但有一個牌河沒有的限制:**手牌會理牌**,整排位置會重新洗。所以逐格
      追蹤只在**一個投票窗之內**有效 —— 窗一開就把格子全部清掉。這也正是
      reopen() 要做的事。窗內手牌是靜止的,位置才穩得住。

    === 張數怎麼決定 ===
    「這一格在這個窗裡出現過幾幀」除以「窗裡總共幾幀」要過 share 才算數。
    偶爾被手擋住一兩幀的格子留得下來;雜訊冒出來的一兩幀格子會被濾掉。
    """

    def __init__(self, votes=CONFIRM_VOTES, share=SETTLED_SHARE,
                 cap=MAX_VOTES, dist=PITCH / 2):
        self.votes_needed = votes
        self.share = share
        self.cap = cap
        self.dist = dist
        self.slots = []         # [{"cx", "names": Counter, "n"}]
        self.frames = 0         # 這一輪餵了幾幀
        self.answer = None
        self.answer_share = 0.0     # **最弱那一格**的票數佔比(木桶效應)
        self.answer_frame = None
        self.weakest = None     # (牌名, 佔比) —— 最沒把握的那一格
        self.shaky = False
        self.counting = True
        self.why = "開場"

    def feed(self, dets, frame=None):
        """吃這一幀手牌模型的偵測(已經扣掉摸入牌)。回傳 True = 剛剛定案。"""
        if not self.counting or not dets:
            return False
        self.frames += 1
        used = set()
        for d in sorted(dets, key=lambda z: z["center"][0]):
            cx = d["center"][0]
            best, bd = None, None
            for i, sl in enumerate(self.slots):
                if i in used:
                    continue
                gap = abs(sl["cx"] - cx)
                if gap <= self.dist and (bd is None or gap < bd):
                    best, bd = i, gap
            if best is None:
                self.slots.append({"cx": float(cx),
                                   "names": Counter([d["pai"]]), "n": 1})
                used.add(len(self.slots) - 1)
            else:
                sl = self.slots[best]
                sl["names"][d["pai"]] += 1
                sl["n"] += 1
                sl["cx"] += (cx - sl["cx"]) / sl["n"]      # 移動平均
                used.add(best)

        if self.frames < self.votes_needed:
            return False
        real = [sl for sl in self.slots
                if sl["n"] >= self.share * self.frames]
        if not real:
            return False
        names, worst, weak = [], 1.0, None
        for sl in sorted(real, key=lambda z: z["cx"]):
            top, c = sl["names"].most_common(1)[0]
            names.append(top)
            sh = c / sl["n"]
            if sh < worst:
                worst, weak = sh, (top, sh)
        # ★ 看**最弱**那一格,不是平均 —— 十二張很穩、一張在 50/50 跳,
        #   平均起來還是很漂亮,但那一張就是會讀錯。木桶效應。
        if worst >= self.share or self.frames >= self.cap:
            self.answer = tuple(sort_hand(names))
            self.answer_share = worst
            self.answer_frame = frame
            self.weakest = weak
            self.shaky = worst < self.share
            self.counting = False
            return True
        return False

    def reopen(self, frame=None, why=""):
        """解凍,重新投一輪。★ 格子全部清掉 —— 手牌理過牌,位置會重新洗。"""
        self.slots = []
        self.frames = 0
        self.counting = True
        self.why = why

    def reset(self):
        self.__init__(self.votes_needed, self.share, self.cap, self.dist)


class Vote:
    """多幀投票確認一個值,**確認完就凍結**,直到 reopen() 被叫才重新投票。

    值是什麼都可以,只要能 hash。目前兩個用途:
        ・整副暗牌(tuple)—— 牌河有新牌 / P0 鳴牌 / 換局時重新投
        ・摸入牌的牌名(str)—— 摸入牌出現時投一輪,消失時重置

    使用者 2026-10-26:「一樣透過之前的多幀投票對手牌進行確認工作,直到下一次
    的摸打牌事件或 P0 發起鳴牌事件前,都不要再對 P0 手牌進行更新。」

    凍結的理由很實際:那個值**只有**在特定事件之後才會變。其他時間重複辨識
    只會讓畫面上的東西因為模型抖動而跳來跳去,而那些跳動**全部都是假的**。

    === ★★ 投票視窗用「最近 N 張票」,不是「最前面 N 張票」★★ ===
    這是跟第三步牌河**刻意不一樣**的地方,不要改回去:
        第三步  一張牌進牌河之後就靜止不動 -> 最前面 8 幀就是最乾淨的
        這裡    reopen() 的時機正好是**理牌/鳴牌動畫剛開始**的時候
                -> 最前面幾幀是最髒的
    所以這裡用滑動視窗:只看最近 CONFIRM_VOTES 張票,動畫那幾幀會自己滑出去。
    最高票佔不到 SETTLED_SHARE 就繼續收,等於自動等動畫結束。
    (舊專案記過同一個坑的另一面:「所有票都投在打牌之後」。這裡 reopen()
     的時機就保證了這件事 —— 所有票一定投在摸入牌消失之後。)

    收滿 MAX_VOTES 還湊不到門檻就用最高票定案,並且標記 shaky。
    ★ 一定要有這條保險絲 —— 寧可給一份標記成可疑的答案,也不要整局都沒有。
    """

    def __init__(self, votes=CONFIRM_VOTES, share=SETTLED_SHARE,
                 cap=MAX_VOTES):
        self.votes_needed = votes
        self.share = share
        self.cap = cap
        self.ballots = []       # 滑動視窗,最多 votes_needed 張
        self.total = 0          # 這一輪總共收了幾張票
        self.answer = None      # 定案的手牌(tuple)
        self.answer_share = 0.0
        self.answer_frame = None
        self.shaky = False      # 是靠保險絲定案的
        self.counting = True    # True = 正在投票,False = 凍結
        self.why = "開場"       # 這一輪是被什麼觸發的

    def feed(self, tiles, frame=None):
        """吃這一幀讀到的手牌(tuple)。回傳 True = 剛剛定案。

        tiles 給 None 代表這一幀沒讀到東西 —— 不投票,也不中斷這一輪。
        凍結中就整個忽略,這正是「不要再更新」的意思。
        """
        if not self.counting or tiles is None:
            return False
        self.ballots.append(tiles)
        self.total += 1
        if len(self.ballots) > self.votes_needed:
            self.ballots.pop(0)
        if len(self.ballots) < self.votes_needed:
            return False
        top, n = Counter(self.ballots).most_common(1)[0]
        share = n / len(self.ballots)
        if share >= self.share or self.total >= self.cap:
            self.answer = top
            self.answer_share = share
            self.answer_frame = frame
            self.shaky = share < self.share
            self.counting = False
            return True
        return False

    def reopen(self, frame=None, why=""):
        """解凍,重新投一輪。只有摸打牌事件結束 / P0 鳴牌才可以叫。"""
        self.ballots = []
        self.total = 0
        self.counting = True
        self.why = why

    def reset(self):
        self.__init__(self.votes_needed, self.share, self.cap)


class Hold:
    """去抖動的開關:連續看到 on 幀才開,連續沒看到 off 幀才關。

    ★ 為什麼不直接看「這一幀有沒有」:
      ・偵測會閃。單幀的假偵測會讓畫面上的提示字閃一下就消失,看起來像壞的。
      ・反過來,真的摸入牌偶爾掉一幀,提示字就會斷一下。
      第三步實測過:**雜訊最長撐 3 幀**連續,真牌至少連續 13 幀,中間
      4~12 幀是空的。所以「連續 2 幀才開、連續 3 幀才關」剛好落在那個空隙,
      擋掉所有已知長度的雜訊,又不會讓真的摸入牌斷掉。
    """

    def __init__(self, on=2, off=3):
        self.on = on
        self.off = off
        self.state = False
        self.hit = 0
        self.miss = 0
        self.since = None       # 目前這個狀態是第幾幀開始的

    def feed(self, seen, frame=None):
        """吃這一幀有沒有看到。回傳 (現在要不要亮, 剛剛有沒有換狀態)。"""
        was = self.state
        if seen:
            self.miss = 0
            self.hit += 1
            if self.hit >= self.on:
                self.state = True
        else:
            self.hit = 0
            self.miss += 1
            if self.miss >= self.off:
                self.state = False
        changed = self.state != was
        if changed:
            self.since = frame
        return self.state, changed

    def reset(self):
        self.state = False
        self.hit = 0
        self.miss = 0
        self.since = None
