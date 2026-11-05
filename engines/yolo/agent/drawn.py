"""摸入牌(drawn_tile)的座標表與摸切/手切判斷。純資料 + 純邏輯,零模型相依。

=== 判斷的根據 ===
摸切 = 把剛摸進來的那張打出去 -> 摸入牌的位置**空了**
手切 = 從手裡拿一張打出去   -> 摸入牌**還在**原位(之後才滑進手牌)

★ 兩種情況「最後」都會變空(手切的理牌動作做完也空了),所以差別只在**時機**:
  要在「牌河剛出現新牌」的那一刻看。看晚了,手切的理牌動作做完,兩種就一樣了。
  這就是使用者 2026-10-24 指定「判斷點取牌河出現新牌的第一幀」的原因。

=== 座標 ===
使用者 2026-10-24 量的「摸入牌最大可能區域」。要蓋住鳴滿四組的情況 ——
每鳴一組暗牌少 3 張,摸入牌會跟著往手牌起點的方向滑。

    P1  (1571, 171, 219, 567)    x 1571-1790   y 171-738
    P2  ( 655,   5, 616,  65)    x  655-1271   y   5- 70
    P3  ( 150, 132, 210, 545)    x  150- 360   y 132-677

三框互不重疊(算過:兩兩的 x 完全沒有交集)。

★ 為什麼框可以畫得這麼寬鬆:drawn_tile 是**模型自己的一個類別**,所以框只要
  回答「這是哪一家的」,不用回答「這是不是摸入牌」。跟牌河框完全不同 ——
  牌河框是裁切範圍,畫錯會把牌切掉;這裡的框只是查表。

★ P0 不需要框:使用者指定 P0 不看、不輸出(P0 的摸切之後用別的方式判斷)。
  而且三框最大只到 y=738,舊專案量到 P0 的 drawn_tile 在 y≈985 ——
  P0 自然落在所有框外面,會被當成 None 忽略。

⚠️ 已知的風險:P3 的框上緣 y 132-205 跟**寶牌指示面板**重疊。舊專案的 P3
   鳴牌區就是因為寶牌面板被誤認而加了排除區 (0,0,400,205),而寶牌是正面牌、
   跟摸入牌同一類外觀。所以 4_tsumogiri.py 的報告會把 P3 落在 y<205 的偵測
   單獨列出來 —— 如果那一帶一直有偵測,就要加排除區。
"""

# (x, y, w, h),1920x1080 基準。P0 刻意不建表,見模組說明。
DRAWN_ZONES = {
    "P1": (1571, 171, 219, 567),
    "P2": (655, 5, 616, 65),
    "P3": (150, 132, 210, 545),
}

# 要判斷摸切的三家。P0 不在裡面 —— 使用者指定跳過。
WATCH_SEATS = ("P1", "P2", "P3")

# 舊專案量到 P0 的 drawn_tile 在 y≈985,其他三家最近的在 y≈629。
# 這個值只用在報告裡把「框外的偵測」分成「大概是 P0」跟「真的不知道」。
P0_DRAWN_MIN_Y = 900

# 寶牌指示面板的範圍(舊專案 P3 鳴牌區的排除區)。只用在報告的警告。
DORA_PANEL = (0, 0, 400, 205)

# arm 這個類別的門檻。舊專案的實測註記:「arm 這個類別信心度天生偏低
# (門檻要放到 0.15),漏偵測是常態」。這裡用它只是為了把「分不出來」標出來,
# 所以寧可寬鬆 —— 漏偵測的代價是少標一個不明,誤偵測的代價也只是多一個不明。
ARM_CONF = 0.15

# ★★ drawn_tile 自己的信心度門檻(跟局勢模型的一般門檻 0.25 分開)。
#
# 2026-10-24 的證據:整整一巡,摸入牌只被偵測到 1 幀(seen=YES(1f))。
# 如果模型對那張牌是穩穩看得見的,應該是幾十幀 —— 只有 1 幀代表它的分數
# **就卡在 0.25 上下抖**。同一輪還看到 seen=0f(完全沒越過)跟 seen=2f。
#
# 代價不對稱,所以往低的調:
#     漏報 -> 摸切/手切直接判錯,而且畫面上看不出來
#     誤報 -> 幾乎無害:第 2 條「取得」一定要中心點落在該家的框裡,
#             框很緊,框外的雜訊根本進不來
# 跟牌河模型 0.25 -> 0.10 是同一個道理。
#
# ⚠️ 這個值 2026-10-24 降過一次又被 revert —— 但那時還有「隨時都在找移位、
#    而且不要求在框內」的追蹤器,0.10 的雜訊被它抓進來,正常判斷才壞掉。
#    那套已經拿掉了(見 DrawnWatch.observe 的三條路),現在重試。
#    真的又壞掉的話:--drawn-conf 0.25 就回到舊行為,不用改程式。
DRAWN_CONF = 0.10

# ★★ 判斷點之後再多看幾幀,才決定「摸入牌還在不在」。
#
# 2026-10-24 實測 f615:摸入牌明明還在原位(人眼看得到,存的座標也對),
# 但偵測**剛好閃掉判斷點那一幀**(last_seen 是前一幀)-> 判成摸切。
# 單幀取樣碰上會閃的信號就是在賭運氣。
#
# 為什麼只能往後看,不能往前看:
#     摸切   ...摸入牌一直在...  [打出去] 之後都不會再出現
#     手切   ...摸入牌一直在...  [牌落地] 還在原位,要等理牌才滑進手牌
# 往前看的話兩種都「有」,分不開。往後看才分得開:摸切之後是空的,
# 手切之後還會再看到幾幀。
#
# 不用付延遲:輸出本來就要等那一格定案(CONFIRM_FRAMES=8 幀),窗口比它短。
# 窗口不能太長 —— 手切的理牌動畫做完摸入牌也會消失,太長就兩種都變成
# 「有看到」。2 幀 @9fps ≈ 220ms,遠短於理牌。
DRAWN_WINDOW = 2

# ============================================================================
# ★★ 摸入牌:框負責「認出來」,座標負責「還在不在」(使用者 2026-10-24 指定)
# ============================================================================
#   認出來  中心點落在該家的最大可能區域 -> 這是**誰的**摸入牌,並把座標存下來
#   還在嗎  判斷點那一幀,**存下來的那個座標**上還有沒有 drawn_tile
#
# 跟「框裡有沒有牌」的差別:框是固定的,座標是那一張牌自己的。摸入牌只要
# 認出來過一次,之後就算排版把它推到框邊,判斷的也還是它本人。

# 「還在原位」的容忍距離。摸入牌是靜止的,只有偵測框會抖幾 px。
# ★ 一定要小於一張牌的寬度(實測河牌 52~80px),不然會配到隔壁那一張,
#   變成「摸入牌還在」—— 摸切就會全部讀成手切。
DRAWN_AT_DIST = 40

# 暗槓之後重新取得座標時,往外找多遠。
# 暗槓把 4 張從暗牌列拿走,暗牌列縮短大約三個牌寬,所以要夠大;
# 但不能大到跨過桌子搶別處的偵測。
ANKAN_REACQ_DIST = 320


# ★★ 立直宣告動畫「擋不擋判斷」用的門檻 —— 跟「這是不是一次立直事件」的
#    門檻(event_zones.ACT_CONF["reach"] = 0.70)是**兩回事**,刻意分開。
#    事件門檻要高:誤判一次立直,整條 mjai 事件流就錯了。
#    遮擋門檻要低:兩種錯的代價差很多 ——
#        誤判成「還在擋」  -> 多等幾幀,沒有損失
#        漏判成「沒在擋」  -> 在動畫還蓋著的時候判斷,摸切/手切直接讀錯
#    而且橫幅滑進滑出的那幾幀分數天生就低,那正是最需要擋住的時候。
REACH_ANIM_CONF = 0.25


# 吃/碰之後那一巡**沒有摸牌動作**,所以不會有 drawn_tile。
# 不特別處理的話「摸入牌不在」會被讀成摸切,而且每次鳴牌都錯。
# ★ 槓不算:槓會補摸嶺上牌,補摸之後還在自己的回合,所以槓之後有摸入牌。
NO_DRAW_AFTER = ("chi", "pon")

# 判斷結果
TSUMOGIRI = "摸切"
TEDASHI = "手切"
# ★ judge() 自 2026-10-24 起不再回報不明(使用者決定,理由見 judge 的說明)。
#   常數留著:報告還會列這一欄(永遠是 0 就代表規則照預期在走),而且
#   「其實不知道」這件事還在 dahai.csv 的 drawn_seen 欄位裡看得到。
UNKNOWN = "不明"


def in_zone(center, zone):
    x, y, w, h = zone
    return x <= center[0] <= x + w and y <= center[1] <= y + h


def seat_of_drawn(center, zones=None):
    """摸入牌的中心點落在哪一家的區域。不在任何區域回傳 None。

    跟 river.seat_of_center 同一個判準(中心點查表),但這裡的三框互不重疊,
    所以不需要處理「同時落在兩家」。真的重疊了就是座標量錯,應該要發現,
    而不是自動挑一家 —— 所以這裡刻意回傳第一個命中的,並由呼叫端的報告
    檢查重疊。
    """
    zones = zones or DRAWN_ZONES
    for seat in sorted(zones):
        if in_zone(center, zones[seat]):
            return seat
    return None


def boxes_overlap(box, zone):
    """框 (x1,y1,x2,y2) 跟區域 (x,y,w,h) 有沒有交集。"""
    x1, y1, x2, y2 = box
    zx, zy, zw, zh = zone
    return not (x2 < zx or x1 > zx + zw or y2 < zy or y1 > zy + zh)


def arm_blocking(arm_dets, seat, zones=None, conf=ARM_CONF):
    """這一家的摸入牌區域,現在有沒有被手臂蓋住。

    ★★ 這是**判斷點的條件**(使用者 2026-10-24 指定):牌河辨識到新牌、
       而且這一家的摸入牌區域沒有手臂,才在那一幀判斷摸入牌在不在。
       手臂動畫會蓋住摸入牌,蓋著的時候讀到的「不在」是假的。

    ★ 刻意**不問**「這是誰的手臂」。arm 這個類別沒有座位資訊(舊專案的紀錄:
      只能歸給當下行動者,而那會繞回指示燈,也就是繞回一個落後指標)。
      要問的是「這一家的摸入牌被擋住了嗎」—— 那只要看手臂的框跟這一家的
      摸入牌區域有沒有交集,不需要知道那是誰的手。
    """
    zones = zones or DRAWN_ZONES
    z = zones.get(seat)
    if z is None:
        return False
    return any(d["conf"] >= conf and boxes_overlap(d["box"], z)
               for d in arm_dets)


def judge(drawn_now, drawn_seen, forced_tedashi):
    """摸切還是手切。回傳 (結論, 理由)。

    drawn_now       判斷點那一幀,這一家的摸入牌在不在
    drawn_seen      這一巡**曾經**看到過這一家的摸入牌嗎
    forced_tedashi  這一打是不是緊接在自己吃/碰之後

    判斷點由呼叫端決定:牌河出現新牌、而且該家沒有手臂也沒有立直宣告動畫
    的那一幀。

    === ★★ 不回報「不明」(使用者 2026-10-24 決定)===
    原本「這一巡從來沒看到過摸入牌」會回報不明。改成**照當下有沒有看到**
    直接判:沒看到就是摸切。理由是使用者要盡量不要有不明。

    這是一個**知道代價的選擇**,不是忘了:
      ・好處:mjai 的 tsumogiri 是必填布林,本來就沒有「不知道」可以填;
              不明到了那一層還是得猜,不如在這裡就給一個答案。
      ・代價:如果模型對 P1/P2/P3 的橘色牌背召回率很差,「摸入牌不在」會
              變成每一打都成立 -> 每一打都判成摸切 -> **而且畫面上、輸出上
              都看不出異常**。不明本來就是在擋這個。
      ・所以代價要留在**報告裡**,不是消失:drawn_seen 照樣逐筆記進
        dahai.csv,報告第 2 節會印「沒有摸入牌紀錄就判的有幾筆」跟摸切率。
        摸切率逼近 100% 就是這個病,看報告抓。
    (舊專案同一個選擇:DEFAULT_TSUMOGIRI = True,並且把猜過幾次數出來。)

    兩種錯的代價不對稱,所以沒看到的時候倒向摸切:
        標成手切(實際摸切) -> 給 Mortal 一個假的「那家手牌動了」強訊號
        標成摸切(實際手切) -> 只是一個假的弱訊號
    """
    if forced_tedashi:
        return TEDASHI, "吃/碰之後那一巡沒有摸牌,必然手切"
    if drawn_now:
        return TEDASHI, "判斷點摸入牌還在"
    if not drawn_seen:
        return TSUMOGIRI, "這一巡沒有摸入牌紀錄,依當下沒看到判摸切"
    return TSUMOGIRI, "判斷點摸入牌已經不在"


class DrawnWatch:
    """逐家追蹤「摸入牌現在在不在」跟「這一巡看過沒」。

    ★ 狀態只有兩個,而且清除的時機是關鍵:
        drawn_now[seat]   這一幀的偵測結果,每幀重算
        seen[seat]        這一巡看過摸入牌沒 —— **在判斷點被取用之後才清掉**

      為什麼不在「牌河出現新牌」的那一刻就清:新格子有可能是雜訊。那一刻就
      清掉的話,真正那一打會因為 seen=False 被判成不明。等取用(= 那一格定案)
      才清,雜訊格子最多讓下一打誤報一次**不明** —— 不明是看得見的,
      假摸切是看不見的。
    """

    def __init__(self, seats=WATCH_SEATS):
        self.seats = tuple(seats)
        self.drawn_now = {s: False for s in self.seats}
        self.seen = {s: False for s in self.seats}
        self.frames = {s: 0 for s in self.seats}     # 這一巡看到幾幀(報告用)
        self.last_frame = {s: None for s in self.seats}
        self.forced = {s: False for s in self.seats}     # 吃/碰之後
        self.riichi = {s: False for s in self.seats}     # 立直了(整局)
        # ★ 下一打是不是「立直宣言牌」(橫放那張)。
        #   mjai 規格:宣告 reach 之後允許任何「打完仍然聽牌」的牌,**不限定
        #   剛摸的那張**;「只能摸切」是從立直成立之後的下一巡才開始
        #   (gimite 原版 possible_dahais 的註記:
        #    "Only tsumogiri is allowed after reach.")。
        #   所以宣言牌是立直後**唯一合法可以手切**的一張,必須排除在
        #   「立直後必然摸切」那條檢查之外,不然每次立直都會噴假矛盾。
        self.reach_pending = {s: False for s in self.seats}
        self.arm = {s: False for s in self.seats}        # 這一幀手臂蓋住了
        self.arm_frames = {s: 0 for s in self.seats}     # 這一巡蓋了幾幀
        # ★ 這一幀這一家的立直宣告動畫還在。使用者 2026-10-24 從判斷點截圖
        #   確認:立直的特殊動畫會讓程式**提前觸發**摸切判斷,所以它跟手臂
        #   一樣是判斷點的條件 —— 動畫還在就不判斷。
        self.reach_anim = {s: False for s in self.seats}
        self.reach_frames = {s: 0 for s in self.seats}
        # ★ 認出來的那一張摸入牌的座標。判斷點就是問「這個座標上還在不在」。
        self.pos = {s: None for s in self.seats}
        # 剛剛暗槓了 -> 摸入牌換位置了,要重新存一次座標。
        self.ankan = {s: False for s in self.seats}
        self.reacq = {s: 0 for s in self.seats}      # 重新取得幾次(報告用)

    def observe_arm(self, arm_dets, zones=None, conf=ARM_CONF):
        """吃這一幀所有的 arm 偵測,算出每一家的摸入牌區域有沒有被蓋住。"""
        for s in self.seats:
            self.arm[s] = arm_blocking(arm_dets, s, zones, conf)
            if self.arm[s]:
                self.arm_frames[s] += 1
        return dict(self.arm)

    def observe_reach_anim(self, seat_of_reach, conf_min=REACH_ANIM_CONF):
        """吃這一幀的立直橫幅偵測,算出每一家的宣告動畫還在不在。

        seat_of_reach  [(det, 那個框歸給哪一家), ...] —— 座位歸屬用
                       event_zones.classify,跟第二步同一套,這裡不重做。
        """
        hit = {s: False for s in self.seats}
        for d, seat in seat_of_reach:
            if seat in hit and d["conf"] >= conf_min:
                hit[seat] = True
        for s in self.seats:
            self.reach_anim[s] = hit[s]
            if hit[s]:
                self.reach_frames[s] += 1
        return dict(self.reach_anim)

    def blocked(self, seat):
        """判斷點的「現在不能判」條件:手臂或立直宣告動畫還在。

        ★ 兩個都是**遮擋**,不是證據。它們不進 judge(),只決定「什麼時候
          去看摸入牌」。
        """
        return self.arm.get(seat, False) or self.reach_anim.get(seat, False)

    def note_kan(self, seat, frame=None):
        """這一家宣告槓了。

        ★ 怎麼認出是**暗槓**(使用者 2026-10-24 給的判準):
          「辨識到玩家有摸入牌之後,沒有該玩家打牌事件,但宣告槓」。
          self.seen[seat] 正好就是這個條件 —— 它在摸入牌第一次被看到時立起來,
          在**那一家的打牌定案輸出時**才被 consume() 清掉。所以宣告槓的當下
          seen 還是 True,就代表「摸過牌、還沒打牌」= 自己的回合 = 暗槓/加槓。
          (大明槓是鳴別人打出的牌,那時 seen 是 False,不會進來。)

        暗槓會把 4 張從暗牌列拿走,摸入牌跟著換位置,所以存下來的座標要作廢,
        等嶺上牌出現再重新存一次。

        ★ 回傳值是**狀態有沒有翻**,不是「這一幀有沒有看到槓」。宣告橫幅
          會連續顯示好幾幀,每幀都回 True 的話終端會被同一次槓洗版五六行。
        """
        if seat in self.ankan and self.seen[seat] and not self.ankan[seat]:
            self.ankan[seat] = True
            return True
        return False

    def observe(self, dets, frame, zones=None):
        """吃這一幀所有的 drawn_tile 偵測。回傳 (歸屬好的, 沒認領的)。

        三件事,依序:
          1. 還在嗎:存下來的座標附近有沒有偵測 —— **不要求在框內**
          2. 認出來:中心點落在這一家的框裡 -> 是這一家的,存座標
          3. 暗槓後重新取得:剛暗槓過,舊座標沒東西了,就用附近最近的那個
             偵測重新存(只做這一次)
        """
        zones = zones or DRAWN_ZONES
        owner = [(d, seat_of_drawn(d["center"], zones)) for d in dets]
        used, hit = set(), {s: None for s in self.seats}

        def near(a, b, dist):
            return abs(a[0] - b[0]) <= dist and abs(a[1] - b[1]) <= dist

        def pick_near(ref, dist, s):
            best = None
            for i, (d, o) in enumerate(owner):
                if i in used or (o is not None and o != s):
                    continue
                if not near(d["center"], ref, dist):
                    continue
                if best is None or d["conf"] > owner[best][0]["conf"]:
                    best = i
            return best

        # --- 1. 存下來的座標上還在不在 ---------------------------------
        for s in self.seats:
            if self.pos[s] is None:
                continue
            i = pick_near(self.pos[s], DRAWN_AT_DIST, s)
            if i is not None:
                used.add(i)
                hit[s] = owner[i][0]
                self.ankan[s] = False      # 還在原位,沒換位置

        # --- 2. 認出來(框內)-> 存座標 ----------------------------------
        for s in self.seats:
            if hit[s] is not None:
                continue
            best = None
            for i, (d, o) in enumerate(owner):
                if i in used or o != s:
                    continue
                if best is None or d["conf"] > owner[best][0]["conf"]:
                    best = i
            if best is not None:
                used.add(best)
                hit[s] = owner[best][0]
                self.ankan[s] = False

        # --- 3. 暗槓之後重新存一次座標 ----------------------------------
        for s in self.seats:
            if hit[s] is not None or not self.ankan[s] or self.pos[s] is None:
                continue
            i = pick_near(self.pos[s], ANKAN_REACQ_DIST, s)
            if i is not None:
                used.add(i)
                hit[s] = owner[i][0]
                self.ankan[s] = False
                self.reacq[s] += 1

        for s in self.seats:
            self.drawn_now[s] = hit[s] is not None
            if hit[s] is not None:
                self.pos[s] = hit[s]["center"]
                self.seen[s] = True
                self.frames[s] += 1
                self.last_frame[s] = frame
        return ({s: d for s, d in hit.items() if d is not None},
                [d for i, (d, _o) in enumerate(owner) if i not in used])

    def evidence(self, seat, frame):
        """牌河剛出現新牌 -> 把判斷點的狀態拍下來。還不下結論。

        ★ 判斷跟輸出分開:這一刻還不知道那是什麼牌(票還沒投完),也還不確定
          那一格不是雜訊。所以這裡只存證據,等那一格定案才用它輸出。
        """
        return {"frame": frame,
                "drawn_now": self.drawn_now[seat],
                "drawn_seen": self.seen[seat],
                "forced_tedashi": self.forced[seat],
                "riichi": self.riichi[seat],
                "arm_blocked": self.arm[seat],
                "reach_anim": self.reach_anim[seat],
                "drawn_frames": self.frames[seat],
                "arm_frames": self.arm_frames[seat],
                "reach_frames": self.reach_frames[seat],
                "pos": self.pos[seat],
                "reacq": self.reacq[seat],
                "last_drawn_frame": self.last_frame[seat]}

    def consume(self, seat):
        """這一家的證據用掉了(那一格定案、已經輸出)-> 這一巡結束。"""
        self.seen[seat] = False
        self.frames[seat] = 0
        self.arm_frames[seat] = 0
        self.reach_frames[seat] = 0
        self.last_frame[seat] = None
        # ★ 座標要清掉:這一巡結束了,下一巡的摸入牌要重新「認出來」(框內)。
        #   不清的話下一巡會拿上一巡的座標去問「還在不在」,而那個位置通常
        #   真的還有一張牌(剛打完牌,手牌補上來),就會變成假的手切。
        self.pos[seat] = None
        self.ankan[seat] = False
        self.reacq[seat] = 0
        self.forced[seat] = False
        self.reach_pending[seat] = False

    def note_naki(self, seat, cls):
        """看到這一家吃/碰 -> 下一打強制手切。槓不算(補摸嶺上牌)。

        ★ 刻意做成「設旗標」而不是「發事件」:同一個橫幅被模型重複偵測成
          chi 又 pon 是實測過的事(第二步的紀錄),設旗標是等冪的,重複偵測
          不會造成任何問題,所以這裡不需要第二步那整套邊緣觸發。
        """
        if seat in self.forced and cls in NO_DRAW_AFTER:
            self.forced[seat] = True

    def note_reach(self, seat):
        """看到這一家立直。

        ★ 兩個狀態:
            riichi         這一局立直了(之後的摸打必然摸切,除暗槓)
            reach_pending  **下一打是宣言牌**,那一張不受「必然摸切」約束
        """
        if seat in self.riichi:
            if not self.riichi[seat]:
                self.reach_pending[seat] = True
            self.riichi[seat] = True

    def new_hand(self):
        """換局:所有旗標清掉。"""
        for s in self.seats:
            self.drawn_now[s] = False
            self.seen[s] = False
            self.frames[s] = 0
            self.arm[s] = False
            self.arm_frames[s] = 0
            self.reach_anim[s] = False
            self.reach_frames[s] = 0
            self.pos[s] = None
            self.ankan[s] = False
            self.reacq[s] = 0
            self.last_frame[s] = None
            self.forced[s] = False
            self.riichi[s] = False
            self.reach_pending[s] = False
