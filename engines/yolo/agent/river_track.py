"""牌河的跨幀追蹤 —— 第三步跟第四步共用的那一塊。

=== 為什麼單獨開一個檔 ===
RiverTracker 原本住在 3_river.py 裡。第四步(摸切判斷)要的就是它:
「這一家的牌河什麼時候出現新的一格」跟「那一格定案成什麼牌」。
但模組名不能以數字開頭,所以 4_tsumogiri.py 要用它只能走 importlib ——
那正是第二步把宣告邏輯寫在 2_events.py 裡踩過的坑(見 tiles.py 的說明)。
這次是**新建**依賴,所以現在搬出來比以後再搬便宜。

這個檔案只有「量測」,沒有畫面、沒有 CSV、沒有報告 —— 兩支程式各自做那些。
"""
from collections import Counter

from river import (SEATS, mark_sideways, rows_of, sort_by_discard_order)


# 跨幀配對:中心點距離小於這個值視為同一個位置。沿用舊 readers.py 的 25。
MATCH_DIST = 25


# 一個位置連續這麼多幀沒看到就封存(視為那一格結束了)。
# ★ 這個值決定兩件事,而且方向相反:
#     太小 -> 牌只是閃爍漏讀幾幀就被當成「結束」,同一張牌裂成兩個位置,
#             閃爍反而量不到(變成兩個各自穩定的位置)
#     太大 -> 跨局不會重置。下一局同樣的格子放了別的牌,會被算成「牌名變了」
#   30 幀 @ 14 fps ≈ 2 秒:比閃爍的漏讀長得多,比局間的結算畫面短得多。
#   報告第 4 節會印出實際觀測到的漏讀長度分布,用來回頭驗這個值。
GONE_FRAMES = 30


# 牌名變動的時候,距離上次看到這個位置幾幀以內算「純閃爍」。
# 間隔大的變動另外歸類 —— 那可能是被鳴走之後那一格換了人,或是換局,
# 兩者都不是模型讀錯。把它們混在一起算閃爍率就是在誇大錯誤。
FLICKER_GAP = 2


# ---------------------------------------------------------------------------
# 畫面上「這一格的答案」—— ★ 只影響顯示,不影響任何統計
# ---------------------------------------------------------------------------
# 一個位置累積到這麼多幀、而且最高票佔這麼多比例,畫面上就當成「有答案了」。
#
# ⚠️ 這**不是**第四步的定案機制。差別很重要:
#     這裡    只是把累積到的最高票畫在螢幕上,給人看的。CSV 照舊逐幀記錄
#             每一幀的原始讀法,第 4 節的閃爍率也是從原始讀法算的,完全
#             不受這兩個數字影響。
#     第四步  要把牌面「鎖住」然後送進 mjai 事件流,送出去就收不回來,
#             所以那裡的規則要另外量、另外定(票數、鎖、drift 都要處理)。
#
# 為什麼不乾脆只畫最高票就好:那會把 24.7% 的閃爍藏起來,畫面上看到一排
# 穩定正確的答案,就會誤以為模型很準。所以這一幀的讀法跟最高票不一致的
# 時候要畫成 `9m!8m`(最高票!這一幀)並標紅 —— 閃爍要看得見。
# ★★ 定案規則(2026-10-23 依使用者的指正重寫)
#
#   使用者的指正很根本:「如果在定案的時候辨識就有問題,那後續改答案既沒有
#   意義也沒用。」—— 對。mjai 的打牌事件送出去就收不回來(舊紀錄 readers.py
#   那條「dahai 送出去就收不回來」是同一件事)。所以**定案當下讀對**才是
#   唯一重要的事,「答案之後會越來越好」是假的安慰。
#
#   所以規則簡化成一條,而且跟量測對得起來:
#       看滿 CONFIRM_FRAMES 幀 -> 取那幾幀的多數決 -> **凍結,之後不再改**
#
#   CONFIRM_FRAMES 從 W 掃描挑(報告第 4 節,那一節量的就是「只看前 W 幀
#   投票,跟看完整段的答案差幾格」):
#       W=2  -> 30.6%      W=3  -> 19.4%      W=5  -> 15.3%
#       W=8  -> 8.3%       W=12 -> 4.2%       W=30 -> 1.4%
#   取 8(≈690ms @11.6fps)。再往上拉,等待時間換到的改善變小,而打牌事件
#   晚送出去是有代價的。
#   ⚠️ 這個 8.3% 裡有一塊是 2s/3s 那種「兩個候選各半」的格子 —— 那種等
#      再久也不會收斂。所以補完資料集之後要重測,說不定 W 可以調小。
CONFIRM_FRAMES = 8


SETTLED_SHARE = 0.6


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def order_frame(seat, dets):
    """把一家這一幀的偵測排好,附上排號與橫放旗標。

    回傳 [(det, order, row, sideways), ...],order 從 0 開始。
    """
    ordered = sort_by_discard_order(seat, dets)
    rows = rows_of(seat, ordered)
    side = mark_sideways(ordered)
    # 打牌順序掛回 det 上 —— RiverTracker 要記進那一格,報告才講得出
    # 「第幾張」。實機跟離線重算兩條路都走這裡,所以只會有一份。
    for i, d in enumerate(ordered):
        d["order"] = i
        d["sideways"] = bool(side[i])
    return list(zip(ordered, range(len(ordered)), rows, side))


def _area(det):
    x1, y1, x2, y2 = det["box"]
    return max(1, x2 - x1) * max(1, y2 - y1)


def winner_conf(slot):
    """這一格勝出的那個牌名,平均信心度多少。算不出來回傳 None。

    ★ 要看**勝出那個牌名自己的**平均分,不是整格的平均。整格平均會被其他
      候選稀釋,而我們要問的是「模型對它給的這個答案有多少把握」。
    """
    if not slot or not slot.get("names") or not slot.get("wnames"):
        return None
    top, n = slot["names"].most_common(1)[0]
    return slot["wnames"].get(top, 0.0) / float(max(1, n))


def is_settled(slot, votes=CONFIRM_FRAMES, share=SETTLED_SHARE):
    """這一格定案了沒 = answer 有沒有被寫進去(看滿 N 幀就定案並凍結)。

    share 留著只是為了舊的離線分析(RiverTracker 沒跑過的資料);正常路徑
    一律看 answer。
    """
    if not slot:
        return False
    if slot.get("answer") is not None:
        return True
    if not slot.get("names"):
        return False
    _top, n_top = slot["names"].most_common(1)[0]
    return (slot["n"] >= votes
            and n_top / float(max(1, slot["n"])) >= share)


def answer_of(slot):
    """這一格定案的答案;還沒定案回傳 None。★ 定案後不會改。"""
    if not slot:
        return None
    if slot.get("answer") is not None:
        return slot["answer"]
    if is_settled(slot) and slot.get("names"):
        return slot["names"].most_common(1)[0][0]
    return None


class RiverTracker:
    """逐家維護「牌河上的一個位置看過哪些牌名」。

    ★ 只記錄,不投票、不定案。
      定案(票數夠了就把牌面鎖住)是下一步的事,而且它是**判斷**;這裡做的
      是**量測**。混在一起的話,量到的「閃爍率」就會被自己的定案規則改掉,
      再也分不出是模型不穩還是規則擋掉了。

    位置用移動平均 —— 偵測框每幀會抖幾 px,不平均的話配對會越飄越遠。
    """

    def __init__(self, match_dist=MATCH_DIST, gone_frames=GONE_FRAMES,
                 flicker_gap=FLICKER_GAP, confirm_frames=CONFIRM_FRAMES):
        self.match_dist = match_dist
        self.gone_frames = gone_frames
        self.flicker_gap = flicker_gap
        self.confirm_frames = confirm_frames
        self.live = {s: [] for s in SEATS}     # 還在的位置
        self.done = {s: [] for s in SEATS}     # 封存的位置(報告用)
        self.gaps = []                         # 觀測到的漏讀長度(驗 GONE)

    HEAD_LEN = 40          # 只留前幾幀的讀法(定案窗口掃描用),不無限長

    def _new_slot(self, det, frame):
        return {"seat": det.get("seat"),
                "center": det["center"], "n": 1, "first": frame,
                "last_frame": frame, "last": det["pai"],
                "names": Counter([det["pai"]]), "confs": [det["conf"]],
                # 用來比「多數決 / 信心度加權 / 單幀最高分」哪個定得準。
                # 使用者 2026-10-23 問「最終判定是不是看信心度最高」——
                # 不是(是多數決),但哪個比較好值得用資料回答。
                "wnames": Counter({det["pai"]: det["conf"]}),
                "top1": (det["conf"], det["pai"]),
                "head": [det["pai"]], "changes": [],
                "order": det.get("order"), "run": det.get("run"),
                # 平均面積 —— 用來把結算畫面那種「尺寸明顯不對」的假格子
                # 濾掉(它們的框比真的河牌大兩倍以上)。
                "area": _area(det), "an": 1,
                # 這一格有幾幀被判成橫放。★ 橫放要看**整段**,不能逐幀判:
                # 手掃過去把框壓扁一幀就會被當成橫放。見 slot_sideways。
                "side_n": 1 if det.get("sideways") else 0,
                # ★ 定案後凍結的答案。一旦寫進去就不再改 —— 這是使用者
                #   2026-10-23 的要求,也是對的:送出去的事件收不回來。
                "answer": None, "first_ms": det.get("t_ms"),
                # 這一行已經印過了沒。一張牌只印一次。
                "said": False}

    @staticmethod
    def _restart(s, det, frame):
        """這一格換人了(隔很久之後變成別的牌)。票要重算,不能跨過去累積。

        ★ 隔了很久才變,代表這一格被結算畫面擋住之後重新讀到,或是換局之後
          同一格放了別的牌。把兩段的票加在一起有兩個後果:
            ・畫面上會顯示上一張牌的最高票(明明已經換人了)
            ・「等幾幀才能定案」會量成好幾百幀,因為第一次看到是上一段的
          2026-10-23 兩個都踩到了。

        ★★ 只有**還沒定案**的格子才會走到這裡,見 update()。定案過的格子
            斷檔再回來不算換牌 —— 那是手擋住,不是新的牌。
        """
        s["n"] = 1
        s["first"] = frame
        s["names"] = Counter([det["pai"]])
        s["wnames"] = Counter({det["pai"]: det["conf"]})
        s["top1"] = (det["conf"], det["pai"])
        s["head"] = [det["pai"]]
        s["confs"] = [det["conf"]]
        s["side_n"] = 1 if det.get("sideways") else 0
        s["answer"] = None
        s["first_ms"] = det.get("t_ms")
        s["said"] = False

    def update(self, seat, dets, frame):
        """吃一家這一幀的偵測。回傳 [(slot, det, change_kind), ...]。

        change_kind: None / "flicker" / "gapped"
        """
        slots = self.live[seat]
        # 1. 先封存太久沒看到的。★ 一定要在配對之前做,不然上一局同一個格子
        #    的位置會把這一局的牌吃進去,變成「牌名變了」。
        keep = []
        for s in slots:
            if frame - s["last_frame"] > self.gone_frames:
                self.done[seat].append(s)
            else:
                keep.append(s)
        self.live[seat] = slots = keep

        out, used = [], set()
        # 信心度高的先挑位置 —— 同一個位置有兩個候選時,該讓有把握的那個占。
        for d in sorted(dets, key=lambda z: -z["conf"]):
            best, bd = None, None
            for i, s in enumerate(slots):
                if i in used:
                    continue
                dd = dist2(s["center"], d["center"])
                if dd <= self.match_dist ** 2 and (bd is None or dd < bd):
                    best, bd = i, dd
            if best is None:
                slots.append(self._new_slot(d, frame))
                used.add(len(slots) - 1)
                out.append((slots[-1], d, None))
                continue
            s = slots[best]
            used.add(best)
            gap = frame - s["last_frame"]
            if gap > 1:
                self.gaps.append(gap)
            kind = None
            if d["pai"] != s["last"]:
                kind = "flicker" if gap <= self.flicker_gap else "gapped"
                # ★ 這一次變動,是發生在「已經有穩定答案之後」還是「進場期」?
                #   2026-10-23 使用者的觀察:剛打出的牌會被手、特殊動畫或
                #   閃爍特效擋住,干擾一結束就讀對了 —— 那是**暫態**,不是
                #   讀錯。兩種混在一起算,閃爍率就沒有意義。
                #   判斷要在這一幀的票投進去**之前**做,不然剛變的那一票
                #   會把 settled 自己推翻掉。
                s["changes"].append((frame, s["last"], d["pai"], gap, kind,
                                     is_settled(s)))
                s["last"] = d["pai"]
            s["last_frame"] = frame
            # ★★ 2026-10-23:定案過的格子**永遠不重算**。
            #
            # 原本只要「斷檔之後牌名變了」就 _restart(票清掉、answer 清掉、
            # said 清掉),結果同一個位置會重新定案、重新輸出。P0 整場重複
            # 輸出就是這個:
            #   f651 3s / f652 2s(手掃過去) / f653-654 完全讀不到 /
            #   f655 3s  -> gap=3 > flicker_gap=2 -> 判成「換牌了」-> 重算
            # 但那三幀是 P0 自己的手伸過去打牌擋住牌河,牌根本沒變。
            # P0 特別嚴重是因為**只有 P0 的手會蓋在自己的牌河上**,
            # 其他三家的手在畫面上離自己的牌河很遠。
            #
            # 真正的「同一格換成別的牌」是換局,那時牌河會空上好幾百幀,
            # 上面 gone_frames=30 的封存就處理掉了,不需要靠這條。
            if kind == "gapped" and s["answer"] is None:
                self._restart(s, d, frame)      # 還沒定案才重算
            else:
                s["n"] += 1
                s["names"][d["pai"]] += 1
                s["wnames"][d["pai"]] += d["conf"]
                if d["conf"] > s["top1"][0]:
                    s["top1"] = (d["conf"], d["pai"])
                s["confs"].append(d["conf"])
                s["area"] += _area(d)
                s["an"] += 1
                if d.get("sideways"):
                    s["side_n"] += 1
                for _k in ("order", "run"):
                    if d.get(_k) is not None:
                        s[_k] = d[_k]
                if len(s["head"]) < self.HEAD_LEN:
                    s["head"].append(d["pai"])
                # ★ 看滿 N 幀就定案,取那幾幀的多數決,然後**凍結**。
                if (s["answer"] is None
                        and s["n"] >= self.confirm_frames):
                    s["answer"] = Counter(
                        s["head"][:self.confirm_frames]).most_common(1)[0][0]
                    s["answer_frame"] = frame
            n = s["n"]
            s["center"] = ((s["center"][0] * (n - 1) + d["center"][0]) // n,
                           (s["center"][1] * (n - 1) + d["center"][1]) // n)
            out.append((s, d, kind))
        return out

    def all_slots(self, seat=None):
        seats = (seat,) if seat else SEATS
        out = []
        for s in seats:
            out += self.done[s] + self.live[s]
        return out



# 張數剛好少一張、連續這麼多幀才算「被鳴走了」。
# 依據第三步實測:雜訊最長撐 3 幀連續。遮擋造成的張數起伏回升途中會經過
# 「剛好少一張」,但只有一兩幀 —— 真的被鳴走則會一直少著。
DROP_FRAMES = 3


class DahaiGate:
    """這個新格子**是不是一次真的打牌**。

    === 為什麼需要 ===
    RiverTracker 的「新格子」只代表「牌河裁圖裡冒出一個沒配到既有格子的
    偵測」,它不知道那是不是打牌。2026-10-24 實測踩到的:暗槓的角色立繪
    動畫整片蓋過 P3 的牌河 -> 偵測漂移/消失超過 GONE_FRAMES -> P3 所有
    格子被封存 -> 動畫散掉之後**每一張牌都變成新格子** -> 每一張都重新
    定案、重新輸出。f889 打的 9p 在 f937 又輸出一次,而且插在 P2 還沒打牌
    的中間。送進 mjai 就是重複的 dahai,整條事件流從此對不上。

    === 規則(使用者 2026-10-24 給的)===
    「沒有新開一局或被鳴牌的話,牌河最新牌的座標位置不應該倒退;就算被
      鳴牌,一次也只會倒退到前一格。」

    實作成**位置**而不是張數:
        已經輸出過的位置  -> 不可能再是新打牌(那是同一張牌被重新偵測到)
        被鳴走            -> 最後輸出的那個位置釋放掉,可以再放一張新的

    ⚠️ 走過的死路:一開始用「當幀張數有沒有超過歷史水位」判斷。**不行** ——
       單幀張數太吵(實測五局:少 1 張佔 12.4%、少 2 張佔 2.3%),新牌出現
       那一幀剛好有兩張舊牌沒讀到就會被誤擋,一局誤擋 7~14 筆真的打牌。
       第二條死路是「少 3 張以上就當作看不見、凍結格子不老化」,一樣會掉
       6~7 筆。位置守門五局實測:擋下 1~2 個假格子,真的打牌一筆都沒掉。

    ★ 「被鳴走」刻意用張數判斷,不用鳴牌事件 —— 第二步的鳴牌偵測本來就會
      漏(實測有整次槓沒抓到)。少一個依賴。

    ★ 被擋下來的格子**還是要建立、還是要追蹤**(不然下一幀又變成新的),
      只是不觸發摸切判斷、不輸出。所以這個類別只回答 yes/no,不碰 slot。
    """

    def __init__(self, seats=SEATS, drop_frames=DROP_FRAMES,
                 dist=MATCH_DIST):
        self.drop_frames = drop_frames
        self.dist = dist
        self.said = {s: [] for s in seats}   # 已經輸出過的位置,照順序
        self.peak = {s: 0 for s in seats}    # 這一家的張數水位
        self.drop = {s: 0 for s in seats}    # 「剛好少一張」連續幾幀了

    def observe(self, seat, n):
        """每一幀先吃該家這一幀的張數。要在 is_dahai 之前呼叫。"""
        if n == self.peak[seat] - 1:
            self.drop[seat] += 1
            if self.drop[seat] >= self.drop_frames:
                # 少一張而且撐住了 = 最新那張被鳴走 -> 位置退一格
                self.peak[seat] = n
                self.drop[seat] = 0
                if self.said[seat]:
                    self.said[seat].pop()
        else:
            self.drop[seat] = 0
        if n > self.peak[seat]:
            self.peak[seat] = n

    def is_dahai(self, seat, center):
        """新格子在 center:它是不是一次真的打牌。"""
        if center is None:
            return False
        return not any(abs(a[0] - center[0]) < self.dist
                       and abs(a[1] - center[1]) < self.dist
                       for a in self.said[seat])

    def note_said(self, seat, center):
        """這一格輸出過了 -> 登記位置,之後同一個位置不再算新打牌。"""
        self.said[seat].append(tuple(center))

    def new_hand(self):
        for s in self.said:
            self.said[s] = []
            self.peak[s] = 0
            self.drop[s] = 0


def force_confirm(tracker, seats, frame, only_dahai=True):
    """鳴牌來了 -> 把還在投票中的格子用**現有的票**直接定案。

    回傳 [(seat, slot), ...] —— 這一次被強制定案的。

    === 為什麼一定要有 ===
    2026-10-24 實測,四次全中:
        f77  P3 牌河出現新牌  -> +6 幀 f83  P2 碰
        f121 P3 牌河出現新牌  -> +5 幀 f126 P2 碰
        f96  P3 牌河出現新牌  -> +6 幀 f102 P2 碰
        f142 P3 牌河出現新牌  -> +6 幀 f148 P2 碰
    被碰走的牌在第 5~6 幀就從牌河消失,而定案要 CONFIRM_FRAMES(8)票 ——
    **永遠湊不滿**,所以那一格沒有答案、沒有輸出。P3 那兩張打牌就這樣整個
    不見了。

    對 mjai 來說這是致命的,不是少印一行:被碰走的那張牌一定要有對應的
    dahai 事件,不然 Mortal 收到的是「P2 碰了一張不存在的牌」,整條事件流
    從此對不上。票數不夠的牌面不可靠,但**缺一筆 dahai 嚴重得多**。
    (舊專案 discard.py 的 force_resolve_all 就是這件事。)

    ★ 對**四家都試一次**,不去推「被鳴的是誰」。被鳴的是上一個行動者,而
      那只能從指示燈拿 —— 指示燈是落後指標(實測打牌那家的手臂收回之前
      根本不會切換)。已經定案的格子不會有反應,還在投票中的那一家自然是
      唯一有回應的。

    ★ 摸切判斷**不要**改用鳴牌那一幀。鳴牌發生在打牌之後一段時間,那時
      理牌可能早就做完,拿鳴牌那幀去讀會把手切全部讀成摸切。判斷點照舊用
      slot 自己存的那一份證據。
    """
    out = []
    for seat in seats:
        for slot in tracker.live.get(seat, []):
            if slot.get("answer") is not None or slot.get("said"):
                continue
            if only_dahai and not slot.get("dahai"):
                continue
            if not slot.get("head"):
                continue
            slot["answer"] = Counter(slot["head"]).most_common(1)[0][0]
            slot["answer_frame"] = frame
            slot["forced"] = slot["n"]        # 用幾票定的,附在結果上
            out.append((seat, slot))
    return out
