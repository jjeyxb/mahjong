"""牌河轉接層:把 RiverTracker + DahaiGate 包成「事件」給狀態機用。

v6.2 牌河轉接層(A02~A12)的實作。重用 river_track.py,不改它。

=== 這一層輸出什麼 ===
    RiverCandidate  新格子通過 DahaiGate -> 一次打牌的「佔位」(牌面未知)
                    t_first 是那張牌**第一次被看到**的時刻,不是定案時刻
    RiverConfirmed  那一格 8 幀定案(或被鳴走時強制定案)-> 補上牌面
    RiverReleased   那一格被封存了(被鳴走 / 張數掉了)-> 同位置下一張是新的

=== 這一層補了舊程式的兩個洞 ===
D1 被鳴走那一格會吞掉下一張牌
   舊程式:DahaiGate 放掉位置,但 RiverTracker 裡那一格的 answer/said 還凍著,
   同位置下一張牌在 GONE_FRAMES(30 幀)內出現就被配回舊格子,不會輸出。
   這裡:兩層一起處理,而且有兩條觸發路徑 ——
     (a) 狀態機確認鳴牌 -> claim(seat, slot_id, generation)
     (b) 沒抓到鳴牌,但 DahaiGate 的張數連續少一張 -> 同樣封存那一格
   兩條路徑都可能先到,所以有防重複的帳(expect_drop)。
D3 P0 不在舊的 WATCH_SEATS 裡
   這裡四家都走同一條路。

=== 身分 ===
slot_id     這一局唯一,`{hand}-{seat}-S{n}`
generation  同一個位置第幾次放牌(被鳴走後同位置再放 = 2)
claim 一定要帶 slot_id + generation,對不上就拒絕 —— 遲到的鳴牌事件
不可以封存同位置的新牌(v6.2 測試 4)。
"""
from collections import Counter
from dataclasses import dataclass, field

import _paths  # noqa: F401
from river import SEATS
from river_track import (CONFIRM_FRAMES, DROP_FRAMES, GONE_FRAMES,
                         MATCH_DIST, DahaiGate, RiverTracker, order_frame)

# 一格裡被判成橫放的幀數比例超過這個值,才算立直宣言牌。
# 橫放要看整段,不能逐幀判(手掃過去把框壓扁一幀就會被當成橫放)。
SIDEWAYS_SHARE = 0.5

# ★ 新格子要被看到幾幀才送出候選。
#   2026-10-26 實機錄影:打牌的飛行動畫中途會被偵測到一幀(位置偏、框比定位後
#   大 30~50%),下一幀才落到定位。只看一幀就送出的話,每次都會多一筆假打牌
#   (那一局 9 筆 EVENT_GAP 有 8 筆是這個)。飛行中的那一格只存在一幀,
#   要求 2 幀就濾掉了;t_first 仍然是第一次看到的那一幀。
#   舊程式只在定案(8 幀)才輸出,所以從來沒遇過這個問題。
CAND_MIN_FRAMES = 2


@dataclass
class RiverCandidate:
    seat: str
    slot_id: str
    generation: int
    frame: int
    t_first_ms: float
    center: tuple
    kind: str = "candidate"


@dataclass
class RiverConfirmed:
    seat: str
    slot_id: str
    generation: int
    frame: int
    pai: str
    votes: int
    forced: bool
    sideways: bool
    conf: float
    kind: str = "confirmed"


@dataclass
class RiverReleased:
    seat: str
    slot_id: str
    generation: int
    frame: int
    reason: str           # "claim" / "count_drop" / "new_hand"
    kind: str = "released"


@dataclass
class ClaimResult:
    ok: bool
    reason: str
    events: list = field(default_factory=list)


def _near(a, b, dist=MATCH_DIST):
    return abs(a[0] - b[0]) < dist and abs(a[1] - b[1]) < dist


class GuardedGate(DahaiGate):
    """DahaiGate 加兩件事:

    1. 可以「指定位置」釋放(鳴牌確認時),而不是只能 pop 最後一個。
    2. 防重複:鳴牌先釋放了位置的話,之後張數掉下來那一下不可以再 pop 一次
       —— 再 pop 就會把**上一張**已輸出的位置也放掉,那一格之後被重新偵測
       就會變成一次假打牌。

    observe() 回傳這一幀「因為張數掉了而釋放的位置」(沒有就 None),讓轉接層
    同步封存 RiverTracker 那一格。
    """

    def __init__(self, seats=SEATS, drop_frames=DROP_FRAMES, dist=MATCH_DIST):
        super().__init__(seats, drop_frames, dist)
        self.expect_drop = {s: 0 for s in seats}

    def observe(self, seat, n):
        released = None
        if n == self.peak[seat] - 1:
            self.drop[seat] += 1
            if self.drop[seat] >= self.drop_frames:
                self.peak[seat] = n
                self.drop[seat] = 0
                if self.expect_drop[seat] > 0:
                    # 鳴牌已經先釋放過了,這一下張數掉落就是它,不再 pop
                    self.expect_drop[seat] -= 1
                elif self.said[seat]:
                    released = self.said[seat].pop()
        else:
            self.drop[seat] = 0
        if n > self.peak[seat]:
            self.peak[seat] = n
        return released

    def release_at(self, seat, center):
        """指定位置釋放。回傳有沒有真的找到並釋放。"""
        for i in range(len(self.said[seat]) - 1, -1, -1):
            if _near(self.said[seat][i], center, self.dist):
                del self.said[seat][i]
                # 那張牌消失時張數會掉一張 -> 預先登記,不要再 pop 別人
                self.expect_drop[seat] += 1
                return True
        return False

    def new_hand(self):
        super().new_hand()
        self.expect_drop = {s: 0 for s in self.said}


# ★ 2026-10-29 實機:錄製端牌河門檻 0.10,P1 的牌河區自動擴張到右上角後,
#   那裡兩個 0.16/0.22 的假框在新局一開始就被當成 P1 打了兩張 -> EVENT_GAP。
#   實機打出去的牌,定案信心最低 0.33(三場、373 張)。
MIN_DET_CONF = 0.30


def is_sliver(d):
    """太小或太細長的框:中央面板四角的風位標記、牌河區邊緣切到半張的東西。

    ★ 2026-10-30 實戰東1 f2531:P2 牌河區左下角 33x20 的框讀成 E(0.45),
      跟真的 8s 同時出現 -> P2 連打兩張 -> EVENT_GAP。
      真的牌河牌(五份錄影、32 萬個框)面積 0.1% 分位 1716、長寬比 >= 0.5。"""
    w = d["box"][2] - d["box"][0]
    h = d["box"][3] - d["box"][1]
    return w * h < 1400 or min(w, h) / max(w, h, 1) < 0.42


def is_hover_tip(seat, d):
    """P0 手牌滑鼠移上去時跳出的「剩幾張」提示框(聽牌時一次列好幾張)。

    ★ 2026-10-30 實戰:提示框裡的牌被牌河模型當成 P0 的棄牌(東1 f2147~2395、
      東2-1 f5418 讀成 W、東4 f7099~7122 兩張)-> 多一張打牌、手牌帳對不上、
      守門器停送。特徵(1080p):橫的框(寬 80~84 > 高 64~76)、y 718~724、
      x 在 960 ± 60k 上(一張在 960,兩張在 900/1020)。
      真的 P0 第三排直牌是 65x85;第三排的橫牌(立直)中心算起來在
      799/864/929/994/1059,離 960 ± 60k 都超過 19 px。"""
    if seat != "P0":
        return False
    x, y = d["center"]
    b = d["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    if not (712 <= y <= 732 and w > h):
        return False
    r = (x - 960) % 60
    return min(r, 60 - r) <= 12


def is_bottom_clip(seat, d, zone):
    """P0 牌河區底邊切到的扁框:不是牌河的牌。

    ★ 2026-11-06 錄影 human-20261103 東3局 f4332:P0 牌河區底邊一個 83x45 的扁框
      讀成白(0.56)、停了 8 幀 -> 被當成一次打牌定案,佔住第三排第一格的位置。
      之後真的九索落在那裡,Gate 以為那格已經打過 -> 那一打整個漏掉,之後
      P0 每一摸都錯開一位。
      真的第三排直牌也貼著底邊,但高 81;立直橫放的高約 64。五份錄影裡貼底邊、
      橫的、高不到 55 的框一共 9 段,全部是假的(最寬 163)。"""
    if seat != "P0" or not zone:
        return False
    b = d["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    return w > h and h < 55 and b[3] >= zone[1] + zone[3] - 3


class RiverAdapter:
    """四家牌河 -> RiverCandidate / RiverConfirmed / RiverReleased。

    用法:每一幀呼叫 feed(frame_index, t_ms, river_obs),拿回事件清單。
    river_obs 就是觀測檔裡的 `river` 欄位(seat -> {tiles: [...]});
    某一家是 None 或 tiles 是 None 代表那一幀沒跑,那一家這一幀不更新。
    """

    def __init__(self, hand_id="H000", confirm_frames=CONFIRM_FRAMES,
                 gone_frames=GONE_FRAMES, match_dist=MATCH_DIST,
                 seats=SEATS, cand_min_frames=CAND_MIN_FRAMES):
        self.seats = tuple(seats)
        self.cand_min_frames = cand_min_frames
        self.confirm_frames = confirm_frames
        self.gone_frames = gone_frames
        self.match_dist = match_dist
        self.tracker = RiverTracker(match_dist, gone_frames,
                                    confirm_frames=confirm_frames)
        self.gate = GuardedGate(self.seats, dist=match_dist)
        self.hand_id = hand_id
        self._n = 0
        self.by_id = {}                 # slot_id -> slot(活的跟封存的都有)
        self.positions = {s: [] for s in self.seats}   # [(center, count)]
        self.visible = {s: 0 for s in self.seats}       # 這一幀看到幾張
        self.last_cand = {}                              # 每一家上一個候選的位置
        self.stats = Counter()

    # ------------------------------------------------------------------
    def new_hand(self, hand_id, frame=0):
        """換局:三層同步重設(v6.2 D4)。舊格子封存,歷史保留在 by_id。"""
        out = []
        for s in self.seats:
            for sl in list(self.tracker.live[s]):
                self.tracker.live[s].remove(sl)
                self.tracker.done[s].append(sl)
                if sl.get("slot_id") and not sl.get("released"):
                    sl["released"] = "new_hand"
                    out.append(RiverReleased(s, sl["slot_id"],
                                             sl["generation"], frame,
                                             "new_hand"))
        self.gate.new_hand()
        self.hand_id = hand_id
        self._n = 0
        self.positions = {s: [] for s in self.seats}
        self.last_cand = {}
        return out

    # ------------------------------------------------------------------
    def _generation(self, seat, center):
        for i, (c, k) in enumerate(self.positions[seat]):
            if _near(c, center, self.match_dist):
                self.positions[seat][i] = (c, k + 1)
                return k + 1
        self.positions[seat].append((tuple(center), 1))
        return 1

    def _assign_id(self, seat, slot):
        self._n += 1
        slot["slot_id"] = f"{self.hand_id}-{seat}-S{self._n:03d}"
        slot["generation"] = self._generation(seat, slot["center"])
        self.by_id[slot["slot_id"]] = slot

    def _archive(self, seat, slot, frame, reason):
        if slot in self.tracker.live[seat]:
            self.tracker.live[seat].remove(slot)
            self.tracker.done[seat].append(slot)
        slot["released"] = reason
        slot["released_frame"] = frame
        return RiverReleased(seat, slot.get("slot_id"), slot.get("generation"),
                             frame, reason)

    def _resurrected(self, seat, slot, frame):
        """這個新格子是不是「因為張數掉落被封存的那一格」又出現了。

        ★ 張數掉落那條備援路徑(沒抓到鳴牌時用)會誤傷一種情況:最新那張牌
          被手或動畫擋住超過 DROP_FRAMES 幀。擋完再出現,它會變成新格子 ->
          通過 Gate(位置已被放掉)-> 一次假打牌。
          判準:同位置、GONE_FRAMES 內、牌種一樣 -> 是同一張。
          真的被鳴走後同家再打到同一格,中間要隔三家行動,遠超過 GONE_FRAMES;
          而且還得剛好是同一種牌。
        ★ 2026-11-07 被鳴走的那一格也一樣:鳴牌確認時牌還留在牌河上幾幀,封存之後
          同位置同一張又被看成新格子 -> 一次假打牌 -> 狀態機以為那一家又打了一張,
          等鳴牌那家打牌時再「推定」一次鳴牌。錄影 selfplay-20261106-110134
          東1局 f1464 吃 8s(8s 留到 f1469),事件流丟了四巡左右。
        """
        from tiles import base_pai
        pai = slot["last"]
        for old in reversed(self.tracker.done[seat]):
            if old.get("released") not in ("count_drop", "claim", "claim_ghost"):
                continue
            if frame - old.get("released_frame", -10 ** 9) > self.gone_frames:
                continue
            if not _near(old["center"], slot["center"], self.match_dist):
                continue
            if old.get("answer") and base_pai(old["answer"]) == base_pai(pai):
                old["resurrected_from"] = ("claim" if old["released"]
                                           .startswith("claim") else old["released"])
                old["released"] = "resurrected"
                return old
        return None

    @staticmethod
    def _sideways(slot):
        return slot.get("side_n", 0) / float(max(1, slot.get("n", 1))) \
            >= SIDEWAYS_SHARE

    def _confirm_event(self, seat, slot, frame):
        slot["said"] = True
        self.gate.note_said(seat, slot["center"])
        top = slot["answer"]
        wn = slot.get("wnames", {})
        n = slot["names"].get(top, 0) if slot.get("names") else 0
        conf = (wn.get(top, 0.0) / float(max(1, n))) if n else 0.0
        return RiverConfirmed(seat, slot["slot_id"], slot["generation"], frame,
                              top, slot.get("forced") or slot["n"],
                              bool(slot.get("forced")), self._sideways(slot),
                              round(conf, 3))

    # ------------------------------------------------------------------
    def feed(self, frame, t_ms, river_obs, frozen=()):
        """frozen:這一幀被宣告橫幅蓋住的家(pipeline 算好傳進來)。
        那幾家這一幀整個不更新 —— 張數不算、新牌不收,等橫幅走了再照常處理。

        ★ 2026-11-04 實戰 runs\\20261030_183152 東4局 f7618~7620:P2 開槓的
          橫幅蓋住 P3 牌河唯一那張 8s 三幀,新牌河模型讀不到 -> Gate 以為牌被
          拿走、放掉位置 -> 8s 回來時被當成一次新的打牌 -> 之後 P3 的牌全部錯位,
          守門器擋下整局。舊模型剛好在橫幅下還讀得到,所以沒事。
          不靠模型「看穿」橫幅,改成橫幅蓋住時那一家先不動:代價是那一家的新
          打牌晚一點(橫幅約 1 秒)才進來 —— 使用者 2026-11-04 確認可以接受。"""
        events = []
        if river_obs is None:
            return events
        for seat in self.seats:
            part = river_obs.get(seat)
            if part is None or part.get("tiles") is None:
                continue
            if seat in frozen:
                self.stats["banner_hold"] += 1
                continue
            dets = []
            for d in part["tiles"]:
                if d.get("conf", 1) < MIN_DET_CONF or is_hover_tip(seat, d) \
                        or is_sliver(d) or is_bottom_clip(seat, d, part.get("zone")):
                    continue
                dd = dict(d)
                dd["box"] = tuple(d["box"])
                dd["center"] = tuple(d["center"])
                dd["seat"] = seat
                dd["t_ms"] = t_ms
                dets.append(dd)
            items = order_frame(seat, dets)
            self.visible[seat] = len(items)
            released_center = self.gate.observe(seat, len(items))
            marks = self.tracker.update(seat, [d for d, *_ in items], frame)
            # 被鳴走的牌的殘影一不見就封存:留著的話,這一家下一張打到同一格
            # (GONE_FRAMES 內)會被配回殘影,不會變成新格子(selfplay-20261102-155238
            # 東4局4本場:下家碰白,上家接著打 3s 到同一格,整串錯一位)
            seen_now = {id(m[0]) for m in marks}
            for sl in list(self.tracker.live[seat]):
                if sl.get("claim_ghost") and id(sl) not in seen_now:
                    self._archive(seat, sl, frame, "claim_ghost")

            # --- (b) 張數掉了:Gate 放掉的那個位置,tracker 那一格也封存 --
            if released_center is not None:
                matched = {id(m[0]) for m in marks}
                hit = None
                for sl in list(self.tracker.live[seat]):
                    if (sl.get("said") and id(sl) not in matched
                            and _near(sl["center"], released_center,
                                      self.match_dist)):
                        hit = sl
                        break
                if hit is not None:
                    events.append(self._archive(seat, hit, frame,
                                                "count_drop"))
                    self.stats["release_count_drop"] += 1
                else:
                    # ★ 放錯人了:那個位置這一幀還看得到牌(或根本沒有輸出過
                    #   的格子在那裡)。最常見的是「被鳴走的牌還沒定案就消失」
                    #   -> 張數掉落 pop 到的是**上一張**已輸出的位置。放回去,
                    #   不然上一張被重新偵測時會變成一次假打牌。
                    self.gate.said[seat].append(released_center)
                    self.stats["release_undone"] += 1

            # --- 新格子 -> 候選 ------------------------------------------
            for slot, d, kind in marks:
                if slot["n"] == 1 and kind is None and "slot_id" not in slot:
                    ghost = self._resurrected(seat, slot, frame)
                    self._assign_id(seat, slot)
                    if ghost is not None:
                        # 同一張牌被擋了幾幀又出現,不是新打牌
                        slot["dahai"] = False
                        slot["said"] = True
                        slot["answer"] = ghost.get("answer")
                        slot["ghost_of"] = ghost.get("slot_id")
                        if ghost.get("resurrected_from") == "claim":
                            # 被鳴走的牌還沒消失:位置已經放掉,別再佔住,
                            # 不然這一家下一張打到同一格會被擋
                            slot["claim_ghost"] = True
                            self.stats["claimed_lingering"] += 1
                            continue
                        self.gate.note_said(seat, slot["center"])
                        self.stats["resurrected"] += 1
                        continue
                    slot["dahai"] = self.gate.is_dahai(seat, slot["center"])
                    if not slot["dahai"]:
                        self.stats["gate_blocked"] += 1
            # 看滿 CAND_MIN_FRAMES 幀才送候選(濾掉飛行動畫中途的那一格)
            ready = [slot for slot, d, kind in marks
                     if (slot.get("dahai") and not slot.get("cand_sent")
                         and slot["n"] >= self.cand_min_frames)]
            if len(ready) > 1 and self.last_cand.get(seat) is not None:
                # ★ 同一家同一幀冒出兩張新牌:一次只會打一張。2026-10-27 實機
                #   H003:P2 立直那張落下的同時,P3 牌河的一張被畫進 P2 的區域
                #   (錄製時自動擴張的牌河區重疊)-> 兩個候選 -> EVENT_GAP。
                #   留下離這一家上一張最近的(牌河是一張接一張排的),其他的當成
                #   不是這一家的牌,不送。
                lc = self.last_cand[seat]
                ready.sort(key=lambda sl: abs(sl["center"][0] - lc[0])
                           + abs(sl["center"][1] - lc[1]))
                for sl in ready[1:]:
                    sl["dahai"] = False
                    sl["intruder"] = True
                    self.stats["simultaneous_dropped"] += 1
                ready = ready[:1]
            for slot in ready:
                if True:
                    slot["cand_sent"] = True
                    self.last_cand[seat] = tuple(slot["center"])
                    events.append(RiverCandidate(
                        seat, slot["slot_id"], slot["generation"],
                        slot["first"],
                        slot.get("first_ms") if slot.get("first_ms")
                        is not None else t_ms,
                        tuple(slot["center"])))
                    self.stats["candidate"] += 1

            # --- 定案 -> 補牌面 ------------------------------------------
            # ★ 走 live 而不是只走 marks:強制定案的格子這一幀可能已經不在
            #   畫面上(被鳴走了),只看 marks 會漏掉(4_tsumogiri 實測踩過)。
            for slot in list(self.tracker.live[seat]):
                if (slot.get("dahai") and slot.get("cand_sent")
                        and slot.get("answer") is not None
                        and not slot.get("said")):
                    events.append(self._confirm_event(seat, slot, frame))
                    self.stats["confirmed"] += 1
        return events

    # ------------------------------------------------------------------
    def find(self, slot_id):
        return self.by_id.get(slot_id)

    def claim(self, seat, slot_id, generation, frame, expected_pai=None):
        """(a) 狀態機確認吃/碰/大明槓:來源那一格先補定案,再兩層一起封存。

        順序(v6.2 第 4 節):核對身分 -> 補回牌面 -> 封存 tracker -> 釋放 Gate。
        回傳 ClaimResult;身分對不上就拒絕,什麼都不動。
        """
        slot = self.by_id.get(slot_id)
        if slot is None:
            return ClaimResult(False, f"沒有這一格 {slot_id}")
        if slot.get("generation") != generation:
            return ClaimResult(False, f"{slot_id} 的 generation 是 "
                                      f"{slot.get('generation')},不是 "
                                      f"{generation}")
        if slot.get("released") == "claim":
            return ClaimResult(True, "已經處理過")
        out = []
        # 1. 補回牌面:不足 8 幀就用現有票數強制定案
        if slot.get("answer") is None and slot.get("head"):
            slot["answer"] = Counter(slot["head"]).most_common(1)[0][0]
            slot["answer_frame"] = frame
            slot["forced"] = slot["n"]
            self.stats["force_confirm"] += 1
        if slot.get("answer") is not None and not slot.get("said"):
            out.append(self._confirm_event(seat, slot, frame))
        # 2+3. 兩層一起封存
        if slot in self.tracker.live[seat]:
            out.append(self._archive(seat, slot, frame, "claim"))
        else:
            slot["released"] = "claim"
            out.append(RiverReleased(seat, slot_id, generation, frame,
                                     "claim"))
        if not self.gate.release_at(seat, slot["center"]):
            # Gate 已經因為張數掉落 pop 過了(備援路徑先到)——正常,不用再動
            self.stats["claim_after_drop"] += 1
        self.stats["claim"] += 1
        return ClaimResult(True, "ok", out)
