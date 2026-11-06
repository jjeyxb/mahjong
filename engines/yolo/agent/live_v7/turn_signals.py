"""摸牌訊號與切法(摸切/手切)判斷。

重用:
    drawn.DrawnWatch / judge   P1~P3 的摸入牌追蹤與摸切判斷(第四步的規則)
    hand.pick_p0_drawn / concealed_of / name_of_drawn / judge_dahai
                               P0 的摸入牌與手牌差分(第五步的規則)

=== 輸出 ===
    DrawSeen(seat)       這一家的摸入牌「出現了」(缺席一陣子之後連續看到)
                         ★ 只是訊號。是不是一次真的摸牌由狀態機決定:
                           手切之後摸入牌還在原位那幾幀、閃爍,都會產生雜訊,
                           狀態機只接受「該輪到他」的那一家。
    DrawName(P0, pai)    P0 摸到的牌面(手牌模型在摸入牌位置的多數決)
    CutVerdict(slot_id)  某一次打牌的切法

=== 切法的兩套方法(使用者定的,不混用)===
P1~P3  牌河出現新牌、且該家沒有手臂/立直動畫的那一幀(判斷點),往後看
       DRAWN_WINDOW 幀,摸入牌還在 -> 手切,不在 -> 摸切。吃/碰後必然手切。
P0     打牌前後兩份「穩定的暗牌」比對:內容一樣 -> 摸切,不一樣 -> 手切。
       吃/碰後必然手切。暗牌 = 手牌模型的輸出扣掉摸入牌那一張
       (手牌模型會連摸入牌一起讀到,不扣的話會重複計數)。
       等不到打牌後的穩定手牌時,退回 P1~P3 那一套,並標記 fallback。
"""
from collections import Counter, deque
from dataclasses import dataclass

import _paths  # noqa: F401
from drawn import (ARM_CONF, DRAWN_CONF, DRAWN_WINDOW, REACH_ANIM_CONF,
                   TEDASHI, TSUMOGIRI, DrawnWatch, judge)
from event_zones import classify
from tiles import base_pai
from drawn_lib import apply_hand_fix
from hand import (concealed_of, diff, judge_dahai, match_drawn, name_of_drawn,
                  pick_p0_drawn)

SEATS = ("P0", "P1", "P2", "P3")
OPP = ("P1", "P2", "P3")

# 摸入牌「出現」的判準:缺席至少 ABSENT_MIN 幀之後,連續 PRESENT_MIN 幀看到。
ABSENT_MIN = 3
PRESENT_MIN = 2
# P1~P3:新牌出現後一直被手臂擋住,最多等幾幀就硬判(並記 arm_blocked)。
MAX_BLOCK_WAIT = 8
# P0:暗牌要連續幾幀內容完全一樣才算穩定。
STABLE_FRAMES = 3
# P0:打牌後最多等幾幀穩定手牌,等不到就退回摸入牌判斷。
P0_MAX_WAIT = 45
# P0 摸入牌名字的最少票數。
# ★ 2026-10-26 實機:手牌模型每 2 幀跑一次、摸入牌只停留約 4 幀 -> 最多 2 票。
#   原本要 3 票,P0 的摸牌牌面 10 筆讀不到。改 2 票;摸入牌消失時還不到 2 票
#   就用現有的 1 票(之後還有牌河/手牌差分兩個獨立來源可以對)。
NAME_VOTES = 2
# ★ 2026-10-27 實機 H005:P0 摸 9m,手牌模型在摸入牌位置讀成 E(0.56)兩次 ->
#   tsumo 送了 E,手牌帳少一張 9m,後面打 9m 被守門器擋下。低信心的牌名不投票
#   (摸入牌通常會停好幾幀,等得到高信心的讀數;等不到就由打牌後的手牌差分補)。
NAME_CONF = 0.70
REACH_LOOKBACK = 20    # 打牌候選前這麼多幀內有這一家的立直橫幅 -> 宣言牌
REACH_AFTER = 2        # 橫幅結束後看幾幀摸入牌


@dataclass
class DrawSeen:
    seat: str
    frame: int
    t_ms: float
    kind: str = "draw_seen"


@dataclass
class DrawName:
    seat: str
    frame: int
    pai: str
    votes: int
    kind: str = "draw_name"


@dataclass
class CutVerdict:
    seat: str
    slot_id: str
    frame: int
    tsumogiri: bool
    why: str
    method: str          # drawn_position / p0_hand_diff / p0_fallback
    inferred: bool       # True = 沒有直接證據、依規則推的
    pai_guess: object = None
    drawn_guess: object = None   # P0 手切時「手上多出來的那張」= 摸到的牌
    kind: str = "cut"
    pai_conf: object = None      # 手牌差分那張牌在打牌前暗牌裡的信心
    missing: object = None       # P0 手牌差分「少了」的牌(差分有歧義時有好幾張)


# 樣板比對(drawn_lib.py)的門檻。2026-10-28 用 runs\20261027_181804 量:
# 分數 >= 0.80 而且贏第二名 0.05 以上,全部讀對;再低就開始出現 5m/3m 混淆。
# 兩檔:(分數 >= 0.80、領先 >= 0.05)或(>= 0.75、領先 >= 0.10)。
#   這場 121 個摸入牌畫面,兩檔合起來採用 92 次、0 次錯;0.70/0.10 也 0 錯,留餘裕。
LIB_SOLO = 0.80
LIB_SOLO_MARGIN = 0.05
LIB_SOLO2 = 0.75
LIB_SOLO2_MARGIN = 0.10
LIB_AGREE = 0.65
WEAK_VOTES = 6         # 手牌每 2 幀跑一次 -> 約 12 幀 = 2 秒
WEAK_HAND_CONF = 0.40
WEAK_LIB = 0.60
WEAK_LIB_MARGIN = 0.15       # 手牌模型低信心、樣板也指向同一張 -> 兩個弱證據一致


def drawn_vote(hit, lib):
    """摸入牌這一幀投哪一張(或不投)。
        手牌模型 >= NAME_CONF                     -> 手牌模型
        樣板比對 >= LIB_SOLO 且領先 >= MARGIN       -> 樣板(五要手牌模型同意:赤五)
        手牌模型有框但低信心、樣板同一張 >= LIB_AGREE -> 那一張"""
    hm = hit.get("pai") if hit else None
    hc = hit.get("conf", 0) if hit else 0
    if hm and hc >= NAME_CONF:
        return hm
    if not lib or not lib.get("pai"):
        return None
    lp, ls = lib["pai"], lib.get("score", 0)
    margin = ls - (lib.get("second_score") or 0)
    five = base_pai(lp)[0] == "5"
    if hm and base_pai(hm) == base_pai(lp) and ls >= LIB_AGREE:
        return lp if (not five or lib.get("red_checked")) else hm
    strong = (ls >= LIB_SOLO and margin >= LIB_SOLO_MARGIN) or \
        (ls >= LIB_SOLO2 and margin >= LIB_SOLO2_MARGIN)
    if strong and (not five or lib.get("red_checked")):
        return lp            # 五:赤不赤已經用顏色判過(drawn_lib.is_red_five)
    return None


def _separated_right(hit, hand_dets, pitch=94.0):
    """摸入牌一定在最右邊、跟暗牌隔一段空隙。
    ★ 2026-10-27 實機 H002 f1138:手切後理牌動畫中,局勢模型的 drawn_tile 框
      跳到最左邊那張(3m)上 -> 牌名投給 3m,實際摸的是 5mr。不在最右邊、或跟
      左邊那張沒有空隙的,不投票。"""
    others = [d["center"][0] for d in hand_dets if d is not hit]
    if not others:
        return True
    return hit["center"][0] - max(others) >= 1.15 * pitch


def _apart_from_row(det, hand_x, pitch=94.0):
    """局勢模型的摸入牌框,跟上一份手牌讀數比:在最右邊、跟暗牌隔一段空隙才算。

    ★ 2026-11-06 錄影 selfplay-20261106-103105 東3局1本場 f4192~f4272:打出最右邊的東,
      摸到的 5s 補進東的位置;滑鼠還停在那裡,那張牌浮起來(y 989 -> 944),局勢模型
      把它當成摸入牌(信心多半 0.10~0.18,剛點完 0.7~0.9)。暗牌扣掉它只剩 12 張,
      打牌前後兩份都少同一張 -> 「內容一樣」-> 判成摸切,牌河那張變成摸到的牌。
      十段錄影 P0「12 張沒變」判摸切 85 次錯、1 次對。真人打牌滑鼠也會停在牌上。
      牌名投票早就有同一條(_separated_right),這裡補在「哪一張是摸入牌」本身。"""
    if not hand_x:
        return True
    x = det["center"][0]
    others = [h for h in hand_x if abs(h - x) > pitch / 2]
    return not others or x - max(others) >= 1.15 * pitch


class _Edge:
    def __init__(self):
        self.present = 0
        self.absent = ABSENT_MIN      # 開局視為「已經缺席夠久」
        self.armed = True

    def step(self, now):
        fired = False
        if now:
            self.present += 1
            if self.armed and self.present >= PRESENT_MIN:
                fired = True
                self.armed = False
            self.absent = 0
        else:
            self.absent += 1
            self.present = 0
            if self.absent >= ABSENT_MIN:
                self.armed = True
        return fired


class TurnSignals:
    def __init__(self):
        self.watch = DrawnWatch(OPP)
        self.edge = {s: _Edge() for s in SEATS}
        self.p0_drawn = None            # 這一幀 P0 的摸入牌偵測(局勢模型)
        self.p0_name = Counter()        # 這一次摸牌的牌名票
        self.p0_name_sent = False
        self.pending = {}               # slot_id -> 等切法的打牌
        # P0 手牌
        self.p0_hist = deque(maxlen=STABLE_FRAMES)
        self.p0_stable = []             # [(first_frame, last_frame, tiles)]
        self.p0_stable_drawn = {}       # first_frame -> 那時摸入牌在不在
        self.p0_draw_frame = None       # P0 最近一次摸牌(摸入牌出現)的幀
        self.p0_conf_by_tiles = {}      # 暗牌(tuple) -> {牌: 信心}
        self.p0_weak_hand = Counter()   # 低信心但一直讀到的牌名(這一次摸牌)
        self.p0_weak_lib = Counter()
        self.p0_with_drawn = None       # 摸入牌還在時的穩定暗牌(打牌前基準)
        self.p0_hand_x = None           # 上一份手牌讀數每張的 x(認摸入牌用)
        self.forced = {s: False for s in SEATS}
        self.pai_of_slot = None         # slot_id -> 牌河定案的牌面(pipeline 設)
        self.drawn_hist = {s: deque(maxlen=60) for s in OPP}  # (frame, 在不在)
        self.reach_last = {}            # 這一家立直橫幅最後出現的那一幀

    # ------------------------------------------------------------------
    def new_hand(self):
        self.watch.new_hand()
        self.edge = {s: _Edge() for s in SEATS}
        self.p0_name = Counter()
        self.p0_name_sent = False
        self.pending = {}
        self.p0_hist.clear()
        self.p0_stable = []
        self.p0_stable_drawn = {}
        self.p0_with_drawn = None
        self.p0_draw_frame = None
        self.p0_hand_x = None
        self.forced = {s: False for s in SEATS}
        self.drawn_hist = {s: deque(maxlen=60) for s in OPP}
        self.reach_last = {}

    def _weak_vote(self, hit, lib):
        """一幀不夠有把握,但很多幀都指向同一張 -> 也算數。
        ★ 2026-10-29 實機:你等建議時摸入牌一直停在那裡(幾十幀),手牌模型
          每一幀都讀 1m(0.58)、或樣板每一幀都說 1s(0.66,領先 0.19),
          但沒有一幀過門檻 -> 牌名一直出不來 -> Mortal 那巡沒有建議。
            手牌模型 >= WEAK_HAND_CONF 的讀數累積 WEAK_VOTES 幀,而且樣板沒有
            強烈反對(樣板指向別張、分數 >= LIB_SOLO2) -> 那一張
            樣板 >= WEAK_LIB 且領先 >= WEAK_LIB_MARGIN 累積 WEAK_VOTES 幀,
            而且手牌模型沒有一再讀成別張 -> 樣板那張(五要判過赤)"""
        if hit and hit.get("conf", 0) >= WEAK_HAND_CONF:
            self.p0_weak_hand[hit["pai"]] += 1
        if lib and lib.get("pai") and lib.get("score", 0) >= WEAK_LIB and \
                lib["score"] - (lib.get("second_score") or 0) >= \
                WEAK_LIB_MARGIN and (base_pai(lib["pai"])[0] != "5"
                                     or lib.get("red_checked")):
            self.p0_weak_lib[lib["pai"]] += 1
        lib_top = lib.get("pai") if lib else None
        lib_strong = lib and lib.get("score", 0) >= LIB_SOLO2
        if self.p0_weak_hand:
            p, n = self.p0_weak_hand.most_common(1)[0]
            if n >= WEAK_VOTES and not (lib_strong and base_pai(lib_top)
                                        != base_pai(p)):
                return p
        if self.p0_weak_lib:
            p, n = self.p0_weak_lib.most_common(1)[0]
            others = sum(v for k, v in self.p0_weak_hand.items()
                         if base_pai(k) != base_pai(p))
            if n >= WEAK_VOTES and others < 3:
                return p
        return None

    def note_call(self, seat, cls):
        """狀態機確認吃/碰 -> 這一家下一打必然手切。"""
        if cls in ("chi", "pon"):
            self.forced[seat] = True
            self.watch.note_naki(seat, cls)

    def note_kan(self, seat, frame):
        if seat in OPP:
            self.watch.note_kan(seat, frame)

    def note_reach(self, seat):
        if seat in OPP:
            self.watch.note_reach(seat)

    def p0_tehai_snapshot(self):
        """這一局最早的穩定暗牌(給 start_kyoku 的 tehais 用)。"""
        return self.p0_stable[0] if self.p0_stable else None

    def p0_drawn_present(self):
        return self.p0_drawn is not None

    # ------------------------------------------------------------------
    def feed(self, frame, t_ms, region_dets, p0_hand):
        """每幀先呼叫:更新摸入牌/手臂/手牌狀態。回傳 DrawSeen / DrawName。
        切法結論在 judge() 才下(要等牌河候選登記)。"""
        out = []
        if region_dets is not None:
            dets = [dict(d, box=tuple(d["box"]), center=tuple(d["center"]))
                    for d in region_dets]
            arms = [d for d in dets if d["cls"] == "arm"
                    and d["conf"] >= ARM_CONF]
            self.watch.observe_arm(arms)
            reaches = [(d, classify(d["center"][0], d["center"][1],
                                    "reach")[0])
                       for d in dets if d["cls"] == "reach"
                       and d["conf"] >= REACH_ANIM_CONF]
            self.watch.observe_reach_anim(reaches)
            for _d, rs in reaches:
                if rs in OPP:
                    self.reach_last[rs] = frame
            # ★ 錄製端用較低的門檻(0.10)全部記下來,這裡套第四步量過的
            #   drawn_tile 門檻 —— 跟實機舊程式同一個值。
            drawn_dets = [d for d in dets if d["cls"] == "drawn_tile"
                          and d["conf"] >= DRAWN_CONF]
            self.watch.observe(drawn_dets, frame)
            was = self.p0_drawn is not None
            self.p0_drawn = pick_p0_drawn(
                [d for d in drawn_dets if _apart_from_row(d, self.p0_hand_x)])
            if self.p0_drawn is not None and not was:
                self.p0_weak_hand = Counter()
                self.p0_weak_lib = Counter()
                # 摸入牌剛出現 = 新的一次摸牌:牌名票從這一幀開始算。
                # ★ 原本在「確認出現」(PRESENT_MIN 幀後)才清票,第一幀投的票會
                #   被清掉(2026-10-27 實機 H007:莊家第 14 張 F 只在第一幀讀到)
                self.p0_name = Counter()
                self.p0_name_sent = False
            for s in OPP:
                self.drawn_hist[s].append((frame, bool(self.watch.drawn_now[s])))
                if self.edge[s].step(self.watch.drawn_now[s]):
                    out.append(DrawSeen(s, frame, t_ms))
            if (self.p0_drawn is None and self.p0_name
                    and not self.p0_name_sent):
                top, n = self.p0_name.most_common(1)[0]
                self.p0_name_sent = True
                out.append(DrawName("P0", frame, top, n))
            if self.edge["P0"].step(self.p0_drawn is not None):
                out.append(DrawSeen("P0", frame, t_ms))
                self.p0_draw_frame = frame - 1   # PRESENT_MIN 幀前就出現了
                self.p0_with_drawn = None

        # --- P0 手牌:暗牌穩定快照、摸入牌牌名 -----------------------
        if p0_hand is not None and p0_hand.get("tiles") is not None:
            hd = [dict(d, box=tuple(d["box"]), center=tuple(d["center"]))
                  for d in p0_hand["tiles"]]
            # 暗牌的樣板比對修正(drawn_lib.fix_hand,錄製端算好的)
            hd = apply_hand_fix(hd, p0_hand.get("hand_fix"))
            self.p0_hand_x = [d["center"][0] for d in hd]
            if self.p0_drawn is not None:
                hit = match_drawn(hd, self.p0_drawn)
                if hit is not None and not _separated_right(hit, hd):
                    hit = None
                lib = p0_hand.get("drawn_lib")
                nm = drawn_vote(hit, lib)
                if nm is None and not self.p0_name_sent:
                    nm = self._weak_vote(hit, lib)
                if nm:
                    self.p0_name[nm] += 1
                    top, n = self.p0_name.most_common(1)[0]
                    if n >= NAME_VOTES and not self.p0_name_sent:
                        self.p0_name_sent = True
                        out.append(DrawName("P0", frame, top, n))
            conc = concealed_of(hd, self.p0_drawn)
            tiles = tuple(sorted(d["pai"] for d in conc))
            cm = {}
            for d in conc:
                cm[d["pai"]] = min(cm.get(d["pai"], 1.0), d.get("conf", 1.0))
            self.p0_conf_by_tiles[tiles] = cm     # 這份暗牌裡每種牌最低的信心
            if len(self.p0_conf_by_tiles) > 200:
                self.p0_conf_by_tiles.pop(next(iter(self.p0_conf_by_tiles)))
            self.p0_hist.append(tiles)
            if (len(self.p0_hist) == STABLE_FRAMES
                    and len(set(self.p0_hist)) == 1):
                if self.p0_stable and self.p0_stable[-1][2] == tiles:
                    f0, _f1, t = self.p0_stable[-1]
                    self.p0_stable[-1] = (f0, frame, t)
                else:
                    self.p0_stable.append(
                        (frame - STABLE_FRAMES + 1, frame, tiles))
                    # 這一份穩定暗牌開始時,摸入牌還分開擺著嗎(起手牌倒推要用)
                    self.p0_stable_drawn[frame - STABLE_FRAMES + 1] = \
                        self.p0_drawn is not None
                    del self.p0_stable[:-40]
                if self.p0_drawn is not None:
                    self.p0_with_drawn = tiles
        return out

    def judge(self, frame):
        """每幀呼叫(在牌河候選登記之後):收證據、到時候就下切法結論。"""
        return self._judge_pending(frame)

    # ------------------------------------------------------------------
    def on_candidate(self, seat, slot_id, frame, first_of_hand=False):
        """牌河轉接層送來一個打牌候選 -> 開始收切法證據。"""
        p = {"seat": seat, "slot_id": slot_id, "first": frame, "ev": None,
             "forced": self.forced[seat],
             "p0_drawn_at": self.p0_drawn is not None,
             "p0_drawn_name": (self.p0_name.most_common(1)[0][0]
                               if self.p0_name else None),
             "first_of_hand": first_of_hand}
        rl = self.reach_last.get(seat)
        if seat in OPP and rl is not None and frame - rl <= REACH_LOOKBACK:
            p["reach_end"] = rl
        if seat == "P0":
            # ★ 打牌前的基準 = 「摸入牌還在」時的穩定暗牌。
            #   2026-10-26 實機 f432~f442:P0 摸 4p 手切 6s,理牌(f436)比牌河
            #   出現新牌(f439)還早。拿「牌河新牌之前」最後一份穩定暗牌當基準,
            #   拿到的其實是理牌後的手牌 -> 前後一樣 -> 判成摸切。
            #   吃碰後那一打沒有摸入牌,才退回「打牌之前」那一份。
            # ★ 2026-10-27 實機 H006/H007:摸入牌那段沒有穩定讀數(換局剛開始、
            #   手牌模型漏一張),而理牌又比牌河新牌早 -> 「打牌前最後一份」其實
            #   是打完理好的手牌 -> 前後一樣 -> 判成摸切、牌名變成摸到的那張。
            #   摸入牌出現之後才開始的那幾份不能當打牌前基準。
            lim = frame if self.p0_draw_frame is None else \
                min(frame, self.p0_draw_frame)
            p["prev_lim"] = lim
            p["prev"] = self.p0_with_drawn or self._p0_before(lim)
        self.pending[slot_id] = p
        self.forced[seat] = False

    def cancel(self, slot_id):
        self.pending.pop(slot_id, None)

    def _p0_before(self, frame):
        # 打牌之前就開始的最後一段(摸切時這一段會一路延續過打牌那一幀)
        for f0, f1, t in reversed(self.p0_stable):
            if f0 < frame:
                return t
        return None

    def _p0_after(self, frame):
        """打牌之後第一份穩定暗牌。

        ★ 摸切時暗牌前後完全一樣,穩定快照會是**同一段**一路延續過打牌那一幀
          (f0 <= frame)。所以判準不是「這一段在打牌之後才開始」,而是「打牌
          之後這一段至少又穩定了 STABLE_FRAMES 幀」。
        """
        for f0, f1, t in self.p0_stable:
            if f1 >= frame + STABLE_FRAMES:
                return t
        return None

    def _judge_pending(self, frame):
        out = []
        for sid, p in list(self.pending.items()):
            seat = p["seat"]
            if seat == "P0":
                v = self._judge_p0(p, frame)
            else:
                v = self._judge_opp(p, frame)
            if v is not None:
                out.append(v)
                del self.pending[sid]
        return out

    def _judge_reach_decl(self, p, frame):
        """立直宣言牌:判斷點改成「立直橫幅結束後」。

        ★ 2026-10-27 實機(H002 P3、H003 P2,牌譜確認都是手切):立直特效讓
          牌河那張晚 3~5 幀才被偵測到,到那時摸入牌已經理進手牌 -> 判成摸切。
          畫面上實際是:橫幅結束時宣言牌已經落在牌河,摸入牌還在原位
          (f1006、f1559/1560)。所以看橫幅結束後 REACH_AFTER 幀內摸入牌還在不在。
        ⚠️ 這場沒有對手「立直摸切」的例子,摸切那一邊是照規則推的,還沒驗證。"""
        s, re_ = p["seat"], p["reach_end"]
        if frame < re_ + REACH_AFTER:
            return None
        seen = [on for f, on in self.drawn_hist[s]
                if re_ < f <= re_ + REACH_AFTER]
        present = any(seen)
        self.watch.consume(s)
        return CutVerdict(s, p["slot_id"], frame, not present,
                          f"立直宣言牌:橫幅結束後 {REACH_AFTER} 幀內摸入牌"
                          + ("還在 -> 手切" if present else "不在 -> 摸切"),
                          "reach_decl_drawn", not seen)

    def _judge_opp(self, p, frame):
        s = p["seat"]
        if p.get("reach_end") is not None and not p["forced"]:
            return self._judge_reach_decl(p, frame)
        w = self.watch
        if p["ev"] is None:
            waited = frame - p["first"]
            if not w.blocked(s) or waited >= MAX_BLOCK_WAIT:
                ev = w.evidence(s, frame)
                ev["arm_blocked_final"] = w.blocked(s)
                ev["drawn_win"] = 1 if w.drawn_now[s] else 0
                ev["win_until"] = frame + DRAWN_WINDOW
                p["ev"] = ev
            return None
        ev = p["ev"]
        if frame <= ev["win_until"]:
            if w.drawn_now[s]:
                ev["drawn_win"] += 1
                ev["drawn_now"] = True
            return None
        forced = p["forced"] or ev["forced_tedashi"]
        verdict, why = judge(ev["drawn_now"], ev["drawn_seen"], forced)
        if p.get("first_of_hand") and not forced and not ev["drawn_seen"] \
                and not ev["drawn_now"]:
            # ★ 2026-10-30 實戰(東1 P2、東2 P3,牌譜都是手切):一局的第一打
            #   一定是莊家打的(14 張、開局沒有「摸牌」動作),畫面上沒有摸入牌
            #   紀錄是正常的。14 張裡打第 14 張的機率只有 1/14 -> 判手切。
            verdict = TEDASHI
            why = "莊家第一打(開局 14 張、沒有摸入牌紀錄)-> 13/14 是手切"
        inferred = (not forced and not ev["drawn_seen"]) or \
            ev.get("arm_blocked_final", False)
        w.consume(s)
        return CutVerdict(s, p["slot_id"], frame, verdict == TSUMOGIRI, why,
                          "drawn_position", inferred)

    def _judge_p0(self, p, frame):
        if p.get("prev") is None:
            p["prev"] = self._p0_before(p.get("prev_lim", p["first"]))
        cur = self._p0_after(p["first"])
        naki = "chi" if p["forced"] else None
        if cur is not None and p.get("prev") is not None:
            verdict, pai, why = judge_dahai(list(p["prev"]), list(cur),
                                            naki=naki,
                                            drawn=p.get("p0_drawn_name"))
            if verdict is not None:
                inn = diff(list(cur), list(p["prev"]))
                out = diff(list(p["prev"]), list(cur))
                pc = self.p0_conf_by_tiles.get(tuple(p["prev"]), {}).get(pai)
                return CutVerdict("P0", p["slot_id"], frame,
                                  verdict == TSUMOGIRI, why, "p0_hand_diff",
                                  False, pai,
                                  inn[0] if len(inn) == 1 else None,
                                  pai_conf=pc, missing=out)
        # ★ 2026-10-30 實戰:手牌一直沒有穩定讀數,等了 45 幀(7 秒)才退回 ->
        #   守門器 6 秒就停送。摸入牌牌名跟牌河那張不同種 = 一定手切,不必等。
        drawn0 = p.get("p0_drawn_name")
        river0 = self.pai_of_slot(p["slot_id"]) if self.pai_of_slot else None
        if drawn0 and river0 and base_pai(drawn0) != base_pai(river0) and \
                not p["forced"]:
            return CutVerdict("P0", p["slot_id"], frame, False,
                              f"摸入牌 {drawn0}、牌河 {river0} -> 不同,一定手切",
                              "p0_drawn_vs_river", False, drawn_guess=drawn0)
        if frame - p["first"] < P0_MAX_WAIT:
            return None
        # 等不到:退回摸入牌位置判斷(判斷點那一刻摸入牌還在 -> 手切)
        if p["forced"]:
            return CutVerdict("P0", p["slot_id"], frame, False,
                              "吃/碰之後必然手切", "p0_fallback", False)
        # ★ 實機 runs\20261027_181804 H001:P0 手切之後摸入牌會被理進手牌,
        #   摸入牌位置一樣是空的 -> 「摸入牌不在 = 摸切」對 P0 不成立。
        #   有摸入牌牌名、牌河也定案了:不同種 -> 一定手切;同種 -> 摸切(約定)
        drawn = p.get("p0_drawn_name")
        river = self.pai_of_slot(p["slot_id"]) if self.pai_of_slot else None
        if drawn and river:
            same = base_pai(drawn) == base_pai(river)
            return CutVerdict("P0", p["slot_id"], frame, same,
                              f"等不到打牌後的穩定手牌;摸入牌 {drawn}、牌河 "
                              f"{river} -> " + ("同種,摸切(約定)" if same
                                               else "不同,一定手切"),
                              "p0_drawn_vs_river", same,
                              drawn_guess=drawn)
        if frame - p["first"] < P0_MAX_WAIT * 2 and drawn is None:
            return None             # 再等一下牌河或牌名
        tsumogiri = not p["p0_drawn_at"]
        return CutVerdict("P0", p["slot_id"], frame, tsumogiri,
                          "等不到打牌後的穩定手牌,依判斷點摸入牌"
                          + ("不在" if tsumogiri else "還在")
                          + "(★P0 手切後摸入牌會被理進手牌,這條不可靠)",
                          "p0_fallback", True)
