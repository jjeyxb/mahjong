"""回合狀態機(v7 精簡版 + v7.1 立直規則)。

狀態(majsoul_v7_fsm.csv):
    UNKNOWN / WAIT_ACTION(Px) / WAIT_DISCARD_AFTER_CALL(Px) /
    WAIT_REACTION(Pd, discard) / WAIT_CHANKAN(Px, kan) / WAIT_RINSHAN(Px) /
    WIN_COLLECTION(source) / ROUND_END / RESYNC

這一版實作的範圍(第 3 步):
    ✅ T01 T02 T06 T09 T10   一般摸打、開局首打、中途開始
    ✅ T11 T12 T13           吃/碰/大明槓的回合轉移 + 來源棄牌 CLAIMED + 兩層失效
    ✅ T07 T15 T17           自家槓、嶺上摸牌
    ✅ T08 T14 T16 T18 G01   自摸、榮和(多家收集)、搶槓、流局
    ✅ S01~S05               立直 pending / accepted / cancelled
    ✅ C01 C03 C06 C07       矛盾 -> EVENT_GAP + 以強觀測重新定錨
    ⏳ consumed(吃碰槓用掉哪幾張)、槓型細分、寶牌翻新 —— 第 4 步
       (這些事件會建立,但欄位缺,export_eligible=False,不會被當成完整)

★ 狀態機只吃「已經過濾過的訊號」:牌河候選(DahaiGate 過了)、宣告(段落化
  過了)、摸牌出現(邊緣觸發過了)。它不再投票,也不看模型信心度。
★ 摸牌出現是**弱訊號**:只接受「該輪到他」的那一家。其他家的摸入牌出現
  (手切後摸入牌還在原位、閃爍)只記數,不觸發矛盾。
★ 牌河新牌是**強訊號**:跟預期不符時不丟掉,記 EVENT_GAP 再用它重新定錨。
"""
from collections import Counter

SEATS = ("P0", "P1", "P2", "P3")


def nxt(seat):
    return SEATS[(SEATS.index(seat) + 1) % 4]


# 榮和收集窗口:第一個 ron 之後多久沒有新的 ron 就結束(毫秒)。
# ⚠️ 起始值,要用雙響的錄影量。
WIN_WINDOW_MS = 3000
# ROUND_END 之後四家牌河連續全空幾幀算新的一局。
NEW_HAND_EMPTY = 5
# 不在 ROUND_END 但四家牌河全空這麼久 -> 漏看了終局,強制換局。
# ★ 2026-10-30 實戰:牌河被蓋住 11 秒(70 幀)之後原樣回來 -> 原本 40 幀就當成
#   漏看終局、換局,回來的牌全部變成新局的打牌。換局一律由 pipeline 看「牌河
#   回來時的張數」決定,這條關掉(保留參數,方便之後要再開)。
LOST_END_EMPTY = 10 ** 9
SAME_SUIT_RIVER_CONF = 0.85   # 牌河信心低於這個,才用「同花色」改 P0 打的牌
HAND_OVER_RIVER = 0.60   # P0 打牌:手牌那張信心 >= 這個才蓋過牌河


class State:
    def __init__(self, name, actor=None, ref=None):
        self.name = name
        self.actor = actor        # 輪到誰 / 剛打牌的是誰
        self.ref = ref            # discard event id / kan event id / source

    def __repr__(self):
        a = f"({self.actor}" + (f",{self.ref}" if self.ref else "") + ")" \
            if self.actor or self.ref else ""
        return f"{self.name}{a}"


class RoundFSM:
    """hooks:
        claim(seat, slot_id, generation, frame) -> list of river events
            叫牌河轉接層做兩層失效;回傳的事件(強制定案)由 pipeline 再餵回來
        note_call(seat, cls) / note_kan(seat, frame) / note_reach(seat)
            通知切法判斷那一層
    """

    def __init__(self, ledger, hooks=None, hand_id="H001"):
        self.ledger = ledger
        self.hooks = hooks or {}
        self.hand_id = hand_id
        self.state = State("UNKNOWN")
        self.trace = []
        self.stats = Counter()
        self._reset_hand_vars()

    def _reset_hand_vars(self):
        self.slot_event = {}          # slot_id -> dahai event id
        self.ghost_next = {}          # 被當成飛行中假格子取消的 dahai -> 接手的那一筆
        self.last_discard = None      # 最新一張未被鳴走的 dahai event id
        self.last_p0_tsumo = None
        self.riichi_pending = {}      # seat -> {"reach": eid, "decl": eid}
        self.riichi = set()
        self.pons = {s: [] for s in SEATS}     # 碰過的牌(base)
        self.win_started_ms = None
        self.win_last_ms = None
        self.winners = []
        self._win_target = None
        self.any_event = False
        self.had_discard = False
        self.empty_frames = 0
        self.dealer_guess = None

    # ------------------------------------------------------------------
    def _go(self, name, actor=None, ref=None, why="", frame=None):
        old = repr(self.state)
        self.state = State(name, actor, ref)
        self.trace.append({"frame": frame, "from": old,
                           "to": repr(self.state), "why": why})

    def _hook(self, name, *a):
        f = self.hooks.get(name)
        return f(*a) if f else None

    def _ev(self, type_, t, frame, status="PENDING", **kw):
        self.any_event = True
        return self.ledger.add(type_, self.hand_id, t, frame, status, **kw)

    def _gap(self, t, frame, expected, observed, why):
        self.stats["gap"] += 1
        return self._ev("event_gap", t - 0.2, frame, "CONFIRMED",
                        expected=expected, observed=observed, reason=why,
                        order_bias=-2)

    def _infer_tsumo(self, seat, t, frame, why, origin="NORMAL"):
        """漏看的摸牌。對手的牌面本來就是 ?,所以推出來的摸牌是完整的;
        P0 的牌面不知道 -> pai=None,欄位缺,不會被當成可送出。"""
        self.stats["inferred_tsumo"] += 1
        ev = self._ev("tsumo", t - 0.1, frame, "INFERRED", actor=seat,
                      pai=("?" if seat != "P0" else None), origin=origin,
                      order_bias=-1)
        self.ledger.note(ev["event_id"], why)
        if seat == "P0":
            self.last_p0_tsumo = ev["event_id"]
        return ev

    # ------------------------------------------------------------------
    # 立直
    def _close_reaction(self, t, frame, how):
        """離開 WAIT_REACTION(沒有榮和):宣言牌活下來 -> 立直成立(S03)。"""
        d = self.ledger.get(self.state.ref) if self.state.ref else None
        if d is None:
            return
        seat = d["actor"]
        rp = self.riichi_pending.get(seat)
        if rp and rp.get("decl") == d["event_id"]:
            self._ev("reach_accepted", t - 0.05, frame, "CONFIRMED",
                     actor=seat, reach_id=rp["reach"], via=how,
                     order_bias=-1)
            self.riichi.add(seat)
            del self.riichi_pending[seat]

    def _cancel_riichi_on_ron(self, discard_eid):
        d = self.ledger.get(discard_eid) if discard_eid else None
        if d is None:
            return
        seat = d["actor"]
        rp = self.riichi_pending.get(seat)
        if rp and rp.get("decl") == discard_eid:
            self.ledger.update(rp["reach"], status="CANCELLED",
                               reason="宣言牌被榮和,立直不成立(S04)")
            del self.riichi_pending[seat]

    # ------------------------------------------------------------------
    # 讓 seat 變成「輪到他、已經摸過牌」;必要時補一筆推定的摸牌。
    def _ensure_turn(self, seat, t, frame, why):
        s = self.state
        if s.name == "WAIT_ACTION" and s.actor == seat:
            return True
        if s.name == "WAIT_REACTION" and nxt(s.actor) == seat:
            self._close_reaction(t, frame, "inferred_draw")
            self._infer_tsumo(seat, t, frame, why)
            self._go("WAIT_ACTION", seat, why="推定摸牌:" + why, frame=frame)
            return True
        if s.name in ("WAIT_RINSHAN", "WAIT_CHANKAN") and s.actor == seat:
            self._infer_tsumo(seat, t, frame, why, origin="RINSHAN")
            self._go("WAIT_ACTION", seat, why="推定嶺上摸牌:" + why,
                     frame=frame)
            return True
        if s.name == "UNKNOWN":
            # 開局:莊家起手 14 張,第一個動作前沒有摸牌動畫 -> 莊家的第 14 張
            # 在 mjai 是一筆 tsumo,補上(牌面:對手 ?,P0 待定)
            self.dealer_guess = self.dealer_guess or seat
            self._infer_tsumo(seat, t, frame, "開局莊家第 14 張(推定)")
            self._go("WAIT_ACTION", seat, why="開局:" + why, frame=frame)
            return True
        return False

    # ------------------------------------------------------------------
    def on_candidate(self, c, t, frame):
        """牌河轉接層:RiverCandidate(強訊號)。"""
        s = self.state
        seat = c.seat
        if s.name in ("ROUND_END", "WIN_COLLECTION"):
            # 終局畫面期間冒出來的格子(結算面板的假格子實測過)不算打牌
            self.stats["candidate_after_end"] += 1
            return None
        ok = False
        ghost = None
        if s.name == "WAIT_REACTION" and s.actor == seat:
            prev = self.ledger.get(s.ref)
            if prev is not None and not (prev.get("evidence") or {}).get(
                    "river") and prev.get("call_status") == "UNCLAIMED":
                # ★ 2026-10-30 實戰:P0 打牌飛行中的那張在兩排之間停了 4 幀
                #   (牌河從沒定案)-> 一次假打牌,真的那張落地又一次 -> 原本
                #   EVENT_GAP。同一家緊接著又一張、而且前一張從沒在牌河定案:
                #   前一張是假的,取消,這一張接手。
                self.ledger.update(prev["event_id"], t_ms=t,
                                   reason="飛行中的假格子", status="CANCELLED")
                self.ledger.note(prev["event_id"],
                                 "牌河從沒定案、同一家緊接著又出現一張 -> "
                                 "判定為打牌飛行中的假格子,取消")
                self._hook("cancel_cut", prev.get("slot_id"))
                self.stats["ghost_discard_cancelled"] += 1
                ghost = prev["event_id"]
                self._go("WAIT_ACTION", seat, why="取消假打牌", frame=frame)
                s = self.state
                if prev.get("p0_tsumo"):
                    self.last_p0_tsumo = prev["p0_tsumo"]
                # ★ 2026-11-04 實戰 紀錄\20261104_211402 東2局 f2461:立直宣言牌
                #   先落在這個假格子上 -> 假格子取消後,立直還掛在它身上,
                #   reach_accepted 永遠出不來 -> Mortal 以為你沒立直,之後每巡
                #   次選都寫「立直」。宣言牌改由接手的這一張擔任。
                rp = self.riichi_pending.get(seat)
                if rp and rp.get("decl") == prev["event_id"]:
                    rp["decl"] = None
        if s.name == "WAIT_DISCARD_AFTER_CALL" and s.actor == seat:
            ok = True
        # ★ 漏看鳴牌宣告:先問鳴牌區。下家也要問 —— 2026-10-27 實機 H005:
        #   P1 碰 P0 的 2s(下家碰),沒看到宣告、也沒看到摸牌,原本被當成
        #   「推定摸牌 + 摸切」,少了一個碰。真的摸牌會先走 on_draw_seen,
        #   到這裡還在 WAIT_REACTION 就代表沒看到摸牌。
        if not ok and s.name == "WAIT_REACTION" and seat != s.actor:
            kind = self._hook("infer_call", seat, s.actor, s.ref, frame)
            if kind:
                src = self.ledger.get(s.ref)
                tc = (src["t_first_ms"] + t) / 2 if src else t - 1
                cev = self._on_call(kind, seat, tc, frame)
                if cev is not None:
                    cev["call_frame"] = src["frame_first"] if src else frame
                    self.ledger.update(cev["event_id"], t_ms=t,
                                       reason="推定鳴牌", status="INFERRED")
                    self.ledger.note(cev["event_id"],
                                     f"沒看到{kind}宣告(可能被全螢幕角色動畫"
                                     f"蓋住),但 {seat} 的鳴牌區多了一組、"
                                     f"接著由 {seat} 打牌 -> 推定")
                    self.stats["inferred_call"] += 1
                    s = self.state
                    ok = (s.name == "WAIT_DISCARD_AFTER_CALL" and
                          s.actor == seat) or self._ensure_turn(
                              seat, t, frame, "推定鳴牌後打牌")
        if not ok and self._ensure_turn(seat, t, frame, "看到他的新棄牌"):
            ok = True
        if not ok:
            exp = repr(s)
            self._gap(t, frame, exp, f"{seat} 新棄牌",
                      "C01:棄牌者不符預期;可能漏了打牌或鳴牌,不補猜測的事件")
            self._go("WAIT_ACTION", seat, why="C01 以新棄牌重新定錨",
                     frame=frame)
        self.had_discard = True
        ev = self._ev("dahai", t, frame, actor=seat, pai=None, tsumogiri=None,
                      slot_id=c.slot_id, generation=c.generation,
                      call_status="UNCLAIMED",
                      tedashi_forced=(s.name == "WAIT_DISCARD_AFTER_CALL"))
        self.slot_event[c.slot_id] = ev["event_id"]
        if ghost is not None:
            self.ghost_next[ghost] = ev["event_id"]
        if seat == "P0" and self.last_p0_tsumo:
            ev["p0_tsumo"] = self.last_p0_tsumo
        rp = self.riichi_pending.get(seat)
        if rp and rp.get("decl") is None:
            rp["decl"] = ev["event_id"]
            ev["riichi_declaration"] = True
        self.last_discard = ev["event_id"]
        self._go("WAIT_REACTION", seat, ev["event_id"], "打牌", frame)
        return ev

    def on_confirmed(self, c, t, frame):
        eid = self.slot_event.get(c.slot_id)
        if eid is None:
            return
        cur = self.ledger.get(eid)
        nxt = self.ledger.get(self.ghost_next.get(eid))
        if cur["status"] == "CANCELLED" and nxt is not None \
                and nxt["status"] == "PENDING" and nxt.get("pai") is None:
            # ★ 2026-11-06 錄影 selfplay-20261102-170335 東1局:對家的立直宣言牌(橫放)
            #   先有一格,接著同一家又冒出一格 -> 前一格被當成飛行中的假格子取消。
            #   結果前一格之後在牌河定案(西),接手的那一格永遠沒有牌面,那一打
            #   一直 PENDING,之後整整一輪的事件都對不上輪次。
            #   假格子不會定案;會定案就是真的那張,牌面交給接手的那一筆
            #   (兩筆之間沒有別的事件,順序一樣)。
            self.stats["ghost_confirmed_late"] += 1
            self.ledger.note(nxt["event_id"], f"被取消的 {eid} 之後在牌河定案"
                                              f"({c.pai}),牌面由這一筆接手")
            eid, cur = nxt["event_id"], nxt
        hconf = cur.get("p0_hand_conf")
        if cur.get("p0_hand_pai") and cur["p0_hand_pai"] != c.pai and \
                hconf is not None and c.conf is not None and hconf < c.conf:
            # 手牌差分先到、牌河後定案而且比較有把握 -> 改用牌河(見 _p0_pai_from_hand)
            self.stats["p0_river_vs_hand"] += 1
            self.ledger.note(eid, f"★牌河讀 {c.pai}({c.conf:.2f})、手牌差分是 "
                                  f"{cur['p0_hand_pai']}({hconf:.2f})—— 用牌河")
            cur.pop("p0_hand_pai")
            upd = {"pai": c.pai}
        elif cur.get("p0_hand_pai") and cur["p0_hand_pai"] != c.pai:
            self.stats["p0_river_vs_hand"] += 1
            self.ledger.note(eid, f"★牌河讀 {c.pai}、手牌差分是 "
                                  f"{cur['p0_hand_pai']} —— 用手牌差分")
            upd = {}
        else:
            upd = {"pai": c.pai}
        ev = self.ledger.update(eid, t_ms=t, reason="牌河定案", **upd)
        ev.setdefault("evidence", {})["river"] = {
            "votes": c.votes, "forced": c.forced, "conf": c.conf,
            "sideways": c.sideways}
        if ev.get("actor") == "P0" and not ev.get("p0_hand_pai"):
            self._p0_same_suit(ev, t)
        seat = ev["actor"]
        if c.sideways and not ev.get("riichi_declaration"):
            # 橫放的牌 = 立直宣言牌。漏看了 reach 橫幅 -> 補一筆推定的 reach
            self.stats["inferred_reach"] += 1
            r = self._ev("reach", ev["t_first_ms"] - 0.05, frame, "INFERRED",
                         actor=seat, order_bias=-1)
            self.ledger.note(r["event_id"], "牌河那一格是橫放的(宣言牌),"
                                            "但沒有看到 reach 橫幅")
            ev["riichi_declaration"] = True
            if seat not in self.riichi:
                self.riichi_pending[seat] = {"reach": r["event_id"],
                                             "decl": eid}
                # 這張牌如果已經被下家摸牌帶過去了,立刻結案
                if self.state.name != "WAIT_REACTION" or \
                        self.state.ref != eid:
                    if self.ledger.get(eid).get("call_status") != "RON":
                        self._ev("reach_accepted", t, frame, "INFERRED",
                                 actor=seat, reach_id=r["event_id"])
                        self.riichi.add(seat)
                        del self.riichi_pending[seat]
        self._backfill_p0_tsumo(ev, t)
        # 被吃/碰的牌:鳴牌事件的 pai 也補上
        for k in ev.get("claimed_by", []):
            self.ledger.update(k, t_ms=t, reason="來源棄牌定案", pai=ev["pai"])

    def on_cut(self, v, t, frame):
        eid = self.slot_event.get(v.slot_id)
        if eid is None:
            return
        ev = self.ledger.update(eid, t_ms=t, reason="切法判斷",
                                tsumogiri=v.tsumogiri)
        ev.setdefault("evidence", {})["cut"] = {
            "method": v.method, "why": v.why, "inferred": v.inferred,
            "pai_guess": v.pai_guess}
        if v.inferred:
            ev["tsumogiri_inferred"] = True
        if v.drawn_guess:
            ev["evidence"]["cut"]["drawn_guess"] = v.drawn_guess
        if v.missing:
            ev["evidence"]["cut"]["missing"] = list(v.missing)
        if ev["actor"] == "P0" and v.pai_guess and \
                v.method == "p0_hand_diff":
            # ★ 2026-11-04 實戰 紀錄\20261104_202637 f136:打出北,前後兩份手牌
            #   都漏讀成 12 張 -> 差分說「少了 F」,蓋掉牌河正確的北 -> Mortal
            #   以為北還在手上,整局一直叫你打北。張數不合法的差分不可信,
            #   當成「沒把握」,以牌河為準。
            conf = 0.0 if "不是合法的暗牌" in (v.why or "") else v.pai_conf
            # ★ 2026-11-07 判成摸切時「差分」那張其實是摸入牌的牌名,不是手牌的證據
            #   (錄影 selfplay-20261102-053001 西4局:牌河讀對東 0.95,摸入牌名北
            #   蓋掉了它;113550 南2局1本場 西 -> 白 同樣)。
            if v.tsumogiri:
                conf = 0.0
            self._p0_pai_from_hand(ev, v.pai_guess, t, conf)
        elif ev["actor"] == "P0":
            self._p0_same_suit(ev, t)
        self._backfill_p0_tsumo(ev, t)
        # 免費檢查:立直成立後(宣言牌除外)必然摸切
        if (not v.tsumogiri and ev["actor"] in self.riichi
                and not ev.get("riichi_declaration")):
            self.stats["riichi_tedashi"] += 1
            self.ledger.note(eid, "★立直後判出手切 —— 切法判斷或立直判斷"
                                  "有一個錯了(Q13)")

    def _p0_pai_from_hand(self, ev, pai, t, conf=None):
        """P0 打的牌:手牌差分優先於牌河 —— 前提是手牌那張讀得有把握。
        實機 runs\\20261027_181804:牌河模型把 P0 打的 7p 讀成 6p(0.94)、
        9p 讀成 6p/5p 三次;同一時間手牌模型(牌大、正面)四次都對,信心 0.94~0.98。
        ★ 2026-10-29 實機反例:手牌裡的中只讀到 8m(0.29),打出中之後手牌差分
          說「少了 8m」,牌河讀中才是對的 -> 手牌那張信心低於 HAND_OVER_RIVER
          時不採用,以牌河為準。
        而且 P0 的牌面一定要跟手牌帳一致,否則 libriichi 會 panic。"""
        ev["evidence"]["cut"]["pai_from_hand"] = pai
        ev["evidence"]["cut"]["pai_from_hand_conf"] = conf
        river = ev.get("pai")
        # ★ 2026-11-07 牌河已經定案、而且比手牌那張有把握 -> 用牌河。十段錄影兩邊
        #   不一致 4 次:手牌信心 0.97 > 牌河 0.88 那次手牌對;手牌 0.81 < 0.96、
        #   0.62 < 0.92 那兩次牌河對(selfplay-20261106-113550 東3局、
        #   20261024-123746 東1局)。實機 20261027 的反例(牌河 0.94 讀錯)手牌
        #   0.94~0.98,照樣用手牌。
        rconf = ((ev.get("evidence") or {}).get("river") or {}).get("conf")
        weak = conf is not None and (
            conf < HAND_OVER_RIVER
            or (river is not None and rconf is not None and conf < rconf))
        ev["p0_hand_conf"] = conf
        if weak:
            if river is not None and river != pai:
                self.stats["p0_river_vs_hand"] += 1
                self.ledger.note(ev["event_id"],
                                 f"★牌河讀 {river}、手牌差分是 {pai}"
                                 f"(信心 {conf:.2f} 太低)—— 用牌河")
            return
        ev["p0_hand_pai"] = pai
        if river is not None and river != pai:
            self.stats["p0_river_vs_hand"] += 1
            self.ledger.note(ev["event_id"], f"★牌河讀 {river}、手牌差分是 "
                                             f"{pai} —— 用手牌差分")
        if river != pai:
            self.ledger.update(ev["event_id"], t_ms=t,
                               reason="P0 手牌差分(優先於牌河)", pai=pai)

    def _p0_same_suit(self, ev, t):
        """手牌差分有歧義(少了好幾張)、牌河讀到的那張又不在「少了」裡面:
        「少了」裡面跟牌河同花色的只有一張 -> 就是那張。

        ★ 2026-10-30 實戰東1 f716:暗牌漏讀一張,差分「少了 1m, 9p」;牌河把
          9p 讀成 6p(0.69)—— 手上根本沒有 6p,守門器擋下整局。牌河模型把筒子
          讀錯數字很常見(9p->6p/5p),花色很少錯。"""
        cut = (ev.get("evidence") or {}).get("cut") or {}
        rv = (ev.get("evidence") or {}).get("river") or {}
        if rv.get("conf") is None or rv["conf"] >= SAME_SUIT_RIVER_CONF:
            return          # 牌河還沒定案,或牌河讀得很有把握
        miss = cut.get("missing") or []
        river = ev.get("pai")
        if len(miss) < 2 or not river or river in miss or river[-1] == "r" \
                and river[:2] in miss:
            return
        suit = river[1] if river[0].isdigit() else None
        if suit is None:
            return
        same = sorted({m for m in miss if m[0].isdigit() and m[1] == suit})
        if len(same) != 1:
            return
        self.stats["p0_same_suit_fix"] += 1
        self.ledger.note(ev["event_id"], f"★牌河讀 {river},但手上少的是 "
                                         f"{', '.join(miss)} —— 取同花色的 "
                                         f"{same[0]}")
        ev["p0_hand_pai"] = same[0]
        self.ledger.update(ev["event_id"], t_ms=t,
                           reason="P0 手牌差分(同花色)", pai=same[0])

    def _backfill_p0_tsumo(self, dahai, t):
        """P0 摸到什麼牌,還有兩個獨立來源可以補:
            摸切  牌河那一格就是摸到的那張(牌河是權威)
            手切  打牌前後暗牌比對「多出來的那一張」
        摸入牌位置的手牌模型讀值(DrawName)沒讀到時用;讀到了但不一致就記下來。"""
        tid = dahai.get("p0_tsumo")
        if not tid:
            return
        ts = self.ledger.get(tid)
        if ts is None:
            return
        if dahai.get("tsumogiri") is True and dahai.get("pai"):
            src, pai = "摸切:牌河那張就是摸到的牌", dahai["pai"]
        elif dahai.get("tsumogiri") is False:
            pai = (dahai.get("evidence", {}).get("cut") or {}).get(
                "drawn_guess")
            src = "手切:打牌後暗牌多出來的那張"
        else:
            return
        if not pai:
            return
        if ts.get("pai") is None:
            self.ledger.update(tid, t_ms=t, reason=src, pai=pai)
            ts["pai_source"] = src
        elif ts["pai"] != pai and not ts.get("_mismatch_noted"):
            ts["_mismatch_noted"] = True
            self.stats["p0_tsumo_mismatch"] += 1
            self.ledger.note(tid, f"★摸入牌位置讀到 {ts['pai']},{src} 是 "
                                  f"{pai} —— 兩個模型有一個讀錯")

    # ------------------------------------------------------------------
    def on_draw_seen(self, d, t, frame):
        """摸入牌出現(弱訊號):只接受該輪到的人。"""
        s = self.state
        seat = d.seat
        if s.name == "WAIT_REACTION" and nxt(s.actor) == seat:
            self._close_reaction(t, frame, "draw")
            ev = self._ev("tsumo", t, frame, actor=seat,
                          pai=("?" if seat != "P0" else None))
            if seat == "P0":
                self.last_p0_tsumo = ev["event_id"]
            self._go("WAIT_ACTION", seat, why="T10 下家摸牌", frame=frame)
            return ev
        if s.name in ("WAIT_RINSHAN", "WAIT_CHANKAN") and s.actor == seat:
            if s.name == "WAIT_CHANKAN":
                k = self.ledger.get(s.ref)
                if k:
                    self.ledger.note(s.ref, "看到嶺上摸牌 -> 槓成立、無搶槓")
            ev = self._ev("tsumo", t, frame, actor=seat,
                          pai=("?" if seat != "P0" else None),
                          origin="RINSHAN")
            if seat == "P0":
                self.last_p0_tsumo = ev["event_id"]
            self._go("WAIT_ACTION", seat, why="嶺上摸牌", frame=frame)
            return ev
        if s.name == "UNKNOWN":
            self.dealer_guess = self.dealer_guess or seat
            ev = self._ev("tsumo", t, frame, actor=seat,
                          pai=("?" if seat != "P0" else None))
            if seat == "P0":
                self.last_p0_tsumo = ev["event_id"]
            self.ledger.note(ev["event_id"], "T02:第一個觀測是摸牌(開局莊家"
                                             "第 14 張,或中途開始觀測)")
            self._go("WAIT_ACTION", seat, why="T02", frame=frame)
            return ev
        self.stats[f"draw_ignored:{s.name}"] += 1
        return None

    def on_draw_name(self, d, t, frame):
        if d.seat == "P0" and self.last_p0_tsumo:
            ev = self.ledger.get(self.last_p0_tsumo)
            if ev and ev["pai"] is None:
                self.ledger.update(self.last_p0_tsumo, t_ms=t,
                                   reason="P0 摸入牌牌面", pai=d.pai)

    # ------------------------------------------------------------------
    def on_declaration(self, dec, t, frame):
        c, seat = dec.cls, dec.seat
        if c == "ryukyoku":
            self._ev("ryukyoku", t, frame, "CONFIRMED")
            self._end(t, frame, "G01 流局")
            return
        if c == "reach":
            return self._on_reach(seat, t, frame)
        if c in ("chi", "pon", "kan"):
            return self._on_call(c, seat, t, frame)
        if c == "ron":
            return self._on_ron(seat, t, frame)
        if c == "tsumo":
            return self._on_tsumo_win(seat, t, frame)

    def _on_reach(self, seat, t, frame):
        if seat in self.riichi or seat in self.riichi_pending:
            return
        if not self._ensure_turn(seat, t, frame, "看到立直宣告"):
            self._gap(t, frame, repr(self.state), f"{seat} 立直",
                      "立直宣告者不是當前行動者")
            self._go("WAIT_ACTION", seat, why="以立直宣告重新定錨",
                     frame=frame)
        r = self._ev("reach", t, frame, "CONFIRMED", actor=seat)
        self.riichi_pending[seat] = {"reach": r["event_id"], "decl": None}
        self._hook("note_reach", seat)

    def _on_call(self, c, seat, t, frame):
        s = self.state
        if s.name == "WAIT_ACTION" and s.actor == seat and c == "kan":
            return self._self_kan(seat, t, frame)
        if s.name == "WAIT_REACTION" and seat == s.actor and c != "kan":
            # 「剛打完牌的那一家」宣告吃/碰 —— 吃/碰自己的牌不可能。
            if self._undo_flying_call_tile(seat, t, frame):
                # ★ 2026-11-04 實戰 紀錄\20261104_224154 東3局 f4499~4515:上家打
                #   3m,P0 吃。被吃的那張從上家牌河「橫著」飛過 P0 的牌河區 ->
                #   被當成 P0 打了一張橫的牌(= 立直宣言牌)-> 推定 P0 摸牌 +
                #   推定立直;接著真的「P0 吃」就變成吃自己的牌。那一打的摸牌
                #   從沒看到過 -> 那一打是飛過去的鳴牌,撤掉,回到上家那一打。
                s = self.state
            elif c == "chi":
                # 真的打過牌:宣告橫幅算錯家。吃只有下家能吃 -> 一定是下家
                self.stats["call_by_discarder_fixed"] += 1
                seat = nxt(s.actor)
            else:
                self.stats["call_by_discarder_ignored"] += 1
                return None
        if s.name == "WAIT_REACTION" and seat != s.actor:
            pd = s.actor
            if c == "chi" and seat != nxt(pd):
                # 吃只能吃上家。不是下家 -> 幾乎一定是碰被讀成吃(truth.txt)
                c = "pon"
                self.stats["chi_as_pon"] += 1
            src = self.ledger.get(s.ref)
            typ = {"chi": "chi", "pon": "pon", "kan": "daiminkan"}[c]
            ev = self._ev(typ, t, frame, actor=seat, target=pd,
                          pai=src.get("pai") if src else None,
                          consumed=None, source_discard=s.ref)
            if src is not None:
                # ★ 2026-10-30 實戰:吃的宣告晚到,鳴牌區的新牌組在宣告之前就讀到
                #   了 -> 「鳴牌前」那一份已經含新牌組,相減是空的,consumed 永遠
                #   讀不出來。鳴牌區只可能在來源棄牌之後才變,以它為準。
                ev["call_frame"] = src["frame_first"]
            self.ledger.note(ev["event_id"], "consumed 待第 4 步(鳴牌區)")
            if src is not None:
                src.setdefault("claimed_by", []).append(ev["event_id"])
                self.ledger.update(src["event_id"], t_ms=t,
                                   reason="被鳴走", call_status="CLAIMED")
                self._close_reaction(t, frame, typ)
                for rev in (self._hook("claim", pd, src.get("slot_id"),
                                       src.get("generation"), frame) or []):
                    if getattr(rev, "kind", "") == "confirmed":
                        self.on_confirmed(rev, t, frame)
            if typ == "pon":
                self.pons[seat].append(ev["event_id"])
            self._hook("note_call", seat, c)
            if typ == "daiminkan":
                self._go("WAIT_RINSHAN", seat, ev["event_id"], "T13 大明槓",
                         frame)
            else:
                self._go("WAIT_DISCARD_AFTER_CALL", seat, why=f"T1x {typ}",
                         frame=frame)
            self.last_discard = None
            return ev
        if c == "kan" and self._ensure_turn(seat, t, frame, "看到他宣告槓"):
            return self._self_kan(seat, t, frame)
        # 其他:漏了東西
        self._gap(t, frame, repr(s), f"{seat} {c}",
                  "鳴牌不符預期(來源棄牌不明);保留鳴牌,來源未知")
        typ = {"chi": "chi", "pon": "pon", "kan": "daiminkan"}[c]
        ev = self._ev(typ, t, frame, actor=seat, target=None, pai=None,
                      consumed=None)
        self._hook("note_call", seat, c)
        if typ == "daiminkan":
            self._go("WAIT_RINSHAN", seat, ev["event_id"], "C02 定錨",
                     frame)
        else:
            self._go("WAIT_DISCARD_AFTER_CALL", seat, why="C02 定錨",
                     frame=frame)
        return ev

    def _undo_flying_call_tile(self, seat, t, frame):
        """WAIT_REACTION(seat) 時 seat 自己宣告吃/碰:如果 seat 那一打之前的
        摸牌是推定的(沒看到),那一打就是「被鳴的牌飛過牌河區」的假格子。
        撤掉那一打、那次推定摸牌、跟著推定出來的立直,狀態回到前一家的那一打。
        撤成功回 True。"""
        ghost = self.ledger.get(self.state.ref) if self.state.ref else None
        if ghost is None or ghost.get("type") != "dahai":
            return False
        evs = [e for e in self.ledger.ordered(self.hand_id)
               if e["status"] != "CANCELLED"]
        if ghost not in evs:
            return False
        i = evs.index(ghost)
        tsu = next((e for e in reversed(evs[:i]) if e.get("actor") == seat
                    and e["type"] in ("tsumo", "dahai", "chi", "pon",
                                      "daiminkan")), None)
        if tsu is None or tsu["type"] != "tsumo" or \
                tsu["status"] != "INFERRED":
            return False
        prev = next((e for e in reversed(evs[:i]) if e["type"] == "dahai"
                     and e.get("actor") != seat), None)
        if prev is None:
            return False
        why = "宣告吃/碰的是自己 -> 這一打是被鳴的牌飛過牌河區"
        for e in (ghost, tsu):
            self.ledger.update(e["event_id"], t_ms=t, reason=why,
                               status="CANCELLED")
        self._hook("cancel_cut", ghost.get("slot_id"))
        for e in evs[i - 3 if i >= 3 else 0:]:
            if e.get("actor") == seat and e["type"] in ("reach",
                                                        "reach_accepted") \
                    and e["status"] == "INFERRED" or (
                        e["type"] == "reach_accepted" and e.get("actor") == seat
                        and e.get("reach_id") and (self.ledger.get(
                            e["reach_id"]) or {}).get("status") == "INFERRED"):
                self.ledger.update(e["event_id"], t_ms=t, reason=why,
                                   status="CANCELLED")
        rp = self.riichi_pending.get(seat)
        if rp and rp.get("decl") == ghost["event_id"]:
            self.ledger.update(rp["reach"], t_ms=t, reason=why,
                               status="CANCELLED")
            del self.riichi_pending[seat]
        if seat in self.riichi and not any(
                e["type"] == "reach_accepted" and e.get("actor") == seat
                and e["status"] != "CANCELLED"
                for e in self.ledger.ordered(self.hand_id)):
            self.riichi.discard(seat)
        self.stats["flying_call_tile_undone"] += 1
        self._go("WAIT_REACTION", prev["actor"], prev["event_id"],
                 "撤掉飛過牌河的鳴牌", frame)
        return True

    def _self_kan(self, seat, t, frame):
        # 槓型:有碰過 -> 可能是加槓;牌面要等鳴牌區(第 4 步)才分得出來
        typ = "kan_self"
        ev = self._ev(typ, t, frame, actor=seat, pai=None, consumed=None,
                      kan_type=("kakan_or_ankan" if self.pons[seat]
                                else "ankan"))
        self.ledger.note(ev["event_id"], "槓型與牌面待第 4 步(鳴牌區)")
        self._hook("note_kan", seat, frame)
        self._go("WAIT_CHANKAN", seat, ev["event_id"], "T07 自家槓", frame)
        return ev

    def _on_ron(self, seat, t, frame):
        s = self.state
        if s.name == "WIN_COLLECTION":
            if seat not in self.winners:
                self.winners.append(seat)
                self._ev("hora", t, frame, "CONFIRMED", actor=seat,
                         target=self._win_target, source_id=s.ref)
                self.win_last_ms = t
            return
        if s.name == "WAIT_REACTION" and seat != s.actor:
            target, src = s.actor, s.ref
            self._cancel_riichi_on_ron(src)
            d = self.ledger.get(src)
            if d:
                self.ledger.update(src, t_ms=t, reason="被榮和",
                                   call_status="RON")
        elif s.name == "WAIT_CHANKAN" and seat != s.actor:
            target, src = s.actor, s.ref
            self.ledger.note(src, "被搶槓:不成立,不會有嶺上摸牌")
        else:
            self._gap(t, frame, repr(s), f"{seat} 榮和", "C06:榮和找不到來源")
            target, src = None, None
        self._win_target = target
        self.winners = [seat]
        self.win_started_ms = self.win_last_ms = t
        self._ev("hora", t, frame, "CONFIRMED", actor=seat, target=target,
                 source_id=src)
        self._go("WIN_COLLECTION", None, src, "T14/T16 榮和", frame)

    def _on_tsumo_win(self, seat, t, frame):
        s = self.state
        if not (s.name == "WAIT_ACTION" and s.actor == seat):
            if not self._ensure_turn(seat, t, frame, "看到自摸"):
                self._gap(t, frame, repr(s), f"{seat} 自摸",
                          "自摸者不是當前行動者")
        self._ev("hora", t, frame, "CONFIRMED", actor=seat, target=seat)
        self._end(t, frame, "T08 自摸")

    def _end(self, t, frame, why):
        self._ev("end_kyoku", t + 0.1, frame, "CONFIRMED", order_bias=5)
        self._go("ROUND_END", why=why, frame=frame)

    # ------------------------------------------------------------------
    def tick(self, t, frame, rivers_empty):
        """每幀呼叫:時間窗、換局。回傳 True 代表要換局(pipeline 去重設)。"""
        if self.state.name == "WIN_COLLECTION" and \
                t - self.win_last_ms > WIN_WINDOW_MS:
            self._end(t, frame, "T18 榮和收集窗結束")
        if rivers_empty is None:
            return False
        self.empty_frames = self.empty_frames + 1 if rivers_empty else 0
        # 一般換局由 pipeline 判斷(看牌河回來時的張數);這裡只剩漏看終局
        # ★ 2026-10-29 實機:開局莊家想了十幾秒才打第一張,牌河「本來就」是空的
        #   -> 被當成漏看終局,一直換局,整場停送。要這一局已經有人打過牌才算。
        if (self.state.name not in ("ROUND_END", "UNKNOWN") and self.any_event
                and self.had_discard
                and self.empty_frames >= LOST_END_EMPTY):
            self._gap(t, frame, repr(self.state), "四家牌河長時間全空",
                      "沒有看到終局宣告就換局了(漏看和牌/流局)")
            self._end(t, frame, "漏看終局")
            return True
        return False

    def new_hand(self, hand_id, t, frame):
        self.hand_id = hand_id
        self._reset_hand_vars()
        self._go("UNKNOWN", why="T19 新的一局", frame=frame)
