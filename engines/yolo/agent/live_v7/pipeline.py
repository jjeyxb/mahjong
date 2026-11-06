"""把一幀觀測接成本地事件:轉接層 -> 狀態機 -> 事件帳。

    觀測(一幀)
      ├─ TurnSignals.feed      摸入牌/手臂/P0 手牌狀態 -> DrawSeen / DrawName
      ├─ RiverAdapter.feed     牌河 -> RiverCandidate / RiverConfirmed / Released
      ├─ DeclarationDetector   宣告橫幅 -> Declaration
      ├─ TurnSignals.judge     切法 -> CutVerdict
      ├─ KyokuInfo.feed        場況 -> start_kyoku 欄位、寶牌
      └─ RoundFSM              全部訊號 -> 狀態轉移 + Ledger 事件

同一幀內的處理順序是刻意的:棄牌候選 -> 宣告 -> 摸牌 -> 切法。
(打牌一定先於下家摸牌;鳴牌引用的是剛打出的那張。)

錄影重播(replay.py)與即時錄製(record_obs.py --fsm)走同一個 Pipeline,
只有一份實作。
"""
from collections import Counter

from tiles import base_pai

import _paths  # noqa: F401
from declare import DeclarationDetector
from fsm import RoundFSM
from kyoku_info import KyokuInfo
from ledger import Ledger
from meld_reader import MeldReader
from river_adapter import RiverAdapter, is_bottom_clip, is_hover_tip, is_sliver
from fsm import nxt
from turn_signals import TurnSignals

SEATS = ("P0", "P1", "P2", "P3")

# === 換局判斷(2026-10-27 實機 runs\20261027_181804 之後改寫)===
# 四家牌河全空,有兩種:
#   真的換局   全空 15~20 幀(結算畫面),但九種九牌之後只空 4 幀;
#              回來的時候牌河只有 1 張(新一局第一打)
#   全螢幕特效  角色立繪的鳴牌/和牌動畫把牌河蓋住 7 幀;回來的時候還是原本的 46 張
# 所以看「回來的時候有幾張」,不看空多久:
#   全空 >= EMPTY_MIN 幀,回來時 <= NEW_HAND_MAX_TILES 張、而且比全空之前少
#   (或已經看到終局)-> 換局。「比之前少」:開局第一巡只有 1 張時被遮,
#   回來還是 1 張,不是換局
#   回來時很多張 -> 特效遮擋。全空之前有很多張的話,全空的幀先不餵給牌河
#   轉接層(否則重新出現的 46 張會被當成 46 次打牌);之前只有 1~2 張的話
#   照餵(唯一的那張被鳴走,牌河本來就會全空)
EMPTY_MIN = 3
COUNT_CONF = 0.30    # 判斷「牌河全空/回來幾張」時只算信心 >= 這個的框
# ★ 2026-10-29:牌河只少數幾張回來還不能馬上換局 —— 和牌的全螢幕特效遮住牌河時,
#   中間會有一兩幀只讀到 1 張,接著 46 張全部回來(20261026_232930 f647~654)。
#   所以先看 NH_CONFIRM 幀:一直都只有少數幾張 -> 換局;又冒出一大堆 -> 特效。
NH_CONFIRM = 4
# 余牌對帳(見 Pipeline._wall_kans)。余牌由 MIA 讀好放在觀測的 hud 欄位,YOLO 不讀。
WALL_STABLE = 2          # 同一個讀值連續幾幀才算數
WALL_MAX_STEP = 4        # 一次掉超過這麼多張當成讀錯(稀疏幀最多差一兩巡)
WALL_SETTLE_MS = 2000    # 下一打出現後再等多久才對帳(推定的摸牌要先補上)
WALL_KAN_WINDOW_MS = 20000   # 槓寶牌被讀到之前多久內找槓(寶牌要投票,晚好幾秒)
NH_CANCEL = 6
LATE_DRAW_VOTES = 1     # P0 漏看的摸牌:幾份打牌後的暗牌指向同一張才補(原本就是一份),見 _late_p0_draw
NO_START_EMPTY = 20  # 沒有場況的局:牌河空這麼久再回來才算換局
REPLAY_EMPTY = 6     # 換局時重新餵給新局的「牌河全空」幀數(只取最後幾幀:
                     # 更早的是結算畫面)
NEW_HAND_MAX_TILES = 2
# 宣告橫幅蓋住牌河時,那一家牌河先不更新(river_adapter.feed 的 frozen)。
BANNER_CLS = ("chi", "pon", "kan", "reach", "ron", "tsumo", "ryukyoku")
BANNER_CONF = 0.10       # 橫幅剛滑進來時信心很低,但已經蓋到牌了
BANNER_MIN_AREA = 400    # 跟牌河框重疊至少這麼多 px²(約 20x20)才算蓋住
BANNER_GRACE = 2         # 橫幅消失後再多等幾幀(滑出去的最後一兩幀常常偵測不到)
# ★ 2026-11-04 實戰 紀錄\20261104_195328 f147~148:P0 打牌的手臂伸到 P3 牌河
#   右下角,牌河模型把白色指尖讀成「白」(0.31 -> 0.71)-> P3 多一張打牌 ->
#   EVENT_GAP,整局停送。跟橫幅一樣處理:哪一家牌河「有偵測到的牌」被手臂框
#   蓋住,那一家這一幀先不動,手臂走了再照常讀。
ARM_CONF = 0.30
ARM_TILE_OVERLAP = 0.30  # 牌框有這麼多比例落在手臂框裡,就算被手臂蓋到
ARM_GRACE = 1


class Pipeline:
    def __init__(self, run_id="RUN", verbose=False, guard=None):
        self.run_id = run_id
        self.verbose = verbose
        self.ledger = Ledger(run_id)
        self.hand_no = 1
        self.adapter = RiverAdapter(self.hand_id)
        self.signals = TurnSignals()
        self.declare = DeclarationDetector()
        self.kyoku = KyokuInfo()
        self.meld = MeldReader()
        self.meld.slot_votes = self._slot_votes
        self.fsm = RoundFSM(self.ledger, hooks={
            "claim": self._claim,
            "note_call": self.signals.note_call,
            "note_kan": self.signals.note_kan,
            "note_reach": self.signals.note_reach,
            "infer_call": self._infer_call,
            "cancel_cut": self.signals.cancel,
        }, hand_id=self.hand_id)
        self.guard = guard          # mortal_guard.MortalGuard 或 None
        if guard is not None:
            guard.p0_concealed = self._p0_concealed
        self.sent = []              # 這一幀送出的 mjai 事件
        self.signals.pai_of_slot = self._pai_of_slot
        self.stats = Counter()
        self.frames = 0
        self.empty_run = 0
        self.empty_buf = []
        self.last_total = 0
        self.nh_pending = None
        self.start_eid = None
        self.hand_start_ms = None
        self.log = []

    @property
    def hand_id(self):
        return f"H{self.hand_no:03d}"

    def _infer_call(self, seat, pd, src_eid, frame):
        """漏看鳴牌宣告(角色立繪特效蓋掉橫幅)時,從鳴牌區推:
        棄牌之後 seat 的鳴牌區多了一組 -> 是哪一種鳴牌。推不出來回 None。
        ★ 實機 runs\\20261027_181804:P1 兩次碰 P3 的牌,都是全螢幕角色動畫,
          局勢模型沒有 pon 框 -> 原本記成 EVENT_GAP,整局停送。"""
        src = self.ledger.get(src_eid)
        if src is None:
            return None
        new = self.meld.new_group(seat, src["frame_first"])
        if not new:
            # 鳴牌區只多兩張(橫放被鳴的那張讀不到)、或還沒穩定:
            # 一定要有來源棄牌的牌面,兩張 + 那張 = 碰/吃 才算
            sp = src.get("pai")
            new = self.meld.new_group(seat, src["frame_first"], loose=True)
            if not new or not sp:
                return None
            if len(new) == 2:
                new = list(new) + [sp]
        kinds = {base_pai(x) for x in new}
        if src.get("pai") and base_pai(src["pai"]) not in kinds:
            return None
        if len(kinds) == 1:
            return "pon" if len(new) == 3 else "kan"
        if len(new) == 3 and seat == nxt(pd) and _is_run(new):
            return "chi"
        return None

    def _slot_votes(self, call_eid):
        from collections import Counter as _C
        ev = self.ledger.get(call_eid)
        src = self.ledger.get(ev.get("source_discard")) if ev else None
        sl = self.adapter.find(src.get("slot_id")) if src else None
        return _C(sl.get("head") or []) if sl else {}

    def _pai_of_slot(self, slot_id):
        eid = self.fsm.slot_event.get(slot_id)
        ev = self.ledger.get(eid) if eid else None
        return ev.get("pai") if ev else None

    def _claim(self, seat, slot_id, generation, frame):
        if not slot_id:
            return []
        r = self.adapter.claim(seat, slot_id, generation, frame)
        if not r.ok:
            self.stats["claim_rejected"] += 1
            self._say(f"f{frame} ⚠️ 兩層失效被拒絕:{r.reason}")
        return r.events

    def _say(self, text):
        self.log.append(text)
        if self.verbose:
            print(text)

    # ------------------------------------------------------------------
    def _ensure_start_event(self, t, frame):
        if self.start_eid is None:
            ev = self.ledger.add("start_kyoku", self.hand_id, t - 1.0, frame,
                                 "PENDING", order_bias=-9)
            self.start_eid = ev["event_id"]
            self.hand_start_ms = t

    def _update_start(self, t):
        ev = self.ledger.get(self.start_eid)
        if ev is not None and ev.get("export_eligible"):
            if self.kyoku.p0_actor is None and ev.get("seat_map"):
                self.kyoku.p0_actor = ev["seat_map"]["P0"]
            self._dealer_check()
        if ev is None or ev.get("export_eligible"):
            return          # 齊了就凍結:供託等欄位局中會變,start_kyoku 要開局值
        # ★ 2026-10-30 實戰:手牌模型在某一格一直漏框(最早的穩定暗牌只有
        #   11/12 張)-> 原本只看最早那一份,整局起手牌都缺、整局不送。
        #   現在依序試每一份穩定暗牌,第一份能倒推成 13 張的就用。
        tiles, note = None, None
        for snap in list(self.signals.p0_stable)[:20]:
            tiles, n_ = self._rewind_p0(snap)
            if note is None:
                note = n_
            if tiles is not None:
                note = n_
                break
        f, miss = self.kyoku.start_kyoku_fields(tiles, note)
        # 每個欄位第一次有值就固定,不被局中的變化覆寫(例如立直後供託 +1)
        upd = {k: v for k, v in f.items()
               if v is not None and (ev.get(k) is None or k == "tehais_note")
               and ev.get(k) != v}
        if upd.get("tehais_note") == "":
            upd["tehais_note"] = None
        if upd:
            self.ledger.update(self.start_eid, t_ms=t, reason="場況", **upd)
        self._dealer_check()

    def _rewind_p0(self, snap):
        """P0 起手 13 張 = 最早的穩定暗牌,倒推回這一局 P0 的第一個動作之前。

        ★ 實機 runs\\20261027_181804 H001:錄影開始時 P0(莊家)正要打第一張,
          前幾幀手牌在理牌、不穩定,第一份穩定暗牌已經是「打掉 9s 之後」的。
          直接拿來當起手牌 -> 手牌帳少一張 9s、多一張摸到的 N,守門器擋下。
        倒推:暗牌 + 之前打掉的 + 之前鳴牌用掉的 − 之前摸到的。
        最後一筆如果是摸牌(還在摸入牌位置、沒進暗牌),那一張不扣。
        之前的事件牌面還不知道 -> 先不填(等)。"""
        f0, _f1, tiles = snap
        tiles = list(tiles)
        # ★ 2026-11-06 錄影 human-20261103 東1局:第 170 幀一筆摸牌被取消(沒有牌面),
        #   原本照樣算進倒推 ->「之前的摸牌牌面還不知道」,十四份穩定暗牌全部放棄,
        #   整局沒有 start_kyoku。被取消的事件沒有發生過,不能拿來倒推。
        evs = [e for e in self.ledger.ordered(self.hand_id)
               if e.get("actor") == "P0" and e["frame_first"] is not None
               and e.get("status") != "CANCELLED"
               and e["frame_first"] < f0
               and e["type"] in ("tsumo", "dahai", "chi", "pon",
                                 "daiminkan", "ankan", "kakan", "kan_self")]
        if evs and evs[-1]["type"] == "tsumo":
            if self.signals.p0_stable_drawn.get(f0, True):
                # 摸入牌還分開擺著:那一張不在暗牌裡,不扣
                evs = evs[:-1]
            else:
                # ★ 2026-10-27 實機 H006/H007:摸了之後已經打掉、理好牌,但牌河
                #   那張晚 2~4 幀才被偵測到 -> 打牌事件的幀數比這份暗牌晚。
                #   暗牌裡已經沒有摸入牌位置 = 那一打已經發生,把它算進來。
                nxt_d = [e for e in self.ledger.ordered(self.hand_id)
                         if e.get("actor") == "P0" and e["type"] == "dahai"
                         and e.get("status") != "CANCELLED"
                         and e["frame_first"] is not None
                         and e["frame_first"] >= f0]
                if not nxt_d:
                    return None, "倒推起手牌:等 P0 第一打的牌河"
                evs.append(nxt_d[0])
        if not evs:
            pass
        else:
            add, sub = [], []
            for e in evs:
                t = e["type"]
                if t == "tsumo":
                    if not e.get("pai") or e["pai"] == "?":
                        return None, "倒推起手牌:之前的摸牌牌面還不知道"
                    sub.append(e["pai"])
                elif t == "dahai":
                    if not e.get("pai"):
                        return None, "倒推起手牌:之前的打牌牌面還不知道"
                    add.append(e["pai"])
                else:
                    if not e.get("consumed"):
                        return None, "倒推起手牌:之前的鳴牌還沒讀出來"
                    add += list(e["consumed"])
                    if t == "kakan":
                        add.append(e["pai"])
            from hand import diff
            left = diff(tiles + add, sub)
            if len(left) != len(tiles) + len(add) - len(sub):
                return None, (f"倒推起手牌失敗:暗牌裡沒有之前摸到的 {sub}"
                              f"(暗牌 {sorted(tiles)})")
            tiles = sorted(left)
            note0 = (f"起手牌由 f{f0} 的暗牌倒推:加回打掉/鳴牌用掉的 "
                     f"{add}、扣掉摸到的 {sub}")
            if len(tiles) == 13:
                return tiles, note0
            return None, (f"倒推起手牌後是 {len(tiles)} 張,不是 13 張"
                          f"(暗牌 {sorted(snap[2])})")
        if len(tiles) == 14:
            # ★ 2026-11-04 實戰 紀錄\20261104_211402 東3局:P0 是莊家,14 張裡
            #   哪張是「摸的」要等打出去才倒推得出來 -> start_kyoku 晚到,
            #   第一巡沒有建議。開局那筆 P0 摸牌的牌面已經讀到(摸入牌位置)
            #   就直接用它:14 張扣掉那一張 = 起手 13 張。(其實扣哪一張都
            #   一樣,只要跟送出去的摸牌一致,Mortal 看到的 14 張就是對的。)
            first = next((e for e in self.ledger.ordered(self.hand_id)
                          if e.get("actor") == "P0" and e["status"] != "CANCELLED"
                          and e["type"] in ("tsumo", "dahai", "chi", "pon",
                                            "daiminkan", "ankan", "kakan")),
                         None)
            if first is not None and first["type"] == "tsumo" and \
                    first.get("pai") not in (None, "?"):
                from hand import diff
                left = diff(tiles, [first["pai"]])
                if len(left) == 13:
                    return sorted(left), (f"莊家 14 張扣掉開局摸的 "
                                          f"{first['pai']}(f{f0})")
            return None, ("P0 最早的穩定暗牌是 14 張(P0 是莊家,而第 14 張沒有"
                          "分開擺放)-> 無法決定哪一張是摸牌,不任選")
        if len(tiles) != 13:
            return None, f"P0 最早的穩定暗牌是 {len(tiles)} 張,不是 13 張"
        return tiles, ""

    CALL_TYPES = ("chi", "pon", "daiminkan", "kan_self")

    def _p0_concealed(self, frame, after=False):
        """重新同步用:畫面上 P0 的暗牌(穩定、而且在 frame 之後還是這樣)。
        摸入牌分開擺著,不在裡面 -> 張數是 3n+1。
        after=True:要「frame 之後才開始穩定」的那一份(剛打完牌,要打完的手牌)。"""
        for f0, f1, tiles in reversed(self.signals.p0_stable):
            if after and f0 <= frame:
                continue
            if f1 >= frame and len(tiles) % 3 == 1:
                return list(tiles)
        return None

    def _p0_after(self, frame):
        for f0, f1, tiles in self.signals.p0_stable:
            if f0 > frame:
                return tiles
        return None

    def _melds(self, obs, n_before, frame, t):
        fsm = self.fsm
        if fsm.state.name in ("ROUND_END", "WIN_COLLECTION"):
            self.meld.freeze()
        self.meld.feed(frame, obs.get("meld"))
        for ev in self.ledger.events[n_before:]:
            if ev["type"] not in self.CALL_TYPES:
                continue
            seat = ev["actor"]
            pons = []
            for pid in fsm.pons.get(seat, []):
                pe = self.ledger.get(pid)
                if pe and pe["event_id"] != ev["event_id"]:
                    pons.append((pe.get("pai"), pe.get("consumed")))
            p0 = (self.signals._p0_before(frame) if seat == "P0" else None)
            self.meld.request(ev["event_id"], "kan_self"
                              if ev["type"] == "kan_self" else ev["type"],
                              seat, ev.get("call_frame", frame), pons, p0)
        for r in self.meld.resolve(
                frame, lambda eid: self.ledger.get(eid).get("pai"),
                self._p0_after):
            ev = self.ledger.get(r.event_id)
            ev.setdefault("evidence", {})["meld"] = {
                "method": r.method, "ok": r.ok, "note": r.note}
            if r.note:
                self.ledger.note(r.event_id, r.note)
            if not r.ok:
                self.stats["meld_unresolved"] += 1
                self.ledger.note(r.event_id, "consumed 讀不出來:" + r.note)
                continue
            upd = {"consumed": r.consumed}
            if r.method == "meld_run_fix" and r.pai and r.pai != ev.get("pai"):
                upd["pai"] = r.pai
                src = self.ledger.get(ev.get("source_discard"))
                if src is not None and src.get("pai") != r.pai:
                    self.ledger.update(src["event_id"], t_ms=t,
                                       reason="依鳴牌區修正被鳴的牌", pai=r.pai)
            elif ev.get("pai") is None and r.pai is not None:
                upd["pai"] = r.pai
                src = self.ledger.get(ev.get("source_discard"))
                if src is not None and src.get("pai") is None:
                    self.ledger.update(src["event_id"], t_ms=t,
                                       reason="牌河沒定案就被鳴走,牌面取自鳴牌區",
                                       pai=r.pai)
            if ev["type"] == "kan_self":
                upd["type"] = r.kind
                if r.kind == "kakan":
                    upd["pai"] = r.pai
            self.ledger.update(r.event_id, t_ms=t, reason="鳴牌區", **upd)
            self._say(f"f{frame} 鳴牌 {ev['actor']} {ev['type']} "
                      f"consumed={r.consumed} ({r.method})")

    def _add_dora(self, idx, pai, t, frame):
        """槓後新寶牌。放在事件序的位置照雀魂翻寶牌的時機:
            暗槓        槓之後立刻翻
            大明槓/加槓 槓的那一家打出下一張之後才翻
        (觀測到寶牌的時間會晚很多 —— 寶牌每 N 幀才讀一次、要 3 票 —— 所以
         不能用觀測時間排序,否則 dora 會被排到好幾筆之後。)"""
        kans = [e for e in self.ledger.ordered(self.hand_id)
                if e["type"] in ("ankan", "kakan", "daiminkan", "kan_self")]
        tf, note = t, "找不到對應的槓,照觀測時間排"
        if len(kans) >= idx:
            k = kans[idx - 1]
            if k["type"] == "ankan":
                tf, note = k["t_first_ms"] + 0.02, f"第 {idx} 個槓(暗槓)之後"
            else:
                nxt = [e for e in self.ledger.ordered(self.hand_id)
                       if e["type"] == "dahai" and e.get("actor") == k["actor"]
                       and e["t_first_ms"] > k["t_first_ms"]]
                if nxt:
                    tf = nxt[0]["t_first_ms"] + 0.02
                    note = f"第 {idx} 個槓({k['type']})之後那一打之後"
                else:
                    tf = k["t_first_ms"] + 0.02
                    note = f"第 {idx} 個槓之後(那一打還沒看到)"
        ev = self.ledger.add("dora", self.hand_id, tf, frame, "PENDING",
                             dora_marker=pai, index=idx, order_bias=1)
        self.ledger.note(ev["event_id"], note)

    def _dealer_check(self):
        """莊家應該是這一局第一個行動的人(舊專案 session 的第四重檢查)。"""
        ev = self.ledger.get(self.start_eid)
        sm, oya, dg = ev.get("seat_map"), ev.get("oya"), self.fsm.dealer_guess
        if sm and oya is not None and dg and not ev.get("_dealer_checked"):
            ev["_dealer_checked"] = True
            if sm.get(dg) != oya:
                self.ledger.note(self.start_eid,
                                 f"★莊家不一致:場況說 actor {oya},但第一個"
                                 f"行動的是 {dg}(actor {sm.get(dg)})")
                self.stats["dealer_mismatch"] += 1
                if dg == "P0":
                    self._drop_fake_opening_tsumo()

    FRESH_STABLE = 3          # 新配牌要連續幾次(手牌每 2 幀讀一次)一樣
    FRESH_CHANGED = 6         # 跟上一局最後的手牌至少差幾張才算新配牌

    def _fresh_deal(self, ph, empty, frame):
        """上一局已經結束、牌河全空、手牌出現一份「整手換掉」的 13/14 張
        -> 新的一局已經發牌了,不用等第一張牌落到牌河。

        ★ 2026-11-04 實戰 紀錄\20261104_224154 東3局:P0 是莊家。原本換局要等
          牌河出現第一張 —— 莊家的第一張就是你自己打的 -> 上一局 550s 結束,
          這一局到 574s 你打出第一張才認出來,第一巡的建議等你打完才出現。"""
        if self.fsm.state.name != "ROUND_END" or not empty or \
                self.empty_run < EMPTY_MIN:
            self._fresh = []
            return False
        if not ph:
            return False                     # 這一幀沒讀手牌
        tiles = tuple(sorted(x["pai"] for x in ph.get("tiles") or ()))
        if len(tiles) not in (13, 14):
            self._fresh = []
            return False
        hist = getattr(self, "_fresh", [])
        hist = (hist + [tiles])[-self.FRESH_STABLE:]
        self._fresh = hist
        if len(hist) < self.FRESH_STABLE or len(set(hist)) != 1:
            return False
        old = [s for _f0, _f1, s in self.signals.p0_stable]
        if old:
            prev = Counter(old[-1])
            now_ = Counter(tiles)
            changed = sum((now_ - prev).values())
            if changed < self.FRESH_CHANGED:
                return False                 # 還是上一局的手牌(結算畫面)
        self._fresh = []
        self.stats["fresh_deal_new_hand"] += 1
        self._say(f"f{frame} 牌河全空、手牌整手換新 -> 新的一局已發牌")
        return True

    def start_rush(self, frame):
        """這一局的場況(局數、寶牌、點數)還沒讀齊、而且畫面上已經有手牌了
        -> record_obs 這段時間每一幀都讀場況(平常每 5 幀)。

        ★ 2026-11-04 實戰 紀錄\20261104_211402 東3局:P0 是莊家,開局 2 秒就
          打出第一張;場況每 5 幀讀一次、要投 2~3 票 -> start_kyoku 3.3 秒才齊,
          第一巡沒有建議。"""
        ev = self.ledger.get(self.start_eid) if self.start_eid else None
        if ev is None or ev.get("export_eligible"):
            return False
        return frame - getattr(self, "_hand_seen", -999) <= 30

    def _drop_fake_opening_tsumo(self):
        """★ 2026-11-04 實戰 紀錄\20261104_202352 東3局 f920:P0 不是莊家,
        發牌動畫被當成「P0 開局摸 發」-> 接著莊家打牌變 EVENT_GAP、P0 真的
        摸牌變成連續摸兩次 -> Mortal 手牌帳對不上。場況確定莊家不是 P0、
        P0 也還沒打過牌 -> 那一筆開局摸牌是假的,取消。"""
        evs = self.ledger.ordered(self.hand_id)
        if any(e["type"] == "dahai" and e.get("actor") == "P0"
               and e["status"] != "CANCELLED" for e in evs):
            return
        for e in evs:
            if e["type"] == "tsumo" and e.get("actor") == "P0" and \
                    e["status"] != "CANCELLED" and \
                    any("T02" in n for n in e.get("notes") or ()):
                self.ledger.update(e["event_id"], status="CANCELLED",
                                   reason="莊家不是 P0:開局的 P0 摸牌是發牌動畫")
                self.stats["fake_opening_tsumo"] += 1
                self._say(f"↷ 取消開局的 P0 摸牌 {e.get('pai')}(莊家不是 P0,"
                          f"那是發牌動畫)")
                break

    def _new_hand(self, t, frame):
        if self.fsm.state.name not in ("ROUND_END", "UNKNOWN"):
            self.stats["hand_end_unseen"] += 1
            evs = self.ledger.ordered(self.hand_id)
            per = Counter(e.get("actor") for e in evs if e["type"] == "dahai")
            calls = [e for e in evs if e["type"] in
                     ("chi", "pon", "daiminkan", "ankan", "kakan", "kan_self")]
            if not calls and per and max(per.values()) <= 1:
                # ★ 第一巡、沒有人鳴牌就結束了 -> 只可能是途中流局(九種九牌,
                #   或四風連打)。兩種在 mjai 都是 ryukyoku。2026-10-27 實機 H006:
                #   P2 九種九牌,局勢模型沒有這個類別。
                last = evs[-1]["t_first_ms"] if evs else t - 1
                r = self.fsm._ev("ryukyoku", last + 1, frame, "INFERRED")
                self.ledger.note(r["event_id"],
                                 "第一巡沒人鳴牌就換局 -> 途中流局"
                                 "(九種九牌/四風連打),推定")
                self.fsm._end(last + 1, frame, "推定途中流局")
                self.stats["abortive_draw_inferred"] += 1
            else:
                # 例:漏看的和牌/流局
                self.fsm._gap(t - 1, frame, repr(self.fsm.state), "換局",
                              "沒看到終局宣告就換局了(漏看和牌/流局)")
                self.fsm._end(t - 1, frame, "漏看終局")
        self._update_start(t)
        # 上一局有沒有真的打過(要在 hand_no 往前之前看:hand_id 跟著 hand_no 變)。
        # 只算定案的:錄影開頭載入畫面的假牌河會留下牌面讀不到的待定打牌。
        played = any(e["type"] in ("dahai", "chi", "pon", "daiminkan", "ankan",
                                   "kakan", "hora", "ryukyoku")
                     and e.get("export_eligible")
                     for e in self.ledger.ordered(self.hand_id))
        self.hand_no += 1
        rel = self.adapter.new_hand(self.hand_id, frame)
        self.signals.new_hand()
        self.declare = DeclarationDetector()
        st_ev = self.ledger.get(self.start_eid) if self.start_eid else None
        # ★ 2026-11-07 錄影 20261024-123746:錄影一開始就是東1局發牌,第 1 幀開的那一局
        #   讀到「東1/0」,f98 手牌整手出現又換一次局(_fresh_deal)-> 新局的「東1/0」
        #   被當成上一局的舊面板擋掉,局末讀到「東2」才填 -> 之後每一局都錯一位。
        #   上一局什麼都沒發生(沒人打牌、鳴牌、和牌、流局)就不算數。
        self.kyoku.reset(remember=bool(st_ev and st_ev.get("export_eligible"))
                         and played)
        self.meld.reset()
        self.fsm.new_hand(self.hand_id, t, frame)
        self.start_eid = None
        self._wall_reset()
        self.stats["hands"] += 1
        self._say(f"f{frame} ── 換局 -> {self.hand_id}(封存 {len(rel)} 格)")

    # ------------------------------------------------------------------
    def feed(self, obs):
        # ★ 2026-11-07 還沒進過牌桌的畫面整幀不看:selfplay-20261106-103105 / 110134 /
        #   113550 開局前的載入畫面(約 40~50 秒)四家牌河都讀到假的「白」-> 一串假打牌,
        #   假局的場況後來被真的東1局填齊,第一打時換局條件一條都不成立 -> 假牌佔住的
        #   位置把下家真的打牌擋掉(DahaiGate),東1局丟了四巡。只擋牌河不夠:
        #   selfplay-20261102-170335 載入畫面一個假的「下家摸牌」讓狀態機等下家,
        #   莊家(自己)第一打前沒有推定第 14 張 -> MIA 丟掉那一打,整局錯一位。
        #   從開始到現在一次余牌都沒讀到、也從沒看到一整手手牌 = 還沒進過牌桌。
        #   手牌一出現就不再管(余牌讀取壞掉時不會一直擋)。沒有 hud 欄位的呼叫端不受影響。
        if not getattr(self, "_at_table", False):
            ph0 = obs.get("p0_hand")
            if "hud" not in obs or (obs["hud"] or {}).get("wall") is not None \
                    or (ph0 and len(ph0.get("tiles") or ()) >= 10):
                self._at_table = True
            else:
                return []
        if self.nh_pending is not None:
            return self._pending_step(obs)
        return self._feed(obs)

    @staticmethod
    def _total(obs):
        river = obs.get("river")
        if river is None or not all(river.get(s) and
                                    river[s].get("tiles") is not None
                                    for s in SEATS):
            return None
        return sum(1 for s in SEATS for d in river[s]["tiles"]
                   if d.get("conf", 1) >= COUNT_CONF
                   and not is_hover_tip(s, d) and not is_sliver(d)
                   and not is_bottom_clip(s, d, river[s].get("zone")))

    def _pending_step(self, obs):
        """疑似換局:先觀察 NH_CONFIRM 幀再決定(見 NH_CONFIRM 的說明)。"""
        P = self.nh_pending
        P["obs"].append(obs)
        total = self._total(obs)
        if total is not None and total > NH_CANCEL:
            # 牌河又回來一大堆 -> 是特效遮住,不是換局
            self.nh_pending = None
            self.stats["overlay_hidden_rivers"] += 1
            self._say(f"f{obs['frame_index']} 牌河只回來幾張又全部回來 -> "
                      f"特效,不是換局")
            out = []
            for o in P["obs"][:-1]:
                out += self._feed(o, hide_river=True)
            self.last_total = P["last_total"]
            out += self._feed(obs)
            return out
        if len(P["obs"]) < NH_CONFIRM:
            return []
        self.nh_pending = None
        o0 = P["obs"][0]
        self._new_hand(o0["t_ms"], o0["frame_index"])
        fsm = self.fsm
        # ★ 2026-10-27 實機 H006/H007:九種九牌後新局開得很快,牌河還空著的
        #   那幾幀,莊家 P0 已經摸了第 14 張(F)、而且打掉了 9s。那幾幀原本餵給
        #   的是舊局 -> 新局一開始就少了 P0 的摸牌與牌名,起手牌倒推錯。換局後
        #   把牌河全空那幾幀的摸牌/手牌訊號重新餵給新局。
        rp = P["replay"]
        if rp:
            self._ensure_start_event(rp[0][1], rp[0][0])
        for f_, t_, r_, h_ in rp:
            for d in self.signals.feed(f_, t_, r_, h_):
                if d.kind == "draw_seen":
                    fsm.on_draw_seen(d, d.t_ms, f_)
                elif d.kind == "draw_name":
                    fsm.on_draw_name(d, t_, f_)
        self.empty_run = 0
        self.empty_buf = []
        self.last_total = 0
        out = []
        for o in P["obs"]:
            out += self._feed(o)
        return out

    def _banner_frozen(self, frame, region, river):
        """這一幀哪幾家的牌河被宣告橫幅或手臂蓋住(含消失後的寬限幀)。"""
        if not river:
            return ()
        hold = getattr(self, "_banner_until", None)
        if hold is None:
            hold = self._banner_until = {}
        for d in region or []:
            if d.get("cls") not in BANNER_CLS or d.get("conf", 0) < BANNER_CONF:
                continue
            x1, y1, x2, y2 = d["box"]
            for s, part in river.items():
                if not part or not part.get("zone"):
                    continue
                zx, zy, zw, zh = part["zone"]
                iw = min(x2, zx + zw) - max(x1, zx)
                ih = min(y2, zy + zh) - max(y1, zy)
                if iw > 0 and ih > 0 and iw * ih >= BANNER_MIN_AREA:
                    hold[s] = frame + BANNER_GRACE
        # 只凍結「現在不該打牌的那幾家」:輪到打牌的那家,手臂本來就會蓋著
        # 自己剛放下的牌 —— 凍結它會讓真的打牌晚收,排到下家摸牌後面
        # (20261030_183152 開局 P2 第一打就是這樣錯位)。
        st = self.fsm.state
        turn = st.actor if st.name in ("WAIT_ACTION",
                                       "WAIT_DISCARD_AFTER_CALL") else None
        for d in region or []:
            if turn is None:
                break
            if d.get("cls") != "arm" or d.get("conf", 0) < ARM_CONF:
                continue
            x1, y1, x2, y2 = d["box"]
            for s, part in river.items():
                if s == turn:
                    continue
                for t in (part or {}).get("tiles") or ():
                    tx1, ty1, tx2, ty2 = t["box"]
                    area = max(1, (tx2 - tx1) * (ty2 - ty1))
                    iw = min(x2, tx2) - max(x1, tx1)
                    ih = min(y2, ty2) - max(y1, ty1)
                    if iw > 0 and ih > 0 and iw * ih >= ARM_TILE_OVERLAP * area:
                        hold[s] = max(hold.get(s, -1), frame + ARM_GRACE)
                        self.stats["arm_hold"] = \
                            self.stats.get("arm_hold", 0) + 1
                        break
        return {s for s, until in hold.items() if frame <= until}

    def _feed(self, obs, hide_river=False):
        frame = obs["frame_index"]
        t = obs["t_ms"]
        self.frames += 1
        self._ensure_start_event(t, frame)
        fsm = self.fsm
        n_before = len(self.ledger.events)

        region = obs.get("region_detections")

        river = obs.get("river")
        total = self._total(obs)
        empty = None if total is None else total == 0
        # ★ 2026-11-06 selfplay-20261106-110134 南4局1本場 -> 西1局:和牌的結算畫面蓋住
        #   余牌,攤開的和牌手牌落在牌河區 -> 讀成牌河「回來 1~2 張」-> 多開一局幽靈局,
        #   還把西1局的場況吃走(本場是舊面板的 1),真的西1局被當成「還在看上一局的
        #   面板」整局不送。上一局結束之後余牌讀不到 = 還不是牌桌,牌河先不算(牌留在
        #   牌河上,余牌讀得到時一樣看得到,只是晚幾幀)。只管 ROUND_END:發牌動畫時
        #   余牌也常讀不到,連「這一局還沒有人打牌」都擋的話,莊家第一打會漏掉
        #   (human-20261103 東3局、selfplay-170335 東1局實測)。
        #   沒有 hud 欄位(MIA 以外的呼叫端)不受影響。
        if "hud" in obs and (obs["hud"] or {}).get("wall") is None \
                and self.fsm.state.name == "ROUND_END":
            hide_river = True
        if hide_river:
            river = None
            empty = None
        if empty:
            self.empty_run += 1
            self.empty_buf.append((frame, t, region, obs.get("p0_hand")))
            del self.empty_buf[:-60]
            if self.last_total > NEW_HAND_MAX_TILES:
                river = None        # 很多張一起不見:先不餵,可能只是特效遮住
        elif empty is False:
            if self.empty_run >= EMPTY_MIN:
                st_ev = self.ledger.get(self.start_eid) if self.start_eid \
                    else None
                no_start = not (st_ev and st_ev.get("export_eligible"))
                if total <= NEW_HAND_MAX_TILES and (
                        total < self.last_total
                        or fsm.state.name == "ROUND_END"
                        # ★ 2026-10-30 實戰:錄影開始時還在大廳/載入畫面,讀到
                        #   一張假牌就變成一局 -> 真的東1局接在後面。這一局連
                        #   場況都沒讀到 -> 牌河空過再回來少數幾張就是新局。
                        #   (要空很久才算:大廳->開局空了 61 幀;鳴牌拿走唯一一張
                        #    牌河、接著打牌只空幾幀)
                        or (no_start and self.empty_run >= NO_START_EMPTY)):
                    if any(e["type"] == "dahai" for e in
                           self.ledger.ordered(self.hand_id)):
                        # 疑似換局:先別動,多看 NH_CONFIRM 幀
                        self.nh_pending = {
                            "obs": [obs], "last_total": self.last_total,
                            "replay": list(self.empty_buf[-REPLAY_EMPTY:])}
                        return []
                elif total > NEW_HAND_MAX_TILES:
                    self.stats["overlay_hidden_rivers"] += 1
                    self._say(f"f{frame} 牌河被遮 {self.empty_run} 幀"
                              f"(回來時 {total} 張)-> 特效,不是換局")
            self.empty_run = 0
            self.empty_buf = []
            self.last_total = total
        ph = obs.get("p0_hand")
        if ph and len(ph.get("tiles") or ()) >= 10:
            self._hand_seen = frame
        if self._fresh_deal(ph, empty, frame):
            self._new_hand(t, frame)
        draw_evs = self.signals.feed(frame, t, region, obs.get("p0_hand"))

        for e in self.adapter.feed(frame, t, river,
                                   frozen=self._banner_frozen(frame, region,
                                                              river)):
            if e.kind == "candidate":
                ev = fsm.on_candidate(e, e.t_first_ms, frame)
                if ev is not None:
                    # 一局的第一打(牌河本來是空的、帳上還沒有別的打牌)
                    first = (total is not None and total <= 1 and not any(
                        x["type"] == "dahai" and x["status"] != "CANCELLED"
                        and x["event_id"] != ev["event_id"]
                        for x in self.ledger.ordered(self.hand_id)))
                    self.signals.on_candidate(e.seat, e.slot_id, frame,
                                              first_of_hand=first)
            elif e.kind == "confirmed":
                fsm.on_confirmed(e, t, frame)
            elif e.kind == "released":
                self.stats[f"released_{e.reason}"] += 1

        for d in self.declare.feed(frame, t, region):
            self._say(f"f{frame} 宣告 {d.seat} {d.cls} ({d.conf})")
            fsm.on_declaration(d, d.t_ms, frame)

        for d in draw_evs:
            if d.kind == "draw_seen":
                fsm.on_draw_seen(d, d.t_ms, frame)
            elif d.kind == "draw_name":
                fsm.on_draw_name(d, t, frame)

        for v in self.signals.judge(frame):
            fsm.on_cut(v, t, frame)

        # 余牌對帳:沒看到宣告的槓
        self._see_wall(obs.get("hud"), t, frame)
        self._wall_kans(t, frame)

        # 鳴牌區(第 4 步):consumed、槓型
        self._melds(obs, n_before, frame, t)

        # 場況(第 0 步)與槓後新寶牌
        for idx, pai in self.kyoku.feed(obs.get("ui_context"),
                                        obs.get("dora_observation")):
            if idx >= 1:
                self._add_dora(idx, pai, t, frame)
        self._update_start(t)
        self._late_p0_draw(t)

        if fsm.tick(t, frame, empty):     # 只剩「長時間全空 = 漏看終局」
            self._new_hand(t, frame)

        if self.verbose:
            for ev in self.ledger.events[n_before:]:
                self._say(f"f{frame} + {describe(ev)}  [{fsm.state!r}]")
        self._pump(t)
        return self.ledger.events[n_before:]

    # ------------------------------------------------------------------
    # 余牌對帳
    def _wall_reset(self):
        self._wall = None                 # 目前認定的余牌
        self._wall_seen = (None, 0, None, None)   # (讀值, 連續幾幀, 第一次的 t, frame)
        self._drops = []                  # 余牌每少一張一筆 {"t", "frame", "done"}

    def _see_wall(self, hud, t, frame):
        """記下余牌每一次減少的時刻。MIA 的余牌讀取(樣板版用來排輪次)每幀都讀。"""
        if not hasattr(self, "_drops"):
            self._wall_reset()
        w = (hud or {}).get("wall")
        if w is None:
            return
        was, n, t0, f0 = self._wall_seen
        self._wall_seen = (w, n + 1, t0, f0) if w == was else (w, 1, t, frame)
        w, n, t0, f0 = self._wall_seen
        if n != WALL_STABLE or w == self._wall:
            return
        if self._wall is not None and 0 < self._wall - w <= WALL_MAX_STEP:
            self._drops += [{"t": t0, "frame": f0, "done": False}
                            for _ in range(self._wall - w)]
        elif self._wall is not None and 0 < w - self._wall <= WALL_MAX_STEP:
            # 往回跳:剛才那幾張是讀錯,撤掉(換局會整個重設,不會走到這裡)
            del self._drops[len(self._drops) - (w - self._wall):]
        self._wall = w

    def _wall_kans(self, t, frame):
        """沒看到宣告的自家槓。

        ★ 2026-11-06 錄影 human-20261103 東3局:上家暗槓八筒,槓的橫幅與鳴牌區的變化都沒
          抓到,只看到新翻的寶牌 -> 帳上那一巡少了暗槓與嶺上牌。
        兩個獨立的證據都要有才推:
          ・翻出了新的槓寶牌,帳上卻沒有對應的槓;
          ・余牌比帳上的摸牌多少了一張(嶺上牌)。一打到下一打之間余牌只少一張,
            槓多摸一張嶺上牌就少兩張,而且摸牌與下一打是同一家 -> 就是那一家槓的。
        只看余牌不夠:電腦對局一個動作 0.5 秒,牌河的牌比下一家摸牌還晚被看到時,
        下一家那一摸會算進上一段(一場錄影 30 筆假的)。
        槓型與牌面照常交給鳴牌區(kan_self)。P0 不推:自己的槓看手牌就知道。
        帳上有鳴牌的那一段也不推(大明槓、碰後加槓太多種)。"""
        if not getattr(self, "_drops", None):
            return
        evs = [e for e in self.ledger.ordered(self.hand_id)
               if e.get("status") != "CANCELLED" and e.get("t_first_ms") is not None]
        kans = [e for e in evs if e["type"] in ("ankan", "kakan", "daiminkan", "kan_self")]
        doras = sorted((e for e in evs if e["type"] == "dora" and (e.get("index") or 0) >= 1),
                       key=lambda e: e["index"])
        if len(doras) <= len(kans):
            return
        seen = doras[len(kans)]["t_first_ms"]       # 沒有槓的寶牌照觀測時間排
        bounds = ("dahai",) + self.CALL_TYPES + ("ankan", "kakan")
        found = None
        for d in self._drops:
            if d["done"] or not seen - WALL_KAN_WINDOW_MS <= d["t"] <= seen:
                continue
            nxt_d = next((e for e in evs if e["type"] == "dahai"
                          and e["t_first_ms"] > d["t"]), None)
            if nxt_d is None or t - nxt_d["t_first_ms"] < WALL_SETTLE_MS:
                return          # 寶牌之前還有沒對完帳的一段,等它
            prev = max((e["t_first_ms"] for e in evs if e["type"] in bounds
                        and e["t_first_ms"] < d["t"]), default=float("-inf"))
            end = nxt_d["t_first_ms"]
            seg = [x for x in self._drops if prev < x["t"] <= end]
            inside = [e for e in evs if prev < e["t_first_ms"] < end]
            tsumos = [e for e in inside if e["type"] == "tsumo"]
            seat = nxt_d.get("actor")
            if len(seg) - len(tsumos) == 1 and tsumos and seat != "P0" \
                    and all(e.get("actor") == seat for e in tsumos) \
                    and not any(e["type"] in bounds for e in inside):
                found = (seg, tsumos, seat, end)      # 留最晚的一段:離寶牌最近
        if found is None:
            return
        seg, tsumos, seat, end = found
        for x in seg:
            x["done"] = True
        last = max(e["t_first_ms"] for e in tsumos)
        rin = max(seg, key=lambda x: x["t"])
        tk = rin["t"] if last < rin["t"] < end else (last + end) / 2
        k = self.fsm._ev("kan_self", tk, frame, "INFERRED", actor=seat,
                         pai=None, consumed=None,
                         kan_type=("kakan_or_ankan" if self.fsm.pons[seat]
                                   else "ankan"))
        # 鳴牌區的槓比嶺上牌早出現(實測早 5 幀),「槓之前」要從槓前那次摸牌算
        k["call_frame"] = max(tsumos, key=lambda e: e["t_first_ms"])["frame_first"]
        self.ledger.note(k["event_id"], f"翻了新的槓寶牌、帳上沒有槓;余牌在 {seat} 摸牌與"
                                        "打牌之間多少了一張(嶺上牌)-> 推定自家槓")
        r = self.fsm._ev("tsumo", tk + (end - tk) / 2, frame, "INFERRED",
                         actor=seat, pai="?", origin="RINSHAN")
        self.ledger.note(r["event_id"], "推定槓的嶺上牌")
        self.stats["wall_inferred_kan"] += 1
        self._say(f"f{frame} 余牌對帳:{seat} 推定自家槓")

    def _late_p0_draw(self, t):
        """P0 手切了,摸到什麼牌卻一直沒補上:拿打牌後**之後**的穩定暗牌再比一次。

        ★ 2026-11-06 錄影 20261024 東2局 f1119:摸三索,打東。打牌後第一份穩定暗牌把
          七索看成八索 -> 差分「少了 7s、E,多了 3s、8s」,判不出摸到哪張,摸牌一直
          PENDING。下游等不到這一摸,之後每一次摸牌都錯開一位。兩秒後的那份暗牌
          已經讀對(少了 E、多了 3s)。切法判斷只看第一份,所以這裡往後找:
          打牌前那份暗牌 → 打牌後任何一份,恰好少了牌河那張、多了一張,那一張就是摸到的。
          只看到暗牌下一次改變之前。"""
        from hand import diff
        evs = [e for e in self.ledger.ordered(self.hand_id) if e.get("actor") == "P0"
               and e.get("status") != "CANCELLED"]
        for i, ts in enumerate(evs):
            # ★ 2026-11-06 也補 INFERRED(摸牌沒看到、牌河出現新棄牌才推出來的那種)。
            #   selfplay-20261106-113550 東1局4本場、東3局、東4局1本場:都是這種,下游
            #   等這一摸的牌面等到局尾,整局後半沒有事件流。
            if ts["type"] != "tsumo" or ts.get("pai") is not None \
                    or ts["status"] not in ("PENDING", "INFERRED"):
                continue
            d = next((e for e in evs[i + 1:] if e["type"] == "dahai"
                      and e.get("p0_tsumo") == ts["event_id"]), None)
            if d is None or d.get("tsumogiri") is not False or not d.get("pai"):
                continue
            # 打牌前的暗牌:推出來的摸牌,幀數是看到棄牌那一幀(比真的摸牌晚),最後
            # 一份可能已經是打完的。所以從上一次 P0 動作之後的每一份都試。
            lower = max((e["frame_first"] for e in evs[:i]
                         if e["type"] != "tsumo" and e["frame_first"] is not None),
                        default=-1)
            prevs = [h for f0, f1, h in reversed(self.signals.p0_stable)
                     if f0 < ts["frame_first"] and f1 > lower][:6]
            # 更早的暗牌用之間已知的動作往後推到「摸牌前」。★ 113550 東4局1本場:
            # 摸牌前那兩份一份少讀 9s、一份把 6s 讀成 6m,再早一份是讀對的。
            # 「之後的動作」從哪一幀算:摸到的牌放旁邊時暗牌不變,一份暗牌會跨過摸牌;
            # 牌河又比手牌晚一兩幀。開始幀 +3 與結束幀兩種都試,算錯的那種不會剛好
            # 差一張,投不出票。
            for f0, f1, h in list(reversed(self.signals.p0_stable)):
                if f1 > lower:
                    continue
                if len(prevs) >= 16:
                    break
                for cut in (f0 + 3, f1):
                    fwd = _redo_p0(h, [e for e in evs[:i] if e["frame_first"] is not None
                                       and e["frame_first"] > cut])
                    if fwd is not None and fwd not in prevs:
                        prevs.append(fwd)
            # 打牌後的暗牌:不只看到下一次改變為止。★ 2026-11-06 113550 東1局4本場:
            # 打牌後只有一份,而那份把 2p 讀成 7m,差兩張判不出來。之後的每一份用
            # 已知的後續動作倒推回「剛打完」,各投一票。
            later = evs[evs.index(d) + 1:]
            votes = Counter()
            for f0, f1, cur in self.signals.p0_stable:
                if f1 <= d["frame_first"]:
                    continue
                back = _undo_p0(cur, [e for e in later if e["frame_first"] is not None
                                      and e["frame_first"] < f0])
                if back is None:    # 這份讀錯(倒推出負的)或之後的動作牌面還不知道
                    continue
                for prev in prevs:
                    inn, out = diff(back, list(prev)), diff(list(prev), back)
                    if len(inn) == 1 and out == [d["pai"]]:
                        votes[inn[0]] += 1
                        break
            top = votes.most_common(2)
            if top and top[0][1] >= LATE_DRAW_VOTES and (
                    len(top) == 1 or top[0][1] >= 2 * top[1][1]):
                src = (f"手切:打牌後的暗牌倒推,多出來的那張 {top[0][0]}"
                       f"(投票 {dict(votes)})")
                self.ledger.update(ts["event_id"], t_ms=t, reason=src, pai=top[0][0])
                ts["pai_source"] = src
                self.fsm.stats["p0_tsumo_late"] += 1

    def _pump(self, t):
        """第 4 步:Mortal 守門器。上一局的尾巴(end_kyoku)也要送完。"""
        self.sent = []
        if self.guard is None:
            return
        hands = self.guard.active()
        if self.hand_id not in hands:
            hands.append(self.hand_id)
        for h in hands:
            # 已經換局了還沒送完的舊局:不再等,能送的送、其餘略過,送 end_kyoku
            # (★ 2026-11-04 實戰:流局前最後一張 P0 打牌等不到切法判斷 -> 舊局
            #   一直開著,新局的 start_kyoku 直接疊上去)
            self.sent += self.guard.pump(self.ledger, h, t,
                                         final=(h != self.hand_id))

    def finish(self):
        if self.nh_pending is not None:
            P, self.nh_pending = self.nh_pending, None
            for o in P["obs"]:
                self._feed(o, hide_river=True)
        if self.start_eid:
            self._update_start(None)
        self._pump(None)
        if self.guard is not None:
            self.guard.finish()


def _redo_p0(tiles, events):
    """_undo_p0 的反方向:把一份暗牌往後推過這些 P0 動作。"""
    from hand import diff
    add, sub = [], []
    for e in events:
        if e.get("actor") != "P0" or e.get("status") == "CANCELLED":
            continue
        t = e["type"]
        if t == "tsumo":
            if not e.get("pai") or e["pai"] == "?":
                return None
            add.append(e["pai"])
        elif t == "dahai":
            if not e.get("pai"):
                return None
            sub.append(e["pai"])
        elif t in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            if not e.get("consumed"):
                return None
            sub += list(e["consumed"])
            if t == "kakan":
                sub.append(e["pai"])
    whole = list(tiles) + add
    left = diff(whole, sub)
    if len(left) != len(whole) - len(sub):
        return None
    return left


def _undo_p0(tiles, events):
    """把一份暗牌倒推回這些 P0 動作之前:加回打掉/鳴牌用掉的、扣掉摸到的。

    任何一個動作牌面還不知道、或倒推出負的張數,回傳 None。摸切(打的就是剛摸的)
    兩邊抵銷,不碰暗牌。"""
    from hand import diff
    add, sub = [], []
    for e in events:
        if e.get("actor") != "P0" or e.get("status") == "CANCELLED":
            continue
        t = e["type"]
        if t == "tsumo":
            if not e.get("pai") or e["pai"] == "?":
                return None
            sub.append(e["pai"])
        elif t == "dahai":
            if not e.get("pai"):
                return None
            add.append(e["pai"])
        elif t in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            if not e.get("consumed"):
                return None
            add += list(e["consumed"])
            if t == "kakan":
                add.append(e["pai"])
    whole = list(tiles) + add
    left = diff(whole, sub)
    if len(left) != len(whole) - len(sub):
        return None
    return left


def _is_run(tiles):
    b = sorted(base_pai(x) for x in tiles)
    if any(len(x) != 2 or x[1] not in "mps" for x in b):
        return False
    if len({x[1] for x in b}) != 1:
        return False
    n = [int(x[0]) for x in b]
    return n[1] == n[0] + 1 and n[2] == n[1] + 1


def describe(ev):
    t = ev["type"]
    a = ev.get("actor") or ""
    st = ev["status"]
    if t == "dahai":
        cut = {True: "摸切", False: "手切", None: "?"}[ev.get("tsumogiri")]
        return f"{a} 打 {ev.get('pai') or '?'} {cut} ({st})"
    if t == "tsumo":
        return f"{a} 摸 {ev.get('pai') or '?'} ({st})"
    if t in ("chi", "pon", "daiminkan", "kan_self"):
        return f"{a} {t} 來源 {ev.get('target')} 牌 {ev.get('pai')} ({st})"
    if t == "event_gap":
        return f"EVENT_GAP 預期 {ev.get('expected')} 看到 {ev.get('observed')}"
    if t == "hora":
        return f"{a} 和 (放銃 {ev.get('target')})"
    return f"{t} {a} ({st})"
