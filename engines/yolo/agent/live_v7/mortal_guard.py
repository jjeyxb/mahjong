r"""第 4 步:Mortal 守門器(v7.1 第 3 節)。本地事件帳 -> 不可逆的 mjai 串流。

=== 三個生命週期是分開的 ===
    本地事件帳  可以補、可以更正(ledger.py)
    串流        已送出的不能改、不能重送、不能在前面插東西(這裡)
這一層每一幀看一次事件帳,把「照因果順序排在最前面、而且欄位齊了」的那一段
送出去;遇到第一筆不齊的就停(不越過缺口)。

=== 串流狀態(每一局)===
    WAITING_START    start_kyoku 欄位還沒齊
    BLOCKED_START    start_kyoku 齊了但不可信(P0 的 actor 跟前幾局不同)
    RUNNING          正常送
    BLOCKED_EVENT    某一筆等太久(BLOCK_AFTER_MS)還是不齊、遇到 EVENT_GAP、
                     或 P0 手牌帳對不上 -> 這一局停止送
    STREAM_DIVERGED  已送出的事件後來被證明錯了,或有更早的事件插到已送出的前面
                     -> 這一局停止送,**不送補丁**(Mortal 的狀態已經錯了)
    DONE             end_kyoku 已送
停下來的那一局,Mortal 的建議不再可信;下一局有可信的 start_kyoku 就重新開始。

=== 2026-11-04 改成「不停」(使用者:不要因為辨識錯誤整局沒建議)===
libriichi 只檢查三件事(Mortal\libriichi\src\state\update.rs 實際看過):
  P0 自己打/鳴的牌手上要有、同一種牌看得見的不能超過 4 張、
  P0 手牌張數在 3n+1 / 3n+2 之間交替(不然 assert panic)。
別家的出牌順序、有沒有漏摸牌,它完全不檢查。所以:
  別家的事件對不上 / 等太久 / 會變成第 5 張   -> 略過那一筆,照常往下送
  EVENT_GAP                                   -> 記一筆,照常往下送
  已送出的別家事件後來被更正                   -> 記一筆,Mortal 那邊不改
  P0 自己的帳對不上(或 Mortal 程序掛了)      -> DESYNC:先不送,下一次 P0
     摸牌(牌面讀得到)時,用「畫面上的暗牌」+ 事件帳重建這一局,整段重送給
     Mortal(can_act=false,不會吐建議),接著送這次摸牌 -> 建議恢復。
重建時別家的牌照事件帳送(安全牌判斷大致還在),P0 自己的每一次「摸 -> 打」
改成「摸 X -> 打 X」,起手牌由畫面上的暗牌倒推,所以 P0 的手牌一定對。

=== 會讓 libriichi 整個 process panic 的(舊專案 to_mjai.py 實測)===
    自己的 tsumo.pai / dahai.pai / tehais 填 "?"
    自己打一張手上沒有的牌(當下不報錯,下一個對手打牌時才 panic)
所以這裡自己記 P0 的手牌帳:起手 13 + 摸 - 打 - 鳴牌用掉。對不上就停,不送。
另外自己的 actor 整場固定,算出不同的值就代表某一局的自風或局數讀錯了。

=== 送出的時機 ===
「欄位齊了」由事件帳決定(pai、tsumogiri、consumed...)。reach 被榮和而
不成立時,reach 事件本身照送(宣告確實發生了),只是不會有 reach_accepted。
"""
import json
from collections import Counter

from tiles import base_pai

# 最前面那一筆不齊的事件,從 t_first 算起等這麼久就放棄這一局(毫秒)。
# ⚠️ 起始值。鳴牌的 consumed 實測要 1~2 秒,牌面定案 8 幀(6 fps 下 1.3 秒)。
BLOCK_AFTER_MS = 6000
# 槓之後等新寶牌最多多久(寶牌要投票,比一般欄位慢)
DORA_WAIT_MS = 10000
P0_TSUMO_WAIT_MS = 180000
NOTICE_MS = 2500

SKIP = ("correction", "start_kyoku")
STOPPED = ("BLOCKED_START", "BLOCKED_EVENT", "STREAM_DIVERGED", "DONE")
P0_TYPES = ("tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan")
START_KEYS = ("bakaze", "kyoku", "honba", "kyotaku", "oya", "scores",
              "dora_marker", "seat_map")
CONSUMED_LEN = {"chi": 2, "pon": 2, "daiminkan": 3, "ankan": 4, "kakan": 3}
# 已送出之後,這些欄位再改就代表送錯了
CONTENT = ("type", "actor", "target", "pai", "tsumogiri", "consumed",
           "dora_marker")


def _version(e):
    return tuple(tuple(e[k]) if isinstance(e.get(k), list) else e.get(k)
                 for k in CONTENT)


def _actor(sm, seat):
    return None if seat is None else sm.get(seat)


def _visible(e):
    """這一筆送出後「新看得見」的牌(P0 自己手上的牌在起手/摸牌時已經算過)。"""
    t, a = e["type"], e.get("actor")
    if t == "dora":
        return [e.get("dora_marker")]
    if a == "P0":
        return [e.get("pai")] if t == "tsumo" else []
    if t in ("dahai", "kakan"):
        return [e.get("pai")]
    if t in ("chi", "pon", "daiminkan", "ankan"):
        return list(e.get("consumed") or [])
    return []


def to_mjai(ev, sm):
    t = ev["type"]
    if t == "start_kyoku":
        return {"type": t, "bakaze": ev["bakaze"],
                "dora_marker": ev["dora_marker"], "kyoku": ev["kyoku"],
                "honba": ev["honba"], "kyotaku": ev["kyotaku"],
                "oya": ev["oya"], "scores": ev["scores"],
                "tehais": ev["tehais"]}
    out = {"type": t}
    if t in ("tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan",
             "reach", "reach_accepted", "hora"):
        out["actor"] = _actor(sm, ev.get("actor"))
    if t in ("chi", "pon", "daiminkan", "hora"):
        out["target"] = _actor(sm, ev.get("target"))
    if t in ("tsumo", "dahai", "chi", "pon", "daiminkan", "kakan"):
        out["pai"] = ev["pai"]
    if t == "dahai":
        out["tsumogiri"] = bool(ev["tsumogiri"])
    if t in CONSUMED_LEN:
        out["consumed"] = list(ev["consumed"])
    if t == "dora":
        out["dora_marker"] = ev["dora_marker"]
    return out


class HandStream:
    def __init__(self, hand_id):
        self.hand_id = hand_id
        self.state = "WAITING_START"
        self.reason = None
        self.sent_ids = []
        self.sent_versions = {}
        self.sm = None
        self.hand = None          # P0 手牌帳(基本牌名)
        self.seen = Counter()     # 已送出、看得見的牌(每種最多 4 張、赤最多 1 張)
        self.last_sent = None     # 這一局上一次送出的時間
        self.dora_hold = None     # 槓之後在等新寶牌:{"kind", "actor", "armed"}
        self.noticed = {}
        self.handled = set()      # 已送出或已略過的事件
        self.sent_p0 = set()      # 已送出的 P0 事件(被更正就要重新同步)
        self.corr_noted = set()
        self.p0_turn = False      # P0 手牌是 3n+2(該打牌)
        self.mortal_started = False   # 這一局 Mortal 已經收過 start_kyoku
        self.resync_tried = {}    # 錨點事件 -> 上次失敗的理由(同一個理由只印一次)


class MortalGuard:
    def __init__(self, sink=None, block_after_ms=BLOCK_AFTER_MS, log=None):
        self.sink = sink                  # callable(dict) 或 None
        self.block_after_ms = block_after_ms
        self.log = log or (lambda s: None)
        self.hands = {}
        self.game_started = False
        self.self_actor = None
        self.delivery = []                # 稽核紀錄(delivery.ndjson)
        self.stats = Counter()
        self.p0_concealed = None          # callable(frame) -> 畫面上的 P0 暗牌(pipeline 給)
        self.resync_count = 0

    # ------------------------------------------------------------------
    def _send(self, hs, ev, msg, now):
        if now is not None:
            hs.last_sent = now
        rec = {"run_hand": hs.hand_id, "event_id": ev["event_id"] if ev
               else None, "type": msg["type"],
               "t_first_ms": ev.get("t_first_ms") if ev else None,
               "t_sent_ms": now,
               "latency_ms": (round(now - ev["t_first_ms"], 1)
                              if ev and ev.get("t_first_ms") is not None
                              and now is not None else None),
               "send_status": "SENT", "serialized": json.dumps(msg)}
        self.delivery.append(rec)
        self.stats["sent"] += 1
        if self.sink:
            self.sink(msg)

    def _stop(self, hs, state, reason, now):
        if hs.state in STOPPED:
            return
        hs.state, hs.reason = state, reason
        self.stats[state] += 1
        self.delivery.append({"run_hand": hs.hand_id, "send_status": state,
                              "blocked_reason": reason, "t_ms": now})
        self.log(f"⛔ {hs.hand_id} {state}:{reason}")

    def _desync(self, hs, reason, now):
        if hs.state in STOPPED or hs.state == "DESYNC":
            return
        hs.state, hs.reason = "DESYNC", reason
        self.stats["DESYNC"] += 1
        self.delivery.append({"run_hand": hs.hand_id, "send_status": "DESYNC",
                              "blocked_reason": reason, "t_ms": now})
        self.log(f"⚠️ {hs.hand_id} 對不上:{reason} —— 先不送,"
                 f"下次輪到你摸牌時從畫面重新同步")

    def _skip(self, hs, e, why, now):
        hs.handled.add(e["event_id"])
        self.stats["skipped"] += 1
        self.delivery.append({"run_hand": hs.hand_id,
                              "event_id": e["event_id"], "type": e["type"],
                              "send_status": "SKIPPED", "blocked_reason": why,
                              "t_ms": now})
        self.log(f"↷ {hs.hand_id} 略過 {e['type']} {e.get('actor') or ''}"
                 f"({e['event_id']}):{why}")

    # ------------------------------------------------------------------
    def pump(self, ledger, hand_id, now, final=False):
        """看一次事件帳,送出能送的。回傳這一次送出的 mjai 事件。

        final=True:這一局已經被新的一局取代(pipeline 換局了)。不再等任何
        欄位:能送的照順序送、缺欄位的略過,最後一定送 end_kyoku 收尾 ——
        不然新局的 start_kyoku 會直接疊在沒結束的舊局上。"""
        hs = self.hands.get(hand_id)
        if hs is None:
            hs = self.hands[hand_id] = HandStream(hand_id)
        if hs.state in STOPPED:
            return []
        before = len(self.delivery)
        evs = ledger.ordered(hand_id)

        # --- start_kyoku -------------------------------------------------
        if hs.state == "WAITING_START":
            st = next((e for e in evs if e["type"] == "start_kyoku"), None)
            if final and (st is None or not st.get("export_eligible")):
                hs.state = "DONE"
                return []
            if st is None:
                return []
            if not st.get("export_eligible"):
                # 起手牌一直倒推不出來:等 P0 摸牌時直接從畫面同步
                self._try_resync(hs, ledger, evs, now, quiet=True)
                return self._new(before)
            sm = st["seat_map"]
            if self.self_actor is None:
                self.self_actor = sm["P0"]
                if hasattr(self.sink, "start"):
                    self.sink.start(self.self_actor)
            elif sm["P0"] != self.self_actor:
                self._stop(hs, "BLOCKED_START",
                           f"P0 的 actor 從 {self.self_actor} 變成 {sm['P0']}"
                           f"(整場固定,這一局的自風或局數讀錯了)", now)
                return []
            hs.sm = sm
            me = sm["P0"]
            mine = st["tehais"][me]
            if "?" in mine or len(mine) != 13:
                self._desync(hs, "自己的起手牌不完整", now)
                return self._new(before)
            hs.hand = Counter(base_pai(p) for p in mine)
            hs.seen = Counter()
            why = self._witness(hs, list(mine) + [st["dora_marker"]])
            if why:
                self._desync(hs, why, now)
                return self._new(before)
            if not self.game_started:
                self._send(hs, None, {"type": "start_game"}, now)
                self.game_started = True
            self._send(hs, st, to_mjai(st, sm), now)
            hs.sent_ids.append(st["event_id"])
            hs.handled.add(st["event_id"])
            hs.mortal_started = True
            hs.p0_turn = False
            hs.state = "RUNNING"

        if hs.state == "RUNNING" and getattr(self.sink, "dead", False):
            self._desync(hs, "Mortal 程序停了", now)
        if hs.state == "DESYNC":
            if final:
                self._close(hs, now)
            else:
                self._try_resync(hs, ledger, evs, now)
            return self._new(before)

        body = [e for e in evs if e["type"] not in SKIP
                and not (e["status"] == "CANCELLED" and e["type"] != "reach")]

        # --- 已送出的部分有沒有被改寫 ----------------------------------
        for sid in hs.sent_ids[1:]:
            if sid in hs.corr_noted:
                continue
            e = ledger.get(sid)
            gone = e is None or (e["status"] == "CANCELLED"
                                 and e["type"] != "reach")
            if not gone and _version(e) == hs.sent_versions.get(sid):
                continue
            what = (f"{e['type']} {e.get('actor') or ''}" if e else "")
            if sid in hs.sent_p0 and not final:
                self._desync(hs, f"已送出的 {what}({sid})後來被"
                             + ("取消" if gone else "更正"), now)
                self._try_resync(hs, ledger, evs, now)
                return self._new(before)
            hs.corr_noted.add(sid)
            self.stats["correction_ignored"] += 1
            self.log(f"↷ {hs.hand_id} 已送出的 {what}({sid})後來被"
                     + ("取消" if gone else "更正")
                     + " —— 別家的牌,Mortal 那邊不改,照常往下送")

        # --- 往下送 ------------------------------------------------------
        for e in body:
            if e["event_id"] in hs.handled:
                continue
            if e["type"] == "event_gap":
                hs.handled.add(e["event_id"])
                self.stats["gap_ignored"] += 1
                self.delivery.append({"run_hand": hs.hand_id,
                                      "event_id": e["event_id"],
                                      "send_status": "GAP_IGNORED",
                                      "blocked_reason": f"預期 "
                                      f"{e.get('expected')},看到 "
                                      f"{e.get('observed')}", "t_ms": now})
                self.log(f"↷ {hs.hand_id} 這裡漏了事件(預期 "
                         f"{e.get('expected')},看到 {e.get('observed')})"
                         f"—— 不管它,照常往下送")
                continue
            # --- 槓寶牌:在對的位置等它(v7.1;實機 H007 兩次暗槓)---
            # 寶牌要投票,讀到時事件已經往下送了,會變成「插到已送出的前面」。
            # 所以槓送出後就在寶牌該出現的位置停下來等:
            #   暗槓          槓之後馬上
            #   大明槓/加槓   槓的那一家打出下一張之後
            dh = hs.dora_hold
            if dh and dh["armed"] and e["type"] != "dora" and \
                    e["type"] not in ("hora", "ryukyoku", "end_kyoku"):
                since = hs.last_sent or 0
                if not final and (now is None or now - since <= DORA_WAIT_MS):
                    break
                self.log(f"↷ {hs.hand_id} {dh['kind']} 之後等了 "
                         f"{(now - since) / 1000:.1f}s 還沒讀到新寶牌 —— "
                         f"不等了(Mortal 少知道一張寶牌)")
                hs.dora_hold = None
            ok = e.get("export_eligible") or (
                e["type"] == "reach" and e["status"] == "CANCELLED")
            if ok and e["type"] == "dahai" and e.get("actor") == "P0" and \
                    not (e.get("evidence") or {}).get("river"):
                # P0 的打牌:牌面可能先由手牌差分知道,但要等牌河那一格定案才送
                # —— 飛行中的假格子不會定案(會被取消),送出去就收不回來了
                ok = False
            if ok and e["type"] == "dahai" and \
                    e.get("call_status") == "CLAIMED" and any(
                        not (ledger.get(k) or {}).get("export_eligible")
                        for k in e.get("claimed_by", [])):
                # ★ 被吃/碰的那張先不送,等鳴牌讀出來:鳴牌區可能會修正它的牌面
                #   (2026-10-29 實機:立直特效中把 9m 讀成 5mr,吃了之後才看出來)
                ok = False
            if not ok:
                # 從「輪到它」開始算:它出現之後、前一筆送出之後
                # (等 start_kyoku 那段不算在它頭上)
                since = max(x for x in (e.get("t_first_ms"), hs.last_sent, 0)
                            if x is not None)
                age = (now - since) if now is not None else 0
                limit = self.block_after_ms
                if e["type"] == "dahai" and e.get("actor") == "P0":
                    limit = max(limit, 20000)    # P0 的切法/牌河要等手牌穩定
                if e["type"] == "tsumo" and e.get("actor") == "P0":
                    # ★ 2026-10-29 實機:P0 摸牌讀不出牌名 -> 6 秒就整局停送。
                    #   你在等建議、不會打牌 -> 手牌差分也補不出來 -> 卡死。
                    #   P0 的摸牌一律等到打出去(打完會由手牌差分/牌河補上)。
                    limit = P0_TSUMO_WAIT_MS
                    if age > NOTICE_MS and not hs.noticed.get(e["event_id"]):
                        hs.noticed[e["event_id"]] = True
                        self.log("⏳ 你這次摸的牌讀不出來,Mortal 這巡給不了建議"
                                 " —— 請自己打,打完會自動接上")
                if final:
                    self._skip(hs, e, "已經換局,不等了", now)
                    continue
                if age > limit:
                    why = (f"等了 {age / 1000:.1f}s 還缺 "
                           f"{', '.join(e.get('missing') or [])}")
                    if e.get("actor") == "P0" and e["type"] in P0_TYPES:
                        self._desync(hs, f"{e['type']} P0({e['event_id']})"
                                     + why, now)
                        break
                    self._skip(hs, e, why, now)
                    continue
                break
            why = self._own_hand(hs, e)
            if why:
                if final:
                    self._skip(hs, e, why, now)
                    continue
                self._desync(hs, why, now)
                break
            why = self._witness(hs, _visible(e))
            if why:
                if e.get("actor") == "P0" and e["type"] in P0_TYPES \
                        and not final:
                    self._desync(hs, why, now)
                    break
                self._skip(hs, e, why, now)
                continue
            msg = to_mjai(e, hs.sm)
            self._send(hs, e, msg, now)
            hs.sent_ids.append(e["event_id"])
            hs.handled.add(e["event_id"])
            if e.get("actor") == "P0" and e["type"] in P0_TYPES:
                hs.sent_p0.add(e["event_id"])
            hs.sent_versions[e["event_id"]] = _version(e)
            t_ = e["type"]
            if t_ == "dora":
                hs.dora_hold = None
            elif t_ == "ankan":
                hs.dora_hold = {"kind": t_, "actor": e.get("actor"),
                                "armed": True}
            elif t_ in ("daiminkan", "kakan"):
                hs.dora_hold = {"kind": t_, "actor": e.get("actor"),
                                "armed": False}
            elif t_ == "dahai" and hs.dora_hold and \
                    hs.dora_hold["actor"] == e.get("actor"):
                hs.dora_hold["armed"] = True
            if e["type"] == "end_kyoku":
                hs.state = "DONE"
                break
        if final and hs.state not in STOPPED:
            self._close(hs, now)
        if hs.state == "DESYNC":
            # 剛好卡在 P0 摸牌上(例如摸到的牌讓帳對不上):馬上試著同步
            self._try_resync(hs, ledger, evs, now)
        return self._new(before)

    def _close(self, hs, now):
        """換局了:這一局收尾(Mortal 收過 start_kyoku 才需要 end_kyoku)。"""
        if hs.mortal_started:
            self._send(hs, None, {"type": "end_kyoku"}, now)
        hs.state = "DONE"
        self.stats["closed_by_new_hand"] += 1

    # ------------------------------------------------------------------
    def _try_resync(self, hs, ledger, evs, now, quiet=False):
        """DESYNC(或起手牌一直不齊)時:輪到 P0 摸牌、牌面讀得到、畫面上的暗牌
        穩定 -> 重建這一局送給 Mortal。條件不夠就什麼都不做(下一幀再試)。"""
        st = next((e for e in evs if e["type"] == "start_kyoku"), None)
        end = next((e for e in evs if e["type"] == "end_kyoku"
                    and e.get("export_eligible")), None)
        if end is not None:
            if hs.mortal_started:
                self._send(hs, end, {"type": "end_kyoku"}, now)
            hs.state = "DONE"
            return
        if st is None or any(st.get(k) is None for k in START_KEYS):
            return
        body = [e for e in evs if e["type"] not in SKIP
                and e["status"] != "CANCELLED"
                and e["type"] not in ("event_gap",)]
        if not body:
            return
        anc = body[-1]
        p0last = next((e for e in reversed(body) if e.get("actor") == "P0"
                       and e["type"] in P0_TYPES), None)
        if anc["type"] == "tsumo" and anc.get("actor") == "P0" and \
                anc.get("pai") not in (None, "?"):
            hist, ref, after = body[:-1], anc["frame_first"], False
        elif p0last is not None and p0last["type"] == "dahai" and \
                p0last.get("pai"):
            # ★ 2026-11-04 實戰 紀錄\20261104_224154 東3局:立直後摸到的牌
            #   馬上自動打掉,摸牌那一刻太短、牌面也沒讀到 -> 一直等不到
            #   「P0 摸牌」這個同步點,這一局後面都沒建議。
            #   P0 剛打完牌(手牌 3n+1)也是同步點:用打完之後的穩定暗牌。
            hist, ref, after = body, p0last["frame_first"], True
            anc = None
        else:
            return
        key = anc or p0last
        sm = st["seat_map"]
        if self.self_actor is not None and sm["P0"] != self.self_actor:
            self._note_resync(hs, key, f"P0 的 actor 從 {self.self_actor} 變成 "
                              f"{sm['P0']}(這一局的自風或局數讀錯了)", quiet)
            return
        conc = self.p0_concealed(ref, after=after) \
            if self.p0_concealed else None
        if conc is None:
            self._note_resync(hs, key, "畫面上的暗牌還不穩定", True)
            return
        plan, why = build_resync(st, hist, anc, conc)
        if plan is None:
            self._note_resync(hs, key, why, quiet)
            return

        # --- 送出 -----------------------------------------------------------
        if self.self_actor is None:
            self.self_actor = sm["P0"]
            if hasattr(self.sink, "start"):
                self.sink.start(self.self_actor)
        if not self.game_started:
            self._send(hs, None, {"type": "start_game"}, now)
            self.game_started = True
        elif hs.mortal_started:
            self._send(hs, None, {"type": "end_kyoku", "can_act": False}, now)
        for ev, msg in plan["msgs"]:
            self._send(hs, ev, msg, now)
        if anc is not None:
            self._send(hs, anc, to_mjai(anc, sm), now)
        hs.sm = sm
        hs.mortal_started = True
        hs.hand = Counter(base_pai(p) for p in list(conc)
                          + ([anc["pai"]] if anc else []))
        hs.seen = plan["seen"]
        hs.p0_turn = anc is not None
        hs.dora_hold = None
        hs.handled = {e["event_id"] for e in evs}
        hs.sent_ids = [st["event_id"]] + plan["ids"] + \
            ([anc["event_id"]] if anc else [])
        hs.sent_versions = {e["event_id"]: _version(e) for e in evs}
        hs.sent_p0 = {e["event_id"] for e in body
                      if e.get("actor") == "P0" and e["type"] in P0_TYPES}
        hs.corr_noted = set()
        old = hs.reason
        hs.state, hs.reason = "RUNNING", None
        self.resync_count += 1
        self.stats["resync"] += 1
        self.delivery.append({"run_hand": hs.hand_id, "send_status": "RESYNC",
                              "blocked_reason": old, "t_ms": now,
                              "tehai": plan["tehai"],
                              "dropped": plan["dropped"]})
        self.log(f"🔄 {hs.hand_id} 已從畫面重新同步(起手牌倒推為 "
                 f"{' '.join(plan['tehai'])};重送 {len(plan['msgs'])} 筆"
                 + (f",略過 {len(plan['dropped'])} 筆讀錯的牌"
                    if plan["dropped"] else "") + ")")

    def _note_resync(self, hs, anc, why, quiet):
        if hs.resync_tried.get(anc["event_id"]) == why:
            return
        hs.resync_tried[anc["event_id"]] = why
        if not quiet:
            self.log(f"⏳ {hs.hand_id} 這次還不能重新同步:{why}")

    def _new(self, before):
        return [json.loads(r["serialized"]) for r in self.delivery[before:]
                if r.get("send_status") == "SENT"]

    # ------------------------------------------------------------------
    def _own_hand(self, hs, e):
        """P0 手牌帳 + 張數節奏(3n+1 摸/鳴 -> 3n+2 打)。回傳擋下來的理由。

        節奏錯了 libriichi 會 assert panic("tehai is not 3n+2"),所以跟牌名
        一樣要先擋。"""
        if e.get("actor") != "P0":
            return None
        t = e["type"]
        if t == "tsumo":
            if not e.get("pai") or e["pai"] == "?":
                return "自己的摸牌牌面不知道(送 ? 會讓 libriichi panic)"
            if hs.p0_turn:
                return "P0 連續摸牌(中間少了一次打牌)"
            hs.hand[base_pai(e["pai"])] += 1
            hs.p0_turn = True
            return None
        if t == "dahai":
            if not hs.p0_turn:
                return f"P0 打 {e.get('pai')},但之前沒有摸牌或鳴牌"
            why = self._spend(hs, [e["pai"]], "打")
            if not why:
                hs.p0_turn = False
            return why
        if t in ("chi", "pon", "daiminkan", "ankan"):
            if (t == "ankan") != hs.p0_turn:
                return (f"P0 {t} 的時機不對(手牌張數對不上)")
            if len(e["consumed"]) != CONSUMED_LEN[t]:
                return f"{t} 的 consumed 張數不對:{e['consumed']}"
            why = self._spend(hs, e["consumed"], t)
            if not why:
                # 吃/碰之後要打牌;大明槓/暗槓之後要嶺上摸牌
                hs.p0_turn = t in ("chi", "pon")
            return why
        if t == "kakan":
            if not hs.p0_turn:
                return "P0 加槓的時機不對(手牌張數對不上)"
            why = self._spend(hs, [e["pai"]], "加槓補上")
            if not why:
                hs.p0_turn = False
            return why
        return None

    @staticmethod
    def _witness(hs, pais):
        """看得見的牌超過 4 張(或赤五超過 1 張)-> 擋下。

        ★ 2026-10-31 重播:東4 的槓寶牌 2m 被讀成 1m,1m 變成第 5 張 ->
          Mortal(libriichi)"attempt to witness the fifth 1m" 當掉,之後整場
          都沒有建議。擋下這一局,總比讓 Mortal 整個掛掉好。"""
        add = Counter(base_pai(p) for p in pais if p and p != "?")
        red = Counter(p for p in pais if p and p.endswith("r"))
        for p, n in add.items():
            if hs.seen[p] + n > 4:
                return (f"{p} 會變成第 {hs.seen[p] + n} 張(看得見的牌最多 4 張)"
                        f" —— 某一張牌讀錯了,送出去 Mortal 會當掉")
        for p, n in red.items():
            if hs.seen[p] + n > 1:
                return f"赤 {p} 出現第 2 張 —— 某一張牌讀錯了"
        hs.seen.update(add)
        hs.seen.update(red)
        return None

    @staticmethod
    def _spend(hs, pais, what):
        need = Counter(base_pai(p) for p in pais)
        short = {p: n for p, n in need.items() if hs.hand[p] < n}
        if short:
            return (f"自己要{what} {list(pais)},但手牌帳上"
                    + "、".join(f"{p} 只有 {hs.hand[p]} 張" for p in short)
                    + " —— 起手牌、摸牌、打牌至少有一個讀錯了,"
                      "硬送出去 libriichi 之後會 panic")
        hs.hand.subtract(need)
        return None

    def active(self):
        return [h for h, hs in self.hands.items() if hs.state not in STOPPED]

    def finish(self):
        """檔尾:還在送的局記成「錄影結束」,不算擋下。"""
        for hs in self.hands.values():
            if hs.state not in STOPPED:
                hs.reason = (hs.reason or "") + "(錄影結束時還在這一局)"

    def report(self):
        lines = []
        for h, hs in self.hands.items():
            n = len(hs.sent_ids)
            lines.append(f"    {h}: {hs.state}  已送 {n} 筆"
                         + (f"  ⛔ {hs.reason}" if hs.reason else ""))
        lat = [r["latency_ms"] for r in self.delivery
               if r.get("latency_ms") is not None
               and r["type"] in ("dahai", "chi", "pon", "daiminkan",
                                 "ankan", "kakan", "tsumo")]
        if lat:
            s = sorted(lat)
            lines.append(f"    延遲(第一次看到 -> 送出):中位 "
                         f"{s[len(s) // 2] / 1000:.2f}s,90% "
                         f"{s[int(len(s) * 0.9)] / 1000:.2f}s,最慢 "
                         f"{s[-1] / 1000:.2f}s")
            by = {}
            for r in self.delivery:
                if r.get("latency_ms") is not None:
                    by.setdefault(r["type"], []).append(r["latency_ms"])
            for t, v in sorted(by.items()):
                v.sort()
                lines.append(f"      {t:<14} 中位 {v[len(v) // 2] / 1000:.2f}s"
                             f"  最慢 {v[-1] / 1000:.2f}s  ({len(v)} 筆)")
        if self.self_actor is not None:
            lines.append(f"    P0 的 Mortal player_id = {self.self_actor}")
        return lines


def build_resync(st, hist, anc, conc):
    """重新同步要送給 Mortal 的整段(不含 end_kyoku、最後那次 P0 摸牌)。

    st    start_kyoku 事件(場況欄位要齊,起手牌可以沒有)
    hist  這一局到 anc 之前的事件(已排序、不含取消/缺口)
    anc   P0 這次的摸牌(牌面已知)
    conc  畫面上 P0 的暗牌(不含這次摸到的那張),3n+1 張

    P0 每一次「摸 X -> 打 Y」改送「摸 Y -> 打 Y」:手牌不變,牌河/振聽照實;
    鳴牌後的打牌、鳴牌用掉的牌一定是起手就在手上 -> 起手牌 = 暗牌 + 這些。
    回傳 (plan, None) 或 (None, 理由)。"""
    sm = st["seat_map"]
    A = lambda seat: None if seat is None else sm.get(seat)
    msgs = []            # [ev, msg, 看得見的牌, 可丟(別家打牌)]
    start_add = []
    pend = False
    after_call = False   # 吃/碰之後那一打:牌本來就在手上(不是摸來的)
    nca = {"can_act": False}
    for e in hist:
        t, a = e["type"], e.get("actor")
        if t == "dora":
            if e.get("dora_marker"):
                msgs.append([e, {"type": "dora",
                                 "dora_marker": e["dora_marker"], **nca},
                             [e["dora_marker"]], False])
            continue
        if t in ("tsumo", "dahai", "chi", "pon", "daiminkan", "ankan",
                 "kakan", "reach", "reach_accepted") and A(a) is None:
            continue
        if a == "P0":
            p = e.get("pai")
            if t == "tsumo":
                # 前一次摸牌之後沒有打牌就又摸:前一筆是假的(發牌動畫)
                # 或是漏看了一次打牌 -> 都以畫面暗牌為準,前一筆不送
                pend = True
            elif t == "dahai":
                if not p:
                    return None, "P0 有一張打牌牌面不知道"
                if pend or not after_call:
                    # 摸 X -> 打 Y 改成 摸 Y -> 打 Y;沒看到摸牌(立直後自動
                    # 摸切太快)也一樣補一筆
                    msgs.append([None, {"type": "tsumo", "actor": A(a),
                                        "pai": p, **nca}, [p], False])
                else:
                    start_add.append(p)
                after_call = False
                msgs.append([e, {"type": "dahai", "actor": A(a), "pai": p,
                                 "tsumogiri": bool(e.get("tsumogiri")),
                                 **nca}, [], False])
                pend = False
            elif t in ("chi", "pon", "daiminkan"):
                c = e.get("consumed")
                if not c or len(c) != CONSUMED_LEN[t] or not p or \
                        A(e.get("target")) is None:
                    # 讀不完整的鳴牌:多半是假的(別家的宣告被算到 P0)。先當作
                    # 沒發生;如果是真的,下面「起手牌 13 張」的檢查會擋下來
                    continue
                pend = False        # 鳴牌前那次「摸牌」一定是假的:不送
                after_call = t in ("chi", "pon")
                start_add += list(c)
                msgs.append([e, {"type": t, "actor": A(a),
                                 "target": A(e.get("target")), "pai": p,
                                 "consumed": list(c), **nca}, [], False])
            elif t == "ankan":
                c = e.get("consumed")
                if not c or len(c) != 4 or not pend:
                    return None, "P0 的暗槓讀不完整"
                msgs.append([None, {"type": "tsumo", "actor": A(a),
                                    "pai": c[0], **nca}, [c[0]], False])
                start_add += list(c[1:])
                msgs.append([e, {"type": "ankan", "actor": A(a),
                                 "consumed": list(c), **nca}, [], False])
                pend = False
            elif t == "kakan":
                if not p or not pend:
                    return None, "P0 的加槓讀不完整"
                msgs.append([None, {"type": "tsumo", "actor": A(a),
                                    "pai": p, **nca}, [p], False])
                msgs.append([e, {"type": "kakan", "actor": A(a), "pai": p,
                                 **nca}, [], False])
                pend = False
            elif t in ("reach", "reach_accepted"):
                msgs.append([e, {"type": t, "actor": A(a), **nca}, [], False])
            continue
        # --- 別家:讀不完整的就不送 -------------------------------------
        if t == "tsumo":
            msgs.append([e, {"type": "tsumo", "actor": A(a), "pai": "?",
                             **nca}, [], False])
        elif t == "dahai" and e.get("pai"):
            msgs.append([e, {"type": "dahai", "actor": A(a), "pai": e["pai"],
                             "tsumogiri": bool(e.get("tsumogiri")), **nca},
                         [e["pai"]], True])
        elif t in ("chi", "pon", "daiminkan") and e.get("pai") and \
                A(e.get("target")) is not None and \
                len(e.get("consumed") or ()) == CONSUMED_LEN[t]:
            msgs.append([e, {"type": t, "actor": A(a),
                             "target": A(e.get("target")), "pai": e["pai"],
                             "consumed": list(e["consumed"]), **nca},
                         list(e["consumed"]), False])
        elif t == "ankan" and len(e.get("consumed") or ()) == 4:
            msgs.append([e, {"type": t, "actor": A(a),
                             "consumed": list(e["consumed"]), **nca},
                         list(e["consumed"]), False])
        elif t == "kakan" and e.get("pai"):
            msgs.append([e, {"type": t, "actor": A(a), "pai": e["pai"],
                             **nca}, [e["pai"]], False])
        elif t in ("reach", "reach_accepted"):
            msgs.append([e, {"type": t, "actor": A(a), **nca}, [], False])
    # 最後還 pend:前一次摸牌之後的打牌沒看到(或那次摸牌是假的)-> 不送它
    tehai = sorted(list(conc) + start_add)
    if len(tehai) != 13:
        return None, (f"畫面暗牌 {len(conc)} 張 + 之前打掉/鳴牌用掉 "
                      f"{len(start_add)} 張 = {len(tehai)},不是 13 張")

    # --- 同一種牌看得見的不超過 4 張(赤五 1 張):超過就丟別家的打牌 -------
    def count():
        base, red = Counter(), Counter()
        allp = tehai + [st["dora_marker"]] + ([anc["pai"]] if anc else []) + \
            [p for m in msgs for p in m[2]]
        for p in allp:
            base[base_pai(p)] += 1
            if p.endswith("r"):
                red[p] += 1
        return base, red
    dropped = []
    while True:
        base, red = count()
        over = [p for p, n in base.items() if n > 4] + \
            [p for p, n in red.items() if n > 1]
        if not over:
            break
        p = over[0]
        hit = [m for m in msgs if m[3] and (
            base_pai(m[2][0]) == p if not p.endswith("r") else m[2][0] == p)]
        if not hit:
            return None, f"{p} 看得見超過上限,而且是 P0 自己的牌 —— 暗牌讀錯了"
        m = hit[-1]
        msgs.remove(m)
        dropped.append(m[1]["pai"])
    base, red = count()
    tehais = [["?"] * 13 for _ in range(4)]
    tehais[sm["P0"]] = list(tehai)
    start = {"type": "start_kyoku", "bakaze": st["bakaze"],
             "dora_marker": st["dora_marker"], "kyoku": st["kyoku"],
             "honba": st["honba"], "kyotaku": st["kyotaku"],
             "oya": st["oya"], "scores": st["scores"], "tehais": tehais,
             **nca}
    seen = Counter(base)
    seen.update(red)
    return {"msgs": [(None, start)] + [(m[0], m[1]) for m in msgs],
            "ids": [m[0]["event_id"] for m in msgs if m[0] is not None],
            "seen": seen, "tehai": tehai, "dropped": dropped}, None


# ---------------------------------------------------------------------------
# 送到哪裡
# ---------------------------------------------------------------------------
class FileSink:
    """mjai 串流寫成檔(一行一個事件)。可以拿去餵 Mortal 或 validate_mjai.py。"""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg):
        self.fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


class MultiSink:
    def __init__(self, *sinks):
        self.sinks = [x for x in sinks if x]

    def start(self, player_id):
        for x in self.sinks:
            if hasattr(x, "start"):
                x.start(player_id)

    def __call__(self, msg):
        for x in self.sinks:
            x(msg)

    @property
    def dead(self):
        return any(getattr(x, "dead", False) for x in self.sinks)

    def close(self):
        for x in self.sinks:
            x.close()


def _trigger(ev, pid):
    """回覆是針對哪一筆:「你摸 北 → 」「上家打 5p → 」。"""
    from mortal_view import tile_zh
    if not ev:
        return ""
    who = {0: "你", 1: "下家", 2: "對家", 3: "上家"}.get(
        (ev.get("actor", pid) - pid) % 4, "")
    t = ev.get("type")
    verb = {"tsumo": "摸", "dahai": "打", "chi": "吃", "pon": "碰",
            "reach": "立直", "kakan": "加槓"}.get(t)
    if not verb:
        return ""
    return f"{who}{verb}{(' ' + tile_zh(ev['pai'])) if ev.get('pai') else ''} → "


# --mortal docker 的預設指令:mortal.pth 放在 專案\Model,掛成容器裡的 /mnt
# (docker 映像 mortal 的 config.toml 寫的是 /mnt/mortal.pth)
# ★ 2026-10-31:映像裡的 mortal.py 是沒改過的版本(torch.load weights_only=True),
#   載這顆 mortal.pth 會 UnpicklingError。把專案 Mortal\mortal\mortal.py(已改成
#   weights_only=False)蓋進容器,不必重建映像,而且用的一定是專案裡的那一份。
DEFAULT_MORTAL_CMD = ('docker run -i --rm -v "{model_dir}:/mnt" '
                      '-v "{mortal_py}:/src/mortal/mortal.py" mortal {player_id}')


# --mortal native:不用 Docker,用目前這個 Python 直接跑(Mac 版預設)。
#   需要先編好 Mortal/libriichi.so(Mac:建立Mortal.command),見 Mortal/run_native.py
NATIVE_MORTAL_CMD = '"{python}" "{run_native}" {player_id}'


def mortal_command(cmd):
    import _paths
    import os
    import sys
    if cmd.strip().lower() == "docker":
        cmd = DEFAULT_MORTAL_CMD
    elif cmd.strip().lower() == "native":
        cmd = (NATIVE_MORTAL_CMD.replace("{python}", sys.executable)
               .replace("{run_native}", os.path.join(
                   _paths.PROJECT_DIR, "Mortal", "run_native.py")))
    mortal_py = os.path.join(_paths.PROJECT_DIR, "Mortal", "mortal", "mortal.py")
    return (cmd.replace("{model_dir}", _paths.MODEL_DIR)
            .replace("{mortal_py}", mortal_py))


class ProcessSink:
    """即時送給 Mortal 程序(stdin 一行一個事件),Mortal 的回覆另一條執行緒收。

    ★ Mortal 的回覆是「建議」,不是遊戲裡發生的事(v7.1 第 3 節第 6 點):
      這裡只印出來、存檔,絕不回灌成事件。
    例:--mortal docker(= DEFAULT_MORTAL_CMD,掛載 專案\\Model)
        {player_id} 會換成 P0 的 actor(從第一局 start_kyoku 算出來)。

    Mortal 只在「P0 能做決定」的事件後才回覆(實測 576 筆事件回 94 筆)。
    每一筆回覆記下「收到時最後送出的是第幾筆」(after),重播報告靠它對照。
    sync=True(重播用):送出 P0 有可能要決定的事件後,先等回覆再送下一筆。
      重播一下子送完幾百筆,不等的話回覆會全部擠在最後,對不回去。
      即時錄製不用等:事件之間隔幾百毫秒,Mortal 幾毫秒就回了。
    """

    SYNC_WAIT_SURE = 5.0    # P0 摸牌/鳴牌/立直之後一定會回(秒)
    SYNC_WAIT_MAYBE = 0.3   # 別家打牌之後,能鳴/能和才會回

    def __init__(self, cmd, reply_path, on_reply=print, sync=False):
        self.cmd = cmd
        self.reply_path = reply_path
        self.on_reply = on_reply
        self.sync = sync
        self.proc = None
        self.pending = []
        self.stream = []          # 送出去的每一筆
        self.pairs = []           # (after 索引, reply)
        self.pid = None
        self._q = None
        self.last_actionable = None
        self.dead = False
        self._out = None
        self.on_reply_obj = None    # agent.py:收到回覆時另外給(觸發事件, 回覆 dict)
        # 立直的預先詢問(見 _speculate_reach)
        import threading
        self.lock = threading.RLock()
        self.speculate = not sync
        self.spec = None            # None / "pending"(已送假立直、等回覆)/ "done"
        self.drop_replies = 0       # 撤回假立直後,還在路上的回覆要丟掉

    def start(self, player_id):
        import queue
        import subprocess
        import threading
        import time
        from mortal_view import describe
        self.pid = player_id
        cmd = mortal_command(self.cmd).replace("{player_id}", str(player_id))
        self.proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     encoding="utf-8", bufsize=1)
        if getattr(self, "_out", None) is None:
            self._out = open(self.reply_path, "w", encoding="utf-8")
        out = self._out
        self._q = queue.Queue()
        self.dead = False
        self.spec, self.drop_replies = None, 0

        def pump():
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    self.on_reply("Mortal:" + line)
                    continue
                with self.lock:
                    if self.drop_replies:
                        self.drop_replies -= 1
                        continue
                    spec_reply = self.spec == "pending"
                    if spec_reply:
                        self.spec = "done"
                if spec_reply:
                    # 假立直的回覆 = 立直後要打哪一張
                    trig = {"type": "reach", "actor": self.pid,
                            "speculative": True}
                    out.write(json.dumps({"after": len(self.stream) - 1,
                                          "t": time.time(), "event": trig,
                                          "reply": r}, ensure_ascii=False)
                              + "\n")
                    out.flush()
                    self.on_reply("🀄 立直的話 → " + describe(r))
                    if self.on_reply_obj is not None:
                        try:
                            self.on_reply_obj({"type": "reach_spec",
                                               "actor": self.pid}, r)
                        except Exception as e:
                            self.on_reply(f"⚠️ 字幕更新失敗:{e}")
                    self._q.put(r)
                    continue
                # 回覆對到「最後一筆 P0 可能要決定的事件」:即時錄製時回覆到達前
                # 下一筆(例如下家摸牌)可能已經送出了(2026-10-29 實機:
                # 「對家摸 ? → 不鳴」其實是針對前一筆下家打 6s)
                i = self.last_actionable if self.last_actionable is not None \
                    else len(self.stream) - 1
                self.pairs.append((i, r))
                out.write(json.dumps({"after": i, "t": time.time(),
                                      "event": self.stream[i] if i >= 0
                                      else None, "reply": r},
                                     ensure_ascii=False) + "\n")
                out.flush()
                self.on_reply("🀄 " + _trigger(self.stream[i] if i >= 0
                                               else None, self.pid)
                              + describe(r))
                if self.on_reply_obj is not None:
                    try:
                        self.on_reply_obj(self.stream[i] if i >= 0 else None,
                                          r)
                    except Exception as e:      # 字幕出錯不能拖垮 Mortal
                        self.on_reply(f"⚠️ 字幕更新失敗:{e}")
                self._q.put(r)
                if r.get("type") == "reach" and r.get("actor") == self.pid:
                    self._speculate_reach(i)
        threading.Thread(target=pump, daemon=True).start()
        for m in self.pending:
            self._write(m)
        self.pending = []

    def _speculate_reach(self, trigger_index):
        """Mortal 建議立直:先偷偷送一個「我立直了」給它,它就會回「立直後打哪張」。

        ★ 2026-11-04 使用者:立直後要打哪一張,它不會講。Mortal 只有在收到
          reach 之後才回宣言牌;畫面上的立直宣言要等你點了牌才看得到,那時
          才問已經來不及(實測只早 0.9 秒)。
        之後:
          你真的立直     -> 守門器送來的 reach 不再送(Mortal 已經收過了)
          你沒有立直     -> 撤回:end_kyoku + 這一局送過的全部重送一次
                            (can_act=false,Mortal 不回),狀態回到立直之前
        """
        with self.lock:
            if not self.speculate or self.spec or self.dead or \
                    self.proc is None or self.proc.poll() is not None:
                return
            if trigger_index != len(self.stream) - 1:
                return          # 摸牌之後已經又送了別的:來不及,不猜
            try:
                self.proc.stdin.write(json.dumps(
                    {"type": "reach", "actor": self.pid}) + "\n")
                self.proc.stdin.flush()
                self.spec = "pending"
            except (BrokenPipeError, OSError):
                pass

    def _rollback_spec(self):
        """撤回假立直:Mortal 狀態回到「送假立直之前」。"""
        if self.spec == "pending":
            self.drop_replies += 1          # 它對假立直的回覆還沒到:到了丟掉
        self.spec = None
        start = max((i for i, m in enumerate(self.stream)
                     if m.get("type") == "start_kyoku"), default=None)
        if start is None:
            return
        msgs = [{"type": "end_kyoku", "can_act": False}] + \
            [dict(m, can_act=False) for m in self.stream[start:]]
        try:
            for m in msgs:
                self.proc.stdin.write(json.dumps(m) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _expect(self, msg):
        if msg.get("can_act") is False:
            return 0            # 重新同步時重送的歷史:Mortal 不會回
        t, a = msg.get("type"), msg.get("actor")
        if a == self.pid and t in ("tsumo", "chi", "pon", "reach"):
            return self.SYNC_WAIT_SURE
        if a is not None and a != self.pid and t in ("dahai", "kakan"):
            return self.SYNC_WAIT_MAYBE
        return 0

    def _write(self, msg):
        import queue
        with self.lock:
            if self.spec:
                if msg.get("type") == "reach" and \
                        msg.get("actor") == self.pid:
                    # 你真的立直了:Mortal 已經收過(假立直),不再送
                    self.spec = None
                    self.stream.append(msg)
                    return
                self._rollback_spec()
            w = self._write_locked(msg)
        # 重播(sync)時等回覆:一定要在鎖外面等,收回覆的執行緒也要用這把鎖
        if w:
            try:
                self._q.get(timeout=w)
            except queue.Empty:
                pass

    def _write_locked(self, msg):
        """寫進 Mortal;回傳要等回覆幾秒(只有 sync 重播才會 > 0)。"""
        if self.dead or self.proc.poll() is not None:
            # ★ Mortal 當掉了(例如 libriichi 判定牌數不合理):這一局剩下的
            #   不送;下一局 start_kyoku 時重開一個 Mortal(先送 start_game)
            if not self.dead:
                self.dead = True
                self.on_reply("⚠️ Mortal 程序結束了,這一局之後的建議沒了;"
                              "下一局開始時會自動重開")
            if msg.get("type") != "start_kyoku":
                return 0
            self.on_reply("🔄 重開 Mortal")
            self.start(self.pid)
            self._write_locked({"type": "start_game"})
        try:
            self.stream.append(msg)
            if self._expect(msg):
                self.last_actionable = len(self.stream) - 1
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            if not self.dead:
                self.dead = True
                self.on_reply(f"⚠️ Mortal 程序寫不進去({e}),這一局之後的建議"
                              "沒了;下一局開始時會自動重開")
            return 0
        w = self._expect(msg) if self.sync else 0
        if w and not self.pairs:
            w = max(w, 60.0)        # 第一次:Docker 啟動 + 載入模型
        return w

    def __call__(self, msg):
        if self.proc is None:
            self.pending.append(msg)
        else:
            self._write(msg)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
