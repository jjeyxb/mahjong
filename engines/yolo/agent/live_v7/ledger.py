"""本地事件帳(events.ndjson)。可以補欄位、可以更正,但保留每一次改動。

v7.1 第 3 節的「本地事件帳(mutable)」:
    PENDING    佔位(例如牌河剛出現新牌,牌面還沒定案)
    CONFIRMED  需要的欄位都有了
    INFERRED   沒有直接觀測、依規則推出來的(例如漏看的摸牌)
    CANCELLED  後來證明不成立(例如被榮和的立直宣言)
另外兩種特殊事件:
    event_gap  這裡漏了東西,不知道是什麼 —— 不猜
    correction 某一筆被後來的證據改掉,記下舊值、新值、理由

★ 本地事件帳**不是** Mortal 的輸入。export_eligible 只代表「這一筆的欄位
  齊了」;能不能真的送出,要等第 4 步的 Mortal 守門器檢查因果順序。
"""
import itertools

SCHEMA = "v7-local-event-1"

# 每一種事件要齊哪些欄位才算完整(依 Mortal libriichi event.rs)
REQUIRED = {
    "start_kyoku": ("bakaze", "kyoku", "honba", "kyotaku", "oya", "scores",
                    "dora_marker", "tehais"),
    "tsumo": ("actor", "pai"),
    "dahai": ("actor", "pai", "tsumogiri"),
    "chi": ("actor", "target", "pai", "consumed"),
    "pon": ("actor", "target", "pai", "consumed"),
    "daiminkan": ("actor", "target", "pai", "consumed"),
    "ankan": ("actor", "consumed"),
    "kakan": ("actor", "pai", "consumed"),
    "dora": ("dora_marker",),
    "reach": ("actor",),
    "reach_accepted": ("actor",),
    "hora": ("actor", "target"),
    "ryukyoku": (),
    "end_kyoku": (),
}


class Ledger:
    def __init__(self, run_id="RUN"):
        self.run_id = run_id
        self.events = []
        self.by_id = {}
        self._seq = itertools.count(1)

    def add(self, type_, hand_id, t_first_ms, frame, status="PENDING",
            **fields):
        eid = f"{hand_id}-E{next(self._seq):04d}"
        ev = {"schema_version": SCHEMA, "event_id": eid, "hand_id": hand_id,
              "run_id": self.run_id, "type": type_, "status": status,
              "t_first_ms": t_first_ms, "frame_first": frame,
              "t_confirm_ms": None, "version": 1, "notes": [],
              "history": []}
        ev.update(fields)
        self.events.append(ev)
        self.by_id[eid] = ev
        self._refresh(ev)
        return ev

    def get(self, eid):
        return self.by_id.get(eid)

    def update(self, eid, t_ms=None, reason=None, **fields):
        """補欄位。原本是 None 的欄位直接填;已經有值而且不同 -> 記成更正。"""
        ev = self.by_id[eid]
        changed = {}
        for k, v in fields.items():
            old = ev.get(k)
            if old == v:
                continue
            changed[k] = (old, v)
            ev[k] = v
        if changed:
            ev["version"] += 1
            ev["history"].append({"t_ms": t_ms, "reason": reason,
                                  "changed": {k: list(v) for k, v in
                                              changed.items()}})
            corrected = {k: v for k, v in changed.items()
                         if v[0] is not None and k not in
                         ("status", "call_status", "t_confirm_ms",
                          "evidence", "missing", "export_eligible",
                          "type", "t_first_ms", "order_bias")}
            if corrected:
                self.add("correction", ev["hand_id"], t_ms, None,
                         status="CONFIRMED", correction_of=eid,
                         changed={k: list(v) for k, v in corrected.items()},
                         reason=reason)
            self._refresh(ev, t_ms)
        return ev

    def note(self, eid, text):
        self.by_id[eid]["notes"].append(text)

    def _refresh(self, ev, t_ms=None):
        need = REQUIRED.get(ev["type"])
        if need is None:
            ev["missing"] = []
            ev["export_eligible"] = False
            return
        missing = [k for k in need if ev.get(k) is None]
        ev["missing"] = missing
        ev["export_eligible"] = (not missing
                                 and ev["status"] in ("CONFIRMED",
                                                      "INFERRED"))
        if not missing and ev["status"] == "PENDING":
            ev["status"] = "CONFIRMED"
            ev["t_confirm_ms"] = t_ms
            ev["export_eligible"] = True

    def ordered(self, hand_id=None):
        evs = [e for e in self.events if e["type"] != "correction"
               and (hand_id is None or e["hand_id"] == hand_id)]
        return sorted(evs, key=lambda e: (e["hand_id"],
                                          e["t_first_ms"] if e["t_first_ms"]
                                          is not None else float("inf"),
                                          e.get("order_bias", 0),
                                          e["event_id"]))
