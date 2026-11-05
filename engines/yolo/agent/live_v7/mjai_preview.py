"""本地事件帳 -> 「如果現在要送給 Mortal,能送到哪裡」的預覽。

★ 這**不是**第 4 步的 Mortal 守門器,只是先看看資料夠不夠:
    ・照 t_first 的因果順序走
    ・遇到第一筆欄位不齊的事件就停(v7.1:不可越過缺口)
    ・座位換成 mjai actor 需要 start_kyoku 的 seat_map;沒有就整局 BLOCKED_START
用來回答「這段錄影,欄位齊全度卡在哪裡」。
"""
SKIP = ("event_gap", "correction", "kan_self")


def _actor(sm, seat):
    return None if seat is None or sm is None else sm.get(seat)


def to_mjai(ev, sm):
    t = ev["type"]
    if t == "start_kyoku":
        return {"type": t, "bakaze": ev["bakaze"], "dora_marker":
                ev["dora_marker"], "kyoku": ev["kyoku"], "honba": ev["honba"],
                "kyotaku": ev["kyotaku"], "oya": ev["oya"],
                "scores": ev["scores"], "tehais": ev["tehais"]}
    out = {"type": t}
    if "actor" in ev and ev.get("actor") is not None:
        out["actor"] = _actor(sm, ev["actor"])
    if t in ("chi", "pon", "daiminkan", "hora"):
        out["target"] = _actor(sm, ev.get("target"))
    for k in ("pai", "tsumogiri", "consumed", "dora_marker"):
        if ev.get(k) is not None:
            out[k] = ev[k]
    return out


def preview(ledger):
    """回傳 {hand_id: {"sent": [...], "blocked": 原因 or None}}。"""
    res = {}
    hands = sorted({e["hand_id"] for e in ledger.events})
    for h in hands:
        evs = [e for e in ledger.ordered(h) if e["type"] not in SKIP
               and e["status"] != "CANCELLED"]
        start = next((e for e in evs if e["type"] == "start_kyoku"), None)
        sent, blocked = [], None
        sm = start.get("seat_map") if start else None
        if start is None or not start.get("export_eligible"):
            blocked = ("BLOCKED_START: start_kyoku 缺 "
                       + ", ".join(start.get("missing", []) if start else
                                   ["整筆"]))
        else:
            for e in evs:
                if not e.get("export_eligible"):
                    blocked = (f"BLOCKED_EVENT at {e['event_id']} "
                               f"{e['type']} {e.get('actor') or ''} 缺 "
                               f"{', '.join(e.get('missing') or ['status=' + e['status']])}")
                    break
                sent.append(to_mjai(e, sm))
        res[h] = {"sent": sent, "blocked": blocked,
                  "total": len(evs)}
    return res
