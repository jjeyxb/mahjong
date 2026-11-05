"""Mortal 回覆 -> 人看得懂的一行;重播時對照「實際上怎麼打」。

=== Mortal 的 meta ===
    mask_bits  這一刻能做的動作(46 個位元,libriichi 的動作編號)
    q_values   照 mask_bits 由小到大、每個能做的動作一個分數;最大的就是建議
動作編號(2026-10-28 用實機 94 筆回覆驗證過:每一筆 q 最大的都等於 Mortal 的回覆):
    0~8 1m~9m  9~17 1p~9p  18~26 1s~9s  27~33 東南西北白發中  34~36 赤 5m/5p/5s
    37 立直  38/39/40 吃(被吃的牌在 低/中/高)  41 碰  42 槓  43 和  44 九種九牌流局  45 不鳴
"""
import json

TILES = ([f"{i}m" for i in range(1, 10)] + [f"{i}p" for i in range(1, 10)]
         + [f"{i}s" for i in range(1, 10)] + list("ESWNPFC")
         + ["5mr", "5pr", "5sr"])
ACTIONS = TILES + ["reach", "chi_low", "chi_mid", "chi_high", "pon", "kan",
                   "hora", "ryukyoku", "none"]
ZH = {"E": "東", "S": "南", "W": "西", "N": "北", "P": "白", "F": "發", "C": "中"}
ACT_ZH = {"reach": "立直", "chi_low": "吃", "chi_mid": "吃", "chi_high": "吃",
          "pon": "碰", "kan": "槓", "hora": "和", "ryukyoku": "九種九牌",
          "none": "不鳴"}


def tile_zh(p):
    if not p:
        return "?"
    if p in ZH:
        return ZH[p]
    return p.replace("r", "(赤)") if p.endswith("r") else p


def ranked(reply):
    """[(分數, 動作), ...] 由高到低。沒有 meta(例如立直後自動摸切)回 []。"""
    meta = reply.get("meta") or {}
    mb, q = meta.get("mask_bits"), meta.get("q_values")
    if mb is None or q is None:
        return []
    acts = [ACTIONS[b] for b in range(len(ACTIONS)) if mb >> b & 1]
    if len(acts) != len(q):
        return []
    return sorted(zip(q, acts), reverse=True)


def _act_zh(a):
    return ACT_ZH.get(a) or f"打 {tile_zh(a)}"


def describe(reply):
    """一行。例:「打 北(摸切)  次選 打 9s(差 0.06)  向聽 1」"""
    t = reply.get("type")
    if t == "dahai":
        main = (f"打 {tile_zh(reply.get('pai'))}"
                f"({'摸切' if reply.get('tsumogiri') else '手切'})")
    elif t in ("chi", "pon", "daiminkan"):
        zh = {"chi": "吃", "pon": "碰", "daiminkan": "大明槓"}[t]
        main = (f"{zh} {tile_zh(reply.get('pai'))}(用 "
                + " ".join(tile_zh(x) for x in reply.get("consumed") or [])
                + ")")
    elif t in ("ankan", "kakan"):
        main = ("暗槓 " if t == "ankan" else "加槓 ") + tile_zh(
            (reply.get("consumed") or [reply.get("pai")])[0])
    elif t == "reach":
        main = "立直"
    elif t == "hora":
        main = "和了!"
    elif t == "ryukyoku":
        main = "九種九牌(流局)"
    elif t == "none":
        main = "不鳴(跳過)"
    else:
        main = json.dumps(reply, ensure_ascii=False)
    r = ranked(reply)
    extra = ""
    if len(r) >= 2:
        d = r[0][0] - r[1][0]
        extra = f"  次選 {_act_zh(r[1][1])}(差 {d:.2f})"
    meta = reply.get("meta") or {}
    if meta.get("shanten") is not None:
        extra += f"  向聽 {meta['shanten']}"
    return main + extra


def advice_text(reply):
    """字幕列用:「建議:打 北　次選:打 9s」。不顯示分數。

    ★ 2026-11-04 使用者:摸切/手切不用寫。同一種牌打哪一張對自己的手牌
      完全一樣,差別只在對手看得到「是不是剛摸的」—— 進階的讀牌資訊。
      要打剛摸的那張時,手牌上的紅點會直接標在摸入牌上(advice_marks)。"""
    t = reply.get("type")
    if t == "dahai":
        main = f"打 {tile_zh(reply.get('pai'))}"
    elif t in ("chi", "pon", "daiminkan"):
        zh = {"chi": "吃", "pon": "碰", "daiminkan": "大明槓"}[t]
        main = (f"{zh} {tile_zh(reply.get('pai'))}(用 "
                + " ".join(tile_zh(x) for x in reply.get("consumed") or [])
                + ")")
    elif t in ("ankan", "kakan"):
        main = ("暗槓 " if t == "ankan" else "加槓 ") + tile_zh(
            (reply.get("consumed") or [reply.get("pai")])[0])
    elif t == "reach":
        main = "立直"
    elif t == "hora":
        main = "和了!"
    elif t == "ryukyoku":
        main = "九種九牌(流局)"
    elif t == "none":
        main = "不鳴(跳過)"
    else:
        return None
    r = ranked(reply)
    second = f"　次選:{_act_zh(r[1][1])}" if len(r) >= 2 else ""
    return f"建議:{main}{second}"


def advice_marks(reply):
    """手牌上要標的牌:[(牌, "main"/"second", 是不是摸入牌那張), ...]

    主選  打牌 -> 那一張(摸切就標摸入牌);吃/碰/槓 -> 要拿出來的那幾張
    次選  只有「打某一張」才標(次選是不鳴、立直之類就不標)"""
    t = reply.get("type")
    out = []
    if t == "dahai" and reply.get("pai"):
        out.append((reply["pai"], "main", bool(reply.get("tsumogiri"))))
    elif t in ("chi", "pon", "daiminkan", "ankan"):
        out += [(p, "main", False) for p in reply.get("consumed") or []]
    elif t == "kakan" and reply.get("pai"):
        out.append((reply["pai"], "main", False))
    r = ranked(reply)
    if len(r) >= 2 and r[1][1] in TILES:
        out.append((r[1][1], "second", False))
    return out


# ---------------------------------------------------------------------------
# 重播對照:Mortal 建議 vs 實際
# ---------------------------------------------------------------------------
def _same(reply, actual, pid):
    t = reply.get("type")
    if actual is None:
        return t == "none"
    at = actual["type"]
    if t == "none":
        return at not in ("chi", "pon", "daiminkan", "hora") \
            or actual.get("actor") != pid
    if t != at:
        return False
    if t == "dahai":
        return reply.get("pai", "").rstrip("r") == \
            actual.get("pai", "").rstrip("r")
    return True


def _actual(stream, i, pid, reply):
    """第 i 筆之後,P0 實際做了什麼。
    對別家打牌的回應(鳴/和/不鳴):看下一筆是不是 P0 的鳴牌/和。
    其他:P0 的下一個動作。"""
    after = stream[i]
    if reply.get("type") in ("none", "chi", "pon", "daiminkan") or (
            after.get("type") == "dahai" and after.get("actor") != pid):
        for e in stream[i + 1:i + 4]:
            if e.get("type") in ("chi", "pon", "daiminkan", "hora") and \
                    e.get("actor") == pid:
                return e
            if e.get("type") in ("tsumo", "dahai", "end_kyoku"):
                return None
        return None
    for e in stream[i + 1:]:
        if e.get("type") == "end_kyoku":
            return None
        if e.get("actor") == pid and e.get("type") in (
                "dahai", "reach", "ankan", "kakan", "hora", "chi", "pon"):
            return e
    return None


def review(stream, pairs, pid):
    """pairs: [(stream 的索引, reply)]。回傳 (報告行, 一致數, 總數)。"""
    lines, ok, n = [], 0, 0
    kyoku = ""
    for i, reply in pairs:
        for e in stream[:i + 1][::-1]:
            if e.get("type") == "start_kyoku":
                kyoku = (f"{ {'E': '東', 'S': '南'}.get(e['bakaze'], e['bakaze']) }"
                         f"{e['kyoku']}-{e['honba']}").replace(" ", "")
                break
        act = _actual(stream, i, pid, reply)
        same = _same(reply, act, pid)
        n += 1
        ok += same
        if act is None:
            a = "不鳴(跳過)" if reply.get("type") in (
                "none", "chi", "pon") else "-"
        elif act["type"] == "dahai":
            a = f"打 {tile_zh(act['pai'])}"
        else:
            a = {"reach": "立直", "chi": "吃", "pon": "碰", "ankan": "暗槓",
                 "kakan": "加槓", "hora": "和", "daiminkan": "大明槓"}.get(
                     act["type"], act["type"])
        trig = stream[i]
        tz = (f"{trig.get('type')}"
              + (f" {trig.get('pai')}" if trig.get("pai") else "")
              + (f" by {trig.get('actor')}" if trig.get("actor") is not None
                 else ""))
        lines.append(f"{'✓' if same else '✗'} {kyoku:<6} #{i:<4} 收到 {tz:<16}"
                     f" 建議 {describe(reply)}  | 實際 {a}")
    return lines, ok, n


def riichi_advice(reach_reply, discard_reply):
    """「建議:立直,打 2m　次選:不立直,打 2p」+ 手牌標記。

    discard_reply 是送了(預先的)reach 之後 Mortal 回的宣言牌;
    次選取自建議立直那一筆回覆的第二名(通常是不立直、默聽打某一張)。"""
    if (discard_reply or {}).get("type") != "dahai":
        return None, []
    pai = discard_reply.get("pai")
    text = f"建議:立直,打 {tile_zh(pai)}"
    marks = [(pai, "main", bool(discard_reply.get("tsumogiri")))]
    r = ranked(reach_reply or {})
    if len(r) >= 2:
        a = r[1][1]
        if a == pai:
            text += "　次選:不立直(一樣打這張)"
        elif a in TILES:
            text += f"　次選:不立直,打 {tile_zh(a)}"
            marks.append((a, "second", False))
        else:
            text += f"　次選:{_act_zh(a)}"
    return text, marks
