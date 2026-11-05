"""第 4 步:吃/碰/槓用掉了哪幾張(consumed)、槓型。

=== 做法:鳴牌前後「穩定牌組」的多重集合相減 ===
鳴牌區模型讀到的是一張一張的牌。鳴牌之後那一家的鳴牌區會多出一組,但動畫
期間牌組會滑動(實機 P2 碰 8s:x 821 -> 734),偶爾也會閃一兩幀讀錯
(f57 三張 9m 讀成 1m 0.18~0.44)。所以:
    穩定牌組  同一家連續 STABLE 次鳴牌模型讀數的多重集合完全一樣
    新牌組    鳴牌之後第一份「包含鳴牌前全部牌、而且多出預期張數」的穩定牌組
              減掉鳴牌前最後一份穩定牌組
不看位置,所以滑動不影響。

=== 每一種鳴牌 ===
    chi / pon      新增 3 張 -> 扣掉被鳴的那張(來源棄牌)= consumed[2]
    daiminkan      新增 4 張 -> 扣掉被鳴的那張 = consumed[3]
    自家槓         新增 1 張而且跟自己碰過的牌同種 -> kakan
                       pai = 新的那張,consumed = 原本碰組的 3 張(mjai 規格)
                   新增 2 張(暗槓只露兩張,使用者確認)或 4 張 -> ankan
                       consumed = 4 張;五的話照雀魂一種花色一張赤:5xr + 5x*3
赤寶牌:鳴牌模型讀得出 5mr/5pr/5sr,照實記錄;比對「是不是同一種」才正規化。

=== P0 的第二個來源 ===
P0 鳴牌之後手牌會少 2/3 張:鳴牌前後的穩定暗牌相減 = consumed。
兩個來源都有就交叉檢查(不一致記下來);鳴牌區讀不到就用手牌那一份。

=== 讀不出來 ===
等 RESOLVE_WAIT 幀還湊不出來就放棄:consumed 維持缺,那一筆鳴牌不會被當成
可以送出(Mortal 守門器會擋在那裡),不猜。
"""
from collections import Counter, deque
from dataclasses import dataclass

import _paths  # noqa: F401
from hand import diff
from tiles import base_pai

SEATS = ("P0", "P1", "P2", "P3")
MELD_CONF = 0.30       # 鳴牌模型門檻 0.15 是「框得到」,這裡取比較穩的
STABLE = 2             # 連續幾次鳴牌讀數一樣才算穩定
# ★ 鳴牌模型每 3 幀跑一次,STABLE=3 要等 9 幀(6 fps 下 1.5 秒)。P0 鳴牌之後
#   Mortal 要馬上決定打哪張,鳴牌事件不能拖這麼久,所以取 2。
RESOLVE_WAIT = 90      # 鳴牌後最多等幾幀
PARTIAL_WAIT = 15      # 碰/大明槓少讀一張時,等這麼久還是少一張才接受
# ★ 實機 runs\20261027_181804 H003:P1 碰中,鳴牌模型三張只框到兩張(中間
#   那張漏掉,畫面上清楚是三張)。碰/大明槓是同一種牌,少一張不影響答案
#   (五除外:赤五在哪一張決定 consumed,不猜)。


@dataclass
class MeldResult:
    event_id: str
    ok: bool
    kind: str              # chi/pon/daiminkan/ankan/kakan
    pai: object
    consumed: object
    method: str            # meld_diff / p0_hand_diff / both
    note: str = ""


def _sub(a, b):
    """多重集合 a - b(b 必須是 a 的子集,否則回 None)。"""
    c = Counter(a)
    c.subtract(Counter(b))
    if any(v < 0 for v in c.values()):
        return None
    return [p for p, k in c.items() for _ in range(k)]


def _remove_one(tiles, pai):
    """拿掉被鳴的那張:先找完全一樣的(含赤),沒有再找同種的。"""
    t = list(tiles)
    if pai in t:
        t.remove(pai)
        return t
    for i, x in enumerate(t):
        if base_pai(x) == base_pai(pai):
            del t[i]
            return t
    return None


def _loose_sub(a, b):
    """多重集合 a - b,b 裡 a 沒有的就忽略(鳴牌區之前少讀的牌後來補讀到)。"""
    c = Counter(a)
    c.subtract(Counter(b))
    return [p for p, k in c.items() for _ in range(max(0, k))]


def _num(p):
    b = base_pai(p)
    return (int(b[0]), b[1]) if len(b) == 2 and b[1] in "mps" else None


def find_consumed(new, kind, pai):
    """從「新增的牌」裡找出 consumed:吃 = 跟 pai 組成順子的兩張,碰 = 兩張同種,
    大明槓 = 三張同種。被鳴的那張(橫放)常常讀不到,不需要它在 new 裡。
    多出來的牌(例如舊牌組裡之前漏讀、這時才讀到的那張)不影響。找不到回 None。"""
    pb = base_pai(pai)
    if kind in ("pon", "daiminkan"):
        k = 2 if kind == "pon" else 3
        same = [x for x in new if base_pai(x) == pb]
        if len(same) >= k + 1:
            # 被鳴的那張也在:拿掉一張(優先拿掉跟 pai 完全一樣的)
            same = _remove_one(same, pai)
        if len(same) == k:
            return sorted(same)
        return None
    if kind == "chi":
        pn = _num(pai)
        if pn is None:
            return None
        opts = []
        for i, a in enumerate(new):
            for b in new[i + 1:]:
                na, nb = _num(a), _num(b)
                if not na or not nb or na[1] != pn[1] or nb[1] != pn[1]:
                    continue
                ns = sorted([na[0], nb[0], pn[0]])
                if ns[1] == ns[0] + 1 and ns[2] == ns[1] + 1:
                    opts.append(tuple(sorted([a, b])))
        opts = set(opts)
        return list(next(iter(opts))) if len(opts) == 1 else None
    return None


def _run3(tiles):
    ns = sorted(_num(t) for t in tiles if _num(t))
    return (len(ns) == 3 and len({x[1] for x in ns}) == 1
            and ns[1][0] == ns[0][0] + 1 and ns[2][0] == ns[1][0] + 1)


def ankan_tiles(pai):
    b = base_pai(pai)
    if b in ("5m", "5p", "5s"):
        return [b + "r", b, b, b]
    return [b] * 4


class MeldReader:
    def __init__(self):
        self.reset()

    slot_votes = None     # event_id -> {牌: 票數}(來源棄牌那一格的牌河票;pipeline 設)

    def reset(self):
        self.hist = {s: deque(maxlen=STABLE) for s in SEATS}
        self.stable = {s: [] for s in SEATS}     # [(f0, f1, tiles tuple)]
        self.raw = {s: deque(maxlen=12) for s in SEATS}   # [(frame, tiles)]
        self.pending = []
        self.frozen = False
        self.stats = Counter()

    def freeze(self):
        """終局之後不再讀:局末攤牌會被讀成鳴牌(meld_model 的已知缺陷)。"""
        self.frozen = True

    # ------------------------------------------------------------------
    def feed(self, frame, meld_obs):
        if meld_obs is None or self.frozen:
            return
        for s in SEATS:
            part = meld_obs.get(s)
            if not part or part.get("tiles") is None:
                continue
            tiles = tuple(sorted(d["pai"] for d in part["tiles"]
                                 if d["conf"] >= MELD_CONF))
            self.raw[s].append((frame, tiles))
            h = self.hist[s]
            h.append(tiles)
            if len(h) == STABLE and len(set(h)) == 1:
                st = self.stable[s]
                if st and st[-1][2] == tiles:
                    st[-1] = (st[-1][0], frame, tiles)
                else:
                    st.append((frame, frame, tiles))
                    del st[:-30]

    def _before(self, seat, frame):
        for f0, f1, t in reversed(self.stable[seat]):
            if f0 < frame:
                return t
        return ()

    def _after(self, seat, frame, before, sizes):
        for f0, f1, t in self.stable[seat]:
            if f1 <= frame:
                continue
            new = _sub(t, before)
            if new is not None and len(new) in sizes:
                return new
        return None

    def new_group(self, seat, since_frame, loose=False):
        """since_frame 之後這一家鳴牌區多出來的牌(最新一份穩定牌組 − 之前)。
        沒有、或不是 3/4 張 -> None。給「漏看鳴牌宣告」的推定用。

        loose=True(★ 2026-10-30 實戰東2 f3540:P1 碰 P2 的 4m,宣告橫幅只有
        0.19、橫放的那張 4m 只有 0.2 -> 穩定牌組只多兩張 4m,而且打牌的時候
        穩定牌組還沒成形):也接受 2 張,也看還沒穩定的最新讀數。
        要不要採用由呼叫端配合來源棄牌判斷(2 張 + 被鳴那張 = 碰/吃)。"""
        sizes = (2, 3, 4) if loose else (3, 4)
        before = self._before(seat, since_frame)
        for f0, f1, t in reversed(self.stable[seat]):
            if f1 <= since_frame:
                break
            new = _sub(t, before)
            if new is not None and len(new) in sizes:
                return new
        if loose:
            for f, t in reversed(self.raw[seat]):
                if f <= since_frame:
                    break
                new = _sub(t, before)
                if new is not None and len(new) in sizes:
                    return new
        return None

    # ------------------------------------------------------------------
    def request(self, event_id, kind, seat, frame, pons=None, p0_hand=None):
        """kind: chi / pon / daiminkan / kan_self。
        pons: 這一家碰過的 [(pai, consumed)],加槓要用。
        p0_hand: (鳴牌前穩定暗牌, signals) —— P0 才有。"""
        self.pending.append({"event_id": event_id, "kind": kind,
                             "seat": seat, "frame": frame,
                             "before": self._before(seat, frame),
                             "pons": pons or [], "p0": p0_hand})

    def resolve(self, frame, get_pai, p0_after=None):
        """每幀呼叫。get_pai(event_id) -> 被鳴的那張(來源棄牌的牌面,可能還沒有)。
        p0_after(frame) -> P0 在 frame 之後第一份穩定暗牌。"""
        out = []
        for p in list(self.pending):
            r = self._try(p, frame, get_pai, p0_after)
            if r is not None:
                out.append(r)
                self.pending.remove(p)
                self.stats[f"{r.kind}:{'ok' if r.ok else 'fail'}"] += 1
        return out

    def _try(self, p, frame, get_pai, p0_after):
        kind, seat = p["kind"], p["seat"]
        late = frame - p["frame"] > RESOLVE_WAIT
        if kind == "kan_self":
            return self._try_self_kan(p, late)
        need = 3 if kind in ("chi", "pon") else 4
        pai = get_pai(p["event_id"])
        new = self._after(seat, p["frame"], p["before"], (need,))
        from_meld = None
        if new is not None and pai is None and kind != "chi" and \
                len({base_pai(x) for x in new}) == 1:
            # 被鳴的那張在牌河上一閃就被拿走、沒定案:碰/大明槓是同一種,
            # 牌面可以從鳴牌區知道(赤五:優先當成不是被鳴的那張)
            plain = [x for x in new if not x.endswith("r")]
            pai = plain[0] if plain else new[0]
            p["pai_from_meld"] = True
        if new is not None and pai is not None:
            from_meld = _remove_one(new, pai)
        from_hand = None
        if seat == "P0" and p0_after is not None and p.get("p0"):
            cur = p0_after(p["frame"])
            if cur:
                gone = diff(list(p["p0"]), list(cur))
                if len(gone) == need - 1:
                    from_hand = gone
        if from_meld is None and pai is None and kind in ("pon", "daiminkan"):
            # 被鳴的牌面還不知道:碰/大明槓的新增牌全是同一種,牌面就是那一種
            k = 2 if kind == "pon" else 3
            for f0, f1, t in self.stable[seat]:
                if f1 <= p["frame"]:
                    continue
                nw = _loose_sub(t, p["before"])
                kinds = {base_pai(x) for x in nw}
                if len(kinds) == 1 and len(nw) >= k:
                    plain = [x for x in nw if not x.endswith("r")]
                    pai = plain[0] if plain else nw[0]
                    p["pai_from_meld"] = True
                    break
        if from_meld is None and pai is not None:
            # ★ 2026-10-27 實機 H003:P1 吃 7s,橫放的 7s 只有 0.18 讀不到,同時
            #   舊的碰組補讀到一張中 -> 「新增 3 張」= 中 5s 6s,嚴格比對失敗。
            #   改成在新增的牌裡找「能跟 pai 組成這種鳴牌」的那幾張。
            for f0, f1, t in self.stable[seat]:
                if f1 <= p["frame"]:
                    continue
                cons = find_consumed(_loose_sub(t, p["before"]), kind, pai)
                if cons:
                    from_meld = cons
                    p["method"] = "meld_loose"
                    break
        if from_meld is None and kind == "chi" and pai is not None:
            # ★ 2026-10-30 實戰東2-1 f5460:P1 吃 4m,鳴牌區讀到 4m 6m,中間那張
            #   5m 只有 0.24(低於門檻)-> 永遠湊不到兩張。吃的另外兩張裡只看到
            #   一張、而且跟被鳴的牌差 2(4m 與 6m):順子只有一種,中間那張一定是
            #   5m(赤不赤讀不到,當普通的)。差 1 的(4m 與 5m)有兩種可能,不猜。
            pn = _num(pai)
            for f0, f1, t in self.stable[seat]:
                if f1 <= p["frame"] or pn is None:
                    continue
                nw = [x for x in _loose_sub(t, p["before"])
                      if _num(x) and _num(x)[1] == pn[1]
                      and base_pai(x) != base_pai(pai)]
                gap = [x for x in nw if abs(_num(x)[0] - pn[0]) == 2]
                if len(nw) == 1 and len(gap) == 1:
                    mid = f"{(_num(gap[0])[0] + pn[0]) // 2}{pn[1]}"
                    from_meld = sorted([gap[0], mid])
                    p["method"] = "meld_gap_fill"
                    p["gap_fill"] = mid
                    break
        if from_meld is None and kind == "chi" and pai is not None and \
                new is not None and len(new) == 3 and self.slot_votes:
            # ★ 2026-10-29 實機:P3 吃 P2 的立直宣言牌,鳴牌區清楚是 7m8m9m,
            #   但宣言牌在立直特效中被讀成 5mr(特效結束後才讀到 9m)、又因為被吃
            #   而提早定案 -> 被鳴的牌不在順子裡,consumed 永遠讀不出來。
            #   鳴牌區是正面、沒有特效:順子成立的話,被鳴的牌一定是這三張之一,
            #   取牌河那一格的票裡出現過、而且最多票的那一張。
            if _run3(new):
                votes = self.slot_votes(p["event_id"]) or {}
                opts = [(votes.get(t, 0), t) for t in new]
                best = max(opts)
                if best[0] > 0:
                    rest = list(new)
                    rest.remove(best[1])
                    p["method"] = "meld_run_fix"
                    p["pai_fix"] = best[1]
                    from_meld = rest
                    pai = best[1]
        if from_meld is not None:
            note, method = "", p.get("method", "meld_diff")
            if p.get("pai_fix"):
                note = (f"★牌河讀的被鳴牌不在順子裡,依鳴牌區改成 {p['pai_fix']}")
            if p.get("gap_fill"):
                note = (f"★鳴牌區只讀到一張,中間那張 {p['gap_fill']} 是由順子推的"
                        f"(赤五讀不到)")
            if from_hand is not None:
                if Counter(map(base_pai, from_meld)) == \
                        Counter(map(base_pai, from_hand)):
                    method = "both"
                else:
                    note = (f"★鳴牌區說 {sorted(from_meld)}、手牌少了 "
                            f"{from_hand} —— 兩個模型有一個讀錯")
            return MeldResult(p["event_id"], True, kind, pai,
                              sorted(from_meld), method, note)
        if from_meld is None and kind in ("pon", "daiminkan") and pai and \
                base_pai(pai)[0] != "5" and \
                frame - p["frame"] >= PARTIAL_WAIT and from_hand is None:
            part = self._after(seat, p["frame"], p["before"], (need - 1,))
            if part and all(base_pai(x) == base_pai(pai) for x in part):
                return MeldResult(p["event_id"], True, kind, pai,
                                  [base_pai(pai)] * (need - 1),
                                  "meld_partial",
                                  f"鳴牌區只讀到 {len(part)} 張 {base_pai(pai)}"
                                  f"(應該 {need} 張);同一種牌,consumed 照推")
        if from_hand is not None:
            # P0:手牌差分先到就先用(Mortal 要馬上決定鳴牌後打哪張);
            # 鳴牌區晚到的那一份不再比對
            return MeldResult(p["event_id"], True, kind, pai,
                              sorted(from_hand), "p0_hand_diff",
                              "鳴牌區還沒穩定,用 P0 手牌差分")
        if late:
            why = ("鳴牌區沒有出現多 {} 張的穩定牌組".format(need)
                   if new is None else "被鳴的牌面還不知道")
            return MeldResult(p["event_id"], False, kind, pai, None,
                              "none", why)
        return None

    def _try_self_kan(self, p, late):
        seat = p["seat"]
        new = self._after(seat, p["frame"], p["before"], (1, 2, 4))
        if new is not None:
            if len(new) == 1:
                t = new[0]
                for pon_pai, pon_cons in p["pons"]:
                    if pon_pai and base_pai(pon_pai) == base_pai(t):
                        cons = sorted([pon_pai] + list(pon_cons or []))
                        if len(cons) != 3:
                            return MeldResult(p["event_id"], False, "kakan",
                                              t, None, "meld_diff",
                                              "原本的碰組不完整")
                        return MeldResult(p["event_id"], True, "kakan", t,
                                          cons, "meld_diff")
                return MeldResult(p["event_id"], False, "kakan", t, None,
                                  "meld_diff", "多一張但找不到同種的碰")
            if len({base_pai(x) for x in new}) == 1:
                return MeldResult(p["event_id"], True, "ankan", None,
                                  ankan_tiles(new[0]), "meld_diff")
            return MeldResult(p["event_id"], False, "ankan", None, None,
                              "meld_diff", f"新增的牌不是同一種:{new}")
        if late:
            return MeldResult(p["event_id"], False, "kan_self", None, None,
                              "none", "鳴牌區沒有出現新的槓")
        return None
