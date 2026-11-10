"""第 0 步:start_kyoku 需要的場況(局數、本場、供託、四家點數、自風、寶牌)。

辨識本身沿用舊專案(mjai_live)已經驗證過的樣板比對,程式搬在 ui/:
    ui/panel.py   局數 / 本場 / 供託
    ui/winds.py   四個角的自風 -> 莊家、mjai actor
    ui/scores.py  四家點數(含「總和 = 100000」檢查)
寶牌指示牌用舊專案的 tile_model(整張畫面推論)篩 DORA_ZONE,邏輯同舊的
readers.DoraReader。

=== 分工(跟觀測檔的原則一致)===
錄製端  raw_ui(frame)  每 N 幀讀一次,只存「這一幀讀到什麼、分數多少」
下游    KyokuInfo      吃 raw 讀值做投票/檢查/確認。規則改了不用重錄。

=== 確認規則(全部沿用舊專案量過的值)===
局數/本場  累計 MIN_VOTES 票、最高票是第二名的 VOTE_MARGIN 倍;而且
           (局數,本場) 不可以跟上一局一樣(換局瞬間面板還是舊的)
供託       連續 LIVE_CONFIRM 幀一樣才改(局中有人立直會變)
自風       以 P0 為準推其他三家;其他三家讀到的要一致、紅底那家要是東;
           連續 MIN_VOTES 幀
點數       四家都讀到 + 總和檢查通過(需要供託),同一組 MIN_VOTES 票;
           只有一家讀不到時用總和補出那一家(2026-11-10),票數照舊
寶牌       同一格累計 DORA_VOTES 票、而且是第二名的 VOTE_MARGIN 倍;
           定了就不改(槓寶牌事件已經建了,改值會變成重複的 dora 事件)
           ★ 實機 runs\20261026_232930:寶牌每 15 幀讀一次、5 票 -> start_kyoku
             要等 14.7 秒才齊,Mortal 前 14 秒什麼都收不到。改 3 票,錄製端
             --dora-every 預設改 5(3 票約 15 幀 = 2.4 秒)

=== mjai 座次(v7.1 2.1)===
actor 0 = 東一局的東家。某一家的 actor = (莊家 actor + 自風 index) mod 4,
莊家 actor = 局數 - 1。scores 要照 actor 排。
"""
from collections import Counter

import _paths  # noqa: F401

WINDS = ("E", "S", "W", "N")
SEAT_ORDER = ("P0", "P1", "P2", "P3")
PANEL_MATCH_MIN = 0.70
# ★ 2026-11-06 本場 8 以上沒有樣板(錄影最多錄到 7 本場),改用推的:
#   本場只會是 0 或「上一局 + 1」,而 0 讀得到。所以上一局已知、本場又讀不出來,
#   就是上一局 + 1。只在「上一局 + 1」沒有樣板時才推(有樣板就等它讀)。
#   難的是舊面板:上一局是 8 的話,換局那幾幀的舊面板也是「讀不出來」,
#   跟新的 9 分不開。所以讀不出來的那幀帶一個縮小的字形指紋(HONBA_SIG),
#   跟上一局讀不出來時的指紋一樣就是舊面板。實測 24x12 同一個字 1.0、
#   不同字最高 0.57。
HONBA_TEMPLATED_MAX = 7     # templates_v7/honbax__6、7 是 selfplay-20261102 切的
HONBA_SIG_BELOW = 0.85      # 分數低於這個就帶指紋(對的讀值實測最低 0.88)
HONBA_SIG = (24, 12)
HONBA_SAME_SIG = 0.85
HONBA_GUESS_VOTES = 4       # 同一個新字形要看到幾次才推
WIND_MATCH_MIN = 0.70
MIN_VOTES = 2
VOTE_MARGIN = 2
LIVE_CONFIRM = 3
DORA_VOTES = 3

# 舊 readers.py 量的寶牌指示牌位置(1920x1080)
DORA_ZONE = (25, 44, 285, 84)
DORA_SLOT_X = (55, 111, 167, 223, 279)
DORA_SLOT_Y = 86
DORA_MATCH_DIST = 32


# ---------------------------------------------------------------------------
# 錄製端:只讀,不判斷
# ---------------------------------------------------------------------------
class RawUiReader:
    """需要 cv2 + 樣板。tile_model 可選(沒有就不讀寶牌)。"""

    def __init__(self, old_dir, tile_model=None):
        import os
        import panel
        import scores
        import winds
        self.panel, self.scores, self.winds = panel, scores, winds
        self.tpl = panel.Templates(os.path.join(old_dir, "templates"))
        self._extra_templates(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "ui", "templates_v7"))
        self.wtpl = winds.WindTemplates(os.path.join(old_dir,
                                                     "templates_wind"))
        self.wreader = winds.WindReader(self.wtpl)
        self.tile_model = tile_model
        n_panel = sum(self.tpl.count(f) for f in ("kyoku", "honba",
                                                  "kyotaku"))
        print(f"✅ 面板樣板 {n_panel} 張、自風樣板 {self.wtpl.count()} 張"
              + ((",寶牌用 " + type(tile_model).__name__) if tile_model
                 else ",不讀寶牌"))

    def _extra_templates(self, folder):
        """補點數樣板(ui\\templates_v7)。舊樣板缺的數字才補,不覆蓋。
        ★ 2026-10-27 從 runs\\20261027_181804 的錄影切出來(六局的點數都由
          總和檢查與畫面確認過)。舊的缺很多:P0 缺 5、P1 缺 7/8/9、P2 缺
          3/7/8/9、P3 缺 1/4/6 —— 缺的數字會去比對到別的數字(P2 的 23000
          讀成 25000)。補完仍缺:P1 8、P2 8/9、P3 4/6(這場沒出現),那些
          由同一組(左右/上下)另一家的樣板代替。"""
        import os
        import cv2
        if not os.path.isdir(folder):
            return
        n = 0
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".png") or "__" not in name:
                continue
            field, label = name[:-4].split("__", 1)
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            have = self.tpl.items.setdefault(field, {})
            if label not in have:
                have[label] = (img > 127).astype("uint8")
                n += 1
        self.tpl._build_pools()
        print(f"✅ 補點數樣板 {n} 張(ui\\templates_v7)")

    def read(self, frame):
        p = {}
        for f in ("kyoku", "honba", "kyotaku"):
            n = self.panel.normalize(self.panel.mask_of(frame, f))
            if n is None:
                p[f] = None
            else:
                lab, sc = self.tpl.match(f, n)
                p[f] = [lab, round(float(sc), 3)]
                if f == "honba" and sc < HONBA_SIG_BELOW:
                    p[f].append(honba_sig(n))
        w, red, wsc = self.wreader.read(frame)
        winds = {s: ([w[s], round(float(wsc[s]), 3)] if w[s] else None)
                 for s in SEAT_ORDER}
        sc = {}
        for s in SEAT_ORDER:
            v, score, why = self.scores.read_seat(frame, s, self.tpl)
            # 讀不出來也留下原因(之後查「為什麼這一家整局 None」要用)
            sc[s] = ([v, round(float(score), 3)] if v is not None
                     else [None, round(float(score), 3), why])
        ui = {"panel": p, "winds": winds, "wind_red": red, "scores": sc}
        dora = None
        if self.tile_model is not None:
            from tiles import in_region
            dets = self.tile_model.detect(frame)
            dora = {"zone": list(DORA_ZONE),
                    "tiles": [{"pai": d["pai"], "conf": round(d["conf"], 4),
                               "box": [int(v) for v in d["box"]],
                               "center": [int(v) for v in d["center"]]}
                              for d in in_region(dets, DORA_ZONE)]}
        return ui, dora


def honba_sig(norm):
    """正規化後的本場遮罩縮成 HONBA_SIG,壓成十六進位字串(存進觀測檔)。"""
    import cv2
    import numpy as np
    small = cv2.resize(norm.astype(np.float32), HONBA_SIG,
                       interpolation=cv2.INTER_AREA) > 0.5
    return np.packbits(small.ravel()).tobytes().hex()


def sig_similarity(a, b):
    """兩個 honba_sig 的交集/聯集。"""
    x, y = int(a, 16), int(b, 16)
    union = bin(x | y).count("1")
    return bin(x & y).count("1") / union if union else 0.0


# ---------------------------------------------------------------------------
# 下游:投票、檢查、組 start_kyoku
# ---------------------------------------------------------------------------
def derive_from_p0(w0):
    i = WINDS.index(w0)
    return {s: WINDS[(i + k) % 4] for k, s in enumerate(SEAT_ORDER)}


def actors_of(winds, kyoku):
    oya = (int(kyoku) - 1) % 4
    return {s: (oya + WINDS.index(w)) % 4 for s, w in winds.items()}


class KyokuInfo:
    def __init__(self):
        self.previous = None        # 上一局的 (kyoku_label, honba)
        self.previous_sig = None    # 上一局本場是推出來的:它的字形指紋(認舊面板用)
        self.p0_actor = None        # 整場固定;第一局 start_kyoku 確定後由 pipeline 設
        self.reset()

    def reset(self, remember=True):
        """remember=False:上一局不是真的一局(錄影開始時的大廳/載入畫面),
        它讀到的局數不能拿來擋下一局 ——
        ★ 2026-10-30 實戰:大廳那段讀到「東1」,真的東1局就被當成「跟上一局
          一樣 = 舊畫面」一直擋,到局末讀到下一局的「東2」才填進去 -> 局數、
          P0 的 actor 全錯。"""
        if remember and getattr(self, "value", None) \
                and self.value.get("kyoku_label") \
                and self.value.get("honba") is not None:
            self.previous = (self.value["kyoku_label"], self.value["honba"])
            self.previous_sig = (max(self.honba_sigs, key=lambda c: c[1])[0]
                                 if self.honba_guessed and self.honba_sigs
                                 else None)
        self.honba_sigs = []        # [[指紋, 次數]],讀不出來的本場依字形分組
        self.honba_guessed = False
        self.value = {"kyoku_label": None, "honba": None, "kyotaku": None,
                      "winds": None, "scores": None, "dora": []}
        self.votes = {"kyoku_label": Counter(), "honba": Counter()}
        self._streak = (None, 0)
        self._wind = (None, 0)
        self.score_votes = Counter()
        self.derived_scores = 0     # 用總和補出第四家點數的次數,見 _scores
        self.dora_slots = [Counter() for _ in DORA_SLOT_X]
        self.stale = 0
        self.rejected = Counter()

    # ------------------------------------------------------------------
    def feed(self, ui, dora):
        """吃一幀的 raw 讀值。回傳新確認的寶牌 [(第幾格, 牌)]。"""
        new_dora = []
        if ui:
            self._panel(ui.get("panel") or {})
            self._winds(ui.get("winds") or {}, ui.get("wind_red"))
            self._scores(ui.get("scores") or {})
        if dora and dora.get("tiles") is not None:
            before = list(self.value["dora"])
            for d in dora["tiles"]:
                cx, cy = d["center"]
                if abs(cy - DORA_SLOT_Y) > 60:
                    continue
                k = min(range(len(DORA_SLOT_X)),
                        key=lambda i: abs(DORA_SLOT_X[i] - cx))
                if abs(DORA_SLOT_X[k] - cx) <= DORA_MATCH_DIST:
                    self.dora_slots[k][d["pai"]] += 1
            marks = []
            for k, c in enumerate(self.dora_slots):
                if k < len(before) and before[k] is not None:
                    marks.append(before[k])
                    continue
                if not c:
                    marks.append(None)
                    continue
                top = c.most_common(2)
                best, n = top[0]
                second = top[1][1] if len(top) > 1 else 0
                marks.append(best if n >= DORA_VOTES
                             and n >= VOTE_MARGIN * second else None)
            while marks and marks[-1] is None:
                marks.pop()
            self.value["dora"] = marks
            new_dora = [(i, p) for i, p in enumerate(marks)
                        if p is not None and (i >= len(before)
                                              or before[i] != p)]
        return new_dora

    def _panel(self, p):
        for f, key in (("kyoku", "kyoku_label"), ("honba", "honba")):
            r = p.get(f)
            if f == "honba" and r and self._honba_unread(r, p.get("kyoku")):
                continue
            if not r or r[0] is None or r[1] < PANEL_MATCH_MIN:
                continue
            lab = r[0] if f == "kyoku" else int(r[0])
            self.votes[key][lab] += 1
            top = self.votes[key].most_common(2)
            best, n = top[0]
            second = top[1][1] if len(top) > 1 else 0
            if n >= MIN_VOTES and n >= VOTE_MARGIN * max(second, 1):
                self.value[key] = best
        now = (self.value["kyoku_label"], self.value["honba"])
        if None not in now and now == self.previous:
            # 還在看上一局的面板,清掉重讀
            self.stale += 1
            self.value["kyoku_label"] = self.value["honba"] = None
            self.votes = {"kyoku_label": Counter(), "honba": Counter()}
        r = p.get("kyotaku")
        if r and r[0] is not None and r[1] >= PANEL_MATCH_MIN:
            lab = int(r[0])
            was, n = self._streak
            n = n + 1 if was == lab else 1
            self._streak = (lab, n)
            if n >= LIVE_CONFIRM:
                self.value["kyotaku"] = lab

    def _honba_unread(self, r, rk):
        """這一幀的本場算不算「讀不出來」;是的話順便看能不能推(見 HONBA_SIG)。

        回傳 True 表示這一幀不投票。弱的讀值(低於 HONBA_SIG_BELOW)如果不是
        0 也不是上一局 + 1,一樣當讀不出來:那是沒有樣板的字被硬配到別的字。
        """
        expect = self.previous[1] + 1 if self.previous else None
        weak = r[0] is None or r[1] < PANEL_MATCH_MIN
        if not weak:
            if expect is None or r[1] >= HONBA_SIG_BELOW \
                    or int(r[0]) in (0, expect):
                return False
            self.rejected["honba_impossible"] += 1
        sig = r[2] if len(r) > 2 else None
        if sig is None:
            return True
        if self.previous_sig and \
                sig_similarity(sig, self.previous_sig) >= HONBA_SAME_SIG:
            # 跟下面「(局數,本場) 跟上一局一樣」同一件事:舊面板,清掉重讀
            # (不清的話舊面板的局數票會一直壓著新的)
            self.rejected["honba_old_panel"] += 1
            self.stale += 1
            self.value["kyoku_label"] = self.value["honba"] = None
            self.votes = {"kyoku_label": Counter(), "honba": Counter()}
            return True
        for c in self.honba_sigs:
            if sig_similarity(sig, c[0]) >= HONBA_SAME_SIG:
                c[1] += 1
                break
        else:
            c = [sig, 1]
            self.honba_sigs.append(c)
        label = self.value["kyoku_label"]
        if (self.value["honba"] is None and expect is not None
                and expect > HONBA_TEMPLATED_MAX and label
                and rk and rk[0] == label and rk[1] >= PANEL_MATCH_MIN
                and not self.votes["honba"][0]
                and c[1] >= HONBA_GUESS_VOTES):
            self.value["honba"] = expect
            self.honba_guessed = True
            self.rejected["honba_guessed"] += 1
        return True

    def _winds(self, w, red):
        if self.value["winds"] is not None:
            return
        p0 = w.get("P0")
        if not p0 or p0[1] < WIND_MATCH_MIN:
            self._wind = (None, 0)
            return
        full = derive_from_p0(p0[0])
        for s in SEAT_ORDER:
            r = w.get(s)
            if r and r[1] >= WIND_MATCH_MIN and r[0] != full[s]:
                self.rejected["wind_mismatch"] += 1
                self._wind = (None, 0)
                return
        if red is not None and full.get(red) != "E":
            self.rejected["wind_red"] += 1
            self._wind = (None, 0)
            return
        key = tuple(full[s] for s in SEAT_ORDER)
        was, n = self._wind
        n = n + 1 if was == key else 1
        self._wind = (key, n)
        if n >= MIN_VOTES:
            self.value["winds"] = full

    def _scores(self, sc):
        if self.value["scores"] is not None:
            return
        vals = {s: (sc.get(s) or [None])[0] for s in SEAT_ORDER}
        unread = [s for s, v in vals.items() if v is None]
        if len(unread) > 1:
            return
        if self.value["kyotaku"] is None:
            self.rejected["scores_no_kyotaku"] += 1
            return
        total = 100000 - self.value["kyotaku"] * 1000
        if unread:
            # ★ 2026-11-10 實打一家點數一直讀不到,start_kyoku 整局差這一欄,Mortal 整局收不到。
            #   三家加供託跟總和的差就是第四家;算出來的要是非負的整百才信,照樣要投到 MIN_VOTES 票。
            rest = total - sum(v for v in vals.values() if v is not None)
            if rest < 0 or rest % 100:
                self.rejected["scores_derived_bad"] += 1
                return
            vals[unread[0]] = rest
            self.derived_scores += 1
        elif sum(vals.values()) != total:
            self.rejected["scores_total"] += 1
            return
        key = tuple(vals[s] for s in SEAT_ORDER)
        self.score_votes[key] += 1
        if self.score_votes[key] >= MIN_VOTES:
            self.value["scores"] = vals

    # ------------------------------------------------------------------
    def seat_map(self):
        """{seat: mjai actor};算不出來回 None。

        ★ 2026-10-30 實戰:mjai 的 actor 是「東一局時的座位」,整場固定 ——
          P0 的 actor 一旦確定(第一局),之後每一局都是 P0=a、P1=a+1、
          P2=a+2、P3=a+3,不必再讀自風。原本每局重讀自風,換局瞬間讀到的
          還是上一局的自風(西),配上新的局數(東2)-> P0 的 actor 算成 3,
          守門器以為座位變了,那一局整局不送。現在自風只拿來核對。"""
        v = self.value
        if self.p0_actor is not None:
            sm = {s: (self.p0_actor + i) % 4 for i, s in enumerate(SEAT_ORDER)}
            if v["winds"] and v["kyoku_label"]:
                rd = actors_of(v["winds"], int(v["kyoku_label"][1]))
                if rd != sm:
                    self.rejected["winds_vs_fixed_actor"] += 1
            return sm if v["kyoku_label"] else None
        if not v["winds"] or not v["kyoku_label"]:
            return None
        return actors_of(v["winds"], int(v["kyoku_label"][1]))

    def start_kyoku_fields(self, tehais_p0=None, tehais_note=None):
        """回傳 (fields, missing)。fields 的座位一律換成 mjai actor 排序。"""
        v = self.value
        miss = []
        f = {}
        if v["kyoku_label"]:
            f["bakaze"] = v["kyoku_label"][0]
            f["kyoku"] = int(v["kyoku_label"][1])
        else:
            miss += ["bakaze", "kyoku"]
        # ★ 2026-11-06 selfplay-20261106-110134 西1局:本場比局數先湊滿票,舊面板的「1」
        #   還沒被「跟上一局一樣」那條識破就寫進 start_kyoku(欄位第一次有值就固定)。
        #   局數還沒定的時候本場也不交出去。
        f["honba"] = v["honba"] if v["kyoku_label"] else None
        f["kyotaku"] = v["kyotaku"]
        if f["honba"] is None:
            miss.append("honba")
        if v["kyotaku"] is None:
            miss.append("kyotaku")
        sm = self.seat_map()
        f["seat_map"] = sm
        if sm is None:
            miss += ["oya", "seat_map"]
        else:
            f["oya"] = (f["kyoku"] - 1) % 4
        if v["scores"] and sm:
            arr = [None] * 4
            for s, a in sm.items():
                arr[a] = v["scores"][s]
            f["scores"] = arr
        else:
            miss.append("scores")
        if v["dora"] and v["dora"][0]:
            f["dora_marker"] = v["dora"][0]
        else:
            miss.append("dora_marker")
        if tehais_p0 is not None and len(tehais_p0) == 13 and sm:
            th = [["?"] * 13 for _ in range(4)]
            th[sm["P0"]] = list(tehais_p0)
            f["tehais"] = th
        else:
            miss.append("tehais")
        if tehais_note is not None:
            f["tehais_note"] = tehais_note
        return f, miss
