"""P0 摸入牌的第二意見:用「這一場手牌模型很有把握的牌」當樣板比對。

=== 為什麼需要 ===
2026-10-27 實機(runs\\20261027_181804):P0 摸牌 64 次有 12 次牌名讀不到,
要等打出去之後才從手牌差分補 -> 那幾巡 Mortal 收到摸牌時你已經打了。
看畫面,摸入牌清清楚楚,是手牌模型在「摸入牌位置」特別弱:
    5m 在摸入牌位置 4 次完全沒有框(同一張 5m 在手牌裡 1.00)
    9m 讀成 E(0.53/0.56)兩次,3s 只有 0.55~0.65,2m 0.38
同一個畫面裡,手牌裡的牌模型都很有把握 —— 而且雀魂同一種牌每次畫得一模一樣。

=== 做法 ===
樣板庫:每一幀手牌模型信心 >= LIB_CONF 的暗牌,切下來(灰階、縮成 40x56),
        每種牌留 LIB_PER 張。也可以從檔案預載(ui\\p0_tile_lib.npz)。
比對:  摸入牌位置(局勢模型的 drawn_tile 中心;大小、高度用同一幀暗牌的中位數)
        上下左右挪幾個像素,跟每一種牌的樣板算相關係數。
        分數 = min(整張、上半、下半) —— 萬子只看整張的話「萬」字佔一大半,
        5m/3m 會很像;上半是數字,分開算才分得出來。
輸出:  {"pai", "score", "second", "second_score"},寫進觀測的 p0_hand.drawn_lib。
        用不用、怎麼用由 turn_signals 決定(錄製端只記錄)。

⚠️ 赤五:樣板庫裡沒有那一種赤五的時候,赤五會被比成普通五。turn_signals 對
   五不單獨採用,要手牌模型也同意。
"""
import numpy as np

import os

DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui",
                           "p0_tile_lib.npz")
LIB_CONF = 0.90
LIB_PER = 4
SIZE = (40, 56)          # (w, h)
FIX_CONF = 0.70          # 暗牌信心低於這個才拿樣板比
FIX_SCORE = 0.80         # 樣板分數要 >= 這個
FIX_MARGIN = 0.10        #   而且贏第二名這麼多
OFFSETS = [(dx, dy) for dx in (-8, -4, 0, 4, 8) for dy in (-4, 0, 4)]


def _gray(img, box):
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    c = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(c, SIZE).astype(np.float32)


def _n(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum())
                                  + 1e-6))


def similarity(a, b):
    k = a.shape[0] // 2
    return min(_n(a, b), _n(a[:k], b[:k]), _n(a[k:], b[k:]))


class DrawnLibrary:
    def __init__(self, path=None):
        self.lib = {}
        self._mx = None
        if path:
            self.load(path)

    def load(self, path):
        import os
        if not os.path.isfile(path):
            return 0
        d = np.load(path)
        for k in d.files:
            self.lib[k] = [x for x in d[k]]
        self._mx = None
        return sum(len(v) for v in self.lib.values())

    def save(self, path):
        np.savez_compressed(path, **{k: np.stack(v) for k, v in
                                     self.lib.items() if v})

    def update(self, img, hand_dets, drawn_x=None):
        """把這一幀有把握的暗牌加進樣板庫(摸入牌那張不加)。"""
        for d in hand_dets:
            if d["conf"] < LIB_CONF:
                continue
            if drawn_x is not None and abs(d["center"][0] - drawn_x) < 47:
                continue
            L = self.lib.setdefault(d["pai"], [])
            if len(L) >= LIB_PER:
                continue
            g = _gray(img, d["box"])
            if g is not None:
                L.append(g)
                self._mx = None

    def _matrix(self):
        """樣板全部攤平、分三段(整張/上半/下半)各自標準化,矩陣乘法一次算完。"""
        if self._mx is not None:
            return self._mx
        labels, rows = [], []
        for p, L in self.lib.items():
            for t in L:
                labels.append(p)
                rows.append(t)
        T = np.stack(rows)
        self._mx = (labels, [_norm_rows(T, part)
                             for part in ("all", "top", "bot")])
        return self._mx

    def _rank(self, crops):
        labels, mats = self._matrix()
        C = np.stack(crops)
        sims = None
        for part, M in zip(("all", "top", "bot"), mats):
            S = _norm_rows(C, part) @ M.T          # (crops, templates)
            sims = S if sims is None else np.minimum(sims, S)
        per = sims.max(axis=0)                     # 每張樣板在最佳位移下的分數
        at = sims.argmax(axis=0)
        best = {}
        for p, v in zip(labels, per):
            if v > best.get(p, -1.0):
                best[p] = float(v)
        r = sorted(((v, p) for p, v in best.items()), reverse=True)
        return labels, per, at, r

    def classify_box(self, img, box):
        """一個框(上下左右挪幾個像素)跟樣板庫比:回傳 (牌, 分數, 第二名分數)。"""
        if not self.lib:
            return None
        x1, y1, x2, y2 = box
        crops = []
        for dx, dy in OFFSETS:
            g = _gray(img, (x1 + dx, y1 + dy, x2 + dx, y2 + dy))
            if g is not None:
                crops.append(g)
        if not crops:
            return None
        _l, _p, _a, r = self._rank(crops)
        return r[0][1], r[0][0], (r[1][0] if len(r) > 1 else 0.0)

    def fix_hand(self, img, hand_dets, drawn_x=None):
        """暗牌的第二意見:手牌模型沒把握的牌、還有漏框的空格,拿樣板庫比。

        ★ 2026-10-30 實戰東1:暗牌裡的 1m 整局讀不好 —— 漏框(只讀到 11/12 張)
          或讀成 8m(0.54)/9m(0.46)。起手牌倒推就多一張 8m、少一張 1m,打出 1m
          時守門器擋下整局。同一幀用樣板比:那張 8m(0.54) -> 1m 0.96(第二名
          2m 0.61)。
        回傳 [{"x": 中心 x, "pai", "score", "was", "was_conf", "box"}],
        "was" 是 None 表示那一格原本沒有框(補上的)。"""
        if not self.lib or not hand_dets:
            return []
        dets = sorted((d for d in hand_dets if drawn_x is None
                       or abs(d["center"][0] - drawn_x) >= 47),
                      key=lambda d: d["center"][0])
        if not dets:
            return []
        w = float(np.median([d["box"][2] - d["box"][0] for d in dets]))
        out = []

        def judge(box, was, was_conf, x):
            r = self.classify_box(img, box)
            if r is None:
                return
            pai, sc, sc2 = r
            if sc < FIX_SCORE or sc - sc2 < FIX_MARGIN:
                return
            if pai[:2] in ("5m", "5p", "5s"):
                red = is_red_five(img, box, pai[:2])
                if red is not None:
                    pai = pai[:2] + ("r" if red else "")
            out.append({"x": round(float(x), 1), "pai": pai,
                        "score": round(float(sc), 3), "was": was,
                        "was_conf": was_conf,
                        "box": [round(float(v), 1) for v in box]})

        for d in dets:
            if d.get("conf", 1.0) < FIX_CONF:
                judge(tuple(d["box"]), d["pai"], round(d["conf"], 3),
                      d["center"][0])
        for a, b in zip(dets, dets[1:]):
            gap = b["box"][0] - a["box"][2]
            n = int(round(gap / w)) if w > 0 else 0
            if gap < 0.6 * w or n < 1 or n > 3:
                continue
            step = gap / n
            y1 = min(a["box"][1], b["box"][1])
            y2 = max(a["box"][3], b["box"][3])
            for k in range(n):
                x1 = a["box"][2] + k * step
                judge((x1, y1, x1 + step, y2), None, None, x1 + step / 2)
        return out

    def classify(self, img, hand_dets, drawn_center):
        if not self.lib or not hand_dets or drawn_center is None:
            return None
        ws = [d["box"][2] - d["box"][0] for d in hand_dets]
        hs = [d["box"][3] - d["box"][1] for d in hand_dets]
        w, h = float(np.median(ws)), float(np.median(hs))
        cy = float(np.median([d["center"][1] for d in hand_dets]))
        cx = float(drawn_center[0])
        crops, boxes = [], []
        for dx, dy in OFFSETS:
            b = (cx + dx - w / 2, cy + dy - h / 2, cx + dx + w / 2,
                 cy + dy + h / 2)
            g = _gray(img, b)
            if g is not None:
                crops.append(g)
                boxes.append(b)
        if not crops:
            return None
        labels, per, at, r = self._rank(crops)
        out = {"pai": r[0][1], "score": round(r[0][0], 3)}
        if len(r) > 1:
            out["second"] = r[1][1]
            out["second_score"] = round(r[1][0], 3)
        out["has_red"] = sorted(p for p in self.lib if p.endswith("r"))
        b0 = out["pai"][:2]
        if b0 in ("5m", "5p", "5s"):
            j = max((k for k, p in enumerate(labels) if p[:2] == b0),
                    key=lambda k: per[k])
            red = is_red_five(img, boxes[int(at[j])], b0)
            if red is not None:
                out["pai"] = b0 + ("r" if red else "")
                out["red_checked"] = True
        return out


def is_red_five(img, box, base):
    """赤五看顏色(2026-10-28 用這場的手牌量,紅色像素比例):
        5m  上半(數字)  普通 <= 0.023    赤 >= 0.083   -> 門檻 0.05
        5s  上+下       普通 <= 0.133    赤 >= 0.30    -> 門檻 0.20
        5p  上+下       普通 <= 0.086    (這場沒有赤 5p)-> 門檻 0.20
    """
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    t = img[max(0, y1):y2, max(0, x1):x2]
    if t.size == 0:
        return None
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))
           & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70))
    k = red.shape[0] // 2
    top, bot = float(red[:k].mean()), float(red[k:].mean())
    if base == "5m":
        return top >= 0.05
    return top + bot >= 0.20


def _norm_rows(A, part):
    """(n, h, w) -> 每一列標準化(零均值、單位長度)的攤平向量。"""
    k = A.shape[1] // 2
    if part == "top":
        A = A[:, :k]
    elif part == "bot":
        A = A[:, k:]
    X = A.reshape(len(A), -1).astype(np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    return X


def observe(lib, img, hand_dets, region_dets):
    """錄製端每次跑完手牌模型呼叫:更新樣板庫、回傳摸入牌的比對結果(或 None)。"""
    from hand import pick_p0_drawn
    from drawn import DRAWN_CONF
    drawn = pick_p0_drawn([d for d in region_dets or []
                           if d["cls"] == "drawn_tile"
                           and d["conf"] >= DRAWN_CONF])
    dx = drawn["center"][0] if drawn else None
    lib.update(img, hand_dets, dx)
    if drawn is None:
        return None
    # 摸入牌一定在最右邊、跟暗牌隔一段空隙(理牌動畫中 drawn_tile 框會跳到
    # 別張牌上,2026-10-27 實機 H002 f1138)
    others = [d["center"][0] for d in hand_dets if abs(d["center"][0] - dx) >= 47]
    if others and dx - max(others) < 1.15 * 94.0:
        return None
    return lib.classify(img, hand_dets, drawn["center"])


def observe_hand(lib, img, hand_dets, region_dets):
    """錄製端:暗牌的樣板比對修正(寫進 p0_hand.hand_fix)。"""
    from hand import pick_p0_drawn
    from drawn import DRAWN_CONF
    drawn = pick_p0_drawn([d for d in region_dets or []
                           if d["cls"] == "drawn_tile"
                           and d["conf"] >= DRAWN_CONF])
    return lib.fix_hand(img, hand_dets, drawn["center"][0] if drawn else None)


def apply_hand_fix(hand_dets, fixes):
    """重播端:把 hand_fix 套回手牌框(改牌名、補漏掉的框)。"""
    if not fixes:
        return hand_dets
    out = [dict(d) for d in hand_dets]
    for f in fixes:
        if f.get("was") is None:
            b = f["box"]
            out.append({"pai": f["pai"], "conf": f["score"], "box": tuple(b),
                        "center": ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                        "lib_fix": True})
            continue
        for d in out:
            if abs(d["center"][0] - f["x"]) < 10 and d["pai"] == f["was"]:
                d["pai"] = f["pai"]
                d["conf"] = f["score"]
                d["lib_fix"] = True
                break
    return out
