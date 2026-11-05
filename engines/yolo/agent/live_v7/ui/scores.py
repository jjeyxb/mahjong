"""四家點數的辨識。

mjai 的 start_kyoku 有一個 scores 欄位(四家的點數),這是最後一個還沒讀的
必要欄位。做法跟 panel.py 同一套(固定 ROI + 樣板比對 + 整局投票),差別有
三個,都是量出來才知道的:

=== 1. 四家的字是四種方向,不是三種 ===
    P0(下)  0°     P1(右)  逆時針 90°(np.rot90 k=3 轉正)
    P2(上)  180°   P3(左)  順時針 90°(np.rot90 k=1 轉正)
  先把 ROI 轉正再做後面所有事,這樣四家共用同一套「由左到右切數字」的邏輯。

=== 2. 上下家跟左右家的字型不一樣,要分兩個池 ===
  用實機截圖量的(四家分別是 24000 / 25000 / 25000 / 25000,答案確定):
      同一個數字,P0 vs P2(都是水平)     0.70 ~ 0.80
      同一個數字,P1 vs P3(都是垂直)     0.74 ~ 0.83
      同一個數字,水平那組 vs 垂直那組    0.11 ~ 0.36   ← 完全不能共用
  原因是桌面有透視:上下兩家的數字大約 14x19,左右兩家轉正後大約 10x26,
  細長得多。正規化是等比例放進畫布(保留長寬比),所以長寬比不同就對不上。
  ⚠️ 這也是為什麼不能像本場/供託那樣四家共用一個池:那兩個計數器在畫面上
     是並排的同一種字,這裡不是。

  同一組之內的間距目前是「同數字最低 0.70 vs 不同數字最高 0.49」。夠用但不
  寬裕,所以 run_step9.py --report 要拿實機資料再量一次;如果間距守不住,就
  改成四家各自一個池(檔名已經是 score<座位>__<數字>.png,改的是查詢時的
  分組,不必重收樣板)。

=== 3. 有一個免費而且很強的檢查:總和 ===
    四家點數 + 供託×1000 = 100000(四人麻將,起始 25000×4)
  立直棒一放上桌,那家的點數立刻少 1000、供託加 1,所以這條在任何時刻都成立。
  它的價值在於「兩邊都可能說錯,而且錯了就對不起來」——供託是 panel.py 讀的,
  點數是這裡讀的,兩個獨立來源(鐵則 11、13)。任何一位數字看錯,總和幾乎
  一定不會剛好還是 100000。
  ⚠️ 供託還沒讀出來的時候要回報「無從檢查」,不能當成通過,也不能當成失敗。

  另外兩條規則不需要任何參數:
      點數一定是 100 的倍數(雀魂沒有 10 點、1 點的計算)
      位數 3~6(最少 3 位如 900,最多 6 位如 100000)

=== 讀取時機 ===
點數只在局與局之間變(局中只有立直會動,而那 1000 點的變化 mjai 是用 reach
事件表示的,不需要重讀)。所以跟面板一樣:一局讀一次,四家都確認而且總和
對得起來就 done,之後整局不再辨識。結算動畫期間數字會滾動,那時候讀到的
組合總和不會對,自然就被擋掉了——這是總和檢查的另一個好處,不必另外寫
「等動畫停下來」的邏輯。
"""
from collections import namedtuple

import cv2
import numpy as np

from panel import Templates, as_mask, normalize, similarity   # noqa: F401

# 量自實機截圖 panel.png(1920x1080)。ROI 都留了 6~9px 邊界,而且確認過
# 內容不貼邊——貼邊代表字被切掉,樣板就永遠對不上(panel.py 同樣的檢查)。
ROIS = {
    "P0": (900, 462, 120, 38),
    "P1": (1020, 378, 46, 82),
    "P2": (900, 346, 120, 38),
    "P3": (850, 378, 46, 82),
}

# np.rot90 的 k:把那一家的字轉成正的
ROT = {"P0": 0, "P1": 3, "P2": 2, "P3": 1}

# 樣板池:水平的兩家一組,垂直的兩家一組(見模組說明)
GROUP = {"P0": "flat", "P2": "flat", "P1": "side", "P3": "side"}

# 金色的點數字。青色的局數/余N 落在 H≈94,離這個範圍很遠,所以不會混進來。
GOLD_H = (10, 45)
GOLD_S = 80
GOLD_V = 110

# 數字的大小範圍(轉正之後)。實機量到水平 13~16 x 17~19、垂直 9~10 x 25~26。
# ⚠️ 只用「高度」擋雜訊,寬度不設上限:兩個數字黏在一起的時候寬度會變兩倍,
#    設上限會把整串點數丟掉。面板右側那條 11x66 的黃色裝飾是「細而高」,
#    高度這一關就擋掉了,不需要靠寬度。
MIN_AREA = 15
MIN_H, MAX_H = 8, 40
MIN_W = 5

# 字距(左緣到左緣)與單一數字的寬/高比,量自 panel.png:
#     P0 字距 16.0 寬高比 0.68~0.84      P2 字距 15.0 寬高比 0.76~0.82
#     P1 字距 10.8 寬高比 0.35~0.38      P3 字距 10.8 寬高比 0.36~0.38
# ★ 這個字型是固定字距的,四家各自固定。這件事很重要,因為數字之間的空隙
#   只有 1~2px(四家都是),稍微一點光暈或反鋸齒就會讓兩個數字連在一起,
#   連通元件就會把 5 位數看成 4 位數(實機真的發生了)。
#   固定字距讓「這一塊到底是幾個數字」變成算術而不是猜測:
#       k = round((這一塊的寬度 + 空隙) / 字距)
#   空隙 = 字距 - 單一數字寬度,而單一數字寬度用同一幀裡最窄的那一塊估
#   (最窄的幾乎不可能是黏起來的);整串全部黏成一塊時才退回用寬高比推。
PITCH = {"P0": 16.0, "P1": 10.8, "P2": 15.0, "P3": 10.8}
RATIO = {"P0": 0.76, "P1": 0.36, "P2": 0.78, "P3": 0.37}
PITCH_TOL = 0.25        # 同一幀量到的字距跟上面差超過這個比例就不採用

MIN_DIGITS, MAX_DIGITS = 3, 6
TOTAL = 100000          # 四人麻將的點數總和
STEP = 100              # 點數一定是 100 的倍數

Digit = namedtuple("Digit", "x norm merged")

# 這個門檻的角色只是「不要誤殺」,不是「保證答對」。量出來的數字:
#     數字清清楚楚分開時,正確答案 0.72~0.89
#     五位全部黏成一塊、靠字距切開時,正確答案最低掉到 0.49
#     但每一個字的「正確答案」都贏「最像的錯誤答案」至少 0.15(40/40)
# 所以比對本身(取最像的那一張)是可靠的,門檻只要低到不會把切出來的字擋掉。
# 真正把關的是總和 —— 那個檢查不需要門檻,而且錯了就是錯了。
# ⚠️ 目前只有 0/2/4/5 四種數字的樣板,錯誤答案的候選比較少,間距是樂觀的。
#    十種都收齊之後要用 run_step9.py --report 重新量一次。
MATCH_MIN = 0.45
MIN_VOTES = 2           # 同一個答案要出現幾幀才確認(跟 panel.py 同理由)


def upright(frame, seat):
    """把某一家的點數區切出來並轉正。回傳 BGR 影像。"""
    x, y, w, h = ROIS[seat]
    return np.ascontiguousarray(np.rot90(frame[y:y + h, x:x + w], ROT[seat]))


def gold_mask(bgr):
    """金色字的二值遮罩。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= GOLD_H[0]) & (h <= GOLD_H[1]) & (s > GOLD_S) & (v > GOLD_V))
    return as_mask(m, "點數遮罩").astype(np.uint8)


def _runs(mask, seat="P0"):
    """連通元件,依 x 排序,只留大小像數字的。回傳 [(x, y, w, h), …]。

    用連通元件而不是逐欄投影當第一步:元件自帶大小,可以直接把裝飾條那種
    「不是數字」的東西擋掉,逐欄投影做不到。黏在一起的數字會變成同一個
    元件,那由下一步(_count)處理。

    ★ 2026-10-26(live_v7)加:同一個數字斷成上下兩塊時先合回去。
      實機截圖(173.jpg、P0 莊家那張)左右兩家的「0」「3」頂端細筆畫斷開,
      上半塊太小被 MIN_AREA 擋掉,剩下半塊去比對 -> 0.34,整家讀不出來。
      x 範圍重疊一半以上的碎塊屬於同一個數字(數字是左右並排的,不會上下
      疊),先合併外接框再套大小條件。遮罩像素沒有任何改動,所以舊樣板照用。
      原本只對左右兩家做(上下兩家數字正下方的黃色行動指示條會被併進來)。
    ★ 2026-10-27:指示條改成合併前先依形狀拿掉(寬 > 2 字距、高 < MIN_H),
      四家都做合併 —— 實機 P0 的「5」也會斷成上下兩塊(15200 讀不出來)。
    """
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    # ★ 2026-10-27 實機:輪到那一家時,點數旁邊會出現黃色的行動指示條(轉正後
    #   在數字正下方、橫跨整串)。它跟每一個數字的 x 都重疊,合併會把整串點數
    #   跟指示條併成一塊 -> 讀不出來(E4 局 P3 的 44100 一整局都是 None)。
    #   指示條又寬又扁(寬 > 2 個字距、高 < 數字高度的一半),合併前先拿掉。
    parts = [list(int(v) for v in stats[i][:4]) for i in range(1, n)
             if stats[i, 4] >= 2
             and not (stats[i, 2] > 2 * PITCH[seat] and stats[i, 3] < MIN_H)]
    parts.sort(key=lambda b: b[0])
    merged = []
    for x, y, w, h in parts:
        hit = None
        for m in merged:
            ov = min(x + w, m[0] + m[2]) - max(x, m[0])
            if ov >= 0.5 * min(w, m[2]):
                hit = m
                break
        if hit is None:
            merged.append([x, y, w, h])
        else:
            x1, y1 = min(hit[0], x), min(hit[1], y)
            x2 = max(hit[0] + hit[2], x + w)
            y2 = max(hit[1] + hit[3], y + h)
            hit[:] = [x1, y1, x2 - x1, y2 - y1]
    out = []
    for x, y, w, h in merged:
        area = int(mask[y:y + h, x:x + w].sum())
        if area >= MIN_AREA and MIN_H <= h <= MAX_H and w >= MIN_W:
            out.append((x, y, w, h))
    out.sort(key=lambda b: b[0])
    return out


def _metrics(runs, seat):
    """這一幀的字距與空隙。回傳 (pitch, gap)。

    字距優先用同一幀量到的(相鄰左緣的最小間隔),但要跟量過的常數對得上
    ——只剩一塊、或每一塊都是黏起來的時候,量到的間隔會是字距的兩三倍,
    那時候寧可用常數。
    """
    pitch = PITCH[seat]
    lefts = [b[0] for b in runs]
    diffs = [b - a for a, b in zip(lefts, lefts[1:])]
    if diffs:
        got = min(diffs)
        if abs(got - pitch) <= PITCH_TOL * pitch:
            pitch = float(got)
    if len(runs) >= 2:
        # 最窄的那一塊當作單一數字的寬度 —— 但「1」本身就特別窄,不能當代表
        # (★ 2026-10-27 實機 44100:1 寬 5px -> 空隙算成 5.8 -> 每個 0 都被
        #   切成兩位,讀成 7 位數)。比正常寬高比窄很多的塊不列入。
        normal = [b[2] for b in runs if b[2] >= 0.7 * RATIO[seat] * b[3]]
        glyph = min(normal) if normal else RATIO[seat] * runs[0][3]
    else:
        glyph = RATIO[seat] * runs[0][3] if runs else pitch
    return pitch, max(0.0, pitch - glyph)


def _count(w, pitch, gap):
    """這一塊裡有幾個數字。k 個數字佔的寬度是 (k-1)×字距 + 數字寬,
    加回一個空隙就剛好是 k×字距。"""
    return max(1, int(round((w + gap) / pitch)))


def _cuts(mask, box, k, pitch):
    """把黏在一起的一塊切成 k 段。回傳每一段的 x 範圍 [(x0, x1), …]。

    切點不是等分:等分會從筆畫正中間切下去。先算出等分的位置,再在附近
    找「這一欄的筆畫最少」的地方切——兩個數字接在一起的地方本來就是整塊
    裡最細的地方,那個位置才是真正的邊界。
    """
    x, y, w, h = box
    cols = mask[y:y + h, x:x + w].sum(axis=0)
    win = max(1, int(round(pitch * 0.25)))
    edges = [0]
    for i in range(1, k):
        want = int(round(i * w / k))
        lo, hi = max(1, want - win), min(w - 1, want + win + 1)
        edges.append(min(range(lo, hi), key=lambda c: (cols[c], abs(c - want))))
    edges.append(w)
    return [(x + edges[i], x + edges[i + 1]) for i in range(k)]


def split_digits(mask, seat="P0"):
    """把遮罩切成一個一個數字(由左到右)。

    回傳 [Digit(x, norm, merged), …]。merged=True 表示這一位是從黏在一起的
    一塊切出來的——切出來的圖形邊緣可能沾到隔壁,拿來當樣板會愈用愈糟,
    所以收集工具要跳過這種。
    norm 可能是 None(切出來幾乎沒有筆畫),呼叫端要當成讀取失敗,不可以
    默默跳過——跳過會讓位數變少,把 25000 讀成 2500。
    """
    runs = _runs(mask, seat)
    if not runs:
        return []
    pitch, gap = _metrics(runs, seat)
    out = []
    for box in runs:
        x, y, w, h = box
        k = _count(w, pitch, gap)
        if k == 1:
            out.append(Digit(x, normalize(mask[y:y + h, x:x + w]), False))
            continue
        for x0, x1 in _cuts(mask, box, k, pitch):
            out.append(Digit(x0, normalize(mask[y:y + h, x0:x1]), True))
    out.sort(key=lambda d: d.x)
    return out


def read_seat(frame, seat, templates):
    """讀一家的點數。回傳 (點數 or None, 最弱的那一位的分數, 說明)。

    任何一位認不出來就整個回報失敗——五位裡讀對四位不是「差不多對」,而是
    一個完全錯誤的數字(鐵則 5)。
    """
    digits = split_digits(gold_mask(upright(frame, seat)), seat)
    if not digits:
        return None, 0.0, "看不到數字(被蓋住或還沒開始)"
    if not (MIN_DIGITS <= len(digits) <= MAX_DIGITS):
        return None, 0.0, f"切出 {len(digits)} 位,不在 {MIN_DIGITS}~{MAX_DIGITS} 之間"
    field = f"score{seat}"
    text, worst = "", 1.0
    for d in digits:
        norm = d.norm
        if norm is None:
            return None, 0.0, f"第 {len(text)+1} 位切出來是空的"
        label, score = templates.match(field, norm)
        if label is None or not label.isdigit() or score < MATCH_MIN:
            return None, score, f"第 {len(text)+1} 位認不出來(最像 {label},{score:.2f})"
        text += label
        worst = min(worst, score)
    value = int(text)
    if value % STEP:
        return None, worst, f"{value} 不是 {STEP} 的倍數,一定有一位看錯"
    return value, worst, ""


def check_total(values, kyotaku):
    """總和檢查。回傳 (True/False/None, 說明);None = 無從檢查。"""
    if any(v is None for v in values.values()) or len(values) != 4:
        return None, "四家還沒都讀到"
    if kyotaku is None:
        return None, "供託還沒讀到,無從檢查"
    got = sum(values.values()) + int(kyotaku) * 1000
    if got == TOTAL:
        return True, f"總和 {got} ✓"
    return False, f"總和 {got} != {TOTAL}(四家 {sum(values.values())} + 供託 {kyotaku}×1000)"


class ScoreReader:
    """四家點數。一局讀一次,四家都確認而且總和對得起來才算數。

    為什麼四家一起做而不是各自獨立:總和檢查需要四家同時成立,那是這裡最強
    的證據。分開做的話每一家都只能靠自己的相似度,而相似度是會騙人的。
    """

    def __init__(self, templates, match_min=MATCH_MIN, min_votes=MIN_VOTES):
        self.templates = templates
        self.match_min = match_min
        self.min_votes = min_votes
        self.value = {}          # 已確認的四家點數
        self.votes = {}          # 候選組合 -> 票數
        self.last = {}           # 這一幀各家的原始讀值(給畫面顯示)
        self.frames_used = 0
        self.unread = 0          # 四家沒讀齊的幀數
        self.unchecked = 0       # 讀齊了但供託還沒讀到,無從檢查的幀數
        self.rejected = []       # 被總和擋掉的組合,連理由一起留著

    @property
    def done(self):
        return len(self.value) == 4

    def update(self, frame, kyotaku=None):
        """吃一幀。回傳剛確認的四家點數 dict,還沒確認回傳 None。

        已經 done 就完全不辨識——一局之內不會變。
        """
        if self.done:
            return None
        self.frames_used += 1
        seats, notes = {}, {}
        for seat in ("P0", "P1", "P2", "P3"):
            v, sc, why = read_seat(frame, seat, self.templates)
            seats[seat] = v
            notes[seat] = (v, round(sc, 2), why)
        self.last = notes
        if any(v is None for v in seats.values()):
            self.unread += 1
            return None

        ok, why = check_total(seats, kyotaku)
        if ok is False:
            # 總和不對:一定有一位看錯,或是正在播結算動畫。不投票。
            self.rejected.append((dict(seats), why))
            del self.rejected[:-20]
            return None
        if ok is None:
            # 供託還沒讀到 -> 沒有證據,不能確認。這是刻意的:相似度自己
            # 說了不算(它是「最像哪一張樣板」,不是「這一定對」),總和
            # 才是這裡唯一能證偽的東西。供託讀不出來的時候寧可一直等,
            # 也不要送一組沒被檢查過的點數給 Mortal。
            self.unchecked += 1
            return None

        key = tuple(seats[s] for s in ("P0", "P1", "P2", "P3"))
        self.votes[key] = self.votes.get(key, 0) + 1
        # 總和對得起來的組合已經很強了,但還是要 MIN_VOTES 票——擋的是
        # 「動畫播到一半剛好被讀成另一組也剛好加得起來」這種巧合。
        if self.votes[key] < self.min_votes:
            return None
        self.value = dict(seats)
        return dict(self.value)

    def state(self):
        """給預覽視窗的一行(ASCII)。"""
        if self.done:
            return "SCORES " + " ".join(f"{s}={self.value[s]}"
                                        for s in ("P0", "P1", "P2", "P3"))
        bits = []
        for s in ("P0", "P1", "P2", "P3"):
            v, sc, why = self.last.get(s, (None, 0.0, "-"))
            bits.append(f"{s}={v if v is not None else '?'}({sc:.2f})")
        return "SCORES " + " ".join(bits)

    def report(self):
        lines = [f"  實際做辨識的幀數 {self.frames_used},"
                 f"四家沒讀齊 {self.unread} 幀,"
                 f"供託還沒讀到 {self.unchecked} 幀"]
        if self.done:
            lines.append("  確認:" + "  ".join(
                f"{s}={self.value[s]}" for s in ("P0", "P1", "P2", "P3")))
        else:
            lines.append("  還沒確認。這一幀各家的讀值:")
            for s in ("P0", "P1", "P2", "P3"):
                v, sc, why = self.last.get(s, (None, 0.0, "沒有紀錄"))
                lines.append(f"    {s}: {v if v is not None else '(讀不出來)'}"
                             f"  分數 {sc:.2f}  {why}")
        for combo, why in self.rejected[-5:]:
            lines.append(f"  ⚠️ 被總和擋掉:{combo} —— {why}")
        return lines

    def reset(self):
        self.value = {}
        self.votes = {}
        self.unread = 0
        self.unchecked = 0
