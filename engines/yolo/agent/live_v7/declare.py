"""宣告橫幅(吃/碰/槓/立直/榮和/自摸/流局)-> 一次宣告一個事件。

重用 event_zones.py 的座位歸屬(classify)、逐類別門檻(act_conf_for)、
最小幀數(min_frames_for),不另訂數字。

=== 一次宣告一個事件 ===
橫幅會連續顯示好幾幀。這裡用「段」:同一家、同一組形狀(吃碰槓共用一組,
因為模型會在三者之間改變心意)連續出現就是同一段,斷掉超過 GAP_END 幀
才算結束。

=== 吃/碰被讀錯 ===
truth.txt 記錄過「真的是碰,前幾幀讀成吃 0.22~0.45」以及「同一個橫幅先報吃
後報碰,被算成兩次」。所以吃碰槓不在第一幀就下結論,而是累積 DECIDE_FRAMES
幀之後取「信心度總和最高」的類別,一段只送一次。
立直/榮和/自摸/流局沒有這個問題,達到最小幀數就送。

=== 貼邊 ===
被畫面邊緣切到的框不算(真橫幅一定完整在畫面內),region_model 說明過理由。
"""
from collections import defaultdict
from dataclasses import dataclass

import _paths  # noqa: F401
from event_zones import act_conf_for, classify, min_frames_for

CALL_CLASSES = ("chi", "pon", "kan")
OTHER_CLASSES = ("reach", "ron", "tsumo", "ryukyoku")
ALL_CLASSES = CALL_CLASSES + OTHER_CLASSES

# 吃碰槓累積幾幀再決定是哪一類。3 幀 @9fps ≈ 330ms。
DECIDE_FRAMES = 3
# 一段中斷超過幾幀算結束。
GAP_END = 6


@dataclass
class Declaration:
    cls: str            # chi/pon/kan/reach/ron/tsumo/ryukyoku
    seat: object        # P0..P3;流局是 None
    frame: int          # 這一段第一幀
    t_ms: float
    decided_frame: int
    conf: float         # 勝出類別的平均信心度
    votes: dict         # 各類別的信心度總和(除錯用)
    kind: str = "declaration"


class _Episode:
    def __init__(self, frame, t_ms):
        self.first = frame
        self.t_ms = t_ms
        self.last = frame
        self.frames = 0
        self.sum = defaultdict(float)
        self.cnt = defaultdict(int)
        self.sent = False


class DeclarationDetector:
    def __init__(self, decide_frames=DECIDE_FRAMES, gap_end=GAP_END,
                 act_conf=None):
        self.decide_frames = decide_frames
        self.gap_end = gap_end
        self.act_conf = act_conf          # 全部類別共用的覆寫(實驗用)
        self.ep = {}                      # (group, seat) -> _Episode
        self.rejected = defaultdict(int)  # 理由 -> 次數

    @staticmethod
    def _group(cls):
        return "call" if cls in CALL_CLASSES else cls

    def feed(self, frame, t_ms, region_dets):
        out = []
        if region_dets is None:
            return out
        seen = set()
        for d in region_dets:
            c = d["cls"]
            if c not in ALL_CLASSES:
                continue
            if d.get("clipped"):
                self.rejected["clipped"] += 1
                continue
            if c == "ryukyoku":
                seat = None
            else:
                seat, _shape = classify(d["center"][0], d["center"][1], c)
                if seat not in ("P0", "P1", "P2", "P3"):
                    self.rejected[f"zone:{seat}"] += 1
                    continue
            if d["conf"] < act_conf_for(c, seat, self.act_conf):
                self.rejected["conf"] += 1
                continue
            key = (self._group(c), seat)
            ep = self.ep.get(key)
            if ep is None or frame - ep.last > self.gap_end:
                ep = self.ep[key] = _Episode(frame, t_ms)
            if key not in seen:
                ep.frames += 1
                seen.add(key)
            ep.last = frame
            ep.sum[c] += d["conf"]
            ep.cnt[c] += 1

        for key, ep in list(self.ep.items()):
            if ep.sent:
                if frame - ep.last > self.gap_end:
                    del self.ep[key]
                continue
            group, seat = key
            best = max(ep.sum, key=lambda k: ep.sum[k])
            need = min_frames_for(best)
            ended = frame - ep.last > self.gap_end
            if group == "call":
                ready = ep.frames >= max(need, self.decide_frames) or (
                    ended and ep.frames >= need)
            else:
                ready = ep.frames >= need
            if ready:
                ep.sent = True
                out.append(Declaration(
                    best, seat, ep.first, ep.t_ms, frame,
                    round(ep.sum[best] / max(1, ep.cnt[best]), 3),
                    {k: round(v, 3) for k, v in ep.sum.items()}))
            elif ended:
                self.rejected[f"too_short:{best}"] += 1
                del self.ep[key]
        return out
