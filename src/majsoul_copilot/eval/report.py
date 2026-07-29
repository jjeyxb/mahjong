"""CV 準確率報告:拿封包當標準答案,量手牌辨識到底對多少。

這是專題最強的論述點 —— 純 CV 沒辦法宣稱 100%,但**可以把錯誤變成一個可量測、
可追蹤的數字**,並且指出錯在哪幾種牌上。混淆對照表通常比總體準確率有用得多:
「9s 被認成 4s」是可以修的,「準確率 93%」不是。

三個層級的數字
--------------
* **張數**:讀到的暗手牌張數對不對。錯了代表槽位佔用判斷壞掉,後面都不用看。
* **每張牌**:逐槽比對牌名。這是主要指標。
* **整手**:整副手牌完全正確的比例。比每張牌的準確率低很多是正常的
  —— 13 張裡錯一張,整手就算錯。實際使用時要看的是這個,因為向聽計算
  吃的是整手牌。

赤寶牌另外算
------------
``5m`` 與 ``5mr`` 在向聽計算上等價,但打點差一番。分開統計才看得出模板庫
有沒有把赤寶牌那三張認出來 —— 混在一起的話,把所有赤五都認成普通五,
總體準確率只掉 2% 左右,很容易被忽略過去。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from majsoul_copilot.eval.align import AlignedFrame
from majsoul_copilot.mjai.tiles import ms_to_mjai
from majsoul_copilot.vision.tiles.classify import TemplateSet, classify_hand
from majsoul_copilot.vision.tiles.hand import read_hand

__all__ = ["FrameResult", "TileReport", "evaluate_frame", "summarize"]


@dataclass(frozen=True, slots=True)
class FrameResult:
    """一幀的比對結果。

    Attributes:
        index: 幀序號。
        truth: 標準答案(MJAI 表示法,已排序)。
        predicted: CV 讀出來的(MJAI 表示法,已排序)。
        count_ok: 張數對不對。
        low_confidence: 有幾張的比對分數不夠有把握。
    """

    index: int
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    count_ok: bool
    low_confidence: int

    @property
    def exact(self) -> bool:
        """整手完全正確。"""
        return self.truth == self.predicted

    @property
    def correct_tiles(self) -> int:
        """對了幾張。張數不同時以多重集合取交集 —— 逐位比對會因為一張錯位
        就整串往後錯,那個數字會低估到沒有參考價值。"""
        return sum((Counter(self.truth) & Counter(self.predicted)).values())


@dataclass(slots=True)
class TileReport:
    """整份報告。

    Attributes:
        frames: 每一幀的結果。
        confusions: ``(標準答案, 認成什麼)`` → 次數。
    """

    frames: list[FrameResult] = field(default_factory=list)
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def count_accuracy(self) -> float:
        return _ratio(sum(f.count_ok for f in self.frames), self.total_frames)

    @property
    def exact_accuracy(self) -> float:
        return _ratio(sum(f.exact for f in self.frames), self.total_frames)

    @property
    def tile_accuracy(self) -> float:
        correct = sum(f.correct_tiles for f in self.frames)
        total = sum(len(f.truth) for f in self.frames)
        return _ratio(correct, total)

    @property
    def red_five_recall(self) -> float:
        """標準答案裡的赤五,有多少被認成赤五。

        分母是 0(這份錄影裡根本沒摸到赤五)時回 ``nan`` —— 回 0 或 1 都會
        被誤讀成「認不出來」或「全對」,而實際上是「沒測到」。
        """
        held = correct = 0
        for frame in self.frames:
            truth, predicted = Counter(frame.truth), Counter(frame.predicted)
            for tile, count in truth.items():
                if tile.endswith("r"):
                    held += count
                    correct += min(count, predicted[tile])
        return correct / held if held else float("nan")

    def summary(self) -> str:
        if not self.frames:
            return "沒有任何可評分的幀 —— 見 align 的說明,轉場期間的幀不算數。"
        lines = [
            f"可評分的幀 {self.total_frames}",
            f"  張數正確   {self.count_accuracy:.1%}",
            f"  每張牌正確 {self.tile_accuracy:.1%}",
            f"  整手正確   {self.exact_accuracy:.1%}",
            f"  赤寶牌召回 {self.red_five_recall:.1%}",
        ]
        if self.confusions:
            lines.append("  最常認錯的:")
            for (truth, wrong), count in self.confusions.most_common(8):
                lines.append(f"    {truth} → {wrong}  ×{count}")
        return "\n".join(lines)


def evaluate_frame(
    image: np.ndarray, frame: AlignedFrame, templates: TemplateSet
) -> FrameResult:
    """對一幀跑完整的手牌辨識,與標準答案比對。

    Args:
        image: ``own_hand`` ROI 的 BGR 影像。
        frame: :func:`~majsoul_copilot.eval.align.align` 配好的幀。
        templates: 牌面模板集。

    Returns:
        這一幀的比對結果。
    """
    hand = read_hand(image)
    concealed, drawn = classify_hand(image, hand, templates)
    matches = [*concealed, *([drawn] if drawn is not None else [])]

    # classify 回的是雀魂記法(0m/1z),標準答案是 MJAI 記法(5mr/E)
    predicted = tuple(sorted(ms_to_mjai(m.label) for m in matches))
    truth = tuple(sorted(frame.truth.tiles))

    return FrameResult(
        index=frame.index,
        truth=truth,
        predicted=predicted,
        count_ok=len(predicted) == len(truth),
        low_confidence=sum(not m.is_confident for m in matches),
    )


def summarize(results: list[FrameResult]) -> TileReport:
    """把每一幀的結果匯總,並統計混淆對照。

    混淆只在**張數正確**的幀上統計。張數不對時無法確定哪張對應哪張,
    硬配出來的對照表只是雜訊。
    """
    report = TileReport(frames=list(results))
    for frame in results:
        if not frame.count_ok:
            continue
        missing = Counter(frame.truth) - Counter(frame.predicted)
        extra = Counter(frame.predicted) - Counter(frame.truth)
        # 張數相同時,少掉的與多出來的一樣多 —— 兩兩配對就是認錯的方向。
        # 一幀裡錯超過一張時這個配對未必是真的對應關係,但統計上仍然有指示性。
        for truth, wrong in zip(sorted(missing.elements()), sorted(extra.elements()), strict=False):
            report.confusions[(truth, wrong)] += 1
    return report


def _ratio(part: int, total: int) -> float:
    return part / total if total else 0.0
