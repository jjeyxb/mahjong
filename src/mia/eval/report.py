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

信心 × 正確
-----------
總體準確率回答不了實際使用時最要緊的那個問題:**錯的時候,系統知不知道自己錯了?**
把每一張牌按「有沒有信心」與「對不對」交叉統計,四格的意義完全不同:

===========  ==============================  ==============================
             對                              錯
===========  ==============================  ==============================
有信心       正常                            **靜默錯誤** —— 唯一真正危險的
沒信心       棄權的代價                      系統自己舉手了,上層丟掉就好
===========  ==============================  ==============================

只有左下與右下兩格是可以靠門檻換的:門檻拉高,右上的靜默錯誤會往右下移
(變成「我不確定」),但左上也會往左下移(本來對的被白白丟掉)。
:data:`SWEEP` 就是在量這個交換比。

:class:`TileOutcome` 存的是**分數本身**而不是現行門檻下的布林值,所以門檻掃描
不需要重跑辨識 —— 4327 幀跑一次要幾分鐘,而掃描是對同一份結果重新計算。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from mia.eval.align import AlignedFrame
from mia.mjai.tiles import ms_to_mjai
from mia.vision.tiles.classify import (
    MIN_BRIGHTNESS,
    MIN_MARGIN,
    MIN_SCORE,
    TemplateSet,
    classify_hand,
)
from mia.vision.tiles.hand import read_hand

__all__ = ["SWEEP", "FrameResult", "TileOutcome", "TileReport", "evaluate_frame", "summarize"]

#: 門檻掃描用的 ``(最低分數, 最小差距)`` 組合。第一組是**現行值** ——
#: 掃描要看得出「現在在哪裡」,不然後面幾組沒有比較基準。
SWEEP = (
    (MIN_SCORE, MIN_MARGIN),
    (0.60, 0.06),
    (0.65, 0.08),
    (0.70, 0.12),
    (0.75, 0.18),
)


@dataclass(frozen=True, slots=True)
class TileOutcome:
    """一張牌的辨識結果,連同它的信心分數。

    Attributes:
        label: CV 讀出來的牌(MJAI 記法)。
        correct: 這一張算不算對。配對方式見 :func:`evaluate_frame`。
        score: 相關係數。
        margin: 與次高分的差距。
        brightness: 比對區與模板的亮度比。見
            :data:`~mia.vision.tiles.classify.MIN_BRIGHTNESS`。
    """

    label: str
    correct: bool
    score: float
    margin: float
    brightness: float = 1.0

    def confident_at(self, min_score: float, min_margin: float) -> bool:
        """在指定門檻下算不算有信心。判定與 :attr:`Match.is_confident` 同一套。

        **亮度不隨門檻掃描而變。** 分數與差距問的是「像不像」,可以在準確率
        與棄權成本之間換;亮度問的是「這一格到底是不是牌」,那沒有什麼好換的
        —— 一塊桌布不會因為門檻放寬就變成一張牌。
        """
        return (
            self.score >= min_score
            and self.margin >= min_margin
            and self.brightness >= MIN_BRIGHTNESS
        )

    @property
    def confident(self) -> bool:
        """現行門檻下算不算有信心。"""
        return self.confident_at(MIN_SCORE, MIN_MARGIN)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """一幀的比對結果。

    Attributes:
        index: 幀序號。
        truth: 標準答案(MJAI 表示法,已排序)。
        predicted: CV 讀出來的(MJAI 表示法,已排序)。
        count_ok: 張數對不對。
        tiles: 每一張的辨識結果與信心分數。**預設是空的** —— 只驗統計邏輯的
            測試不必編出假的分數,而需要明細的統計會跳過沒有明細的幀。
    """

    index: int
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    count_ok: bool
    tiles: tuple[TileOutcome, ...] = ()

    @property
    def low_confidence(self) -> int:
        """有幾張的比對分數不夠有把握。"""
        return sum(not tile.confident for tile in self.tiles)

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
        confident_confusions: 同上,但**只算高信心卻認錯的**。這張表才是要修的
            清單 —— 渲染是確定性的,所以同一張牌在同一個位置會一直錯成同一個
            答案,它是有限且可列舉的 bug,不是一個機率。
    """

    frames: list[FrameResult] = field(default_factory=list)
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)
    confident_confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

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

    @property
    def scored_tiles(self) -> tuple[TileOutcome, ...]:
        """所有帶著信心分數的牌。沒有明細的幀不計入。"""
        return tuple(tile for frame in self.frames for tile in frame.tiles)

    @property
    def accepted_tiles(self) -> tuple[TileOutcome, ...]:
        """只算**張數通得過檢查**的幀上的牌。

        張數不對的幀上層本來就會整幀丟掉(即時路徑看
        :attr:`~mia.vision.tiles.hand.Hand.is_plausible`),所以那些幀上的
        靜默錯誤永遠不會流到使用者面前。把它們算進去會高估實際的風險 ——
        而這個數字正是用來決定「要不要逼使用者逐項確認」的,高估的代價是
        做了一個沒必要的功能。

        反過來也要留著 :attr:`scored_tiles`:張數檢查本身可能漏掉
        「剛好多讀一張又少讀一張」的情形,兩個數字一起看才知道差距在哪。
        """
        return tuple(tile for frame in self.frames if frame.count_ok for tile in frame.tiles)

    def _tiles(self, *, accepted: bool) -> tuple[TileOutcome, ...]:
        return self.accepted_tiles if accepted else self.scored_tiles

    def confidence_table(
        self,
        min_score: float = MIN_SCORE,
        min_margin: float = MIN_MARGIN,
        *,
        accepted: bool = False,
    ) -> dict[tuple[bool, bool], int]:
        """``(有信心, 對) → 張數``。四格的意義見模組說明。"""
        table: Counter[tuple[bool, bool]] = Counter()
        for tile in self._tiles(accepted=accepted):
            table[(tile.confident_at(min_score, min_margin), tile.correct)] += 1
        return dict(table)

    def silent_error_rate(
        self,
        min_score: float = MIN_SCORE,
        min_margin: float = MIN_MARGIN,
        *,
        accepted: bool = False,
    ) -> float:
        """高信心卻認錯,佔全部牌的比例。

        **這是整份報告最要緊的一個數字。** 自檢抓不到它(它跟正確答案在系統
        內部無法區分),也只有人工確認擋得住它 —— 所以它多大,直接決定了
        「要不要逼使用者逐項確認」這個設計問題的答案。

        沒有任何明細時回 ``nan``:回 0 會被讀成「完全沒有靜默錯誤」,
        而實際上是「沒測」。
        """
        tiles = self._tiles(accepted=accepted)
        if not tiles:
            return float("nan")
        bad = sum(1 for t in tiles if not t.correct and t.confident_at(min_score, min_margin))
        return bad / len(tiles)

    def abstain_rate(
        self,
        min_score: float = MIN_SCORE,
        min_margin: float = MIN_MARGIN,
        *,
        accepted: bool = False,
    ) -> float:
        """其實是對的,卻被判為沒信心 —— 棄權的代價。

        與 :meth:`silent_error_rate` 是同一個門檻的兩端。單看任一邊都會得到
        錯的結論:門檻拉到極高能讓靜默錯誤歸零,代價是什麼都不敢講。
        """
        tiles = self._tiles(accepted=accepted)
        if not tiles:
            return float("nan")
        wasted = sum(1 for t in tiles if t.correct and not t.confident_at(min_score, min_margin))
        return wasted / len(tiles)

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

        tiles = self.scored_tiles
        if tiles:
            table = self.confidence_table()
            lines += [
                "",
                f"信心 × 正確(共 {len(tiles)} 張)",
                f"  有信心   對 {table.get((True, True), 0):6d}"
                f"   錯 {table.get((True, False), 0):5d}  ← 靜默錯誤",
                f"  沒信心   對 {table.get((False, True), 0):6d}"
                f"   錯 {table.get((False, False), 0):5d}  ← 自己舉手了",
                f"  靜默錯誤率 {self.silent_error_rate():.3%}  ← 全部 {len(tiles)} 張",
                f"  實際風險   {self.silent_error_rate(accepted=True):.3%}"
                f"  ← 只算張數通過的 {len(self.accepted_tiles)} 張,"
                f"張數不對的幀上層本來就會丟掉",
                f"  棄權成本   {self.abstain_rate():.3%}",
                "  門檻掃描(分數/差距 → 靜默錯誤 / 棄權成本):",
            ]
            for min_score, min_margin in SWEEP:
                lines.append(
                    f"    {min_score:.2f} / {min_margin:.2f}"
                    f"  {self.silent_error_rate(min_score, min_margin):>8.3%}"
                    f"  {self.abstain_rate(min_score, min_margin):>8.3%}"
                )

        if self.confident_confusions:
            lines.append("  高信心卻認錯的(自檢抓不到的那些):")
            for (truth, wrong), count in self.confident_confusions.most_common(8):
                lines.append(f"    {truth} → {wrong}  ×{count}")
        return "\n".join(lines)


def evaluate_frame(
    image: np.ndarray, frame: AlignedFrame, templates: TemplateSet, *, headroom: int = 0
) -> FrameResult:
    """對一幀跑完整的手牌辨識,與標準答案比對。

    Args:
        image: ``own_hand`` ROI 的 BGR 影像;``headroom`` 不為 0 時是上緣多切
            了一段的版本。
        frame: :func:`~mia.eval.align.align` 配好的幀。
        templates: 牌面模板集。
        headroom: 牌框上方額外的搜尋高度。**要與即時管線用同一個值** ——
            不然這份報告量的就不是真的會跑的那條路。

    Returns:
        這一幀的比對結果。
    """
    hand = read_hand(image[headroom:])
    concealed, drawn = classify_hand(image, hand, templates, headroom=headroom)
    matches = [*concealed, *([drawn] if drawn is not None else [])]

    # classify 回的是雀魂記法(0m/1z),標準答案是 MJAI 記法(5mr/E)
    labels = tuple(ms_to_mjai(m.label) for m in matches)  # 未排序,要與 matches 對齊
    truth = tuple(sorted(frame.truth.tiles))

    # 哪幾張算錯:CV 讀出來、但標準答案裡沒有的那些(多重集合差)。這與
    # FrameResult.correct_tiles 的交集是同一件事的兩面,刻意**不逐槽比對** ——
    # 一張錯位就會讓後面整串都算錯,那個數字沒有參考價值。
    #
    # 代價是一幀裡錯超過一張時,「哪一張對應哪一張」未必是真的對應關係。
    # 但這裡只要知道「這張算不算對」,不需要知道它本來該是什麼。
    extra = Counter(labels) - Counter(truth)
    tiles: list[TileOutcome] = []
    for match, label in zip(matches, labels, strict=True):
        wrong = extra[label] > 0
        if wrong:
            extra[label] -= 1
        tiles.append(TileOutcome(label, not wrong, match.score, match.margin, match.brightness))

    return FrameResult(
        index=frame.index,
        truth=truth,
        predicted=tuple(sorted(labels)),
        count_ok=len(labels) == len(truth),
        tiles=tuple(tiles),
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

        # 同樣的配對,但帶著信心。錯的那幾張牌的 label 恰好就是 extra 這個
        # 多重集合,所以按 label 排序之後兩邊對得起來。
        wrong_tiles = sorted((t for t in frame.tiles if not t.correct), key=lambda t: t.label)
        for truth, outcome in zip(sorted(missing.elements()), wrong_tiles, strict=False):
            if outcome.confident:
                report.confident_confusions[(truth, outcome.label)] += 1
    return report


def _ratio(part: int, total: int) -> float:
    return part / total if total else 0.0
