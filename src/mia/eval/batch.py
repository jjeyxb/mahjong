"""逐幀平行跑評測。

為什麼需要這個
--------------
評測的成本幾乎全在模板比對:實測每幀 483 ms,其中 472 ms 是
:func:`~mia.vision.tiles.classify.classify` —— 每張牌要對 37 種牌各跑一次
正規化互相關,一幀 14 張就是 518 次。4327 幀跑完要四十分鐘。

而那四十分鐘裡機器只用了一核半。``cv2.matchTemplate`` 自己會開執行緒,但
搜尋區只有 121x307,太小,分不動。**平行化要切在幀這一層** —— 每一幀完全
獨立,是教科書上的那種可平行。

為什麼是行程而不是執行緒
------------------------
比對是 CPU 密集的 C 迴圈,執行緒會被 GIL 綁住。行程各自有自己的解譯器,
代價是每個工人要重新載入 session 索引與 37 張模板 —— 所以用 ``initializer``
載一次,而不是每個任務載一次。

⚠ **每個工人都要把 OpenCV 的執行緒數設回 1。** 不設的話 N 個行程各開 15 條
執行緒互搶核心,實測比單行程還慢。這是平行化最容易踩到的那個洞。

結果的順序與輸入相同(``ProcessPoolExecutor.map`` 保證),所以報告裡的幀序
不會因為平行而亂掉。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mia.eval.align import AlignedFrame
from mia.eval.report import FrameResult, evaluate_frame
from mia.recorder import SessionReader
from mia.recorder.session import FrameRecord
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet

__all__ = ["Crop", "run"]

#: 每個工人拿到的幀數。小一點負載比較均勻(張數判斷失敗的幀成本不同),
#: 大一點則省下跨行程傳遞的次數。64 幀約 30 秒,夠細也夠粗。
CHUNK = 64


@dataclass(frozen=True, slots=True)
class Crop:
    """從整張畫面切出手牌搜尋區所需的全部幾何。

    打包成一個物件而不是三個參數:它要跨行程傳給每個工人,而三個散裝的
    數字很容易在某一邊弄錯順序 —— 而弄錯的後果是切到別的地方,然後得到一份
    看起來正常但完全沒有意義的準確率報告。

    Attributes:
        roi: ``own_hand`` 在畫面上的像素框。
        top: 實際要切的上緣(``roi.y`` 減去 headroom,夾在 0 以上)。
        headroom: 牌框上方多留了幾個像素給滑鼠抬起來的那張牌。
    """

    roi: Rect
    top: int
    headroom: int

    @classmethod
    def around(cls, roi: Rect, headroom: int) -> Crop:
        """算出夾過邊界的切法。``headroom`` 會依實際切得到的量縮小。"""
        top = max(0, roi.y - headroom)
        return cls(roi=roi, top=top, headroom=roi.y - top)

    def cut(self, image: np.ndarray) -> np.ndarray:
        """切出含 headroom 的搜尋區(回傳 view,非複本)。"""
        roi = self.roi
        return image[self.top : roi.y + roi.height, roi.x : roi.x + roi.width]


@dataclass(frozen=True, slots=True)
class _Worker:
    """一個工人行程裡載一次就好的東西。"""

    session: SessionReader
    templates: TemplateSet
    crop: Crop
    by_index: dict[int, FrameRecord]

    def evaluate(self, frame: AlignedFrame) -> FrameResult:
        image = self.session.load_image(self.by_index[frame.index])
        return evaluate_frame(
            self.crop.cut(image), frame, self.templates, headroom=self.crop.headroom
        )


#: 工人行程裡的狀態。模組層的全域是 ``ProcessPoolExecutor`` 的 initializer
#: 唯一能把東西留給後續任務的地方。
_STATE: _Worker | None = None


def _setup(session_dir: Path, crop: Crop, skin: str | None) -> None:
    global _STATE
    # 見模組說明:不設回 1 的話 N 個行程會互搶核心,比單行程還慢。
    cv2.setNumThreads(1)
    session = SessionReader(session_dir)
    _STATE = _Worker(
        session=session,
        templates=TemplateSet.load(skin) if skin else TemplateSet.load(),
        crop=crop,
        by_index={record.index: record for record in session.manifest.frames},
    )


def _work(chunk: Sequence[AlignedFrame]) -> list[FrameResult]:
    assert _STATE is not None, "工人沒有經過 _setup —— initializer 沒跑?"
    return [_STATE.evaluate(frame) for frame in chunk]


def run(
    session_dir: Path,
    frames: Sequence[AlignedFrame],
    crop: Crop,
    *,
    skin: str | None = None,
    jobs: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[FrameResult]:
    """對配好的幀跑辨識。

    Args:
        session_dir: ``tools/record.py`` 錄的 session 目錄。傳路徑而不是
            :class:`SessionReader`,因為每個工人要自己開一個。
        frames: :func:`~mia.eval.align.align` 配好的幀。
        crop: 手牌搜尋區的幾何。
        skin: 牌面皮膚,``None`` 用內建的。
        jobs: 幾個工人。``None`` 表示依可用核心數;``1`` 走單行程
            —— profiling 與除錯時要的,平行化會讓堆疊變得看不懂。
        progress: ``(做完幾幀, 總共幾幀)``。四十分鐘的工作沒有進度就只能
            用 ``ps`` 猜,而 stdout 導到檔案時連印出來的東西都看不到。

    Returns:
        與 ``frames`` 同順序的結果。
    """
    if not frames:
        return []
    workers = jobs if jobs is not None else (os.process_cpu_count() or 1)
    chunks = [list(frames[i : i + CHUNK]) for i in range(0, len(frames), CHUNK)]
    results: list[FrameResult] = []

    if workers <= 1:
        _setup(session_dir, crop, skin)
        for chunk in chunks:
            results.extend(_work(chunk))
            if progress:
                progress(len(results), len(frames))
        return results

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_setup, initargs=(session_dir, crop, skin)
    ) as pool:
        for block in pool.map(_work, chunks):
            results.extend(block)
            if progress:
                progress(len(results), len(frames))
    return results
