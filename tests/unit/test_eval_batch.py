"""逐幀平行跑評測。

**這裡測的是接線,不是準確率。** 平行化唯一不該改變的事情就是答案 ——
所以重點在「單行程與多行程得到一樣的結果,而且順序沒亂」。真正的準確率
由成對的錄影量出來(見 ``tools/evaluate.py``)。

素材刻意縮到三分之一:全尺寸一幀要 424 ms,縮小之後 49 ms 而 13 張牌仍然
全部認得出來 —— 測接線不需要全解析度,而讓整份測試慢上十倍是有代價的。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.capture.base import Frame, WindowInfo
from mia.eval.align import AlignedFrame, HandState
from mia.eval.batch import Crop, run
from mia.recorder import SessionWriter
from mia.utils.geometry import Rect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 真實手牌縮到三分之一。見模組說明。
SCALE = 3


class TestCrop:
    def test_headroom_is_taken_above_the_roi(self) -> None:
        crop = Crop.around(Rect(10, 100, 50, 80), headroom=30)
        assert (crop.top, crop.headroom) == (70, 30)

    def test_headroom_shrinks_to_what_the_image_actually_has(self) -> None:
        """ROI 離上緣只有 5px 時要不到 30px —— 要了就會切到負索引,

        而 numpy 的負索引不會報錯,它會從**另一頭**切,安靜地給你畫面底部。
        """
        crop = Crop.around(Rect(10, 5, 50, 80), headroom=30)
        assert (crop.top, crop.headroom) == (0, 5)

    def test_cut_returns_the_roi_plus_its_headroom(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        crop = Crop.around(Rect(20, 60, 100, 40), headroom=10)
        assert crop.cut(image).shape[:2] == (50, 100)


@pytest.fixture(scope="module")
def hand() -> np.ndarray:
    image = cv2.imread(str(FIXTURES / "hand_13_full.png"))
    height, width = image.shape[:2]
    return cv2.resize(image, (width // SCALE, height // SCALE), interpolation=cv2.INTER_AREA)


def _session(tmp_path: Path, window: WindowInfo, hand: np.ndarray, count: int) -> Path:
    """寫一個每一幀都是同一副手牌的 session。

    內容一樣,所以每一幀的答案也該一樣 —— 結果只要有一幀不同,就是平行化
    把幀與答案配錯了,而那正是這裡要擋的。
    """
    base = time.monotonic()
    with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
        for i in range(count):
            writer.add_frame(Frame(hand, window=window, scale=1.0, captured_at=base + i))
    return tmp_path / "s"


def _frames(count: int) -> list[AlignedFrame]:
    """標準答案隨幀序變化 —— 配錯了的話 truth 會對不上。"""
    return [
        AlignedFrame(i, float(i), HandState(float(i), (f"{i % 9 + 1}m",) * 13, None, 0, True))
        for i in range(count)
    ]


class TestRun:
    def test_nothing_to_do_returns_nothing(self) -> None:
        assert run(Path("不存在"), [], Crop.around(Rect(0, 0, 1, 1), 0)) == []

    def test_parallel_matches_serial(
        self, tmp_path: Path, window: WindowInfo, hand: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """平行化唯一不該改變的事情就是答案。

        ``CHUNK`` 調小才切得出多個批次 —— 預設 64 的話這幾幀全在同一批,
        跨批次的順序就根本沒被測到。
        """
        monkeypatch.setattr("mia.eval.batch.CHUNK", 2)
        count = 5
        directory = _session(tmp_path, window, hand, count)
        crop = Crop.around(Rect(0, 0, hand.shape[1], hand.shape[0]), headroom=0)
        frames = _frames(count)

        serial = run(directory, frames, crop, jobs=1)
        parallel = run(directory, frames, crop, jobs=3)

        assert [r.index for r in serial] == list(range(count)), "順序不該亂"
        assert serial == parallel

    def test_progress_counts_up_to_the_total(
        self, tmp_path: Path, window: WindowInfo, hand: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("mia.eval.batch.CHUNK", 2)
        count = 5
        directory = _session(tmp_path, window, hand, count)
        crop = Crop.around(Rect(0, 0, hand.shape[1], hand.shape[0]), headroom=0)

        seen: list[tuple[int, int]] = []
        run(directory, _frames(count), crop, jobs=1, progress=lambda d, t: seen.append((d, t)))

        assert seen[-1] == (count, count)
        assert [d for d, _ in seen] == sorted(d for d, _ in seen), "進度不該往回跳"

    def test_the_same_image_gives_the_same_reading(
        self, tmp_path: Path, window: WindowInfo, hand: np.ndarray
    ) -> None:
        """每一幀都是同一張圖,所以讀出來的牌也該每幀相同。

        不同的話代表工人之間有狀態殘留 —— 模板快取是 TemplateSet 的實例狀態,
        而每個工人各有一份。
        """
        directory = _session(tmp_path, window, hand, 3)
        crop = Crop.around(Rect(0, 0, hand.shape[1], hand.shape[0]), headroom=0)
        results = run(directory, _frames(3), crop, jobs=1)
        assert len({r.predicted for r in results}) == 1
        assert len(results[0].predicted) == 13
