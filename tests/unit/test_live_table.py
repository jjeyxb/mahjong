"""畫面 → 整桌狀態 → 危險度,完全不靠封包。

這一組的重點是**整條鏈真的接得起來**:素材圖進去,危險度出來。零件各自的
正確率由 test_vision_river / test_vision_dora / test_tracker_* 守著,
這裡守的是接線。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.calibration.table import Calibration
from mia.config.loader import load_config
from mia.live.bus import Dangers, UpdateBus
from mia.live.table import TableLink, TableWorker
from mia.utils.geometry import Rect, Size
from mia.vision.tiles.classify import TemplateSet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 素材那一幀錄製時鎖定的牌桌矩形。
TABLE = Rect(0, 174, 2536, 1430)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / "table_full.jpg"))
    assert frame is not None, "讀不到 table_full.jpg"
    return frame


@pytest.fixture(scope="module")
def calibration(image: np.ndarray) -> Calibration:
    return Calibration(
        table_rect=TABLE,
        image_size=Size(image.shape[1], image.shape[0]),
        source="manual",
    )


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    return TemplateSet.load()


def _worker(templates: TemplateSet) -> tuple[TableWorker, UpdateBus]:
    bus = UpdateBus()
    return TableWorker(bus, config=load_config().roi, templates=templates), bus


class TestTheWholeChain:
    def test_one_frame_produces_a_danger_report(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        worker, bus = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.offer_hand(["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z"])
        worker.tick()
        posts = [u for u in bus.drain() if isinstance(u, Dangers)]
        assert posts, "一張有牌河的畫面應該算得出危險度"
        assert posts[-1].report.tiles

    def test_it_reads_the_four_rivers(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """素材那一幀四家合計 65 張。讀不到的格子仍然佔位,所以長度要對。"""
        worker, _ = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.offer_hand(["1m"])
        worker.tick()
        counts = [len(river) for river in worker._rivers.rivers]  # noqa: SLF001
        assert sum(counts) == 65
        assert counts == [15, 17, 17, 16]

    def test_it_reads_the_dora_indicator(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        worker, _ = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.offer_hand(["1m"])
        worker.tick()
        assert worker._table.dora_markers == ["3s"]  # noqa: SLF001

    def test_seeing_the_same_frame_twice_posts_once(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """牌河沒變就不該再投 —— 投了會讓 UI 每秒重畫一次一模一樣的東西。"""
        worker, bus = _worker(templates)
        worker.offer_hand(["1m", "9p", "5z"])
        worker.offer_frame(image, calibration)
        worker.tick()
        bus.drain()
        worker.offer_frame(image, calibration)
        worker.tick()
        assert not [u for u in bus.drain() if isinstance(u, Dangers)]

    def test_without_a_hand_it_stays_quiet(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """危險度是「我切這張會不會放銃」—— 沒有手牌就沒有問題可以回答。"""
        worker, bus = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.tick()
        assert not [u for u in bus.drain() if isinstance(u, Dangers)]


class TestItOnlyKeepsTheLatestFrame:
    def test_an_older_frame_is_dropped(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """畫面每秒來十五張而這裡一秒才吃一張 —— 積起來補播是在顯示
        已經不成立的局面,與 UpdateBus 同一個理由。"""
        worker, _ = _worker(templates)
        worker.offer_frame(np.zeros_like(image), calibration)
        worker.offer_frame(image, calibration)
        worker.offer_hand(["1m"])
        worker.tick()
        assert sum(len(r) for r in worker._rivers.rivers) == 65  # noqa: SLF001

    def test_nothing_to_do_without_a_frame(self, templates: TemplateSet) -> None:
        worker, bus = _worker(templates)
        worker.tick()
        assert not bus.drain()


class TestTheLink:
    """兩個功能各自開關,所以中間要一個不會被重建的東西。"""

    def test_it_forwards_to_whoever_is_attached(self, templates: TemplateSet) -> None:
        link = TableLink()
        worker, _ = _worker(templates)
        link.attach(worker)
        link.offer_hand(["1m"])
        assert worker._hand == ("1m",)  # noqa: SLF001

    def test_with_nobody_attached_it_just_drops(self) -> None:
        """牌桌那條路關著的時候,擷取那條路照跑不誤。"""
        link = TableLink()
        link.offer_hand(["1m"])
        assert not link.attached

    def test_a_later_worker_is_not_cleared_by_an_earlier_one(
        self, templates: TemplateSet
    ) -> None:
        """舊的 worker 收尾時會拔掉自己 —— 但不該拔掉已經接上的新的那個。"""
        link = TableLink()
        old, _ = _worker(templates)
        new, _ = _worker(templates)
        link.attach(old)
        link.attach(new)
        link.detach(old)
        assert link.attached
