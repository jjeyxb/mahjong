"""畫面 → 整桌狀態 → 危險度,完全不靠封包。

這一組的重點是**整條鏈真的接得起來**:素材圖進去,危險度出來。零件各自的
正確率由 test_vision_river / test_vision_dora / test_tracker_* 守著,
這裡守的是接線。
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.calibration.table import Calibration
from mia.config.loader import load_config
from mia.live.bus import Dangers, UpdateBus
from mia.live.table import TableLink, TableWorker, _PendingFaces
from mia.utils.geometry import Rect, Size
from mia.vision.tiles.classify import TemplateSet
from mia.vision.tiles.melds import MeldFaces

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


def _settle(worker: TableWorker, image: np.ndarray, calibration: Calibration) -> None:
    """等背景認完副露牌面,再讀一輪把結果收回來。"""
    for job in worker._pending_faces:  # noqa: SLF001
        job.future.result(timeout=60)
    worker.offer_frame(image, calibration)
    worker.tick()


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
        """牌河沒變就不該再投 —— 投了會讓 UI 每秒重畫一次一模一樣的東西。

        先讀兩次:這張圖上有副露,而副露要連兩次讀到同樣的長度才採信
        (``mia.tracker.melds``),第二次才算定;它的牌面在另一條執行緒認,
        再等它填回來。之後同一張圖就不該再投。
        """
        worker, bus = _worker(templates)
        worker.offer_hand(["1m", "9p", "5z"])
        for _ in range(2):
            worker.offer_frame(image, calibration)
            worker.tick()
        _settle(worker, image, calibration)
        bus.drain()
        worker.offer_frame(image, calibration)
        worker.tick()
        assert not [u for u in bus.drain() if isinstance(u, Dangers)]

    def test_a_called_group_is_recorded_with_what_it_could_read(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """這張圖的下家吃了 1 2 3 筒(2、3 筒橫放疊在一起)。

        讀得到的記下來,讀不到的留未知 —— 不猜。
        """
        worker, _ = _worker(templates)
        worker.offer_hand(["1m", "9p", "5z"])
        for _ in range(2):
            worker.offer_frame(image, calibration)
            worker.tick()
        melds = worker._table.players[1].melds  # noqa: SLF001
        assert melds == [("?", "?", "?")], "組數不等牌面,當下就要算數"
        _settle(worker, image, calibration)
        melds = worker._table.players[1].melds  # noqa: SLF001
        assert len(melds) == 1
        known = {tile for tile in melds[0] if tile != "?"}
        assert known
        assert known <= {"1p", "2p", "3p"}

    def test_faces_from_the_last_kyoku_are_dropped(self, templates: TemplateSet) -> None:
        """認牌面要好幾秒,中間換局的話,認好的那組屬於上一局。"""
        worker, _ = _worker(templates)
        old = worker._table  # noqa: SLF001
        old.players[1].melds.append(("?", "?", "?"))
        done: Future[MeldFaces] = Future()
        done.set_result(MeldFaces(frozenset({"7z"})))
        worker._pending_faces.append(_PendingFaces(old, 1, 0, done))  # noqa: SLF001
        worker._new_kyoku()  # noqa: SLF001
        worker._collect_meld_faces()  # noqa: SLF001
        assert worker._table.players[1].melds == []  # noqa: SLF001
        assert worker._pending_faces == []  # noqa: SLF001

    def test_the_hud_gives_the_round_and_the_winds(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """東 4 局,莊家是下家 —— 畫面這條路終於知道場風與自風了。"""
        worker, _ = _worker(templates)
        for _ in range(2):
            worker.offer_frame(image, calibration)
            worker.tick()
        assert worker.round is not None
        assert worker.round.kyoku == 4
        assert worker._table.oya == 1  # noqa: SLF001
        assert worker._table.bakaze == "E"  # noqa: SLF001
        # 下家是莊家,自己坐他的上家 —— 北家。畫面左下角印的也是北。
        assert "N" in worker._table.players[0].yakuhai  # noqa: SLF001
        assert "S" in worker._table.players[2].yakuhai  # noqa: SLF001

    def test_joining_mid_kyoku_gives_no_event_stream(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """一上來就滿河,誰先打的已經無從得知 —— 這一局不出事件流,不要亂排。

        放銃分析照常:河是成員問題,不是順序問題。
        """
        worker, _ = _worker(templates)
        for _ in range(3):
            worker.offer_frame(image, calibration)
            worker.tick()
        assert worker.steps == []
        assert any(player.river for player in worker._table.players)  # noqa: SLF001

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


class TestItSaysWhatItIsWaitingFor:
    """**沒有一條路可以安靜地卡住。**

    使用者回報:只打開放銃分析的話畫面一直停在「等待對局…」,而且完全
    不出聲 —— 拿不到畫面就直接 return。撥了開關什麼都沒發生是最難查的
    一種壞法,因為它跟「還沒輪到你」長得一模一樣。
    """

    def test_without_a_frame_it_says_so(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker.tick()
        assert "畫面辨識" in worker.status.read()

    def test_without_a_hand_it_says_so(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        worker, _ = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.tick()
        assert "手牌" in worker.status.read()

    def test_with_both_it_says_nothing(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """好好運作的時候不要佔版面 —— 狀態列是給異常用的。"""
        worker, _ = _worker(templates)
        worker.offer_frame(image, calibration)
        worker.offer_hand(["1m", "9p"])
        worker.tick()
        assert not worker.status.read()
