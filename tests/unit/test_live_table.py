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
from mia.mjai.events import Dahai
from mia.tracker import Action, Move, OwnMove, Step, TurnTracker
from mia.utils.geometry import Rect, Size
from mia.vision.hud import Hud
from mia.vision.tiles.classify import Match, TemplateSet
from mia.vision.tiles.melds import MeldFaces
from mia.vision.tiles.river import River

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

    def test_joining_mid_kyoku_still_gives_an_event_stream(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet
    ) -> None:
        """一上來就滿河。家與家之間的先後照輪替補,這一局才有建議。"""
        worker, _ = _worker(templates)
        for _ in range(3):
            worker.offer_frame(image, calibration)
            worker.tick()
        seen = sum(step.action is Action.DAHAI for step in worker.steps)
        assert seen == sum(worker._rivers.thrown(position) for position in range(4))  # noqa: SLF001
        assert seen > 4

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

    def test_a_waiting_hand_is_read_as_soon_as_a_frame_arrives(
        self, templates: TemplateSet, calibration: Calibration
    ) -> None:
        """手牌先到、畫面被上一輪拿走了:不等滿一秒。"""
        worker, _ = _worker(templates)
        worker.offer_hand(["1m"])
        worker.tick()
        worker._wake.clear()  # noqa: SLF001
        worker.offer_frame(np.zeros((90, 160, 3), dtype=np.uint8), calibration)
        assert worker._wake.is_set()  # noqa: SLF001

    def test_a_frame_alone_does_not_wake_it(
        self, templates: TemplateSet, calibration: Calibration
    ) -> None:
        worker, _ = _worker(templates)
        worker.offer_frame(np.zeros((90, 160, 3), dtype=np.uint8), calibration)
        assert not worker._wake.is_set()  # noqa: SLF001


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


class TestTheDeclaredTile:
    """點數說有人立直了,宣言牌就知道是哪一張。實測一場讀錯的牌有 9 張是宣言牌。"""

    @staticmethod
    def _worker(templates: TemplateSet, known: list[str], serials: list[int]) -> TableWorker:
        worker, _ = _worker(templates)
        worker._known[3] = list(known)  # noqa: SLF001 - 直接擺出要測的狀態
        worker._rivers.serials[3] = list(serials)  # noqa: SLF001
        return worker

    def test_the_marked_tile_becomes_the_sideways_one(self, templates: TemplateSet) -> None:
        """當列最後一張也算。橫放偵測對這種情況永遠說不知道。"""
        worker = self._worker(templates, ["1m", "2m", "?"], [0, 1, 2])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] == 2  # noqa: SLF001
        assert worker._known[3] == [], "下一輪用挪過的格線重認"  # noqa: SLF001

    def test_a_tile_read_upright_is_not_the_declared_one(self, templates: TemplateSet) -> None:
        """宣言牌直放從來認不出來。認得出來就是標記晚了一張。"""
        worker = self._worker(templates, ["1m", "2m", "3m"], [0, 1, 2])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] is None  # noqa: SLF001
        assert worker._known[3] == ["1m", "2m", "3m"]  # noqa: SLF001

    def test_the_mark_on_the_spill_over_goes_back_a_cell(self, templates: TemplateSet) -> None:
        """橫放的牌比一格寬,後面那格也讀得到牌,標記會落在那一格上。"""
        worker, _ = _worker(templates)
        worker._rivers.serials[1] = [0, 1, 2]  # noqa: SLF001
        worker._known[1] = ["1m", "?", "?"]  # noqa: SLF001
        worker._reach_marks[1] = 2  # noqa: SLF001
        worker._leaning[1] = {1}  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[1] == 1  # noqa: SLF001

    def test_a_called_tile_shifts_the_cell(self, templates: TemplateSet) -> None:
        """第 1 張被鳴走了,第 3 張宣言牌在第 2 格。"""
        worker = self._worker(templates, ["1m", "?"], [0, 2])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] == 1  # noqa: SLF001

    def test_it_waits_when_the_river_and_the_count_disagree(self, templates: TemplateSet) -> None:
        worker = self._worker(templates, ["1m"], [0, 1, 2])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] is None  # noqa: SLF001


class _AnyRoi:
    """切什麼都回一小塊。"""

    @staticmethod
    def crop(image: np.ndarray, _name: str) -> np.ndarray:
        return image


class TestTheChiChooser:
    """選擇框蓋在自己牌河的第三列,框裡的牌讀起來是河裡的牌。實測整局的事件流因此停了。"""

    @staticmethod
    def _worker(templates: TemplateSet, discards: int) -> TableWorker:
        from mia.tracker.hand import OwnMove

        worker, _ = _worker(templates)
        worker._own.moves = [  # noqa: SLF001 - 直接擺出要測的狀態
            OwnMove(Move.DISCARD, ("1m",)) for _ in range(discards)
        ]
        return worker

    def test_four_more_than_the_hand_explains_is_the_box(self, templates: TemplateSet) -> None:
        worker = self._worker(templates, 13)
        assert worker._own_river_covered(17)  # noqa: SLF001

    def test_the_hand_being_a_step_behind_is_normal(self, templates: TemplateSet) -> None:
        worker = self._worker(templates, 13)
        assert not worker._own_river_covered(14)  # noqa: SLF001
        assert not worker._own_river_covered(15)  # noqa: SLF001

    def test_without_any_moves_it_says_nothing(self, templates: TemplateSet) -> None:
        """中途接上時手牌什麼都不知道,不能拿它擋。"""
        worker = self._worker(templates, 0)
        assert not worker._own_river_covered(10)  # noqa: SLF001


class TestTurnClock:
    def test_called_directly_each_read_is_a_round(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker.reads = 7
        assert worker._turn_clock() == 7  # noqa: SLF001

    def test_running_it_counts_seconds_not_reads(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """手牌一變就提早讀,一秒讀好幾次,數次數的話等副露的時間只剩不到一秒。"""
        worker, _ = _worker(templates)
        worker.start()
        worker.stop()
        worker.join(timeout=5)
        assert not worker.is_alive()
        began = worker._began  # noqa: SLF001
        assert began is not None
        worker.reads = 100
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: began + 2.5)
        assert worker._turn_clock() == pytest.approx(2.5)  # noqa: SLF001


class TestLanding:
    """牌河每格只認一次,所以要等牌停穩才認。"""

    @staticmethod
    def _river(*cells: int) -> River:
        rect = Rect(10, 10, 90, 120)
        occupied = tuple(index in cells for index in range(18))
        return River(cells=(rect,) * 18, occupied=occupied)

    @staticmethod
    def _flats(level: int = 200) -> list[np.ndarray]:
        return [np.full((200, 200, 3), level, np.uint8) for _ in range(4)]

    def _running(self, templates: TemplateSet) -> TableWorker:
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        return worker

    def test_the_first_sighting_is_not_enough(self, templates: TemplateSet) -> None:
        worker = self._running(templates)
        rivers = [self._river(0), *[self._river() for _ in range(3)]]
        assert not worker._watch_landing(self._flats(), rivers)  # noqa: SLF001

    def test_the_same_pixels_twice_have_landed(self, templates: TemplateSet) -> None:
        worker = self._running(templates)
        rivers = [self._river(0), *[self._river() for _ in range(3)]]
        worker._watch_landing(self._flats(), rivers)  # noqa: SLF001
        assert worker._watch_landing(self._flats(), rivers)  # noqa: SLF001
        assert worker._steady(0, 0)  # noqa: SLF001

    def test_moving_pixels_have_not(self, templates: TemplateSet) -> None:
        worker = self._running(templates)
        rivers = [self._river(0), *[self._river() for _ in range(3)]]
        worker._watch_landing(self._flats(200), rivers)  # noqa: SLF001
        assert not worker._watch_landing(self._flats(150), rivers)  # noqa: SLF001
        assert not worker._steady(0, 0)  # noqa: SLF001

    def test_cells_already_read_are_not_watched(self, templates: TemplateSet) -> None:
        worker = self._running(templates)
        worker._known[1] = ["1m"]  # noqa: SLF001
        rivers = [self._river(), self._river(0), self._river(), self._river()]
        worker._watch_landing(self._flats(), rivers)  # noqa: SLF001
        assert not worker._watch_landing(self._flats(), rivers)  # noqa: SLF001

    def test_a_cell_read_too_early_is_watched_until_it_lands(
        self, templates: TemplateSet
    ) -> None:
        worker = self._running(templates)
        worker._known[1] = ["1m"]  # noqa: SLF001
        worker._unsteady[1] = {0}  # noqa: SLF001
        rivers = [self._river(), self._river(0), self._river(), self._river()]
        worker._watch_landing(self._flats(), rivers)  # noqa: SLF001
        assert worker._watch_landing(self._flats(), rivers)  # noqa: SLF001

    def test_nothing_new_means_no_round_for_a_second(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._running(templates)
        rivers = [self._river() for _ in range(4)]
        worker._last_round = 100.0  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 100.5)
        assert not worker._round_due(self._flats(), rivers, urgent=False)  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 101.1)
        assert worker._round_due(self._flats(), rivers, urgent=False)  # noqa: SLF001

    def test_the_hand_changing_cannot_wait(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._running(templates)
        rivers = [self._river() for _ in range(4)]
        worker._last_round = 100.0  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 100.1)
        assert worker._round_due(self._flats(), rivers, urgent=True)  # noqa: SLF001

    def _moving(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch, answer: str, seat: int = 1
    ) -> tuple[TableWorker, bool]:
        worker = self._running(templates)
        monkeypatch.setattr(worker, "_read_cell", lambda *_: (answer, answer))
        rivers = [self._river(), self._river(), self._river(), self._river()]
        rivers[seat] = self._river(0)
        worker._watch_landing(self._flats(200), rivers)  # noqa: SLF001
        return worker, worker._watch_landing(self._flats(150), rivers)  # noqa: SLF001

    def test_a_clear_tile_of_another_seat_is_taken_at_once(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """途中會讀錯的幾乎都是白。1142 張先收,只錯 2 張,與等停穩一樣。"""
        worker, landed = self._moving(templates, monkeypatch, "3s")
        assert landed
        assert worker._clear[1]  # noqa: SLF001

    def test_a_white_tile_has_to_land(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, landed = self._moving(templates, monkeypatch, "P")
        assert not landed

    def test_an_unreadable_tile_has_to_land(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, landed = self._moving(templates, monkeypatch, "?")
        assert not landed

    def test_ones_own_tile_has_to_land(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """牌面手牌已經知道,而且飛得慢。"""
        _, landed = self._moving(templates, monkeypatch, "3s", seat=0)
        assert not landed

    def test_called_directly_everything_counts_as_landed(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        assert worker._steady(0, 3)  # noqa: SLF001


class TestABatchOfTilesAtTheEnd:
    """和了之後的結算畫面會一次讀出一排假牌。"""

    @staticmethod
    def _worker(templates: TemplateSet) -> TableWorker:
        worker, _ = _worker(templates)
        worker._turns = TurnTracker(oya=0)  # noqa: SLF001
        worker._turns.drawn = 30  # noqa: SLF001
        return worker

    BATCH = tuple((1, "7s", "7s", serial) for serial in range(5))

    def test_the_wall_has_to_move_too(self, templates: TemplateSet) -> None:
        worker = self._worker(templates)
        assert worker._order(1.0, self.BATCH, (), 40, ()) == []  # noqa: SLF001

    def test_an_unread_wall_proves_nothing(self, templates: TemplateSet) -> None:
        worker = self._worker(templates)
        assert worker._order(1.0, self.BATCH, (), None, ()) == []  # noqa: SLF001

    def test_tiles_the_wall_agrees_with_go_in(self, templates: TemplateSet) -> None:
        worker = self._worker(templates)
        steps = worker._order(1.0, self.BATCH, (), 35, ())  # noqa: SLF001
        assert sum(step.action is Action.DAHAI for step in steps) == 5


class TestRevisit:
    """輪到自己時還沒停穩的牌也要先讀,停穩之後讀到不一樣的要改掉。"""

    def test_the_new_answer_replaces_the_old_one(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        worker._rivers.observe(1, ["P"])  # noqa: SLF001
        worker._known[1] = ["P"]  # noqa: SLF001
        worker._readings[(1, 0)] = ("P", "P")  # noqa: SLF001
        worker._unsteady[1] = {0}  # noqa: SLF001
        worker._still[1] = {0}  # noqa: SLF001
        monkeypatch.setattr(worker, "_read_cell", lambda *_: ("3s", "3s"))
        river = TestLanding._river(0)  # noqa: SLF001
        worker._revisit(1, np.zeros((200, 200, 3), np.uint8), river, worker._known[1])  # noqa: SLF001
        assert worker._known[1] == ["3s"]  # noqa: SLF001
        assert worker._rivers.rivers[1] == ["3s"]  # noqa: SLF001
        assert worker._readings[(1, 0)] == ("3s", "3s")  # noqa: SLF001
        assert not worker._unsteady[1]  # noqa: SLF001

    def test_the_danger_side_is_corrected_too(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """讀在動畫中途的那張不改的話,會一直留著當現物。"""
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        worker._table.players[1].river = ["2m", "P"]  # noqa: SLF001
        worker._known[1] = ["2m", "P"]  # noqa: SLF001
        worker._unsteady[1] = {1}  # noqa: SLF001
        worker._still[1] = {1}  # noqa: SLF001
        monkeypatch.setattr(worker, "_read_cell", lambda *_: ("3s", "3s"))
        river = TestLanding._river(0, 1)  # noqa: SLF001
        worker._revisit(1, np.zeros((200, 200, 3), np.uint8), river, worker._known[1])  # noqa: SLF001
        assert worker._table.players[1].river == ["2m", "3s"]  # noqa: SLF001

    def test_it_waits_until_the_tile_has_landed(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        worker._known[1] = ["P"]  # noqa: SLF001
        worker._unsteady[1] = {0}  # noqa: SLF001
        monkeypatch.setattr(worker, "_read_cell", lambda *_: ("3s", "3s"))
        river = TestLanding._river(0)  # noqa: SLF001
        worker._revisit(1, np.zeros((200, 200, 3), np.uint8), river, worker._known[1])  # noqa: SLF001
        assert worker._known[1] == ["P"]  # noqa: SLF001
        assert worker._unsteady[1] == {0}  # noqa: SLF001


class TestTheDeclaredTileBeforeThePoints:
    """要等點數才知道誰立直,但牌面現在就看得出來。"""

    FUZZY = Match("3m", 0.45, "8m", 0.44)

    def _guess(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch, turned: Match
    ) -> tuple[str, str]:
        worker, _ = _worker(templates)
        monkeypatch.setattr("mia.live.table.classify_river_tile", lambda *_, **__: self.FUZZY)
        monkeypatch.setattr("mia.live.table.classify_sideways", lambda *_, **__: turned)
        flat = np.full((400, 600, 3), 200, np.uint8)
        return worker._read_cell(1, flat, TestLanding._river(0), 0)  # noqa: SLF001

    def test_a_turned_tile_that_fits_better_is_the_guess(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pai, guess = self._guess(templates, monkeypatch, Match("7p", 0.7, "8p", 0.7))
        assert (pai, guess) == ("?", "7p")

    def test_it_stays_a_guess(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """只當猜測,不算立直:假立直的代價太大。"""
        pai, _ = self._guess(templates, monkeypatch, Match("7p", 0.9, "8p", 0.9))
        assert pai == "?"

    def test_otherwise_the_upright_guess_stands(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pai, guess = self._guess(templates, monkeypatch, Match("7p", 0.4, "8p", 0.4))
        assert (pai, guess) == ("?", "3m")


class TestUrgentReadsOfAFlyingTile:
    """輪到自己時還在飛的牌只收看得清楚的:它常常馬上被鳴走,沒機會停穩重讀。"""

    def _read(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> list[str]:
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        worker._known[1] = ["1m"]  # noqa: SLF001
        monkeypatch.setattr(worker, "_read_cell", lambda *_: (answer, "E"))
        monkeypatch.setattr(worker, "_emit", lambda *_: None)
        river = TestLanding._river(0, 1)  # noqa: SLF001
        worker._read_seat(1, np.zeros((400, 600, 3), np.uint8), river, urgent=True)  # noqa: SLF001
        return worker._known[1]  # noqa: SLF001

    def test_a_guess_is_not_taken(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._read(templates, monkeypatch, "?") == ["1m"]

    def test_a_clear_tile_is(self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._read(templates, monkeypatch, "3s") == ["1m", "3s"]

    def test_white_still_needs_a_second_look(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._read(templates, monkeypatch, "P") == ["1m"]


class TestReadCell:
    def test_the_same_pixels_are_not_read_twice(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """河被蓋住或鳴牌之後重讀一家,一家 300 毫秒,那段時間自己的手牌在排隊。"""
        worker, _ = _worker(templates)
        calls: list[int] = []

        def counted(*_: object, **__: object) -> Match:
            calls.append(1)
            return Match("1m", 0.9, "2m", 0.1)

        monkeypatch.setattr("mia.live.table.classify_river_tile", counted)
        flat = np.full((400, 600, 3), 200, np.uint8)
        river = TestLanding._river(0)  # noqa: SLF001
        worker._read_cell(1, flat, river, 0)  # noqa: SLF001
        worker._read_cell(1, flat, river, 0)  # noqa: SLF001
        assert len(calls) == 1

    def test_a_tile_that_was_called_only_shortens_the_cache(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._known[1] = ["1m", "2m"]  # noqa: SLF001
        monkeypatch.setattr(worker, "_look_for_sideways", lambda *_: None)
        monkeypatch.setattr(worker, "_emit", lambda *_: None)
        river = TestLanding._river(0)  # noqa: SLF001
        worker._read_seat(1, np.zeros((400, 600, 3), np.uint8), river)  # noqa: SLF001
        assert worker._known[1] == ["1m"]  # noqa: SLF001


class TestAStreamThatStopped:
    """事件流停住了就用畫面重寫,不要停到局尾。"""

    @staticmethod
    def _stalled(templates: TemplateSet) -> TableWorker:
        worker, _ = _worker(templates)
        worker._round.settled = Hud(  # noqa: SLF001
            bakaze="E", kyoku=1, honba=0, kyotaku=0, scores=(25000,) * 4, dealer=0
        )
        worker._dora_marks = [(0, "9m")]  # noqa: SLF001
        tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "S"]
        worker._own.observe(tiles)  # noqa: SLF001
        worker._own.moves = [OwnMove(Move.DISCARD, ("1m",))]  # noqa: SLF001
        worker.steps = [Step(Action.TSUMO, 0), Step(Action.DAHAI, 0, "N", serial=0)]
        worker._seen_hand = (tiles, None)  # noqa: SLF001
        return worker

    def test_it_waits_a_moment_first(self, templates: TemplateSet) -> None:
        """手牌慢一步是常有的事。"""
        worker = self._stalled(templates)
        worker.reads = 1
        assert worker.stream().stalled
        worker.reads = 3
        assert worker.stream().stalled

    def test_then_the_hand_on_screen_wins(self, templates: TemplateSet) -> None:
        worker = self._stalled(templates)
        worker.reads = 1
        worker.stream()
        worker.reads = 5
        worker._seen_at = 4  # noqa: SLF001
        stream = worker.stream()
        assert not stream.stalled
        assert Dahai(actor=0, pai="N", tsumogiri=True) in stream.events

    def test_without_a_hand_to_go_by_it_stays(self, templates: TemplateSet) -> None:
        worker = self._stalled(templates)
        worker._seen_hand = None  # noqa: SLF001
        worker.reads = 1
        worker.stream()
        worker.reads = 5
        worker._seen_at = 4  # noqa: SLF001
        assert worker.stream().stalled

    def test_a_hand_from_before_it_stopped_is_no_evidence(self, templates: TemplateSet) -> None:
        """局尾手牌被結算畫面蓋住,河裡卻還有最後一張。"""
        worker = self._stalled(templates)
        worker._seen_at = 0  # noqa: SLF001
        worker.reads = 1
        worker.stream()
        worker.reads = 9
        assert worker.stream().stalled

    def test_the_clock_restarts_when_it_moves_again(self, templates: TemplateSet) -> None:
        worker = self._stalled(templates)
        worker.reads = 1
        worker.stream()
        assert worker._stalled_at == 1  # noqa: SLF001
        worker.steps = []
        worker._own.moves = []  # noqa: SLF001
        worker.stream()
        assert worker._stalled_at is None  # noqa: SLF001


class TestAFlashOverOnesOwnRiver:
    """立直的白光掃過自己的河會多讀出一張牌。照補的話憑空多一手,之後全部晚一輪。"""

    @staticmethod
    def _after_a_discard(templates: TemplateSet) -> TableWorker:
        worker, _ = _worker(templates)
        tiles = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "S", "S"]
        worker._own.observe(tiles)  # noqa: SLF001
        worker._own.observe([*tiles, "P"], "P")  # noqa: SLF001
        worker._own.observe(tiles)  # noqa: SLF001
        worker._rivers.observe(0, ["P"])  # noqa: SLF001
        worker._catch_up_own()  # noqa: SLF001
        worker._readings[(0, 1)] = ("C", "C")  # noqa: SLF001
        return worker

    def test_no_catching_up_before_the_turn_comes_round(self, templates: TemplateSet) -> None:
        worker = self._after_a_discard(templates)
        worker._rivers.observe(0, ["P", "C"])  # noqa: SLF001
        for _ in range(6):
            worker._catch_up_own()  # noqa: SLF001
        assert len(worker._own.moves) == 2  # noqa: SLF001

    def test_once_the_seat_before_has_played_it_is_real(self, templates: TemplateSet) -> None:
        worker = self._after_a_discard(templates)
        worker._rivers.observe(0, ["P", "C"])  # noqa: SLF001
        worker._rivers.observe(3, ["1s"])  # noqa: SLF001
        for _ in range(6):
            worker._catch_up_own()  # noqa: SLF001
        assert worker._own.moves[-1] == OwnMove(Move.DISCARD, ("C",), tsumogiri=True)  # noqa: SLF001

    def test_a_tile_that_vanished_with_nobody_calling_never_was(
        self, templates: TemplateSet
    ) -> None:
        worker = self._after_a_discard(templates)
        worker._rivers._thrown[0] = 2  # noqa: SLF001
        assert worker._own_thrown() == 1  # noqa: SLF001

    def test_a_tile_someone_called_was_real(self, templates: TemplateSet) -> None:
        worker = self._after_a_discard(templates)
        worker._rivers._thrown[0] = 2  # noqa: SLF001
        worker._melds.groups[1] = 1  # noqa: SLF001
        assert worker._own_thrown() == 2  # noqa: SLF001


class TestMeldSpacing:
    def test_melds_are_not_read_twice_in_a_moment(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """要連讀兩次一樣才採信,隔太近的話兩次都是動畫的過渡值。"""
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        seen: list[int] = []
        monkeypatch.setattr(worker._melds, "observe", lambda p, _: seen.append(p) or 0)  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 10.0)
        image = np.zeros((1440, 2560, 3), np.uint8)
        table = Rect(0, 0, 2560, 1440)
        worker._read_melds(image, table, [])  # noqa: SLF001
        first = len(seen)
        worker._read_melds(image, table, [])  # noqa: SLF001
        assert first and len(seen) == first
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 10.5)
        worker._read_melds(image, table, [])  # noqa: SLF001
        assert len(seen) == 2 * first


class TestOwnMovesGoFirst:
    """自己的牌飛得最慢。排序照輪替,自己沒排出來,下家、對家的牌就全部排在後面。"""

    DROP = OwnMove(Move.DISCARD, ("1m",))
    PON = OwnMove(Move.CALL, ("E", "E"))

    def test_the_hand_tells_the_order_at_once(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker._own.moves = [self.DROP]  # noqa: SLF001
        assert worker._own_first([], []) == ([(0, "1m", None, 0)], [])  # noqa: SLF001

    def test_the_river_does_not_tell_it_again(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker._own.moves = [self.DROP]  # noqa: SLF001
        worker._own_first([], [])  # noqa: SLF001
        assert worker._own_first([(0, "1m", "1m", 0)], []) == ([], [])  # noqa: SLF001

    def test_the_river_first_is_fine_too(self, templates: TemplateSet) -> None:
        """立直之後的摸切手牌跟不上,河先看到。"""
        worker, _ = _worker(templates)
        seen = (0, "1m", "1m", 0)
        assert worker._own_first([seen], []) == ([seen], [])  # noqa: SLF001
        worker._own.moves = [self.DROP]  # noqa: SLF001
        assert worker._own_first([], []) == ([], [])  # noqa: SLF001

    def test_other_seats_pass_straight_through(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        seen = (1, "2p", "2p", 0)
        assert worker._own_first([seen], [2]) == ([seen], [2])  # noqa: SLF001

    def test_a_call_goes_first_too(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker._own.moves = [self.PON]  # noqa: SLF001
        assert worker._own_first([], []) == ([], [0])  # noqa: SLF001
        assert worker._own_first([], [0]) == ([], [])  # noqa: SLF001


class TestDoraOfTheLastKyoku:
    """結算畫面上的寶牌面板還是上一局的。實測五次開局寶牌讀錯,五次都是上一局那張。"""

    @staticmethod
    def _reads(templates: TemplateSet, monkeypatch: pytest.MonkeyPatch, wall: int) -> bool:
        worker, _ = _worker(templates)
        worker._rois = _AnyRoi()  # noqa: SLF001
        worker._digits = object()  # noqa: SLF001
        calls: list[object] = []
        monkeypatch.setattr("mia.live.table.read_wall", lambda *_: wall)
        monkeypatch.setattr("mia.live.table.read_dora", lambda *_: calls.append(1) or [])
        worker._read_dora(np.zeros((10, 10, 3), np.uint8), Rect(0, 0, 10, 10))  # noqa: SLF001
        return bool(calls)

    def test_not_while_the_table_is_empty(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert not self._reads(templates, monkeypatch, 12)

    def test_from_the_moment_the_wall_says_it_began(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """自己是莊家時,打第一張之前就要知道寶牌。"""
        assert self._reads(templates, monkeypatch, 70)
