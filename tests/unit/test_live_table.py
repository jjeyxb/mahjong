"""畫面 → 整桌狀態 → 危險度,完全不靠封包。

這一組的重點是**整條鏈真的接得起來**:素材圖進去,危險度出來。零件各自的
正確率由 test_vision_river / test_vision_dora / test_tracker_* 守著,
這裡守的是接線。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from mia.calibration.table import Calibration
from mia.config.loader import load_config
from mia.live.bus import Dangers, UpdateBus
from mia.live.table import TableLink, TableWorker, _MeldLook, _PendingFaces
from mia.mjai.events import Chi, Dahai
from mia.tracker import Action, Move, OwnMove, Step, TurnTracker
from mia.tracker.stream import MeldPick, MeldShape
from mia.utils.geometry import Rect, Size
from mia.vision.hud import Hud
from mia.vision.tiles.classify import Match, TemplateSet
from mia.vision.tiles.meld_fit import MeldPicture, Pick
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


class TestMeldPictureRetry:
    def _picture(self, value: int) -> MeldPicture:
        image = np.full((1440, 2560, 3), value, dtype=np.uint8)
        return MeldPicture(image, Rect(0, 0, 2560, 1440), "toimen", end=6.0)

    def test_an_obscured_picture_can_be_replaced_after_abstaining(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        first: Future[Pick | None] = Future()
        second: Future[Pick | None] = Future()
        pool.submit.side_effect = [first, second]
        picture, clear = self._picture(140), self._picture(200)
        look = _MeldLook(pool, picture, templates)
        shapes = (MeldShape(("8p", "7p", "9p"), 0),)
        assert look.pick(shapes) is None
        assert look.pending
        assert not look.needs_picture
        # 還沒比完時不換掉它,否則每一輪都會重新排工作、永遠等不到結果。
        look.refresh(clear)
        first.set_result(None)
        assert not look.pending
        assert look.needs_picture
        look.refresh(picture)
        assert look.pick(shapes) is None
        assert pool.submit.call_count == 1
        look.refresh(clear)
        assert look.pick(shapes) is None
        assert pool.submit.call_args.args[1] is clear
        second.set_result(Pick(0, (False, False, False), 0.3))
        assert look.pick(shapes) == MeldPick(0, (False, False, False))
        assert not look.needs_picture
        # 已看清楚的結果不被下一次遮擋蓋掉。
        look.refresh(picture)
        assert look.pick(shapes) == MeldPick(0, (False, False, False))
        assert pool.submit.call_count == 2

    def test_a_failed_picture_is_not_still_pending(self, templates: TemplateSet) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        future: Future[Pick | None] = Future()
        pool.submit.return_value = future
        look = _MeldLook(pool, self._picture(140), templates)
        shapes = (MeldShape(("7p", "5p", "6p"), 0),)
        assert not look.pending
        assert look.pick(shapes) is None
        assert look.pending
        future.set_exception(ValueError("unreadable"))
        assert not look.pending
        assert look.pick(shapes) is None

    def test_abstaining_waits_until_the_face_scan_is_adopted(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        future: Future[Pick | None] = Future()
        pool.submit.return_value = future
        look = _MeldLook(pool, self._picture(140), templates, wait_faces=True)
        # 字牌碰等不需要問畫面的組,不能因為牌面仍在掃描而增加等候。
        assert not look.pending
        shapes = (MeldShape(("7p", "5p", "6p"), 0),)
        assert look.pick(shapes) is None
        future.set_result(None)
        assert look.pick(shapes) is None
        assert look.pending
        look.finish_faces()
        assert not look.pending

    @pytest.mark.parametrize("failed", [False, True])
    def test_face_completion_releases_the_fallback_after_adopting_reds(
        self, templates: TemplateSet, failed: bool
    ) -> None:
        worker, _ = _worker(templates)
        pool = Mock(spec=ThreadPoolExecutor)
        picked: Future[Pick | None] = Future()
        picked.set_result(None)
        pool.submit.return_value = picked
        look = _MeldLook(pool, self._picture(140), templates, wait_faces=True)
        look.pick((MeldShape(("7p", "5p", "6p"), 0),))
        worker._meld_looks[(2, 0)] = look  # noqa: SLF001
        worker._table.players[2].melds.append(("?", "?", "?"))  # noqa: SLF001
        faces: Future[MeldFaces] = Future()
        if failed:
            faces.set_exception(ValueError("unreadable"))
        else:
            faces.set_result(MeldFaces(frozenset(("0p", "6p", "7p")),
                                       red_suits=frozenset(("p",))))
        worker._pending_faces = [_PendingFaces(worker._table, 2, 0, faces)]  # noqa: SLF001
        assert look.pending
        worker._collect_meld_faces()  # noqa: SLF001
        assert not look.pending
        assert not worker._pending_faces  # noqa: SLF001
        if not failed:
            assert worker._meld_reds[(2, 0)] == frozenset(("p",))  # noqa: SLF001

    def test_new_kyoku_cancels_queued_candidate_jobs(self, templates: TemplateSet) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        job: Future[Pick | None] = Future()
        pool.submit.return_value = job
        look = _MeldLook(pool, self._picture(140), templates)
        look.pick((MeldShape(("8p", "7p", "9p"), 0),))
        worker, _ = _worker(templates)
        worker._meld_looks[(1, 0)] = look  # noqa: SLF001
        worker._new_kyoku()  # noqa: SLF001
        assert job.cancelled()
        assert worker._meld_looks == {}  # noqa: SLF001

    def test_a_weak_wrong_chi_can_be_corrected_by_a_clearer_picture(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        first: Future[Pick | None] = Future()
        second: Future[Pick | None] = Future()
        pool.submit.side_effect = [first, second]
        look = _MeldLook(pool, self._picture(140), templates)
        shapes = (
            MeldShape(("6s", "5s", "7s"), 0),
            MeldShape(("6s", "7s", "8s"), 0),
        )
        look.pick(shapes)
        first.set_result(Pick(1, (False,) * 3, 0.1, quality=0.52))
        assert look.pick(shapes) == MeldPick(1, (False,) * 3)
        assert look.needs_picture
        look.refresh(self._picture(200))
        # 新圖還沒比完,保留先前答案,不退回張數猜測。
        assert look.pick(shapes) == MeldPick(1, (False,) * 3)
        second.set_result(Pick(0, (False,) * 3, 0.3, quality=0.74))
        assert look.pick(shapes) == MeldPick(0, (False,) * 3)
        assert not look.needs_picture

    def test_rechecks_keep_the_best_picture_and_stop_after_two_tries(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        jobs: list[Future[Pick | None]] = [Future() for _ in range(3)]
        pool.submit.side_effect = jobs
        look = _MeldLook(pool, self._picture(140), templates)
        shapes = (MeldShape(("6s", "5s", "7s"), 0),)
        look.pick(shapes)
        jobs[0].set_result(Pick(0, (False,) * 3, 0.2, quality=0.6))
        look.refresh(self._picture(180))
        assert look.pick(shapes) == MeldPick(0, (False,) * 3)
        jobs[1].set_result(None)
        assert look.pick(shapes) == MeldPick(0, (False,) * 3)
        assert look.needs_picture
        look.refresh(self._picture(220))
        assert look.pick(shapes) == MeldPick(0, (False,) * 3)
        jobs[2].set_result(Pick(0, (False, True, False), 0.25, quality=0.61))
        # 分數只差一點、赤五判斷卻變了,不改寫原本答案。
        assert look.pick(shapes) == MeldPick(0, (False,) * 3)
        assert not look.needs_picture
        look.refresh(self._picture(100))
        assert pool.submit.call_count == 3

    def test_an_old_candidate_answer_does_not_stop_retrying_new_candidates(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        first: Future[Pick | None] = Future()
        second: Future[Pick | None] = Future()
        pool.submit.side_effect = [first, second]
        look = _MeldLook(pool, self._picture(140), templates)
        look.pick((MeldShape(("7s", "5s", "6s"), 0),))
        first.set_result(Pick(0, (False,) * 3, 0.2, quality=0.8))
        assert not look.needs_picture
        look.pick((MeldShape(("6s", "5s", "7s"), 0),))
        second.set_result(None)
        assert look.needs_picture

    def test_a_completed_fit_wakes_the_table_without_waiting_for_the_interval(
        self, templates: TemplateSet
    ) -> None:
        pool = Mock(spec=ThreadPoolExecutor)
        job: Future[Pick | None] = Future()
        pool.submit.return_value = job
        worker, _ = _worker(templates)
        look = _MeldLook(pool, self._picture(140), templates, worker._on_meld_ready)  # noqa: SLF001
        look.pick((MeldShape(("8p", "7p", "9p"), 0),))
        assert not worker._faces_ready.is_set()  # noqa: SLF001
        job.set_result(Pick(0, (False, False, False), 0.3))
        assert worker._faces_ready.is_set()  # noqa: SLF001
        assert worker._wake.is_set()  # noqa: SLF001

    def test_background_completion_publishes_without_rereading_the_rivers(
        self, image: np.ndarray, calibration: Calibration, templates: TemplateSet,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        worker, _ = _worker(templates)
        publish, stream = Mock(), Mock()
        monkeypatch.setattr(worker, "_publish", publish)
        monkeypatch.setattr(worker, "_on_stream", stream)
        monkeypatch.setattr(worker, "_round_due", lambda *_args: False)
        worker._faces_ready.set()  # noqa: SLF001
        worker.offer_frame(image, calibration)
        worker.tick()
        assert worker.reads == 0
        assert not worker._faces_ready.is_set()  # noqa: SLF001
        publish.assert_called_once()
        stream.assert_called_once()


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
        assert worker._reach_marks[1] == 1  # noqa: SLF001

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

    def test_after_the_declared_tile_is_called_the_next_one_lies_down(
        self, templates: TemplateSet
    ) -> None:
        """宣言牌(第 3 張)被鳴走了,規則上他的下一張改成橫放(真人東一局上家)。"""
        worker = self._worker(templates, ["1m", "2m", "?"], [0, 1, 3])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] == 2  # noqa: SLF001
        assert worker._known[3] == []  # noqa: SLF001

    def test_a_called_declaration_waits_for_the_next_tile(self, templates: TemplateSet) -> None:
        worker = self._worker(templates, ["1m", "2m"], [0, 1])
        worker._reach_marks[3] = 2  # noqa: SLF001
        worker._mark_declared()  # noqa: SLF001
        assert worker._sideways[3] is None  # noqa: SLF001

    def test_a_next_tile_read_upright_is_not_lying_down(self, templates: TemplateSet) -> None:
        worker = self._worker(templates, ["1m", "2m", "3m"], [0, 1, 3])
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

    def test_an_unread_wall_holds_the_batch_for_the_next_reading(
        self, templates: TemplateSet
    ) -> None:
        """牌河那一層已經發過這批牌,丟掉就永遠漏了。等下一輪讀到余牌再判。"""
        worker = self._worker(templates)
        assert worker._order(1.0, self.BATCH, (), None, ()) == []  # noqa: SLF001
        assert worker._order(1.5, (), (), None, ()) == []  # noqa: SLF001
        steps = worker._order(2.0, (), (), 35, ())  # noqa: SLF001
        assert sum(step.action is Action.DAHAI for step in steps) == 5
        assert worker._joined == ([], [], [])  # noqa: SLF001

    def test_a_batch_the_wall_never_confirms_is_dropped(self, templates: TemplateSet) -> None:
        worker = self._worker(templates)
        assert worker._order(1.0, self.BATCH, (), None, ()) == []  # noqa: SLF001
        assert worker._order(4.5, (), (), None, ()) == []  # noqa: SLF001
        assert worker._joined == ([], [], [])  # noqa: SLF001
        assert worker._order(5.0, (), (), 35, ()) == []  # noqa: SLF001


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

    def test_an_unknown_tile_is_retried_even_when_the_first_picture_was_still(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._began = 1.0  # noqa: SLF001
        worker._still[1] = {0}  # noqa: SLF001
        read = Mock(side_effect=[("?", "P"), ("?", "P"), ("6s", "6s")])
        monkeypatch.setattr(worker, "_read_cell", read)
        river = TestLanding._river(0)  # noqa: SLF001
        flat = np.zeros((200, 200, 3), np.uint8)
        worker._read_seat(1, flat, river)  # noqa: SLF001
        assert worker._readings[(1, 0)] == (None, "P")  # noqa: SLF001
        worker._read_seat(1, flat, river)  # noqa: SLF001
        assert worker._unsteady[1] == {0}  # noqa: SLF001
        worker._read_seat(1, flat, river)  # noqa: SLF001
        assert worker._readings[(1, 0)] == ("6s", "6s")  # noqa: SLF001
        assert worker._rivers.rivers[1] == ["6s"]  # noqa: SLF001
        assert not worker._unsteady[1]  # noqa: SLF001

    def test_a_weak_revisit_does_not_erase_a_confirmed_tile(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._known[1] = ["6s"]  # noqa: SLF001
        worker._readings[(1, 0)] = ("6s", "6s")  # noqa: SLF001
        worker._unsteady[1] = {0}  # noqa: SLF001
        monkeypatch.setattr(worker, "_read_cell", lambda *_: ("?", "P"))
        worker._revisit(1, np.zeros((200, 200, 3), np.uint8), TestLanding._river(0),  # noqa: SLF001
                        worker._known[1])  # noqa: SLF001
        assert worker._known[1] == ["6s"]  # noqa: SLF001
        assert worker._readings[(1, 0)] == ("6s", "6s")  # noqa: SLF001

    def test_an_unreadable_cell_does_not_retry_forever(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._known[1] = ["?"]  # noqa: SLF001
        worker._unsteady[1] = {0}  # noqa: SLF001
        worker._uncertain_until[1][0] = 10.0  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 11.0)
        read = Mock()
        monkeypatch.setattr(worker, "_read_cell", read)
        worker._revisit(1, np.zeros((200, 200, 3), np.uint8), TestLanding._river(0),  # noqa: SLF001
                        worker._known[1])  # noqa: SLF001
        read.assert_not_called()
        assert not worker._unsteady[1]  # noqa: SLF001

    def test_recorded_flying_six_sou_is_corrected_when_the_ink_appears(
        self, templates: TemplateSet
    ) -> None:
        from mia.vision.tiles.river import read_river

        worker, _ = _worker(templates)
        worker._river_templates = TemplateSet.load_river()  # noqa: SLF001
        worker._began = 1.0  # noqa: SLF001
        known = ["S", "S", "9p", "N", "7s", "1m", "C", "8m", "8s", "W", "2p", "F", "9p"]
        worker._known[3] = list(known)  # noqa: SLF001
        worker._rivers.observe(3, known)  # noqa: SLF001
        worker._table.players[3].river = list(known)  # noqa: SLF001
        worker._still[3] = {13}  # noqa: SLF001
        flying = cv2.imread(str(FIXTURES / "river_flying_6s_kamicha.png"))
        clear = cv2.imread(str(FIXTURES / "river_landed_6s_kamicha.png"))
        assert flying is not None and clear is not None
        worker._read_seat(3, flying, read_river(flying))  # noqa: SLF001
        assert worker._readings[(3, 13)] == (None, "P")  # noqa: SLF001
        worker._read_seat(3, clear, read_river(clear))  # noqa: SLF001
        assert worker._readings[(3, 13)] == ("6s", "6s")  # noqa: SLF001


class TestTsumogiriAtTheDiscard:
    @pytest.mark.parametrize(("age", "expected"), [(0.1, False), (0.8, False), (2.0, True)])
    def test_hand_evidence_is_limited_to_the_discard_window(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch, age: float, expected: bool
    ) -> None:
        worker, _ = _worker(templates)
        worker._kyoku.empty = False  # noqa: SLF001
        watcher = Mock()
        watcher.tsumogiri.return_value = True
        worker._tedashi = watcher  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 1.0)
        worker._emit(2, ["4p"])  # noqa: SLF001
        watcher.tsumogiri.return_value = False
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 1.0 + age)
        monkeypatch.setattr("mia.live.table.read_backs", Mock())
        worker._watch_backs(np.zeros((20, 20, 3), np.uint8), Rect(0, 0, 20, 20))  # noqa: SLF001
        assert worker._seen_tsumogiri[(2, 0)] == (1.0, expected)  # noqa: SLF001

    def test_published_verdict_does_not_change_with_a_later_hand(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        worker._kyoku.empty = False  # noqa: SLF001
        worker._seen_tsumogiri[(2, 0)] = (1.0, True)  # noqa: SLF001
        worker._tsumogiri[(2, 0)] = True  # noqa: SLF001
        watcher = Mock()
        watcher.tsumogiri.return_value = False
        worker._tedashi = watcher  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 1.1)
        monkeypatch.setattr("mia.live.table.read_backs", Mock())
        worker._watch_backs(np.zeros((20, 20, 3), np.uint8), Rect(0, 0, 20, 20))  # noqa: SLF001
        assert worker._seen_tsumogiri[(2, 0)] == (1.0, True)  # noqa: SLF001

    def test_ordering_later_uses_the_shape_seen_with_the_discard(
        self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, _ = _worker(templates)
        watcher = Mock()
        watcher.tsumogiri.return_value = False
        worker._tedashi = watcher  # noqa: SLF001
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 1.0)
        worker._emit(2, ["4p"])  # noqa: SLF001
        watcher.tsumogiri.return_value = True
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: 7.0)
        worker._decide_tsumogiri([Step(Action.DAHAI, 2, "4p", serial=0)])  # noqa: SLF001
        assert worker._tsumogiri[(2, 0)] is False  # noqa: SLF001

    def test_a_new_discard_cannot_reuse_the_previous_discard_verdict(
        self, templates: TemplateSet
    ) -> None:
        worker, _ = _worker(templates)
        watcher = Mock()
        watcher.tsumogiri.side_effect = [False, True]
        worker._tedashi = watcher  # noqa: SLF001
        worker._emit(2, ["4p"])  # noqa: SLF001
        worker._emit(2, ["4p", "5p"])  # noqa: SLF001
        worker._decide_tsumogiri([Step(Action.DAHAI, 2, "4p", serial=0),  # noqa: SLF001
                                Step(Action.DAHAI, 2, "5p", serial=1)])
        assert worker._tsumogiri == {(2, 0): False, (2, 1): True}  # noqa: SLF001

    def test_new_kyoku_drops_the_saved_shapes(self, templates: TemplateSet) -> None:
        worker, _ = _worker(templates)
        worker._seen_tsumogiri[(2, 0)] = (1.0, False)  # noqa: SLF001
        worker._new_kyoku()  # noqa: SLF001
        assert not worker._seen_tsumogiri  # noqa: SLF001


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


class TestOwnCombinedCall:
    @pytest.mark.parametrize(
        "position,pai,guess", [(0, "8s", None), (0, None, "8s"), (1, "8s", None)]
    )
    def test_only_a_confident_own_discard_corrects_the_chi(
        self, templates: TemplateSet, position: int, pai: str | None, guess: str | None
    ) -> None:
        worker, _ = _worker(templates)
        worker._round.settled = Hud(  # noqa: SLF001
            bakaze="E", kyoku=1, honba=0, kyotaku=0, scores=(25000,) * 4, dealer=3
        )
        worker._dora_marks = [(0, "9m")]  # noqa: SLF001
        hand = ["5s", "7s", "8s", "1m", "2m", "3m", "4p", "5p", "6p", "E", "E", "S", "S"]
        worker._own.observe(hand)  # noqa: SLF001
        worker._own.observe(hand[3:])  # noqa: SLF001
        worker.steps = [
            Step(Action.TSUMO, 3),
            Step(Action.DAHAI, 3, "6s"),
            Step(Action.CALL, 0, target=3),
            Step(Action.DAHAI, 0, "5s", serial=0),
        ]
        worker._readings[(position, 0)] = (pai, guess)  # noqa: SLF001
        stream = worker.stream()
        corrected = position == 0 and pai is not None
        consumed = ["5s", "7s"] if corrected else ["7s", "8s"]
        discarded = "8s" if corrected else "5s"
        assert not stream.stalled
        assert stream.events[-2:] == (
            Chi(actor=0, target=3, pai="6s", consumed=consumed),
            Dahai(actor=0, pai=discarded, tsumogiri=False),
        )
        assert worker.own_moves[0].tiles == tuple(consumed)
        worker.stop()


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


class TestHoldBehindALeaningTile:
    """宣言牌橫放、格線還沒挪:它的後半截落在下一格,不能當成一張新的捨牌。

    錄影 selfplay-20261102-043327 東3局2本場:上家宣言牌 8p 一落下,後半截就被讀成一張讀不出來的
    新牌送出去,之後猜成白。點數 2 秒後才變,那張假牌留在事件流裡。
    """

    @pytest.fixture
    def worker(self, templates: TemplateSet, monkeypatch: pytest.MonkeyPatch) -> TableWorker:
        worker, _ = _worker(templates)
        worker._leaning[3].add(13)  # noqa: SLF001
        worker._leaning_at[3][13] = 100.0  # noqa: SLF001
        self.now = 100.0
        monkeypatch.setattr("mia.live.table.time.monotonic", lambda: self.now)
        return worker

    def _held(self, worker: TableWorker, pai: str = "?") -> bool:
        return worker._behind_leaning(3, 14, pai)  # noqa: SLF001

    def test_the_cell_after_a_leaning_tile_waits(self, worker: TableWorker) -> None:
        assert self._held(worker)
        self.now += 1.0
        assert self._held(worker)

    def test_it_gives_up_after_the_hold(self, worker: TableWorker) -> None:
        for _ in range(7):  # 0~6 秒
            assert self._held(worker)
            self.now += 1.0
        assert not self._held(worker)  # 滿 7 秒

    def test_a_break_restarts_the_wait(self, worker: TableWorker) -> None:
        self._held(worker)
        self.now += 1.0
        self._held(worker)
        worker._leaning_at[3][13] = self.now + 3.0  # noqa: SLF001 - 宣言牌又被重讀成疑似
        self.now += 3.0  # 那半截不見了一陣子,之後又冒出來
        assert self._held(worker)
        self.now += 1.0
        assert self._held(worker)

    def test_a_readable_tile_or_a_known_sideways_cell_does_not_wait(
        self, worker: TableWorker
    ) -> None:
        assert not self._held(worker, "6p")
        worker._sideways[3] = 13  # noqa: SLF001
        assert not self._held(worker)

    def test_a_tile_thrown_long_after_the_leaning_one_does_not_wait(
        self, worker: TableWorker
    ) -> None:
        """錄影 155238 南1局6本場:中被當成疑似,5 秒後真的宣言牌落在後面。"""
        self.now += 5.0
        assert not self._held(worker)

    def test_a_leaning_tile_read_again_this_round_still_counts(self, worker: TableWorker) -> None:
        """錄影 043327 東3局2本場:宣言牌那格閃過一下被剪掉,下一張打出來時兩格一起讀回來。"""
        self.now += 7.0
        assert worker._behind_leaning(3, 14, "?", together=True)  # noqa: SLF001

    def test_no_leaning_tile_no_wait(self, worker: TableWorker) -> None:
        worker._leaning[3].clear()  # noqa: SLF001
        assert not self._held(worker)
