"""擷取執行緒:畫面 → 手牌。

用真實截圖貼進一張合成的整幀影像,再讓 worker 走完整條路(校正 → ROI →
read_hand → classify)。手牌本身是真的,只有「它在畫面上的位置」是安排的
—— 那正好是這一層要負責的部分。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.calibration.manual import HandRoiChoice
from mia.capture.base import CaptureBackend, CaptureFailedError, Frame, WindowInfo
from mia.config.models import AppConfig, CalibrationConfig, CaptureConfig, RoiConfig
from mia.live import vision
from mia.live.bus import CvHand, UpdateBus
from mia.live.vision import VisionWorker
from mia.platforms import STEAM
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import Match, TemplateSet

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 手牌截圖要貼在合成整幀影像的哪裡(像素)。數字本身沒有意義,
#: 重點是設定裡的正規化 ROI 必須指回同一塊。
PASTE = Rect(300, 1100, 0, 0)

#: 整幀影像的尺寸。取真實錄影的解析度(2560x1440),因為手牌 fixture 是從
#: 那個解析度裁出來的 1813x207 —— 塞進比它小的畫布會直接放不下。
FRAME_SIZE = (2560, 1440)  # (寬, 高)


def _hand_image(name: str = "hand_13_full.png") -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"讀不到 fixture {name}"
    return image


def _frame_with_hand(window: WindowInfo, hand: np.ndarray) -> tuple[Frame, RoiConfig]:
    """造一張整幀影像,把手牌貼在 :data:`PASTE`,並回傳指回那裡的 ROI 設定。"""
    width, height = FRAME_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (40, 90, 40)  # 桌布綠,不是純黑 —— 純黑會讓校正把整張剝掉
    hand_h, hand_w = hand.shape[:2]
    assert PASTE.x + hand_w <= width and PASTE.y + hand_h <= height
    canvas[PASTE.y : PASTE.y + hand_h, PASTE.x : PASTE.x + hand_w] = hand

    roi = RoiConfig(
        own_hand=(PASTE.x / width, PASTE.y / height, hand_w / width, hand_h / height)
    )
    return Frame(canvas, window, scale=1.0), roi


def _config(roi: RoiConfig, *, fps: float = 120.0) -> AppConfig:
    """校正走手動指定的逃生門 —— 這裡要測的不是邊框剝除,而且手動指定會立刻鎖定。"""
    return AppConfig(
        capture=CaptureConfig(target_fps=fps),
        calibration=CalibrationConfig(manual_table_rect=(0.0, 0.0, 1.0, 1.0)),
        roi=roi,
    )


class FakeBackend(CaptureBackend):
    """吐預先排好的幀。``None`` 表示這一次擷取失敗。"""

    name = "fake"

    def __init__(self, window: WindowInfo, frames: list[Frame | None]) -> None:
        self._window = window
        self._frames = frames
        self.calls = 0
        self.closed = False

    @staticmethod
    def is_available() -> bool:
        return True

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG002
        return [self._window]

    def capture(self, window: WindowInfo) -> Frame:  # noqa: ARG002
        self.calls += 1
        index = min(self.calls - 1, len(self._frames) - 1)
        frame = self._frames[index]
        if frame is None:
            raise CaptureFailedError("測試刻意讓這一次失敗")
        return frame

    def close(self) -> None:
        self.closed = True


class EmptyBackend(CaptureBackend):
    """一個視窗都沒有。"""

    name = "empty"

    @staticmethod
    def is_available() -> bool:
        return True

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG002
        return []

    def capture(self, window: WindowInfo) -> Frame:  # noqa: ARG002
        raise CaptureFailedError("沒有視窗")


def _run_until(worker: VisionWorker, predicate, *, timeout: float = 5.0) -> None:
    worker.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    worker.stop()
    worker.join(2)


class _Sink:
    def __init__(self) -> None:
        self.hands: list[list[str]] = []

    def offer_frame(self, image: np.ndarray, calibration: object) -> None:  # noqa: ARG002
        return

    def offer_hand(self, tiles: list[str], drawn: str | None = None) -> None:  # noqa: ARG002
        self.hands.append(list(tiles))


class TestWhatTheTableGets:
    TILE = Match(label="1m", score=0.9, runner_up="2m", runner_up_score=0.1, brightness=1.0)
    FUZZY = Match(label="5s", score=0.3, runner_up="2m", runner_up_score=0.29, brightness=1.0)

    def _offered(
        self, window: WindowInfo, drawn: Match, monkeypatch: pytest.MonkeyPatch
    ) -> list[list[str]]:
        hand = _hand_image("hand_14_with_draw.png")
        frame, roi = _frame_with_hand(window, hand)
        sink = _Sink()
        worker = VisionWorker(
            UpdateBus(),
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
            sink=sink,
        )
        worker._templates = TemplateSet.load()  # noqa: SLF001
        monkeypatch.setattr(
            vision, "classify_hand", lambda *_, **__: ((self.TILE,) * 13, drawn)
        )
        worker._read(hand, 0)  # noqa: SLF001
        return sink.hands

    def test_a_fuzzy_drawn_tile_is_not_a_draw(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """打牌的特效飄過摸牌那格,實測被讀成一張低信心的牌。"""
        assert self._offered(window, self.FUZZY, monkeypatch) == []

    def test_a_clear_drawn_tile_goes_through(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert len(self._offered(window, self.TILE, monkeypatch)) == 1

    def _doubtful_worker(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[VisionWorker, _Sink]:
        """手牌裡有一張沒把握,摸牌那張有把握。"""
        hand = _hand_image("hand_14_with_draw.png")
        frame, roi = _frame_with_hand(window, hand)
        sink = _Sink()
        worker = VisionWorker(
            UpdateBus(),
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
            sink=sink,
        )
        worker._templates = TemplateSet.load()  # noqa: SLF001
        tiles = (self.FUZZY, *(self.TILE,) * 12)
        monkeypatch.setattr(vision, "classify_hand", lambda *_, **__: (tiles, self.TILE))
        worker._read(hand, 0)  # noqa: SLF001
        return worker, sink

    def test_a_doubtful_hand_waits_for_the_screen_to_stay_still(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, sink = self._doubtful_worker(window, monkeypatch)
        assert sink.hands == []
        for _ in range(vision._STILL_FRAMES - 1):  # noqa: SLF001
            worker._offer_doubtful()  # noqa: SLF001
        assert sink.hands == []
        worker._offer_doubtful()  # noqa: SLF001
        assert len(sink.hands) == 1

    def test_a_forced_read_needs_the_same_doubtful_hand_twice(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """打牌特效貼著手牌右邊,強制讀取剛好讀到那一幀,就憑空多一張摸牌。"""
        worker, sink = self._doubtful_worker(window, monkeypatch)
        hand = _hand_image("hand_14_with_draw.png")
        worker._read(hand, 0, forced=True)  # noqa: SLF001
        assert sink.hands == []
        worker._read(hand, 0, forced=True)  # noqa: SLF001
        assert len(sink.hands) == 1

    def test_a_doubtful_hand_is_dropped_when_the_screen_moves(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, sink = self._doubtful_worker(window, monkeypatch)
        moved = np.zeros((20, 20, 3), dtype=np.uint8)
        worker._previous = None  # noqa: SLF001
        worker._settle_and_read(moved, 0)  # noqa: SLF001
        for _ in range(vision._STILL_FRAMES):  # noqa: SLF001
            worker._offer_doubtful()  # noqa: SLF001
        assert sink.hands == []


class TestGlancingWhileItMoves:
    """實打時滑鼠滑過手牌,手牌區停不下來。動的時候也看一眼。"""

    TILE = Match(label="1m", score=0.9, runner_up="2m", runner_up_score=0.1, brightness=1.0)
    OTHER = Match(label="9p", score=0.9, runner_up="2m", runner_up_score=0.1, brightness=1.0)
    FUZZY = Match(label="5s", score=0.3, runner_up="2m", runner_up_score=0.29, brightness=1.0)

    def _worker(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch, seen: list[object]
    ) -> tuple[VisionWorker, _Sink, np.ndarray]:
        hand = _hand_image("hand_14_with_draw.png")
        frame, roi = _frame_with_hand(window, hand)
        sink = _Sink()
        worker = VisionWorker(
            UpdateBus(),
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
            sink=sink,
        )
        worker._templates = TemplateSet.load()  # noqa: SLF001
        answers = iter(seen)
        monkeypatch.setattr(vision, "classify_hand", lambda *_, **__: next(answers))
        return worker, sink, hand

    def test_a_clear_draw_goes_through_at_once(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker, sink, hand = self._worker(window, monkeypatch, [((self.TILE,) * 13, self.TILE)])
        worker._resting = ("1m",) * 13  # noqa: SLF001
        assert not worker._glance(hand, 0)  # noqa: SLF001
        assert sink.hands == [["1m"] * 14]

    def test_it_is_offered_once(self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch) -> None:
        both = ((self.TILE,) * 13, self.TILE)
        worker, sink, hand = self._worker(window, monkeypatch, [both, both])
        worker._resting = ("1m",) * 13  # noqa: SLF001
        worker._glance(hand, 0)  # noqa: SLF001
        worker._glance(hand, 0)  # noqa: SLF001
        assert len(sink.hands) == 1

    def test_a_fuzzy_draw_waits(self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch) -> None:
        worker, sink, hand = self._worker(window, monkeypatch, [((self.TILE,) * 13, self.FUZZY)])
        worker._resting = ("1m",) * 13  # noqa: SLF001
        worker._glance(hand, 0)  # noqa: SLF001
        assert sink.hands == []

    def test_the_rest_of_the_hand_has_to_be_the_same(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = [((self.TILE,) * 12 + (self.OTHER,), self.TILE)]
        worker, sink, hand = self._worker(window, monkeypatch, seen)
        worker._resting = ("1m",) * 13  # noqa: SLF001
        worker._glance(hand, 0)  # noqa: SLF001
        assert sink.hands == []

    def test_twice_the_same_counts_as_stopped(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """滑鼠一直在動時等不到靜止,實測大多要等滿一秒強制讀。"""
        both = ((self.TILE,) * 13, self.TILE)
        worker, _, hand = self._worker(window, monkeypatch, [both, both, both])
        assert not worker._glance(hand, 0)  # noqa: SLF001
        assert worker._glance(hand, 0)  # noqa: SLF001
        assert not worker._glance(hand, 0), "同一手不必一直重讀"  # noqa: SLF001

    def test_once_is_not_enough(self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch) -> None:
        one = ((self.TILE,) * 13, self.TILE)
        other = ((self.OTHER,) * 13, self.TILE)
        worker, _, hand = self._worker(window, monkeypatch, [one, other])
        worker._glance(hand, 0)  # noqa: SLF001
        assert not worker._glance(hand, 0)  # noqa: SLF001

    def test_anything_unsure_is_not_stopped(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fuzzy = ((self.TILE,) * 13, self.FUZZY)
        worker, _, hand = self._worker(window, monkeypatch, [fuzzy, fuzzy])
        worker._glance(hand, 0)  # noqa: SLF001
        assert not worker._glance(hand, 0)  # noqa: SLF001

    def test_the_settled_read_does_not_offer_the_draw_again(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """再交一次只是白白叫醒牌桌。"""
        both = ((self.TILE,) * 13, self.TILE)
        worker, sink, hand = self._worker(window, monkeypatch, [both, both])
        worker._resting = ("1m",) * 13  # noqa: SLF001
        worker._glance(hand, 0)  # noqa: SLF001
        worker._read(hand, 0)  # noqa: SLF001
        assert len(sink.hands) == 1


class TestReadingAHand:
    def test_posts_the_thirteen_tiles_it_sees(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
        )
        _run_until(worker, lambda: len(bus) > 0)

        posted = bus.drain()
        assert posted, "什麼都沒投出來"
        hand = posted[0]
        assert isinstance(hand, CvHand)
        assert len(hand.tiles) == 13
        assert hand.drawn is None  # 這張 fixture 是滿手 13 張、不是自己的回合

    def test_a_hand_with_a_drawn_tile_reports_it_separately(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image("hand_14_with_draw.png"))
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: len(bus) > 0)

        hand = bus.drain()[0]
        assert isinstance(hand, CvHand)
        assert len(hand.tiles) == 13
        assert hand.drawn is not None

    def test_tiles_are_in_majsoul_notation(self, window: WindowInfo) -> None:
        """投出去的是 classify 的原始輸出。轉成 MJAI 記法是 ViewModel 的事,
        在這裡先轉會讓兩邊都以為對方負責。
        """
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: len(bus) > 0)

        hand = bus.drain()[0]
        assert isinstance(hand, CvHand)
        for tile in hand.tiles:
            assert tile[-1] in "mpsz", f"{tile} 不是雀魂記法"


def _fading_frame(window: WindowInfo, step: int) -> Frame:
    """手牌半透明、每一幀深淺不同 —— 摸打動畫中途的樣子,認不出來。"""
    hand = _hand_image()
    frame, _ = _frame_with_hand(window, hand)
    image = frame.image.copy()
    hand_h, hand_w = hand.shape[:2]
    alpha = 0.15 + 0.03 * step
    base = np.full_like(hand, (40, 90, 40))
    faded = (hand * alpha + base * (1 - alpha)).astype(np.uint8)
    image[PASTE.y : PASTE.y + hand_h, PASTE.x + step : PASTE.x + step + hand_w] = faded
    return Frame(image, window, scale=1.0)


def _moving_frame(window: WindowInfo, offset: int) -> Frame:
    """手牌往右挪 ``offset`` 個像素 —— 每一幀都與前一幀不同,模擬動畫中途。"""
    hand = _hand_image()
    frame, _ = _frame_with_hand(window, hand)
    image = frame.image.copy()
    hand_h, hand_w = hand.shape[:2]
    image[PASTE.y : PASTE.y + hand_h, PASTE.x : PASTE.x + hand_w] = (40, 90, 40)
    image[
        PASTE.y : PASTE.y + hand_h, PASTE.x + offset : PASTE.x + offset + hand_w
    ] = hand
    return Frame(image, window, scale=1.0)


class TestOnlyClassifyingWhatHasStopped:
    """動的時候不認,停下來才認。

    動畫中途的畫面沒有正確答案 —— 牌是半透明、位移中的,模板比對只會給出一個
    沒有意義的結果,而它仍然會被投到 UI 上。離線評測從一開始就把轉場的幀整段
    丟掉(見 eval.align.SETTLE),即時這條路原本沒有對應的機制。
    """

    def test_an_identical_frame_is_not_classified_twice(self, window: WindowInfo) -> None:
        """這是省下九成模板比對的地方 —— 等別人打牌時畫面是靜止的。"""
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: worker.skipped >= 3)
        assert worker.reads == 1, f"同一張畫面認了 {worker.reads} 次"
        assert worker.skipped >= 3

    def test_a_frame_that_settles_is_classified_again(self, window: WindowInfo) -> None:
        first, roi = _frame_with_hand(window, _hand_image())
        second, _ = _frame_with_hand(window, _hand_image("hand_14_with_draw.png"))
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            # first 要出現兩次才算停下來過;second 之後由 FakeBackend 一直重複
            backend=FakeBackend(window, [first, first, second]),
            window=window,
        )
        _run_until(worker, lambda: worker.reads >= 2)
        assert worker.reads >= 2

    def test_a_frame_that_never_settles_is_not_classified(self, window: WindowInfo) -> None:
        """動畫中途每一幀都不一樣,那些幀一張都不該認 —— 認了就是把
        半透明、位移中的牌投到畫面上。"""
        roi = _frame_with_hand(window, _hand_image())[1]
        moving = [_fading_frame(window, step) for step in range(1, 8)]
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            # 幀用完之後模擬擷取失敗,而不是讓 FakeBackend 重複最後一張 ——
            # 重複的話會被 VisionWorker 誤判成「畫面停下來了」而觸發一次
            # classify,使這個測試在背景執行緒的時間點上變成賭運氣。
            backend=FakeBackend(window, [*moving, None]),
            window=window,
        )
        _run_until(worker, lambda: worker.skipped >= len(moving), timeout=3.0)
        assert worker.reads == 0, f"動畫中途認了 {worker.reads} 次"
        assert len(bus) == 0

    def test_tiles_that_move_but_read_clearly_are_read(self, window: WindowInfo) -> None:
        """滑鼠滑過手牌時一直在動,但每一張都認得出來:連兩眼一樣就讀,不必等滿一秒。"""
        roi = _frame_with_hand(window, _hand_image())[1]
        moving = [_moving_frame(window, offset) for offset in range(1, 8)]
        worker = VisionWorker(
            UpdateBus(),
            config=_config(roi),
            backend=FakeBackend(window, [*moving, None]),
            window=window,
        )
        _run_until(worker, lambda: worker.reads >= 1, timeout=3.0)
        assert worker.reads >= 1

    def test_something_that_never_stops_still_gets_read(self, window: WindowInfo) -> None:
        """ROI 裡有東西永遠在動的話(會呼吸的牌背皮膚、擷取雜訊),寧可退回
        舊行為也不要讓整個功能安靜地停擺。"""
        roi = _frame_with_hand(window, _hand_image())[1]
        forever = [_moving_frame(window, offset) for offset in range(1, 60)]
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, forever), window=window
        )
        _run_until(worker, lambda: worker.reads >= 1, timeout=5.0)
        assert worker.reads >= 1


class TestDegradedConditions:
    def test_missing_own_hand_roi_stops_with_an_explanation(self, window: WindowInfo) -> None:
        """設定裡沒量測 ROI 時要說清楚,不是靜靜地什麼都不做。"""
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(RoiConfig()),
            backend=FakeBackend(window, []),
            window=window,
        )
        worker.start()
        worker.join(3)
        assert not worker.is_alive()
        assert "own_hand" in worker.status.read()
        assert len(bus) == 0

    def test_no_game_window_says_so_and_keeps_trying(self) -> None:
        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(RoiConfig(own_hand=(0.1, 0.7, 0.6, 0.1))), backend=EmptyBackend()
        )
        _run_until(worker, lambda: "找不到" in worker.status.read(), timeout=3)
        assert "找不到" in worker.status.read()

    def test_it_says_which_version_it_was_looking_for(self) -> None:
        """「找不到視窗」最常見的原因是**選錯版本**。

        不講版本的話使用者會去找遊戲為什麼沒被認到,而真正的問題是他開著
        Steam 版、MIA 在找瀏覽器(或反過來)。
        """
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(RoiConfig(own_hand=(0.1, 0.7, 0.6, 0.1))),
            backend=EmptyBackend(),
            platform=STEAM,
        )
        _run_until(worker, lambda: "找不到" in worker.status.read(), timeout=3)
        assert "Steam 版" in worker.status.read()

    def test_capture_failures_are_reported_but_not_fatal(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        # 前三次失敗,之後成功 —— 要驗的是它撐過去了
        backend = FakeBackend(window, [None, None, None, frame])
        worker = VisionWorker(bus, config=_config(roi), backend=backend, window=window)
        _run_until(worker, lambda: len(bus) > 0, timeout=5)
        assert len(bus) > 0, "擷取失敗之後就再也沒有認出手牌"

    def test_an_animation_frame_is_skipped_without_clearing_the_hand(
        self, window: WindowInfo
    ) -> None:
        """張數不合法幾乎都是抓在動畫中間。這時上一巡的手牌仍是最好的資訊。"""
        good, roi = _frame_with_hand(window, _hand_image())
        # 只留左邊 6 張,其餘整條塗成桌布 —— 6 張不是合法的手牌張數
        partial = good.image.copy()
        hand_w = round(roi.own_hand[2] * FRAME_SIZE[0])
        hand_h = round(roi.own_hand[3] * FRAME_SIZE[1])
        cut = PASTE.x + round(hand_w * 0.42)
        partial[PASTE.y : PASTE.y + hand_h, cut : PASTE.x + hand_w] = (40, 90, 40)
        broken = Frame(partial, window, scale=1.0)

        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [broken]), window=window
        )
        _run_until(worker, lambda: "合法手牌" in worker.status.read(), timeout=3)
        assert worker.reads == 0
        assert len(bus) == 0

    def test_a_hand_too_small_to_classify_does_not_kill_the_thread(
        self, window: WindowInfo
    ) -> None:
        """視窗縮太小是**使用者救得回來**的狀況,不能是「不玩了」。

        `classify` 對太小的牌框會拋 ``ValueError`` 而不是硬算(那是對的 ——
        相關係數在十幾個像素上只是在比雜訊)。但那個例外原本一路竄到
        ``run()`` 的 blanket except,整條擷取執行緒就收掉了:使用者把視窗
        放大也回不來,只能重開 App。

        走得到這裡的除了「把瀏覽器縮到畫布 480 CSS 像素以下」,還有更常見的
        一條 —— 校正鎖到一個偏小的假牌桌(單幀校正實測 16% 明顯錯誤)。
        """
        tiny = cv2.resize(_hand_image(), (200, 23), interpolation=cv2.INTER_AREA)
        frame, roi = _frame_with_hand(window, tiny)

        bus = UpdateBus()
        worker = VisionWorker(
            bus, config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        _run_until(worker, lambda: "太小" in worker.status.read(), timeout=5)

        assert "太小" in worker.status.read(), f"狀態沒說原因: {worker.status.read()!r}"
        assert "已停止" not in worker.status.read(), "執行緒被打死了"
        assert len(bus) == 0

    def test_the_backend_is_closed_when_the_thread_ends(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        backend = FakeBackend(window, [frame])
        worker = VisionWorker(UpdateBus(), config=_config(roi), backend=backend, window=window)
        _run_until(worker, lambda: worker.reads > 0)
        assert backend.closed


class TestStopping:
    def test_stop_ends_the_thread_promptly(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        worker = VisionWorker(
            UpdateBus(),
            # 低幀率:停止不該等滿一個週期
            config=_config(roi, fps=1.0),
            backend=FakeBackend(window, [frame]),
            window=window,
        )
        worker.start()
        time.sleep(0.05)
        started = time.monotonic()
        worker.stop()
        worker.join(2)
        assert not worker.is_alive()
        assert time.monotonic() - started < 1.0, "停止等了超過一個擷取週期"


@pytest.mark.parametrize("count", [1, 4])
def test_repr_is_informative(window: WindowInfo, count: int) -> None:
    frame, roi = _frame_with_hand(window, _hand_image())
    worker = VisionWorker(
        UpdateBus(),
        config=_config(roi),
        backend=FakeBackend(window, [frame] * count),
        window=window,
    )
    assert "VisionWorker" in repr(worker)


def test_threading_module_is_used_for_the_stop_event() -> None:
    """釘住「用 Event 而不是 sleep」這個決定 —— 它是停止延遲的來源。"""
    worker = VisionWorker(UpdateBus(), config=_config(RoiConfig()))
    assert isinstance(worker._stop_event, threading.Event)  # noqa: SLF001


class TestTheManuallyFramedHand:
    """使用者自己框的手牌範圍 —— 框過就整個跳過校正。"""

    def _worker(self, window: WindowInfo, frame: Frame, roi: RoiConfig, box) -> tuple:
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            # 校正刻意給一個**錯的**手動矩形:框過的話根本不會走到那裡,
            # 走到了就會認出不一樣的牌,測試就會失敗。
            config=AppConfig(
                capture=CaptureConfig(target_fps=120.0),
                calibration=CalibrationConfig(manual_table_rect=(0.0, 0.0, 0.5, 0.5)),
                roi=roi,
            ),
            backend=FakeBackend(window, [frame]),
            window=window,
            hand_roi=HandRoiChoice(box),
        )
        return bus, worker

    def test_it_reads_the_hand_from_the_box(self, window: WindowInfo) -> None:
        frame, roi = _frame_with_hand(window, _hand_image())
        width, height = FRAME_SIZE
        hand_h, hand_w = _hand_image().shape[:2]
        box = (PASTE.x / width, PASTE.y / height, hand_w / width, hand_h / height)
        bus, worker = self._worker(window, frame, roi, box)
        _run_until(worker, lambda: len(bus) > 0)

        posted = bus.drain()
        assert posted, "框過了卻什麼都沒投出來"
        assert len(posted[0].tiles) == 13

    def test_the_box_bypasses_calibration(self, window: WindowInfo) -> None:
        """校正設成一個錯的矩形,框過的話不該有人去讀它。

        這一條才是重點:手動框**不是**另一種校正來源,它是整段跳過。
        Steam 版上校正那條路走不通,所以不能只是「優先用框」。
        """
        frame, roi = _frame_with_hand(window, _hand_image())
        width, height = FRAME_SIZE
        hand_h, hand_w = _hand_image().shape[:2]
        box = (PASTE.x / width, PASTE.y / height, hand_w / width, hand_h / height)
        bus, worker = self._worker(window, frame, roi, box)
        _run_until(worker, lambda: len(bus) > 0)
        assert "校正" not in worker.status.read()

    def test_without_a_box_it_still_calibrates(self, window: WindowInfo) -> None:
        """沒框過就走原本那條路 —— 網頁版的行為一個字都不該變。"""
        frame, roi = _frame_with_hand(window, _hand_image())
        bus = UpdateBus()
        worker = VisionWorker(
            bus,
            config=_config(roi),
            backend=FakeBackend(window, [frame]),
            window=window,
            hand_roi=HandRoiChoice(),
        )
        _run_until(worker, lambda: len(bus) > 0)
        assert bus.drain(), "沒框過時原本那條路壞了"


class TestStallWarnings:
    """實打時擷取停過 30 秒,事後只知道沒有畫面。中斷與恢復都要記下來,連同當時的原因。"""

    def test_a_stall_and_its_recovery_are_logged_with_the_reason(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mia.utils.logging import logger

        frame, roi = _frame_with_hand(window, _hand_image())
        monkeypatch.setattr(vision, "_STALL_WARN", 1.0)
        backend = FakeBackend(window, [frame] + [None] * 250 + [frame])
        worker = VisionWorker(
            UpdateBus(), config=_config(roi), backend=backend, window=window
        )
        messages: list[str] = []
        sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
        try:
            _run_until(worker, lambda: any("恢復" in m for m in messages), timeout=5)
        finally:
            logger.remove(sink)
        stalls = [m for m in messages if "沒有交出畫面" in m]
        assert len(stalls) == 1
        assert "擷取失敗" in stalls[0]
        assert any("擷取恢復" in m for m in messages)

    def test_a_slow_capture_is_logged(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mia.utils.logging import logger

        frame, roi = _frame_with_hand(window, _hand_image())
        monkeypatch.setattr(vision, "_SLOW_CAPTURE", 0.0)
        worker = VisionWorker(
            UpdateBus(), config=_config(roi), backend=FakeBackend(window, [frame]), window=window
        )
        messages: list[str] = []
        sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
        try:
            _run_until(worker, lambda: any("花了" in m for m in messages), timeout=3)
        finally:
            logger.remove(sink)
        assert any("擷取一次花了" in m for m in messages)
