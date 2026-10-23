"""多幀穩定化校正測試。

這一組測試的骨架直接對應實測到的失敗模式:單幀校正在一份 216 幀的錄影裡
產生了 21 種不同的 ``table_rect``,約 16% 明顯錯誤(最糟的寬度少 258 px)。
原本的錄製迴圈是「第一幀算完就鎖住」,所以那 16% 有機會被鎖進整場對局。

合成畫面刻意重現三種真實的干擾:
  * 正確畫面(佔多數,但彼此有 ±1 px 抖動 —— 實測 1430 佔 108 幀、1429 佔 61 幀)
  * 少剝一截(角色立繪、和了動畫貼到邊緣)
  * 完全沒剝掉(暗轉、載入畫面,整張影像被當成內容)
"""

from __future__ import annotations

import numpy as np
import pytest

from mia.calibration import stable as stable_module
from mia.calibration.manual import TableRectChoice
from mia.calibration.stable import StableCalibrator
from mia.capture.base import Frame, WindowInfo
from mia.config.models import CalibrationConfig
from mia.utils.geometry import Rect, Size
from tests.conftest import make_letterboxed

#: 縮小版的實測情境:視窗含一條「瀏覽器工具列」,底下是 16:9 畫布。
IMAGE = (1280, 846)
TRUE_CANVAS = Rect(0, 126, 1280, 720)  # 1.7778


def _frame(image: np.ndarray, window: WindowInfo) -> Frame:
    return Frame(image=image, window=window, scale=1.0)


def good(window: WindowInfo, jitter: int = 0) -> Frame:
    """正確的一幀。``jitter`` 重現多數樣本之間的 ±1 px 抖動。

    抖動加在**上緣**、下緣固定貼齊影像底部 —— 真實的干擾也是這樣:工具列
    下方的邊界會因為畫面內容而多剝或少剝幾列,畫布底部則沒得跑。
    """
    rect = Rect(
        TRUE_CANVAS.x, TRUE_CANVAS.y + jitter, TRUE_CANVAS.width, TRUE_CANVAS.height - jitter
    )
    return _frame(make_letterboxed(IMAGE, rect), window)


def under_peeled(window: WindowInfo) -> Frame:
    """少剝一截 —— 立繪或動畫貼到工具列下緣,邊界偵測提早停住。"""
    rect = Rect(0, 40, 1280, 806)  # 1.5881,偏離 16:9 達 10.7%
    return _frame(make_letterboxed(IMAGE, rect), window)


def not_peeled(window: WindowInfo) -> Frame:
    """整張影像都是內容 —— 暗轉或載入畫面,四邊都不是純色。"""
    rng = np.random.default_rng(7)
    return _frame(rng.integers(80, 256, (IMAGE[1], IMAGE[0], 3), dtype=np.uint8), window)


#: 一個長寬比完全正常、位置卻完全不同的畫布。
#:
#: 這是「鎖在載入畫面上」那種失敗的樣子:候選本身挑不出毛病(1.7778 正好是
#: 16:9),兩道既有的防線一道都不會響,而它就是不對。實測
#: ``20260801-112437`` 鎖到的是 ``1708x968@(69,218)``,長寬比 1.7645。
OTHER_CANVAS = Rect(60, 200, 1120, 630)


def wrong_place(window: WindowInfo) -> Frame:
    """畫布在別的地方 —— 鎖定當下畫面還不是牌桌。"""
    return _frame(make_letterboxed(IMAGE, OTHER_CANVAS), window)


def featureless(window: WindowInfo) -> Frame:
    """整張同一個顏色 —— 剝除找不到任何內容區,只能回 fallback。"""
    return _frame(np.zeros((IMAGE[1], IMAGE[0], 3), dtype=np.uint8), window)


class TestLocking:
    def test_stays_unlocked_until_enough_samples(self, window: WindowInfo) -> None:
        """蒐集期間必須回傳 None,不能先給一個「暫時的」答案。

        給暫時值等於把單幀校正的問題原封不動搬過來 —— 呼叫端拿到一個看起來
        可用的 Calibration,不會知道它只是抽樣抽到的。
        """
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        for _ in range(4):
            assert stable.feed(good(window)) is None
            assert not stable.locked
        assert stable.feed(good(window)) is not None
        assert stable.locked

    def test_progress_reports_accepted_not_fed(self, window: WindowInfo) -> None:
        """進度算的是**被接受**的候選,被丟掉的不能算數。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        for _ in range(10):
            stable.feed(not_peeled(window))
        assert stable.progress == (0, 5)
        assert not stable.locked

    def test_locked_result_is_returned_without_recomputing(self, window: WindowInfo) -> None:
        """鎖定後再餵爛畫面也不該動搖結果 —— 這是「鎖住」的意義。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(3):
            stable.feed(good(window))
        locked = stable.result
        assert locked is not None
        for _ in range(20):
            assert stable.feed(not_peeled(window)) == locked


class TestOutlierRejection:
    def test_a_minority_of_bad_frames_cannot_move_the_answer(self, window: WindowInfo) -> None:
        """實測的核心情境:16% 的幀明顯錯誤,結果必須仍然正確。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=15))
        frames = [good(window, jitter=i % 2) for i in range(17)]
        frames += [under_peeled(window)] * 2 + [not_peeled(window)] * 2
        # 打散,免得測試只證明「壞幀剛好排在後面所以沒被取樣到」
        order = [0, 17, 1, 2, 18, 3, 4, 5, 19, 6, 7, 8, 20, 9, 10, 11, 12, 13, 14, 15, 16]
        result = None
        for i in order:
            result = stable.feed(frames[i]) or result

        assert result is not None
        assert result.table_rect.x == TRUE_CANVAS.x
        assert result.table_rect.width == TRUE_CANVAS.width
        assert abs(result.table_rect.y - TRUE_CANVAS.y) <= 1
        assert abs(result.table_rect.height - TRUE_CANVAS.height) <= 1

    def test_first_frame_being_wrong_does_not_poison_the_session(self, window: WindowInfo) -> None:
        """原本的 bug 就是這個 —— 遊戲剛啟動常停在載入畫面,而那正是剝除最容易失敗的時候。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        result = None
        for frame in [not_peeled(window), under_peeled(window), *[good(window) for _ in range(5)]]:
            result = stable.feed(frame) or result
        assert result is not None
        assert result.table_rect == TRUE_CANVAS

    def test_wildly_off_aspect_is_rejected_outright(self, window: WindowInfo) -> None:
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(5):
            stable.feed(under_peeled(window))
        assert stable.progress[0] == 0, "偏離 10.7% 的候選不該被納入"

    def test_a_mildly_off_aspect_is_still_accepted(self, window: WindowInfo) -> None:
        """正確答案本身就會偏離 16:9(實測 1.7902,偏 0.7%),不能因為示警就丟掉。

        aspect_reject 比 aspect_tolerance 寬鬆是刻意的:前者是「這幀根本沒看到
        牌桌」的門檻,後者只是「值得提醒使用者」的門檻。
        """
        mild = Rect(0, 130, 1280, 716)  # 1.7877,偏 0.6%,在 tolerance 邊緣
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=1))
        result = stable.feed(_frame(make_letterboxed(IMAGE, mild), window))
        assert result is not None
        assert result.table_rect == mild


class TestSpreadWarning:
    def test_consistent_frames_produce_no_warning(self, window: WindowInfo) -> None:
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        result = None
        for i in range(5):
            result = stable.feed(good(window, jitter=i % 2)) or result
        assert result is not None
        assert result.warnings == ()
        assert result.is_reliable

    def test_noisy_frames_lock_anyway_but_warn(self, window: WindowInfo) -> None:
        """畫面很不穩時仍要給答案(中位數就是為此而生),但必須讓使用者知道。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=5))
        result = None
        for jitter in (0, 30, -30, 25, -25):
            result = stable.feed(good(window, jitter=jitter)) or result
        assert result is not None
        assert result.warnings, "候選差異這麼大卻毫無警告,是靜默地給錯答案"
        assert "中位數" in result.warnings[0]


class TestResize:
    def test_size_change_triggers_a_fresh_collection(self, window: WindowInfo) -> None:
        """使用者縮放視窗後,舊的取樣全部作廢 —— 混在一起會得到一個誰都不對的矩形。"""
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(3):
            stable.feed(good(window))
        assert stable.locked

        bigger = Rect(0, 252, 2560, 1440)
        larger_frame = _frame(make_letterboxed((2560, 1692), bigger), window)
        assert stable.feed(larger_frame) is None, "尺寸變了必須重新蒐集,不能沿用"
        assert stable.progress[0] == 1

        result = None
        for _ in range(2):
            result = stable.feed(larger_frame) or result
        assert result is not None
        assert result.table_rect == bigger
        assert result.image_size == Size(2560, 1692)


class TestManual:
    def test_manual_rect_locks_immediately(self, window: WindowInfo) -> None:
        """手動指定就沒有「不穩定」可言,不該還要等 15 幀。"""
        config = CalibrationConfig(manual_table_rect=(0.0, 0.1, 1.0, 0.8), stabilize_frames=15)
        stable = StableCalibrator(config)
        result = stable.feed(good(window))
        assert result is not None
        assert result.source == "manual"


class TestTheLockIsRechecked:
    """鎖定之後還要繼續複查。

    既有的兩道防線問的都是「候選看起來對不對」。有一種失敗躲得過它們:
    鎖定當下畫面根本還不是牌桌,而那一幀剝出來的矩形看起來完全正常。
    實測 ``20260801-112437`` 錄製開頭遊戲正在重連,前十幾幀是黑色載入畫面,
    剝出 ``1708x968@(69,218)``(長寬比 1.7645,離 16:9 只差 0.7%)——
    **採用 15 個候選、丟棄 0 個**,沒有任何警告,然後整場都用那個值。

    唯一分得出來的證據是:牌桌出現之後,畫面說的跟當初鎖的不一樣。
    """

    def _tight(self, monkeypatch: pytest.MonkeyPatch, window_len: int = 6) -> None:
        """把複查的節奏調快,否則一個測試要餵一千多幀。

        調的是節奏不是邏輯 —— 真正的值由
        :meth:`TestTheRecheckWindow.test_the_window_outlasts_an_animation` 釘住。
        """
        monkeypatch.setattr(stable_module, "_VERIFY_EVERY", 1)
        monkeypatch.setattr(stable_module, "_VERIFY_WINDOW", window_len)

    def test_a_lock_made_before_the_table_appeared_is_dropped(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._tight(monkeypatch)
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(3):
            stable.feed(wrong_place(window))
        assert stable.result is not None
        assert stable.result.table_rect == OTHER_CANVAS, "前提:先鎖到錯的地方"

        result = None
        for _ in range(20):
            result = stable.feed(good(window)) or result
        assert result is not None
        assert result.table_rect == TRUE_CANVAS, "牌桌出現之後該改過來"

    def test_a_burst_of_bad_frames_does_not_drop_a_good_lock(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """和了動畫、結算、配牌那段剝除會一直失敗 —— 那是**沒看到**,不是鎖錯了。

        解錯的代價是整個辨識停擺,所以寧可晚一點發現,也不能被一次和了解掉。
        """
        self._tight(monkeypatch, window_len=10)
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(3):
            stable.feed(good(window))
        locked = stable.result
        assert locked is not None

        for _ in range(5):  # 短於視窗的一陣壞幀
            stable.feed(not_peeled(window))
        assert stable.locked, "一陣雜訊就解鎖,等於把 16% 的單幀誤差放大成停擺"
        assert stable.result is locked

    def test_undetectable_frames_do_not_push_toward_unlocking(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """整張暗掉的幀問不出答案 —— 那既不是支持也不是反對,不該計入。

        計入的話每次過場都往解鎖推一把,而過場正是最常出現的東西。
        """
        self._tight(monkeypatch, window_len=3)
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=3))
        for _ in range(3):
            stable.feed(good(window))
        assert stable.locked
        for _ in range(50):
            stable.feed(featureless(window))
        assert stable.locked

    def test_a_manual_rect_is_never_rechecked(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """使用者講了什麼就是什麼。畫面對不上是他要知道的事,不是這裡自作主張改掉的事。"""
        self._tight(monkeypatch, window_len=2)
        config = CalibrationConfig(manual_table_rect=(0.0, 0.1, 1.0, 0.8))
        stable = StableCalibrator(config)
        locked = stable.feed(good(window))
        assert locked is not None
        for _ in range(20):
            assert stable.feed(wrong_place(window)) is locked


class TestTheRecheckWindow:
    """複查視窗的長度是量出來的,不是估的。"""

    def test_the_window_outlasts_an_animation(self) -> None:
        """實測鎖定正確的那份錄影,最長連續 27 次複查都不一致(和了 + 結算 + 配牌)。

        視窗短過那個,一次和了就能解掉一個正確的鎖定。掃過各種長度之後取 80
        —— 那時正確的那份在最糟的視窗裡也只有 36% 不一致,離半數還有 14 個
        百分點;視窗 60 只剩 5 個百分點。
        """
        longest_outage = 27
        assert longest_outage * 2 <= stable_module._VERIFY_WINDOW  # noqa: SLF001

    def test_a_check_is_about_a_second(self) -> None:
        """視窗真正該管的是涵蓋多少**時間** —— 查得更密不會更準,
        只會讓同一段動畫塞滿視窗。擷取實測 15 fps。
        """
        assert stable_module._VERIFY_EVERY == 15  # noqa: SLF001


class TestManualFraming:
    """使用者自己框的牌桌矩形。自動剝除走不通時的逃生門。

    與「框選手牌範圍」是兩件事:那個**取代**校正,這個**就是**校正的結果 ——
    ROI 與四家牌河的角點仍然照常相對它推導,只是那個矩形改成人給的。
    """

    def test_a_framed_rect_locks_on_the_first_frame(self, window: WindowInfo) -> None:
        """人講的話不必「蒐集多幀取中位數」—— 那是為了對付猜的結果的雜訊。"""
        manual = TableRectChoice((0.0, 0.25, 1.0, 0.5))
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=15), manual=manual)
        result = stable.feed(good(window))
        assert result is not None
        assert result.source == "manual"
        assert result.table_rect == Rect(0, 212, 1280, 423)

    def test_the_framed_rect_beats_the_peeling(self, window: WindowInfo) -> None:
        """剝除仍然會算出 TRUE_CANVAS,但人指定了就以人為準。

        反過來做(啟發式蓋過人)的話,使用者會框了一次、看不出任何變化,
        而那正是他去框的原因。
        """
        manual = TableRectChoice((0.1, 0.1, 0.5, 0.5))
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=1), manual=manual)
        result = stable.feed(good(window))
        assert result is not None
        assert result.table_rect != TRUE_CANVAS

    def test_reframing_takes_effect_on_the_next_frame(self, window: WindowInfo) -> None:
        """框完要立刻生效,不能等複查 —— 那最快也要八十秒。

        那八十秒裡畫面上會「框完了卻沒反應」,而那正是這個功能要解決的
        那種沉默。
        """
        manual = TableRectChoice()
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=1), manual=manual)
        first = stable.feed(good(window))
        assert first is not None
        assert first.table_rect == TRUE_CANVAS

        manual.set((0.0, 0.25, 1.0, 0.5))
        second = stable.feed(good(window))
        assert second is not None
        assert second.source == "manual"
        assert second.table_rect == Rect(0, 212, 1280, 423)

    def test_clearing_goes_back_to_the_peeling(self, window: WindowInfo) -> None:
        """「改回自動」也要立刻生效,不然使用者會以為按鈕壞了。"""
        manual = TableRectChoice((0.0, 0.25, 1.0, 0.5))
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=1), manual=manual)
        assert stable.feed(good(window)) is not None
        manual.clear()
        back = stable.feed(good(window))
        assert back is not None
        assert back.table_rect == TRUE_CANVAS

    def test_a_framed_rect_is_never_second_guessed(
        self, window: WindowInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """複查是用來抓「程式自己猜錯了」的。人講的話不在它管轄範圍內 ——
        畫面對不上是使用者要知道的事,不是這裡自作主張改掉的事。
        """
        monkeypatch.setattr(stable_module, "_VERIFY_EVERY", 1)
        monkeypatch.setattr(stable_module, "_VERIFY_WINDOW", 2)
        manual = TableRectChoice((0.0, 0.25, 1.0, 0.5))
        stable = StableCalibrator(CalibrationConfig(stabilize_frames=1), manual=manual)
        locked = stable.feed(good(window))
        assert locked is not None
        for _ in range(20):
            assert stable.feed(wrong_place(window)) is locked
