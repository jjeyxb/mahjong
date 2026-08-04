"""擷取層抽象的測試 —— 不碰任何真實螢幕。"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from mia.capture.base import (
    CaptureBackend,
    Frame,
    WindowInfo,
    WindowNotFoundError,
)
from mia.utils.geometry import Rect, Size


class FakeBackend(CaptureBackend):
    """回傳預設視窗清單與純色影像的假後端。"""

    name = "fake"

    def __init__(self, windows: list[WindowInfo]) -> None:
        self._windows = windows

    @staticmethod
    def is_available() -> bool:
        return True

    def list_windows(self, *, include_all: bool = False) -> list[WindowInfo]:  # noqa: ARG002
        return list(self._windows)

    def capture(self, window: WindowInfo) -> Frame:
        image = np.zeros((window.bounds.height, window.bounds.width, 3), dtype=np.uint8)
        return Frame(image=image, window=window)


def _win(handle: int, title: str, owner: str, w: int, h: int) -> WindowInfo:
    return WindowInfo(handle=handle, title=title, owner=owner, bounds=Rect(0, 0, w, h))


class TestWindowInfoMatches:
    def test_case_insensitive_substring(self, window: WindowInfo) -> None:
        assert window.matches(["MAHJONG"])
        assert window.matches(["雀魂"])

    def test_matches_owner_too(self, window: WindowInfo) -> None:
        assert window.matches(["chrome"])

    def test_field_restriction(self, window: WindowInfo) -> None:
        assert window.matches(["chrome"], fields="owner")
        assert not window.matches(["chrome"], fields="title")

    def test_no_match(self, window: WindowInfo) -> None:
        assert not window.matches(["firefox"])

    def test_empty_patterns_never_match(self, window: WindowInfo) -> None:
        assert not window.matches([])

    def test_empty_string_pattern_is_ignored(self, window: WindowInfo) -> None:
        """空字串是所有字串的子字串,若不過濾會誤判所有視窗都命中。"""
        assert not window.matches([""])


class TestFrame:
    def test_size_uses_image_not_window_bounds(self, window: WindowInfo) -> None:
        """Retina 下影像是邏輯尺寸的兩倍,Frame.size 必須回報像素而非邏輯尺寸。"""
        frame = Frame(image=np.zeros((1440, 2560, 3), dtype=np.uint8), window=window, scale=2.0)
        assert frame.size == Size(2560, 1440)
        assert window.bounds.size == Size(1280, 720)

    def test_rejects_bgra(self, window: WindowInfo) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            Frame(image=np.zeros((10, 10, 4), dtype=np.uint8), window=window)

    def test_rejects_grayscale(self, window: WindowInfo) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            Frame(image=np.zeros((10, 10), dtype=np.uint8), window=window)

    def test_rejects_float_dtype(self, window: WindowInfo) -> None:
        with pytest.raises(ValueError, match="uint8"):
            Frame(image=np.zeros((10, 10, 3), dtype=np.float32), window=window)

    def test_crop_clips_to_image_bounds(self, window: WindowInfo) -> None:
        frame = Frame(image=np.zeros((100, 200, 3), dtype=np.uint8), window=window)
        cropped = frame.crop(Rect(150, 50, 200, 200))  # 超出右下角
        assert cropped.shape == (50, 50, 3)

    def test_crop_entirely_outside_raises(self, window: WindowInfo) -> None:
        frame = Frame(image=np.zeros((100, 200, 3), dtype=np.uint8), window=window)
        with pytest.raises(ValueError, match="之外"):
            frame.crop(Rect(500, 500, 10, 10))


class TestFindWindow:
    def test_matches_title(self) -> None:
        backend = FakeBackend([_win(1, "雀魂 MahjongSoul", "Chrome", 1280, 720)])
        assert backend.find_window(["雀魂"]).handle == 1

    def test_matches_owner(self) -> None:
        backend = FakeBackend([_win(1, "未命名", "MajSoul.exe", 1280, 720)])
        assert backend.find_window([], ["majsoul"]).handle == 1

    def test_prefers_largest_candidate(self) -> None:
        """多個分頁/視窗都叫雀魂時,遊戲本體通常是最大的那個。"""
        backend = FakeBackend(
            [
                _win(1, "雀魂 - 小視窗", "Chrome", 640, 360),
                _win(2, "雀魂 - 主視窗", "Chrome", 1920, 1080),
                _win(3, "雀魂 - 中視窗", "Chrome", 1280, 720),
            ]
        )
        assert backend.find_window(["雀魂"]).handle == 2

    def test_owner_match_beats_larger_title_match(self) -> None:
        """迴歸測試:編輯器開著檔名含「雀魂」的檔案,標題會命中且視窗更大。

        實機撞過這個 —— VS Code (1512x874) 蓋過了遊戲本體 (1512x870)。
        程式名稱比視窗標題可靠,必須優先。
        """
        backend = FakeBackend(
            [
                _win(1, "雀魂麻將輔助軟體架構設計 — mahjong", "Code", 1512, 874),
                _win(2, "雀魂麻將", "雀魂麻將", 1512, 870),
            ]
        )
        chosen = backend.find_window(["雀魂", "majsoul"], ["雀魂", "majsoul", "jantama"])
        assert chosen.handle == 2, "應選中遊戲本體而非編輯器"

    def test_exact_title_beats_substring_when_no_owner_match(self) -> None:
        backend = FakeBackend(
            [
                _win(1, "關於雀魂的筆記 - Notion", "Notion", 1920, 1080),
                _win(2, "雀魂", "Google Chrome", 1280, 720),
            ]
        )
        assert backend.find_window(["雀魂"], []).handle == 2

    def test_score_zero_for_non_matching_window(self) -> None:
        window = _win(1, "某個網頁", "Safari", 800, 600)
        assert FakeBackend([]).score_window(window, ["雀魂"], ["majsoul"]) == 0


class TestBrowserHostedWindow:
    """網頁版雀魂跑在瀏覽器裡,程式名是瀏覽器而不是遊戲。"""

    TITLES: ClassVar = ("雀魂", "majsoul")
    OWNERS: ClassVar = ("雀魂", "majsoul", "jantama")
    HOSTS: ClassVar = ("chrome", "chromium", "edge", "firefox", "safari")

    def test_browser_beats_larger_editor_window(self) -> None:
        """迴歸測試:實機撞過這個。

        VS Code 開著檔名含「雀魂」的檔案(1512x865),與真正跑著網頁版雀魂的
        Chromium(1282x865)**拿到完全相同的分數** —— 兩者程式名都不符,
        都只命中標題子字串,然後同分比面積,較大的 VS Code 就贏了。
        瀏覽器承載必須是一個獨立的加分項才能拆開這種平手。
        """
        backend = FakeBackend(
            [
                _win(1, "Code — 雀魂麻將輔助軟體架構設計 — mahjong", "Code", 1512, 865),
                _win(2, "Google Chrome for Testing — 雀魂麻將", "Google Chrome for Testing",
                     1282, 865),
            ]
        )
        chosen = backend.find_window(self.TITLES, self.OWNERS, self.HOSTS)
        assert chosen.handle == 2, "應選中瀏覽器裡的雀魂,而非碰巧同名的編輯器"

    def test_game_beats_a_larger_browser_window_that_merely_mentions_it(self) -> None:
        """迴歸測試:2026-08-04 實機撞到,而且是自己造成的。

        本專案的 GitHub repo 描述裡有「雀魂」兩個字,於是 Safari 開著那一頁時
        標題就命中了。它與真正的遊戲視窗**同樣**拿到「標題子字串 + 瀏覽器」,
        然後面積比遊戲大,於是贏了 —— 功能 1 整個對著一個網頁跑 CV,而畫面上
        只表現成「向聽分析沒反應」。

        瀏覽器加分擋不住這種,因為對手也是瀏覽器。要靠的是覆蓋率:
        「雀魂麻將」裡 pattern 佔一半,那個 80 幾字的網頁標題裡佔 2%。
        """
        backend = FakeBackend(
            [
                _win(
                    1,
                    "jjeyxb/mahjong: MIA — Mahjong Intelligence Assistant:"
                    "雀魂輔助工具(螢幕擷取 + CV + Mortal AI)",
                    "Safari",
                    1512,
                    870,
                ),
                _win(2, "雀魂麻將", "Google Chrome for Testing", 1200, 824),
            ]
        )
        chosen = backend.find_window(self.TITLES, self.OWNERS, self.HOSTS)
        assert chosen.handle == 2, "應選中遊戲,而非碰巧提到遊戲名的網頁"

    def test_a_long_title_still_counts_as_a_weak_match(self) -> None:
        """覆蓋率低不等於不算 —— 那可能真的是目標,只是標題很長。

        所以仍然給分(會被選中並記一句警告),只是輸給覆蓋率高的對手。
        """
        window = _win(1, "關於雀魂這款遊戲的一些非常冗長的個人筆記與心得", "Notion", 800, 600)
        assert FakeBackend([]).score_window(window, self.TITLES, self.OWNERS, self.HOSTS) > 0

    def test_the_longest_matching_pattern_decides_coverage(self) -> None:
        """多個 pattern 命中時取最長的 —— 長的那個更難是巧合。

        「雀魂 majsoul」裡兩個 pattern 都命中:短的「雀魂」佔 2/12,長的
        「majsoul」佔 7/12。取長的才過得了覆蓋率門檻。
        """
        long_hit = _win(1, "雀魂 majsoul", "Google Chrome", 1280, 720)
        short_only = _win(2, "雀魂之類的一大串無關緊要的文字啊啊啊", "Google Chrome", 1280, 720)
        backend = FakeBackend([])
        score = backend.score_window(long_hit, self.TITLES, self.OWNERS, self.HOSTS)
        weaker = backend.score_window(short_only, self.TITLES, self.OWNERS, self.HOSTS)
        assert score > weaker

    def test_browser_bonus_requires_a_title_match(self) -> None:
        """沒有標題佐證時瀏覽器不加分,否則任何一個分頁都會被誤選。"""
        window = _win(1, "GitHub - 某個專案", "Google Chrome", 1600, 900)
        assert FakeBackend([]).score_window(window, self.TITLES, self.OWNERS, self.HOSTS) == 0

    def test_unrelated_browser_window_never_wins(self) -> None:
        backend = FakeBackend(
            [
                _win(1, "YouTube", "Google Chrome", 1920, 1080),
                _win(2, "雀魂麻將", "Google Chrome", 1280, 720),
            ]
        )
        assert backend.find_window(self.TITLES, self.OWNERS, self.HOSTS).handle == 2

    def test_native_client_still_outranks_browser(self) -> None:
        """Steam 桌面版的程式名直接命中,應該勝過瀏覽器分頁。"""
        backend = FakeBackend(
            [
                _win(1, "Google Chrome for Testing — 雀魂麻將", "Google Chrome", 1920, 1080),
                _win(2, "雀魂麻將", "雀魂麻將", 1280, 720),
            ]
        )
        assert backend.find_window(self.TITLES, self.OWNERS, self.HOSTS).handle == 2

    def test_weak_match_is_still_returned(self) -> None:
        """只靠標題命中時仍然回傳,但要留下警告 —— 靜靜錄錯視窗最糟。"""
        backend = FakeBackend([_win(1, "雀魂筆記", "Notion", 1280, 720)])
        chosen = backend.find_window(self.TITLES, self.OWNERS, self.HOSTS)
        assert chosen.handle == 1

    def test_min_size_filters_out_small_windows(self) -> None:
        backend = FakeBackend([_win(1, "雀魂", "Chrome", 320, 200)])
        with pytest.raises(WindowNotFoundError):
            backend.find_window(["雀魂"], min_width=640, min_height=360)

    def test_error_lists_visible_windows_for_diagnosis(self) -> None:
        backend = FakeBackend([_win(1, "某個網頁", "Safari", 1280, 720)])
        with pytest.raises(WindowNotFoundError) as excinfo:
            backend.find_window(["雀魂"])
        message = str(excinfo.value)
        assert "某個網頁" in message and "Safari" in message

    def test_error_when_no_windows_at_all(self) -> None:
        with pytest.raises(WindowNotFoundError, match="沒有任何可擷取的視窗"):
            FakeBackend([]).find_window(["雀魂"])


class TestContextManager:
    def test_close_called_on_exit(self) -> None:
        closed = False

        class Tracking(FakeBackend):
            def close(self) -> None:
                nonlocal closed
                closed = True

        with Tracking([]):
            pass
        assert closed
