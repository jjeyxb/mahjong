"""畫面與封包的對齊。

對齊錯了,準確率報告會給出一個看起來很正常但完全錯誤的數字 —— 那比報錯還糟。
所以這裡測得比一般的工具函式嚴,尤其是「什麼時候該拒絕回答」。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from majsoul_copilot.capture.base import Frame, WindowInfo
from majsoul_copilot.eval.align import HandState, HandTimeline, align, build_timeline
from majsoul_copilot.recorder import SessionReader, SessionWriter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_game_excerpt.jsonl"


def timeline(*walls: float) -> HandTimeline:
    """每個時刻一個狀態,手牌內容用時刻本身標記,方便斷言。"""
    built = HandTimeline(
        states=[HandState(w, (f"{int(w)}m",), None, 0, True) for w in walls],
        seat=0,
    )
    return built.freeze()


class TestStableAt:
    def test_a_settled_state_is_returned(self) -> None:
        assert timeline(100.0).stable_at(101.0, settle=0.4) is not None

    def test_nothing_before_the_first_state(self) -> None:
        assert timeline(100.0).stable_at(99.0) is None

    def test_a_frame_during_the_animation_is_refused(self) -> None:
        """封包已經到了,但畫面還在演動畫 —— 這時畫面與封包必然不一致。

        把這種幀算進去,等於把動畫延遲記成 CV 的錯誤。
        """
        assert timeline(100.0).stable_at(100.2, settle=0.4) is None

    def test_a_frame_just_before_the_next_change_is_refused(self) -> None:
        """下一個狀態馬上就到,畫面隨時會變 —— 同樣不能算。"""
        assert timeline(100.0, 101.0).stable_at(100.8, settle=0.4) is None

    def test_a_frame_in_the_quiet_middle_is_accepted(self) -> None:
        state = timeline(100.0, 102.0).stable_at(101.0, settle=0.4)
        assert state is not None
        assert state.tiles == ("100m",)

    def test_a_larger_settle_rejects_more(self) -> None:
        """穩定秒數是可調的嚴格度 —— 加大會讓可評分的幀變少但更乾淨。"""
        built = timeline(100.0, 102.0)
        assert built.stable_at(100.5, settle=0.4) is not None
        assert built.stable_at(100.5, settle=0.8) is None

    def test_a_desynced_state_is_never_returned(self) -> None:
        """事件流已經對不上,這之後的手牌不能當標準答案。"""
        built = HandTimeline(states=[HandState(100.0, ("1m",), None, 0, False)]).freeze()
        assert built.stable_at(101.0) is None

    def test_at_ignores_stability(self) -> None:
        """:meth:`at` 是給診斷用的,不做穩定性檢查。"""
        assert timeline(100.0).at(100.1) is not None


@pytest.fixture(scope="module")
def real() -> HandTimeline:
    """真實對局的手牌時間軸。解析一次全班共用 —— 建一次約 0.5 秒。"""
    return build_timeline(FIXTURE)


class TestBuildTimeline:

    def test_it_reads_the_real_recording(self, real: HandTimeline) -> None:
        assert len(real) > 0
        assert real.seat == 2

    def test_the_stream_stays_in_sync(self, real: HandTimeline) -> None:
        """一場真實對局從頭到尾都對得上 —— 有 desync 就代表 to_mjai 漏了東西。"""
        assert all(s.in_sync for s in real.states)

    def test_hand_sizes_are_always_legal(self, real: HandTimeline) -> None:
        """13/14 張(或副露後對應的張數)。melds 為 -1 就是不合法。"""
        assert all(s.melds >= 0 for s in real.states)

    def test_states_are_sorted_by_time(self, real: HandTimeline) -> None:
        walls = [s.wall for s in real.states]
        assert walls == sorted(walls)

    def test_red_fives_are_kept_distinct(self, real: HandTimeline) -> None:
        """赤五若被還原成普通五,標準答案就測不出模板庫漏了那三張。"""
        assert any(t.endswith("r") for s in real.states for t in s.tiles)

    def test_states_only_appear_when_the_hand_changes(self, real: HandTimeline) -> None:
        """別家的動作不該產生狀態 —— 那會讓每一幀都落在「剛變化」而被拒絕。"""
        pairs = zip(real.states, real.states[1:], strict=False)
        assert all(a.tiles != b.tiles or a.drawn != b.drawn for a, b in pairs)


class TestAlign:
    def _session(self, tmp_path: Path, window: WindowInfo, walls: list[float]) -> SessionReader:
        """造一個 session,其幀落在指定的牆上時鐘。

        ``SessionWriter`` 用第一幀的 ``captured_at``(單調時鐘)推算
        ``first_frame_wall``,所以這裡要用 :func:`time.monotonic` 的基準。
        """
        base_mono = time.monotonic()
        base_wall = time.time()
        with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
            for i, wall in enumerate(walls):
                image = np.full((80, 120, 3), (i * 40) % 255, dtype=np.uint8)
                writer.add_frame(
                    Frame(image, window=window, scale=1.0, captured_at=base_mono + wall - base_wall)
                )
        return SessionReader(tmp_path / "s")

    def test_frames_in_quiet_windows_get_a_truth(
        self, tmp_path: Path, window: WindowInfo
    ) -> None:
        now = time.time()
        session = self._session(tmp_path, window, [now, now + 1.0, now + 2.0])
        built = timeline(now - 5.0)
        paired = list(align(session, built))
        assert len(paired) == 3

    def test_frames_outside_the_recording_are_dropped(
        self, tmp_path: Path, window: WindowInfo
    ) -> None:
        """畫面錄得比封包早 —— 那幾幀沒有標準答案,略過而不是配一個最近的。"""
        now = time.time()
        session = self._session(tmp_path, window, [now, now + 1.0])
        paired = list(align(session, timeline(now + 100.0)))
        assert paired == []

    def test_an_empty_timeline_pairs_nothing(self, tmp_path: Path, window: WindowInfo) -> None:
        session = self._session(tmp_path, window, [time.time()])
        assert list(align(session, HandTimeline().freeze())) == []

    def test_indices_can_be_restricted(self, tmp_path: Path, window: WindowInfo) -> None:
        now = time.time()
        session = self._session(tmp_path, window, [now, now + 1.0, now + 2.0])
        paired = list(align(session, timeline(now - 5.0), indices=[1]))
        assert [f.index for f in paired] == [1]

    def test_a_session_without_the_anchor_is_rejected(
        self, tmp_path: Path, window: WindowInfo
    ) -> None:
        """舊格式沒有 first_frame_wall。硬對齊會得到一個完全錯誤但看似正常的結果。"""
        session = self._session(tmp_path, window, [time.time()])
        session.manifest.first_frame_wall = 0.0
        with pytest.raises(ValueError, match="無法自動對齊"):
            list(align(session, timeline(0.0)))
