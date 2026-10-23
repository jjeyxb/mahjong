"""素材夠不夠格拿來量準確率。

擋的是一種**沒有症狀**的失敗:校正錄錯了,報告照樣跑完、照樣給出一份格式
完全正常的數字 —— 而那份數字量的是校正,不是辨識。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.eval.material import require_one_calibration
from mia.recorder.session import SessionReader, SessionWriter
from tests.conftest import make_letterboxed

from mia.capture.base import Frame, WindowInfo  # isort: skip
from mia.utils.geometry import Rect  # isort: skip


def _session(tmp_path: Path, window: WindowInfo, rects: list[Rect]) -> SessionReader:
    """造一份素材,每一幀帶一個校正結果。"""
    image = make_letterboxed((320, 240), Rect(0, 30, 320, 180))
    with SessionWriter(tmp_path, session_id="s", change_threshold=0.0) as writer:
        for i, rect in enumerate(rects):
            writer.add_frame(
                Frame(image=image, window=window, captured_at=float(i)), table_rect=rect
            )
    return SessionReader(tmp_path / "s")


class TestRequireOneCalibration:
    def test_a_steady_session_passes(self, tmp_path: Path, window: WindowInfo) -> None:
        session = _session(tmp_path, window, [Rect(0, 30, 320, 180)] * 3)
        require_one_calibration(session)  # 不該拋

    def test_a_session_whose_calibration_changed_is_refused(
        self, tmp_path: Path, window: WindowInfo
    ) -> None:
        """ROI 與牌河角點全部正規化到 table_rect —— 換過等於前後兩套座標。

        實測 20260801-112437 校正鎖在黑色載入畫面上,牌河每張牌 20.2%、
        手牌 26.3%;換上正確的矩形之後同一幀的手牌一張不差。
        """
        session = _session(
            tmp_path, window, [Rect(0, 30, 320, 180), Rect(0, 30, 320, 180), Rect(5, 5, 300, 170)]
        )
        with pytest.raises(SystemExit, match="第 2 幀換過"):
            require_one_calibration(session)

    def test_anyway_downgrades_it_to_a_warning(self, tmp_path: Path, window: WindowInfo) -> None:
        """使用者明講要看就讓他看 —— 擋的是「不知情」,不是「不准」。"""
        session = _session(tmp_path, window, [Rect(0, 30, 320, 180), Rect(5, 5, 300, 170)])
        require_one_calibration(session, anyway=True)

    def test_an_old_session_without_the_field_passes(
        self, tmp_path: Path, window: WindowInfo
    ) -> None:
        """這個欄位是後來才加的。舊素材讀成 0,不該被當成「換過」。"""
        session = _session(tmp_path, window, [Rect(0, 30, 320, 180)])
        session.manifest.table_rect_from = 0
        require_one_calibration(session)
