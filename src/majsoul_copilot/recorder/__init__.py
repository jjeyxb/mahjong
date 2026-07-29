"""錄製與離線回放 —— 產生「畫面 ↔ Ground Truth」的成對資料集。"""

from majsoul_copilot.recorder.loop import RecorderStats, record_session
from majsoul_copilot.recorder.session import (
    FrameRecord,
    SessionManifest,
    SessionReader,
    SessionWriter,
)

__all__ = [
    "FrameRecord",
    "RecorderStats",
    "SessionManifest",
    "SessionReader",
    "SessionWriter",
    "record_session",
]
