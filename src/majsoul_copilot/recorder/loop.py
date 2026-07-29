"""錄製迴圈:定速擷取畫面並寫進 session。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from majsoul_copilot.calibration.table import Calibration, TableCalibrator
from majsoul_copilot.capture.base import CaptureBackend, CaptureFailedError, WindowInfo
from majsoul_copilot.recorder.session import SessionWriter
from majsoul_copilot.utils.logging import logger

__all__ = ["RecorderStats", "record_session"]


@dataclass
class RecorderStats:
    """一次錄製的統計。"""

    frames: int = 0
    stored: int = 0
    errors: int = 0
    dropped: int = 0  # 來不及跟上目標幀率的次數
    duration: float = 0.0

    @property
    def actual_fps(self) -> float:
        return self.frames / self.duration if self.duration > 0 else 0.0

    @property
    def store_ratio(self) -> float:
        return self.stored / self.frames if self.frames else 0.0


def record_session(
    backend: CaptureBackend,
    window: WindowInfo,
    writer: SessionWriter,
    *,
    target_fps: float = 15.0,
    duration: float | None = None,
    calibrator: TableCalibrator | None = None,
    max_consecutive_errors: int = 30,
    stop_check: object = None,
) -> RecorderStats:
    """持續擷取並寫入,直到時間到、使用者中斷,或連續失敗過多。

    Args:
        duration: 錄製秒數;None 表示錄到被 Ctrl-C 中斷為止。
        calibrator: 提供時會在第一幀做牌桌校正,並把結果寫進 manifest。
            畫面尺寸改變(使用者縮放視窗)時會自動重新校正。
        max_consecutive_errors: 連續擷取失敗超過此數就中止。單次失敗很正常
            (視窗剛好在切換),但持續失敗代表遊戲關了或權限沒了。
        stop_check: 可呼叫物件,回傳 True 時提前結束。給 GUI / 測試用。

    Returns:
        本次錄製的統計。
    """
    interval = 1.0 / target_fps
    stats = RecorderStats()
    calibration: Calibration | None = None
    consecutive_errors = 0

    start = time.monotonic()
    next_tick = start

    logger.info(
        "開始錄製 {} — 目標 {:.0f} fps{}",
        window,
        target_fps,
        f",{duration:.0f} 秒" if duration else ",按 Ctrl-C 停止",
    )

    try:
        while True:
            now = time.monotonic()
            if duration is not None and now - start >= duration:
                break
            if stop_check is not None and stop_check():  # type: ignore[operator]
                logger.info("收到停止訊號")
                break

            if now < next_tick:
                time.sleep(next_tick - now)
            elif now - next_tick > interval:
                # 已經落後超過一個週期:放棄追趕,把時基重設到現在,
                # 否則會進入「越追越忙」的惡性循環。
                stats.dropped += 1
                next_tick = now
            next_tick += interval

            try:
                frame = backend.capture(window)
            except CaptureFailedError as exc:
                stats.errors += 1
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("連續 {} 次擷取失敗,中止錄製: {}", consecutive_errors, exc)
                    break
                continue
            consecutive_errors = 0

            if calibrator is not None and (calibration is None or not calibration.matches(frame)):
                calibration = calibrator.calibrate(frame)
                logger.info("牌桌校正: {}", calibration)

            record = writer.add_frame(
                frame, table_rect=calibration.table_rect if calibration else None
            )
            stats.frames += 1
            stats.stored += int(record.changed)

            if stats.frames % 100 == 0:
                elapsed = time.monotonic() - start
                logger.info(
                    "已錄 {} 幀({:.0f}% 有變化),{:.0f} 秒,{:.1f} fps",
                    stats.frames,
                    100 * stats.stored / stats.frames,
                    elapsed,
                    stats.frames / elapsed,
                )
    except KeyboardInterrupt:
        logger.info("使用者中斷")

    stats.duration = time.monotonic() - start
    return stats
