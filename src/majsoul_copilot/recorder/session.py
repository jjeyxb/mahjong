"""錄製工作階段的磁碟格式:寫入、讀取、離線回放。

一個 session 就是一個目錄::

    data/recordings/20260726-190312_majsoul/
        manifest.json      後設資料 + 幀索引
        frames/000042.png  畫面(僅存有變化的幀)
        events.jsonl       Ground Truth 事件流(有 proxy 時)

為什麼只存「有變化」的幀
------------------------
麻將牌桌大部分時間是靜止的。以 15 fps 錄一場 10 分鐘的半莊,若每幀都存,
在 3024x1740 的解析度下是 9000 張 × 約 3 MB ≈ 27 GB,完全不可行。

實際做法是:**每幀都記錄時間戳,但只有畫面真的變了才寫入像素**。
索引裡每一筆都保留時間資訊與所指向的影像檔,沒變化的幀就指回前一張。
這樣既保有完整的時間軸(事件對齊需要),檔案量又降到可接受的範圍。
實測一場對局約有 80% 的幀是靜止的。

為什麼預設用 JPEG 而不是 PNG
----------------------------
這批資料的用途是量測辨識準確率,直覺上應該用無損格式。但在 3024x1740 下實測
(單張真實牌局畫面):

===============  ========  ========  ==================
格式             檔案大小  寫入耗時  模板比對峰值分數
===============  ========  ========  ==================
PNG(級別 1)     4.10 MB   127 ms    1.000000
PNG(級別 9)     3.22 MB   7710 ms   1.000000
JPEG q95         1.00 MB   9 ms      0.999800
===============  ========  ========  ==================

PNG 的 127 ms 寫入**超過 10 fps 的取樣週期**,錄製時會直接掉幀;而 JPEG q95
對模板比對的影響只出現在小數第四位,峰值位置完全一致(平均像素差 1.01,
99 百分位 9)。四倍的空間差距與十四倍的速度差距換這個誤差,是划算的。

需要嚴格無損時(例如要證明壓縮確實不影響結論),用 ``image_format="png"``
錄一小段對照組即可。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from majsoul_copilot.capture.base import Frame, WindowInfo
from majsoul_copilot.utils.geometry import Rect
from majsoul_copilot.utils.logging import logger

__all__ = ["FrameRecord", "SessionManifest", "SessionReader", "SessionWriter"]

MANIFEST_NAME = "manifest.json"
FRAMES_DIR = "frames"
EVENTS_NAME = "events.jsonl"

_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """索引中的一筆幀紀錄。

    Attributes:
        index: 幀序號,從 0 起算且連續(包含沒有寫入像素的幀)。
        timestamp: 相對於 session 開始的秒數(單調時鐘)。
        image: 影像檔名(相對於 frames/)。沒有變化時指向前一張,
            因此多筆紀錄可能共用同一個檔名。
        changed: 這一幀是否觸發了寫檔。
        scale: 擷取當下的 pixel/logical 比值。
    """

    index: int
    timestamp: float
    image: str
    changed: bool
    scale: float = 1.0


@dataclass
class SessionManifest:
    """session 的後設資料。"""

    format_version: int = _FORMAT_VERSION
    session_id: str = ""
    started_at: str = ""  # ISO-8601 UTC,寫在**建構當下**
    # 第一幀擷取當下的 :func:`time.time`。這是畫面與 GT 對齊的唯一錨點 ——
    # 兩者是各自獨立的行程,各有各的單調時鐘原點,只有牆上時鐘可以共用。
    # 為什麼不直接用 ``started_at``:它記的是建構 SessionWriter 的時刻,與第一幀
    # 之間隔著第一次擷取。實測(macOS/Quartz,tools/record.py 的呼叫順序)只差
    # 21 ms —— 在 10 fps 下不到四分之一幀,所以這不是在修一個已經在發生的錯誤。
    #
    # 真正的理由是那 21 ms 是**偶然**的:它剛好小,是因為 record.py 先建好後端
    # 再建 writer。任何一個呼叫端只要把建構提前(例如先開 writer 再等使用者
    # 按鍵開始),落差就會從毫秒變成秒,而且不會有任何徵兆 —— 錯位的資料集看
    # 起來完全正常。錨點綁在第一幀上,這個相依性就不存在了。
    # 0.0 表示舊格式,無法自動對齊。
    first_frame_wall: float = 0.0
    backend: str = ""
    window_title: str = ""
    window_owner: str = ""
    window_bounds: list[int] = field(default_factory=list)  # [x, y, w, h] 邏輯座標
    image_size: list[int] = field(default_factory=list)  # [width, height] 像素
    table_rect: list[int] | None = None  # 校正結果 [x, y, w, h] 像素
    image_format: str = "png"
    frame_count: int = 0
    stored_frames: int = 0
    duration: float = 0.0
    notes: str = ""
    frames: list[FrameRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["frames"] = [asdict(f) for f in self.frames]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SessionManifest:
        frames = [FrameRecord(**f) for f in data.pop("frames", [])]
        return cls(**data, frames=frames)

    @property
    def window(self) -> WindowInfo:
        """重建 WindowInfo,讓回放出來的 Frame 與線上擷取的形狀一致。"""
        x, y, w, h = self.window_bounds or [0, 0, 0, 0]
        return WindowInfo(
            handle=0, title=self.window_title, owner=self.window_owner, bounds=Rect(x, y, w, h)
        )


def _thumbnail(image: np.ndarray, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    """縮成灰階小圖,用來廉價地比對「畫面有沒有變」。

    直接比對全解析度影像太慢(3024x1740 每幀要處理 500 萬像素),而縮圖
    足以偵測牌張出現/消失這種等級的變化。
    """
    return cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


class SessionWriter:
    """把擷取到的畫面與事件寫成一個 session 目錄。"""

    def __init__(
        self,
        root: Path | str,
        *,
        session_id: str | None = None,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        change_threshold: float = 1.5,
        backend: str = "",
        notes: str = "",
    ) -> None:
        """
        Args:
            root: 所有 session 的上層目錄,例如 ``data/recordings``。
            session_id: 目錄名;省略則用時間戳自動產生。
            image_format: ``jpg``(預設,理由見模組 docstring)或 ``png``(無損)。
            jpeg_quality: JPEG 品質。低於 90 開始明顯影響模板比對,不建議。
            change_threshold: 縮圖的平均絕對差超過此值才視為「畫面有變化」。
                單位是 0~255 的灰階值。設 0 表示每幀都存。
        """
        self.root = Path(root)
        self.session_id = session_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = self.root / self.session_id
        self.frames_dir = self.path / FRAMES_DIR
        self.image_format = image_format.lower().lstrip(".")
        self.change_threshold = change_threshold
        # PNG 級別 1 是刻意的:級別 9 只小 20%,寫入卻要 7.7 秒,完全不能用於錄製。
        self._write_params = (
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            if self.image_format in ("jpg", "jpeg")
            else [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = SessionManifest(
            session_id=self.session_id,
            started_at=datetime.now(UTC).isoformat(),
            backend=backend,
            image_format=self.image_format,
            notes=notes,
        )

        self._start_time: float | None = None
        self._previous_thumb: np.ndarray | None = None
        self._last_image_name = ""
        self._events_file: TextIO | None = None
        self._closed = False

    # ------------------------------------------------------------------

    def add_frame(self, frame: Frame, *, table_rect: Rect | None = None) -> FrameRecord:
        """加入一幀。畫面與前一幀差異夠大時才實際寫檔。"""
        if self._closed:
            raise RuntimeError("session 已關閉")

        if self._start_time is None:
            self._start_time = frame.captured_at
            # 減掉「擷取完成 → 寫入」這段延遲,把牆上時鐘校回擷取的那一瞬間。
            # 單張擷取約需數十毫秒,不扣掉的話整條時間軸會整體偏移。
            self.manifest.first_frame_wall = time.time() - (time.monotonic() - frame.captured_at)
            self.manifest.window_title = frame.window.title
            self.manifest.window_owner = frame.window.owner
            b = frame.window.bounds
            self.manifest.window_bounds = [b.x, b.y, b.width, b.height]
            self.manifest.image_size = [frame.size.width, frame.size.height]

        # 校正**不是**第一幀就有:StableCalibrator 要蒐集數幀才鎖定,在那之前
        # 傳進來的是 None。所以這段不能放在上面的「第一幀」分支裡,否則整份
        # manifest 會永遠沒有 table_rect。第一個拿到的值就定案,後面不再改
        # —— 校正一旦鎖定就不會變,真的變了(視窗縮放)是另一場錄製的事。
        if table_rect is not None and self.manifest.table_rect is None:
            self.manifest.table_rect = [
                table_rect.x,
                table_rect.y,
                table_rect.width,
                table_rect.height,
            ]

        index = self.manifest.frame_count
        timestamp = frame.captured_at - self._start_time

        thumb = _thumbnail(frame.image)
        changed = self._previous_thumb is None or (
            float(np.abs(thumb.astype(np.int16) - self._previous_thumb.astype(np.int16)).mean())
            >= self.change_threshold
        )

        if changed:
            name = f"{index:06d}.{self.image_format}"
            cv2.imwrite(str(self.frames_dir / name), frame.image, self._write_params)
            self._previous_thumb = thumb
            self._last_image_name = name
            self.manifest.stored_frames += 1
        else:
            name = self._last_image_name

        record = FrameRecord(
            index=index, timestamp=timestamp, image=name, changed=changed, scale=frame.scale
        )
        self.manifest.frames.append(record)
        self.manifest.frame_count += 1
        self.manifest.duration = timestamp
        return record

    def add_event(self, event: dict[str, Any]) -> None:
        """附加一筆 Ground Truth 事件(JSON Lines)。

        事件必須自帶 ``timestamp``(同樣以 session 開始為原點),對齊畫面靠它。
        """
        if self._closed:
            raise RuntimeError("session 已關閉")
        if self._events_file is None:
            self._events_file = (self.path / EVENTS_NAME).open("a", encoding="utf-8")
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def elapsed(self, captured_at: float) -> float:
        """把單調時鐘換算成相對於 session 開始的秒數。"""
        return captured_at - self._start_time if self._start_time is not None else 0.0

    def close(self) -> None:
        if self._closed:
            return
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None
        with (self.path / MANIFEST_NAME).open("w", encoding="utf-8") as fh:
            json.dump(self.manifest.to_json(), fh, ensure_ascii=False, indent=2)
        self._closed = True
        logger.info(
            "session {} 已寫入 {}:{} 幀(實際存檔 {} 張),{:.1f} 秒",
            self.session_id,
            self.path,
            self.manifest.frame_count,
            self.manifest.stored_frames,
            self.manifest.duration,
        )

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SessionReader:
    """讀回已錄製的 session,用於離線開發與迴歸測試。

    :meth:`frames` 產出的是與線上擷取完全相同的 :class:`Frame` 物件,
    所以視覺管線的程式碼不需要知道自己吃的是即時畫面還是錄影。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"不是有效的 session 目錄(缺少 {MANIFEST_NAME}): {self.path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            self.manifest = SessionManifest.from_json(json.load(fh))
        self._cache: tuple[str, np.ndarray] | None = None

    @property
    def table_rect(self) -> Rect | None:
        if self.manifest.table_rect is None:
            return None
        return Rect(*self.manifest.table_rect)

    @property
    def can_align(self) -> bool:
        """是否具備與 GT 錄影檔對齊所需的牆上時鐘錨點。

        format_version 1 的早期 session 沒有 ``first_frame_wall``,只能靠人工
        比對畫面來對齊,無法自動化。
        """
        return self.manifest.first_frame_wall > 0

    def wall_at(self, timestamp: float) -> float:
        """把 session 內的相對秒數換算成牆上時鐘。

        換算後即可直接與 GT 錄影檔(``ws.jsonl``)每一行的 ``wall`` 欄位比較 ——
        那是兩個獨立行程之間唯一的共同時間基準。

        Raises:
            ValueError: session 缺少錨點(見 :attr:`can_align`)。
        """
        if not self.can_align:
            raise ValueError(
                f"session {self.manifest.session_id} 沒有 first_frame_wall,無法自動對齊 GT。"
                "這是 2026-07-27 之前錄製的舊格式。"
            )
        return self.manifest.first_frame_wall + timestamp

    def __len__(self) -> int:
        return self.manifest.frame_count

    def load_image(self, record: FrameRecord) -> np.ndarray:
        """讀取某一幀的影像。連續多幀共用同一檔案時會沿用快取,不重複解碼。"""
        if self._cache is not None and self._cache[0] == record.image:
            return self._cache[1]
        image = cv2.imread(str(self.path / FRAMES_DIR / record.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"讀不到影像 {record.image}(session: {self.path})")
        self._cache = (record.image, image)
        return image

    def frames(self, *, changed_only: bool = False) -> Iterator[Frame]:
        """依序產出 :class:`Frame`。

        Args:
            changed_only: 只產出畫面有變化的幀。開發 CV 時通常用這個,
                因為重複的靜止畫面沒有額外資訊。
        """
        window = self.manifest.window
        for record in self.manifest.frames:
            if changed_only and not record.changed:
                continue
            yield Frame(
                image=self.load_image(record),
                window=window,
                scale=record.scale,
                captured_at=record.timestamp,
            )

    def events(self) -> Iterator[dict[str, Any]]:
        """依序產出 Ground Truth 事件。沒有 events.jsonl 時產出空序列。"""
        events_path = self.path / EVENTS_NAME
        if not events_path.is_file():
            return
        with events_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __repr__(self) -> str:
        m = self.manifest
        return (
            f"<SessionReader {m.session_id} {m.frame_count} 幀"
            f"(存檔 {m.stored_frames}){m.duration:.1f}s>"
        )
