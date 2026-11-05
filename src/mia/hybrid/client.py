"""One frame -> YOLO proposals -> template fusion -> one mutable event ledger."""

from __future__ import annotations

from typing import Any

import numpy as np

from mia.engine.base import EngineError
from mia.hybrid.fusion import Fusion, TemplateReader, annotate
from mia.vision.tiles.classify import DEFAULT_SKIN
from mia.yolo.client import YoloClient


class HybridClient(YoloClient):
    def __init__(self, *, skin: str = DEFAULT_SKIN) -> None:
        super().__init__()
        self._skin = skin
        self._fusion: Fusion | None = None
        self._hand_id: str | None = None
        self.last_fusion: dict[str, Any] = {}

    def start(self) -> None:
        super().start()
        if not self.supports_observations:
            raise EngineError("YOLO 程序版本不支援混合辨識")
        self._fusion = Fusion(TemplateReader(self._skin))

    def feed(self, image: np.ndarray, wall: float, *, raw: bool = False) -> dict[str, Any]:
        if raw:
            raise ValueError("混合模式只輸出合併後的事件快照")
        if self._fusion is None:
            raise EngineError("混合辨識尚未啟動")
        observation = super().feed(image, wall, raw=True)
        fused = self._fusion.apply(image, observation)
        payload = super().track(fused)
        self.last_fusion = fused["fusion"]
        annotate(payload, self.last_fusion)
        hand_id = payload.get("hand_id")
        if hand_id != self._hand_id:
            self._fusion.reset()
            self._hand_id = hand_id
        return payload
