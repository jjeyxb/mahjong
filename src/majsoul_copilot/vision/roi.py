"""ROI 解析:把設定檔裡的正規化矩形換算成當下畫面的像素位置。

這是設定與影像之間唯一的橋。上層(hand / meld / river 等辨識模組)只跟
:class:`RoiSet` 要「某個區域的影像」,不需要知道座標是怎麼換算的,也不需要
碰到 :class:`~majsoul_copilot.calibration.Calibration`。

為什麼不直接在各辨識模組裡算座標
--------------------------------
因為換算過程有兩個容易錯的地方,集中在一處才好驗證:

1. ROI 的正規化座標是**相對牌桌矩形**,不是相對整張擷取影像。牌桌矩形本身
   在影像中有偏移(視窗裝飾、黑邊),換算時必須加回去 ——
   :meth:`NormRect.to_pixels` 收 :class:`Rect` 時會處理,收 :class:`Size`
   時不會,傳錯型別就會整片歪掉。
2. 沒量測的 ROI 是 ``None``。這種情況必須**明確炸掉**,不能給個預設框硬跑
   —— 拿錯的框去辨識只會讓準確率不明不白地變差,事後極難追查。

ROI 名稱
--------
用點號表示巢狀,與 ``config/default.yaml`` 和 ROI 標註工具一致:
``own_hand``、``own.river``、``kamicha.melds`` ……。名稱是從
:class:`~majsoul_copilot.config.models.RoiConfig` 的欄位**動態導出**的,
不是寫死的清單 —— 之後往設定模型加區域,這裡自動跟著有,不會漏。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from majsoul_copilot.calibration.table import Calibration
from majsoul_copilot.config.models import RiverGridConfig, RoiConfig, SeatRoiConfig
from majsoul_copilot.utils.geometry import NormQuad, NormRect, Rect
from majsoul_copilot.vision.grid import Grid

__all__ = ["MissingRoiError", "Roi", "RoiSet", "roi_names"]


class MissingRoiError(KeyError):
    """要用的 ROI 在設定檔裡還是 ``null``(尚未量測)。"""


@dataclass(frozen=True, slots=True)
class Roi:
    """一個已解析的區域。

    Attributes:
        name: 點號分隔的 ROI 名稱,例如 ``kamicha.melds``。
        norm: 設定檔裡的正規化座標(相對牌桌矩形)。
        rect: 換算後的像素矩形,**相對整張擷取影像**,可直接拿去切片。
    """

    name: str
    norm: NormRect
    rect: Rect

    def crop(self, image: np.ndarray) -> np.ndarray:
        """從整張擷取影像切出這個區域(回傳 view,非複本)。"""
        return image[self.rect.as_slice()]

    def __str__(self) -> str:
        return f"{self.name}: {self.rect} <- {self.norm}"


def _iter_fields(config: RoiConfig) -> Iterator[tuple[str, tuple[float, ...] | None]]:
    """走訪 RoiConfig 裡所有**矩形**欄位,巢狀的展開成點號名稱。

    牌河網格(``river_grid``)存的是四邊形不是矩形,型別不同,由
    :attr:`RoiSet.grids` 另外處理 —— 混在一起的話 ``NormRect(*value)`` 會拿到
    四個 (x, y) 對而不是四個數字。
    """
    for name in type(config).model_fields:
        value = getattr(config, name)
        if isinstance(value, SeatRoiConfig):
            for sub in type(value).model_fields:
                yield f"{name}.{sub}", getattr(value, sub)
        elif not isinstance(value, RiverGridConfig):
            yield name, value


def roi_names() -> tuple[str, ...]:
    """所有 ROI 名稱,順序與設定模型的欄位順序一致。"""
    return tuple(name for name, _ in _iter_fields(RoiConfig()))


class RoiSet:
    """某一次校正下,所有 ROI 的像素位置。

    校正結果一變(使用者縮放視窗、切全螢幕)就要重建 —— 用
    :meth:`matches` 檢查,不要沿用舊的。
    """

    def __init__(self, config: RoiConfig, calibration: Calibration) -> None:
        self.calibration = calibration
        self._rois: dict[str, Roi] = {}
        self._missing: list[str] = []

        for name, value in _iter_fields(config):
            if value is None:
                self._missing.append(name)
                continue
            norm = NormRect(*value)
            self._rois[name] = Roi(name, norm, calibration.roi_to_pixels(norm))

        self._grids: dict[str, Grid] = {}
        for seat in type(config.river_grid).model_fields:
            quad = getattr(config.river_grid, seat)
            if quad is None:
                self._missing.append(f"river_grid.{seat}")
                continue
            self._grids[seat] = Grid.from_norm(
                NormQuad(tuple(quad)), calibration.table_rect
            )

    # ------------------------------------------------------------------ 查詢

    @property
    def missing(self) -> tuple[str, ...]:
        """尚未量測(設定檔為 null)的 ROI 名稱。"""
        return tuple(self._missing)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(self._rois)

    def __contains__(self, name: str) -> bool:
        return name in self._rois

    def __len__(self) -> int:
        return len(self._rois)

    def __iter__(self) -> Iterator[Roi]:
        return iter(self._rois.values())

    def get(self, name: str) -> Roi | None:
        return self._rois.get(name)

    @property
    def grids(self) -> dict[str, Grid]:
        """四家牌河的網格,鍵是 ``own`` / ``kamicha`` / ``toimen`` / ``shimocha``。"""
        return dict(self._grids)

    def grid(self, seat: str) -> Grid:
        """取某一家的牌河網格;沒標定過就拋錯,不給預設值硬跑。"""
        grid = self._grids.get(seat)
        if grid is None:
            raise MissingRoiError(
                f"{seat} 的牌河網格還沒標定。請用 tools/grid_annotate.py 拉四個角,"
                "填進 config/default.yaml 的 roi.river_grid 區塊。"
            )
        return grid

    def __getitem__(self, name: str) -> Roi:
        """取一個 ROI;沒量測或名稱打錯都會拋出可讀的錯誤。"""
        roi = self._rois.get(name)
        if roi is not None:
            return roi
        if name in self._missing:
            raise MissingRoiError(
                f"ROI {name!r} 在設定檔裡還是 null(尚未量測)。"
                "請用 tools/roi_annotate.py 量測後填進 config/default.yaml 的 roi: 區塊。"
            )
        raise MissingRoiError(
            f"沒有名為 {name!r} 的 ROI。可用的有:{', '.join(roi_names())}"
        )

    # ------------------------------------------------------------------ 使用

    def crop(self, image: np.ndarray, name: str) -> np.ndarray:
        """切出指定區域的影像(回傳 view,非複本)。

        傳進來的必須是**整張擷取影像**,不是已經裁過的牌桌影像 ——
        ROI 的像素座標含牌桌矩形本身的偏移。
        """
        if image.shape[:2] != (self.calibration.image_size.height,
                               self.calibration.image_size.width):
            raise ValueError(
                f"影像尺寸 {image.shape[1]}x{image.shape[0]} 與校正當下的 "
                f"{self.calibration.image_size} 不符 —— 這份 RoiSet 不適用於這張影像。"
                "視窗大小變了就要重新校正並重建 RoiSet。"
            )
        return self[name].crop(image)

    def matches(self, calibration: Calibration) -> bool:
        """這份 RoiSet 是否仍適用於給定的校正結果。"""
        return calibration.table_rect == self.calibration.table_rect

    def __str__(self) -> str:
        tail = f",{len(self._missing)} 個未量測" if self._missing else ""
        return f"RoiSet({len(self._rois)} 個區域{tail})"
