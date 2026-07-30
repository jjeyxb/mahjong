"""牌面圖片的載入與顯示。

圖檔就是分類器用的那 37 張模板(``assets/tiles/<skin>/``,64×109),從雀魂官方
資源切出來的。一份素材兩個用途:比對時當模板,顯示時當圖示。這也讓 UI 上看到的
牌與辨識器認的牌**必然是同一套** —— 各自準備一份圖的話,換皮膚時會一邊對一邊錯。

記法
----
對外一律 MJAI 記法(``5mr`` / ``E``),但檔名是雀魂記法(``0m`` / ``1z``)。
轉換關在 :class:`TileIcons` 裡面,呼叫端不必知道磁碟上長什麼樣。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from majsoul_copilot.mjai.tiles import UNKNOWN, mjai_to_ms
from majsoul_copilot.utils.paths import ASSETS_DIR

__all__ = ["DEFAULT_SKIN", "TILE_ASPECT", "HandStrip", "TileIcons", "TileLabel"]

DEFAULT_SKIN = "mjpface_default"

#: 素材是 64×109。寬高比固定,縮放時只給高度就好。
TILE_ASPECT = 64 / 109


class TileIcons:
    """牌名 → QPixmap。

    Args:
        skin: ``assets/tiles/`` 底下的皮膚目錄名。

    Raises:
        FileNotFoundError: 皮膚目錄不存在。

    Note:
        縮放結果有快取。手牌 14 張 + 候選清單十幾張,每次狀態更新都重新縮放
        是實際會看得出來的卡頓 —— 尺寸固定時直接命中快取。
    """

    def __init__(self, skin: str = DEFAULT_SKIN) -> None:
        self.root = ASSETS_DIR / "tiles" / skin
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"找不到牌面素材 {self.root} —— 執行 tools/fetch_tiles.py 取得"
            )
        self._sources: dict[str, QPixmap] = {}
        self._scaled: dict[tuple[str, int], QPixmap] = {}

    def pixmap(self, tile: str, height: int) -> QPixmap:
        """某張牌縮放到指定高度。認不得的牌回一張空的 pixmap。

        回空的而不是拋例外:UI 在畫的時候不該因為一張牌認不出來就整個崩掉,
        使用者會看到一個空格,那本身就說明了問題。
        """
        key = (tile, height)
        cached = self._scaled.get(key)
        if cached is not None:
            return cached

        source = self._load(tile)
        scaled = (
            source
            if source.isNull()
            else source.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
        )
        self._scaled[key] = scaled
        return scaled

    def _load(self, tile: str) -> QPixmap:
        cached = self._sources.get(tile)
        if cached is not None:
            return cached

        pixmap = QPixmap()
        if tile != UNKNOWN:
            try:
                path: Path = self.root / f"{mjai_to_ms(tile)}.png"
            except ValueError:
                path = self.root / "__unknown__"
            if path.is_file():
                pixmap = QPixmap(str(path))
        self._sources[tile] = pixmap
        return pixmap


class TileLabel(QLabel):
    """單張牌。

    Args:
        icons: 圖示來源。
        height: 顯示高度(像素)。
    """

    def __init__(self, icons: TileIcons, height: int = 48, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icons = icons
        self._height = height
        self.setFixedSize(round(height * TILE_ASPECT), height)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_tile(self, tile: str) -> None:
        self.setPixmap(self._icons.pixmap(tile, self._height))
        self.setToolTip(tile)


class HandStrip(QWidget):
    """一排牌。摸進來那張與暗手牌之間空一格。

    Args:
        icons: 圖示來源。
        height: 牌的顯示高度。
        gap: 牌與牌之間的間距。
    """

    def __init__(
        self,
        icons: TileIcons,
        *,
        height: int = 48,
        gap: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icons = icons
        self._height = height
        self._labels: list[TileLabel] = []

        # 版面固定是 [暗手牌…] [間隔] [摸的那張] [伸縮]。摸牌用專屬的 widget
        # 而不是重複利用最後一個暗手牌槽 —— 那樣就沒地方放中間的間隔。
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(gap)

        self._gap = QWidget(self)
        self._gap.setFixedWidth(round(height * TILE_ASPECT / 2))
        self._drawn = TileLabel(icons, height, self)
        self._layout.addWidget(self._gap)
        self._layout.addWidget(self._drawn)
        self._layout.addStretch(1)
        self._gap.setVisible(False)
        self._drawn.setVisible(False)

    def set_tiles(self, tiles: tuple[str, ...], *, drawn: str | None = None) -> None:
        """換掉整排牌。

        Args:
            tiles: 暗手牌,已排序。
            drawn: 剛摸進來那張;會與前面空一格,對應畫面上的樣子。
        """
        while len(self._labels) < len(tiles):
            label = TileLabel(self._icons, self._height, self)
            # 插在間隔之前 —— 暗手牌永遠在最左邊那一段
            self._layout.insertWidget(len(self._labels), label)
            self._labels.append(label)

        for label, tile in zip(self._labels, tiles, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._labels[len(tiles) :]:
            label.setVisible(False)

        if drawn:
            self._drawn.set_tile(drawn)
        self._gap.setVisible(bool(drawn))
        self._drawn.setVisible(bool(drawn))
