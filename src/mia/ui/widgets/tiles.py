"""牌面圖片的載入與顯示。

圖檔就是分類器用的那 37 張模板(``assets/tiles/<skin>/``,64×109),從雀魂官方
資源切出來的。一份素材兩個用途:比對時當模板,顯示時當圖示。這也讓 UI 上看到的
牌與辨識器認的牌**必然是同一套** —— 各自準備一份圖的話,換皮膚時會一邊對一邊錯。

素材是模板,顯示要再修一次
--------------------------
同一份檔案不代表畫出來就該長一樣。模板是為了**比對**切的,它:

* 上面帶著一條約 4px 的灰邊 —— 那是遊戲畫面裡牌的上緣,不是牌面。比對時
  它每張都一樣,不影響結果;畫在 UI 上就是每張牌頭上一條髒灰條。
* 四個角是直角、沒有邊框。遊戲裡那是貼在 3D 牌桌上的一塊,周圍有光影;
  單獨放在一片灰底的視窗上,就是一塊浮不起來的方形色塊。

所以 :meth:`TileIcons.pixmap` 是**顯示用的管線**:切掉上緣、圓角、描一圈
極淡的框。素材檔本身一個位元都不動 —— 分類器走的是
:mod:`mia.vision.tiles.classify` 自己的載入路徑,兩邊互不相干。

Retina
------
縮放一律做到**裝置像素**(``devicePixelRatio``)。只縮到邏輯像素的話,
2× 螢幕上是先縮小再被系統放大兩倍 —— 畫面上就是一排糊掉的牌,而在 1× 螢幕
上測完全看不出來。素材高 109px,而最大的顯示高度是 64,所以 2× 時要 128px:
放大 17%,仍然遠好過先縮到 64 再被系統拉成 128。

記法
----
對外一律 MJAI 記法(``5mr`` / ``E``),但檔名是雀魂記法(``0m`` / ``1z``)。
轉換關在 :class:`TileIcons` 裡面,呼叫端不必知道磁碟上長什麼樣。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from mia.mjai.tiles import UNKNOWN, mjai_to_ms
from mia.utils.paths import ASSETS_DIR

__all__ = ["DEFAULT_SKIN", "TILE_ASPECT", "HandStrip", "TileIcons", "TileLabel"]

DEFAULT_SKIN = "mjpface_default"

#: 素材本身的寬高比(64×109)。
_SOURCE_ASPECT = 64 / 109

#: 素材最上面有多少**不是牌面** —— 那條灰邊是遊戲裡牌的上緣。
#: 寫成比例而不是「4 像素」:別的皮膚若解析度不同,像素數會跟著變,比例不會。
_EDGE_FRACTION = 4 / 109

#: 顯示用的寬高比。切掉上緣之後牌面變短,所以比素材本身寬一點。
#: 寬高比固定,縮放時只給高度就好。
TILE_ASPECT = _SOURCE_ASPECT / (1 - _EDGE_FRACTION)

#: 圓角半徑佔高度的比例。實測 0.06 看起來還是直角、0.11 已經像撲克牌,
#: 0.08 是「一眼認得出是麻將牌」的那個值。
_RADIUS_FRACTION = 0.08

#: 牌的外框。半透明的黑而不是 palette 的某個角色:它描的是**牌**的邊,
#: 而牌面永遠是米白的,不隨視窗的淺／深色改變。
_BORDER = QColor(0, 0, 0, 70)


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
        self._scaled: dict[tuple[str, int, float], QPixmap] = {}

    def pixmap(self, tile: str, height: int, ratio: float = 1.0) -> QPixmap:
        """某張牌,畫成指定高度。認不得的牌回一張空的 pixmap。

        回空的而不是拋例外:UI 在畫的時候不該因為一張牌認不出來就整個崩掉,
        使用者會看到一個空格,那本身就說明了問題。

        Args:
            tile: MJAI 記法。
            height: **邏輯**像素高度 —— 就是版面上佔的高度。
            ratio: 裝置像素比。實際畫出來的是 ``height * ratio`` 像素,
                然後掛上同樣的 ``devicePixelRatio`` 讓 Qt 知道該縮回去。
                理由見模組說明的「Retina」。
        """
        key = (tile, height, ratio)
        cached = self._scaled.get(key)
        if cached is not None:
            return cached

        face = self._face(tile)
        drawn = face if face.isNull() else _rounded(face, height, ratio)
        self._scaled[key] = drawn
        return drawn

    def _face(self, tile: str) -> QPixmap:
        """只有牌面的那一塊 —— 切掉素材上緣那條灰邊。"""
        source = self._load(tile)
        if source.isNull():
            return source
        edge = round(source.height() * _EDGE_FRACTION)
        return source.copy(0, edge, source.width(), source.height() - edge)

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


def _rounded(face: QPixmap, height: int, ratio: float) -> QPixmap:
    """把牌面縮到指定高度,切圓角,描一圈淡框。

    在**裝置像素**上畫,最後才掛 ``devicePixelRatio`` —— 圓角與框線都是
    1 像素等級的東西,用邏輯像素畫再讓系統放大會糊成兩三像素寬的灰邊。
    """
    width = round(height * TILE_ASPECT * ratio)
    tall = round(height * ratio)
    scaled = face.scaled(
        width,
        tall,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

    out = QPixmap(width, tall)
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 框線畫在半像素上。整數座標的話一條 1px 的線會跨在兩個像素之間,
    # 兩邊各拿一半的濃度 —— 看起來就是「框好像沒對齊」。
    inset = ratio / 2
    radius = height * _RADIUS_FRACTION * ratio
    path = QPainterPath()
    path.addRoundedRect(
        QRectF(inset, inset, width - 2 * inset, tall - 2 * inset), radius, radius
    )

    painter.setClipPath(path)
    painter.drawPixmap(0, 0, scaled)
    painter.setClipping(False)
    painter.setPen(QPen(_BORDER, ratio))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.end()

    out.setDevicePixelRatio(ratio)
    return out


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
        self._tile: str | None = None

    @property
    def tile(self) -> str | None:
        """現在畫的是哪一張。還沒設過就是 ``None``。

        記下來是因為 QPixmap 問不出它是哪張牌 —— 沒有這個屬性的話,
        「這一排牌面的順序對不對」只能靠眼睛看,測試與 debug 都接不上。
        """
        return self._tile

    def set_tile(self, tile: str) -> None:
        self._tile = tile
        # devicePixelRatioF() 在還沒 show 的 widget 上回主螢幕的值,而那正是
        # 絕大多數情況下的正確答案。搬到別的螢幕時 Qt 會送 DevicePixelRatioChange,
        # 見 :meth:`changeEvent`。
        self.setPixmap(self._icons.pixmap(tile, self._height, self.devicePixelRatioF()))
        self.setToolTip(tile)

    def changeEvent(self, event: QEvent) -> None:
        """拖到不同 DPI 的螢幕時重畫。

        不做的話,從 Retina 拖到外接螢幕(或反過來)之後這張牌會一直是用
        舊的比例算出來的 —— 一邊糊、一邊太小,而且要再動一次狀態才會修好。
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.DevicePixelRatioChange and self._tile is not None:
            self.set_tile(self._tile)


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
