"""從 assets/icon.svg 產生打包用的 App 圖示。

    python tools/make_icons.py

產物在 ``packaging/icons/``:``MIA.icns``(Mac)與 ``MIA.ico``(Windows)。兩個都進版控,
Windows 建置機不需要 Mac 的 ``iconutil``。改了 icon.svg 之後要重跑一次。

Mac 的圖示照 macOS 的格線留邊:1024 的畫布裡圓角方塊只佔中間 824,不然在 Dock 上
會比其他 App 大一圈。Windows 的圖示不留邊,工作列與捷徑都是貼滿的。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "icon.svg"
OUT = ROOT / "packaging" / "icons"

#: macOS 圖示格線:1024 的畫布,圖形 824,四周各留 100。
_MAC_CONTENT = 824 / 1024
_ICONSET = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}
_ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def render(renderer: QSvgRenderer, size: int, content: float = 1.0) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    side = size * content
    margin = (size - side) / 2
    renderer.render(painter, QRectF(margin, margin, side, side))
    painter.end()
    return image


def main() -> int:
    app = QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(SOURCE))
    if not renderer.isValid():
        raise SystemExit(f"讀不了 {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        iconset = folder / "MIA.iconset"
        iconset.mkdir()
        for name, size in _ICONSET.items():
            render(renderer, size, _MAC_CONTENT).save(str(iconset / name))
        if sys.platform == "darwin":
            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(OUT / "MIA.icns")], check=True
            )
        else:
            print("不是 Mac,跳過 MIA.icns")
        big = folder / "windows.png"
        render(renderer, 256).save(str(big))
        Image.open(big).save(OUT / "MIA.ico", sizes=_ICO_SIZES)
    del app
    print(f"已產生 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
