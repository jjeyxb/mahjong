"""顏色在**兩種**外觀模式下都要讀得到。

這一組測試存在的原因是實機踩到的:次要文字全部寫 ``color: palette(mid)``,
看起來很對(沒有寫死任何顏色),但 ``Mid`` 是給 3D 邊框陰影用的角色。
macOS 深色模式下它是 (21,21,21),而視窗背景是 (30,30,30) —— 字比背景還暗,
畫面上幾乎讀不到。淺色模式下剛好相反,所以開發時完全看不出來。

**用了 palette 不等於跟著模式走**,要看用的是哪一個角色。

樣式表裡的 ``palette(...)`` 是對著**應用程式** palette 解析的,所以這裡換的
是 ``QApplication`` 那一份,不是單一 widget 的 —— 換錯了測到的會是預設值,
而預設值兩種情況都一樣,測試就永遠會過。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLabel

from mia.ui.style import CAPTION, MUTED, SMALL, off_track
from mia.ui.widgets.toggle import ON_COLOR

#: 實測 macOS 的兩套底色(PySide6 6.11.1)。
DARK = ((30, 30, 30), (255, 255, 255))
LIGHT = ((239, 239, 239), (0, 0, 0))
SCHEMES = [("深色", DARK), ("淺色", LIGHT)]

#: 次要文字與背景至少要差這麼多亮度(0~255)。
#:
#: 這不是 WCAG —— 次要文字本來就該比主文字淡,套 4.5:1 會逼著它跟主文字一樣
#: 黑。門檻取在「原本那個壞掉的組合」(21 對 30,差 9)與「現在這個」之間。
MIN_CONTRAST = 40


def make_palette(window: tuple[int, int, int], text: tuple[int, int, int]) -> QPalette:
    """做一套指定底色的 palette,含 Qt 由文字色推出來的 PlaceholderText。"""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(*window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(*text))
    faded = QColor(*text)
    faded.setAlpha(140)  # Qt 對 PlaceholderText 用的比例
    palette.setColor(QPalette.ColorRole.PlaceholderText, faded)
    # 壞掉的那個角色:深色模式下它比背景還暗
    palette.setColor(QPalette.ColorRole.Mid, QColor(21, 21, 21))
    return palette


@pytest.fixture
def scheme(request: pytest.FixtureRequest, qapp: QApplication) -> Iterator[tuple]:
    """把應用程式 palette 換成指定的那一套,測完換回來。"""
    window, text = request.param
    original = qapp.palette()
    qapp.setPalette(make_palette(window, text))
    yield request.param
    qapp.setPalette(original)


def over(front: QColor, back: QColor) -> QColor:
    """把帶 alpha 的前景合成到背景上 —— 眼睛看到的是這個顏色,不是前景本身。"""
    ratio = front.alphaF()
    channels = (
        (front.red(), back.red()),
        (front.green(), back.green()),
        (front.blue(), back.blue()),
    )
    return QColor(*(round(b + (f - b) * ratio) for f, b in channels))


def luma(color: QColor) -> float:
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()


def rendered_text_color(css: str, window: tuple[int, int, int]) -> QColor:
    """套上樣式之後,這個 label 的文字實際會是什麼顏色(已合成到背景上)。"""
    label = QLabel("x")
    label.setStyleSheet(css)
    label.ensurePolished()
    drawn = label.palette().color(QPalette.ColorRole.WindowText)
    return over(drawn, QColor(*window))


@pytest.mark.parametrize("scheme", [s for _, s in SCHEMES], indirect=True,
                         ids=[n for n, _ in SCHEMES])
class TestMutedText:
    def test_it_is_readable_against_the_background(self, scheme: tuple) -> None:
        window, _ = scheme
        shown = rendered_text_color(MUTED, window)
        gap = abs(luma(shown) - luma(QColor(*window)))
        assert gap >= MIN_CONTRAST, f"次要文字與背景只差 {gap:.0f}"

    def test_it_is_dimmer_than_the_primary_text(self, scheme: tuple) -> None:
        """次要文字要看得出來是次要的,不然就沒有層次可言。"""
        window, text = scheme
        shown = rendered_text_color(MUTED, window)
        background, primary = luma(QColor(*window)), luma(QColor(*text))
        assert abs(luma(shown) - background) < abs(primary - background)


@pytest.mark.parametrize("scheme", [DARK], indirect=True, ids=["深色"])
class TestTheOldBug:
    def test_the_old_role_really_was_broken(self, scheme: tuple) -> None:
        """釘住當初的症狀 —— 沒有這一條,上面那個門檻看起來像是憑空訂的。"""
        window, _ = scheme
        shown = rendered_text_color("color: palette(mid);", window)
        assert abs(luma(shown) - luma(QColor(*window))) < MIN_CONTRAST


class TestCaption:
    def test_the_caption_style_is_the_muted_style_plus_small(self) -> None:
        assert MUTED in CAPTION
        assert SMALL in CAPTION


@pytest.mark.parametrize(("name", "raw"), SCHEMES)
class TestOffTrack:
    """關著的開關也要看得出那裡有一個可以點的東西。``Mid`` 在深色模式下
    比背景還暗,整個開關會消失。"""

    def test_the_switch_is_visible_when_it_is_off(self, name: str, raw: tuple) -> None:
        track = off_track(make_palette(*raw))
        assert abs(luma(track) - luma(QColor(*raw[0]))) >= MIN_CONTRAST, name

    def test_off_stays_grey(self, name: str, raw: tuple) -> None:
        """「開」是 Apple 的綠,那是這個開關的識別。關的顏色必須是無彩的,
        否則兩個狀態靠顏色就分不出來。"""
        track = off_track(make_palette(*raw))
        assert track.saturation() == 0, name
        assert ON_COLOR.saturation() > 100

    def test_it_is_opaque(self, name: str, raw: tuple) -> None:
        """軌道還要與綠色做逐分量線性混色,帶 alpha 進去混出來的中間色不對。"""
        assert off_track(make_palette(*raw)).alpha() == 255, name
