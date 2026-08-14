"""放銃分析分頁。

顯示的是**還剩幾種待牌型**,不是機率 —— 理由見
:mod:`mia.analysis.danger` 的模組說明。這一頁的責任是把那份推論攤開來,
而且**不要把它壓扁**。

為什麼要逐家顯示
----------------
實測踩過的坑:一張牌對立直的那一家是現物,對另一家卻是全新的。只寫一個
「安全」標籤,使用者就會照著打出去 —— 那正是真實牌譜裡發生過的事
(見 ``docs/decisions.md`` 第十八節)。

所以每一列除了等級,還要寫**是對誰**、**憑什麼**。等級取最危險的那一家,
因為放銃只需要中一個人。

座位寫成上家 / 對家 / 下家
--------------------------
「對 3 家危險」要在腦裡換算一次才知道是誰。相對稱呼直接對得上牌桌。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mia.analysis import DangerLevel, DangerReport, TileDanger
from mia.ui.style import CAPTION, MUTED
from mia.ui.viewmodel import ViewState
from mia.ui.widgets.tiles import TileIcons, TileLabel

__all__ = ["AnalysisDangerTab"]

#: 最多列幾張。手牌 14 張全列會把整頁撐得很長,而使用者真正在挑的是最安全的
#: 那幾張與最危險的那幾張 —— 中間那些幾乎不看。
MAX_ROWS = 14

#: 等級的顏色。只有「安全」用綠色 —— 它是規則保證的,與其他三級不是同一種東西。
_LEVEL_COLORS = {
    DangerLevel.SAFE: "#34C759",
    DangerLevel.LIKELY_SAFE: "#8E8E93",
    DangerLevel.RISKY: "#FF9500",
    DangerLevel.VERY_RISKY: "#FF3B30",
}

#: 相對座位。索引是 ``(對方 - 自己) % 4``。
_RELATIVE = ("自己", "下家", "對家", "上家")


def seat_name(mine: int | None, theirs: int) -> str:
    """把絕對座位換成相對稱呼。不知道自己坐哪就退回絕對座位。"""
    if mine is None:
        return f"{theirs} 家"
    return _RELATIVE[(theirs - mine) % 4]


class _DangerRow(QWidget):
    """一列:牌面圖 + 等級 + 對誰、憑什麼。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._tile = TileLabel(icons, 30, self)
        self._level = QLabel("", self)
        self._level.setMinimumWidth(64)
        self._detail = QLabel("", self)
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(MUTED)
        layout.addWidget(self._tile)
        layout.addWidget(self._level)
        layout.addWidget(self._detail, 1)

    def update_from(self, danger: TileDanger, seat: int | None) -> None:
        # 牌名**已經是 MJAI 記法**了 —— 這一條路的資料來自 MJAI 事件流,
        # 不像向聽分析那邊是 classify 的雀魂記法。再轉一次會在字牌上炸掉
        # (``ms_to_mjai("S")`` 不認得),而數牌剛好轉得過去,所以只有摸到
        # 字牌時才會發現。
        self._tile.set_tile(danger.tile)
        colour = _LEVEL_COLORS[danger.level]
        self._level.setText(danger.level.label)
        self._level.setStyleSheet(f"color: {colour}; font-weight: 600;")
        self._detail.setText(_detail(danger, seat))


def _detail(danger: TileDanger, seat: int | None) -> str:
    """「對誰、憑什麼」那一句。

    三家都是現物時收成一句話 —— 那是最好的情況,不必把三行一樣的東西攤開。
    否則只寫最危險那一家:其餘的資訊量低,而這一列的寬度有限。
    """
    if all(s.furiten for s in danger.seats):
        return "三家都是現物"
    worst = max(danger.seats, key=lambda s: (s.level, len(s.waits), s.reach))
    who = seat_name(seat, worst.seat)
    if worst.reach:
        who += "(立直)"
    return f"對{who}:{worst.reason}"


class AnalysisDangerTab(QWidget):
    """放銃分析分頁。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icons = icons
        self._rows: list[_DangerRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._headline = QLabel("等待對局…", self)
        self._headline.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(self._headline)

        self._source = QLabel("", self)
        self._source.setStyleSheet(CAPTION)
        self._source.setWordWrap(True)
        layout.addWidget(self._source)

        self._rule = _rule(self)
        layout.addWidget(self._rule)
        self._caption = QLabel("每張牌切出去(安全的在前)", self)
        self._caption.setStyleSheet(CAPTION)
        layout.addWidget(self._caption)

        self._list = QWidget(self)
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(3)
        layout.addWidget(self._list)

        layout.addStretch(1)

    def update_from(self, state: ViewState, *, enabled: bool = True) -> None:
        report = state.dangers if enabled else None
        if not enabled:
            self._blank("未開啟", "用右上角的開關打開")
            return
        if report is None:
            self._blank("等待對局…", "這一頁要靠封包才能知道別家打了什麼")
            return
        self._show(report)

    # ------------------------------------------------------------------ 內部

    def _blank(self, headline: str, note: str) -> None:
        self._headline.setText(headline)
        self._source.setText(note)
        self._rule.setVisible(False)
        self._caption.setVisible(False)
        for row in self._rows:
            row.setVisible(False)

    def _show(self, report: DangerReport) -> None:
        if report.reached:
            who = "、".join(seat_name(report.seat, s) for s in report.reached)
            self._headline.setText(f"{who}立直")
            note = "以下是排除法的結果:還剩幾種待牌型沒被排除。"
        else:
            self._headline.setText("沒有人立直")
            # 這句話很重要。沒人立直**不等於安全** —— 實測那場三次放銃,
            # 和牌的人一個都沒立直。
            note = (
                "沒有人立直不代表安全 —— 沒立直的人一樣會榮和,"
                "只是我們無法確定他聽不聽牌。以下仍然是排除法的結果。"
            )
        self._source.setText(note)

        tiles = report.tiles[:MAX_ROWS]
        self._rule.setVisible(bool(tiles))
        self._caption.setVisible(bool(tiles))

        while len(self._rows) < len(tiles):
            row = _DangerRow(self._icons, self._list)
            self._list_layout.addWidget(row)
            self._rows.append(row)

        for row, danger in zip(self._rows, tiles, strict=False):
            row.update_from(danger, report.seat)
            row.setVisible(True)
        for row in self._rows[len(tiles) :]:
            row.setVisible(False)


def _rule(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
