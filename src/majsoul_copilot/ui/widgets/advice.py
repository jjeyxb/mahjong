"""「AI 建議」分頁的內容。

三段:誰建議切什麼、各候選的 Q 值長條、其他引擎有沒有不同意見。

Q 值怎麼畫成長條
----------------
Q 值是**相對的** —— 同一手之內互相比較才有意義,不同局面之間的絕對值不可比
(實測見過 +2.7 也見過 -6.8)。所以長條不用固定刻度,每次都以這一手的
最高/最低值重新正規化。固定刻度的話,大部分局面的長條會全部擠在一端。

負值不畫成反向長條。使用者要看的是「哪個選項比較好」,不是「Q 值的正負」
—— 全部選項都是負分很常見(場況不好),那時反向長條會讓整個畫面看起來像出錯了。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from majsoul_copilot.engine.actions import Candidate
from majsoul_copilot.ui.viewmodel import EngineView, ViewState
from majsoul_copilot.ui.widgets.tiles import TileIcons, TileLabel

__all__ = ["AdviceTab"]

#: 候選清單最多列幾個。合法動作可能有十幾個,但排到第八個之後幾乎不看。
MAX_CANDIDATES = 8

_BAR_RANGE = 1000


class _Headline(QWidget):
    """最上面那一行:建議切哪張,大字加牌面圖。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._verb = QLabel("—", self)
        self._verb.setStyleSheet("font-size: 26px; font-weight: 600;")
        self._tile = TileLabel(icons, 64, self)
        self._detail = QLabel("", self)
        self._detail.setStyleSheet("color: palette(mid);")

        layout.addWidget(self._verb)
        layout.addWidget(self._tile)
        layout.addWidget(self._detail)
        layout.addStretch(1)

    def update_from(self, engine: EngineView | None) -> None:
        if engine is None:
            self._verb.setText("等待引擎…")
            self._tile.setVisible(False)
            self._detail.setText("")
            return
        if engine.action is None:
            self._verb.setText("不需要動作")
            self._tile.setVisible(False)
            self._detail.setText(f"{engine.name}")
            return

        top = engine.candidates[0] if engine.candidates else None
        if engine.tile:
            self._verb.setText("切")
            self._tile.set_tile(engine.tile)
            self._tile.setVisible(True)
        else:
            # 立直、吃碰這種沒有對應的單張牌,直接把動作寫成大字
            self._verb.setText(top.display if top else engine.action)
            self._tile.setVisible(False)

        detail = f"{engine.name}"
        if top is not None:
            detail += f"   Q={top.q:+.2f}"
        detail += f"   {engine.latency_ms:.0f} ms"
        self._detail.setText(detail)


class _CandidateRow(QWidget):
    """候選清單的一列:牌面圖 / 名稱 / Q 值 / 長條。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tile = TileLabel(icons, 30, self)
        self._name = QLabel("", self)
        self._name.setMinimumWidth(48)
        self._value = QLabel("", self)
        self._value.setMinimumWidth(56)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, _BAR_RANGE)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self._tile)
        layout.addWidget(self._name)
        layout.addWidget(self._value)
        layout.addWidget(self._bar, 1)

    def update_from(self, candidate: Candidate, *, low: float, high: float) -> None:
        if candidate.is_tile:
            self._tile.set_tile(candidate.label)
            self._tile.setVisible(True)
            self._name.setText("")
        else:
            self._tile.setVisible(False)
            self._name.setText(candidate.display)

        self._value.setText(f"{candidate.q:+.2f}")
        span = high - low
        # 只有一個候選時 span 為 0 —— 畫滿而不是除以零
        fraction = 1.0 if span <= 0 else (candidate.q - low) / span
        self._bar.setValue(round(fraction * _BAR_RANGE))
        weight = "600" if candidate.chosen else "400"
        self._value.setStyleSheet(f"font-weight: {weight};")


class AdviceTab(QWidget):
    """AI 建議分頁。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icons = icons
        self._rows: list[_CandidateRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._headline = _Headline(icons, self)
        layout.addWidget(self._headline)
        layout.addWidget(_rule(self))

        self._candidates_label = QLabel("其他候選", self)
        self._candidates_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self._candidates_label)

        self._candidates = QWidget(self)
        self._candidates_layout = QVBoxLayout(self._candidates)
        self._candidates_layout.setContentsMargins(0, 0, 0, 0)
        self._candidates_layout.setSpacing(3)
        layout.addWidget(self._candidates)

        layout.addWidget(_rule(self))
        self._others = QWidget(self)
        self._others_layout = QGridLayout(self._others)
        self._others_layout.setContentsMargins(0, 0, 0, 0)
        self._others_layout.setSpacing(4)
        layout.addWidget(self._others)
        self._other_labels: list[tuple[QLabel, QLabel]] = []

        layout.addStretch(1)

    def update_from(self, state: ViewState) -> None:
        primary = state.primary
        self._headline.update_from(primary)
        self._update_candidates(primary)
        self._update_others(state)

    def _update_candidates(self, engine: EngineView | None) -> None:
        candidates = engine.candidates[:MAX_CANDIDATES] if engine else ()
        # 沒有 Q 值的引擎(規則式 baseline)不該留著上一手的長條
        self._candidates_label.setVisible(bool(candidates))

        while len(self._rows) < len(candidates):
            row = _CandidateRow(self._icons, self._candidates)
            self._candidates_layout.addWidget(row)
            self._rows.append(row)

        if candidates:
            values = [c.q for c in engine.candidates] if engine else [0.0]
            low, high = min(values), max(values)
            for row, candidate in zip(self._rows, candidates, strict=False):
                row.update_from(candidate, low=low, high=high)
        for row in self._rows[len(candidates) :]:
            row.setVisible(False)
        for row in self._rows[: len(candidates)]:
            row.setVisible(True)

    def _update_others(self, state: ViewState) -> None:
        primary = state.primary
        others = [e for e in state.engines if e is not primary]
        while len(self._other_labels) < len(others):
            index = len(self._other_labels)
            name = QLabel("", self._others)
            name.setStyleSheet("color: palette(mid);")
            value = QLabel("", self._others)
            self._others_layout.addWidget(name, index, 0)
            self._others_layout.addWidget(value, index, 1)
            self._other_labels.append((name, value))

        for (name, value), engine in zip(self._other_labels, others, strict=False):
            name.setText(engine.name)
            text = engine.action or "不需要動作"
            if not state.is_unanimous and engine.action is not None:
                text += "   ⚠ 分歧"
            value.setText(text)
            name.setVisible(True)
            value.setVisible(True)
        for name, value in self._other_labels[len(others) :]:
            name.setVisible(False)
            value.setVisible(False)


def _rule(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
