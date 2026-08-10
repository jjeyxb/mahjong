"""「AI 建議」分頁的內容。

三段:誰建議切什麼、各候選的 Q 值長條、其他引擎有沒有不同意見。

Q 值怎麼畫成長條
----------------
正規化在 :func:`~mia.ui.viewmodel.q_fraction`(Overlay 也用同一個),那裡寫著
為什麼不用固定刻度。

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

from mia.engine.actions import Candidate
from mia.ui.style import CAPTION, MUTED
from mia.ui.viewmodel import EngineView, ViewState, q_fraction
from mia.ui.widgets.tiles import TileIcons, TileLabel

__all__ = ["AdviceTab"]

#: 候選清單最多列幾個。合法動作可能有十幾個,但排到第八個之後幾乎不看。
MAX_CANDIDATES = 8

_BAR_RANGE = 1000

#: 大字那個動詞。牌交給圖去講,所以這一行只有動詞。
_VERB = "font-size: 26px; font-weight: 600;"


#: 「自己要拿出來的牌」最多幾張。吃碰是 2、大明槓是 3。
_MAX_OWN_TILES = 3


class _Headline(QWidget):
    """最上面那一行:大字 + 牌面圖。

    牌面圖是**一排**而不是一張。原本只畫一張,於是:

    * ``吃 3m`` 只看到 3m —— 而 3m 可以用 1m2m、2m4m 或 4m5m 去吃,
      使用者根本不知道該點哪兩張。
    * ``立直`` 什麼牌都沒有 —— 而按下立直的下一秒就要選一張打出去。

    兩個都是實機打一場才發現的。
    """

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._verb = QLabel("—", self)
        self._verb.setStyleSheet(_VERB)
        #: 動作在講的那張。鳴牌時是**別家打出來的**那張,不在自己手上。
        self._subject = TileLabel(icons, 64, self)
        #: 分隔詞。它是唯一讓「桌上那張」與「自己手上那幾張」分得開的東西
        #: —— 三張一樣大小排在一起,使用者看不出該點哪幾張。
        self._joiner = QLabel("", self)
        self._joiner.setStyleSheet(MUTED + "font-size: 15px;")
        #: 自己要拿出來的那幾張。
        self._own = [TileLabel(icons, 64, self) for _ in range(_MAX_OWN_TILES)]
        self._detail = QLabel("", self)
        self._detail.setStyleSheet(MUTED)

        layout.addWidget(self._verb)
        layout.addWidget(self._subject)
        layout.addWidget(self._joiner)
        for label in self._own:
            layout.addWidget(label)
        layout.addWidget(self._detail)
        layout.addStretch(1)

    def _show_tiles(
        self, subject: str | None = None, *, joiner: str = "", own: tuple[str, ...] = ()
    ) -> None:
        """畫出這一手的牌。沒用到的位子要藏起來,不然會留上一手的殘影。"""
        if subject:
            self._subject.set_tile(subject)
        self._subject.setVisible(bool(subject))
        self._joiner.setText(joiner)
        self._joiner.setVisible(bool(joiner))
        own = own[:_MAX_OWN_TILES]
        for label, tile in zip(self._own, own, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._own[len(own) :]:
            label.setVisible(False)

    def update_from(
        self, engine: EngineView | None, *, stale: bool = False, enabled: bool = True
    ) -> None:
        if not enabled:
            # 「等待引擎…」在功能關著的時候是錯的:那句話讓人以為它正在載入,
            # 於是使用者就一直等下去。關著就要說關著,並指出開關在哪。
            self._verb.setText("未開啟")
            self._show_tiles()
            self._detail.setText("用右上角的開關打開")
            self._verb.setStyleSheet(_VERB + MUTED)
            return
        if engine is None:
            self._verb.setText("等待引擎…")
            self._show_tiles()
            self._detail.setText("")
            return
        top = engine.candidates[0] if engine.candidates else None

        if engine.action is None and not engine.declined:
            # 真的沒人在問。這是九成的事件,保持安靜。
            self._verb.setText(engine.verb)
            self._show_tiles()
            self._detail.setText(f"{engine.name}")
            self._verb.setStyleSheet(_VERB + MUTED)
            return

        # 大字是動詞,牌交給圖去講 —— 文字與圖同時寫一次會又長又重複。
        # 「切什麼 / 吃要用哪兩張 / 立直之後切哪張」由 EngineView 決定,
        # Overlay 用的是同一組屬性,兩邊不會講得不一樣。
        #
        # 「跳過」也走這條路:遊戲在問「碰 / 槓 / 跳過」而引擎說不鳴,**那是一個
        # 答案**,底下的候選清單會列出鳴牌各要多少 Q,使用者才看得出差多少。
        self._verb.setText(engine.verb)
        self._show_tiles(engine.subject, joiner=engine.joiner, own=engine.own_tiles)

        detail = f"{engine.name}"
        if top is not None:
            detail += f"   Q={top.q:+.2f}"
        detail += f"   {engine.latency_ms:.0f} ms"
        if stale:
            # 建議的牌已經不在手上。標出來而不是藏起來 —— 使用者看到「切 3s」
            # 而手上沒有 3s 時,要知道那是剛剛打掉了,不是程式算錯。
            detail += "   ·已打出"
        self._detail.setText(detail)
        self._verb.setStyleSheet(_VERB + (MUTED if stale else ""))


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
        self._bar.setValue(round(q_fraction(candidate.q, low=low, high=high) * _BAR_RANGE))
        weight = "600" if candidate.chosen else "400"
        self._value.setStyleSheet(f"font-weight: {weight};")


class AdviceTab(QWidget):
    """AI 建議分頁。"""

    def __init__(self, icons: TileIcons, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icons = icons
        self._rows: list[_CandidateRow] = []
        #: 候選清單最多列幾個。可以在設定列調 —— 合法動作可能十幾個,
        #: 但排到第八個之後幾乎不看,而列太多會把手牌那段推出畫面。
        self.max_candidates = MAX_CANDIDATES

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._headline = _Headline(icons, self)
        layout.addWidget(self._headline)

        # 分隔線要**留住參照**,才能跟著它底下那一區一起隱藏。寫死加進版面的話,
        # 功能關著時會剩下兩條橫線橫在一片空白上,看起來像東西沒載出來。
        self._candidates_rule = _rule(self)
        layout.addWidget(self._candidates_rule)

        self._candidates_label = QLabel("其他候選", self)
        self._candidates_label.setStyleSheet(CAPTION)
        layout.addWidget(self._candidates_label)

        self._candidates = QWidget(self)
        self._candidates_layout = QVBoxLayout(self._candidates)
        self._candidates_layout.setContentsMargins(0, 0, 0, 0)
        self._candidates_layout.setSpacing(3)
        layout.addWidget(self._candidates)

        self._others_rule = _rule(self)
        layout.addWidget(self._others_rule)
        self._others = QWidget(self)
        self._others_layout = QGridLayout(self._others)
        self._others_layout.setContentsMargins(0, 0, 0, 0)
        self._others_layout.setSpacing(4)
        layout.addWidget(self._others)
        self._other_labels: list[tuple[QLabel, QLabel]] = []

        layout.addStretch(1)

    def update_from(self, state: ViewState, *, enabled: bool = True) -> None:
        primary = state.primary if enabled else None
        self._headline.update_from(primary, stale=state.advice_is_stale, enabled=enabled)
        self._update_candidates(primary)
        self._update_others(state if enabled else ViewState())

    def _update_candidates(self, engine: EngineView | None) -> None:
        candidates = engine.candidates[: self.max_candidates] if engine else ()
        # 沒有 Q 值的引擎(規則式 baseline)不該留著上一手的長條
        self._candidates_label.setVisible(bool(candidates))
        self._candidates_rule.setVisible(bool(candidates))

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
        self._others_rule.setVisible(bool(others))
        while len(self._other_labels) < len(others):
            index = len(self._other_labels)
            name = QLabel("", self._others)
            name.setStyleSheet(MUTED)
            value = QLabel("", self._others)
            self._others_layout.addWidget(name, index, 0)
            self._others_layout.addWidget(value, index, 1)
            self._other_labels.append((name, value))

        for (name, value), engine in zip(self._other_labels, others, strict=False):
            name.setText(engine.name)
            text = engine.headline
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
