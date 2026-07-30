"""側邊視窗:兩個分頁 + 一條狀態列。

分頁而不是把兩者塞在一頁:功能 1(向聽/進張)與功能 2(AI 建議)是**兩條獨立的
路** —— 前者不碰連線,後者不碰畫面。擠在一起會讓「只有一邊在運作」的狀態
很難表達,而那是實際使用時的常態(引擎還在載入、或封包還沒接上)。

視窗刻意做窄。它要放在遊戲旁邊,不是蓋在上面 —— 蓋上去是 Overlay(M7)的事。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from majsoul_copilot.ui.viewmodel import ViewModel, ViewState
from majsoul_copilot.ui.widgets.advice import AdviceTab
from majsoul_copilot.ui.widgets.analysis import AnalysisTab
from majsoul_copilot.ui.widgets.tiles import DEFAULT_SKIN, TileIcons

__all__ = ["PanelWindow"]

#: 放得下 14 張 44px 高的牌加一點邊界。
MIN_WIDTH = 380
MIN_HEIGHT = 520


class PanelWindow(QMainWindow):
    """輔助資訊的側邊視窗。

    Args:
        viewmodel: 狀態來源。視窗自己訂閱它,不主動去拉。
        skin: 牌面素材。

    Note:
        :meth:`apply` 是唯一的入口。ViewModel 的通知可能來自別的執行緒,
        呼叫端要負責把它切回主執行緒 —— Qt 的 widget 只能在建立它的執行緒上動。
        ``tools/ui.py`` 用 ``QTimer`` 做這件事。
    """

    def __init__(self, viewmodel: ViewModel, *, skin: str = DEFAULT_SKIN) -> None:
        super().__init__()
        self.setWindowTitle("Majsoul Copilot")
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        icons = TileIcons(skin)
        self._advice = AdviceTab(icons, self)
        self._analysis = AnalysisTab(icons, self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._advice, "AI 建議")
        self._tabs.addTab(self._analysis, "向聽分析")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)
        self.setCentralWidget(central)

        self._notice = QLabel("", self)
        self._notice.setStyleSheet("color: palette(mid);")
        status = QStatusBar(self)
        status.addWidget(self._notice, 1)
        self.setStatusBar(status)

        self._viewmodel = viewmodel
        self.apply(viewmodel.state)

    def apply(self, state: ViewState) -> None:
        """把狀態畫上去。

        兩個分頁都更新,不只是當前可見的那個 —— 切分頁時才更新會讓使用者看到
        一瞬間的舊資料。更新一次的成本是十幾個 label,遠低於那個閃動的代價。
        """
        self._advice.update_from(state)
        self._analysis.update_from(state)
        self._notice.setText(_status_text(state))


def _status_text(state: ViewState) -> str:
    parts: list[str] = []
    if state.hand_conflict:
        parts.append("⚠ 畫面與封包的手牌不一致")
    parts.extend(state.notices)
    if not parts:
        engines = "、".join(e.name for e in state.engines)
        return f"引擎:{engines}" if engines else "等待資料…"
    return "　|　".join(parts)


def _center_on_screen(window: QMainWindow) -> None:
    """把視窗放到螢幕右側 —— 遊戲通常在中間或左邊。"""
    screen = window.screen()
    if screen is None:
        return
    available = screen.availableGeometry()
    size = window.frameGeometry()
    window.move(
        available.right() - size.width() - 24,
        available.top() + max(0, (available.height() - size.height()) // 2),
    )
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
