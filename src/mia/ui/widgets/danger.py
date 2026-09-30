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

指名的理由要寫出來
------------------
「對家(立直)」與「下家(3副露)」在畫面上是同一階 —— 兩者都是「大概真的
在聽」。等級不因此改變(那是排除法數出來的),但**指名要指到**:有人坐在
三副露上,標題卻寫著「沒有人立直」,那是畫面說沒事而實際有事。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mia.analysis import DangerLevel, DangerReport, SeatDanger, TileDanger
from mia.ui.style import CAPTION, MUTED
from mia.ui.viewmodel import ViewState
from mia.ui.widgets.tiles import TileIcons, TileLabel

__all__ = ["LEVEL_COLORS", "AnalysisDangerTab", "danger_note", "seat_name"]

#: 最多列幾張。手牌 14 張全列會把整頁撐得很長,而使用者真正在挑的是最安全的
#: 那幾張與最危險的那幾張 —— 中間那些幾乎不看。
MAX_ROWS = 14

#: 等級的顏色。只有「安全」用綠色 —— 它是規則保證的,與其他三級不是同一種東西。
#:
#: Overlay 也用這一份。兩邊各寫一組的話,遲早會有一邊改了另一邊沒改,而
#: 「同一張牌在側邊視窗是橘的、在 HUD 上是紅的」會讓人不知道該信哪一個。
LEVEL_COLORS = {
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
        colour = LEVEL_COLORS[danger.level]
        self._level.setText(danger.level.label)
        self._level.setStyleSheet(f"color: {colour}; font-weight: 600;")
        self._detail.setText(_detail(danger, seat))


def _detail(danger: TileDanger, seat: int | None) -> str:
    """「對誰、憑什麼」那一句。

    三家都是現物時收成一句話 —— 那是最好的情況,不必把三行一樣的東西攤開。
    否則寫最危險那一家,**再加上哪幾家是現物**。

    Note:
        現物是「其餘的資訊量低」那條規則的例外。原本這裡只寫最危險的一家,
        於是「對立直家是現物、對另外兩家全新」這種局面在畫面上完全看不出
        前半段 —— 而那半段是規則保證的不可能,不是「比較不危險」,正好是
        打牌時最想知道的東西。等級仍然只由最危險那家決定(放銃只需要中一個),
        這裡加的是**描述**,不是等級。
    """
    if all(s.furiten for s in danger.seats):
        return "三家都是現物"
    worst = _worst(danger)
    line = f"對{_named(worst, seat)}:{worst.reason}"
    note = _furiten_note(danger, seat, exclude=worst)
    return f"{line} · {note}" if note else line


def danger_note(danger: TileDanger, seat: int | None) -> str:
    """同一句話的 **Overlay 版**:只寫對誰,不列待牌型。

    為什麼要短:HUD 疊在牌桌上,寬度是直接搶走的畫面。
    「兩面(4s5s)、邊張(1s2s)、嵌張(2s4s)、單騎(3s)、雙碰(3s3s)」那一串
    會把 Overlay 撐到橫跨半個牌桌 —— 而要看型的人本來就會去側邊視窗。

    **但「對誰」不能省。** 一張牌對立直的那家是現物、對另一家全新,只寫
    「安全」使用者就會照著打 —— 那是實測踩過的坑(見模組說明),換成 HUD
    也還是同一個坑。所以砍掉的是型的清單,不是對象。
    """
    if not danger.seats:
        return "沒有人需要防"
    if all(s.furiten for s in danger.seats):
        return "三家都是現物"
    if danger.level is DangerLevel.SAFE:
        return "對三家都排除了"
    worst = _worst(danger)
    note = _furiten_note(danger, seat, exclude=worst, verb=False)
    tail = f"・{note}現物" if note else ""
    return f"對{_who(danger, seat)}・還 {len(worst.waits)} 型{tail}"


def _who(danger: TileDanger, seat: int | None) -> str:
    """指名最危險的那一家,附上他憑什麼被指名(立直 / 三副露)。

    括號裡那個字**不能省**:「對下家危險」與「對下家(3副露)危險」是不同
    份量的話,而使用者要靠那個份量決定要不要繞路。
    """
    return _named(_worst(danger), seat)


def _named(sd: SeatDanger, seat: int | None) -> str:
    """一家的稱呼:相對座位 + 憑什麼被指名。"""
    who = seat_name(seat, sd.seat)
    return f"{who}({sd.stance})" if sd.stance else who


def _furiten_note(
    danger: TileDanger, seat: int | None, *, exclude: SeatDanger, verb: bool = True
) -> str:
    """「這幾家是現物」。沒有可講的就回空字串。

    Args:
        exclude: 已經在句子前半段被指名的那一家。**一定要排除** ——
            0 型又立直的家有可能同時是 :func:`_worst` 選中的那個,那時
            「對對家(立直):現物 · 對家(立直)是現物」會把同一件事說兩次。
        verb: 要不要帶「是現物」那兩個字。Overlay 版不帶,由呼叫端接上 ——
            HUD 的寬度是從牌桌上搶來的,省得下來的就要省。

    Note:
        **括號裡的立直 / 副露不省。** 「對家是現物」與「對家(立直)是現物」
        份量差很多:後者是說那個確定在聽的人打不到你,那正是這一句最有用的
        時候。省掉它省不到幾個字,卻剛好省掉了重點。
    """
    others = [s for s in danger.seats if s.furiten and s is not exclude]
    if not others:
        return ""
    who = "、".join(_named(s, seat) for s in others)
    return f"{who}是現物" if verb else who


def _worst(danger: TileDanger) -> SeatDanger:
    """最該提的那一家。

    **與分析層同一個排序,靠測試釘著。** 這裡原本漏掉 ``s.threat``,
    於是三副露的人與一個什麼都沒做的人平手時,名字落在座位編號小的那個 ——
    畫面上就變成「對下家危險」,而真正露出馬腳的上家一個字都沒提到。
    那正是 ``docs/decisions.md`` 第十八節說「平手不寫死就是拿巧合當道理」
    的那個坑,分析層修好了,這份抄過來的沒有跟上。
    """
    return max(danger.seats, key=lambda s: (s.level, len(s.waits), s.threat, s.reach))


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
        if report.threats:
            who = "、".join(
                f"{seat_name(report.seat, t.seat)}{t.label}" for t in report.threats
            )
            self._headline.setText(who)
            note = "以下是排除法的結果:還剩幾種待牌型沒被排除。"
        else:
            # 不寫死「立直或三副露」—— 門檻現在還包含「兩副露帶役牌」
            # (見 mia.mjai.table.MELD_THREAT_WITH_YAKUHAI),把條件抄在這裡
            # 只會在下次改門檻時變成一句安靜的謊話。
            self._headline.setText("沒有人露出馬腳")
            # 這句話很重要。沒人露出馬腳**不等於安全** —— 實測那場三次榮和,
            # 有兩次是沒立直的人和的;而其中一次到現在仍然讀不出來。
            note = (
                "沒有人立直、也沒有人副露到要防的程度 —— 但那不代表安全,"
                "一樣會有人榮和,只是我們無法確定他聽不聽牌。"
                "以下仍然是排除法的結果。"
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
