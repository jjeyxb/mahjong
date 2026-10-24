"""「框選牌桌範圍」對話框。

為什麼需要手動框
----------------
自動校正是從四邊往中間剝除整條顏色一致的帶狀區域。它是啟發式,而且
**錯的時候很安靜**:

* 實測 ``20260801-112437``(1800x1186):剝除停在瀏覽器工具列裡,剝出來的
  長寬比 1.56,整份錄影沒有一幀校得準。
* 同一份錄影開頭遊戲正在重連,畫面全黑。黑畫面剝得特別順,給出一個長寬比
  1.7645(離 16:9 只差 0.7%)的矩形,既有的兩道防線一道都沒響,於是被鎖進
  整場。手牌辨識 97% → 26%,全程無錯誤訊息。

與「框選手牌範圍」的差別
------------------------
那個是**取代**校正 —— 框了之後那條路上根本不算牌桌矩形。
這個**就是**校正的結果 —— 框了之後 ROI 與四家牌河的角點仍然照常相對它推導,
只是那個矩形改成人給的。所以這個框影響的東西多得多。

為什麼拖完要當場驗一次
----------------------
與 :mod:`~mia.ui.panel.handroi` 同一個理由,而且更要緊:牌桌矩形是所有
正規化座標的共同原點,框歪 3% 手牌的槽位就會一路累加偏移,到第 13 張差了
半張牌 —— 十三張、合法、全錯。

所以拖完會把從這個框推導出來的東西**畫回畫面上**:自己的手牌 ROI,以及
四家牌河的四邊形。框歪了,那些線就不會落在牌上 —— 看得見。再加上當場認一次
手牌,認不出來「確定」就不能按:存不下一個已知壞掉的框。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mia.calibration.table import Calibration
from mia.capture.base import Frame
from mia.config.models import RoiConfig
from mia.mjai.tiles import ms_to_mjai
from mia.ui.panel.picker import DragPicker
from mia.ui.style import CAPTION, MUTED
from mia.ui.widgets.tiles import DEFAULT_SKIN, TileIcons, TileLabel
from mia.utils.geometry import NormRect, Rect, Size
from mia.vision.roi import RoiSet
from mia.vision.tiles.classify import TemplateSet, classify, classify_hand
from mia.vision.tiles.hand import MAX_CONCEALED, read_hand
from mia.vision.tiles.river import (
    DEFAULT_CORNERS,
    SEATS,
    classify_river_tile,
    face_size,
    read_river,
)

__all__ = ["TableRectDialog"]

#: 預覽上畫牌的高度。
_RESULT_TILE = 34

#: 16:9。框出來的比例離它太遠通常是把瀏覽器工具列也框進去了。
_EXPECTED_ASPECT = 16 / 9

#: 長寬比差多少才提醒。不擋 —— 有些環境本來就不是 16:9
#: (實測 Steam 版客戶區 1512x844 = 1.7915)。
_ASPECT_HINT = 0.03

#: 四家的中文稱呼。用字典而不是跟著 :data:`~mia.vision.tiles.river.SEATS`
#: 的順序排一個 tuple —— 順序一旦改掉,tuple 會安靜地對應錯,而症狀是
#: 訊息說「對家」但其實是上家。
SEAT_NAMES = {"own": "自己", "shimocha": "下家", "toimen": "對家", "kamicha": "上家"}

#: 牌河至少要有這麼多張才拿來評分。
#:
#: 一局剛開始時四家加起來只有兩三張,那時的「有信心比例」是兩三個樣本的事,
#: 報出來只會誤導。寧可說「牌河還太空」也不要給一個會晃的數字。
_MIN_RIVER_TILES = 8

#: 牌河的「有信心比例」低於這個就是框歪了。
#:
#: 為什麼要這一條
#: --------------
#: **張數與連續性完全看不出框歪。** 實測同一幀把牌桌矩形的上緣往下推:
#:
#: ============  ======  ======  ==========  ============
#: 偏移          張數    連續    平均把握    有信心比例
#: ============  ======  ======  ==========  ============
#: 0(正確)      65      4/4     0.693       83%
#: +0.2% 高      65      4/4     0.676       74%
#: +0.5% 高      65      4/4     0.632       63%
#: +1.0% 高      65      4/4     0.481       23%
#: ============  ======  ======  ==========  ============
#:
#: 張數一路都是 65、四家一路都連續 —— 即時路徑的採用條件(連續 + 張數)
#: 在這裡**一個都攔不住**。1921 幀的全量也是同一回事:上緣偏 1% 時採用率
#: 仍有 93.3%,而每張牌正確率從 94.6% 掉到 32.9%。
#:
#: 所以能用的只有把握度。取 0.65:正確的框實測 0.83,偏 0.5% 是 0.63。
_MIN_RIVER_CONFIDENT = 0.65


class TableRectDialog(QDialog):
    """框一次牌桌範圍。

    Args:
        frame: 剛抓到的遊戲畫面。
        roi: ROI 設定,用來從框推導出自己的手牌範圍並驗一次。
        skin: 牌面素材,用來把認出來的牌畫出來。

    Note:
        :attr:`result_rect` 在按下「確定」之後才有值,而那顆按鈕只有在**真的
        認出一手合法的牌**時才能按。
    """

    def __init__(
        self,
        frame: Frame,
        *,
        roi: RoiConfig,
        skin: str = DEFAULT_SKIN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("框選牌桌範圍")
        self._image = frame.image
        self._roi = roi
        self._templates: TemplateSet | None = None
        #: 驗過的牌桌矩形,正規化成影像的比例。``None`` 代表還沒有結果。
        self.result_rect: NormRect | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint = QLabel(
            "框出遊戲畫布 —— 藍色牌桌連同左右兩側的 UI,"
            "**不要**含瀏覽器的分頁列與網址列。\n"
            "拖完會把從這個框算出來的手牌範圍與四家牌河畫回畫面上,"
            "線沒落在牌上就是框歪了。",
            self,
        )
        hint.setStyleSheet(CAPTION)
        layout.addWidget(hint)

        self._picker = DragPicker(self._image, self)
        self._picker.on_release = self._check
        layout.addWidget(self._picker, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._message = QLabel("還沒框 —— 在上面的畫面裡拖一個框", self)
        self._message.setStyleSheet(MUTED)
        layout.addWidget(self._message)

        self._tiles_row = QWidget(self)
        tiles_layout = QHBoxLayout(self._tiles_row)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        tiles_layout.setSpacing(2)
        icons = TileIcons(skin)
        # 多開一個位子給摸的那張 —— 使用者在自己摸牌那一巡框選時,
        # 少開的話畫出來會剛好少掉最關鍵的那一張。
        self._tiles = [
            TileLabel(icons, _RESULT_TILE, self._tiles_row) for _ in range(MAX_CONCEALED + 1)
        ]
        for label in self._tiles:
            label.setVisible(False)
            tiles_layout.addWidget(label)
        tiles_layout.addStretch(1)
        layout.addWidget(self._tiles_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        for standard, text in (
            (QDialogButtonBox.StandardButton.Ok, "確定"),
            (QDialogButtonBox.StandardButton.Cancel, "取消"),
        ):
            button = self._buttons.button(standard)
            if button is not None:
                button.setText(text)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._set_ok(False)

    def _set_ok(self, enabled: bool) -> None:
        """「確定」只有在真的認出一手牌時才能按 —— 存不下一個已知壞掉的框。"""
        button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(enabled)

    # ------------------------------------------------------------------ 驗證

    def _check(self) -> None:
        """拖完之後:推導 ROI 與牌河 → 畫出來 → 認一次手牌。

        每一步失敗都說**那一步**的話。「框選失敗」這種句子幫不上忙 ——
        使用者要知道的是框大了、框小了,還是只是這一巡手上有副露。
        """
        self._show_tiles(())
        self.result_rect = None
        self._picker.clear_derived()
        self._set_ok(False)

        box = self._picker.box
        if box is None:
            self._message.setText("還沒框 —— 在上面的畫面裡拖一個框")
            self._picker.update()
            return

        rough = Rect(box.x(), box.y(), box.width(), box.height())
        width, height = self._picker.image_size

        # 微調要四五秒,而那段時間視窗是凍住的。先把話說出來並讓 Qt 把它畫上去,
        # 不然使用者看到的是一個按了沒反應的程式 —— 而他剛拖完框,正等著回應。
        self._message.setText("正在微調框的位置(約五秒)…")
        QApplication.processEvents()
        if self._templates is None:
            self._templates = TemplateSet.load()
        table = _refine(self._image, rough, self._templates)
        nudged = (
            table.x - rough.x,
            table.y - rough.y,
            table.width - rough.width,
            table.height - rough.height,
        )
        self._picker.polys = _river_quads(table)

        hand_roi = _hand_roi(self._roi, table, Size(width, height))
        if hand_roi is None:
            self._message.setText(
                "設定檔裡沒有 own_hand 這個 ROI —— 框還是存得下去,"
                "但這裡沒辦法當場驗給你看(見 config/default.yaml)"
            )
            self._picker.update()
            return
        self._picker.rects = [QRect(hand_roi.x, hand_roi.y, hand_roi.width, hand_roi.height)]

        aspect_note = _aspect_note(table)
        crop = self._image[hand_roi.as_slice()]
        hand = read_hand(crop) if crop.size else None
        if hand is None or not hand.is_plausible:
            found = 0 if hand is None else len(hand)
            self._message.setText(
                f"算出來的手牌範圍裡讀到 {found} 張,不是合法手牌 —— "
                f"框歪了的話重拖一次;手上有副露的話先打到沒有再框。{aspect_note}"
            )
            self._picker.update()
            return

        try:
            concealed, drawn = classify_hand(crop, hand, self._templates, headroom=0)
        except ValueError as exc:
            self._message.setText(f"牌太小認不出來:{exc}")
            self._picker.update()
            return

        matches = [*concealed, *([drawn] if drawn else [])]
        score = float(np.mean([m.score for m in matches]))
        # classify 回的是雀魂記法(5z),而 TileIcons 吃的是 MJAI(P)。
        # 數字牌兩種記法剛好一樣,所以漏掉這一步的症狀是「只有字牌畫不出來」
        # —— 實測白板畫成了「5z」三個字。
        self._show_tiles([ms_to_mjai(m.label) for m in matches])
        self._message.setText(
            f"認出 {len(concealed)} 張暗手牌"
            + (f" + 摸的 {ms_to_mjai(drawn.label)}" if drawn else "")
            + f",平均把握 {score:.2f}。{_nudge_note(nudged)}"
            + f"\n{self._river_note(table)}"
            + _clipped_note(table, self._image)
            + aspect_note
        )
        self.result_rect = NormRect(
            table.x / width, table.y / height, table.width / width, table.height / height
        )
        self._set_ok(True)
        self._picker.update()

    def _river_note(self, table: Rect) -> str:
        """四家牌河讀起來怎麼樣。

        **不要叫使用者用眼睛判斷綠線正不正。** 容忍度比看得出來的還窄:實測
        牌桌矩形的上緣只要偏 0.5% 的高度(1608 px 的畫面上是 8 px),每張牌
        正確率就從 94.6% 掉到 79.2%,偏 1% 掉到 32.9% —— 而那時綠線看起來
        仍然「大致落在牌上」。

        也不能只報張數與連續性:那兩個在上面那些偏移下**完全沒有反應**
        (見 :data:`_MIN_RIVER_CONFIDENT`)。所以這裡報的是把握度。
        """
        if self._templates is None:
            self._templates = TemplateSet.load()
        size = face_size()
        counts: list[int] = []
        holes: list[str] = []
        confident = total = 0
        for seat in SEATS:
            flat = DEFAULT_CORNERS[seat].rectify(self._image, table)
            river = read_river(flat)
            counts.append(river.count)
            if not river.is_plausible():
                holes.append(SEAT_NAMES[seat])
            for index in river.filled:
                match = classify_river_tile(
                    flat[river.cells[index].as_slice()], self._templates, size=size
                )
                total += 1
                confident += match.is_confident

        read = "/".join(str(c) for c in counts)
        if holes:
            return f"⚠ {'、'.join(holes)}的牌河中間有洞(讀到 {read} 張)—— 框歪了,重拖一次。"
        if total < _MIN_RIVER_TILES:
            return f"牌河現在只有 {read} 張,太空了看不準 —— 這一局打久一點再框會比較有把握。"
        ratio = confident / total
        if ratio < _MIN_RIVER_CONFIDENT:
            return (
                f"⚠ 四家牌河讀到 {read} 張,但只有 {ratio:.0%} 有把握"
                f"(正常是八成以上)—— 框歪了,重拖一次。"
            )
        return f"四家牌河讀到 {read} 張,{ratio:.0%} 有把握 —— 可以按「確定」。"

    def _show_tiles(self, labels: list[str] | tuple[str, ...]) -> None:
        for label, tile in zip(self._tiles, labels, strict=False):
            label.set_tile(tile)
            label.setVisible(True)
        for label in self._tiles[len(labels) :]:
            label.setVisible(False)


#: 微調時上下左右各搜尋多少像素,以及粗掃的步長。
#:
#: 為什麼一定要微調
#: ----------------
#: **人框不到那個精度。** 實測牌桌矩形的上緣只要偏 0.5% 的高度,每張牌正確率
#: 就從 94.6% 掉到 79.2%;1608 px 的畫面上那是 8 px,而預覽縮到 820 px 寬之後
#: 只剩 4 px。要使用者用滑鼠拖到那個準度是不合理的要求。
#:
#: 範圍取 ±32 px:實測牌譜畫面的自動校正差了 18 px(連續只有 2/4),微調之後
#: 四家全部連續。粗掃步長 4 px 在容忍度(約 5 px)之內。
_REFINE_RANGE = 32
_REFINE_STEP = 4

#: 細挑階段最多試幾個候選。每一個都要把四家的牌都認一次 —— 實測單一尺度
#: 0.64 秒、兩個尺度 1.27 秒,那是這裡最貴的事。粗掃掃滿整個網格只要 1.1 秒。
#:
#: 細挑**只用一個尺度**。兩個尺度的覆核是為了壓靜默錯誤(見
#: :data:`~mia.vision.tiles.river.CHECK_FACE_HEIGHT`),而這裡問的是
#: 「對得準不準」,兩個尺度一致與否不影響這個判斷的方向,只是讓它貴一倍。
_REFINE_FINALISTS = 5


def _clipped_note(table: Rect, image: np.ndarray) -> str:
    """微調的最佳解是不是被畫面邊界卡住的。

    牌桌矩形是個**座標系**,不是實體邊界 —— 而擷取有可能切到畫布。實測一張
    2560x1614 的牌譜畫面:不限邊界時最好的矩形下緣在 1620(超出 6 px),
    有把握 57 張;硬夾回畫面內之後只剩 33 張。

    夾是必要的(存成比例的型別不收超出邊界的值),但**夾掉之後要講**。
    不講的話使用者看到的是「只有 48% 有把握 —— 框歪了,重拖一次」,
    於是他會一直重框一個其實沒框錯的東西。
    """
    height, width = image.shape[:2]
    edges = []
    if table.y <= _REFINE_STEP:
        edges.append("上")
    if table.y + table.height >= height - _REFINE_STEP:
        edges.append("下")
    if table.x <= _REFINE_STEP:
        edges.append("左")
    if table.x + table.width >= width - _REFINE_STEP:
        edges.append("右")
    if not edges:
        return ""
    return (
        f"\n(微調在畫面的{'、'.join(edges)}緣就被卡住了 —— 擷取可能切到了畫布,那不是你框的問題。)"
    )


def _nudge_note(nudged: tuple[int, int, int, int]) -> str:
    """微調把框移了多少。移動量是 0 就不講 —— 沒事的時候不要佔版面。

    講出來不是為了炫耀,是因為**微調是會錯的**:它在一個小範圍裡找最好的,
    而「最好的」可能只是「一堆爛的裡面比較不爛的那個」。使用者看到「往下
    移了 30 px」就知道該懷疑一下自己原本框的位置。
    """
    dx, dy, dw, dh = nudged
    if not any(nudged):
        return ""
    parts = []
    if dx or dy:
        parts.append(f"位置 ({dx:+d}, {dy:+d})")
    if dw or dh:
        parts.append(f"大小 ({dw:+d}, {dh:+d})")
    return "微調了 " + "、".join(parts) + " px。"


def _refine(image: np.ndarray, table: Rect, templates: TemplateSet) -> Rect:
    """把使用者框的矩形微調到最好的那個。

    兩段,因為兩個訊號的成本差三個數量級:

    1. **粗掃**只看「四家的牌河是不是都從頭連續」。那只要算每一格的中位亮度
       與色相,快得可以掃滿整個網格。它看得出整體縮放錯了(牌譜那次差 18 px,
       連續從 2/4 變成 4/4),但**看不出半格以內的偏移** —— 實測上緣偏 1% 時
       連續性與張數完全沒反應。
    2. **細挑**在粗掃的優勝者裡把牌真的認一次,取「有把握的張數」最多的。
       那是唯一對小偏移有反應的訊號。

    目標是有把握的**張數**而不是比例。比例會獎勵把框縮小 —— 把難認的牌排除
    在外,剩下的比例自然漂亮。張數不會:少框一張牌就少一分。這個陷阱在校四家
    角點時踩過一次,見 :data:`~mia.vision.tiles.river.DEFAULT_CORNERS`。
    """
    height, width = image.shape[:2]
    coarse: list[tuple[int, int, Rect]] = []
    for dy in range(-_REFINE_RANGE, _REFINE_RANGE + 1, _REFINE_STEP):
        for dh in range(-_REFINE_RANGE, _REFINE_RANGE + 1, _REFINE_STEP):
            candidate = Rect(table.x, table.y + dy, table.width, table.height + dh)
            # 長到畫面外的不能要。牌桌矩形之後要存成**相對影像**的比例,
            # 而那個型別不收超出邊界的值 —— 它會拋,而這裡是 Qt 的 slot,
            # 例外漏出去的樣子是「訊息說可以按確定,但按鈕是灰的」。
            if candidate.height <= 0 or candidate.y < 0:
                continue
            if candidate.y + candidate.height > height or candidate.x + candidate.width > width:
                continue
            seats = tiles = 0
            for seat in SEATS:
                river = read_river(DEFAULT_CORNERS[seat].rectify(image, candidate))
                seats += river.is_plausible()
                tiles += river.count
            coarse.append((seats, tiles, candidate))

    if not coarse:
        return table
    coarse.sort(key=lambda row: (row[0], row[1]), reverse=True)

    best, best_score = table, -1
    for _, _, candidate in coarse[:_REFINE_FINALISTS]:
        confident = 0
        for seat in SEATS:
            flat = DEFAULT_CORNERS[seat].rectify(image, candidate)
            river = read_river(flat)
            if not river.is_plausible():
                continue
            size = face_size()
            for index in river.filled:
                match = classify(flat[river.cells[index].as_slice()], templates, size=size)
                confident += match.is_confident
        if confident > best_score:
            best, best_score = candidate, confident
    return best


def _aspect_note(table: Rect) -> str:
    """長寬比離 16:9 太遠時的提醒。沒問題就回空字串。

    不擋,只提醒:有些環境本來就不是 16:9(實測 Steam 版客戶區 1.7915)。
    但差太多最常見的原因是把瀏覽器工具列也框進去了,那值得當場講一聲。
    """
    deviation = abs(table.aspect - _EXPECTED_ASPECT) / _EXPECTED_ASPECT
    if deviation <= _ASPECT_HINT:
        return ""
    return (
        f"\n⚠ 長寬比 {table.aspect:.3f},離 16:9 差 {deviation:.0%} ——"
        "最常見的原因是把瀏覽器工具列也框進去了。"
    )


def _hand_roi(roi: RoiConfig, table: Rect, image: Size) -> Rect | None:
    """從框出來的牌桌矩形推導自己的手牌範圍。``own_hand`` 沒設定時回 ``None``。"""
    rois = RoiSet(roi, Calibration(table_rect=table, image_size=image, source="manual"))
    return rois["own_hand"].rect if "own_hand" in rois else None


def _river_quads(table: Rect) -> list[list[tuple[int, int]]]:
    """四家牌河在畫面上的四邊形。

    畫出來而不只是算一算:牌河的角點是相對牌桌矩形的比例,所以它們落得準不準
    **就是**這個框對不對的證據 —— 而且是比手牌更嚴的證據,因為它們在畫面中央,
    離框的邊緣最遠,框的誤差在那裡放得最大。
    """
    quads = []
    for seat in SEATS:
        quad = DEFAULT_CORNERS[seat].quad(table)
        quads.append([(int(x), int(y)) for x, y in quad])
    return quads
