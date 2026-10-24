"""「框選牌桌範圍」對話框裡的推導。

這裡測的是**不需要 GUI 的那部分**:從使用者拖的框算出手牌 ROI 與四家牌河,
以及長寬比的提醒。對話框本身的安全網就是把這些畫回畫面上 —— 算錯了,
畫出來的線就落在錯的地方,而那正是使用者唯一看得見的證據。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.config.models import RoiConfig
from mia.ui.panel.tablerect import (
    _REFINE_FINALISTS,
    SEAT_NAMES,
    TableRectDialog,
    _aspect_note,
    _coverage,
    _hand_roi,
    _nudge_note,
    _refine,
    _river_quads,
)
from mia.utils.geometry import NormRect, Rect, Size
from mia.vision.tiles.river import DEFAULT_CORNERS, SEATS, read_river

#: 實測素材的牌桌矩形與畫面尺寸(``data/recordings/20260801-110639``)。
TABLE = Rect(0, 174, 2536, 1430)
IMAGE = Size(2536, 1608)


@pytest.fixture(scope="module")
def table_image():
    """真實對局的一整張擷取畫面。答案來自封包,不是目視 —— 見 test_vision_river。"""
    import cv2

    path = Path(__file__).resolve().parents[1] / "fixtures" / "table_full.jpg"
    image = cv2.imread(str(path))
    assert image is not None, "讀不到 table_full.jpg"
    return image


@pytest.fixture(scope="module")
def templates():
    from mia.vision.tiles.classify import TemplateSet

    return TemplateSet.load()


def _wanted(image, table: Rect) -> dict[str, list[int]]:
    """粗掃要量哪些格子。由粗框決定一次,之後每個候選量同一批。"""
    return {
        seat: list(read_river(DEFAULT_CORNERS[seat].rectify(image, table)).filled)
        for seat in SEATS
    }


def _body(func) -> str:
    """函式的原始碼,**去掉說明文字**。

    釘原始碼的測試要看的是程式在做什麼,不是說明在講什麼。說明裡正好會提到
    那些被淘汰掉的做法(例如「連續性不參與」),整段抓下來就會把說明當成程式
    誤判 —— 這個測試自己踩過一次。
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    if ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    return ast.unparse(node)


class TestHandRoi:
    def test_it_matches_what_the_evaluator_computes(self) -> None:
        """對話框算的必須與 ``tools/evaluate.py`` 算的是同一個。

        兩邊算不一樣的話,使用者在對話框裡看到線落在牌上、存下去,然後實際
        跑的是另一個位置 —— 畫面說一套、程式做另一套,最糟的那種。
        """
        roi = _hand_roi(_config(), TABLE, IMAGE)
        assert roi == Rect(292, 1394, 1796, 205)

    def test_it_scales_with_the_frame(self) -> None:
        """ROI 是相對牌桌矩形的比例 —— 框小一半,ROI 要跟著小一半。"""
        half = Rect(0, 87, 1268, 715)
        roi = _hand_roi(_config(), half, Size(1268, 804))
        assert roi is not None
        assert roi.width == pytest.approx(1796 / 2, abs=2)

    def test_a_config_without_own_hand_says_so(self) -> None:
        """設定檔沒有這個 ROI 時要回 None,不能拋 —— 對話框還是要開得起來,
        只是沒辦法當場驗給使用者看。
        """
        assert _hand_roi(RoiConfig(), TABLE, IMAGE) is None


class TestRiverQuads:
    def test_every_seat_gets_a_quad(self) -> None:
        quads = _river_quads(TABLE)
        assert len(quads) == len(SEATS)
        assert all(len(q) == 4 for q in quads)

    def test_the_quads_land_inside_the_table(self) -> None:
        """牌河在牌桌中央 —— 跑到框外面就是推導錯了。"""
        for quad in _river_quads(TABLE):
            for x, y in quad:
                assert TABLE.x <= x <= TABLE.x + TABLE.width
                assert TABLE.y <= y <= TABLE.y + TABLE.height

    def test_they_move_with_the_frame(self) -> None:
        """角點是比例 —— 框往下移,牌河要跟著往下移。

        不跟著動的話,使用者把框拖到別的地方,畫面上的綠線卻不動,
        於是他看不出自己框錯了。
        """
        moved = Rect(TABLE.x, TABLE.y + 100, TABLE.width, TABLE.height)
        before = _river_quads(TABLE)[0][0]
        after = _river_quads(moved)[0][0]
        assert after[1] - before[1] == pytest.approx(100, abs=2)


class TestAspectNote:
    def test_a_normal_frame_says_nothing(self) -> None:
        assert _aspect_note(TABLE) == ""

    def test_a_non_sixteen_nine_client_is_tolerated(self) -> None:
        """Steam 版的客戶區實測 1512x844 = 1.7915,本來就不是 16:9。

        擋下來的話那些使用者永遠按不了「確定」。
        """
        assert _aspect_note(Rect(0, 0, 1512, 844)) == ""

    def test_a_toolbar_left_in_the_frame_is_called_out(self) -> None:
        """實測 20260801-112437 自動剝除剝出 1798x1152 = 1.56,就是這種。"""
        note = _aspect_note(Rect(0, 0, 1798, 1152))
        assert "工具列" in note


def _config() -> RoiConfig:
    from mia.config.loader import load_config

    return load_config().roi


class TestSeatNames:
    def test_every_seat_has_a_name(self) -> None:
        """少一家的話訊息會丟 KeyError —— 而那發生在使用者剛拖完框的時候。"""
        assert set(SEAT_NAMES) == set(SEATS)


class TestRiverNote:
    """拖完之後對牌河的判語。

    **不叫使用者用眼睛判斷綠線正不正。** 容忍度比看得出來的還窄:實測牌桌
    矩形的上緣只要偏 0.5% 的高度(1608 px 的畫面上是 8 px),每張牌正確率
    就從 94.6% 掉到 79.2%,而綠線看起來仍然「大致落在牌上」。
    """

    def _note(self, image, table: Rect) -> str:
        """不開 QDialog —— 這段邏輯不需要視窗。"""

        class Stub:
            _image = image
            _templates = None
            _river_note = TableRectDialog._river_note  # noqa: SLF001

        return Stub()._river_note(table)  # noqa: SLF001

    def test_a_correct_frame_passes(self, table_image) -> None:
        note = self._note(table_image, TABLE)
        assert "可以按" in note
        assert "15/17/17/16" in note, "張數要報出來,那是使用者唯一對得起來的東西"

    @pytest.mark.parametrize(("offset", "label"), [(8, "0.5%"), (16, "1.0%")])
    def test_a_slightly_crooked_frame_is_caught(self, table_image, offset, label) -> None:
        """實測上緣偏 0.5% → 63% 有把握,偏 1% → 23%。門檻 65% 兩個都擋得下。"""
        note = self._note(table_image, Rect(0, 174 + offset, 2536, 1430))
        assert "框歪了" in note, f"偏 {label} 沒被擋下來"

    def test_the_count_alone_cannot_tell(self, table_image) -> None:
        """**這是這條檢查存在的理由。**

        即時路徑的採用條件是「連續 + 張數正確」,而那兩個在這些偏移下一個都
        沒反應 —— 三種偏移讀到的都是 15/17/17/16 張、四家都連續。1921 幀的
        全量也一樣:上緣偏 1% 時採用率仍有 93.3%,每張牌卻只剩 32.9%。
        """
        counts = [_counts(table_image, Rect(0, 174 + d, 2536, 1430)) for d in (0, 8, 16)]
        assert counts[0] == counts[1] == counts[2] == (15, 17, 17, 16)

    def test_an_empty_river_says_so_instead_of_guessing(self) -> None:
        """一局剛開始四家加起來只有兩三張,那時的比例是兩三個樣本的事。

        寧可說「太空了」也不要給一個會晃的數字 —— 使用者會照著它重框。
        """
        import numpy as np

        blank = np.full((1608, 2536, 3), 70, dtype=np.uint8)
        assert "太空" in self._note(blank, TABLE)


def _counts(image, table: Rect) -> tuple[int, ...]:
    from mia.vision.tiles.river import DEFAULT_CORNERS, read_river

    return tuple(read_river(DEFAULT_CORNERS[s].rectify(image, table)).count for s in SEATS)


class TestRefine:
    """把使用者框的矩形微調到最好的那個。

    **人框不到那個精度。** 實測上緣偏 0.5% 的高度,每張牌正確率就從 94.6%
    掉到 79.2%;1608 px 的畫面上那是 8 px,而預覽縮到 820 px 寬之後只剩 4 px。
    要使用者用滑鼠拖到那個準度是不合理的要求,所以程式自己補上最後那幾像素。
    """

    def test_it_recovers_a_frame_nudged_off(self, table_image, templates) -> None:
        """從偏掉的地方出發,要回到跟正確答案一樣好。

        **用有把握的張數衡量,不是用連續性。** 連續性在這些偏移下完全沒反應
        —— 實測上緣偏 +14 px 時四家仍然全部連續,而有把握的張數從 56 掉到 18。
        那正是微調要存在的理由:使用者看得到的訊號(線有沒有落在牌上、張數對
        不對)對這個級距的偏移都是瞎的。
        """
        off = Rect(TABLE.x, TABLE.y + 14, TABLE.width, TABLE.height)
        assert _confident(table_image, off, templates) < 25, "前提:起點確實是壞的"
        after = _refine(table_image, off, templates)
        assert _confident(table_image, after, templates) >= 50

    def test_it_leaves_a_good_frame_good(self, table_image, templates) -> None:
        """已經對的框不能被微調弄壞 —— 那比不微調還糟。"""
        after = _refine(table_image, TABLE, templates)
        assert _confident(table_image, after, templates) >= 50
        assert _seats_ok(table_image, after) == 4

    def test_different_starts_land_somewhere_good(self, table_image, templates) -> None:
        """使用者每次拖的位置都不一樣,結果不能隨之起舞。

        不要求收斂到**同一個**矩形 —— 搜尋空間上有一道脊,脊上的點一樣好
        (實測兩個差 16 px 的答案張數與連續性完全相同)。要求的是都好。

        只取兩個起點,一邊一個。原本有四個,但微調一次要 17 秒(入圍 24 個
        候選,每個都要把四家的牌認一遍),四個起點就是 69 秒 —— 這一個測試
        曾經佔掉整套的三分之一。兩邊各一個已經涵蓋這條斷言要講的事。
        """
        for dy in (-10, 12):
            start = Rect(TABLE.x, TABLE.y + dy, TABLE.width, TABLE.height)
            after = _refine(table_image, start, templates)
            assert _confident(table_image, after, templates) >= 50, f"從 {dy:+d} 出發"

    def test_the_objective_is_a_count_not_a_ratio(self) -> None:
        """比例會獎勵把框縮小 —— 把難認的牌排除在外,剩下的比例自然漂亮。

        這個陷阱在校四家角點時踩過一次(見 DEFAULT_CORNERS 的說明),
        所以這裡釘的是原始碼本身:目標必須是**張數**。
        """
        source = _body(_refine)
        assert "confident += " in source
        assert "/ total" not in source, "一旦改成比例,縮框就會變成最佳解"


class TestNudgeNote:
    def test_no_movement_says_nothing(self) -> None:
        """沒事的時候不要佔版面。"""
        assert _nudge_note((0, 0, 0, 0)) == ""

    def test_movement_is_reported(self) -> None:
        """微調是會錯的 —— 它在小範圍裡找最好的,而那可能只是「比較不爛的」。

        講出來,使用者看到「往下移了 30 px」才知道該懷疑一下自己框的位置。
        """
        note = _nudge_note((0, 18, 0, -6))
        assert "+18" in note
        assert "-6" in note


def _confident(image, table: Rect, templates) -> int:
    """四家合起來有幾張認得有把握。微調的目標函數就是這個。"""
    from mia.vision.tiles.classify import classify
    from mia.vision.tiles.river import DEFAULT_CORNERS, face_size, read_river

    size = face_size()
    total = 0
    for seat in SEATS:
        flat = DEFAULT_CORNERS[seat].rectify(image, table)
        river = read_river(flat)
        for index in river.filled:
            match = classify(flat[river.cells[index].as_slice()], templates, size=size)
            total += match.is_confident
    return total


def _seats_ok(image, table: Rect) -> int:
    from mia.vision.tiles.river import DEFAULT_CORNERS, read_river

    return sum(read_river(DEFAULT_CORNERS[s].rectify(image, table)).is_plausible() for s in SEATS)


class TestRefineStaysInsideTheFrame:
    """微調不可以長到畫面外。

    牌桌矩形之後要存成**相對影像**的比例,而那個型別不收超出邊界的值 ——
    它會拋 ValueError,而這裡是 Qt 的 slot。例外漏出去的樣子是:訊息寫著
    「可以按確定」,而確定鍵是灰的,錯誤只進 stderr。使用者回報過這一幕。
    """

    @pytest.mark.parametrize(
        "start",
        [
            Rect(0, 0, 2536, 1430),  # 貼著上緣
            Rect(0, 174, 2536, 1434),  # 貼著下緣
            Rect(0, 174, 2536, 1430),  # 一般
        ],
    )
    def test_the_result_can_always_be_stored(self, table_image, templates, start) -> None:
        out = _refine(table_image, start, templates)
        height, width = table_image.shape[:2]
        NormRect(out.x / width, out.y / height, out.width / width, out.height / height)


class TestTheCoarseStageOnlyExcludes:
    """粗掃的工作是排除,不是排名。

    這裡釘的是一個踩過的坑。粗掃原本的目標是「有牌的格數」,而那個訊號
    **是反著走的** —— 格線錯位時每一格都壓到隔壁的牌,於是*更多*格子看起來
    有牌。實測一張 2560x1614 的畫面、169 個格點:張數最高(69 張、四家全
    連續)的前五名,真正認得出來的只有 15~21 張,而真正的最佳解 58 張排在
    第 19 名。粗掃不是沒幫上忙,是把最佳解主動淘汰掉了。

    改成覆蓋率之後相關係數 +0.76,但前五名仍只到 53/58 —— 所以它仍然排不出
    名次,入圍名額才要開到 24(24 名 → 57,48 名 → 58)。
    """

    def test_coverage_is_not_fooled_by_a_slipped_grid(self, table_image) -> None:
        """錯位**不可以**讓分數上升。方向錯了,微調就會主動往壞的地方跑。

        這是張數那個目標真正的毛病:格線壓到隔壁的牌時更多格子看起來有牌,
        所以錯得越多分越高。覆蓋率沒有這個問題 —— 錯位的格子會吃到桌布。
        """
        wanted = _wanted(table_image, TABLE)
        aligned = _coverage(table_image, TABLE, wanted)
        for dy in (-20, -12, -6):
            slipped = Rect(TABLE.x, TABLE.y + dy, TABLE.width, TABLE.height)
            assert _coverage(table_image, slipped, wanted) < aligned, f"偏 {dy:+d} px"

    def test_coverage_cannot_rank(self, table_image) -> None:
        """而它**也排不出名次** —— 這條釘的就是它不夠好這件事。

        實測這張 fixture(正確答案在 0,有把握 56 張):

        ====  ======  ========
        偏移  覆蓋率  有把握
        ====  ======  ========
        -20   39.47   13
        0     43.86   **56**
        +6    44.76   49
        +20   **44.03**   10
        ====  ======  ========

        往上偏它分得出來,往下偏幾乎是平的 —— 偏 +20 的覆蓋率還贏過正確答案,
        而那裡只認得出 10 張。所以粗掃只能用來排除,入圍名額要開到
        :data:`_REFINE_FINALISTS` 那麼大,名次交給細挑。

        哪天有人想把名額調回 5,這條會先告訴他為什麼不行。
        """
        wanted = _wanted(table_image, TABLE)
        aligned = _coverage(table_image, TABLE, wanted)
        worse = Rect(TABLE.x, TABLE.y + 20, TABLE.width, TABLE.height)
        assert _coverage(table_image, worse, wanted) > aligned
        assert _REFINE_FINALISTS >= 24, "粗掃排不出名次,名額就不能小"

    def test_the_cell_set_is_fixed_by_the_rough_rect(self) -> None:
        """被評分的格子集合不可以跟著候選框跑。

        這是前面那個陷阱的真正根源:候選框一動,「哪些格子有牌」就跟著動,
        於是任何分數都可以靠「換一批好算的格子」拿到。實測四個便宜訊號
        (連續家數、張數、覆蓋率、牌面重心偏移)全部被這件事破壞,其中重心
        偏移甚至整個反過來 —— 偏移最小的前三名有把握只有 4~6 張。
        """
        source = _body(_refine)
        assert "wanted = {" in source, "格子集合要先由粗框定下來"
        assert source.index("wanted = {") < source.index("for dy in range")

    def test_contiguity_is_not_used_at_all(self) -> None:
        """連續性不參與微調的任何一段。

        牌河被鳴牌拿走的那張會留下真正的空格,所以「不連續」本來就不代表
        框錯。實測那張牌譜畫面的最佳解(有把握 58)只有 3/4 家連續,而拿
        連續家數當第一順位排序正好會把它淘汰掉。
        """
        assert "is_plausible" not in _body(_refine)
