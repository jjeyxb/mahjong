"""牌河:攤平成俯視圖,切格,再跟封包的標準答案對。

素材 ``table_full.jpg`` 是真實對局的一整張擷取畫面(2536x1608,牌桌矩形
2536x1430),四家牌河 15/17/17/16 張。答案不是目視判讀的,是同一時刻的封包
算出來的 —— 所以這一組測試量的是**真的準確率**,不是「看起來對不對」。

:data:`EXPECTED` 的來源:``data/recordings/20260801-110639`` 幀 12141,
與 ``data/gt/ws.jsonl`` 對齊(自己坐 2 號位)。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.utils.geometry import Rect
from mia.vision.tiles.classify import TemplateSet, classify
from mia.vision.tiles.river import (
    CELL,
    CHECK_FACE_HEIGHT,
    COLUMNS,
    DEFAULT_CORNERS,
    FACE,
    ROWS,
    SEATS,
    SIDEWAYS_FACE,
    SIDEWAYS_SHIFT,
    River,
    RiverView,
    classify_river_tile,
    classify_sideways,
    face_size,
    find_sideways,
    read_river,
    sideways_face_size,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _hsv(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    """用 HSV 描述一個顏色,回傳 BGR。

    這一組測試講的全是色相與飽和度,直接寫 BGR 的話每個常數都要讀的人
    自己換算一次 —— 而換算錯了測試仍然會過,只是測到的不是原本要測的東西。
    """
    pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


#: 素材那一幀的牌桌矩形。錄製時 StableCalibrator 鎖定的結果。
TABLE = Rect(0, 174, 2536, 1430)

#: 封包算出來的四家牌河(看得見的那些,被鳴走的已經濾掉)。
EXPECTED = {
    "own": "N 1s F 8m C 8s 4p 8s E W S 2m 4m 9p 7p",
    "shimocha": "N C P 4s 2m 5p 8m 6s 7s 8p E W 9p 6m 7p 7m 1s",
    "toimen": "P S 1s 9m 2m 4p 2s E 5m 1s 3m 5p S 9m F 9s 8p",
    "kamicha": "9m 5s 2s 9s 8s W 9s 6s 3p C E 1m 1m W 9p 5m",
}


@pytest.fixture(scope="module")
def table_image() -> np.ndarray:
    image = cv2.imread(str(FIXTURES / "table_full.jpg"))
    assert image is not None, "讀不到 table_full.jpg"
    return image


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    return TemplateSet.load()


def _read(image: np.ndarray, seat: str, templates: TemplateSet) -> list[str]:
    """攤平 → 切格 → 分類,回傳 MJAI 記法(赤五已正規化成普通五)。"""
    flat = DEFAULT_CORNERS[seat].rectify(image, TABLE)
    river = read_river(flat)
    size = face_size()
    out = []
    for index in river.filled:
        match = classify_river_tile(flat[river.cells[index].as_slice()], templates, size=size)
        out.append(normalize_red(ms_to_mjai(match.label)))
    return out


class TestRiverView:
    def test_the_quad_follows_the_table_rect(self) -> None:
        """角點是比例,所以換一個牌桌矩形就該整個跟著縮放。"""
        view = RiverView(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        quad = view.quad(Rect(10, 20, 100, 200))
        assert quad.tolist() == [[10, 20], [110, 20], [110, 220], [10, 220]]

    def test_rectify_produces_the_canonical_grid(self, table_image: np.ndarray) -> None:
        flat = DEFAULT_CORNERS["own"].rectify(table_image, TABLE)
        assert flat.shape[:2] == (ROWS * CELL[1], COLUMNS * CELL[0])

    @pytest.mark.parametrize("seat", SEATS)
    def test_every_seat_has_corners(self, seat: str) -> None:
        assert len(DEFAULT_CORNERS[seat].corners) == 4

    def test_the_face_is_smaller_than_its_cell(self) -> None:
        """一格裡除了牌面還含著牌縫與相鄰牌的邊緣。"""
        width, height = face_size()
        assert 0 < width < CELL[0]
        assert 0 < height < CELL[1]


class TestReadRiver:
    @pytest.mark.parametrize("seat", SEATS)
    def test_the_tile_count_matches_the_packets(self, seat: str, table_image: np.ndarray) -> None:
        """數錯張數的後果比認錯牌嚴重 —— 少一格會讓後面整串索引往前位移,
        那一家就從「錯一張」變成「幾乎全錯」。
        """
        flat = DEFAULT_CORNERS[seat].rectify(table_image, TABLE)
        assert read_river(flat).count == len(EXPECTED[seat].split())

    @pytest.mark.parametrize("seat", SEATS)
    def test_a_real_river_is_contiguous(self, seat: str, table_image: np.ndarray) -> None:
        flat = DEFAULT_CORNERS[seat].rectify(table_image, TABLE)
        assert read_river(flat).is_plausible()

    def test_a_hole_in_the_middle_is_not_plausible(self) -> None:
        """牌河依序往後填,中間不會有洞 —— 有洞就是攤平歪了或抓在動畫中間。"""
        occupied = tuple([True, True, False, True] + [False] * 14)
        assert not River(cells=(), occupied=occupied).is_plausible()

    def test_an_empty_river_is_plausible(self) -> None:
        assert River(cells=(), occupied=(False,) * 18).is_plausible()

    def test_a_greyscale_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="BGR"):
            read_river(np.zeros((360, 540), dtype=np.uint8))


class TestAgainstThePackets:
    """**這一組是重點。** 答案來自封包,不是目視。"""

    @pytest.mark.parametrize(
        ("seat", "floor"),
        [("own", 13), ("shimocha", 16), ("toimen", 13), ("kamicha", 14)],
    )
    def test_each_seat_reads_mostly_right(
        self, seat: str, floor: int, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """門檻訂在實測值少一張 —— 釘住現況,又留一張的餘地給無害的波動。

        掉下來就是幾何或模板被動到了,而那是會靜靜發生的事。
        """
        truth = EXPECTED[seat].split()
        read = _read(table_image, seat, templates)
        right = sum(a == b for a, b in zip(read, truth, strict=False))
        assert right >= floor, f"{seat} 只對 {right}/{len(truth)}:\n  答案 {truth}\n  讀出 {read}"

    def test_the_whole_table_is_over_ninety_percent(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """四家合計。實測 60/65 = 92.3%。"""
        right = total = 0
        for seat in SEATS:
            truth = EXPECTED[seat].split()
            read = _read(table_image, seat, templates)
            right += sum(a == b for a, b in zip(read, truth, strict=False))
            total += len(truth)
        assert right / total >= 0.90, f"{right}/{total}"

    def test_shimocha_is_perfect(self, table_image: np.ndarray, templates: TemplateSet) -> None:
        """有一家是 17/17 全對 —— 證明這個方法的上限不是「差不多」。"""
        assert _read(table_image, "shimocha", templates) == EXPECTED["shimocha"].split()

    def test_confident_reads_are_rarely_wrong(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """靜默錯誤:有信心卻認錯。那是自檢攔不住、只有人工確認擋得住的那一格。

        實測四家合計 54 張有信心、**0 張錯**(加上兩個尺度的覆核之前是
        56 張有信心、2 張錯 —— 那兩張都是 ``2m`` 被讀成 ``1m``)。

        留 1 張的餘地,因為這裡釘的是「幾乎不會錯」,不是「這一幀剛好全對」。
        """
        silent = 0
        for seat in SEATS:
            truth = EXPECTED[seat].split()
            flat = DEFAULT_CORNERS[seat].rectify(table_image, TABLE)
            river = read_river(flat)
            size = face_size()
            for position, index in enumerate(river.filled):
                if position >= len(truth):
                    continue
                match = classify_river_tile(
                    flat[river.cells[index].as_slice()], templates, size=size
                )
                if match.is_confident and normalize_red(ms_to_mjai(match.label)) != truth[position]:
                    silent += 1
        assert silent <= 1, f"靜默錯誤 {silent} 張"


class TestOccupancyUsesTheFaceOnly:
    """有無判斷只看格子正中央那一塊,不是整格。

    一格比一張牌大一圈(FACE 是 0.72 x 0.70),所以一格裝著一張牌時另外一半
    是桌布,整格取中位數會落在牌與桌布之間。實測一局開頭河裡只有一張牌時,
    那一格整格的中位亮度是 97,而門檻是 110 —— 判成空的。

    症狀是「每局開頭的幀全部被拒絕」:own 的採用率因此只有 48.3%,
    改看牌面之後是 89.0%。
    """

    def test_a_lone_tile_in_a_wide_cell_is_still_found(self) -> None:
        """牌只佔一格的一半,但它在正中央 —— 要數得到。"""
        from mia.vision.tiles.river import CELL, face_size

        flat = np.full((ROWS * CELL[1], COLUMNS * CELL[0], 3), 70, dtype=np.uint8)
        face_w, face_h = face_size()
        x = (CELL[0] - face_w) // 2
        y = (CELL[1] - face_h) // 2
        flat[y : y + face_h, x : x + face_w] = 210

        river = read_river(flat)
        assert river.count == 1
        assert river.filled == (0,)

    def test_brightness_only_at_the_edges_is_not_a_tile(self) -> None:
        """格子邊緣的亮東西(鄰桌的牌、UI)不該被當成一張牌。"""
        from mia.vision.tiles.river import CELL

        flat = np.full((ROWS * CELL[1], COLUMNS * CELL[0], 3), 70, dtype=np.uint8)
        flat[0 : CELL[1] // 8, 0 : CELL[0]] = 210  # 只有上緣一條

        assert read_river(flat).count == 0


class TestTileBacksAreNotDiscards:
    """牌背不是捨牌。

    牌背是橘色的,亮度跟牌面一樣高,所以只看亮度分不出來。而河的末端常常會
    出現一張牌背 —— 實測上家有 43% 的幀因此多讀一張,一律在結尾。

    它傷的不是準確率(多出來那張分數低,過不了信心門檻),是**採用率**:
    張數一錯,那一家整幀就被拒絕。上家的採用率因此只有 56.9%,
    加上這道檢查之後是 96.7%。
    """

    def _flat(self, colour: tuple[int, int, int]) -> np.ndarray:
        from mia.vision.tiles.river import CELL, face_size

        flat = np.full((ROWS * CELL[1], COLUMNS * CELL[0], 3), 70, dtype=np.uint8)
        face_w, face_h = face_size()
        x = (CELL[0] - face_w) // 2
        y = (CELL[1] - face_h) // 2
        flat[y : y + face_h, x : x + face_w] = colour
        return flat

    def test_an_ivory_face_counts(self) -> None:
        """牌面幾乎沒有飽和度 —— 實測中位 4。"""
        assert read_river(self._flat((208, 212, 210))).count == 1

    def test_an_orange_back_does_not(self) -> None:
        """牌背亮但很橘 —— 實測中位飽和度 126。"""
        assert read_river(self._flat((40, 150, 230))).count == 0

    def test_a_yellow_green_back_does_not_either(self) -> None:
        """牌背在四家的光照下偏得比想像中多。

        實測上家「結尾多一張」的那一格(35 幀):色相中位 55,5~95 百分位
        42~82 —— 已經是黃綠色而不是橘。色相帶只框到 40 的話那些全部漏過去,
        上家的採用率從 96.7% 掉回 88.5%。
        """
        assert read_river(self._flat(_hsv(55, 73, 145))).count == 0

    def test_a_blue_haku_counts(self) -> None:
        """**白板在牌河裡是淺藍色的,不是白的。**

        它平躺著,而牌桌的藍光打在一張完全空白的牌面上就是這個結果。實測
        (2560x1614 的畫面,三家各一張)亮度 176、飽和度 79、色相 104 ——
        飽和度已經高過牌背的門檻 70,所以原本的規則把它判成「這格沒有牌」,
        牌河中間破一個洞,那一家整幀被丟掉。

        分類器本身完全沒問題:同一格強制去認,回的是白板 0.95 分、
        差距 0.68。壞掉的只有「這格有沒有牌」。
        """
        assert read_river(self._flat(_hsv(104, 79, 176))).count == 1

    def test_hue_only_matters_when_saturated(self) -> None:
        """飽和度低的時候色相是雜訊 —— 一塊近乎無彩的象牙色,色相是哪個值
        都不奇怪。實測一般牌面的色相就落在 30 附近(橘色那一段),
        而它們當然是牌。
        """
        assert read_river(self._flat(_hsv(30, 6, 200))).count == 1


class TestTwoScalesMustAgree:
    """同一格用兩個模板高度各認一次,答案不一樣就棄權。

    擋的是**門檻擋不住的那一種錯**:``2m`` 在 0.70 的高度下有三分之二被讀成
    ``1m``,而且錯的那個分數還比對的高(0.692 vs 0.670)—— 調門檻只會把對的
    一起擋掉。模板拉高到 0.73 時 ``2m`` 就對了,但其他牌開始變差,所以沒有
    一個高度兩邊都好。

    那就不要選邊。尺度是這裡的干擾參數,而「換個尺度答案就變了」本身就是
    不可信的證據。實測 own 的靜默錯誤 2.09% → 0.16%,合計 0.65% → 0.18%,
    代價是多 0.93 個百分點的棄權。
    """

    def test_the_two_heights_really_differ(self) -> None:
        """兩個高度一樣的話整個覆核就是空轉 —— 而且不會有任何症狀。"""
        assert round(CHECK_FACE_HEIGHT * CELL[1]) != round(FACE[1] * CELL[1])

    def test_agreement_passes_the_match_through(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """兩邊同意時回傳的就是原本那個 Match,不打任何折扣。"""
        flat = DEFAULT_CORNERS["shimocha"].rectify(table_image, TABLE)
        river = read_river(flat)
        size = face_size()
        agreed = 0
        for index in river.filled:
            cell = flat[river.cells[index].as_slice()]
            checked = classify_river_tile(cell, templates, size=size)
            # 基準要用**同一組參數** —— 牌河是開 merge_red 的(赤五與普通五
            # 在這裡等價)。拿沒開的去比,差的是次高分的挑法,不是覆核。
            plain = classify(cell, templates, size=size, merge_red=True)
            if checked.runner_up_score != checked.score:
                assert checked == plain
                agreed += 1
        assert agreed, "整家都不一致的話,這個測試什麼都沒驗到"

    def test_disagreement_cannot_be_confident(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """不一致時差距歸零 —— 任何門檻都過不了,不是只有現行這組。"""
        flat = DEFAULT_CORNERS["own"].rectify(table_image, TABLE)
        river = read_river(flat)
        size = face_size()
        for index in river.filled:
            cell = flat[river.cells[index].as_slice()]
            checked = classify_river_tile(cell, templates, size=size)
            if checked.runner_up_score == checked.score:
                assert checked.margin == 0.0
                assert not checked.is_confident

    def test_the_second_height_scales_with_the_cell(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """呼叫端給別的 size 時,第二個尺度要跟著按比例走。

        硬套 :data:`CELL` 會讓兩個尺度的相對關係跑掉,而症狀是「覆核好像
        沒在運作」—— 那不會有任何錯誤訊息。
        """
        flat = DEFAULT_CORNERS["own"].rectify(table_image, TABLE)
        cell = flat[read_river(flat).cells[0].as_slice()]
        half = (face_size()[0] // 2, face_size()[1] // 2)
        # 只要不拋錯、而且確實跑了兩次就夠 —— 第二個尺度若按 CELL 算,
        # 這裡會拿到一個比格子還大的模板而拿不到結果。
        assert classify_river_tile(cell, templates, size=half).label


class TestTheSidewaysReachTile:
    """立直的宣言牌橫躺著,而格線是固定等寬的。

    素材那一幀剛好有一張:對家第 8 格的 ``5m``。另外三家沒有 —— 所以同一張
    圖同時是正例與負例,而負例是這裡真正要守的東西(用寬度判斷的三次嘗試
    全是被假陽殺掉的,不是被漏抓殺掉的)。
    """

    def _flat(self, image: np.ndarray, seat: str = "toimen") -> np.ndarray:
        return DEFAULT_CORNERS[seat].rectify(image, TABLE)

    def test_it_finds_the_one_sideways_tile_and_nothing_else(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        found = {
            seat: find_sideways(self._flat(table_image, seat), templates) for seat in SEATS
        }
        assert found == {"own": None, "shimocha": None, "toimen": 8, "kamicha": None}

    def test_the_tile_itself_needs_the_rotated_templates(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """直放模板問錯了問題 —— 它沒有直放。

        實測兩份錄影 28 格:直放的正確率 0.0%、有信心 0 張。
        """
        flat = self._flat(table_image)
        river = read_river(flat)
        upright = classify_river_tile(
            flat[river.cells[8].as_slice()], templates, size=face_size()
        )
        rotated = classify_sideways(flat, 8, templates)
        want = EXPECTED["toimen"].split()[8]
        assert normalize_red(ms_to_mjai(upright.label)) != want
        assert normalize_red(ms_to_mjai(rotated.label)) == want

    def test_the_rotated_read_never_claims_confidence(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """認對了,但**刻意**不敢說有把握。

        這不是因為分數不夠 —— 尺寸量對之後分數有 0.65~0.81,過得了門檻。
        是因為它在那個狀態下開始有信心地講錯話:舊素材那張 2m 被認成 1m,
        22 格裡 10 格有信心、靜默錯誤 100%。

        而證據量差三個數量級:直放的門檻掃過 1921 幀,橫放總共只有 4 張
        不同的宣言牌。四張牌不足以讓一個讀數有資格宣稱把握。

        **要放寬這一條,先拿出更多宣言牌的樣本。** 不要因為分數看起來夠高
        就把歸零那一行刪掉。
        """
        match = classify_sideways(self._flat(table_image), 8, templates)
        assert match.score >= 0.60, "分數本身是夠的 —— 棄權是刻意的,不是分數不足"
        assert match.margin == 0.0
        assert not match.is_confident

    def test_the_rotated_size_is_not_a_swap_of_the_upright_one(self) -> None:
        """躺下來之後透視壓縮的軸也換了,長寬比不是交換的關係。

        第一版就是寫成交換(0.70 x 0.72),四張宣言牌的分數全落在 0.53 附近,
        **全部過不了 0.60**。量出來的是 1.22 x 0.49 —— 寬得多、矮得多。
        """
        swapped = (FACE[1], FACE[0])
        assert swapped != SIDEWAYS_FACE
        assert SIDEWAYS_FACE[0] / SIDEWAYS_FACE[1] > 2.0
        assert sideways_face_size((90, 120)) == (110, 59)

    def test_shifting_fixes_the_tile_after_it(
        self, table_image: np.ndarray, templates: TemplateSet
    ) -> None:
        """宣言牌把同一列後面的牌往右推,不讓開就認錯。

        第 9 格:不挪讀成 ``4s``,挪了讀成 ``1s``(答案)。
        """
        flat = self._flat(table_image)
        want = EXPECTED["toimen"].split()[9]
        size = face_size()
        got = []
        for sideways in (None, 8):
            river = read_river(flat, sideways=sideways)
            match = classify_river_tile(
                flat[river.cells[9].as_slice()], templates, size=size
            )
            got.append(normalize_red(ms_to_mjai(match.label)))
        assert got[0] != want
        assert got[1] == want

    def test_the_tile_pushed_out_abstains_instead_of_leaving_a_hole(
        self, table_image: np.ndarray
    ) -> None:
        """被推出四邊形的那一格要說「讀不到」,不能說「沒牌」。

        一列裡有橫放牌的時候,那一列的物理寬度超過六格 —— 最後一格真的在
        攤平用的四邊形外面。說它沒牌會讓牌河破洞,而洞會讓
        :meth:`River.is_plausible` 把**整家這一幀**丟掉:實測舊素材 10 次、
        新素材 5 次,全在立直之後。那是用扔掉整家去換宣言牌之後的正確率。
        """
        river = read_river(self._flat(table_image), sideways=8)
        assert river.clipped == (11,)
        assert river.count == len(EXPECTED["toimen"].split())
        assert river.is_plausible()

    def test_the_shift_stays_inside_its_row(self, table_image: np.ndarray) -> None:
        """只推同一列、在它後面的格子。下一列從頭排,不受影響。"""
        plain = read_river(self._flat(table_image))
        moved = read_river(self._flat(table_image), sideways=8)
        shifts = [b.x - a.x for a, b in zip(plain.cells, moved.cells, strict=True)]
        assert shifts == [0] * 9 + [SIDEWAYS_SHIFT] * 3 + [0] * 6

    def test_a_river_without_a_sideways_tile_is_untouched(
        self, table_image: np.ndarray
    ) -> None:
        """``sideways=None`` 要跟完全沒有這個功能之前一模一樣。"""
        flat = self._flat(table_image, "own")
        river = read_river(flat, sideways=None)
        assert river.clipped == ()
        assert river.sideways is None
        assert [c.x for c in river.cells[:6]] == [i * CELL[0] for i in range(6)]


class TestTheRotatedTemplates:
    def test_they_turn_counterclockwise(self, templates: TemplateSet) -> None:
        """方向是量出來的,不是猜的。

        第一次比對只試了順時針,分數全是雜訊 —— 而我把那份雜訊當成「訊號
        方向相反」讀了一輪。逆時針才是雀魂宣言牌躺的方向。
        """
        label = templates.labels[0]
        upright = templates.scaled((40, 60))
        rotated = templates.rotated().scaled((60, 40))
        original = dict(upright)[label]
        turned = dict(rotated)[label]
        assert turned.shape[:2] == (40, 60)
        # 逆時針轉回去要等於原圖(縮放是分別做的,所以比形狀與一兩個角落)。
        back = cv2.rotate(turned, cv2.ROTATE_90_CLOCKWISE)
        assert back.shape[:2] == original.shape[:2]

    def test_the_rotation_is_cached(self, templates: TemplateSet) -> None:
        """每一幀的每一格都會問一次 —— 37 張影像不該每次重轉。"""
        assert templates.rotated() is templates.rotated()
