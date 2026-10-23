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
    COLUMNS,
    DEFAULT_CORNERS,
    ROWS,
    SEATS,
    River,
    RiverView,
    face_size,
    read_river,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

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
        match = classify(flat[river.cells[index].as_slice()], templates, size=size)
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

        實測四家合計 56 張有信心、2 張錯。
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
                match = classify(flat[river.cells[index].as_slice()], templates, size=size)
                if match.is_confident and normalize_red(ms_to_mjai(match.label)) != truth[position]:
                    silent += 1
        assert silent <= 3, f"靜默錯誤 {silent} 張"


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
