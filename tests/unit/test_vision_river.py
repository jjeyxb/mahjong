"""牌河的格子幾何與有無判斷。

素材 ``river_own_15.png`` 是真實遊戲畫面裁出來的 own 牌河 ROI(514x288),
河裡 15 張:遠排與中排各六張,近排三張。**牌面認成什麼不在這裡測** ——
那是模板比對的事,這裡只管「切在哪、哪幾格有牌」。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.vision.tiles.classify import TemplateSet
from mia.vision.tiles.river import (
    COLUMNS,
    DEFAULT_RIVER,
    ROWS,
    River,
    RowLayout,
    read_river,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE = FIXTURES / "river_own_15.png"


@pytest.fixture(scope="module")
def river_image() -> np.ndarray:
    image = cv2.imread(str(SAMPLE))
    assert image is not None, f"讀不到素材 {SAMPLE}"
    return image


class TestRowLayout:
    def test_cells_step_by_the_pitch(self) -> None:
        row = RowLayout(origin=0.0, pitch=0.1, top=0.0, height=0.4)
        first = row.cell((1000, 200), 0)
        second = row.cell((1000, 200), 1)
        assert second.x - first.x == 100

    def test_the_band_is_taller_than_a_tile(self) -> None:
        """垂直方向刻意給得寬 —— 貼著切會把 matchTemplate 的平移餘裕拿掉。

        實測同一張「北」在寬鬆格子裡 0.845,貼著切掉到 0.403。
        """
        cell = DEFAULT_RIVER.rows[0].cell((514, 288), 0)
        assert cell.height > cell.width


class TestRiverLayout:
    def test_the_index_runs_row_major(self) -> None:
        """由遠而近、由左而右 —— 與捨牌順序一致。"""
        size = (514, 288)
        assert DEFAULT_RIVER.cell(size, 0).y == DEFAULT_RIVER.cell(size, COLUMNS - 1).y
        assert DEFAULT_RIVER.cell(size, COLUMNS).y > DEFAULT_RIVER.cell(size, 0).y

    def test_rows_get_wider_towards_the_viewer(self) -> None:
        """牌河平躺著,鏡頭是斜的 —— 遠排看起來比近排窄。實測差 4%。"""
        pitches = [row.pitch for row in DEFAULT_RIVER.rows]
        assert pitches == sorted(pitches), "由遠而近應該越來越寬"

    def test_an_index_past_the_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="18 格"):
            DEFAULT_RIVER.cell((514, 288), COLUMNS * ROWS)

    def test_the_face_is_smaller_than_its_cell(self) -> None:
        """格子含著牌的前緣與牌縫,牌面只佔其中一部分。"""
        size = (514, 288)
        width, height = DEFAULT_RIVER.face_size(size, 0)
        cell = DEFAULT_RIVER.cell(size, 0)
        assert 0 < width < cell.width
        assert 0 < height < cell.height


class TestReadRiver:
    def test_it_finds_the_fifteen_tiles(self, river_image: np.ndarray) -> None:
        assert read_river(river_image).count == 15

    def test_the_empty_cells_are_the_last_three(self, river_image: np.ndarray) -> None:
        """近排只有三張,所以最後三格是空的。"""
        occupied = read_river(river_image).occupied
        assert occupied[:15] == (True,) * 15
        assert occupied[15:] == (False,) * 3

    def test_a_real_river_is_plausible(self, river_image: np.ndarray) -> None:
        assert read_river(river_image).is_plausible()

    def test_a_hole_in_the_middle_is_not_plausible(self) -> None:
        """牌河依序往後填,中間不會有洞 —— 有洞就是切歪了或抓在動畫中間。"""
        occupied = tuple([True, True, False, True] + [False] * 14)
        assert not River(cells=(), occupied=occupied).is_plausible()

    def test_an_empty_river_is_plausible(self) -> None:
        assert River(cells=(), occupied=(False,) * 18).is_plausible()

    def test_a_greyscale_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="BGR"):
            read_river(np.zeros((288, 514), dtype=np.uint8))


class TestAgainstTheTemplates:
    """幾何對不對,最後還是要看牌認不認得出來。

    這裡不釘住**認成什麼**(那是分類的事),只釘住「大多數格子過得了信心門檻」
    —— 幾何一歪,這個數字會立刻掉下來,而那正是這組常數最容易壞掉的方式。
    """

    def test_most_cells_are_confidently_recognised(self, river_image: np.ndarray) -> None:
        templates = TemplateSet.load()
        river = read_river(river_image)
        size = (river_image.shape[1], river_image.shape[0])
        sure = 0
        for index in river.filled:
            patch = river_image[river.cells[index].as_slice()]
            scaled = templates.scaled(DEFAULT_RIVER.face_size(size, index))
            scores = sorted(
                (float(cv2.matchTemplate(patch, face, cv2.TM_CCOEFF_NORMED).max()), label)
                for label, face in scaled
            )
            best, second = scores[-1], scores[-2]
            if best[0] >= 0.55 and best[0] - second[0] >= 0.04:
                sure += 1
        assert sure >= 11, f"只有 {sure}/15 過關 —— 幾何可能歪了"

    def test_the_far_row_reads_correctly(self, river_image: np.ndarray) -> None:
        """遠排是目視確認過的:北 一索 發 八萬 中。

        第六張刻意不列 —— 索子的點數目視分不出來,而一個標錯的期望值比
        沒有期望值更糟。
        """
        templates = TemplateSet.load()
        river = read_river(river_image)
        size = (river_image.shape[1], river_image.shape[0])
        read = []
        for index in range(5):
            patch = river_image[river.cells[index].as_slice()]
            scaled = templates.scaled(DEFAULT_RIVER.face_size(size, index))
            best = max(
                (float(cv2.matchTemplate(patch, face, cv2.TM_CCOEFF_NORMED).max()), label)
                for label, face in scaled
            )
            read.append(best[1])
        assert read == ["4z", "1s", "6z", "8m", "7z"]


class TestUpright:
    """把四家轉成同一個方向。

    素材是同一幀裡四家各自的 river ROI。轉正的方向是目視確認的 ——
    轉完之後牌面上的字是正的 —— 這裡再用模板分數交叉驗一次:正確的方向
    該明顯優於其餘三個,而不是剛好贏一點點。
    """

    @pytest.mark.parametrize("seat", ["own", "shimocha", "toimen", "kamicha"])
    def test_every_seat_has_a_rotation(self, seat: str) -> None:
        from mia.vision.tiles.river import ROTATION, SEATS, upright

        assert seat in SEATS
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        assert upright(image, seat).ndim == 3
        assert seat in ROTATION

    def test_own_is_already_upright(self) -> None:
        from mia.vision.tiles.river import ROTATION, upright

        assert ROTATION["own"] is None
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        assert upright(image, "own") is image, "不轉就沒有複製的理由"

    def test_a_side_seat_swaps_the_axes(self) -> None:
        from mia.vision.tiles.river import upright

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        assert upright(image, "kamicha").shape[:2] == (30, 20)

    def test_an_unknown_seat_is_refused(self) -> None:
        from mia.vision.tiles.river import upright

        with pytest.raises(ValueError, match="不認得的座位"):
            upright(np.zeros((4, 4, 3), dtype=np.uint8), "toimen_river")

    @pytest.mark.parametrize("seat", ["toimen", "kamicha", "shimocha"])
    def test_the_chosen_rotation_beats_the_other_three(self, seat: str) -> None:
        """四個方向各比一次,正確的那個分數要最高。

        這是在釘住 ROTATION 的內容。轉錯方向的症狀是「準確率莫名其妙地低」,
        而那看起來跟幾何沒調好一模一樣 —— 分不出來就會往錯的方向修。
        """
        from mia.vision.tiles.river import ROTATION, upright

        source = cv2.imread(str(FIXTURES / f"river_{seat}.png"))
        assert source is not None
        templates = TemplateSet.load()

        def mean_top_score(image: np.ndarray) -> float:
            river = read_river(image)
            if not river.filled:
                return 0.0
            size = (image.shape[1], image.shape[0])
            total = 0.0
            for index in river.filled:
                scaled = templates.scaled(DEFAULT_RIVER.face_size(size, index))
                total += max(
                    float(
                        cv2.matchTemplate(
                            image[river.cells[index].as_slice()], face, cv2.TM_CCOEFF_NORMED
                        ).max()
                    )
                    for _label, face in scaled
                )
            return total / len(river.filled)

        chosen = mean_top_score(upright(source, seat))
        others = [
            mean_top_score(source if rot is None else cv2.rotate(source, rot))
            for rot in (
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )
            if rot != ROTATION[seat]
        ]
        assert chosen > max(others), f"{seat}: 選的方向 {chosen:.3f} 不是最好的 {others}"
