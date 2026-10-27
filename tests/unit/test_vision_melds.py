"""副露帶的格線與連續性檢查。

第一階段只問「這一格是不是翻開的牌」,所以這裡測的是帶子的切法與棄權規則,
不是認牌 —— 認牌是第二階段,還沒做。
"""

from __future__ import annotations

import numpy as np

from mia.utils.geometry import Rect
from mia.vision.tiles.melds import (
    CELL,
    COLUMNS,
    DEFAULT_VIEWS,
    MeldStrip,
    MeldView,
    read_melds,
)

TABLE = Rect(0, 0, 1000, 1000)
#: 整個牌桌都是副露區,方便直接畫格子進去。
WHOLE = MeldView(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))


def _table(faces: int) -> np.ndarray:
    """一張 1000x1000 的「桌子」,左邊 ``faces`` 格畫成翻開的牌。"""
    image = np.full((1000, 1000, 3), (90, 60, 30), dtype=np.uint8)  # 暗藍桌布
    for i in range(faces):
        x0 = round(i * 1000 / COLUMNS)
        x1 = round((i + 1) * 1000 / COLUMNS) - 2
        image[:, x0:x1] = (235, 240, 240)  # 象牙色牌面
    return image


class TestTheStrip:
    def test_it_counts_the_face_up_tiles(self) -> None:
        strip = read_melds(_table(9), TABLE, WHOLE)
        assert strip.count == 9
        assert strip.filled == tuple(range(9))

    def test_an_empty_meld_area_reads_zero(self) -> None:
        assert read_melds(_table(0), TABLE, WHOLE).count == 0

    def test_the_strip_is_long_enough_for_four_kans(self) -> None:
        """四組槓是 16 張,但第四張疊在第三張**上面**,畫面上仍是三張寬。

        所以上限是 12 格,留 13 ——寧可末端多一格空的,也不要在真的排滿時
        切掉最後一組。
        """
        assert COLUMNS >= 4 * 3


class TestItAbstainsRatherThanGuess:
    """副露從桌角往外長,中間不會有洞。有洞就是攤平歪了或被蓋住。"""

    def test_a_contiguous_run_is_plausible(self) -> None:
        assert MeldStrip((True, True, True, False, False)).is_plausible()

    def test_a_hole_means_the_whole_seat_is_dropped(self) -> None:
        assert not MeldStrip((True, False, True, False)).is_plausible()

    def test_nothing_at_all_is_plausible(self) -> None:
        """還沒有人鳴牌是正常狀態,不是讀失敗。"""
        assert MeldStrip((False,) * COLUMNS).is_plausible()


class TestTheMeasuredCorners:
    def test_only_the_two_seats_with_samples_are_measured(self) -> None:
        """四家只有兩家 —— 這個測試是**提醒**,量好了就改它。

        * 下家:素材 0 次鳴牌,搜不了,只能補素材。
        * 自己:副露緊貼著**正面朝上**的手牌,is_face_up 分不開。
          但那一家正好最不需要 —— 見 DEFAULT_VIEWS 的說明。
        """
        assert set(DEFAULT_VIEWS) == {"toimen", "kamicha"}

    def test_own_is_deliberately_absent(self) -> None:
        """不是忘了,是**量不出來而且不需要**。

        放銃分析從不對自己算危險,所以自己的副露只餵「壁」,而少算壁是
        保守的方向(remaining 偏高 → 不觸發排除 → 那張仍算危險)。
        """
        assert "own" not in DEFAULT_VIEWS

    def test_the_corners_stay_on_the_table(self) -> None:
        for name, view in DEFAULT_VIEWS.items():
            for x, y in view.corners:
                assert 0.0 <= x <= 1.0, name
                assert 0.0 <= y <= 1.0, name

    def test_the_cell_is_small_enough_to_be_cheap(self) -> None:
        """只判有無,不比對模板 —— 一整條帶子要比牌河一格還便宜。"""
        assert CELL[0] * CELL[1] * COLUMNS < 90 * 120 * 6 * 3
