"""局的邊界:四家牌河同時清空,然後重新開始長。

這裡用假的 River 而不是真影像 —— 要測的是**狀態機**,不是辨識。
辨識那一端由 test_vision_river.py 守著。
"""

from __future__ import annotations

from mia.tracker import KyokuSignal, KyokuWatcher
from mia.utils.geometry import Rect
from mia.vision.tiles.river import COLUMNS, ROWS, River


def _river(count: int, *, holes: tuple[int, ...] = ()) -> River:
    """一家牌河。``holes`` 讓它不連續 —— 那是結算畫面的特徵。"""
    occupied = [i < count for i in range(COLUMNS * ROWS)]
    for hole in holes:
        occupied[hole] = False
        occupied[hole + 1] = True
    cells = tuple(Rect(0, 0, 1, 1) for _ in occupied)
    return River(cells=cells, occupied=tuple(occupied))


def _table(*counts: int) -> list[River]:
    return [_river(c) for c in counts]


class TestItFindsTheBoundary:
    def test_the_first_tile_after_an_empty_table_starts_a_kyoku(self) -> None:
        watcher = KyokuWatcher()
        assert watcher.observe(_table(0, 0, 0, 0)) is KyokuSignal.NONE
        assert watcher.observe(_table(1, 0, 0, 0)) is KyokuSignal.STARTED
        assert watcher.started == 1

    def test_it_only_fires_once(self) -> None:
        """整局都在長牌 —— 每一幀都宣告新局的話會一直清空狀態。"""
        watcher = KyokuWatcher()
        watcher.observe(_table(1, 0, 0, 0))
        for count in range(2, 10):
            assert watcher.observe(_table(count, 3, 3, 3)) is KyokuSignal.NONE

    def test_clearing_the_table_ends_the_kyoku(self) -> None:
        watcher = KyokuWatcher()
        watcher.observe(_table(1, 0, 0, 0))
        watcher.observe(_table(6, 6, 6, 6))
        assert watcher.observe(_table(0, 0, 0, 0)) is KyokuSignal.ENDED

    def test_a_second_kyoku(self) -> None:
        watcher = KyokuWatcher()
        watcher.observe(_table(1, 0, 0, 0))
        watcher.observe(_table(0, 0, 0, 0))
        assert watcher.observe(_table(0, 1, 0, 0)) is KyokuSignal.STARTED
        assert watcher.started == 2


class TestTheScoringScreenIsNotANewKyoku:
    """和了之後整桌被立繪與結算面板蓋住。

    **張數不是掉到零,是亂跳** —— 角色膚色與牌面的判斷條件一模一樣
    (亮、不飽和)。實測結算畫面讀到 5/3/1/4 共 14 張。分得開的是連續性:
    那些假牌散落在格子裡,四家全部不連續。
    """

    def test_garbage_between_kyoku_is_ignored(self) -> None:
        watcher = KyokuWatcher()
        watcher.observe(_table(6, 6, 6, 6))   # 這一局算一局
        watcher.observe(_table(0, 0, 0, 0))
        garbage = [_river(c, holes=(1,)) for c in (5, 3, 1, 4)]
        assert watcher.observe(garbage) is KyokuSignal.NONE
        assert watcher.started == 1, "結算畫面不該被算成新的一局"

    def test_a_real_start_still_fires_after_the_garbage(self) -> None:
        watcher = KyokuWatcher()
        watcher.observe(_table(0, 0, 0, 0))
        watcher.observe([_river(c, holes=(1,)) for c in (5, 3, 1, 4)])
        assert watcher.observe(_table(0, 0, 1, 0)) is KyokuSignal.STARTED


class TestTheGapCanBeLong:
    def test_an_empty_table_stays_quiet(self) -> None:
        """實測有一局空了 **56 秒**(有人長考,牌河空無一物)。

        那段期間每一幀都要是 NONE —— 重複宣告會讓下游一直清狀態。
        對局結束後的空桌也一樣(實測連續四十秒),而那之後根本沒有牌跟上來。
        """
        watcher = KyokuWatcher()
        watcher.observe(_table(6, 6, 6, 6))
        assert watcher.observe(_table(0, 0, 0, 0)) is KyokuSignal.ENDED
        for _ in range(60):
            assert watcher.observe(_table(0, 0, 0, 0)) is KyokuSignal.NONE
        assert watcher.observe(_table(1, 0, 0, 0)) is KyokuSignal.STARTED
