"""牌河的觀察 → 捨牌事件。

這一層的三個難處都有對應的測試:鳴牌造成的遞補、讀不到的格子、
一幀之間發生不只一件事。每一個都是「天真的 diff 會錯」的地方。
"""

from __future__ import annotations

from mia.tracker import UNKNOWN, Discard, RiverWatcher


class TestItTurnsGrowthIntoDiscards:
    def test_the_first_tile(self) -> None:
        watcher = RiverWatcher()
        assert watcher.observe(0, ["1m"]) == [Discard(position=0, index=0, pai="1m")]

    def test_nothing_changed_means_no_events(self) -> None:
        """同一個狀態被看到很多次 —— 15 fps 下這是常態,不是例外。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        assert watcher.observe(0, ["1m"]) == []
        assert watcher.observe(0, ["1m"]) == []

    def test_two_tiles_at_once_come_out_in_order(self) -> None:
        """掉幀或動畫遮擋之後,河可能一次多兩張。河是有序的,照序補發。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        assert watcher.observe(0, ["1m", "2m", "3m"]) == [
            Discard(position=0, index=1, pai="2m"),
            Discard(position=0, index=2, pai="3m"),
        ]

    def test_each_seat_counts_separately(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        assert watcher.observe(1, ["9p"]) == [Discard(position=1, index=0, pai="9p")]


class TestCallsShortenTheRiver:
    """鳴牌會把中間那張拿走,後面整排往前遞補。

    天真的 diff 在這裡會製造一連串假事件,而且之後每一張都對不上 ——
    症狀看起來像「辨識突然變爛」,但辨識是對的。
    """

    def test_a_called_tile_is_not_a_new_discard(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        assert watcher.observe(0, ["1m", "3m"]) == []
        assert watcher.stats.calls == 1

    def test_the_river_shifts_forward_not_leaves_a_hole(self) -> None:
        """雀魂不留空位 —— 所以之後接上去的那張是第 2 格,不是第 3 格。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        watcher.observe(0, ["1m", "3m"])
        assert watcher.observe(0, ["1m", "3m", "4m"]) == [
            Discard(position=0, index=2, pai="4m")
        ]

    def test_a_call_at_the_end(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        assert watcher.observe(0, ["1m"]) == []
        assert watcher.stats.calls == 1


class TestUnknownIsDeferredNotDropped:
    """讀不到的格子**在張數上算數**,只是內容待定。

    省略它會讓後面的位置全部錯開;當成「沒有牌」會讓下一張被當成同一格。
    """

    def test_an_unreadable_tile_still_counts_as_a_discard(self) -> None:
        watcher = RiverWatcher()
        assert watcher.observe(0, [UNKNOWN]) == [
            Discard(position=0, index=0, pai=UNKNOWN)
        ]

    def test_a_later_frame_fills_it_in(self) -> None:
        """牌還在桌上 —— 棄權是延後,不是放棄。"""
        watcher = RiverWatcher()
        watcher.observe(0, [UNKNOWN])
        watcher.observe(0, ["5s"])
        assert watcher.rivers[0] == ["5s"]
        assert watcher.stats.resolved == 1

    def test_filling_in_does_not_re_emit(self) -> None:
        """補上內容不是新的捨牌 —— 重發會讓下游數到兩張。"""
        watcher = RiverWatcher()
        watcher.observe(0, [UNKNOWN])
        assert watcher.observe(0, ["5s"]) == []

    def test_an_unknown_in_the_middle_does_not_block_growth(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", UNKNOWN])
        assert watcher.observe(0, ["1m", UNKNOWN, "3m"]) == [
            Discard(position=0, index=2, pai="3m")
        ]

    def test_a_confirmed_tile_is_not_overwritten(self) -> None:
        """已經確定的格子不會被後來的讀數推翻。

        那種不一致通常代表**對齊**出了問題,不是前一次讀錯。改掉它會讓
        一次攤平歪掉污染已經正確的歷史。
        """
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        watcher.observe(0, ["9p"])
        assert watcher.rivers[0] == ["1m"]


class TestItRefusesToGuess:
    def test_an_impossible_jump_is_dropped(self) -> None:
        """看不懂就整筆丟掉 —— 少發一個只是延後,猜錯會污染整條事件流。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        assert watcher.observe(0, ["7z", "7z"]) == []
        assert watcher.stats.confused == 1
        assert watcher.rivers[0] == ["1m", "2m", "3m"]

    def test_losing_two_at_once_is_not_a_call(self) -> None:
        """一次只會被鳴走一張。少兩張代表讀錯了,不是鳴了兩次。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        assert watcher.observe(0, ["1m"]) == []
        assert watcher.stats.confused == 1

    def test_reset_clears_every_seat(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        watcher.observe(2, ["9p"])
        watcher.reset()
        assert watcher.rivers == [[], [], [], []]
