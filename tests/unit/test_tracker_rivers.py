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
        """掉幀或動畫遮擋之後,河可能一次多兩張。河是有序的,照序補發 ——
        但要等下一次觀察確認過,見 :class:`TestAFlashIsNotADiscard`。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        assert watcher.observe(0, ["1m", "2m", "3m"]) == []
        assert watcher.observe(0, ["1m", "2m", "3m"]) == [
            Discard(position=0, index=1, pai="2m"),
            Discard(position=0, index=2, pai="3m"),
        ]

    def test_each_seat_counts_separately(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        assert watcher.observe(1, ["9p"]) == [Discard(position=1, index=0, pai="9p")]


class TestAFlashIsNotADiscard:
    """和了 / 立直的過場動畫有一道白光橫過牌河,空格被讀成**有信心的白**。

    發出去就是一張假現物,而且收不回來。重播校正素材,42 輪的假現物全部是
    這樣來的。那幾幀一次多了好幾格,正常的出牌一次只多一張。
    """

    def test_a_flash_that_disappears_emits_nothing(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        assert watcher.observe(0, ["1m", "2m", "P", "P", "P"]) == []
        assert watcher.observe(0, ["1m", "2m"]) == []
        assert watcher.rivers[0] == ["1m", "2m"]

    def test_one_new_tile_is_not_delayed(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        assert watcher.observe(0, ["1m", "2m", "P"]) == [Discard(position=0, index=2, pai="P")]

    def test_a_confirmed_jump_may_keep_growing(self) -> None:
        """確認的那一次河又多了一張 —— 還是同一條河,一起發。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m"])
        watcher.observe(0, ["1m", "2m", "3m"])
        assert [d.pai for d in watcher.observe(0, ["1m", "2m", "3m", "4m"])] == ["2m", "3m", "4m"]

    def test_starting_from_an_empty_river_is_not_delayed(self) -> None:
        """開局中途接上,一開始就看到滿河。"""
        watcher = RiverWatcher()
        assert len(watcher.observe(0, ["1m", "2m", "3m"])) == 3

    def test_a_full_river_that_cannot_be_read_is_not_a_river(self) -> None:
        """開局前的白色介面:連續 18 格、只認得出 1 格,而那一格是白。

        延後確認擋不住 —— 中間幾輪整家不連續、沒有觀察,前後兩次看到的
        是同一個介面。
        """
        watcher = RiverWatcher()
        screen = [UNKNOWN] * 10 + ["P"] + [UNKNOWN] * 7
        assert watcher.observe(0, screen) == []
        assert watcher.observe(0, screen) == []
        assert watcher.rivers[0] == []
        # 真的牌河出現時照常收。
        assert [d.pai for d in watcher.observe(0, ["9p"])] == ["9p"]


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

    def test_a_call_at_the_end_waits_for_a_second_look(self) -> None:
        """最後一張被鳴走**沒有位置上的證據** —— 前面每一格都還在原位。

        而「有一格這次沒被偵測到」長得一模一樣。兩者在單一次觀察上無法區分,
        所以先不動,看第二次。慢一輪(一秒)換掉一張假現物,划得來。
        """
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        assert watcher.observe(0, ["1m"]) == []
        assert (watcher.stats.calls, watcher.stats.deferred) == (0, 1)

        assert watcher.observe(0, ["1m"]) == []
        assert watcher.stats.calls == 1
        assert watcher.rivers[0] == ["1m"]


class TestAShortReadIsNotACall:
    """少讀到一格**不是**鳴牌。

    這兩件事在資料上長得一樣,而猜錯的代價天差地遠:猜成鳴牌會把整排往前
    挪一位,之後回來的那張看起來是新的 —— 於是多發一次捨牌,放銃分析多出
    一張**假現物**(說「一定安全」的牌)。實測就是這樣漏的,見 _conflicts。
    """

    def test_an_unknown_first_cell_does_not_make_everything_shift(self) -> None:
        """河的第一格是 ``?`` 的時候,「拿掉第一格」無條件對得上形狀。

        這是實測踩到的那一個:``['?','4m','4p']`` 遇到只讀到兩格的那一幀。
        """
        watcher = RiverWatcher()
        watcher.observe(0, [UNKNOWN, "4m", "4p"])
        assert watcher.observe(0, [UNKNOWN, UNKNOWN]) == []
        assert watcher.rivers[0] == [UNKNOWN, "4m", "4p"]
        assert watcher.stats.deferred == 1

    def test_the_tile_that_comes_back_is_not_a_new_discard(self) -> None:
        """挪掉一位之後,回來的 ``4p`` 會被當成新的 —— 那就是假現物的來源。"""
        watcher = RiverWatcher()
        watcher.observe(0, [UNKNOWN, "4m", "4p"])
        watcher.observe(0, [UNKNOWN, UNKNOWN])
        assert watcher.observe(0, [UNKNOWN, UNKNOWN, "4p"]) == []
        assert watcher.rivers[0] == [UNKNOWN, "4m", "4p"]

    def test_a_shift_with_real_evidence_is_taken_at_once(self) -> None:
        """同一格兩邊都確定而且不一樣 —— 那就是真的挪過了,不必等。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        assert watcher.observe(0, ["1m", "3m"]) == []
        assert (watcher.stats.calls, watcher.stats.deferred) == (1, 0)

    def test_coming_back_cancels_the_pending_shrink(self) -> None:
        """變短了又長回來 → 本來就只是讀漏了,待定要清掉。

        不清的話下一次任何一個變短的河都會被第二次確認規則直接採信。
        """
        watcher = RiverWatcher()
        watcher.observe(0, [UNKNOWN, "4m", "4p"])
        watcher.observe(0, [UNKNOWN, UNKNOWN])
        watcher.observe(0, [UNKNOWN, "4m", "4p"])
        assert watcher.observe(0, [UNKNOWN, UNKNOWN]) == []
        assert watcher.rivers[0] == [UNKNOWN, "4m", "4p"]

    def test_the_next_discard_is_what_proves_the_call(self) -> None:
        """鳴牌最常見的樣子:拿走的是剛打出去那張,所以當下沒有位置證據。

        證據要等他**又打了一張**:長度補回來而最後一格換人了。那一次既不是
        前綴也不是少一張 —— 把上一次的假設套上去才說得通。

        不接這一段的代價實測過:那一家從此每次觀察都進 ``confused``,
        永遠對不回來,捨牌整段漏掉。
        """
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        assert watcher.observe(0, ["1m"]) == []  # 2m 被鳴走,先不動
        assert watcher.observe(0, ["1m", "3m"]) == [
            Discard(position=0, index=1, pai="3m")
        ]
        assert watcher.rivers[0] == ["1m", "3m"]
        assert (watcher.stats.calls, watcher.stats.confused) == (1, 0)

    def test_a_hypothesis_that_never_fits_is_not_forced(self) -> None:
        """假設說不通就是說不通 —— 寧可記一次看不懂,也不要硬套。

        猜錯的捨牌會污染整條事件流,少發一個只是延後。
        """
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        watcher.observe(0, ["1m"])
        assert watcher.observe(0, ["9s", "9p", "9m"]) == []
        assert watcher.stats.confused == 1
        assert watcher.rivers[0] == ["1m", "2m"]

    def test_a_river_that_collapsed_is_counted_apart(self) -> None:
        """換局在清桌、立繪蓋住整條河 —— 看不懂是**對的反應**。

        這種次數會把 ``confused`` 稀釋掉,而那一欄是 tracker 唯一的診斷。
        實測一整場 8 次看不懂,7 次是這種。所以要看的是 confused - collapsed。
        """
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m", "4m", "5m", "6m"])
        assert watcher.observe(0, []) == []
        assert (watcher.stats.confused, watcher.stats.collapsed) == (1, 1)

    def test_a_real_misalignment_is_not_counted_as_a_collapse(self) -> None:
        """河還在、只是對不上 —— 那才是值得查的那一種。"""
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m", "3m"])
        assert watcher.observe(0, ["9s", "9p", "9m", "8m"]) == []
        assert (watcher.stats.confused, watcher.stats.collapsed) == (1, 0)

    def test_a_new_kyoku_clears_the_pending_shrink(self) -> None:
        watcher = RiverWatcher()
        watcher.observe(0, ["1m", "2m"])
        watcher.observe(0, ["1m"])
        watcher.reset()
        assert watcher._shrinking == [None] * 4  # noqa: SLF001


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
