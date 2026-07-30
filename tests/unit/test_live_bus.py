"""郵箱與 ViewModel 批次通知的測試。"""

from __future__ import annotations

import threading

from mia.engine.base import Advice
from mia.live.bus import Advices, CvHand, PacketHand, UpdateBus, WorkerStatus
from mia.mjai import Dahai
from mia.ui.viewmodel import ViewModel


class TestUpdateBus:
    def test_drain_returns_what_was_posted(self) -> None:
        bus = UpdateBus()
        bus.post(CvHand(("1m", "2m")))
        assert bus.drain() == (CvHand(("1m", "2m")),)

    def test_drain_empties_the_box(self) -> None:
        bus = UpdateBus()
        bus.post(CvHand(("1m",)))
        bus.drain()
        assert bus.drain() == ()

    def test_same_slot_keeps_only_the_newest(self) -> None:
        """這是整個設計的重點:UI 卡住的時候要掉幀,不要補播。"""
        bus = UpdateBus()
        for tile in ("1m", "2m", "3m"):
            bus.post(CvHand((tile,)))
        assert bus.drain() == (CvHand(("3m",)),)
        assert bus.stats.dropped == 2

    def test_different_slots_do_not_overwrite_each_other(self) -> None:
        """手牌與建議必須分開合併。

        合在一個 slot 的話,「摸牌→建議切 3s」之後別人打牌的事件會把那個建議
        連帶覆蓋掉,而它明明還該顯示著。
        """
        bus = UpdateBus()
        advice = Advices((Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False)),))
        bus.post(advice)
        bus.post(PacketHand(("1m",)))
        bus.post(PacketHand(("2m",)))

        drained = bus.drain()
        assert advice in drained
        assert PacketHand(("2m",)) in drained
        assert len(drained) == 2

    def test_drain_order_follows_first_post_of_each_slot(self) -> None:
        bus = UpdateBus()
        bus.post(PacketHand(("1m",)))
        bus.post(CvHand(("1m",)))
        bus.post(PacketHand(("2m",)))  # 覆蓋,但不該把 PacketHand 移到後面
        assert [type(u) for u in bus.drain()] == [PacketHand, CvHand]

    def test_post_from_many_threads_loses_nothing_but_stale_frames(self) -> None:
        """真正的用法是跨執行緒 —— 沒有鎖的話計數會漏。"""
        bus = UpdateBus()
        rounds = 200

        def spam(label: str) -> None:
            for i in range(rounds):
                bus.post(CvHand((f"{label}{i}",)))

        threads = [threading.Thread(target=spam, args=(c,)) for c in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        bus.drain()
        assert bus.stats.posted == 4 * rounds
        assert bus.stats.dropped + bus.stats.drained == bus.stats.posted

    def test_len_reports_pending_slots(self) -> None:
        bus = UpdateBus()
        assert len(bus) == 0
        bus.post(CvHand(("1m",)))
        bus.post(CvHand(("2m",)))
        assert len(bus) == 1


class TestWorkerStatus:
    def test_empty_message_means_nothing_to_say(self) -> None:
        status = WorkerStatus("畫面")
        assert status.read() == ""

    def test_say_replaces_rather_than_accumulates(self) -> None:
        status = WorkerStatus("畫面")
        status.say("校正中 3/10")
        status.say("校正中 7/10")
        assert status.read() == "校正中 7/10"


class TestViewModelBatch:
    def test_three_updates_notify_once(self) -> None:
        seen: list[int] = []
        model = ViewModel(lambda _state: seen.append(1))
        with model.batch():
            model.update_packet_hand(("1m", "2m"))
            model.update_advices([Advice("m", None)])
            model.set_notices(["嗨"])
        assert len(seen) == 1

    def test_the_single_notification_carries_the_final_state(self) -> None:
        states = []
        model = ViewModel(states.append)
        with model.batch():
            model.update_packet_hand(("1m",))
            model.set_notices(["最後這個"])
        assert states[-1].notices == ("最後這個",)
        assert states[-1].packet_hand == ("1m",)

    def test_an_empty_batch_notifies_nobody(self) -> None:
        seen: list[int] = []
        model = ViewModel(lambda _state: seen.append(1))
        with model.batch():
            pass
        assert seen == []

    def test_state_is_readable_inside_the_batch(self) -> None:
        """通知延後了,但狀態本身要立刻可讀 —— 不然批次內就不能做決定。"""
        model = ViewModel()
        with model.batch():
            model.update_packet_hand(("1m",))
            assert model.state.packet_hand == ("1m",)

    def test_notifications_resume_after_the_batch(self) -> None:
        seen: list[int] = []
        model = ViewModel(lambda _state: seen.append(1))
        with model.batch():
            model.set_notices(["a"])
        model.set_notices(["b"])
        assert len(seen) == 2

    def test_an_exception_inside_the_batch_still_flushes(self) -> None:
        """批次內出錯不該讓 UI 永遠停在舊畫面上。"""
        seen: list[int] = []
        model = ViewModel(lambda _state: seen.append(1))
        try:
            with model.batch():
                model.set_notices(["有更新到"])
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert len(seen) == 1
        assert model.state.notices == ("有更新到",)
