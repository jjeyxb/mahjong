"""UI 的狀態層。

**不 import Qt。** 真正容易錯的是「該顯示什麼」而不是「像素畫在哪」,而前者
不需要事件迴圈就測得動。
"""

from __future__ import annotations

from majsoul_copilot.analysis import AGARI, TENPAI
from majsoul_copilot.engine.base import Advice
from majsoul_copilot.mjai import Dahai, Reach
from majsoul_copilot.ui.viewmodel import ViewModel, ViewState

# 123m 456m 789m + 11p 對子 + 23p 兩面 = 聽 1p/4p。
#
# 刻意不用單騎聽牌:單騎「留 A 聽 B」與「留 B 聽 A」的進張枚數往往一樣多,
# 是真正的平手,排序只好退回按牌名排 —— 那種牌型測不出「有沒有挑對」。
TENPAI_MJAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"]
#: 同一手的雀魂記法 —— CV 那條路輸出的是這種
TENPAI_MS = list(TENPAI_MJAI)


def model() -> ViewModel:
    return ViewModel()


class TestNotation:
    def test_cv_input_is_converted_to_mjai(self) -> None:
        """CV 給雀魂記法(0m/1z),UI 以下一律 MJAI 記法。"""
        m = model()
        m.update_cv_hand(["0m", "1z", "7z"] + ["1p"] * 10)
        assert m.state.hand[:3] == ("5mr", "E", "C")

    def test_an_unreadable_tile_is_skipped_with_a_notice(self) -> None:
        """認不得的牌不該讓整個 UI 崩掉,但使用者要知道這一幀被跳過了。"""
        m = model()
        m.update_cv_hand(["10m"])
        assert m.state.hand == ()
        assert any("認不得" in n for n in m.state.notices)


class TestHandSources:
    def test_packets_win_over_cv(self) -> None:
        """封包是精確的,CV 是估計的。"""
        m = model()
        m.update_cv_hand(["1m"] * 13)
        m.update_packet_hand(TENPAI_MJAI)
        assert m.state.hand == tuple(TENPAI_MJAI)
        assert m.state.hand_source == "both"

    def test_cv_alone_is_used_when_there_are_no_packets(self) -> None:
        """只有畫面時 UI 仍然要能用 —— 那是功能 1 的整個賣點。"""
        m = model()
        m.update_cv_hand(TENPAI_MS)
        assert m.state.hand_source == "cv"
        assert m.state.analysis is not None

    def test_a_conflict_is_reported_not_resolved(self) -> None:
        """兩邊對不上時不去猜哪個對 —— 那件事本身就是使用者該知道的資訊。"""
        m = model()
        m.update_cv_hand(["1m"] * 13)
        m.update_packet_hand(TENPAI_MJAI)
        assert m.state.hand_conflict

    def test_one_source_alone_is_never_a_conflict(self) -> None:
        """只有一邊時沒有東西可比,那是資訊不足而不是衝突。"""
        m = model()
        m.update_cv_hand(TENPAI_MS)
        assert not m.state.hand_conflict

    def test_agreeing_sources_are_not_a_conflict(self) -> None:
        m = model()
        m.update_cv_hand(TENPAI_MS)
        m.update_packet_hand(TENPAI_MJAI)
        assert not m.state.hand_conflict


class TestDrawnTile:
    def test_the_drawn_tile_is_taken_from_the_source(self) -> None:
        """不從 hand 猜 —— 排序過之後那張不一定在尾巴。"""
        m = model()
        m.update_packet_hand([*TENPAI_MJAI, "9s"], drawn="9s")
        assert m.state.drawn == "9s"

    def test_cv_drawn_is_appended_to_the_hand(self) -> None:
        """CV 那邊摸的那張是分開傳的,但完整手牌要含它 —— 不然向聽會算成 13 張。"""
        m = model()
        m.update_cv_hand(TENPAI_MS, drawn="9s")
        assert len(m.state.hand) == 14
        assert m.state.drawn == "9s"
        assert m.state.analysis is not None
        assert m.state.analysis.needs_discard

    def test_drawn_follows_the_same_priority_as_the_hand(self) -> None:
        """封包的手牌配 CV 的摸牌是混血狀態 —— 那張牌可能根本不在手牌裡。"""
        m = model()
        m.update_cv_hand(TENPAI_MS, drawn="9s")
        m.update_packet_hand([*TENPAI_MJAI, "1z"], drawn="1z")
        assert m.state.drawn == "1z"


class TestAnalysis:
    def test_a_tenpai_hand_is_analysed(self) -> None:
        m = model()
        m.update_packet_hand(TENPAI_MJAI)
        assert m.state.analysis is not None
        assert m.state.analysis.shanten == TENPAI

    def test_a_complete_hand_reports_agari(self) -> None:
        m = model()
        m.update_packet_hand([*TENPAI_MJAI, "4p"], drawn="4p")
        assert m.state.analysis is not None
        assert m.state.analysis.shanten == AGARI

    def test_discards_are_offered_only_after_a_draw(self) -> None:
        """13 張還沒摸牌,沒有「該切什麼」可言。"""
        m = model()
        m.update_packet_hand(TENPAI_MJAI)
        assert m.state.discards == ()

        m.update_packet_hand([*TENPAI_MJAI, "9s"], drawn="9s")
        assert m.state.best_discard is not None
        assert m.state.best_discard.tile == "9s"

    def test_an_illegal_tile_count_is_not_an_error(self) -> None:
        """張數是 3 的倍數很常見(辨識抓在動畫中間)。回 None 讓 UI 維持上一次顯示。"""
        m = model()
        m.update_packet_hand(["1m"] * 12)
        assert m.state.analysis is None
        assert m.state.discards == ()

    def test_an_unknown_tile_blocks_analysis(self) -> None:
        m = model()
        m.update_packet_hand(["?"] * 13)
        assert m.state.analysis is None


class TestAdvices:
    def _advice(self, engine: str, tile: str | None) -> Advice:
        action = Dahai(actor=0, pai=tile, tsumogiri=False) if tile else None
        return Advice(engine, action, None, 12.0)

    def test_engine_order_is_preserved(self) -> None:
        """UI 並排顯示,順序跳動會讓人看錯是哪個引擎給的。"""
        m = model()
        m.update_advices([self._advice("a", "1m"), self._advice("b", "9p")])
        assert [e.name for e in m.state.engines] == ["a", "b"]

    def test_primary_is_the_first_engine_with_an_action(self) -> None:
        m = model()
        m.update_advices([self._advice("quiet", None), self._advice("loud", "1m")])
        assert m.state.primary is not None
        assert m.state.primary.name == "loud"

    def test_disagreement_is_flagged(self) -> None:
        m = model()
        m.update_advices([self._advice("a", "1m"), self._advice("b", "9p")])
        assert not m.state.is_unanimous

    def test_agreement_is_flagged(self) -> None:
        m = model()
        m.update_advices([self._advice("a", "1m"), self._advice("b", "1m")])
        assert m.state.is_unanimous

    def test_no_action_at_all_is_unanimous(self) -> None:
        """大部分事件不需要任何人動作,那不算分歧。"""
        m = model()
        m.update_advices([self._advice("a", None), self._advice("b", None)])
        assert m.state.is_unanimous

    def test_q_values_become_labelled_candidates(self) -> None:
        """9m=8、E=27,最高分在 E。"""
        meta = {"mask_bits": (1 << 8) | (1 << 27), "q_values": [0.1, 0.9]}
        m = model()
        m.update_advices([Advice("mortal", Dahai(actor=0, pai="E", tsumogiri=False), meta, 14.0)])
        primary = m.state.primary
        assert primary is not None
        assert primary.has_reasoning
        assert primary.candidates[0].label == "E"

    def test_a_rule_based_engine_has_no_reasoning(self) -> None:
        m = model()
        m.update_advices([self._advice("baseline", "1m")])
        primary = m.state.primary
        assert primary is not None
        assert not primary.has_reasoning

    def test_a_non_discard_action_has_no_tile(self) -> None:
        """立直沒有對應的單張牌 —— UI 靠 tile 是不是 None 決定要不要畫牌面圖。"""
        m = model()
        m.update_advices([Advice("mortal", Reach(actor=0), None, 10.0)])
        primary = m.state.primary
        assert primary is not None
        assert primary.tile is None
        assert primary.action == "立直"


class TestNotifications:
    def test_listeners_are_called_on_every_change(self) -> None:
        seen: list[ViewState] = []
        m = ViewModel(seen.append)
        m.update_packet_hand(TENPAI_MJAI)
        m.update_advices([])
        assert len(seen) == 2

    def test_multiple_listeners_all_fire(self) -> None:
        """側邊視窗與 Overlay 各自訂閱同一個 ViewModel。"""
        a: list[ViewState] = []
        b: list[ViewState] = []
        m = ViewModel(a.append)
        m.subscribe(b.append)
        m.update_packet_hand(TENPAI_MJAI)
        assert len(a) == len(b) == 1

    def test_state_objects_are_immutable_snapshots(self) -> None:
        """UI 拿到的東西不會在畫的中途被換掉。"""
        seen: list[ViewState] = []
        m = ViewModel(seen.append)
        m.update_packet_hand(TENPAI_MJAI)
        first = seen[-1]
        m.update_packet_hand(["1m"] * 13)
        assert first.hand == tuple(TENPAI_MJAI), "舊快照被後續更新改到了"

    def test_clear_resets_everything(self) -> None:
        m = model()
        m.update_packet_hand(TENPAI_MJAI)
        m.update_advices([Advice("a", Dahai(actor=0, pai="1m", tsumogiri=False))])
        m.clear()
        assert m.state == ViewState()

    def test_notices_replace_wholesale(self) -> None:
        m = model()
        m.set_notices(["甲"])
        m.set_notices(["乙"])
        assert m.state.notices == ("乙",)

    def test_a_repeated_notice_is_not_duplicated(self) -> None:
        m = model()
        m.update_cv_hand(["10m"])
        m.update_cv_hand(["10m"])
        assert len(m.state.notices) == 1


class TestPrimaryEngine:
    """哪個引擎當主角。

    這一段是為了釘住一個實際看到的問題:規則式 baseline 排在模型前面時,
    headline 顯示的是 baseline、真正的模型被擠到下面那排,而 Q 值長條整段
    消失(baseline 沒有 meta)。畫面看起來「有東西」,所以很容易當成正常。
    """

    def _advice(self, engine: str, tile: str, meta: dict | None = None) -> Advice:
        return Advice(engine, Dahai(actor=0, pai=tile, tsumogiri=False), meta, 1.0)

    def test_order_decides_who_is_primary(self) -> None:
        m = model()
        m.update_advices([self._advice("baseline", "1m"), self._advice("mortal", "9p")])
        assert m.state.primary is not None
        assert m.state.primary.name == "baseline", "順序就是優先序"

    def test_a_preferred_engine_overrides_the_order(self) -> None:
        m = model()
        m.set_preferred_engine("mortal")
        m.update_advices([self._advice("baseline", "1m"), self._advice("mortal", "9p")])
        assert m.state.primary is not None
        assert m.state.primary.name == "mortal"

    def test_a_preferred_engine_stays_primary_even_with_no_action(self) -> None:
        """使用者點名要看某個引擎,就該一直看它 —— 不該因為它剛好沒事做就跳走。"""
        m = model()
        m.set_preferred_engine("mortal")
        m.update_advices([self._advice("baseline", "1m"), Advice("mortal", None)])
        assert m.state.primary is not None
        assert m.state.primary.name == "mortal"
        assert m.state.primary.action is None

    def test_an_unknown_preference_falls_back_to_the_order(self) -> None:
        """指定的引擎沒起來(啟動失敗)時,不該整頁空白。"""
        m = model()
        m.set_preferred_engine("不存在的引擎")
        m.update_advices([self._advice("baseline", "1m")])
        assert m.state.primary is not None
        assert m.state.primary.name == "baseline"

    def test_clearing_the_preference_returns_to_the_order(self) -> None:
        m = model()
        m.set_preferred_engine("mortal")
        m.set_preferred_engine(None)
        m.update_advices([self._advice("baseline", "1m"), self._advice("mortal", "9p")])
        assert m.state.primary is not None
        assert m.state.primary.name == "baseline"
