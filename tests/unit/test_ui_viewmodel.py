"""UI 的狀態層。

**不 import Qt。** 真正容易錯的是「該顯示什麼」而不是「像素畫在哪」,而前者
不需要事件迴圈就測得動。
"""

from __future__ import annotations

from mia.analysis import AGARI, TENPAI
from mia.engine.base import Advice
from mia.mjai import Chi, Dahai, Reach
from mia.ui.viewmodel import ViewModel, ViewState, q_fraction, shanten_text

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


class TestStaleAdvice:
    """建議的牌已經不在手上。

    手牌與建議是兩個獨立的更新槽,即時模式下最多差一個事件 —— 打牌的事件到了、
    手牌變成 13 張,但建議還是上一巡的。標出來比藏起來好:使用者要能分辨
    「剛剛打掉了」與「程式算錯了」。
    """

    def _advice(self, tile: str) -> Advice:
        return Advice("mortal", Dahai(actor=0, pai=tile, tsumogiri=False), None, 12.0)

    def test_fresh_advice_is_not_stale(self) -> None:
        model = ViewModel()
        model.update_packet_hand(("1m", "2m", "3s"))
        model.update_advices([self._advice("3s")])
        assert not model.state.advice_is_stale

    def test_advice_for_a_tile_no_longer_held_is_stale(self) -> None:
        model = ViewModel()
        model.update_advices([self._advice("3s")])
        model.update_packet_hand(("1m", "2m"))  # 3s 打掉了
        assert model.state.advice_is_stale

    def test_no_hand_means_nothing_to_compare(self) -> None:
        model = ViewModel()
        model.update_advices([self._advice("3s")])
        assert not model.state.advice_is_stale

    def test_no_advice_means_nothing_to_compare(self) -> None:
        model = ViewModel()
        model.update_packet_hand(("1m", "2m"))
        assert not model.state.advice_is_stale

    def test_a_non_discard_advice_is_never_stale(self) -> None:
        """立直、吃碰沒有對應的單張牌可比。"""
        model = ViewModel()
        model.update_packet_hand(("1m", "2m"))
        model.update_advices([Advice("mortal", Reach(actor=0), None, 5.0)])
        assert not model.state.advice_is_stale

    def test_a_red_five_is_not_confused_with_the_plain_one(self) -> None:
        """切赤五與切普通五是兩件事,不能互相當成「還在手上」。"""
        model = ViewModel()
        model.update_advices([self._advice("5mr")])
        model.update_packet_hand(("5m", "1p"))
        assert model.state.advice_is_stale


#: 實測的那一手:碰 -6.30、槓 -6.55、跳過 -0.15,引擎選跳過。
#: 就是使用者截圖裡那個「碰 / 槓 / 跳過」的局面。
_SKIP_META = {
    "mask_bits": (1 << 41) | (1 << 42) | (1 << 45),
    "q_values": [-6.30, -6.55, -0.15],
}


class TestSkipDecisions:
    """遊戲跳出「碰 / 槓 / 跳過」而引擎說不鳴 —— 那是一個答案,不是沒事發生。

    釘住實測到的 bug:那一手畫面上還掛著上一巡已經打掉的切牌建議,
    而使用者要的答案恰好就是被丟掉的那一個。
    """

    SKIP_META = _SKIP_META

    def test_a_declined_call_is_marked_declined(self) -> None:
        model = ViewModel()
        model.update_advices([Advice("mortal", None, self.SKIP_META, 10.0)])
        assert model.state.engines[0].declined

    def test_the_headline_says_skip_not_no_action_needed(self) -> None:
        model = ViewModel()
        model.update_advices([Advice("mortal", None, self.SKIP_META, 10.0)])
        assert model.state.engines[0].headline == "跳過"

    def test_nothing_being_asked_still_says_no_action_needed(self) -> None:
        """一場東風戰裡有 469 個這種事件 —— 它們不該說「跳過」。"""
        model = ViewModel()
        model.update_advices([Advice("mortal", None, {"mask_bits": 0}, 1.0)])
        view = model.state.engines[0]
        assert not view.declined
        assert view.headline == "不需要動作"

    def test_a_rule_based_engine_is_never_declining(self) -> None:
        """baseline 沒有 meta,而且本來就不鳴牌。"""
        model = ViewModel()
        model.update_advices([Advice("baseline", None, None, 0.4)])
        assert not model.state.engines[0].declined

    def test_the_candidates_are_kept_so_the_user_sees_the_margin(self) -> None:
        """最需要看到的正是「碰 -6.30 對 跳過 -0.15」這個對比。"""
        model = ViewModel()
        model.update_advices([Advice("mortal", None, self.SKIP_META, 10.0)])
        labels = [c.display for c in model.state.engines[0].candidates]
        assert labels == ["跳過", "碰", "槓"]

    def test_a_declining_engine_becomes_primary(self) -> None:
        """「跳過」也算有答案 —— 主角不該被讓給一個什麼都沒說的引擎。"""
        model = ViewModel()
        model.update_advices(
            [
                Advice("mortal", None, self.SKIP_META, 10.0),
                Advice("baseline", None, None, 0.4),
            ]
        )
        primary = model.state.primary
        assert primary is not None
        assert primary.name == "mortal"
        assert primary.headline == "跳過"

    def test_a_skip_is_not_marked_stale(self) -> None:
        """``advice_is_stale`` 是比對「建議的牌還在不在手上」。跳過沒有牌可比。"""
        model = ViewModel()
        model.update_packet_hand(("1m", "2m", "3m"))
        model.update_advices([Advice("mortal", None, self.SKIP_META, 10.0)])
        assert not model.state.advice_is_stale


class TestCallAndReachDisplay:
    """兩個實機打一場才發現的顯示問題。"""

    def test_a_chi_shows_which_two_tiles_to_use(self) -> None:
        """``吃 3m`` 可以用 1m2m、2m4m 或 4m5m —— 不寫出來使用者不知道點哪兩張。"""
        model = ViewModel()
        model.update_advices(
            [Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 9.0)]
        )
        view = model.state.engines[0]
        assert view.tile == "3m"
        assert view.consumed == ("1m", "2m")
        assert view.shown_tiles == ("3m", "1m", "2m")

    def test_a_chi_is_not_a_discard(self) -> None:
        """被鳴的那張是別家打出來的,不在自己手上。"""
        model = ViewModel()
        model.update_advices(
            [Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 9.0)]
        )
        assert not model.state.engines[0].is_discard

    def test_a_chi_is_never_marked_stale(self) -> None:
        """釘住一個會很吵的迴歸:拿被鳴那張去比手牌,每個吃碰建議都會被標
        「已打出」—— 因為它本來就不在手上。
        """
        model = ViewModel()
        model.update_packet_hand(("1m", "2m", "5p"))
        model.update_advices(
            [Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 9.0)]
        )
        assert not model.state.advice_is_stale

    def test_reach_shows_the_tile_to_discard_afterwards(self) -> None:
        """按下立直的下一秒就要選一張打出去。引擎不會主動說,要靠 peek 問。"""
        model = ViewModel()
        model.update_advices(
            [
                Advice(
                    "m",
                    Reach(actor=0),
                    None,
                    14.0,
                    follow_up=Dahai(actor=0, pai="5p", tsumogiri=False),
                )
            ]
        )
        view = model.state.engines[0]
        assert view.follow_up_tile == "5p"
        assert view.shown_tiles == ("5p",)

    def test_reach_without_a_follow_up_still_works(self) -> None:
        """peek 問失敗時仍然要顯示「立直」—— 主要建議已經拿到手了。"""
        model = ViewModel()
        model.update_advices([Advice("m", Reach(actor=0), None, 14.0)])
        view = model.state.engines[0]
        assert view.action == "立直"
        assert view.follow_up_tile is None
        assert view.shown_tiles == ()

    def test_two_engines_choosing_different_chi_are_not_unanimous(self) -> None:
        """分歧判斷靠 action 字串。只寫「吃 3m」的話這兩個會被當成一致。"""
        model = ViewModel()
        model.update_advices(
            [
                Advice("a", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 1.0),
                Advice("b", Chi(actor=0, target=3, pai="3m", consumed=["4m", "5m"]), None, 1.0),
            ]
        )
        assert not model.state.is_unanimous


class TestHeadlineParts:
    """「大字 + 一排牌」那一行的組成。

    側邊視窗與 Overlay 都吃這幾個屬性 —— 這裡驗的就是那個單一來源。兩邊各自
    去拆 ``action`` 字串的話,遲早有一邊會漏掉「吃要寫出 consumed」或
    「立直要畫出後續切牌」,而漏掉的那一邊看起來仍然正常。
    """

    @staticmethod
    def view(advice: Advice):
        model = ViewModel()
        model.update_advices([advice])
        return model.state.engines[0]

    def test_a_discard_says_just_the_verb(self) -> None:
        """牌交給圖去講,所以大字裡不再寫一次牌名。"""
        view = self.view(Advice("m", Dahai(actor=0, pai="3s", tsumogiri=False), None, 1.0))
        assert view.verb == "切"
        assert view.subject == "3s"
        assert view.own_tiles == ()

    def test_a_chi_puts_the_called_tile_and_the_two_of_ours_on_opposite_sides(self) -> None:
        """中間那個「用」是唯一讓「桌上那張」與「自己手上那兩張」分得開的東西。"""
        view = self.view(
            Advice("m", Chi(actor=0, target=3, pai="3m", consumed=["1m", "2m"]), None, 1.0)
        )
        assert view.verb == "吃"
        assert view.subject == "3m"
        assert view.joiner == "用"
        assert view.own_tiles == ("1m", "2m")

    def test_reach_shows_the_follow_up_as_the_tile_to_play(self) -> None:
        """立直本身不指向任何一張牌,該畫的是接下來要切的那張。"""
        view = self.view(
            Advice(
                "m",
                Reach(actor=0),
                None,
                1.0,
                follow_up=Dahai(actor=0, pai="5p", tsumogiri=False),
            )
        )
        assert view.verb == "立直"
        assert view.subject is None
        assert view.joiner == "切"
        assert view.own_tiles == ("5p",)

    def test_a_skip_has_a_verb_but_no_tiles(self) -> None:
        """「跳過」是一個答案,不是「沒有建議」。"""
        view = self.view(Advice("m", None, {"mask_bits": 0b11, "q_values": [0.1, -6.3]}, 1.0))
        assert view.verb == "跳過"
        assert view.subject is None
        assert view.own_tiles == ()

    def test_nobody_asked_is_quiet(self) -> None:
        view = self.view(Advice("m", None, None, 1.0))
        assert view.verb == "不需要動作"
        assert view.own_tiles == ()


class TestShantenText:
    """0 是聽牌、-1 是和了 —— 寫錯了畫面上看起來仍然正常,只是數字差一。"""

    def test_agari(self) -> None:
        assert shanten_text(AGARI) == "和了"

    def test_tenpai(self) -> None:
        assert shanten_text(TENPAI) == "聽牌"

    def test_a_plain_number(self) -> None:
        assert shanten_text(2) == "2 向聽"


class TestQFraction:
    """Q 值長條每一手重新正規化 —— 絕對值不可比,實測見過 +2.7 也見過 -6.8。"""

    def test_the_best_fills_the_bar(self) -> None:
        assert q_fraction(1.0, low=-1.0, high=1.0) == 1.0

    def test_the_worst_is_empty(self) -> None:
        assert q_fraction(-1.0, low=-1.0, high=1.0) == 0.0

    def test_all_negative_still_spans_the_full_range(self) -> None:
        """全部都是負分很常見(場況不好),那時長條不該整排看起來像出錯。"""
        assert q_fraction(-6.8, low=-6.8, high=-0.2) == 0.0
        assert q_fraction(-0.2, low=-6.8, high=-0.2) == 1.0

    def test_a_single_candidate_does_not_divide_by_zero(self) -> None:
        assert q_fraction(0.5, low=0.5, high=0.5) == 1.0
