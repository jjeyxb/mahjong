"""摸切機器人 —— 替人坐在那裡打完整局,只為了錄素材。

測的是**什麼時候該打**,不測點擊本身。點擊是 Playwright 的工作,而「有沒有
輪到自己」是這支程式唯一的判斷 —— 它判斷錯的後果是在別人的回合亂點畫面。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from mia.mjai.tiles import sort_key

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def autoplay():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_autoplay", ROOT / "tools" / "autoplay.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_autoplay"] = module
    spec.loader.exec_module(module)
    return module


def _advice(action=None, follow_up=None, *, declined: bool = False):
    """真的 :class:`~mia.engine.base.Advice`,不是自己捏的假貨。

    捏假貨害過一次:立直那條路的 bug 之所以沒被測出來,就是因為假的 Advice
    直接帶著 follow_up —— 而真實情況是 react() 根本不會填它。而且真的
    Advice 是 frozen dataclass,``dataclasses.replace`` 吃得下,假貨不行。

    ``declined``(被問了但選擇跳過)不是欄位而是由 meta 算的,所以這裡要
    給一個「合法動作數 > 1」的遮罩,見 engine/actions.py 的 is_decision。
    """
    from mia.engine.base import Advice

    meta = {"mask_bits": 0b11} if declined else None
    return Advice(engine="test", action=action, meta=meta, follow_up=follow_up)


class FakePage:
    """照真實情況做的假分頁。

    **``viewport_size`` 是 ``None``** —— 瀏覽器是用 ``no_viewport=True``
    開的,那個選項的意思就是「不要設 viewport」。先前這個假貨回了一個
    字典,於是測試一路綠燈,而實際跑起來 ``act`` 每次都卡在
    ``if not size: return`` 然後安靜地返回 —— 整個自動打牌一動也不動。

    假貨比真貨好用,測試就測不到東西。
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.viewport_size = None
        self._size = [width, height]
        self.clicks: list[tuple[int, int]] = []
        self.mouse = self
        self._at: tuple[int, int] | None = None

    def evaluate(self, script: str):  # noqa: ARG002
        return list(self._size)

    def wait_for_timeout(self, ms: float) -> None:
        pass

    # --- mouse ---
    def move(self, x: int, y: int) -> None:
        self._at = (x, y)

    def down(self) -> None:
        pass

    def up(self) -> None:
        """放開才算點完一次 —— 按住不放不是一次點擊。"""
        if self._at is not None:
            self.clicks.append(self._at)

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))


class TestWhenItActs:
    def test_it_does_nothing_before_any_packet(self, autoplay) -> None:
        """還沒看到任何封包就不可以點。

        瀏覽器剛開、還在登入畫面的時候亂點,後果是按到不該按的東西。
        """
        bot = autoplay.TsumogiriBot()
        page = FakePage()
        bot.act(page)
        assert page.clicks == []

    def test_it_waits_for_its_own_turn(self, autoplay) -> None:
        """沒摸到牌就不是自己的回合。

        ``HandTracker.drawn`` 是 None 代表「這一巡還沒摸,或已經打過了」。
        在那種狀態下點畫面,點到的是別人回合裡的某個東西。
        """
        bot = autoplay.TsumogiriBot(delay=0.0, allowed=True)
        bot._hand.drawn = None  # noqa: SLF001 - 直接擺出要測的狀態
        bot._ready_at = None  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        assert page.clicks == []

    def test_it_clicks_once_it_is_ready(self, autoplay) -> None:
        bot = autoplay.TsumogiriBot(delay=0.0, allowed=True)
        bot._ready_at = 0.0  # noqa: SLF001 - 已經可以打了
        page = FakePage()
        bot.act(page)
        assert len(page.clicks) == 1
        assert bot.discards == 1

    def test_it_does_not_double_click(self, autoplay) -> None:
        """**這是最重要的一條。**

        打出去之後 ``drawn`` 要等伺服器回的事件才會清掉,在那之前狀態仍然
        說「該打牌」。沒有冷卻的話會連點,而連點在雀魂上的後果是**把下一巡
        的牌也打掉** —— 那不是「少錄一點」,是把對局打壞。
        """
        bot = autoplay.TsumogiriBot(delay=0.0, allowed=True)
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        bot._ready_at = 0.0  # noqa: SLF001 - 封包還沒回來,狀態仍說該打
        bot.act(page)
        assert len(page.clicks) == 1, "冷卻時間內不可以再點"

    def test_the_click_lands_on_the_drawn_tile(self, autoplay) -> None:
        """座標要落在手牌那一排的右端 —— 摸進來那張的位置。"""
        bot = autoplay.TsumogiriBot(delay=0.0, allowed=True)
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage(1280, 720)
        bot.act(page)
        x, y = page.clicks[0]
        assert 0.75 * 1280 < x < 0.85 * 1280, "右端,但不到畫面邊緣"
        assert 0.85 * 720 < y < 0.99 * 720, "最下面那一排"

    def test_a_page_that_cannot_report_its_size_is_survivable(self, autoplay) -> None:
        """問不到大小就跳過這一巡,不可以拋例外 ——
        自動打牌壞掉不值得讓正在錄的那一場跟著死。"""
        bot = autoplay.TsumogiriBot(delay=0.0, allowed=True)
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage()
        page._size = []  # noqa: SLF001 - 問不到大小的樣子
        bot.act(page)
        assert page.clicks == []


class TestWatchingRealPackets:
    def test_a_real_game_makes_it_want_to_discard(self, autoplay) -> None:
        """餵真實封包進去,它應該有好幾次「該打牌了」。

        用真的牌譜而不是自己編的事件:輪到誰、摸了什麼,那是 HandTracker
        在管的,而它已經被真實牌譜驗過(見 decisions.md「137 個手牌狀態、
        0 次脫節」)。這條確認的是**接線**對了,不是重測它。
        """
        import json
        from base64 import b64decode

        bot = autoplay.TsumogiriBot(delay=0.0)
        wanted = 0
        for line in (FIXTURES / "real_game_full.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            bot.observe(
                b64decode(row["b64"]),
                from_client=row["dir"] == "c2s",
                flow=str(row.get("flow", "")),
            )
            wanted += bot._ready_at is not None  # noqa: SLF001
        assert wanted > 10, f"整場只想打 {wanted} 次,接線大概錯了"


class TestWhereTheTilesAre:
    """牌在畫面上的位置。幾何來自 HandLayout(量過,殘差 < 1 px)。"""

    def test_the_slots_go_left_to_right(self, autoplay) -> None:
        xs = [autoplay.tile_point(i, 13)[0] for i in range(13)]
        assert xs == sorted(xs)
        assert all(0.0 < x < 1.0 for x in xs)

    def test_the_drawn_tile_sits_right_of_them_all(self, autoplay) -> None:
        """摸的那張要在最右邊,而且與第 13 張之間有縫。

        沒有那道縫就分不出「切摸到的那張」與「切最後一張暗牌」——
        在雀魂上那是兩個不同的格子。
        """
        last = autoplay.tile_point(12, 13)[0]
        drawn = autoplay.tile_point(0, 13, drawn=True)[0]
        assert drawn > last
        assert drawn - last > 0.03, "縫太小,點下去會落在隔壁"

    def test_it_matches_the_hardcoded_tsumogiri_point(self, autoplay) -> None:
        """摸切那個寫死的座標必須跟算出來的一致。

        兩邊各算一次就會各自漂。這條是把它們釘在一起 —— 哪天 HandLayout
        改了,摸切那個常數會被這條抓出來。
        """
        computed = autoplay.tile_point(0, 13, drawn=True)
        assert computed[0] == pytest.approx(autoplay.DRAWN_TILE[0], abs=0.005)
        assert computed[1] == pytest.approx(autoplay.DRAWN_TILE[1], abs=0.005)

    def test_fewer_tiles_moves_the_drawn_one_left(self, autoplay) -> None:
        """副露之後手上剩的牌變少,摸的那張會往左跑。

        實測說明寫在 config/default.yaml:無副露時 x≈0.78,四組副露時
        手上只剩 1 張,它跑到 x≈0.19。寫死一個座標的話,一有副露就點空。
        """
        full = autoplay.tile_point(0, 13, drawn=True)[0]
        melded = autoplay.tile_point(0, 1, drawn=True)[0]
        assert melded < full - 0.4


class TestWhatMortalWouldClick:
    """把引擎的建議翻成「點哪裡」。這是 MortalBot 唯一會想錯的地方。"""

    @staticmethod
    def _bot(autoplay, tiles, drawn=None):
        bot = autoplay.MortalBot(engine=None)
        bot._hand.tiles = list(tiles)  # noqa: SLF001
        bot._hand.drawn = drawn  # noqa: SLF001
        return bot

    def test_nothing_to_do_means_no_click(self, autoplay) -> None:
        """沒被問就不要動。別人在摸打時亂點,點到的是別人回合裡的東西。"""
        assert self._bot(autoplay, [])._plan(_advice()) == []  # noqa: SLF001

    def test_declining_presses_skip(self, autoplay) -> None:
        """被問了而選擇跳過,要真的去按「跳過」。

        不按的話雀魂會等到超時 —— 那不只是慢,是讓每一巡都多幾秒,
        整場錄下來的節奏與真實對局不一樣。
        """
        plan = self._bot(autoplay, [])._plan(_advice(declined=True))  # noqa: SLF001
        assert plan == [autoplay._BUTTONS["skip"]]  # noqa: SLF001

    def test_a_discard_lands_on_that_tile(self, autoplay) -> None:
        """切某一張,就要點那一張所在的槽位 —— 不是隨便一張。"""
        from mia.mjai.events import Dahai

        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
        bot = self._bot(autoplay, tiles)
        points = bot._plan(_advice(Dahai(actor=0, pai="5m", tsumogiri=False)))  # noqa: SLF001
        # 期望值要用專案自己的理牌順序算。原本這裡寫 sorted(tiles) ——
        # 那把**錯的順序寫進斷言裡**,於是它驗證的是 bug 本身。
        assert points == [autoplay.tile_point(sorted(tiles, key=sort_key).index("5m"), 13)]

    def test_discarding_the_drawn_tile_uses_the_drawn_slot(self, autoplay) -> None:
        """摸切要點**摸的那個格子**,不是排序後的位置。

        摸進來那張還沒被排進手牌裡,雀魂把它單獨擺在右邊。照排序去點會
        點到別的牌 —— 而且那張牌是合法的,所以**不會有任何錯誤**,
        只是打錯牌。
        """
        from mia.mjai.events import Dahai

        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "9s"]
        bot = self._bot(autoplay, [*tiles, "9s"], drawn="9s")
        points = bot._plan(_advice(Dahai(actor=0, pai="9s", tsumogiri=True)))  # noqa: SLF001
        assert points == [autoplay.tile_point(0, 13, drawn=True)]

    def test_riichi_is_two_clicks(self, autoplay) -> None:
        """立直是「宣言」加「切牌」兩步。

        少按第二步,雀魂會停在等你選牌的狀態直到超時。而切哪一張引擎得
        **先 peek** —— 等看到 reach 事件才問就來不及了(見 Advice.follow_up)。
        """
        from mia.mjai.events import Dahai, Reach

        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
        bot = self._bot(autoplay, tiles)
        advice = _advice(Reach(actor=0), follow_up=Dahai(actor=0, pai="9m", tsumogiri=False))
        points = bot._plan(advice)  # noqa: SLF001
        assert len(points) == 2
        assert points[0] == autoplay._BUTTONS["reach"]  # noqa: SLF001
        assert points[1] == autoplay.tile_point(sorted(tiles, key=sort_key).index("9m"), 13)

    def test_a_tile_we_do_not_hold_is_refused(self, autoplay) -> None:
        """引擎要打一張手上沒有的牌 —— 不要亂點一個格子。

        那代表事件流與引擎對不上了。亂點會打出一張真的牌,讓狀態更歪。
        """
        from mia.mjai.events import Dahai

        bot = self._bot(autoplay, ["1m", "2m"])
        assert bot._plan(_advice(Dahai(actor=0, pai="5z", tsumogiri=False))) == []  # noqa: SLF001



class TestWhereFollowUpComesFrom:
    """立直之後要切哪一張,**得自己去問**。

    ``AIEngine.react()`` 不會填 ``Advice.follow_up`` —— 那是 ``EngineGroup``
    另外 ``peek`` 一次填的(見 mia/engine/multiplex.py)。MortalBot 直接用
    react(),所以要自己補那一步。

    這一條是被真實封包抓出來的,而上面 ``test_riichi_is_two_clicks`` 抓不到
    它:那條測試**自己捏了一個 follow_up 餵進去**,所以它驗的是「拿到之後
    會不會翻成兩步」,完全沒碰「follow_up 從哪來」。實測餵真實牌譜時立直
    出現 2 次,兩次的 follow_up 都是 None,步數全部是 1。
    """

    def test_it_peeks_when_the_engine_wants_to_declare(self, autoplay) -> None:
        from mia.mjai.events import Dahai, Reach

        class FakeEngine:
            def __init__(self) -> None:
                self.peeked: list = []

            def peek(self, event):
                self.peeked.append(event)
                return _advice(Dahai(actor=2, pai="5p", tsumogiri=False))

        engine = FakeEngine()
        bot = autoplay.MortalBot(engine=engine)
        resolved = bot._resolve(_advice(Reach(actor=2)))  # noqa: SLF001
        assert len(engine.peeked) == 1, "看到立直就要問一次"
        assert resolved.follow_up is not None
        assert resolved.follow_up.pai == "5p"

    def test_it_does_not_peek_for_anything_else(self, autoplay) -> None:
        """切牌、鳴牌都不必問。多問一次就是多一次 5 秒逾時的風險。"""
        from mia.mjai.events import Dahai

        class FakeEngine:
            def peek(self, event):  # noqa: ARG002 - 簽章要對得上,但這裡不看它
                raise AssertionError("不該問")

        bot = autoplay.MortalBot(engine=FakeEngine())
        advice = _advice(Dahai(actor=2, pai="1m", tsumogiri=False))
        assert bot._resolve(advice) is advice  # noqa: SLF001

    def test_a_failed_peek_still_declares(self, autoplay) -> None:
        """問不到就只按立直鈕,不要整個放棄。

        少一次切牌雀魂會超時替你打,那比「立直也沒宣告」好 —— 後者等於
        這一局完全沒有立直,而立直牌正是我們要錄的東西。
        """
        from mia.engine.base import EngineError
        from mia.mjai.events import Reach

        class FakeEngine:
            def peek(self, event):  # noqa: ARG002 - 簽章要對得上,但這裡不看它
                raise EngineError("子程序掛了")

        bot = autoplay.MortalBot(engine=FakeEngine())
        resolved = bot._resolve(_advice(Reach(actor=2)))  # noqa: SLF001
        assert resolved.follow_up is None
        assert bot._plan(resolved) == [autoplay._BUTTONS["reach"]]  # noqa: SLF001


class TestTheHandIsOrderedLikeTheScreen:
    """手牌的順序要跟畫面一致,否則點到的是別張牌。

    這條是被「有時候會按歪」逼出來的。原本用 ``sorted()`` 直接排字串,
    而字串序是**先比點數再比花色** —— 排出 1m 1p 1s 2m 2p 2s…,
    雀魂卻是按花色分組的。字牌更離譜:字母序是 C E F N P S W,
    而畫面上是東南西北白發中。

    mia/mjai/tiles.py 的 ``sort_key`` 一直都在,說明的第一句就是在講
    這個錯。我沒用它。
    """

    @staticmethod
    def _bot(autoplay, tiles):
        bot = autoplay.MortalBot(engine=None)
        bot._hand.tiles = list(tiles)  # noqa: SLF001
        bot._hand.drawn = None  # noqa: SLF001
        return bot

    def test_suits_are_grouped_not_interleaved(self, autoplay) -> None:
        """混花色的手牌:9m 要排在 1p **之前**。

        字串序會把 1p 排到 9m 前面,於是這之後每一張的位置都錯開。
        """
        from mia.mjai.events import Dahai

        tiles = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"]
        bot = self._bot(autoplay, tiles)
        here = bot._plan(_advice(Dahai(actor=0, pai="9m", tsumogiri=False)))  # noqa: SLF001
        assert here == [autoplay.tile_point(1, 13)], "9m 是第 2 張,不是第 6 張"

    def test_honours_follow_the_wind_order(self, autoplay) -> None:
        """字牌是東南西北白發中,不是字母序。

        字母序把中(C)排到最前面,於是七張字牌全部錯位。
        """
        from mia.mjai.events import Dahai

        tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "E", "S", "W", "N", "P", "F", "C"]
        bot = self._bot(autoplay, tiles)
        here = bot._plan(_advice(Dahai(actor=0, pai="C", tsumogiri=False)))  # noqa: SLF001
        assert here == [autoplay.tile_point(12, 13)], "中是最後一張"

    def test_it_matches_the_projects_own_sort(self, autoplay) -> None:
        """直接跟 sort_key 對 —— 哪天理牌規則改了,這裡要跟著改。"""
        from mia.mjai.events import Dahai
        from mia.mjai.tiles import sort_key

        tiles = ["5s", "1p", "E", "3m", "C", "9p", "2s"]
        bot = self._bot(autoplay, tiles)
        for want in tiles:
            here = bot._plan(_advice(Dahai(actor=0, pai=want, tsumogiri=False)))  # noqa: SLF001
            index = sorted(tiles, key=sort_key).index(want)
            assert here == [autoplay.tile_point(index, len(tiles))], want


class TestAClickThatDidNothing:
    """莊家開局還在發牌,點下去被吃掉。實測第一手要 13.9 秒、53.7 秒才打出去。"""

    @staticmethod
    def _clicked(autoplay):
        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._pending = [(0.5, 0.925)]  # noqa: SLF001 - 該打了
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        assert len(page.clicks) == 1
        return bot, page

    def test_it_clicks_again_when_nothing_happened(self, autoplay) -> None:
        bot, page = self._clicked(autoplay)
        bot._retry_at = 0.0  # noqa: SLF001 - 時間到了
        bot.act(page)
        assert page.clicks == [page.clicks[0]] * 2

    def test_it_waits_before_clicking_again(self, autoplay) -> None:
        bot, page = self._clicked(autoplay)
        bot.act(page)
        assert len(page.clicks) == 1

    def test_any_event_means_the_click_worked(self, autoplay) -> None:
        """打出去之後還重點的話,點到的是下一巡的牌。"""
        import json
        from base64 import b64decode

        class Engine:
            def react(self, event):  # noqa: ARG002
                return _advice()

        bot, page = self._clicked(autoplay)
        bot.engine = Engine()
        for line in (FIXTURES / "real_game_full.jsonl").read_text().splitlines():
            row = json.loads(line)
            bot.observe(
                b64decode(row["b64"]),
                from_client=row["dir"] == "c2s",
                flow=str(row.get("flow", "")),
            )
            if bot.events:
                break
        bot._retry_at = 0.0  # noqa: SLF001
        bot.act(page)
        assert len(page.clicks) == 1

    def test_a_reply_during_the_press_counts(self, autoplay) -> None:
        """按壓中 Playwright 會送事件進來。實測打牌的回應就在這時候到,結果局末連重點六次。"""

        class Page(FakePage):
            def __init__(self, bot) -> None:
                super().__init__()
                self.bot = bot

            def down(self) -> None:
                self.bot._unconfirmed = None  # noqa: SLF001 - observe 這時候清掉

        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._pending = [(0.5, 0.925)]  # noqa: SLF001
        bot._ready_at = 0.0  # noqa: SLF001
        page = Page(bot)
        bot.act(page)
        assert bot._unconfirmed is None  # noqa: SLF001

    def test_it_gives_up_eventually(self, autoplay) -> None:
        bot, page = self._clicked(autoplay)
        for _ in range(30):
            bot._retry_at = 0.0  # noqa: SLF001
            bot.act(page)
        assert len(page.clicks) == 9, "第一下加上最多八次重點"

    def test_the_second_step_of_a_riichi_is_not_a_retry(self, autoplay) -> None:
        """立直鈕之後還有切牌要點,那一步不是重點。"""
        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._pending = [(0.533, 0.76), (0.3, 0.925)]  # noqa: SLF001
        bot._ready_at = 0.0  # noqa: SLF001
        bot._snapshot = lambda *_: None  # noqa: SLF001 - 不存畫面
        page = FakePage()
        bot.act(page)
        assert bot._unconfirmed is None  # noqa: SLF001

    def test_a_dry_run_never_retries(self, autoplay) -> None:
        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True, dry_run=True)
        bot._pending = [(0.5, 0.925)]  # noqa: SLF001
        bot._ready_at = 0.0  # noqa: SLF001
        bot._snapshot = lambda *_: None  # noqa: SLF001
        bot.act(FakePage())
        assert bot._unconfirmed is None  # noqa: SLF001


class TestTheDealersFirstTurn:
    """畫面把 14 張一起排序,最後一張放在摸牌位。與封包說摸到哪張無關。

    實測南二局:封包摸 8 萬、手上有西,畫面摸牌位是西。引擎要打西,打出去的是 8 索。
    """

    HAND = ("1s", "2p", "3p", "3p", "3s", "3s", "5m", "6m", "6m", "7p", "7s", "8s", "W")

    def _bot(self, autoplay, *, dealing: bool = True):
        from mia.mjai.events import StartGame, StartKyoku, Tsumo

        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._follow(StartGame(id=1))  # noqa: SLF001
        bot._hand.feed(StartGame(id=1))  # noqa: SLF001
        start = StartKyoku(
            bakaze="S", kyoku=2, honba=0, kyotaku=0, oya=1 if dealing else 2,
            dora_marker="1p", tehais=[["?"] * 13, list(self.HAND), ["?"] * 13, ["?"] * 13],
            scores=[25000] * 4,
        )
        for event in (start, Tsumo(actor=1, pai="8m")):
            bot._follow(event)  # noqa: SLF001
            bot._hand.feed(event)  # noqa: SLF001
        return bot

    def test_the_last_sorted_tile_sits_in_the_draw_slot(self, autoplay) -> None:
        bot = self._bot(autoplay)
        assert bot._tile_point("W") == autoplay.tile_point(0, 13, drawn=True)  # noqa: SLF001

    def test_the_packets_draw_sits_with_the_rest(self, autoplay) -> None:
        bot = self._bot(autoplay)
        assert bot._tile_point("8m") == autoplay.tile_point(3, 13)  # noqa: SLF001

    def test_the_tile_the_old_code_hit_is_a_different_slot(self, autoplay) -> None:
        bot = self._bot(autoplay)
        assert bot._tile_point("8s") == autoplay.tile_point(12, 13)  # noqa: SLF001
        assert bot._tile_point("W") != bot._tile_point("8s")  # noqa: SLF001

    def test_a_non_dealer_follows_the_packet(self, autoplay) -> None:
        bot = self._bot(autoplay, dealing=False)
        assert bot._tile_point("8m") == autoplay.tile_point(0, 13, drawn=True)  # noqa: SLF001

    def test_after_the_first_discard_it_is_a_normal_hand(self, autoplay) -> None:
        from mia.mjai.events import Dahai, Tsumo

        bot = self._bot(autoplay)
        for event in (Dahai(actor=1, pai="W", tsumogiri=False), Tsumo(actor=1, pai="E")):
            bot._follow(event)  # noqa: SLF001
            bot._hand.feed(event)  # noqa: SLF001
        assert bot._tile_point("E") == autoplay.tile_point(0, 13, drawn=True)  # noqa: SLF001

    def test_a_kan_on_the_first_turn_ends_it_too(self, autoplay) -> None:
        from mia.mjai.events import Ankan

        bot = self._bot(autoplay)
        bot._follow(Ankan(actor=1, consumed=["3s"] * 4))  # noqa: SLF001
        assert not bot._dealing  # noqa: SLF001

    def test_other_seats_acting_do_not_end_it(self, autoplay) -> None:
        from mia.mjai.events import Dahai

        bot = self._bot(autoplay)
        bot._follow(Dahai(actor=2, pai="W", tsumogiri=False))  # noqa: SLF001
        assert bot._dealing  # noqa: SLF001


class TestChoosingWhichChi:
    """按「吃」之後有好幾種組合會跳選擇框。實測東三局要吃 1 2 索,吃到的是 2 4 索。"""

    @staticmethod
    def _bot(autoplay, operations):
        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._operations = operations  # noqa: SLF001
        return bot

    @staticmethod
    def _chi(consumed):
        from mia.mjai.events import Chi

        return _advice(Chi(actor=0, target=3, pai="3s", consumed=consumed))

    THREE = ((2, ("1s|2s", "2s|4s", "4s|5s")),)

    def test_the_three_options_of_the_real_case(self, autoplay) -> None:
        """位置是畫面量的:三組的中心在 0.3855、0.4895、0.5935。"""
        assert autoplay.choice_point(0, 3) == pytest.approx((0.3855, 0.71))
        assert autoplay.choice_point(1, 3) == pytest.approx((0.4895, 0.71))
        assert autoplay.choice_point(2, 3) == pytest.approx((0.5935, 0.71))

    def test_two_options_sit_either_side_of_the_middle(self, autoplay) -> None:
        assert autoplay.choice_point(0, 2) == pytest.approx((0.4375, 0.71))
        assert autoplay.choice_point(1, 2) == pytest.approx((0.5415, 0.71))

    def test_it_presses_chi_and_then_the_wanted_pair(self, autoplay) -> None:
        bot = self._bot(autoplay, self.THREE)
        points = bot._plan(self._chi(["1s", "2s"]))  # noqa: SLF001
        assert points == [autoplay.button_point("chi"), autoplay.choice_point(0, 3)]

    def test_the_pair_is_matched_whatever_the_order(self, autoplay) -> None:
        bot = self._bot(autoplay, self.THREE)
        points = bot._plan(self._chi(["5s", "4s"]))  # noqa: SLF001
        assert points[1] == autoplay.choice_point(2, 3)

    def test_one_option_has_no_box(self, autoplay) -> None:
        bot = self._bot(autoplay, ((2, ("1s|2s",)),))
        assert bot._plan(self._chi(["1s", "2s"])) == [autoplay.button_point("chi")]  # noqa: SLF001

    def test_a_red_five_is_written_as_zero_in_the_packet(self, autoplay) -> None:
        bot = self._bot(autoplay, ((2, ("0m|6m", "5m|6m")),))
        points = bot._plan(self._chi(["5mr", "6m"]))  # noqa: SLF001
        assert points[1] == autoplay.choice_point(0, 2)

    def test_a_pair_that_is_not_offered_still_presses_the_button(self, autoplay) -> None:
        bot = self._bot(autoplay, self.THREE)
        assert bot._plan(self._chi(["7s", "8s"])) == [autoplay.button_point("chi")]  # noqa: SLF001


class TestWhichButton:
    """好幾顆按鈕時由左到右是封包的順序,跳過在最右邊。實測三種組合。"""

    @staticmethod
    def _point(autoplay, operations, action):
        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._operations = operations  # noqa: SLF001
        return bot._plan(_advice(action))[0]  # noqa: SLF001

    def test_chi_or_pon(self, autoplay) -> None:
        from mia.mjai.events import Chi, Pon

        both = ((2, ("3s|4s",)), (3, ("5s|5s",)))
        chi = Chi(actor=0, target=3, pai="5s", consumed=["3s", "4s"])
        pon = Pon(actor=0, target=3, pai="5s", consumed=["5s", "5s"])
        assert self._point(autoplay, both, chi) == autoplay._slot_point(2)  # noqa: SLF001
        assert self._point(autoplay, both, pon) == autoplay._slot_point(1)  # noqa: SLF001

    def test_pon_or_ron(self, autoplay) -> None:
        from mia.mjai.events import Hora, Pon

        both = ((3, ("5s|5s",)), (9, ()))
        pon = Pon(actor=0, target=3, pai="5s", consumed=["5s", "5s"])
        assert self._point(autoplay, both, pon) == autoplay._slot_point(2)  # noqa: SLF001
        ron = Hora(actor=0, target=3)
        assert self._point(autoplay, both, ron) == autoplay._slot_point(1)  # noqa: SLF001

    def test_pon_or_kan(self, autoplay) -> None:
        from mia.mjai.events import Daiminkan

        both = ((3, ("5s|5s",)), (5, ("5s|5s|5s",)))
        kan = Daiminkan(actor=0, target=3, pai="5s", consumed=["5s", "5s", "5s"])
        assert self._point(autoplay, both, kan) == autoplay._slot_point(1)  # noqa: SLF001

    def test_without_a_list_it_falls_back_to_the_single_button(self, autoplay) -> None:
        from mia.mjai.events import Pon

        pon = Pon(actor=0, target=3, pai="5s", consumed=["5s", "5s"])
        assert self._point(autoplay, (), pon) == autoplay.button_point("pon")

    def test_the_list_comes_from_the_packet_without_the_discard(self, autoplay) -> None:
        from types import SimpleNamespace as Space

        action = Space(operation=Space(operation_list=[
            Space(type=1, combination=[]), Space(type=7, combination=["5s"]),
        ]))
        assert autoplay._read_operations(action) == ((7, ("5s",)),)  # noqa: SLF001
        assert autoplay._read_operations(Space(operation=None)) == ()  # noqa: SLF001


class TestItNoticesAWrongDiscard:
    """打錯牌不會有任何錯誤訊息:是合法的牌,對局照樣繼續。實測是使用者看出來的。"""

    @staticmethod
    def _bot(autoplay):
        from mia.mjai.events import StartGame

        bot = autoplay.MortalBot(engine=None, delay=0.0, allowed=True)
        bot._follow(StartGame(id=1))  # noqa: SLF001
        return bot

    def test_a_different_tile_is_counted(self, autoplay) -> None:
        from mia.mjai.events import Dahai

        bot = self._bot(autoplay)
        bot._intended = "W"  # noqa: SLF001
        bot._follow(Dahai(actor=1, pai="8s", tsumogiri=False))  # noqa: SLF001
        assert bot.misplays == 1

    def test_the_right_tile_is_not(self, autoplay) -> None:
        from mia.mjai.events import Dahai

        bot = self._bot(autoplay)
        bot._intended = "W"  # noqa: SLF001
        bot._follow(Dahai(actor=1, pai="W", tsumogiri=False))  # noqa: SLF001
        assert bot.misplays == 0

    def test_someone_elses_discard_is_ignored(self, autoplay) -> None:
        from mia.mjai.events import Dahai

        bot = self._bot(autoplay)
        bot._intended = "W"  # noqa: SLF001
        bot._follow(Dahai(actor=2, pai="8s", tsumogiri=False))  # noqa: SLF001
        assert bot.misplays == 0

    def test_the_wrong_chi_is_counted(self, autoplay) -> None:
        """實測要吃 1 2 索,吃到 2 4 索,打出去的牌都是合法的。"""
        from mia.mjai.events import Chi

        bot = self._bot(autoplay)
        bot._wanted_call = ["1s", "2s"]  # noqa: SLF001
        bot._follow(Chi(actor=1, target=0, pai="3s", consumed=["2s", "4s"]))  # noqa: SLF001
        assert bot.misplays == 1

    def test_the_right_chi_is_not(self, autoplay) -> None:
        from mia.mjai.events import Chi

        bot = self._bot(autoplay)
        bot._wanted_call = ["1s", "2s"]  # noqa: SLF001
        bot._follow(Chi(actor=1, target=0, pai="3s", consumed=["2s", "1s"]))  # noqa: SLF001
        assert bot.misplays == 0

    def test_a_riichi_checks_the_tile_after_the_declaration(self, autoplay) -> None:
        from mia.mjai.events import Dahai, Reach

        advice = _advice(Reach(actor=1), follow_up=Dahai(actor=1, pai="W", tsumogiri=False))
        assert autoplay._discard_of(advice) == "W"  # noqa: SLF001


class TestOnlyCpuTables:
    """「只在對電腦的房間用」不能只是一句提醒。"""

    @staticmethod
    def _auth(seats, robots):
        from types import SimpleNamespace

        from mia.groundtruth.liqi import MessageKind

        return SimpleNamespace(
            method=".lq.FastTest.authGame",
            kind=MessageKind.RESPONSE,
            payload=SimpleNamespace(
                seat_list=seats,
                robots=[SimpleNamespace(account_id=a) for a in robots],
            ),
        )

    def test_three_robots_and_me(self, autoplay) -> None:
        """實測的座位表:電腦的帳號是 1、2、3,不是 0。"""
        assert autoplay.cpu_table(self._auth([1, 2, 15457218, 3], [1, 2, 3])) is True

    def test_another_human_at_the_table(self, autoplay) -> None:
        assert autoplay.cpu_table(self._auth([1, 2, 15457218, 777], [1, 2])) is False

    def test_three_humans_at_the_table(self, autoplay) -> None:
        assert autoplay.cpu_table(self._auth([5, 6, 15457218, 777], [])) is False

    def test_other_messages_say_nothing(self, autoplay) -> None:
        from types import SimpleNamespace

        from mia.groundtruth.liqi import MessageKind

        other = SimpleNamespace(method=".lq.Lobby.login", kind=MessageKind.RESPONSE, payload=None)
        assert autoplay.cpu_table(other) is None

    def test_no_click_before_the_table_is_confirmed(self, autoplay) -> None:
        bot = autoplay.TsumogiriBot(delay=0.0)
        bot._ready_at = 0.0  # noqa: SLF001 - 該打了,但還沒確認這一桌
        page = FakePage()
        bot.act(page)
        assert page.clicks == []

    def test_mortal_bot_waits_too(self, autoplay) -> None:
        bot = autoplay.MortalBot(engine=object(), delay=0.0)
        bot._ready_at = 0.0  # noqa: SLF001
        bot._pending = [(0.5, 0.5)]  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        assert page.clicks == []


class TestOpeningARoom:
    """大廳點錯的後果比牌桌上大:每一步都要有憑據才往下走。"""

    class _Bot:
        def __init__(self) -> None:
            self.acted = 0

        def observe(self, payload, *, from_client, flow) -> None: ...

        def act(self, page) -> None:  # noqa: ARG002
            self.acted += 1

    class _Lobby(FakePage):
        """會回截圖的假分頁,``orange`` 決定截到的畫面是不是在對的頁面。"""

        def __init__(self, *, orange: bool) -> None:
            super().__init__()
            self.orange = orange

        def screenshot(self, path=None):
            import cv2
            import numpy as np

            image = np.zeros((720, 1280, 3), np.uint8)
            image[:] = (70, 175, 255) if self.orange else (104, 77, 51)
            data = cv2.imencode(".png", image)[1].tobytes()
            if path is not None:
                Path(path).write_bytes(data)
            return data

    @staticmethod
    def _message(method, payload=None):
        from types import SimpleNamespace

        from mia.groundtruth.liqi import MessageKind

        return SimpleNamespace(method=method, kind=MessageKind.RESPONSE, payload=payload)

    @classmethod
    def _room(cls, mode=2, players=4):
        from types import SimpleNamespace

        return cls._message(
            ".lq.Lobby.createRoom",
            SimpleNamespace(
                room=SimpleNamespace(mode=SimpleNamespace(mode=mode), max_player_count=players)
            ),
        )

    @classmethod
    def _update(cls, *, robots, humans=1):
        from types import SimpleNamespace

        return cls._message(
            ".lq.NotifyRoomPlayerUpdate",
            SimpleNamespace(player_list=[0] * humans, robots=[0] * robots),
        )

    def _pilot(self, autoplay, tmp_path):
        from mia.groundtruth.schema import LiqiSchema

        pilot = autoplay.CpuRoomPilot(bot=self._Bot(), out=tmp_path, schema=LiqiSchema.load())
        pilot.note(self._message(".lq.Lobby.loginSuccess"))
        return pilot

    @staticmethod
    def _go(pilot, page) -> None:
        """動畫已經演完,走一步。"""
        pilot._wait_until = 0.0  # noqa: SLF001 - 不真的等
        pilot.act(page)

    @classmethod
    def _overtime_room(cls, *, fandian, bust):
        from types import SimpleNamespace

        message = cls._room()
        message.payload.room.mode.detail_rule = SimpleNamespace(fandian=fandian, can_jifei=bust)
        return message

    def _overtime_pilot(self, autoplay, tmp_path):
        from mia.groundtruth.schema import LiqiSchema

        return autoplay.CpuRoomPilot(
            bot=self._Bot(), out=tmp_path, schema=LiqiSchema.load(), overtime=True
        )

    def test_overtime_goes_through_the_advanced_settings(self, autoplay, tmp_path) -> None:
        pilot = self._overtime_pilot(autoplay, tmp_path)
        names = [stage.name for stage in pilot._plan]  # noqa: SLF001
        assert names.index("高級設置") < names.index("關閉擊飛") < names.index("創建")
        scrolled = sum(stage.scroll for stage in pilot._plan)  # noqa: SLF001
        assert scrolled == 0, "要捲回去,創建前要看半莊的顏色"

    def test_a_plain_room_has_no_extra_stages(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        assert "高級設置" not in [stage.name for stage in pilot._plan]  # noqa: SLF001

    def test_overtime_stops_if_the_rule_did_not_take(self, autoplay, tmp_path) -> None:
        """按錯位置的話開出來是普通房間,打完才發現白打一場。"""
        pilot = self._overtime_pilot(autoplay, tmp_path)
        pilot.note(self._overtime_room(fandian=30000, bust=False))
        assert pilot.failed
        assert not pilot.room

    def test_overtime_needs_busting_off(self, autoplay, tmp_path) -> None:
        pilot = self._overtime_pilot(autoplay, tmp_path)
        pilot.note(self._overtime_room(fandian=200000, bust=True))
        assert pilot.failed

    def test_overtime_room_as_ordered(self, autoplay, tmp_path) -> None:
        pilot = self._overtime_pilot(autoplay, tmp_path)
        pilot.note(self._overtime_room(fandian=200000, bust=False))
        assert pilot.room

    def test_nothing_before_login(self, autoplay, tmp_path) -> None:
        from mia.groundtruth.schema import LiqiSchema

        pilot = autoplay.CpuRoomPilot(bot=self._Bot(), out=tmp_path, schema=LiqiSchema.load())
        page = self._Lobby(orange=True)
        pilot.act(page)
        assert page.clicks == []

    def test_the_whole_way_to_the_table(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        page = self._Lobby(orange=True)
        for _ in range(4):
            self._go(pilot, page)
        assert len(page.clicks) == 4
        self._go(pilot, page)
        assert len(page.clicks) == 4, "房間還沒開好不可以加電腦"
        pilot.note(self._room())
        self._go(pilot, page)
        assert len(page.clicks) == 5
        for count in (1, 2):
            self._go(pilot, page)
            assert len(page.clicks) == 4 + count, "上一台沒進來不可以加下一台"
            pilot.note(self._update(robots=count))
            self._go(pilot, page)
        pilot.note(self._update(robots=3))
        self._go(pilot, page)
        assert len(page.clicks) == 8, "三台都進來才按開始"
        assert not pilot.failed

    def test_the_wrong_page_stops_everything(self, autoplay, tmp_path) -> None:
        """第一下沒點到友人場的話,第二下會點到「段位場」。絕對不能再往下按。"""
        pilot = self._pilot(autoplay, tmp_path)
        page = self._Lobby(orange=False)
        for _ in range(5):
            self._go(pilot, page)
        assert len(page.clicks) == 2
        assert pilot.failed
        assert (tmp_path / "lobby-failed.png").exists()

    def test_a_room_of_the_wrong_kind_stops(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        pilot.note(self._room(mode=1))
        assert pilot.failed
        assert not pilot.room

    def test_a_human_in_the_room_stops(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        pilot.note(self._update(robots=1, humans=2))
        assert pilot.failed

    def test_no_reply_stops(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        page = self._Lobby(orange=True)
        for _ in range(4):
            self._go(pilot, page)
        pilot._deadline = 0.0  # noqa: SLF001 - 時間到了
        self._go(pilot, page)
        assert pilot.failed

    def _at_the_first_robot(self, autoplay, tmp_path):
        pilot = self._pilot(autoplay, tmp_path)
        page = self._Lobby(orange=True)
        for _ in range(4):
            self._go(pilot, page)
        pilot.note(self._room())
        self._go(pilot, page)
        assert pilot._plan[pilot._index - 1].name == "新增電腦"  # noqa: SLF001
        return pilot, page

    def test_adding_a_robot_is_pressed_again(self, autoplay, tmp_path) -> None:
        """實測第三台電腦的點擊沒有送出請求,房間就開不成。"""
        pilot, page = self._at_the_first_robot(autoplay, tmp_path)
        clicks = len(page.clicks)
        pilot._retry_at = 0.0  # noqa: SLF001
        self._go(pilot, page)
        assert len(page.clicks) == clicks + 1
        assert page.clicks[-1] == page.clicks[-2]
        assert not pilot.failed

    def test_not_before_its_time(self, autoplay, tmp_path) -> None:
        pilot, page = self._at_the_first_robot(autoplay, tmp_path)
        clicks = len(page.clicks)
        self._go(pilot, page)
        assert len(page.clicks) == clicks

    def test_other_steps_are_never_pressed_again(self, autoplay, tmp_path) -> None:
        """按錯會到別的頁面。"""
        pilot = self._pilot(autoplay, tmp_path)
        page = self._Lobby(orange=True)
        for _ in range(4):
            self._go(pilot, page)
        clicks = len(page.clicks)
        pilot._retry_at = 0.0  # noqa: SLF001
        self._go(pilot, page)
        assert len(page.clicks) == clicks

    def test_still_nothing_still_stops(self, autoplay, tmp_path) -> None:
        pilot, page = self._at_the_first_robot(autoplay, tmp_path)
        pilot._deadline = 0.0  # noqa: SLF001
        self._go(pilot, page)
        assert pilot.failed

    def test_the_table_is_handed_to_the_bot(self, autoplay, tmp_path) -> None:
        pilot = self._pilot(autoplay, tmp_path)
        pilot.note(self._message(".lq.FastTest.authGame", None))
        page = self._Lobby(orange=True)
        pilot.act(page)
        assert pilot.bot.acted == 1
        assert page.clicks == []


def test_orange_is_told_apart_from_the_lobby(autoplay) -> None:
    """數字都是截圖量的:選中、「創建」、沒選中、以及大廳同一個位置的顏色。"""
    import numpy as np

    def patch(color):
        return np.full((720, 1280, 3), color, np.uint8)

    assert autoplay.is_orange(patch((70, 175, 255)), 0.5, 0.5)
    assert autoplay.is_orange(patch((58, 154, 239)), 0.5, 0.5)
    assert not autoplay.is_orange(patch((104, 77, 51)), 0.5, 0.5)
    assert not autoplay.is_orange(patch((54, 31, 142)), 0.5, 0.5)
    assert not autoplay.is_orange(patch((203, 204, 221)), 0.5, 0.5)
    assert not autoplay.is_orange(patch((170, 170, 170)), 0.5, 0.5)
