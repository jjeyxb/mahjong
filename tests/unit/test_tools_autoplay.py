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
        bot = autoplay.TsumogiriBot(delay=0.0)
        bot._hand.drawn = None  # noqa: SLF001 - 直接擺出要測的狀態
        bot._ready_at = None  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        assert page.clicks == []

    def test_it_clicks_once_it_is_ready(self, autoplay) -> None:
        bot = autoplay.TsumogiriBot(delay=0.0)
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
        bot = autoplay.TsumogiriBot(delay=0.0)
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage()
        bot.act(page)
        bot._ready_at = 0.0  # noqa: SLF001 - 封包還沒回來,狀態仍說該打
        bot.act(page)
        assert len(page.clicks) == 1, "冷卻時間內不可以再點"

    def test_the_click_lands_on_the_drawn_tile(self, autoplay) -> None:
        """座標要落在手牌那一排的右端 —— 摸進來那張的位置。"""
        bot = autoplay.TsumogiriBot(delay=0.0)
        bot._ready_at = 0.0  # noqa: SLF001
        page = FakePage(1280, 720)
        bot.act(page)
        x, y = page.clicks[0]
        assert 0.75 * 1280 < x < 0.85 * 1280, "右端,但不到畫面邊緣"
        assert 0.85 * 720 < y < 0.99 * 720, "最下面那一排"

    def test_a_page_that_cannot_report_its_size_is_survivable(self, autoplay) -> None:
        """問不到大小就跳過這一巡,不可以拋例外 ——
        自動打牌壞掉不值得讓正在錄的那一場跟著死。"""
        bot = autoplay.TsumogiriBot(delay=0.0)
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
