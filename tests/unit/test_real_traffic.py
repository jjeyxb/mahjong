"""以真實對局流量做端到端迴歸測試。

``tests/fixtures/real_game_excerpt.jsonl`` 是 CDP 從一場真實的雀魂 AI 房對局
錄下來的節錄(從 ``authGame`` 到和了),涵蓋座位解析、配牌、摸打、碰、立直、
自摸和了與流局。

**為什麼需要這個檔案:** 合成測試曾經全數通過,卻在真實流量上一個動作都沒解析出來
—— 因為測試資料是照著同一個錯誤假設造的(見 ``TestActionNameQualification``)。
自己造的資料只能驗證「實作符合我的理解」,無法驗證「我的理解符合現實」。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mia.groundtruth.dump import DumpStats, parse_dump
from mia.groundtruth.liqi import LiqiParser
from mia.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema
from mia.groundtruth.to_mjai import MajsoulToMjai
from mia.mjai.events import MjaiEvent

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_game_excerpt.jsonl"
FULL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_game_full.jsonl"

pytestmark = pytest.mark.skipif(
    not (FIXTURE.is_file() and DEFAULT_LIQI_PATH.is_file()),
    reason="需要 real_game_excerpt.jsonl 與 liqi.json",
)


@pytest.fixture(scope="module")
def schema() -> LiqiSchema:
    return LiqiSchema.load()


@pytest.fixture(scope="module")
def replay(schema: LiqiSchema) -> tuple[DumpStats, list[MjaiEvent]]:
    """把整份錄影跑過一次,回傳解析統計與 MJAI 事件流。"""
    stats = DumpStats()
    converters: dict[str, MajsoulToMjai] = {}
    events: list[MjaiEvent] = []
    for frame, message in parse_dump(FIXTURE, schema, stats=stats):
        converter = converters.get(frame.flow)
        if converter is None:
            converter = converters[frame.flow] = MajsoulToMjai(LiqiParser(schema))
        events.extend(converter.handle(message))
    return stats, events


class TestWireParsing:
    def test_every_frame_parses(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        """真實流量必須 100% 解得開。有任何一則失敗就是協定理解有誤。"""
        stats, _ = replay
        assert stats.total > 400
        assert stats.failed == 0, f"解析失敗 {stats.failed} 則: {stats.failures}"
        assert stats.parsed == stats.total

    def test_actions_are_extracted(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        stats, _ = replay
        assert stats.actions > 200, "一場對局應該有數百個動作"

    def test_all_live_actions_are_obfuscated(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """即時動作的 data 全部經過 XOR —— 這證實了金鑰表與演算法正確。

        若這個比例不是 100%,代表混淆方式變了,或我們選錯了嘗試順序。
        """
        stats, _ = replay
        assert stats.obfuscated_actions == stats.actions

    def test_multiple_flows_kept_separate(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """真實錄製確實會有多條連線(大廳與對局各自一條)。"""
        stats, _ = replay
        assert stats.flows > 1


class TestMjaiConversion:
    def test_game_boundaries(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        _, events = replay
        kinds = Counter(e.TYPE for e in events)
        assert kinds["start_game"] == 1
        assert kinds["end_game"] == 1

    def test_kyoku_boundaries_are_balanced(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """每個 start_kyoku 都要有對應的 end_kyoku,否則狀態機會漏掉重置。"""
        _, events = replay
        kinds = Counter(e.TYPE for e in events)
        assert kinds["start_kyoku"] == kinds["end_kyoku"] > 0

    def test_every_kyoku_ends_in_hora_or_ryukyoku(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        _, events = replay
        kinds = Counter(e.TYPE for e in events)
        terminal_rounds = kinds["ryukyoku"] + len(
            {id(e) for e in events if e.TYPE == "hora"}  # 多人榮和時一局可有多個 hora
        )
        assert terminal_rounds >= kinds["end_kyoku"]

    def test_seat_resolved_from_auth_game(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        _, events = replay
        start = next(e for e in events if e.TYPE == "start_game")
        assert 0 <= start.id <= 3
        assert len(start.names) == 4

    def test_own_hand_is_thirteen_tiles(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """配牌一定是 13 張 —— 雀魂發莊家 14 張,第 14 張要拆成 tsumo。"""
        _, events = replay
        start_game = next(e for e in events if e.TYPE == "start_game")
        for kyoku in (e for e in events if e.TYPE == "start_kyoku"):
            assert len(kyoku.tehais) == 4
            assert all(len(hand) == 13 for hand in kyoku.tehais)
            own = kyoku.tehais[start_game.id]
            assert "?" not in own, "自己的手牌必須全部可見"

    def test_other_hands_are_hidden(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        _, events = replay
        start_game = next(e for e in events if e.TYPE == "start_game")
        for kyoku in (e for e in events if e.TYPE == "start_kyoku"):
            for seat, hand in enumerate(kyoku.tehais):
                if seat != start_game.id:
                    assert hand == ["?"] * 13

    def test_start_kyoku_after_start_game(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        _, events = replay
        types = [e.TYPE for e in events]
        assert types.index("start_game") < types.index("start_kyoku")

    def test_reach_is_followed_by_dahai_then_accepted(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """立直的三段式:宣言 → 宣言牌 → (安全通過後)成立。"""
        _, events = replay
        reach_positions = [i for i, e in enumerate(events) if e.TYPE == "reach"]
        assert reach_positions, "這份錄影應該包含一次立直"
        for i in reach_positions:
            declarer = events[i].actor
            assert events[i + 1].TYPE == "dahai"
            assert events[i + 1].actor == declarer, "宣言牌必須由宣言者打出"
            accepted = next(
                (e for e in events[i + 2 :] if e.TYPE == "reach_accepted"), None
            )
            assert accepted is not None and accepted.actor == declarer

    def test_hora_deltas_balance_against_riichi_sticks(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """和了的點數增減加總必須是 1000 的非負整數倍。

        **不是零。** 立直棒的 1000 點在 MJAI 的記帳模型裡是由 ``reach_accepted``
        扣除的,不出現在 ``delta_scores``;但贏家收走供托時**會**算進 deltas。
        所以差額恰好等於收走的立直棒數 × 1000。

        本份錄影的實例:非莊家自摸,莊家付 2100、另兩家各付 1100(共 4300),
        加上自己那支立直棒 1000 → 贏家 +5300,加總為 +1000。
        """
        _, events = replay
        horas = [e for e in events if e.TYPE == "hora"]
        assert horas
        for hora in horas:
            assert hora.deltas is not None
            total = sum(hora.deltas)
            assert total >= 0, f"和了不可能讓總點數變少: {hora.deltas}"
            assert total % 1000 == 0, f"差額必須是立直棒的整數倍: {hora.deltas}"

    def test_ryukyoku_deltas_sum_to_zero(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        _, events = replay
        for ryukyoku in (e for e in events if e.TYPE == "ryukyoku"):
            if ryukyoku.deltas is not None:
                assert sum(ryukyoku.deltas) == 0, f"流局點數不守恆: {ryukyoku.deltas}"

    def test_tsumo_hora_targets_self(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        _, events = replay
        horas = [e for e in events if e.TYPE == "hora"]
        assert horas
        for hora in horas:
            assert 0 <= hora.target <= 3

    def test_melds_reference_a_different_seat(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """吃碰槓的來源必須是別家,自己不可能鳴自己的牌。"""
        _, events = replay
        melds = [e for e in events if e.TYPE in ("chi", "pon", "daiminkan")]
        assert melds, "這份錄影應該包含鳴牌"
        for meld in melds:
            assert meld.actor != meld.target

    def test_own_draws_are_visible_others_are_not(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """只看得到自己摸的牌;別家的一律是 ``?``。"""
        _, events = replay
        seat = next(e for e in events if e.TYPE == "start_game").id
        own = [e for e in events if e.TYPE == "tsumo" and e.actor == seat]
        others = [e for e in events if e.TYPE == "tsumo" and e.actor != seat]

        assert own and others
        assert all(e.pai != "?" for e in own), "自己摸的牌應該看得到"
        assert all(e.pai == "?" for e in others), "不該洩漏別家摸的牌"

    def test_all_tiles_are_valid_mjai_notation(
        self, replay: tuple[DumpStats, list[MjaiEvent]]
    ) -> None:
        """所有出現的牌都必須是合法的 MJAI 表示法。"""
        valid = {f"{n}{s}" for n in range(1, 10) for s in "mps"}
        valid |= {"5mr", "5pr", "5sr", "E", "S", "W", "N", "P", "F", "C", "?"}

        _, events = replay
        for event in events:
            for attr in ("pai", "dora_marker"):
                tile = getattr(event, attr, None)
                if isinstance(tile, str):
                    assert tile in valid, f"{event.TYPE}.{attr} 出現非法牌 {tile!r}"
            for attr in ("consumed", "tehais", "ura_markers"):
                value = getattr(event, attr, None)
                if not value:
                    continue
                flat = value if isinstance(value[0], str) else [t for row in value for t in row]
                for tile in flat:
                    assert tile in valid, f"{event.TYPE}.{attr} 出現非法牌 {tile!r}"

    def test_red_fives_appear(self, replay: tuple[DumpStats, list[MjaiEvent]]) -> None:
        """赤五要正確從雀魂的 0m/0p/0s 轉成 MJAI 的 5mr/5pr/5sr。"""
        _, events = replay
        tiles = {
            getattr(e, "pai", None) for e in events
        } | {
            t for e in events for t in (getattr(e, "tehais", None) or []) for t in t
        }
        assert tiles & {"5mr", "5pr", "5sr"}, "這份錄影應該有赤五出現"


class TestFullGame:
    """完整一場東風戰。

    ``real_game_excerpt.jsonl`` 只到第一次和了為止,涵蓋不到「一場比賽」層級的
    東西:連續五局、局間過場、``end_game``。這份是 2026-07-29 錄的完整對局,
    550 個 MJAI 事件、解析成功率 100%。

    加碼槓(``kakan``)只在這份素材裡出現過 —— 它與暗槓、大明槓的寶牌翻牌時機
    不同(後乗り),是 ``to_mjai`` 最容易寫錯的地方之一。
    """

    @staticmethod
    def _events() -> list[MjaiEvent]:
        schema = LiqiSchema.load()
        converters: dict[str, MajsoulToMjai] = {}
        events: list[MjaiEvent] = []
        for frame, message in parse_dump(FULL_FIXTURE, schema):
            converter = converters.setdefault(frame.flow, MajsoulToMjai(LiqiParser(schema)))
            events.extend(converter.handle(message))
        return events

    def test_every_frame_parses(self) -> None:
        stats = DumpStats()
        list(parse_dump(FULL_FIXTURE, LiqiSchema.load(), stats=stats))
        assert stats.failed == 0
        assert stats.total > 1000

    def test_the_whole_game_is_covered(self) -> None:
        kinds = Counter(e.TYPE for e in self._events())
        assert kinds["start_game"] == 1
        assert kinds["end_game"] == 1
        assert kinds["start_kyoku"] == kinds["end_kyoku"] == 5, "東風戰應該有五局"

    def test_the_rarer_actions_appear(self) -> None:
        """節錄那份沒有加槓 —— 那是寶牌翻牌時機最容易寫錯的地方。"""
        kinds = Counter(e.TYPE for e in self._events())
        for kind in ("chi", "pon", "kakan", "reach", "reach_accepted", "hora", "dora"):
            assert kinds[kind] > 0, f"完整對局裡應該要有 {kind}"
