"""放銃分析的牌譜評測工具。

這支工具的產出會被寫進論文,所以**它自己也要被驗**。驗的重點有兩個:

1. **評估時點。** 一次捨牌要在打出去**之前**評估。順序寫錯的話那張牌會變成
   打者自己的現物,整件事就變成在問一個已經知道答案的問題 —— 而且不會報錯,
   只會得到一份漂亮到不可能的報告。
2. **對照組真的不一樣。** 三組判定吃同一批捨牌,若實作接錯會得到三組相同的
   數字,而那看起來只像「對照組沒有差別」這個結論。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from eval_danger import Result, scan_file, scan_game, usable_seats
from mia.analysis import DangerLevel
from mia.mjai import parse_event

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_game_full.jsonl"


@pytest.fixture(scope="module")
def result() -> Result:
    """那一場掃一次就好 —— 65 次捨牌 × 一個視角,但沒必要每個測試重跑。"""
    return scan_file(FIXTURE)


def _events(raw: list[dict]) -> list:
    return [parse_event(r) for r in raw]


def _start(tehais: list[list[str]], *, oya: int = 0) -> list[dict]:
    return [
        {"type": "start_game", "id": 0},
        {
            "type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
            "oya": oya, "dora_marker": "1z", "tehais": tehais,
            "scores": [25000] * 4,
        },
    ]


def _hand(*tiles: str) -> list[str]:
    return list(tiles)


class TestUsableSeats:
    """手牌不知道的視角量不出東西,而硬算仍然會得到一個數字 —— 要擋在前面。"""

    def test_a_full_information_log_gives_all_four(self) -> None:
        events = _events(_start([_hand(*(["1m"] * 13)) for _ in range(4)]))
        assert usable_seats(events) == [0, 1, 2, 3]

    def test_hidden_hands_are_skipped(self) -> None:
        tehais = [["?"] * 13, ["?"] * 13, _hand(*(["1m"] * 13)), ["?"] * 13]
        assert usable_seats(_events(_start(tehais))) == [2]

    def test_a_packet_recording_only_has_one_usable_seat(self) -> None:
        """封包錄影只看得到自己那一手。"""
        from eval_danger import load_game

        assert len(usable_seats(load_game(FIXTURE))) == 1


class TestTiming:
    """評估的時點。這一組是整支工具最容易靜靜地算錯的地方。"""

    def test_the_discarded_tile_is_not_its_own_furiten(self) -> None:
        """打 3m 的那一手,**不能**因為「3m 在他自己的河裡」而被判成安全。

        先更新牌河再評估就會變成這樣,而且每一手都安全 —— 報告會漂亮到不可能。
        """
        tehais = [_hand(*(["1m"] * 13)) for _ in range(4)]
        raw = [
            *_start(tehais),
            {"type": "tsumo", "actor": 0, "pai": "3m"},
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True},
        ]
        result = scan_game(_events(raw), seat=0)
        assert result.full.discards == 1
        assert result.full.called_safe == 0

    def test_a_ron_right_after_the_discard_is_counted(self) -> None:
        tehais = [_hand(*(["1m"] * 13)) for _ in range(4)]
        raw = [
            *_start(tehais),
            {"type": "tsumo", "actor": 0, "pai": "3m"},
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True},
            {"type": "hora", "actor": 2, "target": 0, "deltas": [-8000, 0, 8000, 0]},
        ]
        assert scan_game(_events(raw), seat=0).full.rons == 1

    def test_a_tsumo_is_not_a_deal_in(self) -> None:
        """自摸不是放銃。``actor == target`` 要排掉,不然放銃數會憑空多出來。"""
        tehais = [_hand(*(["1m"] * 13)) for _ in range(4)]
        raw = [
            *_start(tehais),
            {"type": "tsumo", "actor": 0, "pai": "3m"},
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "5p"},
            {"type": "hora", "actor": 1, "target": 1, "deltas": [-2000, 6000, -2000, -2000]},
        ]
        assert scan_game(_events(raw), seat=0).full.rons == 0


class TestTheKnownGame:
    """那一場已知答案的牌譜 —— 三次榮和,其中一次是這個錄影的人放的。

    這一組是整支工具的錨:**現行版本必須抓到那一手,第一版必須漏掉它**。
    """

    def test_exactly_one_deal_in(self, result: Result) -> None:
        assert result.full.rons == 1

    def test_the_current_version_never_said_safe(self, result: Result) -> None:
        """模組唯一的絕對主張:寫「安全」就是對三家都不可能。"""
        assert result.full.ron_when_safe == 0

    def test_the_reach_only_baseline_reproduces_the_old_bug(self, result: Result) -> None:
        """第一版對那一手說過「安全 —— 現物」,而那張牌正是放銃牌。

        對照組要能把那個錯誤重現出來,不然它量不到「修掉了多少」。
        """
        assert result.reach_only.ron_when_safe == 1

    def test_the_three_rules_do_not_collapse_into_one(self, result: Result) -> None:
        """三組判定接錯的話會得到三組相同的數字,而那看起來只像個結論。"""
        safes = {
            result.full.called_safe,
            result.furiten_only.called_safe,
            result.reach_only.called_safe,
        }
        assert len(safes) == 3

    def test_furiten_only_is_never_wrong_either(self, result: Result) -> None:
        """只認振聽是純規則,不可能說錯 —— 它的代價是蓋得很少。"""
        assert result.furiten_only.ron_when_safe == 0
        assert result.furiten_only.called_safe < result.full.called_safe


class TestTally:
    def test_caught_counts_risky_and_above(self) -> None:
        result = Result()
        result.full.add(DangerLevel.RISKY, ron=True)
        result.full.add(DangerLevel.VERY_RISKY, ron=True)
        result.full.add(DangerLevel.LIKELY_SAFE, ron=True)
        assert result.full.rons == 3
        assert result.full.caught == 2
