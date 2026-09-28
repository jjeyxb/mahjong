"""打法統計的算術。

`tools/style_profile.py` 的分群那一半靠 k-means,沒什麼好釘的;**統計那一半
全是會算錯而且不會報錯的東西** —— 分母該是局還是場、暗槓算不算副露、自摸算不算
放銃、立直棒的 1000 點誰付。算錯了 k-means 照樣跑得出漂亮的三群,而每一群的
意義是錯的。

所以這裡用手寫的合成牌譜去釘每一條規則,而不是拿真實牌譜對「看起來合理」。
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from style_profile import PlayerStats, _ranks_from, collect, scan_game

NAMES = ["甲", "乙", "丙", "丁"]


def _hand(*events: dict) -> list[dict]:
    """一局:start_kyoku → 事件 → end_kyoku。"""
    return [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1s",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [["1m"] * 13 for _ in NAMES],
        },
        *events,
        {"type": "end_kyoku"},
    ]


def _write(path: Path, events: list[dict]) -> Path:
    body = [{"type": "start_game", "names": NAMES, "kyoku_first": 0, "aka_flag": True}]
    body.extend(events)
    body.append({"type": "end_game"})
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in body:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def game(tmp_path: Path):
    def build(events: list[dict]) -> list[PlayerStats]:
        scanned = scan_game(_write(tmp_path / "g.json.gz", events))
        assert scanned is not None
        names, stats = scanned
        assert names == NAMES
        return stats

    return build


class TestCallRate:
    def test_two_calls_in_one_hand_count_once(self, game) -> None:
        """分母是**局**,所以分子也必須是局 —— 不然副露率會超過 1。"""
        stats = game(
            _hand(
                {"type": "pon", "actor": 1, "target": 0, "pai": "1m", "consumed": ["1m", "1m"]},
                {"type": "chi", "actor": 1, "target": 0, "pai": "2p", "consumed": ["3p", "4p"]},
            )
        )
        assert stats[1].called_kyoku == 1
        assert stats[1].kyoku == 1
        assert stats[1].call_rate == 1.0

    def test_a_concealed_kan_is_not_a_call(self, game) -> None:
        """暗槓不破門前清。算進副露率的話「為了槓而槓」的人會被歸成仕掛け派。"""
        stats = game(_hand({"type": "ankan", "actor": 2, "consumed": ["1m"] * 4}))
        assert stats[2].called_kyoku == 0

    def test_an_added_kan_is_not_a_second_call(self, game) -> None:
        """加槓是加在已經算過的碰上面,再算一次等於同一件事記兩遍。"""
        stats = game(
            _hand(
                {"type": "kakan", "actor": 3, "pai": "1m", "consumed": ["1m", "1m", "1m"]},
            )
        )
        assert stats[3].called_kyoku == 0

    def test_a_called_kan_is_a_call(self, game) -> None:
        stats = game(
            _hand({"type": "daiminkan", "actor": 0, "target": 1, "pai": "1m",
                   "consumed": ["1m", "1m", "1m"]})
        )
        assert stats[0].called_kyoku == 1


class TestHoraAndDealIn:
    def test_a_ron_records_a_deal_in_for_the_discarder(self, game) -> None:
        stats = game(
            _hand({"type": "hora", "actor": 1, "target": 3,
                   "deltas": [0, 5800, 0, -5800], "ura_markers": []})
        )
        assert stats[1].hora == 1
        assert stats[1].hora_points == 5800
        assert stats[3].deal_in == 1
        assert stats[3].deal_in_points == 5800

    def test_a_tsumo_is_nobody_s_deal_in(self, game) -> None:
        """自摸時 ``actor == target``。不擋這條的話自摸的人會被記成放銃給自己。"""
        stats = game(
            _hand({"type": "hora", "actor": 2, "target": 2,
                   "deltas": [-1000, -1000, 3000, -1000], "ura_markers": []})
        )
        assert stats[2].hora == 1
        assert all(s.deal_in == 0 for s in stats)

    def test_a_player_who_never_won_scores_zero_not_nan(self, game) -> None:
        """NaN 會讓標準化整欄變成 NaN,k-means 靜默吐出垃圾。"""
        stats = game(_hand({"type": "ryukyoku", "deltas": [0, 0, 0, 0]}))
        assert stats[0].hora == 0
        assert stats[0].mean_hora_points == 0.0
        assert stats[0].mean_deal_in_points == 0.0


class TestRiichiSticks:
    def test_the_thousand_points_are_deducted(self, game) -> None:
        """立直棒不扣的話,四家合計會變成 101000 而不是 100000。

        `hora.deltas` **已經含**和牌者收走的供託,所以只加不減等於每根棒子
        算兩次。因為每個 `start_kyoku` 都重設點數,錯的只有最後一局 ——
        620 場真實牌譜裡 44% 對不上,而順位在點數接近時就跟著錯。
        """
        stats = game(
            _hand(
                {"type": "reach", "actor": 0},
                {"type": "reach_accepted", "actor": 0},
                # 1300 + 自己那根棒子收回來 = deltas 合計 +1000
                {"type": "hora", "actor": 0, "target": 1,
                 "deltas": [2300, -1300, 0, 0], "ura_markers": []},
            )
        )
        assert stats[0].reach_kyoku == 1
        # 0 家:25000 − 1000(立直棒)+ 2300 = 26300;1 家:25000 − 1300 = 23700
        assert stats[0].ranks == [1]
        assert stats[1].ranks == [4]

    def test_reach_is_counted_per_hand(self, game) -> None:
        stats = game(
            _hand({"type": "reach", "actor": 2}, {"type": "reach_accepted", "actor": 2})
        )
        assert stats[2].reach_rate == 1.0
        assert stats[0].reach_rate == 0.0


class TestRanks:
    def test_higher_score_is_a_better_rank(self) -> None:
        assert _ranks_from([30000, 25000, 24000, 21000]) == [1, 2, 3, 4]
        assert _ranks_from([21000, 24000, 25000, 30000]) == [4, 3, 2, 1]

    def test_ties_get_distinct_ranks(self) -> None:
        """平手照座位序 —— 順位只當檢查值,不值得把天鳳的席次規則搬進來。"""
        assert sorted(_ranks_from([25000, 25000, 25000, 25000])) == [1, 2, 3, 4]

    def test_ranks_come_from_the_last_hand_not_the_first(self, game) -> None:
        """`end_game` 不帶點數,終局點數要用最後一局的 `start_kyoku.scores` 往前推。

        取錯成第一局的話每一場都會是平手,平均順位變成一個常數 2.5 ——
        而那剛好會讓「各群強度接近」那條檢查永遠通過。
        """
        second = _hand({"type": "hora", "actor": 3, "target": 0,
                        "deltas": [-8000, 0, 0, 8000], "ura_markers": []})
        second[0]["scores"] = [25000, 25000, 25000, 25000]
        stats = game(_hand({"type": "ryukyoku", "deltas": [0, 0, 0, 0]}) + second)
        assert stats[3].ranks == [1]
        assert stats[0].ranks == [4]
        assert stats[0].kyoku == 2


class TestCorpusScan:
    def test_stats_accumulate_across_games(self, tmp_path: Path) -> None:
        for index in range(3):
            _write(
                tmp_path / f"g{index}.json.gz",
                _hand({"type": "pon", "actor": 0, "target": 1, "pai": "1m",
                       "consumed": ["1m", "1m"]}),
            )
        players = collect(tmp_path)
        assert players["甲"].games == 3
        assert players["甲"].kyoku == 3
        assert players["甲"].called_kyoku == 3
        assert players["乙"].called_kyoku == 0

    def test_a_corrupt_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """一份壞檔不該讓整趟掃描停下來 —— 語料是幾萬個檔案,總會有壞的。"""
        _write(tmp_path / "good.json.gz", _hand({"type": "ryukyoku", "deltas": [0] * 4}))
        (tmp_path / "bad.json.gz").write_bytes(b"not gzip at all")
        players = collect(tmp_path)
        assert players["甲"].games == 1

    def test_an_empty_corpus_says_what_to_run(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="fetch_tenhou"):
            collect(tmp_path)
