"""「這些群是真的嗎」這個問題本身要有測試。

`tools/style_profile.py` 原本直接相信 k-means 的輸出。實測發現鳳凰卓玩家在那六個
特徵上根本不成團,而 k-means **永遠會給你答案** —— 它不會說「這裡沒有群」,
它會回傳 k 個標籤和一張看起來很有意義的重心表。

所以判定「有沒有群」與「這個特徵量到的是打法還是牌運」的那幾個函式,
本身就是這支工具最重要的部分,而它們都是純函式,拿合成資料就測得動。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from style_profile import (
    PlayerStats,
    adjusted_rand,
    axis_groups,
    cluster_is_real,
    silhouette,
    split_half_reliability,
)


def _blobs(groups: int, per_group: int, spread: float, seed: int = 0) -> np.ndarray:
    """``groups`` 團高斯點。``spread`` 小 = 團與團分得開。"""
    rng = np.random.default_rng(seed)
    centres = np.eye(groups) * 10
    return np.vstack(
        [rng.normal(centre, spread, (per_group, groups)) for centre in centres]
    )


class TestAdjustedRand:
    def test_identical_labellings_score_one(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand(labels, labels) == pytest.approx(1.0)

    def test_relabelling_does_not_matter(self) -> None:
        """ARI 比的是「誰跟誰同組」,不是標籤的號碼。

        k-means 每次跑出來的群編號是任意的 —— 直接比標籤會把同一個切法
        判成完全不同,那個檢定就永遠不會通過。
        """
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([2, 2, 0, 0, 1, 1])
        assert adjusted_rand(a, b) == pytest.approx(1.0)

    def test_an_unrelated_partition_scores_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, 400)
        b = rng.integers(0, 3, 400)
        assert abs(adjusted_rand(a, b)) < 0.05

    def test_everything_in_one_group_is_not_an_error(self) -> None:
        """k-means 偶爾會吐出空群,ARI 不該因此爆掉。"""
        ones = np.zeros(10, dtype=int)
        assert adjusted_rand(ones, ones) == 0.0  # 沒有可比的配對


class TestSilhouette:
    def test_separated_blobs_score_high(self) -> None:
        data = _blobs(3, 40, spread=0.3)
        labels = np.repeat([0, 1, 2], 40)
        assert silhouette(data, labels) > 0.8

    def test_one_uniform_cloud_scores_low(self) -> None:
        """一團均勻的雲硬切三份,分數就會很低 —— 那正是真實語料的情況。"""
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, (120, 3))
        labels = np.repeat([0, 1, 2], 40)
        assert silhouette(data, labels) < 0.2

    def test_a_single_group_scores_zero(self) -> None:
        data = _blobs(1, 20, spread=1.0)
        assert silhouette(data, np.zeros(20, dtype=int)) == 0.0


class TestClusterIsReal:
    def test_real_blobs_pass(self) -> None:
        assert cluster_is_real(_blobs(3, 60, spread=0.4), 3)[0]

    def test_a_uniform_cloud_fails(self) -> None:
        """這是實際語料的形狀。它必須**不通過**。"""
        rng = np.random.default_rng(2)
        ok, detail = cluster_is_real(rng.normal(0, 1, (300, 6)), 3)
        assert not ok
        assert "ARI" in detail

    def test_the_report_always_names_the_control(self) -> None:
        """對照組那一行是整個判定的支點,不能只在失敗時才印。"""
        for data in (_blobs(3, 60, spread=0.4), np.random.default_rng(3).normal(0, 1, (200, 4))):
            assert "對照組" in cluster_is_real(data, 3)[1]


def _player(call_rate: float, games: int) -> list[PlayerStats]:
    """造一個「副露率固定」的玩家,每場一局。"""
    out = []
    for index in range(games):
        one = PlayerStats(games=1, kyoku=1)
        # 用交錯而不是隨機,讓每個玩家的實際比率精確等於 call_rate
        one.called_kyoku = 1 if (index % 100) < round(call_rate * 100) else 0
        one.ranks.append(1)
        out.append(one)
    return out


class TestSplitHalfReliability:
    def test_a_stable_trait_is_reliable(self) -> None:
        """每個玩家的副露率固定、彼此不同 → 拆半之後兩半應該高度相關。"""
        per_game = {
            f"p{i}": _player(call_rate=0.1 + 0.8 * i / 39, games=60) for i in range(40)
        }
        score, people = split_half_reliability(per_game, min_games=60)["副露率"]
        assert people == 40
        assert score > 0.9, f"穩定特質的信度應該很高,得到 {score}"

    def test_pure_luck_is_not_reliable(self) -> None:
        """所有玩家同一個真實比率、差異全來自抽樣 → 信度應該趨近 0。

        這一條是整個檢定的意義所在:量到的「差異」若全是運氣,
        照它選出來的兩端就是隨機挑人。
        """
        rng = np.random.default_rng(0)
        per_game = {}
        for i in range(60):
            games = []
            for _ in range(60):
                one = PlayerStats(games=1, kyoku=1)
                one.called_kyoku = int(rng.random() < 0.33)
                one.ranks.append(1)
                games.append(one)
            per_game[f"p{i}"] = games
        score, _ = split_half_reliability(per_game, min_games=60)["副露率"]
        assert score < 0.3, f"純運氣的信度應該趨近 0,得到 {score}"

    def test_too_few_players_reports_nan_instead_of_guessing(self) -> None:
        per_game = {"only": _player(0.3, games=40)}
        score, people = split_half_reliability(per_game, min_games=40)["副露率"]
        assert people == 1
        assert score != score  # NaN

    def test_players_below_the_threshold_are_excluded(self) -> None:
        per_game = {
            **{f"big{i}": _player(0.2 + 0.02 * i, games=40) for i in range(20)},
            **{f"small{i}": _player(0.9, games=5) for i in range(20)},
        }
        _, people = split_half_reliability(per_game, min_games=40)["副露率"]
        assert people == 20, "場數不足的人不該被算進信度"


class TestAxisGroups:
    @staticmethod
    def _players(rates: list[float]) -> dict[str, PlayerStats]:
        out = {}
        for index, rate in enumerate(rates):
            stats = PlayerStats(games=40, kyoku=100, called_kyoku=round(rate * 100))
            stats.ranks = [2, 3] * 20
            out[f"p{index:02d}"] = stats
        return out

    def test_the_two_tails_are_the_extremes(self) -> None:
        players = self._players([i / 100 for i in range(10, 60, 5)])  # 0.10 ~ 0.55
        groups = axis_groups(players, feature="副露率", tail=20, min_games=40)
        assert set(groups) == {"副露率_low", "副露率_high"}
        assert groups["副露率_low"] == ["p00", "p01"]
        assert groups["副露率_high"] == ["p08", "p09"]

    def test_an_unknown_feature_lists_the_valid_ones(self) -> None:
        with pytest.raises(SystemExit, match="副露率"):
            axis_groups(self._players([0.3] * 10), feature="氣勢", tail=20, min_games=1)

    @pytest.mark.parametrize("tail", [0, 50, 80, -5])
    def test_a_tail_outside_zero_to_fifty_is_rejected(self, tail: float) -> None:
        """50 以上兩端就會重疊 —— 同一個人同時進 low 與 high,而且不會有任何抱怨。"""
        with pytest.raises(SystemExit, match="tail"):
            axis_groups(self._players([0.3] * 10), feature="副露率", tail=tail, min_games=1)

    def test_nobody_qualifies_says_what_to_do(self) -> None:
        with pytest.raises(SystemExit, match="fetch_tenhou"):
            axis_groups(self._players([0.3] * 10), feature="副露率", tail=20, min_games=10_000)

    def test_a_noisy_axis_is_flagged(self, capsys) -> None:
        """選到雜訊特徵時要當場說,不能安靜地寫出一份隨機名單。"""
        axis_groups(
            self._players([i / 100 for i in range(10, 60, 5)]),
            feature="副露率",
            tail=20,
            min_games=40,
            reliability={"副露率": (0.05, 100)},
        )
        assert "基本上是雜訊" in capsys.readouterr().out
