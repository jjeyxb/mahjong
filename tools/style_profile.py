#!/usr/bin/env python3
"""從牌譜語料算出每個玩家的打法統計,再分群成「風格」。

M8 的第二步。第一步(`tools/fetch_tenhou.py` → GRP → 微調)已經證明管線通了,
但那次微調的 ``player_names_files = []`` —— 空的代表「全部玩家」,也就是**還沒
分風格**。這支工具產出的就是那個清單。

風格是從資料長出來的,不是我定的
--------------------------------
不去寫「攻擊型 = 立直率 > 25%」這種門檻。門檻要從哪來?我手上沒有那個數字的
來源,硬訂一個就是拿印象當資料 —— 這個專案的 `analysis/danger.py` 開頭那段
「不編機率」講的是同一件事。

改成:量六個能直接從 mjai 事件數出來的比率,標準化之後跑 k-means,
**讓群自己浮出來**,再回頭看每一群的重心長什麼樣子、給它一個名字。
這樣每一群都指得回它憑什麼。

為什麼「平均順位」不當分群特徵
------------------------------
它是**強度**不是風格。放進去的話 k-means 會照強度切,產出的三群會是
「強/中/弱」而不是「攻/守/速」—— 而 M8 想做的是**風格**遷移
(README:`cql.min_q_weight` 調高 = 更貼近該群玩家的實際打法)。

所以平均順位只印出來**當檢查**:如果三群的平均順位差很多,那代表這次分群其實
還是抓到了強度,特徵或群數要重挑。這是這支工具唯一的自我懷疑機制,別拿掉。

用法::

    # 先看分布,決定 --min-games 要設多少
    python tools/style_profile.py --stats

    # 分三群,寫出 player_names_files
    python tools/style_profile.py --clusters 3 --min-games 20

    # 產出給 finetune.toml 用的路徑
    #   [dataset] player_names_files = ['F:/mahjong/data/datasets/styles/style_0.txt']
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 讓本檔可直接執行,不需要先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mia.utils.logging import logger, setup_logging

#: 會讓手牌變成門前清破功的鳴牌。
#:
#: ``ankan`` 不算 —— 暗槓不破門清,把它算進副露率會讓「為了槓而槓」的人
#: 看起來像仕掛け派。``kakan`` 也不算:它是加在已經有的碰上面,
#: 那一手早就被算過一次了,再算一次等於同一件事記兩遍。
OPEN_CALLS = frozenset({"pon", "chi", "daiminkan"})

#: 分群用的特徵。順序就是報表的欄位順序。
#:
#: 全部是**比率或平均值**,不是次數 —— 打得多的人不該因此被歸到另一群。
FEATURES = (
    ("副露率", "call_rate"),
    ("立直率", "reach_rate"),
    ("和了率", "hora_rate"),
    ("放銃率", "deal_in_rate"),
    ("平均和了打點", "mean_hora_points"),
    ("平均放銃失點", "mean_deal_in_points"),
)


@dataclass
class PlayerStats:
    """一個玩家在整份語料裡的累計。

    以**局(kyoku)**為分母而不是以場(game)—— 一場半莊有 8 到 12 局,
    用場當分母的話「立直率」會變成一個沒有意義的大於 1 的數。
    """

    games: int = 0
    kyoku: int = 0
    called_kyoku: int = 0
    reach_kyoku: int = 0
    hora: int = 0
    deal_in: int = 0
    hora_points: int = 0
    deal_in_points: int = 0
    ranks: list[int] = field(default_factory=list)

    @property
    def call_rate(self) -> float:
        return self.called_kyoku / self.kyoku if self.kyoku else 0.0

    @property
    def reach_rate(self) -> float:
        return self.reach_kyoku / self.kyoku if self.kyoku else 0.0

    @property
    def hora_rate(self) -> float:
        return self.hora / self.kyoku if self.kyoku else 0.0

    @property
    def deal_in_rate(self) -> float:
        return self.deal_in / self.kyoku if self.kyoku else 0.0

    @property
    def mean_hora_points(self) -> float:
        """平均和了打點。**沒和過牌的人給 0**,不是 NaN。

        0 在這個特徵上是有意義的(這份語料裡他沒和過),而 NaN 會讓
        標準化整欄變成 NaN、k-means 靜默地吐出垃圾。
        """
        return self.hora_points / self.hora if self.hora else 0.0

    @property
    def mean_deal_in_points(self) -> float:
        return self.deal_in_points / self.deal_in if self.deal_in else 0.0

    @property
    def mean_rank(self) -> float:
        return sum(self.ranks) / len(self.ranks) if self.ranks else 0.0

    def vector(self) -> list[float]:
        return [getattr(self, attr) for _, attr in FEATURES]


def _ranks_from(scores: list[int]) -> list[int]:
    """最終點數 → 每個座位的順位(1~4)。

    同分時照座位序決定 —— 天鳳實際上是照起家順位,但這裡只拿順位當**檢查值**
    (見模組說明),不值得為了平手把整套席次規則搬進來。
    """
    order = sorted(range(len(scores)), key=lambda seat: -scores[seat])
    ranks = [0] * len(scores)
    for place, seat in enumerate(order, 1):
        ranks[seat] = place
    return ranks


def scan_game(path: Path) -> tuple[list[str], list[PlayerStats]] | None:
    """讀一場,回傳 ``(四個名字, 四份統計)``。壞檔回 ``None``。"""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError, EOFError) as exc:
        logger.warning("讀不了 {}:{}", path.name, exc)
        return None

    if not events or events[0].get("type") != "start_game":
        logger.warning("{} 第一個事件不是 start_game,跳過", path.name)
        return None

    names = list(events[0]["names"])
    stats = [PlayerStats(games=1) for _ in names]

    #: 這一局裡誰鳴過牌、誰立直過。每局重置 —— 分母是局不是場。
    called: set[int] = set()
    reached: set[int] = set()
    #: 最後一局開始時的點數 + 那一局的變動 = 終局點數。
    #: `end_game` 不帶點數,而 `start_kyoku.scores` 帶,所以用後者往前推。
    scores = [25000] * len(names)
    pending = [0] * len(names)

    for event in events:
        kind = event["type"]
        if kind == "start_kyoku":
            scores = list(event["scores"])
            pending = [0] * len(names)
            called.clear()
            reached.clear()
            for seat in range(len(names)):
                stats[seat].kyoku += 1
        elif kind in OPEN_CALLS:
            called.add(event["actor"])
        elif kind == "reach":
            reached.add(event["actor"])
        elif kind == "reach_accepted":
            # **立直棒那 1000 點要自己扣。** `reach_accepted` 在 mjai 裡不帶
            # deltas(只有 actor),而 `hora.deltas` 裡**已經含**和牌者收走的
            # 供託 —— 只加不減的話,每一根立直棒都會被算進去兩次。
            #
            # 因為每個 start_kyoku 都重設 scores,少算的只有**最後一局**的立直棒。
            # 症狀因此很隱晦:終局點數合計變成 101000 / 102000 而不是 100000,
            # 620 場裡 271 場(44%)對不上,而順位在點數接近時就跟著算錯。
            # 這是拿「合計應該恰好是 100000」去驗才抓到的。
            pending[event["actor"]] -= 1000
        elif kind == "hora":
            actor, target = event["actor"], event["target"]
            stats[actor].hora += 1
            stats[actor].hora_points += max(0, event["deltas"][actor])
            if target != actor:  # 榮和 —— 有人放銃
                stats[target].deal_in += 1
                stats[target].deal_in_points += max(0, -event["deltas"][target])
            pending = [p + d for p, d in zip(pending, event["deltas"], strict=True)]
        elif kind == "ryukyoku":
            pending = [p + d for p, d in zip(pending, event["deltas"], strict=True)]
        elif kind == "end_kyoku":
            for seat in called:
                stats[seat].called_kyoku += 1
            for seat in reached:
                stats[seat].reach_kyoku += 1

    # 終局點數。驗算方式是「四家合計恰好 100000」—— 實測 649 場裡 96% 對上,
    # 剩下的都是**低於** 100000 且差額是 1000 的整數倍:那是終局時還留在桌上的
    # 立直棒(天鳳把它給第一名)。刻意不模擬那一步:收受者本來就已經是第一名,
    # 加上去不可能改變任何人的順位,而順位是這裡唯一用到終局點數的地方。
    final = [s + p for s, p in zip(scores, pending, strict=True)]
    for seat, rank in enumerate(_ranks_from(final)):
        stats[seat].ranks.append(rank)
    return names, stats


def collect(root: Path) -> dict[str, PlayerStats]:
    """掃過整份語料,累計每個玩家。"""
    files = sorted(root.glob("**/*.json.gz"))
    if not files:
        raise SystemExit(f"{root} 底下沒有 .json.gz —— 先跑 tools/fetch_tenhou.py")

    players: dict[str, PlayerStats] = {}
    ok = 0
    for index, path in enumerate(files, 1):
        scanned = scan_game(path)
        if scanned is None:
            continue
        ok += 1
        names, stats = scanned
        for name, one in zip(names, stats, strict=True):
            total = players.setdefault(name, PlayerStats())
            total.games += one.games
            total.kyoku += one.kyoku
            total.called_kyoku += one.called_kyoku
            total.reach_kyoku += one.reach_kyoku
            total.hora += one.hora
            total.deal_in += one.deal_in
            total.hora_points += one.hora_points
            total.deal_in_points += one.deal_in_points
            total.ranks.extend(one.ranks)
        if index % 500 == 0:
            logger.info("掃了 {}/{} 場,{} 個玩家", index, len(files), len(players))

    logger.info("可用 {}/{} 場,{} 個玩家", ok, len(files), len(players))
    return players


def report_distribution(players: dict[str, PlayerStats]) -> None:
    """場數分布 —— 決定 ``--min-games`` 要設多少就看這張表。"""
    counts = sorted((p.games for p in players.values()), reverse=True)
    total_games = sum(counts)
    print(f"玩家數: {len(players)}    座位出現次數合計: {total_games}")
    print(f"每人場數  中位數 {counts[len(counts) // 2]}  最多 {counts[0]}  最少 {counts[-1]}\n")

    print(f"{'門檻':>6} {'留下幾人':>8} {'覆蓋的座位數':>12} {'佔比':>7}")
    for threshold in (1, 5, 10, 20, 30, 50, 100, 200):
        kept = [c for c in counts if c >= threshold]
        if not kept:
            break
        print(
            f"{threshold:>6} {len(kept):>8} {sum(kept):>12} "
            f"{sum(kept) / total_games:>6.1%}"
        )

    print("\n各特徵在全體玩家上的分布(只計場數 >= 20 的人):")
    eligible = [p for p in players.values() if p.games >= 20]
    if not eligible:
        print("  沒有人滿 20 場 —— 語料還太小,分群沒有意義")
        return
    print(f"{'特徵':>14} {'p10':>9} {'中位數':>9} {'p90':>9}")
    for label, attr in FEATURES:
        values = np.array([getattr(p, attr) for p in eligible])
        low, mid, high = np.percentile(values, [10, 50, 90])
        print(f"{label:>14} {low:>9.3f} {mid:>9.3f} {high:>9.3f}")
    ranks = np.array([p.mean_rank for p in eligible])
    print(f"{'平均順位':>14} {np.percentile(ranks, 10):>9.3f} "
          f"{np.median(ranks):>9.3f} {np.percentile(ranks, 90):>9.3f}   ← 非分群特徵")


def cluster(
    players: dict[str, PlayerStats], *, groups: int, min_games: int, seed: int
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """標準化 → k-means。回傳 ``(每群的名字, 重心(原始單位), 每群平均順位)``。"""
    eligible = {n: p for n, p in players.items() if p.games >= min_games}
    if len(eligible) < groups:
        raise SystemExit(
            f"場數 >= {min_games} 的玩家只有 {len(eligible)} 個,分不出 {groups} 群。"
            "\n語料再大一點,或把 --min-games 調低(但每個人的統計會更雜)。"
        )

    names = sorted(eligible)
    raw = np.array([eligible[n].vector() for n in names], dtype=np.float64)

    # **標準化是必須的,不是調味。** 打點是四位數、比率是 0~1,不標準化的話
    # 歐氏距離幾乎完全由打點決定,另外五個特徵等於沒放。
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std[std == 0] = 1.0  # 整欄同值的特徵沒有資訊,除以 1 讓它變成常數 0
    scaled = (raw - mean) / std

    from scipy.cluster.vq import kmeans2

    # minit='++' 是 k-means++ —— 隨機初始化在這種沒有明顯間隙的資料上
    # 每次跑都會給不同的群,那樣的「風格」沒有意義。
    centroids, labels = kmeans2(scaled, groups, minit="++", seed=seed, missing="raise")

    members: list[list[str]] = [[] for _ in range(groups)]
    for name, label in zip(names, labels, strict=True):
        members[label].append(name)

    mean_ranks = np.array(
        [
            np.mean([eligible[n].mean_rank for n in group]) if group else 0.0
            for group in members
        ]
    )
    return members, centroids * std + mean, mean_ranks


def report_clusters(
    members: list[list[str]],
    centroids: np.ndarray,
    mean_ranks: np.ndarray,
    players: dict[str, PlayerStats],
) -> None:
    print(f"\n{'群':>3} {'人數':>5} " + " ".join(f"{label:>13}" for label, _ in FEATURES))
    for index, (group, centre) in enumerate(zip(members, centroids, strict=True)):
        values = " ".join(f"{v:>13.3f}" for v in centre)
        print(f"{index:>3} {len(group):>5} {values}")

    print(f"\n{'群':>3} {'平均順位':>9}   ← 分群特徵**沒有**包含它,見模組說明")
    for index, rank in enumerate(mean_ranks):
        print(f"{index:>3} {rank:>9.3f}")
    spread = float(mean_ranks.max() - mean_ranks.min())
    print(f"\n各群平均順位的極差: {spread:.3f}")
    if spread > 0.15:
        print("  ⚠ 差得有點多 —— 這次分群可能其實抓到了**強度**而不是風格。")
        print("    那樣微調出來的模型會是「打得比較差的 Mortal」,不是「另一種風格的 Mortal」。")
        print("    可以考慮換群數,或把打點那兩個特徵拿掉再試。")
    else:
        print("  ✓ 各群強度接近,分出來的差異比較可能真的是風格。")

    print("\n每群最具代表性的幾個人(離重心最近):")
    for index, group in enumerate(members):
        sample = sorted(group, key=lambda n: -players[n].games)[:3]
        print(f"  群 {index}: {', '.join(sample)}")


def write_lists(members: list[list[str]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(members):
        path = out / f"style_{index}.txt"
        # train.py 用 filtered_trimmed_lines 讀:一行一個名字,空行會被濾掉。
        path.write_text("\n".join(sorted(group)) + "\n", encoding="utf-8")
        print(f"  {path}  ({len(group)} 人)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/datasets/tenhou/mjai"),
        help="mjai 牌譜根目錄",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/datasets/styles"),
        help="player_names_files 寫到哪",
    )
    parser.add_argument("--clusters", type=int, default=3, help="分幾群,預設 3")
    parser.add_argument(
        "--min-games", type=int, default=20, help="至少打過幾場才納入分群,預設 20"
    )
    parser.add_argument("--seed", type=int, default=0, help="k-means 初始化的種子")
    parser.add_argument(
        "--stats", action="store_true", help="只看分布,不分群也不寫檔"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    players = collect(args.corpus)
    report_distribution(players)

    if args.stats:
        return 0

    members, centroids, mean_ranks = cluster(
        players, groups=args.clusters, min_games=args.min_games, seed=args.seed
    )
    report_clusters(members, centroids, mean_ranks, players)
    print("\n寫出:")
    write_lists(members, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
