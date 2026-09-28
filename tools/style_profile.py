#!/usr/bin/env python3
"""從牌譜語料算出每個玩家的打法統計,再挑出風格明確的玩家名單。

M8 的第二步。第一步(`tools/fetch_tenhou.py` → GRP → 微調)已經證明管線通了,
但那次微調的 ``player_names_files = []`` —— 空的代表「全部玩家」,也就是**還沒
分風格**。這支工具產出的就是那個清單。

⚠ 鳳凰卓玩家**不構成離散的風格群**(實測,見下)
-------------------------------------------------
原本的做法是「六個特徵標準化之後跑 k-means,讓群自己浮出來」。跑出來的三群
重心看起來很漂亮 —— 一群副露立直都多、一群放銃最少、一群門清高打點,
活像教科書上的攻/守/打點。**但那是 k-means 在一團連續的雲上隨手切的。**

2026-09-28 拿 344 個玩家(場數 >= 20)驗:

=========================================  ==========  ==================
檢查                                        結果        真有群該長的樣子
=========================================  ==========  ==================
換 6 個種子重跑的 Adjusted Rand Index       **+0.274**  0.9 以上
輪廓係數(silhouette)                      **+0.138**  >0.5 才算有結構
對照組:每一欄各自打亂之後的輪廓係數        **+0.120**  應該遠低於真實資料
=========================================  ==========  ==================

最後一列是關鍵:把**欄與欄的關聯完全破壞掉**之後,輪廓係數只從 0.138 掉到
0.120。也就是說那六個特徵裡幾乎沒有「成團」這件事 —— 鳳凰卓本來就是一群被
篩過的強手,打法收斂到很接近,這個結果其實很合理。

所以這支工具**不再假裝分群是發現**:

* ``--axis`` (**預設,建議用這個**) —— 沿一個**指名的**統計量取兩端。
  連續分布的「尾巴」是真的,即使中間沒有群。而且門檻是明說的
  (「副露率最高的 20%,也就是 > 0.381」),每一個名單都指得回它憑什麼。
* ``--clusters`` —— k-means 還留著,但它現在會**自己跑穩定性檢定**,
  不合格就明講「這不是分群,是切線」,不讓人拿一張漂亮的重心表去說服自己。

為什麼「平均順位」不是特徵
--------------------------
它是**強度**不是風格。拿它去選人會挑出「強的」與「弱的」,而 M8 想做的是
**風格**遷移(README:`cql.min_q_weight` 調高 = 更貼近該群玩家的實際打法)。
微調出「打得比較差的 Mortal」沒有意義。

所以平均順位只印出來**當檢查**:兩端的平均順位差很多,就代表這個軸選到的其實
是強度差異,那個軸不該用。

用法::

    # 先看分布,決定 --min-games 要設多少
    python tools/style_profile.py --stats

    # **建議**:沿「副露率」取最高與最低各 20%
    python tools/style_profile.py --axis 副露率 --tail 20

    # k-means(會先跑穩定性檢定,不合格會直說)
    python tools/style_profile.py --clusters 3

    # 產出給 finetune.toml 用的路徑(一次只填一群)
    #   [dataset] player_names_files = ['F:/mahjong/data/datasets/styles/副露率_high.txt']
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
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


def accumulate(target: PlayerStats, one: PlayerStats) -> None:
    """把一場的統計加進累計。"""
    target.games += one.games
    target.kyoku += one.kyoku
    target.called_kyoku += one.called_kyoku
    target.reach_kyoku += one.reach_kyoku
    target.hora += one.hora
    target.deal_in += one.deal_in
    target.hora_points += one.hora_points
    target.deal_in_points += one.deal_in_points
    target.ranks.extend(one.ranks)


def collect_per_game(root: Path) -> dict[str, list[PlayerStats]]:
    """掃過整份語料,每個玩家的**每一場各自留著**。

    不先加總是為了 :func:`split_half_reliability` —— 要拆半就必須還看得見
    單場,而那個檢定決定了哪個特徵能用、`--min-games` 該設多少。
    """
    files = sorted(root.glob("**/*.json.gz"))
    if not files:
        raise SystemExit(f"{root} 底下沒有 .json.gz —— 先跑 tools/fetch_tenhou.py")

    per_game: dict[str, list[PlayerStats]] = {}
    ok = 0
    for index, path in enumerate(files, 1):
        scanned = scan_game(path)
        if scanned is None:
            continue
        ok += 1
        names, stats = scanned
        for name, one in zip(names, stats, strict=True):
            per_game.setdefault(name, []).append(one)
        if index % 1000 == 0:
            logger.info("掃了 {}/{} 場,{} 個玩家", index, len(files), len(per_game))

    logger.info("可用 {}/{} 場,{} 個玩家", ok, len(files), len(per_game))
    return per_game


def collect(root: Path) -> dict[str, PlayerStats]:
    """掃過整份語料,累計每個玩家。"""
    players: dict[str, PlayerStats] = {}
    for name, games in collect_per_game(root).items():
        total = players.setdefault(name, PlayerStats())
        for one in games:
            accumulate(total, one)
    return players


def split_half_reliability(
    per_game: dict[str, list[PlayerStats]], *, min_games: int, seed: int = 0
) -> dict[str, tuple[float, int]]:
    """每個特徵裡有多少是這個人的特質,有多少只是牌運?

    做法:把每個玩家的對局隨機分兩半,各自算一次統計量,看兩半之間的相關,
    再用 Spearman-Brown 把「半份資料的信度」換算成「全份資料的信度」。

    **這是整個 ``--axis`` 做法成立的前提,也是 ``--clusters`` 失敗的原因。**
    實測(4807 場鳳凰卓語料)只有副露率撐得住:

    =============  =========  =========  =========
    特徵            場數 >=30  場數 >=50  場數 >=80
    =============  =========  =========  =========
    副露率          **0.647**  **0.653**  **0.722**
    立直率          0.322      0.528      0.553
    放銃率          0.367      0.346      0.519
    和了率          0.143      -0.029     0.268
    平均和了打點     -0.131     0.159      0.345
    平均放銃失點     0.048      0.026      0.297
    =============  =========  =========  =========

    和了率、打點、失點基本上是**雜訊** —— 會不會和牌、和多大,主要由牌運決定,
    不是個人特質。把六個特徵一起丟給 k-means,等於三分之二的維度是隨機數,
    那當然分不出群。

    Returns:
        ``{特徵名稱: (信度, 參與計算的人數)}``。信度 1.0 = 完全可靠、
        0 = 全是雜訊。負數也是「全是雜訊」(小樣本下相關會是負的)。
    """
    rng = random.Random(seed)
    halves: list[tuple[PlayerStats, PlayerStats]] = []
    for games in per_game.values():
        if len(games) < min_games:
            continue
        shuffled = games[:]
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        first, second = PlayerStats(), PlayerStats()
        for one in shuffled[:mid]:
            accumulate(first, one)
        for one in shuffled[mid : mid * 2]:  # 兩半場數相同,奇數時丟掉最後一場
            accumulate(second, one)
        halves.append((first, second))

    if len(halves) < 10:
        return {label: (float("nan"), len(halves)) for label, _ in FEATURES}

    result: dict[str, tuple[float, int]] = {}
    for label, attr in FEATURES:
        x = np.array([getattr(a, attr) for a, _ in halves])
        y = np.array([getattr(b, attr) for _, b in halves])
        if x.std() == 0 or y.std() == 0:
            result[label] = (float("nan"), len(halves))
            continue
        half = float(np.corrcoef(x, y)[0, 1])
        # Spearman-Brown。相關 <= -1 不會發生,但除零要擋。
        full = 2 * half / (1 + half) if half > -1 else float("nan")
        result[label] = (full, len(halves))
    return result


def report_reliability(reliability: dict[str, tuple[float, int]], min_games: int) -> None:
    people = next(iter(reliability.values()))[1]
    print(f"\n拆半信度(場數 >= {min_games} 的 {people} 人,每半約 {min_games // 2} 場):")
    print(f"{'特徵':>14} {'信度':>8}   判讀")
    for label, (value, _) in sorted(reliability.items(), key=lambda kv: -(kv[1][0] or 0)):
        if value != value:  # NaN
            verdict = "人數不足,算不出來"
        elif value >= 0.6:
            verdict = "可用 —— 這是穩定的個人特質"
        elif value >= 0.4:
            verdict = "勉強 —— 尾巴會混進不少運氣好壞"
        else:
            verdict = "**基本上是雜訊**,不要拿來選人"
        print(f"{label:>14} {value:>8.3f}   {verdict}")
    print("\n  信度低不是語料不夠大的問題,是**每個人的場數**不夠 ——")
    print("  放大語料時要用 tools/fetch_tenhou.py --focus-players 把場次集中,")
    print("  不是均勻多抓幾萬場。")


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


def _eligible(players: dict[str, PlayerStats], min_games: int) -> dict[str, PlayerStats]:
    keep = {n: p for n, p in players.items() if p.games >= min_games}
    if not keep:
        raise SystemExit(
            f"沒有人打滿 {min_games} 場。先跑 --stats 看分布,"
            "或把語料放大(tools/fetch_tenhou.py --since ... --focus-players ...)。"
        )
    return keep


def _standardise(
    eligible: dict[str, PlayerStats],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """回傳 ``(名字, 標準化後的矩陣, 每欄平均, 每欄標準差)``。

    **標準化是必須的,不是調味。** 打點是四位數、比率是 0~1,不標準化的話
    歐氏距離幾乎完全由打點決定,另外五個特徵等於沒放。
    """
    names = sorted(eligible)
    raw = np.array([eligible[n].vector() for n in names], dtype=np.float64)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std[std == 0] = 1.0  # 整欄同值的特徵沒有資訊,除以 1 讓它變成常數 0
    return names, (raw - mean) / std, mean, std


def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """平均輪廓係數。>0.5 算有結構,0.25~0.5 很弱,<0.25 基本沒有。

    自己寫是因為環境裡只有 scipy 沒有 sklearn,而為了一個 20 行的指標
    多一個相依不值得。
    """
    scores = []
    for index in range(len(data)):
        same = labels == labels[index]
        same[index] = False
        others = set(labels.tolist()) - {labels[index]}
        if not same.any() or not others:
            continue
        near = float(np.linalg.norm(data[same] - data[index], axis=1).mean())
        far = min(
            float(np.linalg.norm(data[labels == other] - data[index], axis=1).mean())
            for other in others
        )
        scores.append((far - near) / max(near, far))
    return float(np.mean(scores)) if scores else 0.0


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index:1 = 兩次分群完全一致,0 = 跟隨機分一樣。

    用它來問「換個隨機初始化,還會切出同一組群嗎」。資料真有群的話會;
    是一團連續的雲的話,每次切的位置都不一樣。
    """
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        table[int(x), int(y)] += 1

    def choose2(x: np.ndarray | float) -> np.ndarray | float:
        return x * (x - 1) / 2

    both = float(np.sum(choose2(table)))
    rows = float(np.sum(choose2(table.sum(axis=1))))
    cols = float(np.sum(choose2(table.sum(axis=0))))
    total = float(choose2(len(a)))
    expected = rows * cols / total
    maximum = (rows + cols) / 2
    return (both - expected) / (maximum - expected) if maximum != expected else 0.0


def cluster_is_real(scaled: np.ndarray, groups: int, *, seeds: int = 6) -> tuple[bool, str]:
    """這批資料真的分得出 ``groups`` 群嗎?

    問兩件事,兩件都必須過:

    1. **換種子還是同一個切法嗎**(ARI)。不是的話,「群」只是初始化的殘留。
    2. **比把欄位打亂的對照組有結構嗎**(輪廓係數)。把每一欄各自打亂會破壞欄與欄
       的關聯但保留每欄的分布 —— 那是「完全沒有結構」的基準線。真實資料若只比它
       好一點點,那就沒有群。

    這個函式存在的理由:**k-means 永遠會給你答案。** 它不會說「這裡沒有群」,
    它會回傳 k 個標籤和一張看起來很有意義的重心表。要問「有沒有群」,
    得另外問。
    """
    from scipy.cluster.vq import kmeans2

    runs = [
        kmeans2(scaled, groups, minit="++", seed=seed, missing="raise")[1]
        for seed in range(seeds)
    ]
    pairs = [
        adjusted_rand(runs[i], runs[j])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    stability = float(np.mean(pairs))
    real = silhouette(scaled, runs[0])

    rng = np.random.default_rng(0)
    shuffled = np.column_stack([rng.permutation(column) for column in scaled.T])
    _, control_labels = kmeans2(shuffled, groups, minit="++", seed=0, missing="raise")
    control = silhouette(shuffled, control_labels)

    lines = [
        f"換 {seeds} 個種子重跑的 ARI 平均 {stability:+.3f}(最低 {min(pairs):+.3f})"
        "  ← 真有群該是 0.9 以上",
        f"輪廓係數 {real:+.3f}  ← >0.5 才算有結構",
        f"對照組(每欄各自打亂)輪廓 {control:+.3f}  ← 參考值,不進判定,理由見下",
    ]
    if real <= control * 1.5:
        lines.append(
            "→ 真實資料只比對照組好一點點:那六個特徵裡幾乎沒有「成團」這件事"
        )

    # **判定只看 ARI 與輪廓係數,不看對照組。**
    #
    # 對照組(每欄各自打亂)在連續、單峰的特徵上是個好的虛無基準,但它會被
    # **雙峰的邊際分布**騙:真的分成三團時每一欄本身就是雙峰的,打亂之後會變成
    # 一個格子狀的點陣 —— 那個點陣自己就分得開。實測三個乾淨的高斯團:
    # ARI 1.000、輪廓 0.934,而對照組也有 0.507,於是「要比對照組好兩倍」
    # 這個條件把一個**明顯有群**的資料判成不合格。
    #
    # 這是寫測試時才發現的(`test_real_blobs_pass` 當場紅掉)。留著印出來當
    # 參考值有價值 —— 在真實語料那種單峰特徵上,它就是最直觀的那一行證據。
    ok = stability >= 0.7 and real >= 0.35
    return ok, "\n".join(f"  {line}" for line in lines)


def axis_groups(
    players: dict[str, PlayerStats],
    *,
    feature: str,
    tail: float,
    min_games: int,
    reliability: dict[str, tuple[float, int]] | None = None,
) -> dict[str, list[str]]:
    """沿一個指名的統計量取兩端。

    **這是建議的做法。** 連續分布的尾巴是真的,即使中間沒有群 —— 而門檻是
    明說的(「副露率最高的 20%,也就是 > 0.381」),不是 k-means 的黑盒標籤。

    Args:
        feature: :data:`FEATURES` 裡的中文名稱,例如 ``副露率``。
        tail: 每一端取幾 %(``20`` = 各取 20%,中間 60% 不用)。
    """
    attr = dict((label, name) for label, name in FEATURES).get(feature)
    if attr is None:
        raise SystemExit(
            f"沒有 {feature!r} 這個特徵。可用的是:"
            + "、".join(label for label, _ in FEATURES)
        )
    if not 0 < tail < 50:
        raise SystemExit(f"--tail 要在 0 與 50 之間(拿到 {tail})")

    if reliability is not None:
        score = reliability.get(feature, (float("nan"), 0))[0]
        if score != score or score < 0.4:
            print(f"\n⚠ **「{feature}」的拆半信度只有 {score:.3f},基本上是雜訊。**")
            print("  這個軸的「兩端」大半是運氣好壞,不是打法差異 ——")
            print("  拿它選出來的名單,微調出來的東西不會有可辨識的風格。")
            print("  先跑 --reliability 看哪個特徵撐得住,或把 --min-games 調高。")
        elif score < 0.6:
            print(f"\n⚠ 「{feature}」的信度 {score:.3f} 只算勉強,兩端會混進運氣成分。")

    eligible = _eligible(players, min_games)
    ranked = sorted(eligible, key=lambda n: getattr(eligible[n], attr))
    size = max(1, round(len(ranked) * tail / 100))
    low, high = ranked[:size], ranked[-size:]

    def describe(group: list[str]) -> str:
        values = np.array([getattr(eligible[n], attr) for n in group])
        ranks = np.array([eligible[n].mean_rank for n in group])
        return (
            f"{len(group):>4} 人  {feature} {values.min():.3f}~{values.max():.3f}"
            f"(平均 {values.mean():.3f})  平均順位 {ranks.mean():.3f}"
        )

    print(f"\n沿「{feature}」取兩端各 {tail:g}%(共 {len(eligible)} 人有資格):")
    print(f"  low : {describe(low)}")
    print(f"  high: {describe(high)}")

    low_rank = np.mean([eligible[n].mean_rank for n in low])
    high_rank = np.mean([eligible[n].mean_rank for n in high])
    gap = abs(low_rank - high_rank)
    print(f"\n兩端平均順位差 {gap:.3f}")
    if gap > 0.15:
        print("  ⚠ 差得有點多 —— 這個軸選到的可能是**強度**而不是風格。")
        print("    微調出「打得比較差的 Mortal」沒有意義,換一個軸再試。")
    else:
        print("  ✓ 兩端強度接近,差異比較可能真的是風格。")

    # 其他特徵在兩端各是多少 —— 一個軸不會只動一件事,這張表說明「拿到的是什麼」
    print(f"\n{'特徵':>14} {'low':>12} {'high':>12}")
    for label, name in FEATURES:
        lo = np.mean([getattr(eligible[n], name) for n in low])
        hi = np.mean([getattr(eligible[n], name) for n in high])
        mark = "  ← 選的軸" if label == feature else ""
        print(f"{label:>14} {lo:>12.3f} {hi:>12.3f}{mark}")

    return {f"{feature}_low": low, f"{feature}_high": high}


def cluster(
    players: dict[str, PlayerStats], *, groups: int, min_games: int, seed: int
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """標準化 → k-means。回傳 ``(每群的名字, 重心(原始單位), 每群平均順位)``。

    ⚠ **這條路在鳳凰卓語料上驗過,不合格** —— 見模組說明。留著是因為換了語料
    (別的卓等、別的平台)結果可能不同,而 :func:`cluster_is_real` 會當場說話。
    """
    eligible = _eligible(players, min_games)
    if len(eligible) < groups:
        raise SystemExit(
            f"場數 >= {min_games} 的玩家只有 {len(eligible)} 個,分不出 {groups} 群。"
            "\n語料再大一點,或把 --min-games 調低(但每個人的統計會更雜)。"
        )

    names, scaled, mean, std = _standardise(eligible)

    ok, detail = cluster_is_real(scaled, groups)
    print(f"\n分群檢定({groups} 群、{len(names)} 人):")
    print(detail)
    if ok:
        print("  ✓ 這批資料真的分得出群。")
    else:
        print("  ✗ **這不是分群,是切線。**")
        print("    k-means 永遠會給你答案 —— 它不會說「這裡沒有群」,")
        print("    它會回傳 k 個標籤和一張看起來很有意義的重心表。")
        print("    下面那張表的重心差異是真的,但**邊界是隨機初始化決定的**:")
        print("    換個種子,同一個人會被分到別群。")
        print("    要拿得住的名單請改用 --axis(沿一個指名的統計量取兩端)。")

    from scipy.cluster.vq import kmeans2

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

    print(f"\n{'群':>3} {'平均順位':>9}   ← **不是**分群特徵,見模組說明")
    for index, rank in enumerate(mean_ranks):
        print(f"{index:>3} {rank:>9.3f}")
    spread = float(mean_ranks.max() - mean_ranks.min())
    print(f"\n各群平均順位的極差: {spread:.3f}")
    if spread > 0.15:
        print("  ⚠ 差得有點多 —— 這次分群可能抓到了**強度**而不是風格。")
        print("    那樣微調出來的模型會是「打得比較差的 Mortal」。")
    else:
        print("  ✓ 各群強度接近。")
        print("    注意這只排除了「切到強度」,**不保證群是真的** ——")
        print("    那是上面那個分群檢定在回答的問題。一團均勻的雲隨手切三份,")
        print("    三份的平均順位當然也會很接近。")

    print("\n每群最具代表性的幾個人(離重心最近):")
    for index, group in enumerate(members):
        sample = sorted(group, key=lambda n: -players[n].games)[:3]
        print(f"  群 {index}: {', '.join(sample)}")


def write_lists(groups: dict[str, list[str]], out: Path) -> None:
    """一群一個檔。檔名帶群的來由(``副露率_high``),不是 ``style_0``。

    名字有意義很重要:`finetune.toml` 裡那一行 `player_names_files` 是幾週後
    回頭看還要看得懂的東西,而 `style_0` 什麼都沒說。
    """
    out.mkdir(parents=True, exist_ok=True)

    # **把上一次跑剩的名單清掉。** 這個目錄是工具的輸出,不是使用者的收藏夾。
    #
    # 不清的話會留下一地看起來一樣合法的 .txt:k-means 那版的 style_0.txt、
    # 拿雜訊軸試出來的 平均放銃失點_high.txt …… 而 finetune.toml 裡那行
    # `player_names_files` 只是一個路徑字串,指到哪一個都不會有人抱怨。
    # 幾週後回來看,沒有任何線索指出哪一份是當時真的採用的。
    keep = {f"{label}.txt" for label in groups}
    for stale in sorted(out.glob("*.txt")):
        if stale.name not in keep:
            stale.unlink()
            print(f"  清掉上一次的 {stale.name}")

    for label, group in groups.items():
        path = out / f"{label}.txt"
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--axis",
        metavar="特徵",
        help="**建議用這個**:沿一個指名的統計量取兩端,例如 --axis 副露率。"
        "可用:" + "、".join(label for label, _ in FEATURES),
    )
    mode.add_argument(
        "--clusters",
        type=int,
        help="k-means 分幾群。會先跑穩定性檢定 —— 鳳凰卓語料實測**不合格**,"
        "理由見模組說明",
    )
    parser.add_argument(
        "--tail", type=float, default=20.0, help="--axis 每一端取幾 %%,預設 20"
    )
    parser.add_argument(
        "--min-games", type=int, default=20, help="至少打過幾場才納入分群,預設 20"
    )
    parser.add_argument("--seed", type=int, default=0, help="k-means 初始化的種子")
    parser.add_argument(
        "--stats", action="store_true", help="只看分布,不選人也不寫檔"
    )
    parser.add_argument(
        "--reliability",
        action="store_true",
        help="跑拆半信度檢定就結束 —— 回答「哪個特徵量到的是打法,哪個只是牌運」。"
        "這是選軸的依據,也是 --clusters 失敗的原因",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    # 保留每一場才拆得半。加總很便宜,反過來就不行了。
    per_game = collect_per_game(args.corpus)
    players: dict[str, PlayerStats] = {}
    for name, games in per_game.items():
        total = players.setdefault(name, PlayerStats())
        for one in games:
            accumulate(total, one)

    report_distribution(players)

    if args.stats:
        return 0

    reliability = split_half_reliability(per_game, min_games=args.min_games)
    if args.reliability:
        report_reliability(reliability, args.min_games)
        return 0

    if args.clusters:
        members, centroids, mean_ranks = cluster(
            players, groups=args.clusters, min_games=args.min_games, seed=args.seed
        )
        report_clusters(members, centroids, mean_ranks, players)
        groups = {f"cluster_{i}": group for i, group in enumerate(members)}
    else:
        # 預設走軸模式。沒指定特徵就用副露率 —— 那是最常被當成「攻/守」的那一個,
        # 而且它與其他五個特徵的相關性最容易解釋(鳴牌多 → 和了快、打點低)。
        groups = axis_groups(
            players,
            feature=args.axis or FEATURES[0][0],
            tail=args.tail,
            min_games=args.min_games,
            reliability=reliability,
        )

    print("\n寫出:")
    write_lists(groups, args.out)
    print("\n填進 engines/mortal/finetune.toml(**一次只填一群**):")
    first = next(iter(groups))
    print(f"  [dataset] player_names_files = ['{args.out.as_posix()}/{first}.txt']")
    print("  ⚠ 換名單一定要刪掉 finetune_file_index.pth,理由見那份設定檔。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
