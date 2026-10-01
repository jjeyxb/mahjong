#!/usr/bin/env python
"""拿牌譜量放銃分析:它說「安全」的時候,到底有沒有放銃過。

這個模組唯一的**絕對主張**是:畫面上寫「安全」只在對三家都不可能放銃的時候。
那句話先前只被一場牌譜檢驗過(`tests/fixtures/real_game_full.jsonl`),而它已經
被真實牌譜推翻過兩次 —— 兩次都是「規則想窄了」,而且都沒有任何錯誤訊息。
所以這支工具要回答的是三個數字:

1. **被判「安全」卻放銃的次數。** 核心主張是這個該是 **0**。
2. **實際放銃牌被判成危險以上的比例。** 召回率 —— 它抓不抓得到。
3. **對照組差多少。** 只認振聽 / 只認立直(就是被推翻的第一版)。

第 3 點不能省。前兩個數字沒有尺:一個把每張牌都說成「非常危險」的模組,
放銃召回率是 100%,而它毫無用處。對照組就是那把尺。

四個視角
--------
天鳳牌譜是第三人稱、沒有 ``start_game.id``,而 MIA 的事件要靠它決定自己坐哪。
所以每一場**各從四家的視角重播一次**:注入 ``id``、用那一家的 ``tehais`` 當起手。

這不只是為了繞過格式問題,而是真的要這樣做 —— 放銃分析是對稱的(對三家各算
一次排除法),同一場從四個座位看會得到四組獨立的「他打了什麼、中了沒有」。
資料量 ×4,而且不必多抓牌譜。

量的是什麼時點
--------------
一次捨牌要在**打出去之前**評估:牌河裡還沒有這一張,手上還有它。順序寫錯的話
那張牌會變成打者自己的現物,整件事就變成在問一個已經知道答案的問題 ——
而且不會報錯,只會得到一份漂亮到不可能的報告。

用法::

    # 先對已知答案的那一場驗工具本身(那場有三次榮和,兩次是沒立直的人和的)
    python tools/eval_danger.py tests/fixtures/real_game_full.jsonl

    # 再掃語料。--limit 先跑小樣本,整批 25,543 場要數小時
    python tools/eval_danger.py data/datasets/tenhou/mjai --limit 300
    python tools/eval_danger.py data/datasets/tenhou/mjai --out reports/danger.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.analysis import DangerLevel, TileDanger, assess
from mia.analysis.danger import _level
from mia.mjai import (
    Dahai,
    Hora,
    MjaiEvent,
    MjaiFormatError,
    StartGame,
    StartKyoku,
    parse_event,
)
from mia.mjai.handstate import HandTracker
from mia.mjai.table import TableTracker
from mia.mjai.tiles import normalize_red
from mia.utils.logging import setup_logging

SEATS = 4


@dataclass(slots=True)
class Tally:
    """一組判定的成績。

    Attributes:
        by_level: 每個危險度各判了幾次捨牌。
        ron_by_level: 其中幾次真的被榮和。
        called_safe: 判成「安全」的次數(``by_level`` 的 SAFE 格,另外存一份
            是為了讓最重要的那兩個數字並排)。
        ron_when_safe: **判成「安全」卻放銃。核心主張是這個是 0。**
    """

    by_level: Counter[int] = field(default_factory=Counter)
    ron_by_level: Counter[int] = field(default_factory=Counter)

    def add(self, level: DangerLevel, *, ron: bool) -> None:
        self.by_level[int(level)] += 1
        if ron:
            self.ron_by_level[int(level)] += 1

    @property
    def discards(self) -> int:
        return sum(self.by_level.values())

    @property
    def rons(self) -> int:
        return sum(self.ron_by_level.values())

    @property
    def called_safe(self) -> int:
        return self.by_level[int(DangerLevel.SAFE)]

    @property
    def ron_when_safe(self) -> int:
        return self.ron_by_level[int(DangerLevel.SAFE)]

    @property
    def caught(self) -> int:
        """放銃牌被判成「危險」或「非常危險」的次數 —— 召回率的分子。"""
        return self.ron_by_level[int(DangerLevel.RISKY)] + self.ron_by_level[
            int(DangerLevel.VERY_RISKY)
        ]

    def merge(self, other: Tally) -> None:
        self.by_level.update(other.by_level)
        self.ron_by_level.update(other.ron_by_level)


@dataclass(slots=True)
class Result:
    """一次掃描的全部成績。

    三組判定吃的是**同一批捨牌**,所以可以直接比 —— 差異只來自判定規則。
    """

    full: Tally = field(default_factory=Tally)
    furiten_only: Tally = field(default_factory=Tally)
    reach_only: Tally = field(default_factory=Tally)
    games: int = 0
    desyncs: int = 0

    def merge(self, other: Result) -> None:
        self.full.merge(other.full)
        self.furiten_only.merge(other.furiten_only)
        self.reach_only.merge(other.reach_only)
        self.games += other.games
        self.desyncs += other.desyncs


def _levels(danger: TileDanger, table: TableTracker, tile: str) -> tuple[DangerLevel, ...]:
    """同一次捨牌在三種判定下各是什麼等級:(現行, 只認振聽, 只認立直)。

    **三個都從同一份 ``assess`` 結果推出來。** 分開各算一次的話不只慢一倍,
    還會讓三組數字有機會因為實作走鐘而不可比 —— 而這份報告的全部意義就是
    它們吃的是同一批捨牌。

    型數 → 等級一律借分析層那支 ``_level``,**不在這裡再寫一份**:門檻不是
    線性的(0 / ≤2 / ≤4 / 其餘),自己換算過一次就錯過一次。
    """
    # 現行:對三家都算排除法,取最危險的那一家(放銃只需要中一個人)
    full = danger.level

    # 對照一:**只認振聽**,不算筋也不算壁 —— 唯一完全不需要推論的規則。
    # 現行判成安全的牌一定包含這一組,所以兩者的差就是筋與壁多蓋到的範圍,
    # 而那段範圍有沒有代價正是要量的東西。
    furiten = (
        DangerLevel.SAFE
        if all(tile in table.players[s].safe for s in table.opponents)
        else DangerLevel.VERY_RISKY
    )

    # 對照二:**只評估已宣告立直的家** —— 被真實牌譜推翻的那個第一版。
    # 沒有人立直時它無話可說,只能回「安全」,而那正是它錯的地方:
    # 真實牌譜裡三次榮和有兩次是沒立直的人和的。
    reached = [s for s in danger.seats if table.players[s.seat].reach]
    reach_only = _level(max((len(s.waits) for s in reached), default=0))

    return full, furiten, reach_only


def _ron_targets(events: list[MjaiEvent], index: int) -> set[int]:
    """緊接在 ``index`` 這次捨牌之後,誰被榮和了。

    只看下一個 ``hora``:再往後就是下一巡的事了。``actor == target`` 是自摸,
    不算放銃。雙響(兩家同時榮和一張)會給出兩個 ``hora``,所以回的是集合。
    """
    targets: set[int] = set()
    for event in events[index + 1 :]:
        if isinstance(event, Hora):
            if event.actor != event.target:
                targets.add(event.target)
            continue
        if targets or isinstance(event, Dahai | StartKyoku):
            break
    return targets


def scan_game(events: list[MjaiEvent], seat: int) -> Result:
    """從 ``seat`` 的視角重播一場,量他每一次捨牌。

    座位直接設在兩個追蹤器上,而 ``start_game`` **不餵進去** —— 它帶的 ``id``
    是檔案裡寫的那一家,餵下去會把這裡指定的視角覆蓋掉。這樣也省掉為了四個
    視角把同一個檔案解析四次。
    """
    result = Result(games=1)
    table = TableTracker()
    table.seat = seat
    hand = HandTracker()
    hand.seat = seat

    for index, event in enumerate(events):
        if isinstance(event, StartGame):
            continue
        if isinstance(event, Dahai) and event.actor == seat and hand.in_sync:
            tiles = list(hand.tiles)
            tile = normalize_red(event.pai)
            # `assess` 的 tile 是正規化過的(赤五當普通五),這裡不跟著轉的話
            # 打赤五那幾手會找不到對應那一列 —— 而找不到時下面那個 `next`
            # 會悄悄回 None,整手變成沒被量到
            danger = next((d for d in assess(tiles, table).tiles if d.tile == tile), None)
            if danger is not None:
                ron = seat in _ron_targets(events, index)
                full, furiten, reach_only = _levels(danger, table, tile)
                result.full.add(full, ron=ron)
                result.furiten_only.add(furiten, ron=ron)
                result.reach_only.add(reach_only, ron=ron)
        # 先評估再更新 —— 順序反了的話那張牌會變成打者自己的現物
        table.handle(event)
        hand.feed(event)

    if not hand.in_sync:
        result.desyncs = 1
    return result


def load_game(path: Path) -> list[MjaiEvent]:
    """讀一場。mjai 事件流(可 gzip)與原始 WebSocket 錄影都吃。

    ``start_game`` 會被補上一個 ``id`` —— 天鳳牌譜是第三人稱、沒有這個欄位,
    而 MIA 的 ``StartGame`` 要求它。補的值無所謂(視角由 :func:`scan_game`
    自己設),重點是不要為此放寬那個欄位:事件流是誰的視角該由呼叫端決定,
    不該讓解析器去猜。
    """
    if path.suffix not in (".gz", ".jsonl", ".json"):
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    events: list[MjaiEvent] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or "type" not in raw:
                if number == 1:
                    return _from_ws_dump(path)
                continue
            if raw["type"] == "start_game":
                raw.setdefault("id", 0)
            events.append(parse_event(raw))
    return events


def _from_ws_dump(path: Path) -> list[MjaiEvent]:
    """原始封包錄影 —— 轉成 mjai 之後只有**錄影那一家**的手牌是已知的。

    所以這條路進來的檔案通常只有一個可用視角,見 :func:`usable_seats`。
    """
    from mia.groundtruth.stream import MjaiDecoder

    decoder = MjaiDecoder()
    return [event for decoded in decoder.decode_file(path) for event in decoded.events]


def usable_seats(events: list[MjaiEvent]) -> list[int]:
    """哪幾家的視角量得出東西 —— **手牌是已知的那幾家**。

    天鳳牌譜是完全情報,四家都行。封包錄影只看得到自己那一手,其餘三家的
    ``tehais`` 是一排 ``?``,硬算的話 ``HandTracker`` 會從第一張就脫節,
    而 :func:`assess` 收到一手問號仍然**算得出一個數字** —— 那是這支工具最
    容易產生垃圾資料的地方,所以在這裡就擋掉。
    """
    for event in events:
        if isinstance(event, StartKyoku):
            return [s for s, tiles in enumerate(event.tehais) if "?" not in tiles]
    return []


def scan_file(path: Path) -> Result:
    """一場 × 每一個手牌已知的視角。"""
    total = Result()
    try:
        events = load_game(path)
    except (MjaiFormatError, json.JSONDecodeError, OSError, ValueError):
        return total
    for seat in usable_seats(events):
        total.merge(scan_game(events, seat))
    return total


def _pct(part: int, total: int) -> str:
    return "—" if total == 0 else f"{part / total:.2%}"


def report(result: Result) -> str:
    lines = [
        f"重播 {result.games} 個場次視角(手牌脫節 {result.desyncs} 個)",
        "",
    ]
    for name, tally, note in (
        ("現行(三家排除法)", result.full, "畫面上真正在用的"),
        ("對照:只認振聽", result.furiten_only, "不算筋、不算壁"),
        ("對照:只認立直", result.reach_only, "被真實牌譜推翻的第一版"),
    ):
        lines += [
            f"── {name} —— {note}",
            f"   捨牌 {tally.discards}、其中放銃 {tally.rons}"
            f"({_pct(tally.rons, tally.discards)})",
            f"   判「安全」 {tally.called_safe}({_pct(tally.called_safe, tally.discards)})",
            f"   **判「安全」卻放銃 {tally.ron_when_safe}**"
            f"({_pct(tally.ron_when_safe, tally.called_safe)} of 安全)",
            f"   放銃牌被判危險以上 {tally.caught}/{tally.rons}"
            f"({_pct(tally.caught, tally.rons)})",
            "",
        ]
        for level in DangerLevel:
            count = tally.by_level[int(level)]
            if count:
                lines.append(
                    f"      {level.label:<6} {count:>8}  放銃 {tally.ron_by_level[int(level)]:>5}"
                    f"  ({_pct(tally.ron_by_level[int(level)], count)})"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="一個 mjai 檔,或一個裝著它們的目錄")
    parser.add_argument("--limit", type=int, default=0, help="最多幾場(0 = 全部)")
    parser.add_argument("--out", type=Path, help="把統計寫成 JSON")
    parser.add_argument("--progress", type=int, default=100, help="每幾場印一次進度")
    parser.add_argument("--log-level", default="ERROR", help="日誌等級,預設 ERROR")
    args = parser.parse_args(argv)

    # 手牌脫節會逐張 warning,而掃兩萬五千場時那是幾十萬行 —— 統計裡有 desyncs
    setup_logging(level=args.log_level)

    if args.path.is_dir():
        files = sorted(args.path.rglob("*.json.gz")) + sorted(args.path.rglob("*.jsonl"))
    else:
        files = [args.path]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"{args.path} 底下找不到牌譜")

    print(f"{len(files)} 場,每場量所有手牌已知的視角\n")
    total = Result()
    for number, path in enumerate(files, 1):
        total.merge(scan_file(path))
        if args.progress and number % args.progress == 0:
            print(
                f"  {number}/{len(files)} 場 —— 判安全卻放銃"
                f" {total.full.ron_when_safe}(現行)/"
                f" {total.reach_only.ron_when_safe}(只認立直)",
                flush=True,
            )

    print("\n" + report(total))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    name: {
                        "by_level": dict(tally.by_level),
                        "ron_by_level": dict(tally.ron_by_level),
                    }
                    for name, tally in (
                        ("full", total.full),
                        ("furiten_only", total.furiten_only),
                        ("reach_only", total.reach_only),
                    )
                }
                | {"games": total.games, "desyncs": total.desyncs},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"統計寫到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
