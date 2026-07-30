#!/usr/bin/env python
"""把錄下來的對局重播給引擎,看它們會怎麼打。

這是功能 2 的離線驗證工具,也是功能 3 比較風格的起點。三件事一次做完:

1. **引擎真的接得起來嗎** —— 子程序、握手、協定、逾時,整條路走一遍。
2. **引擎之間差在哪** —— 同一個局面並排多個引擎,只有分歧的那幾手值得看。
3. **與真人比多接近** —— 錄影裡本來就有那個人實際打了什麼,拿它當對照。

第 3 點要小心解讀:一致率**不是**準確率。人打錯的時候引擎不跟著錯,一致率反而
會下降。它衡量的是「像不像這個人」,那正好是功能 3 想調的東西。

用法::

    # 先把錄影轉成 MJAI 事件流(或直接餵原始錄影,會自動轉)
    tools/gt.py inspect data/gt/ws.jsonl --mjai-out data/gt/mjai.jsonl

    # 規則式 baseline 與 Mortal 並排
    python tools/advise.py data/gt/mjai.jsonl --mortal models/mortal_298k.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.engine import AIEngine, DummyEngine, EngineGroup
from mia.groundtruth.liqi import LiqiParser
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.to_mjai import MajsoulToMjai
from mia.mjai import MjaiEvent, MjaiFormatError, parse_event
from mia.utils.logging import setup_logging

#: 這些是「輪到自己時可以做的事」。比對引擎建議與真人實際動作時,只看這些。
DECISIONS = frozenset(
    {"dahai", "reach", "chi", "pon", "ankan", "kakan", "daiminkan", "hora", "ryukyoku", "kita"}
)


def load_events(path: Path) -> tuple[list[MjaiEvent], str]:
    """讀事件流。檔案可以是 MJAI 事件流,也可以是原始的 WebSocket 錄影。

    自動判斷而不是加一個 ``--format`` 參數:兩種檔案的第一行差很多,猜錯的
    機率遠低於使用者傳錯旗標的機率。
    """
    first = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first = line
                break
    if not first:
        raise SystemExit(f"{path} 是空的")

    try:
        parse_event(json.loads(first))
    except (MjaiFormatError, json.JSONDecodeError):
        return _from_dump(path), "WebSocket 錄影"

    events = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(parse_event(json.loads(line)))
            except (MjaiFormatError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{path}:{number} 解析失敗 — {exc}") from exc
    return events, "MJAI 事件流"


def _from_dump(path: Path) -> list[MjaiEvent]:
    from mia.groundtruth.dump import parse_dump

    schema = LiqiSchema.load()
    converters: dict[str, MajsoulToMjai] = {}
    events: list[MjaiEvent] = []
    for frame, message in parse_dump(path, schema):
        converter = converters.get(frame.flow)
        if converter is None:
            converter = converters[frame.flow] = MajsoulToMjai(LiqiParser(schema))
        events.extend(converter.handle(message))
    return events


def own_seat(events: list[MjaiEvent]) -> int:
    for event in events:
        if event.TYPE == "start_game":
            return int(getattr(event, "id", 0))
    raise SystemExit("事件流裡沒有 start_game,不知道自己坐哪 —— 這份錄影可能只錄到大廳")


def actual_next(events: list[MjaiEvent], start: int, seat: int) -> MjaiEvent | None:
    """真人在這個局面之後實際做的第一件事。

    立直會拆成 ``reach`` 與宣言牌的 ``dahai`` 兩個事件,這裡只取第一個 ——
    引擎的回覆也是一次一個,對齊才有意義。
    """
    for event in events[start:]:
        if event.TYPE in DECISIONS and getattr(event, "actor", None) == seat:
            return event
        if event.TYPE in {"end_kyoku", "start_kyoku"}:
            return None
    return None


def build_engines(args: argparse.Namespace) -> list[AIEngine]:
    engines: list[AIEngine] = []
    if not args.no_baseline:
        engines.append(DummyEngine())
    for weights in args.mortal or []:
        from mia.engine.mortal import mortal_engine

        engines.append(mortal_engine(weights, seat=args.seat, name=Path(weights).stem))
    if not engines:
        raise SystemExit("一個引擎都沒有 —— 至少給 --mortal,或別加 --no-baseline")
    return engines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=Path, help="MJAI 事件流或原始 WebSocket 錄影")
    parser.add_argument(
        "--mortal",
        action="append",
        metavar="WEIGHTS",
        help="加一個 Mortal 引擎。可重複 —— 這就是比較多份權重的方式",
    )
    parser.add_argument("--no-baseline", action="store_true", help="不要規則式 baseline")
    parser.add_argument("--limit", type=int, default=30, help="最多印幾個決策點,預設 30")
    parser.add_argument("--only-disagreement", action="store_true", help="只印引擎之間不一致的")
    parser.add_argument("--log-level", default="WARNING", help="日誌等級,預設 WARNING")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)

    events, kind = load_events(args.file)
    if not events:
        raise SystemExit(f"{args.file} 裡沒有事件")
    args.seat = own_seat(events)
    print(f"{args.file}({kind}):{len(events)} 個事件,自己坐 {args.seat}\n")

    engines = build_engines(args)
    shown = 0
    decisions = 0
    agree_with_human: Counter[str] = Counter()
    unanimous = 0

    with EngineGroup(engines) as group:
        names = [e.name for e in group]
        for index, event in enumerate(events):
            result = group.react(event)
            if not result.actions:
                continue

            decisions += 1
            unanimous += result.is_unanimous
            human = actual_next(events, index + 1, args.seat)
            for advice in result.actions:
                if human is not None and str(advice.action) == str(human):
                    agree_with_human[advice.engine] += 1

            if args.only_disagreement and result.is_unanimous:
                continue
            if shown < args.limit:
                shown += 1
                print(f"#{decisions:<4} 局面 {event}")
                for advice in result.actions:
                    mark = "=" if human is not None and str(advice.action) == str(human) else " "
                    print(f"      {mark} {advice}")
                print(f"      真人 {human if human is not None else '(沒有後續動作)'}\n")

        print(f"決策點 {decisions} 個,引擎之間一致 {_pct(unanimous, decisions)}")
        for name in names:
            if name not in group.failed:
                print(f"  {name:<20} 與真人一致 {_pct(agree_with_human[name], decisions)}")
        for name, reason in group.failed.items():
            print(f"  {name:<20} 中途掉隊 — {reason.splitlines()[0]}")

    print("\n一致率不是準確率 —— 人打錯的時候引擎不跟著錯,這個數字反而會下降。")
    return 0


def _pct(part: int, total: int) -> str:
    return f"{part}/{total}" if total == 0 else f"{part}/{total} ({part / total:.0%})"


if __name__ == "__main__":
    sys.exit(main())
