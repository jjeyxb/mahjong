#!/usr/bin/env python
"""開側邊視窗。

三種餵資料的方式:

``--replay <ws.jsonl>``
    把錄下的對局照時間重播 —— 引擎真的在跑,建議是真的算出來的。**不需要開遊戲**,
    所以 UI 隨時可以驗、可以截圖給論文用。這是預設。

``--session <錄影目錄>``
    把錄下的畫面跑一次 CV,顯示辨識出來的手牌與向聽。驗的是功能 1 那條路。

``--demo``
    塞一組寫死的假資料就停住。純粹用來看版面,不需要任何素材。

用法::

    python tools/ui.py --demo
    python tools/ui.py --replay tests/fixtures/real_game_full.jsonl --speed 8
    python tools/ui.py --replay data/gt/ws.jsonl --mortal models/mortal_298k.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from majsoul_copilot.engine import AIEngine, DummyEngine, EngineGroup
from majsoul_copilot.mjai import MjaiEvent
from majsoul_copilot.mjai.handstate import HandTracker
from majsoul_copilot.ui.panel.window import PanelWindow, present
from majsoul_copilot.ui.viewmodel import ViewModel
from majsoul_copilot.utils.logging import setup_logging

DEMO_HAND = ("1m", "1m", "2m", "3s", "4m", "5pr", "6m", "8p", "8s", "8s", "9m", "E", "P")


def build_engines(args: argparse.Namespace) -> list[AIEngine]:
    engines: list[AIEngine] = [DummyEngine()]
    for weights in args.mortal or []:
        from majsoul_copilot.engine.mortal import mortal_engine

        engines.append(mortal_engine(weights, seat=args.seat, name=Path(weights).stem))
    return engines


def load_events(path: Path) -> list[MjaiEvent]:
    """讀事件流。與 ``tools/advise.py`` 同一套,兩種格式都吃。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from advise import load_events as _load

    events, _ = _load(path)
    return events


def run_replay(args: argparse.Namespace, model: ViewModel) -> None:
    """照事件順序重播,每 ``interval`` 毫秒一步。

    用 QTimer 而不是背景執行緒:引擎的 react 是同步的、單次 10~20 ms,直接在
    主執行緒跑不會讓 UI 卡住,而且完全避開「Qt widget 只能在主執行緒動」
    這個限制。真正接上遊戲時才需要執行緒,那是另一件事。
    """
    events = load_events(args.replay)
    if not events:
        raise SystemExit(f"{args.replay} 裡沒有事件")

    seat = next((int(getattr(e, "id", 0)) for e in events if e.TYPE == "start_game"), 0)
    args.seat = seat
    tracker = HandTracker(_name="ui")
    group = EngineGroup(build_engines(args))
    group.start()
    model.set_notices([f"重播 {args.replay.name} — 自己坐 {seat}"])

    step = iter(range(len(events)))

    def advance() -> None:
        try:
            index = next(step)
        except StopIteration:
            model.set_notices([f"重播結束({len(events)} 個事件)"])
            group.close()
            timer.stop()
            return

        event = events[index]
        tracker.feed(event)
        model.update_packet_hand(tracker.tiles, drawn=tracker.drawn)
        result = group.react(event)
        # 只在有人真的要動作時換掉建議 —— 每一步都覆蓋會讓畫面在「切 3s」與
        # 「不需要動作」之間閃,而後者佔了九成的事件
        if result.actions:
            model.update_advices(result.advices)

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(max(1, round(1000 / args.speed)))
    # 掛在 model 上避免被 GC —— QTimer 沒有 parent 時會被當成暫時物件回收
    model._timer = timer  # type: ignore[attr-defined]  # noqa: SLF001


def run_session(args: argparse.Namespace, model: ViewModel) -> None:
    """把錄下的畫面跑一次 CV。驗的是功能 1。"""
    from majsoul_copilot.calibration.table import Calibration
    from majsoul_copilot.config.loader import load_config
    from majsoul_copilot.recorder import SessionReader
    from majsoul_copilot.utils.geometry import Size
    from majsoul_copilot.vision.roi import RoiSet
    from majsoul_copilot.vision.tiles.classify import TemplateSet, classify_hand
    from majsoul_copilot.vision.tiles.hand import read_hand

    session = SessionReader(args.session)
    if session.table_rect is None:
        raise SystemExit("session 沒有 table_rect —— 錄製時校正沒鎖定")

    records = [r for r in session.manifest.frames if r.changed]
    if not records:
        raise SystemExit("session 裡沒有有變化的幀")

    first = session.load_image(records[0])
    height, width = first.shape[:2]
    config = load_config()
    rois = RoiSet(
        config.roi,
        Calibration(session.table_rect, Size(width, height), source="manual"),
    )
    box = rois["own_hand"].pixels
    templates = TemplateSet.load()
    model.set_notices([f"重播畫面 {args.session.name} — {len(records)} 幀"])

    step = iter(range(len(records)))

    def advance() -> None:
        try:
            index = next(step)
        except StopIteration:
            model.set_notices([f"重播結束({len(records)} 幀)"])
            timer.stop()
            return

        roi = session.load_image(records[index])[box.as_slice()]
        hand = read_hand(roi)
        concealed, drawn = classify_hand(roi, hand, templates)
        model.update_cv_hand(
            [m.label for m in concealed],
            drawn=drawn.label if drawn else None,
            confident=all(m.is_confident for m in (*concealed, *(x for x in [drawn] if x))),
        )

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(max(1, round(1000 / args.speed)))
    model._timer = timer  # type: ignore[attr-defined]  # noqa: SLF001


def run_demo(model: ViewModel) -> None:
    """一組寫死的資料。只為了看版面,不代表任何真實局面。"""
    from majsoul_copilot.engine.base import Advice
    from majsoul_copilot.mjai import Dahai

    model.update_packet_hand((*DEMO_HAND, "F"), drawn="F")
    # mask_bits 的位元位置要真的對應那幾張牌,q_values 也要照索引升冪 ——
    # 隨便填的話畫面上會出現「建議切 E」但候選清單裡根本沒有 E,那種示範
    # 比沒有示範更糟。9m=8、8p=16、3s=20、E=27,最高分落在 E。
    model.update_advices(
        [
            Advice(
                "mortal_298k",
                Dahai(actor=0, pai="E", tsumogiri=False),
                {
                    "mask_bits": (1 << 8) | (1 << 16) | (1 << 20) | (1 << 27),
                    "q_values": [0.94, 0.31, -0.55, 1.28],
                },
                14.0,
            ),
            Advice("baseline", Dahai(actor=0, pai="9m", tsumogiri=False), None, 0.4),
        ]
    )
    model.set_notices(["示範資料 —— 不是真實局面"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", type=Path, help="重播封包錄影(引擎真的在跑)")
    source.add_argument("--session", type=Path, help="重播畫面錄影(跑 CV)")
    source.add_argument("--demo", action="store_true", help="塞假資料看版面")
    parser.add_argument("--mortal", action="append", metavar="WEIGHTS", help="加一個 Mortal 引擎")
    parser.add_argument("--speed", type=float, default=4.0, help="重播速度(步/秒),預設 4")
    parser.add_argument("--skin", default=None, help="牌面素材皮膚")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    args.seat = 0

    setup_logging(level=args.log_level)
    app = QApplication(sys.argv[:1])

    model = ViewModel()
    window = PanelWindow(model, **({"skin": args.skin} if args.skin else {}))
    model.subscribe(window.apply)

    if args.replay:
        run_replay(args, model)
    elif args.session:
        run_session(args, model)
    else:
        run_demo(model)

    present(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
