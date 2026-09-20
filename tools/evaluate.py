#!/usr/bin/env python
"""量手牌 CV 的準確率:拿同時錄下的封包當標準答案。

需要**一組成對的錄影**:``tools/record.py`` 錄的畫面 session,加上同一段時間
``tools/gt.py`` 錄的封包。兩者靠牆上時鐘對齊,所以必須是同一次遊玩期間錄的。

用法::

    # 兩個終端機同時開錄(順序無所謂,重疊的那段才算數)
    .venv/bin/python tools/gt.py cdp --out data/gt/ws.jsonl
    .venv/bin/python tools/record.py --notes "評測用"

    # 錄完之後
    .venv/bin/python tools/evaluate.py data/recordings/20260729-201530 data/gt/ws.jsonl

先跑 ``--dry-run`` 確認兩份檔案真的對得上 —— 那一步不跑 CV,幾秒就有結果,
可以在還登入著的時候就發現「其實根本沒重疊」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.calibration.table import Calibration
from mia.config.loader import load_config
from mia.eval import align, build_timeline, evaluate_frame, summarize
from mia.eval.align import SETTLE
from mia.recorder import SessionReader
from mia.utils.geometry import Rect, Size
from mia.utils.logging import setup_logging
from mia.vision.roi import RoiSet
from mia.vision.tiles.classify import HOVER_HEADROOM, TemplateSet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path, help="tools/record.py 錄的 session 目錄")
    parser.add_argument("dump", type=Path, help="tools/gt.py 錄的 WebSocket .jsonl")
    parser.add_argument("--skin", default=None, help="牌面模板皮膚,預設用內建的")
    parser.add_argument(
        "--settle",
        type=float,
        default=None,
        help="動畫穩定秒數。加大會變嚴格(可評分的幀變少但更乾淨)",
    )
    parser.add_argument("--dry-run", action="store_true", help="只檢查對得上對不上,不跑 CV")
    parser.add_argument("--limit", type=int, default=0, help="最多評幾幀,0 表示全部")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)

    try:
        session = SessionReader(args.session)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"畫面 {args.session}:{len(session.manifest.frames)} 幀")
    timeline = build_timeline(args.dump)
    print(f"封包 {args.dump}:{len(timeline)} 個手牌狀態,自己坐 {timeline.seat}")

    if not _overlaps(session, timeline):
        return 1

    # 掃描值涵蓋 SETTLE 兩側,才看得出曲線在哪裡平掉(見 align.SETTLE 的量測)
    settle_values = [args.settle] if args.settle is not None else [0.4, 0.6, SETTLE, 1.3, 2.0]
    if args.dry_run:
        print("\n穩定秒數 → 可評分的幀數")
        for settle in settle_values:
            paired = list(align(session, timeline, settle=settle))
            print(f"  {settle:.1f}s  {len(paired)} 幀")
        print("\n沒有 0 的話兩份錄影就對得上,可以拿掉 --dry-run 跑完整評測。")
        return 0

    # 預設一律取 align.SETTLE —— 這裡另外寫一個 0.4 的話,改了那個常數也不會生效,
    # 而報告上仍然印著「穩定秒數 0.4s」,看起來完全正常
    settle = args.settle if args.settle is not None else SETTLE
    paired = list(align(session, timeline, settle=settle))
    if args.limit:
        paired = paired[: args.limit]
    if not paired:
        print("\n配不到任何幀。用 --dry-run 看看是不是穩定秒數設得太嚴。")
        return 1

    print(f"\n穩定秒數 {settle:.1f}s,配到 {len(paired)} 幀,開始辨識…")
    templates = TemplateSet.load(args.skin) if args.skin else TemplateSet.load()
    roi = _own_hand_roi(session)

    # 與即時管線同一個 headroom,否則這份報告量的不是真的會跑的那條路
    headroom = round(roi.height * HOVER_HEADROOM)
    top = max(0, roi.y - headroom)
    headroom = roi.y - top
    print(f"牌框上方多留 {headroom}px 給滑鼠抬起來的那張牌")

    by_index = {r.index: r for r in session.manifest.frames}
    results = []
    for frame in paired:
        image = session.load_image(by_index[frame.index])
        tall = image[top : roi.y + roi.height, roi.x : roi.x + roi.width]
        results.append(evaluate_frame(tall, frame, templates, headroom=headroom))

    report = summarize(results)
    print()
    print(report.summary())
    return 0


def _overlaps(session: SessionReader, timeline) -> bool:  # type: ignore[no-untyped-def]
    """兩份錄影的時間有沒有交集。沒有的話後面全都白做。"""
    if not session.can_align:
        print("\n這個 session 沒有 first_frame_wall(2026-07-27 之前的舊格式),無法自動對齊。")
        return False
    if not len(timeline):
        print("\n封包錄影裡沒有任何手牌狀態 —— 可能只錄到大廳,對局中再錄一次。")
        return False

    frames = session.manifest.frames
    video = (session.wall_at(frames[0].timestamp), session.wall_at(frames[-1].timestamp))
    packets = timeline.span
    overlap = min(video[1], packets[1]) - max(video[0], packets[0])
    if overlap <= 0:
        print(
            f"\n兩份錄影**沒有重疊**(差 {-overlap:.0f} 秒)—— 不是同一次遊玩期間錄的,"
            "或其中一邊在對局開始前就停了。"
        )
        return False
    print(f"重疊 {overlap:.0f} 秒")
    return True


def _own_hand_roi(session: SessionReader) -> Rect:
    """算出 own_hand 在畫面上的像素框。

    回傳型別要寫出來:原本這裡掛著 ``type: ignore[no-untyped-def]``,於是回傳
    值是 ``Any``,``.pixels``(根本不存在的屬性)就一路過了 mypy,直到真的錄到
    素材、跑到這一步才在使用者面前炸開。沒有型別的地方就是沒有人在看的地方。

    直接用 session 存下來的 ``table_rect`` —— 那是錄製當下 StableCalibrator
    蒐集多幀之後鎖定的結果。在這裡重跑一次校正不但慢,還可能因為挑到不同的
    幀而得到不同的答案,那樣準確率報告就不再對應當初錄下的東西了。
    """
    table_rect = session.table_rect
    if table_rect is None:
        raise SystemExit(
            "session 沒有 table_rect —— 錄製時校正沒鎖定。"
            "通常是錄太短(StableCalibrator 要蒐集數幀)或畫面一直停在載入中。"
        )

    frames = session.manifest.frames
    if not frames:
        raise SystemExit("session 裡沒有任何幀")

    config = load_config()
    image = session.load_image(frames[0])
    height, width = image.shape[:2]
    calibration = Calibration(
        table_rect=table_rect,
        image_size=Size(width, height),
        source="manual",  # 不是這次算的,是錄製時就定好的
    )

    rois = RoiSet(config.roi, calibration)
    if "own_hand" not in rois:
        raise SystemExit("設定檔沒有 own_hand 這個 ROI —— 見 config/default.yaml")
    return rois["own_hand"].rect


if __name__ == "__main__":
    sys.exit(main())
