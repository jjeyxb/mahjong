"""Evaluate same-frame fusion on stored YOLO observations and original images.

    .venv/bin/python tools/replay_hybrid.py <session> --observations <ndjson> --out <dir>

No packets are read by this replay. Audit files preserve raw versus fused labels.
--unlimited-time removes the live time budget for deterministic comparisons;
the per-frame count budget still applies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from mia.engine.multiplex import EngineGroup
from mia.hybrid.fusion import Fusion, TemplateReader, annotate, guard_stream
from mia.live.advice import StreamFeeder
from mia.yolo.stream import LedgerStream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--unlimited-time", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.session / "manifest.json").read_text())
    cv2.setNumThreads(1)
    sys.path.insert(0, str(ROOT / "engines/yolo"))
    from worker import Session

    session = Session()
    adapter = LedgerStream()
    fusion = Fusion(
        TemplateReader("mjpface_default"), budget_ms=float("inf") if args.unlimited_time else 50.0
    )
    group = feeder = None
    if args.engine:
        from mia.engine.mortal import mortal_engine

        group = EngineGroup([mortal_engine(ROOT / "models/mortal_298k.pth", seat=0)])
        group.start()
        feeder = StreamFeeder(group)
    last_image = None
    image = None
    previous = None
    hand_id = None
    versions = errors = doubts = 0
    counts = Counter()
    times = []
    started = time.monotonic()
    try:
        with (
            args.observations.open() as input_file,
            (args.out / "vision.jsonl").open("w") as log,
            (args.out / "fusion.jsonl").open("w") as audit,
        ):
            for i, line in enumerate(input_file, 1):
                if args.limit and i > args.limit:
                    i -= 1
                    break
                obs = json.loads(line)
                record = manifest["frames"][obs["frame_index"] - 1]
                if record["image"] != last_image:
                    raw = cv2.imread(str(args.session / "frames" / record["image"]))
                    x, y, w, h = manifest["table_rect"]
                    image = cv2.resize(
                        raw[y : y + h, x : x + w], (1920, 1080), interpolation=cv2.INTER_AREA
                    )
                    last_image = record["image"]
                fused = fusion.apply(image, obs)
                with redirect_stdout(sys.stderr):
                    payload = session.feed(fused)
                annotate(payload, fused["fusion"])
                if payload["hand_id"] != hand_id:
                    fusion.reset()
                    hand_id = payload["hand_id"]
                stream = guard_stream(adapter.update(payload), payload)
                wall = manifest["first_frame_wall"] + obs["t_ms"] / 1000
                counts.update(fused["fusion"]["counts"])
                times.append(fused["fusion"]["elapsed_ms"])
                changes = []
                for kind in ("p0_hand", "river", "meld"):
                    parts = (
                        [fused[kind]]
                        if kind == "p0_hand" and fused.get(kind)
                        else list((fused.get(kind) or {}).values())
                        if kind != "p0_hand"
                        else []
                    )
                    for part in parts:
                        changes += [
                            dict(kind=kind, **d) for d in part.get("tiles", []) if d.get("raw_yolo")
                        ]
                if changes or fused["fusion"]["pending"]:
                    audit.write(
                        json.dumps(
                            dict(
                                frame=obs["frame_index"],
                                wall=wall,
                                changes=changes,
                                pending=fused["fusion"]["pending"],
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                current = (adapter.kyoku, stream)
                if current != previous:
                    log.write(
                        json.dumps(
                            dict(
                                kind="stream",
                                wall=wall,
                                kyoku=adapter.kyoku,
                                events=[e.to_dict() for e in stream.events],
                                note=stream.doubt or "",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    versions += 1
                    doubts += bool(stream.doubt)
                    if feeder and not stream.doubt and stream.events:
                        result = feeder.update(adapter.kyoku, stream.events)
                        errors += len(result.failures) if result else 0
                    previous = current
                if i % 250 == 0:
                    print(
                        f"{i} frames, {time.monotonic() - started:.1f}s, "
                        f"changes={counts['corrected']}, conflicts={counts['disagree']}",
                        flush=True,
                    )
    finally:
        if group:
            group.close()
    stats = dict(
        frames=i,
        versions=versions,
        engine_errors=errors,
        doubt_versions=doubts,
        fusion_counts=dict(counts),
        fusion_ms=times,
        elapsed_seconds=time.monotonic() - started,
        unlimited_time=args.unlimited_time,
    )
    (args.out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "fusion_ms"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
