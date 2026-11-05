"""Replay recorded frames or cached YOLO observations into the MIA adapter.

    .venv/bin/python tools/replay_yolo.py <session> --observations <ndjson> --out <dir>
    .venv/bin/python tools/replay_yolo.py <session> --out <dir>

Packets are only read after replay, by the evaluator. --observations preserves
all original model/FSM ticks and needs no torch in the main environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from mia.engine.multiplex import EngineGroup
from mia.live.advice import StreamFeeder
from mia.yolo.client import YoloClient
from mia.yolo.stream import LedgerStream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--engine", action="store_true", help="Also replay accepted snapshots into Mortal"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.session / "manifest.json").read_text())
    first = manifest["first_frame_wall"]
    sys.path.insert(0, str(ROOT / "engines/yolo"))
    from worker import Session

    adapter = LedgerStream()
    session = Session() if args.observations else None
    client = None if args.observations else YoloClient()
    group = None
    feeder = None
    if args.engine:
        from mia.engine.mortal import mortal_engine

        group = EngineGroup([mortal_engine(ROOT / "models/mortal_298k.pth", seat=0)])
        group.start()
        feeder = StreamFeeder(group)
    started = time.monotonic()
    previous = None
    versions = doubts = failures = 0
    rejected = set()
    try:
        if client:
            client.start()
        if args.observations:
            records = (json.loads(line) for line in args.observations.open() if line.strip())
        else:
            records = iter(manifest["frames"])
        with (args.out / "vision.jsonl").open("w") as log:
            for i, record in enumerate(records, 1):
                if session:
                    with redirect_stdout(sys.stderr):
                        payload = session.feed(record)
                    wall = first + record["t_ms"] / 1000
                else:
                    assert client is not None
                    image = cv2.imread(str(args.session / "frames" / record["image"]))
                    x, y, w, h = manifest["table_rect"]
                    frame = cv2.resize(
                        image[y : y + h, x : x + w], (1920, 1080), interpolation=cv2.INTER_AREA
                    )
                    wall = first + record["timestamp"]
                    payload = client.feed(frame, wall)
                stream = adapter.update(payload)
                rejected.update(adapter.rejected)
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
                        failures += len(result.failures) if result else 0
                    previous = current
                if i % 1000 == 0:
                    print(
                        f"{i} frames, {versions} snapshots, {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    finally:
        if client:
            client.close()
        if group:
            group.close()
    stats = dict(
        frames=i,
        versions=versions,
        doubt_versions=doubts,
        engine_failures=failures,
        rejected_event_ids=sorted(rejected),
        elapsed_seconds=time.monotonic() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
