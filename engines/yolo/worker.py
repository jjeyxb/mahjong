"""Python 3.12 inference child. No Qt, capture, packets, or Mortal dependencies.

stdin: uint32 JSON header length, header {shape,wall}, then raw uint8 BGR bytes.
stdout: one JSON response per frame; upstream diagnostics go to stderr.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import struct
import sys

BASE = Path(__file__).resolve().parent
AGENT = BASE / "agent"
sys.path.insert(0, str(AGENT / "live_v7"))
import _paths  # noqa: F401


class Session:
    """The same stateful path for live frames and cached-observation replay."""

    def __init__(self):
        from pipeline import Pipeline
        self.pipeline = Pipeline("MIA-YOLO")
        self.hand = {}
        self.previous = None
        self.repeats = 0
        self.hand_id = None

    def feed(self, obs):
        from drawn_lib import apply_hand_fix
        from hand import name_of_drawn, pick_p0_drawn
        self.pipeline.feed(obs)
        if self.hand_id != self.pipeline.hand_id:
            self.hand_id = self.pipeline.hand_id
            self.hand, self.previous, self.repeats = {}, None, 0
        ph = obs.get("p0_hand")
        if ph is not None:
            dets = apply_hand_fix(ph.get("tiles") or [], ph.get("hand_fix"))
            dets = sorted(dets, key=lambda d: d["center"][0])
            tiles = [d["pai"] for d in dets]
            key = tuple(sorted(tiles))
            self.repeats = self.repeats+1 if key == self.previous else 1
            self.previous = key
            confident = bool(dets) and all(d.get("conf", 0) >= 0.7 for d in dets)
            self.hand = dict(tiles=tiles, frame=obs["frame_index"],
                stable=confident and self.repeats >= 2 and len(tiles)%3 in (1,2),
                confident=confident,
                drawn=name_of_drawn(dets,pick_p0_drawn(obs.get("region_detections") or [])))
        # Retain revision metadata and physical identities, not append-only output.
        events = []
        for ev in self.pipeline.ledger.ordered(self.pipeline.hand_id):
            events.append({k:v for k,v in ev.items() if k not in ("history","notes","evidence")})
        return dict(hand_id=self.pipeline.hand_id,events=events,hand=self.hand)


class Detector:
    def __init__(self):
        import cv2
        import torch
        import platform_compat
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        self.device = platform_compat.setup_device()
        from region_model import RegionModel
        from river_model import RiverModel
        from hand_model import HandModel
        from meld_model import MeldModel
        from dora_reader import DoraReader
        from kyoku_info import RawUiReader
        import drawn_lib
        models = Path(os.environ.get("MIA_YOLO_MODELS", BASE.parents[1]/"models/yolo"))
        self.region = RegionModel(str(models/"region_model_best.pt"),default_conf=0.1)
        self.river = RiverModel(str(models/"river_model_best.pt"))
        self.hand = HandModel(str(models/"hand_model_best.pt"),region=self.region)
        self.meld = MeldModel(str(models/"meld_model_v32_1280.pt"))
        self.ui = RawUiReader(str(AGENT/"assets"),DoraReader(model=self.hand.model))
        self.library = drawn_lib.DrawnLibrary(drawn_lib.DEFAULT_LIB)
        self.session = Session()
        self.extra = {f"P{i}":0 for i in range(4)}
        self.n = 0

    def feed(self, frame, wall, raw=False):
        from obs import new_frame, det_region, det_tile
        from river import RIVER_ZONES, grow_zone, needed_growth, sort_by_discard_order
        import drawn_lib
        self.n += 1
        obs = new_frame("MIA-YOLO",self.n,wall*1000)
        rd = self.region.detect(frame)
        obs["region_detections"] = [det_region(d) for d in rd]
        zones = {s:grow_zone(s,RIVER_ZONES[s],self.extra[s]) for s in self.extra}
        rv = self.river.detect(frame,zones=zones)
        obs["river"] = {s:dict(zone=list(zones[s]),inference_ok=True,
            tiles=[det_tile(d) for d in rv if d["seat"]==s]) for s in self.extra}
        for s in self.extra:
            mine = [d for d in rv if d["seat"]==s]
            if not mine:
                self.extra[s] = 0
            else:
                need = needed_growth(s,sort_by_discard_order(s,mine),zones[s],2,10)
                if need:
                    self.extra[s] = min(200,self.extra[s]+need)
        if self.n%2==0:
            hd = self.hand.detect(frame,region_dets=[d for d in rd if d["cls"]=="meld_area" and d["conf"]>=0.25])
            obs["p0_hand"] = dict(tiles=[det_tile(d) for d in hd],includes_drawn=True)
            obs["p0_hand"]["drawn_lib"] = drawn_lib.observe(self.library,frame,obs["p0_hand"]["tiles"],rd)
            obs["p0_hand"]["hand_fix"] = drawn_lib.observe_hand(self.library,frame,obs["p0_hand"]["tiles"],rd)
        if self.n%3==0:
            md = self.meld.detect(frame)
            obs["meld"] = {s:dict(tiles=[det_tile(d) for d in md if d["seat"]==s]) for s in self.extra}
        if self.n%5==0 or self.session.pipeline.start_rush(self.n):
            obs["ui_context"],obs["dora_observation"] = self.ui.read(frame)
        return obs if raw else self.session.feed(obs)


def read_exact(stream,n):
    chunks = bytearray()
    while len(chunks)<n:
        chunk = stream.read(n-len(chunks))
        if not chunk:
            raise EOFError("Truncated YOLO frame")
        chunks.extend(chunk)
    return bytes(chunks)


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR",str(BASE/".cache"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        detector = Detector()
    output.write(json.dumps(dict(ready=True,device=detector.device,protocol=1,observations=True))+"\n")
    output.flush()
    while True:
        prefix = sys.stdin.buffer.read(4)
        if not prefix:
            return
        if len(prefix)!=4:
            raise EOFError("Invalid frame header")
        size = struct.unpack("!I",prefix)[0]
        if size>4096:
            raise ValueError("Frame header too large")
        header = json.loads(read_exact(sys.stdin.buffer,size))
        if header.get("kind") == "observation":
            length = header["size"]
            if not isinstance(length,int) or not 0<length<=2*1024*1024:
                raise ValueError("Invalid observation size")
            observation = json.loads(read_exact(sys.stdin.buffer,length))
            with redirect_stdout(sys.stderr):
                reply = detector.session.feed(observation)
            output.write(json.dumps(reply,ensure_ascii=False,separators=(",",":"))+"\n")
            output.flush()
            continue
        shape = header["shape"]
        if shape != [1080,1920,3]:
            raise ValueError("YOLO input must be 1920x1080 BGR")
        frame = np.frombuffer(read_exact(sys.stdin.buffer,1080*1920*3),dtype=np.uint8).reshape(shape)
        with redirect_stdout(sys.stderr):
            reply = detector.feed(frame,float(header["wall"]),raw=bool(header.get("raw_observations")))
        output.write(json.dumps(reply,ensure_ascii=False,separators=(",",":"))+"\n")
        output.flush()


if __name__=="__main__":
    main()
