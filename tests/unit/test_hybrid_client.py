import sys

import numpy as np
import pytest

from mia.engine.base import EngineError
from mia.hybrid.client import HybridClient

CHILD = """
import json,sys,struct
print(json.dumps(dict(ready=True,protocol=1,device="stub",observations=True)),flush=True)
def read(n):
    out=bytearray()
    while len(out)<n:
        part=sys.stdin.buffer.read(n-len(out))
        if not part:raise EOFError()
        out.extend(part)
    return out
frames=tracks=0
while True:
    prefix=sys.stdin.buffer.read(4)
    if not prefix:break
    header=json.loads(read(struct.unpack("!I",prefix)[0]))
    if header.get("kind")=="observation":
        obs=json.loads(read(header['size']))
        tracks+=1
        reply=dict(hand_id="H001",hand={},events=[],received=obs,frames=frames,tracks=tracks)
    else:
        read(1080*1920*3)
        frames+=1
        assert header['raw_observations'] is True
        reply=dict(frame_index=frames,t_ms=header['wall']*1000,p0_hand=None)
    print(json.dumps(reply),flush=True)
"""


def test_raw_frame_is_fused_before_exactly_one_tracking_request(tmp_path, monkeypatch):
    import mia.hybrid.client as module

    class Fusion:
        def __init__(self, reader):
            pass

        def reset(self):
            pass

        def apply(self, _image, obs):
            return {**obs, "fused": True, "fusion": dict(counts={}, pending=[])}

    monkeypatch.setenv("MIA_YOLO_PYTHON", sys.executable)
    monkeypatch.setattr(module, "TemplateReader", lambda _skin: None)
    monkeypatch.setattr(module, "Fusion", Fusion)
    (tmp_path / "worker.py").write_text(CHILD)
    client = HybridClient()
    monkeypatch.setattr(client, "_base", tmp_path)
    try:
        client.start()
        reply = client.feed(np.zeros((1080, 1920, 3), dtype=np.uint8), 123.5)
        assert reply["frames"] == reply["tracks"] == 1
        assert reply["received"]["fused"]
        assert reply["received"]["t_ms"] == 123500
        assert "torch" not in sys.modules
    finally:
        client.close()


def test_old_inference_protocol_fails_clearly_in_hybrid_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MIA_YOLO_PYTHON", sys.executable)
    (tmp_path / "worker.py").write_text(
        'import json;print(json.dumps(dict(ready=True,protocol=1,device="stub")),flush=True)\n'
    )
    client = HybridClient()
    monkeypatch.setattr(client, "_base", tmp_path)
    try:
        with pytest.raises(EngineError, match="不支援混合"):
            client.start()
    finally:
        client.close()
