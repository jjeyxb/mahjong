import sys
from pathlib import Path

import numpy as np
import pytest

from mia.engine.base import EngineError
from mia.yolo.client import YoloClient

CHILD = """
import json,sys,struct
print(json.dumps(dict(ready=True,protocol=1,device="stub")),flush=True)
def read(n):
    data=bytearray()
    while len(data)<n:
        part=sys.stdin.buffer.read(n-len(data))
        if not part: raise EOFError()
        data.extend(part)
    return data
while True:
    p=sys.stdin.buffer.read(4)
    if not p: break
    h=json.loads(read(struct.unpack("!I",p)[0]))
    pixels=read(1080*1920*3)
    print(json.dumps(dict(shape=h["shape"],wall=h["wall"],pixel=pixels[0])),flush=True)
"""


def test_real_pipe_roundtrip_and_shutdown_without_torch(tmp_path, monkeypatch):
    (tmp_path / "worker.py").write_text(CHILD)
    client = YoloClient(python=Path(sys.executable), timeout=3)
    monkeypatch.setattr(client, "_base", tmp_path)
    try:
        client.start()
        assert client.device == "stub"
        reply = client.feed(np.full((1080, 1920, 3), 37, dtype=np.uint8), 123.5)
        assert reply == dict(shape=[1080, 1920, 3], wall=123.5, pixel=37)
        assert "torch" not in sys.modules
    finally:
        client.close()
        client.close()


def test_child_exit_during_start_reports_failure_and_can_be_closed(tmp_path, monkeypatch):
    (tmp_path / "worker.py").write_text("raise SystemExit(1)\n")
    client = YoloClient(python=Path(sys.executable), timeout=1)
    monkeypatch.setattr(client, "_base", tmp_path)
    try:
        with pytest.raises(EngineError, match="已結束"):
            client.start()
    finally:
        client.close()


def test_missing_optional_environment_is_a_clear_error(tmp_path):
    with pytest.raises(EngineError, match="setup_yolo"):
        YoloClient(python=tmp_path / "missing")
