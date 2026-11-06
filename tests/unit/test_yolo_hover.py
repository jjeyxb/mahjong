"""YOLO 代理:滑鼠停在暗牌上浮起來的那張,不是摸入牌(live_v7/turn_signals.py)。

情境照錄影 selfplay-20261106-103105 東3局1本場:打出最右邊的牌之後滑鼠還停在那裡,
補進來的那張浮起來(y 989 -> 944),局勢模型給它 drawn_tile 框。舊程式把它當摸入牌扣掉,
暗牌只剩 12 張,打牌前後一比「沒變」-> 判成摸切。
另外:錄影一開始那一局什麼都沒發生就換局,它讀到的局數不可以拿來擋下一局(pipeline.py)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "engines" / "yolo" / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "live_v7"), str(AGENT / "live_v7" / "ui")]
turn_signals = pytest.importorskip("turn_signals")
pipeline = pytest.importorskip("pipeline")

HAND = ["1m", "3m", "6m", "7m", "8m", "1p", "3p", "5p", "6p", "8p", "9p", "3s", "5s"]
ROW_Y = 989
HOVER_Y = 944


def _tile(pai, x, y=ROW_Y):
    return {"pai": pai, "conf": 0.95, "box": (x - 45, y - 60, x + 45, y + 60), "center": (x, y)}


def _drawn(x, y, conf):
    return {"cls": "drawn_tile", "conf": conf, "box": (x - 45, y - 60, x + 45, y + 60),
            "center": (x, y)}


def _row(tiles, hover=None):
    """暗牌一排,牌距 94;hover = 浮起來的那一張的序號。"""
    return [_tile(p, 268 + 94.5 * i, HOVER_Y if i == hover else ROW_Y) for i, p in enumerate(tiles)]


def _feed(sig, frames, region, tiles):
    out = []
    for f in frames:
        out += sig.feed(f, f * 66.0, region, {"tiles": tiles})
    return out


def test_a_raised_tile_inside_the_row_is_not_the_drawn_tile():
    sig = turn_signals.TurnSignals()
    _feed(sig, range(1, 4), [], _row(HAND))
    hover = _drawn(268 + 94.5 * 12, HOVER_Y, 0.15)  # 滑鼠停在最右邊那張
    out = _feed(sig, range(4, 10), [hover], _row(HAND, hover=12))
    assert sig.p0_drawn is None
    assert not [d for d in out if d.seat == "P0"]
    assert len(sig.p0_stable[-1][2]) == 13


def test_the_real_drawn_tile_apart_from_the_row_is_still_taken_out():
    sig = turn_signals.TurnSignals()
    _feed(sig, range(1, 4), [], _row(HAND))
    drawn = _drawn(1531, ROW_Y, 0.85)
    out = _feed(sig, range(4, 10), [drawn], [*_row(HAND), _tile("E", 1531)])
    assert sig.p0_drawn is not None
    assert [d for d in out if d.seat == "P0" and d.kind == "draw_seen"]
    assert sig.p0_stable[-1][2] == tuple(sorted(HAND))


def _start(p):
    ev = p.ledger.add("start_kyoku", p.hand_id, 0.0, 1, "PENDING", bakaze="E", kyoku=1,
                      honba=0, kyotaku=0, oya=0, scores=[25000] * 4, dora_marker="1m",
                      tehais=[["?"] * 13] * 4)
    assert ev["export_eligible"]
    p.start_eid = ev["event_id"]


def _remembered(p, monkeypatch):
    seen = []
    monkeypatch.setattr(p.kyoku, "reset", lambda remember=True: seen.append(remember))
    p._new_hand(30_000.0, 98)  # noqa: SLF001
    return seen[-1]


def test_a_hand_where_nothing_happened_does_not_block_the_next_panel(monkeypatch):
    p = pipeline.Pipeline("t")
    _start(p)
    # 載入畫面的假牌河:牌面讀不到的待定打牌不算
    p.ledger.add("dahai", p.hand_id, 9_600.0, 37, "PENDING", actor="P1")
    assert _remembered(p, monkeypatch) is False


def test_a_played_hand_still_blocks_its_old_panel(monkeypatch):
    p = pipeline.Pipeline("t")
    _start(p)
    p.ledger.add("dahai", p.hand_id, 5_000.0, 20, "CONFIRMED", actor="P1", pai="E",
                 tsumogiri=True)
    assert _remembered(p, monkeypatch) is True
