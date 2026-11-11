# Optional YOLO inference environment

Run `.venv/bin/python tools/setup_yolo.py --python python3.12` from the repository root.
On Windows with an AMD GPU, add `--gpu rocm` to use AMD's ROCm build of torch
(pinned with hashes in `requirements-rocm.txt`, which must stay UTF-8 with BOM).
See [docs/yolo.md](../../docs/yolo.md) for UI selection, protocol, replay, and validation.

`agent/` contains the inference/state-machine portion of the team-provided
`majongsoul ai agent_mac/app` package. Original source and asset hashes are recorded
in `upstream.json`. The copies preserve upstream implementation; MIA's protocol and
causal/revision adapter live separately in `worker.py` and `src/mia/yolo/`.
Upstream capture, its Mortal checkout, launcher and UI entry points are not included.
The inference wrappers require four `.pt` weights and the accompanying crop/inference
JSON files in `models/yolo/`, which are local assets and are not committed. They are
published at [jjeyxb/mia-yolo](https://huggingface.co/jjeyxb/mia-yolo); fetch them with
`tools/fetch_models.py` (hash-checked) before running `setup_yolo.py`.

The runtime uses torch and Ultralytics in a separate environment. MIA remains on
Python 3.14 without those dependencies. `worker.py` uses only the upstream visual
pipeline and exports its mutable ledger; upstream MortalGuard is not used.

## Local changes to `agent/`

`upstream.json` records the hashes of the files as received. The following files
were changed afterwards to fix failures found by replaying recordings against
packet ground truth (2026-11-06); each change carries a dated `★` comment at the site:

| File | Change | Recording that failed |
|---|---|---|
| `live_v7/ui/scores.py` | A score of exactly 0 (one digit) is accepted | 20261024 E3-1: a seat at 0 points, `start_kyoku` never confirmed |
| `dora_reader.py` | A face-up slot unread at 2× is retried at 1.6×/2.2×/2.5× | selfplay-170335 E1-1: 8p indicator read at 0.147, hand never confirmed |
| `live_v7/pipeline.py` | Cancelled events are ignored when rewinding the starting hand | human-20261103 E1: one cancelled draw blocked all rewinds |
| `live_v7/pipeline.py` | An own draw left unnamed is filled from a later stable hand | 20261024 E2: one misread after-hand left the draw pending |
| `live_v7/fsm.py` | A discard cancelled as "in flight" that later confirms hands its face to the successor | selfplay-170335 E1: riichi tile W lost, a full round misaligned |
| `live_v7/river_adapter.py` | Flat boxes clipped by the bottom of P0's river zone are ignored | human-20261103 E3: a phantom blocked the next discard's slot |
| `live_v7/pipeline.py` | Wall count (read by MIA, `obs["hud"]`) is reconciled against draws; an orphan kan dora plus one extra draw infers a self kan | human-20261103 E3: P3's closed kan of 8p was never seen |
| `live_v7/ui/scores.py` | Minus sign detected (alone or touching the first digit), negative scores read | selfplay-20261106 ×3: no-bust room, every hand after someone went negative was lost (24 hands) |
| `live_v7/ui/panel.py`, `ui/templates_v7/score*x__*.png` | Second score template per seat and digit, picked from packet-verified digits; old ones kept | selfplay-20261106: P0's "8" read as "6" half the time. On five other recordings misreads went 13 → 1 |
| `live_v7/ui/panel.py`, `ui/templates_v7/honbax__{5,6,7}.png` | Honba "5", "6", "7" templates | selfplay-20261106-113550: three 5-honba hands lost; selfplay-20261102-053001 / 155238: 6 and 7 read at 0.58 / 0.52 |
| `live_v7/pipeline.py` | A missed own draw (also the INFERRED kind created when only the discard was seen) is named by voting: hands before it are moved forward and hands after it moved back through known own moves; previously only the first hand after the discard was compared | selfplay-20261106-113550 E1-4, E4-1: one misread in that hand left the draw unnamed and the rest of the hand unsent. Ten recordings: discards 5873 → 6139 / 6262, same engine advice 1233 → 1286 |
| `live_v7/turn_signals.py` | A P0 `drawn_tile` box counts only when it sits right of the concealed row with a gap (checked against the last hand reading); a raised tile inside the row is the mouse hovering, not the drawn tile | selfplay-20261106-103105 E3-1: after cutting the rightmost tile the cursor stayed there, the tile filling the gap was raised and taken out of the concealed hand → 12-tile hands before and after looked "unchanged" → hand-cut judged tsumogiri and the river tile became the draw. Ten recordings: P0 cut judgement 1395/171 → 1551/21 right/wrong, same engine advice 1286 → 1426, engine errors 107 → 28 |
| `live_v7/pipeline.py` | On a new hand the previous panel is remembered only if that hand had a confirmed discard, call, win or draw | 20261024-123746: the recording began at the E1 deal; the frame-1 hand read "E1/0", the deal then started another hand whose "E1/0" was rejected as the old panel → every hand labelled one round late |
| `live_v7/river_adapter.py` | A new slot at the place of a just-claimed tile, with the same tile, within `GONE_FRAMES` is that tile still on screen, not a discard (the position stays released for the seat's next discard, and that lingering slot is archived the first frame it is not seen) | selfplay-20261106-110134 E1 f1464: P0 chi of 8s, the 8s stayed until f1469 → a second P3 discard → the call was "inferred" again at P0's discard. Kept alive, the lingering slot swallowed the next discard on that spot (selfplay-20261102-155238 E4-4) |
| `live_v7/pipeline.py` | Frames are ignored entirely until the table has appeared once (a wall count read, or a full P0 hand seen); callers without `hud` are unaffected | selfplay-20261106-103105 / 110134 / 113550: the loading screen before E1 read as rows of 白 in every river; those fake discards held DahaiGate positions and blocked P1's real discards for four turns. selfplay-20261102-170335: a fake P1 draw on the loading screen left the dealer's first discard without its inferred 14th tile. With these three changes and the hover fix, ten recordings: same engine advice 1426 → 1455, own draws 1469 → 1498 / 1504, discards 6204 → 6249 / 6262, engine errors 28 → 19 |
| `live_v7/fsm.py` | P0 discard: the hand diff overrides the river only when its confidence is not below the confirmed river read (either arrival order); a tsumogiri verdict's "diff" tile is the drawn-tile name, not hand evidence | ten recordings had four river/hand disagreements: hand 0.97 > river 0.88 was the hand; hand 0.81 < 0.96 and 0.62 < 0.92 were the river (selfplay-20261106-113550 E3-0, 20261024-123746 E1-0); selfplay-20261102-053001 W4-0 and 113550 S2-1 lost a correct river read to a misread drawn name. Same engine advice 1455 → 1464, engine errors 19 → 14 |
| `live_v7/kyoku_info.py` | Honba 8+ (no template) is inferred as previous + 1 when unreadable; the reader adds a 24x12 glyph signature to weak honba reads so the previous hand's lingering panel is not taken for the next number | no recording reaches 8 honba |
| `live_v7/pipeline.py` | After a hand ends, river readings are ignored while the wall count is unreadable | selfplay-20261106-110134 S4-1 → W1: the win screen's revealed hand became a phantom hand that took W1's start info |
| `dora_reader.py` | A slot counts as a tile back at saturation ≥ 60 instead of ≥ 150. Measured backs: web Mac 181, web Windows 212, Steam (Windows, pale yellow) 115; face-up tiles 3–6 | Steam 2560x1440, 2026-11-11: the four face-down indicator slots were read as `P`, and four fake `dora P` events were confirmed in the first hand |
| `live_v7/kyoku_info.py` | Honba is not exported before the round label is known | same: the old panel's honba 1 was frozen into W1-0 before the stale-panel check could reject it |

MIA reads the centre panel's wall count and turn light with its own template reader
(`YoloClient.hud`) and adds them to every observation as `hud: {wall, turn}` before
`track`. The upstream pipeline does not read them itself.
