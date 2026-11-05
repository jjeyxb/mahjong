# Optional YOLO inference environment

Run `.venv/bin/python tools/setup_yolo.py --python python3.12` from the repository root.
See [docs/yolo.md](../../docs/yolo.md) for UI selection, protocol, replay, and validation.

`agent/` contains the inference/state-machine portion of the team-provided
`majongsoul ai agent_mac/app` package. Original source and asset hashes are recorded
in `upstream.json`. The copies preserve upstream implementation; MIA's protocol and
causal/revision adapter live separately in `worker.py` and `src/mia/yolo/`.
Upstream capture, its Mortal checkout, launcher and UI entry points are not included.
The inference wrappers require four `.pt` weights and the accompanying crop/inference
JSON files in `models/yolo/`, which are local assets and are not committed.

The runtime uses torch and Ultralytics in a separate environment. MIA remains on
Python 3.14 without those dependencies. `worker.py` uses only the upstream visual
pipeline and exports its mutable ledger; upstream MortalGuard is not used.
