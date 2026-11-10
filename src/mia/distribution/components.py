"""Resolve a previously verified component; incompatible installs fall back to base."""

from __future__ import annotations

import json
from pathlib import Path

from mia.distribution.layout import Layout
from mia.distribution.release import ReleaseError, verified_release
from mia.distribution.service import configuration


def yolo_worker(layout: Layout) -> Path:
    base = layout.worker("yolo")
    try:
        active = json.loads((layout.components / "active.json").read_text(encoding="utf-8"))
        config = configuration(layout)
        for key in ("component", "version"):
            if not isinstance(active[key], str) or Path(active[key]).name != active[key]:
                return base
        folder = layout.components / config["platform"] / active["component"] / active["version"]
        if not folder.resolve().is_relative_to(layout.components.resolve()):
            return base
        release = verified_release(
            (folder / "release.json").read_bytes(), config["public_key"], config["platform"]
        )
        if (
            release.kind != "component"
            or release.app_version != config["version"]
            or release.version != active["version"]
            or release.component != active["component"]
        ):
            return base
        candidate = folder / base.name
        return candidate if candidate.is_file() and not candidate.is_symlink() else base
    except OSError, KeyError, TypeError, ValueError, ReleaseError:
        return base
