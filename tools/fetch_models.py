#!/usr/bin/env python3
"""下載執行需要的模型權重到 ``models/``。新的電腦同步專案之後跑一次就好。

    python tools/fetch_models.py

權重不在 git 裡(太大)。來源:

* ``mortal_298k.pth``:原作者 https://huggingface.co/VoidShine/mortal-298k (AGPL-3.0)。
* 四份風格權重:https://huggingface.co/jjeyxb/mia-mortal-style 。
* YOLO 辨識:https://huggingface.co/jjeyxb/mia-yolo 。

每個檔案的大小與 SHA-256 寫死在下面,下載完逐一核對,對不上就刪掉並報錯:上游換了檔案或
下載壞掉都不會被安靜地用上。已經在、而且雜湊對得上的檔案不重抓。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Weight:
    repo: str
    remote: str
    local: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.remote}"


WEIGHTS = (
    _Weight(
        "VoidShine/mortal-298k",
        "mortal_298k.pth",
        "mortal_298k.pth",
        130774416,
        "bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8",
    ),
    _Weight(
        "jjeyxb/mia-mortal-style",
        "style_call_high.pth",
        "style_call_high.pth",
        130799620,
        "2aaad62d0be379ae891609b67b3ef2495aae08e53a7557b3368490c734fbece9",
    ),
    _Weight(
        "jjeyxb/mia-mortal-style",
        "style_call_low.pth",
        "style_call_low.pth",
        130797731,
        "dfd7586ff00b87b7fa1153d9506ff53038ec49d6d417c3e7f00bc9254a969769",
    ),
    _Weight(
        "jjeyxb/mia-mortal-style",
        "style_reach_high.pth",
        "style_reach_high.pth",
        130801509,
        "bb1ade1c5c0ba916804e006b0d277036c1c118f873a9ac6f00275a751c69a610",
    ),
    _Weight(
        "jjeyxb/mia-mortal-style",
        "style_reach_low.pth",
        "style_reach_low.pth",
        130799620,
        "5253d5afec1cdf3ecb92a17bb69ad35e33db5126febd80dde90183361146e791",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "hand_model_best.pt",
        "yolo/hand_model_best.pt",
        6221667,
        "04a47c872baea0f548e4c473e80d6e685f06e31775ac375f016f79c2cd87ea94",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "meld_model_v32_1280.crop_meta.json",
        "yolo/meld_model_v32_1280.crop_meta.json",
        2961,
        "c0e4752a2045f0529a7fcf38a2db8153a571925e8236248232676c24b762dd87",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "meld_model_v32_1280.infer.json",
        "yolo/meld_model_v32_1280.infer.json",
        19,
        "73f83759cf5a3368ce5a7eda25fe8b6d032764f5720edfd73b0cd475f0039431",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "meld_model_v32_1280.pt",
        "yolo/meld_model_v32_1280.pt",
        5465342,
        "742f0720c64abaa095a66d3b0035d329b05c8dcbbde3f40184721c4c8f6fba38",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "region_model_best.pt",
        "yolo/region_model_best.pt",
        20301630,
        "160570dbd1e3f9977dd80267076622d59821e27447ab4194d7e5af2595c0ff68",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "river_model_best.infer.json",
        "yolo/river_model_best.infer.json",
        19,
        "73f83759cf5a3368ce5a7eda25fe8b6d032764f5720edfd73b0cd475f0039431",
    ),
    _Weight(
        "jjeyxb/mia-yolo",
        "river_model_best.pt",
        "yolo/river_model_best.pt",
        20363582,
        "54b5cb9eacc8361541953c1edc445d35767c1490571a1af4ff41cbae4020e367",
    ),
)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _ready(path: Path, weight: _Weight) -> bool:
    return path.is_file() and path.stat().st_size == weight.size and _digest(path) == weight.sha256


def _download(weight: _Weight, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(weight.url, headers={"User-Agent": "MIA-fetch-models"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {path.name} {done / 1048576:.0f}/{weight.size / 1048576:.0f} MB", end="")
    print()
    if part.stat().st_size != weight.size or _digest(part) != weight.sha256:
        part.unlink()
        raise SystemExit(f"{weight.url} 的內容與預期不同(大小或 SHA-256),已刪除")
    part.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="放權重的目錄")
    args = parser.parse_args(argv)
    for weight in WEIGHTS:
        path = args.models / weight.local
        if _ready(path, weight):
            print(f"已有 {weight.local}")
            continue
        print(f"下載 {weight.local} ← {weight.repo}")
        _download(weight, path)
    print(f"完成,權重在 {args.models}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
