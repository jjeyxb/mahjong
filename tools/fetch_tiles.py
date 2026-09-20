#!/usr/bin/env python3
"""從雀魂官方 CDN 取得牌面圖集,切成 37 張模板存進 ``assets/tiles/<皮膚>/``。

為什麼從官方資源取而不是從遊戲畫面裁
------------------------------------
從畫面裁需要有人先告訴程式「這張是幾萬」。要嘛人工標(37 類、容易標錯,而且
一張錯的模板是**永久且靜默**的錯誤源),要嘛用封包當 GT 自動標(需要一份畫面
與封包配對的錄影)。

官方資源直接給出**已經標好的** 37 張牌面,而且是逐格排列、順序固定,不需要
任何猜測。實測把圖集模板拿去比對真實遊戲畫面,24 張已知答案的牌全數正確
(見 ``docs/decisions.md``)。

順帶的好處是**牌面皮膚可以整組換** —— 資源路徑裡的 ``mjpface_*`` 就是各種
牌面皮膚,換個 ``--skin`` 就能為它產生一整套模板。

取得路徑與 ``fetch_liqi.py`` 相同的三段式:

1. ``/1/version.json`` → 客戶端版本
2. ``/1/resversion{version}.json`` → 資源索引,查出該圖集的版本前綴
3. ``/1/{prefix}/{path}`` → 實際檔案

用法::

    .venv/bin/python tools/fetch_tiles.py                    # 預設皮膚
    .venv/bin/python tools/fetch_tiles.py --skin mjpface_25summer
    .venv/bin/python tools/fetch_tiles.py --list-skins       # 列出有哪些牌面皮膚
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

BASE_URL = "https://game.maj-soul.com/1"
TIMEOUT = 60

#: 圖集裡「手牌用」的那張。同目錄還有 mjp.png(牌河與副露用的 3D 貼圖),
#: 目前只有手牌在 CV 範圍內,所以只取這張。
ATLAS_FILE = "hand.png"
SKIN_DIR = "scene/Assets/Resource/mjpaimian"

#: 圖集是 10 欄 x 4 列。實測 800x516,每格 80x129。
COLS, ROWS = 10, 4

#: 每一列的花色,以及每列第 0 格是赤寶牌。第 4 列是字牌 1z~7z,後 3 格空白。
#: 這個順序是從圖集目視確認的,不是猜的 —— 見 docs/decisions.md 的截圖。
ROW_SUITS = ("s", "m", "p")

#: 牌面在每一格裡的內距(像素)。切掉圓角外框,只留牌面圖案。
FACE_INSET_Y, FACE_INSET_X = 10, 8

#: 圖集是帶 alpha 的去背圖,合成到牌面底色上而不是純白 —— 純白會在筆畫邊緣
#: 造出遊戲畫面裡不存在的高對比,反而拉低比對分數。
FACE_BACKGROUND = 235


def _get(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _resource_index() -> dict:
    client_version = json.loads(_get(f"{BASE_URL}/version.json"))["version"]
    print(f"  客戶端版本: {client_version}")
    return json.loads(_get(f"{BASE_URL}/resversion{client_version}.json"))["res"]


def list_skins(index: dict) -> list[str]:
    """有哪些牌面皮膚。"""
    pattern = re.compile(rf"^{re.escape(SKIN_DIR)}/([^/]+)/{re.escape(ATLAS_FILE)}$")
    return sorted({m.group(1) for k in index if (m := pattern.match(k))})


def labels() -> list[str | None]:
    """圖集每一格對應的牌(雀魂記法),空格為 None。"""
    out: list[str | None] = []
    for suit in ROW_SUITS:
        out += [f"{i}{suit}" for i in range(COLS)]  # 第 0 格是赤寶牌
    out += [f"{i}z" for i in range(1, 8)] + [None] * (COLS - 7)
    return out


def slice_atlas(atlas: np.ndarray) -> dict[str, np.ndarray]:
    """把圖集切成 {牌: BGR 牌面影像}。"""
    if atlas.ndim != 3 or atlas.shape[2] != 4:
        raise SystemExit(f"預期帶 alpha 的圖集,拿到 shape={atlas.shape}")
    height, width = atlas.shape[:2]
    if height % ROWS or width % COLS:
        raise SystemExit(f"圖集尺寸 {width}x{height} 不是 {COLS}x{ROWS} 的整數倍")

    colour = atlas[..., :3].astype(np.float32)
    alpha = atlas[..., 3:4].astype(np.float32) / 255.0
    flat = (colour * alpha + FACE_BACKGROUND * (1 - alpha)).astype(np.uint8)

    ch, cw = height // ROWS, width // COLS
    faces: dict[str, np.ndarray] = {}
    for index, label in enumerate(labels()):
        if label is None:
            continue
        row, col = divmod(index, COLS)
        cell = flat[row * ch : (row + 1) * ch, col * cw : (col + 1) * cw]
        faces[label] = cell[
            FACE_INSET_Y : ch - FACE_INSET_Y, FACE_INSET_X : cw - FACE_INSET_X
        ]
    return faces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="取得雀魂牌面模板")
    parser.add_argument("--skin", default="mjpface_default", help="牌面皮膚目錄名")
    parser.add_argument("--list-skins", action="store_true", help="列出可用皮膚後結束")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="輸出目錄,預設 assets/tiles/<皮膚>/",
    )
    args = parser.parse_args(argv)

    print("查詢資源索引…")
    index = _resource_index()

    if args.list_skins:
        for skin in list_skins(index):
            print(f"  {skin}")
        return 0

    key = f"{SKIN_DIR}/{args.skin}/{ATLAS_FILE}"
    entry = index.get(key)
    if entry is None:
        print(f"\n資源索引中找不到 {key}", file=sys.stderr)
        print("可用的皮膚:", ", ".join(list_skins(index)), file=sys.stderr)
        return 1

    print(f"下載圖集({args.skin})…")
    raw = _get(f"{BASE_URL}/{entry['prefix']}/{key}")
    atlas = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if atlas is None:
        print("\n下載到的檔案不是能解碼的影像", file=sys.stderr)
        return 1
    print(f"  {len(raw):,} 位元組,{atlas.shape[1]}x{atlas.shape[0]}")

    faces = slice_atlas(atlas)
    expected = sum(label is not None for label in labels())
    if len(faces) != expected:
        print(f"\n切出 {len(faces)} 張,預期 {expected} 張", file=sys.stderr)
        return 1

    out_dir = args.out or Path(__file__).resolve().parents[1] / "assets/tiles" / args.skin
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, face in sorted(faces.items()):
        cv2.imwrite(str(out_dir / f"{label}.png"), face)

    sample = next(iter(faces.values()))
    print(f"\n已寫入 {len(faces)} 張模板({sample.shape[1]}x{sample.shape[0]})→ {out_dir}")
    print(f"  資源前綴: {entry['prefix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
