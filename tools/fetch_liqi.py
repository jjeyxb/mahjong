#!/usr/bin/env python3
"""從雀魂官方 CDN 取得 / 更新協定定義 ``liqi.json``。

為什麼需要這個工具
------------------
Steam 桌面版是 Unity 原生客戶端,協定定義沒有以檔案形式散布(實測 4.6 GB 的
StreamingAssets 全是混淆過的 ``.majset``,``Assembly-CSharp.dll`` 裡也沒有
protobuf descriptor)。但**網頁版**會公開提供同一份 ``liqi.json`` ——
兩者走的是同一套協定,所以從網頁版取得即可。

取得路徑分三段:

1. ``/1/version.json`` → 目前的客戶端版本
2. ``/1/resversion{version}.json`` → 資源索引(約 12 MB),查出 liqi.json 的
   版本前綴。**不能直接猜路徑** —— 資源有自己的版本號,通常落後客戶端版本。
3. ``/1/{prefix}/res/proto/liqi.json`` → 實際檔案

用法::

    .venv/bin/python tools/fetch_liqi.py              # 檢查並在有更新時下載
    .venv/bin/python tools/fetch_liqi.py --check      # 只比對,不寫檔
    .venv/bin/python tools/fetch_liqi.py --force      # 強制重新下載
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.groundtruth.schema import DEFAULT_LIQI_PATH, LiqiSchema
from mia.utils.logging import setup_logging

BASE_URL = "https://game.maj-soul.com/1"
VERSION_PATH = DEFAULT_LIQI_PATH.with_name("liqi.version.json")
TIMEOUT = 60


def _get(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def resolve_liqi_url() -> tuple[str, str, str]:
    """回傳 (liqi.json 的 URL, 客戶端版本, 資源前綴)。"""
    client_version = _get_json(f"{BASE_URL}/version.json")["version"]
    print(f"  客戶端版本: {client_version}")

    index = _get_json(f"{BASE_URL}/resversion{client_version}.json")
    entry = index.get("res", {}).get("res/proto/liqi.json")
    if entry is None:
        raise SystemExit("資源索引中找不到 res/proto/liqi.json,雀魂可能改了資源結構")
    prefix = entry["prefix"]
    print(f"  資源前綴  : {prefix}")

    return f"{BASE_URL}/{prefix}/res/proto/liqi.json", client_version, prefix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="取得雀魂協定定義 liqi.json")
    parser.add_argument("--check", action="store_true", help="只比對版本,不寫檔")
    parser.add_argument("--force", action="store_true", help="即使雜湊相同也重新寫入")
    parser.add_argument("--out", type=Path, default=DEFAULT_LIQI_PATH)
    args = parser.parse_args(argv)

    setup_logging("INFO", log_dir=None)

    current = json.loads(VERSION_PATH.read_text(encoding="utf-8")) if VERSION_PATH.is_file() else {}
    if current:
        print(f"目前版本  : {current.get('res_prefix')} (取得於 {current.get('fetched_at')})")

    print("查詢遠端版本…")
    url, client_version, prefix = resolve_liqi_url()

    if prefix == current.get("res_prefix") and not args.force:
        print(f"\n已是最新({prefix}),無需更新。")
        return 0

    print("下載 liqi.json…")
    raw = _get(url)
    digest = hashlib.sha256(raw).hexdigest()
    print(f"  {len(raw):,} 位元組  sha256={digest[:16]}…")

    # 寫檔前先確認真的解析得動,免得把壞檔案蓋上去
    try:
        schema = LiqiSchema(json.loads(raw))
    except Exception as exc:  # noqa: BLE001 - 任何解析失敗都不該寫檔
        print(f"\n下載到的檔案無法解析成協定 schema,已中止寫入:\n  {exc}", file=sys.stderr)
        return 1
    print(f"  驗證通過: {schema!r}")

    if args.check:
        print(f"\n有新版本可用({current.get('res_prefix')} → {prefix}),--check 模式未寫檔。")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    VERSION_PATH.write_text(
        json.dumps(
            {
                "source": url,
                "client_version": client_version,
                "res_prefix": prefix,
                "fetched_at": dt.date.today().isoformat(),
                "sha256": digest,
                "bytes": len(raw),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n已更新 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
