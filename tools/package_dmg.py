"""Create a clean offline DMG without recordings, logins, or update/test keys."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已組裝的 MIA/ 目錄")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("DMG 必須在 Mac 建置")
    app = args.source / "MIA.app"
    config = json.loads((app / "Contents/Resources/resources/release.json").read_text())
    if not (app / "Contents/MacOS/MIA").is_file():
        parser.error("缺少 MIA.app")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mia-dmg-", dir=args.output.parent) as temporary:
        folder = Path(temporary) / "MIA"
        folder.mkdir()
        shutil.copytree(app, folder / "MIA.app", symlinks=True)
        (folder / "install-info.json").write_text(
            json.dumps(
                {
                    "app_id": config["app_id"],
                    "version": config["version"],
                }
            )
        )
        (Path(temporary) / "安裝說明.txt").write_text(
            "將 MIA 資料夾完整拖到你選擇的位置，再開啟其中的 MIA.app。\n"
            "建議放在個人 Applications 資料夾；請勿直接在磁碟映像中使用。\n"
            "設定與錄影會保存在 MIA/userdata，更新 App 時會保留。\n"
            "此為尚未公證的內測版本。\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "MIA",
                "-srcfolder",
                temporary,
                "-format",
                "UDZO",
                "-ov",
                str(args.output.resolve()),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
