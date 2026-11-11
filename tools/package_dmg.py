"""Create a clean offline DMG without recordings, logins, or update/test keys."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

README_NAME = "安裝說明 - Read Me.txt"

README = """\
MIA 安裝說明
============

安裝
  把 MIA.app 拖到旁邊的「應用程式」（Applications）資料夾，再從「應用程式」開啟。
  請不要直接在這個磁碟映像裡執行：那樣無法自動更新。

第一次開啟
  這是尚未經 Apple 公證的內測版。如果系統說無法驗證開發者，到「系統設定 →
  隱私權與安全性」，往下找到 MIA，按「仍要打開」。
  開啟畫面辨識或 AI 建議時，請到「系統設定 → 隱私權與安全性 → 螢幕錄製」允許 MIA，
  然後完整結束 MIA 再重新開啟。

資料存放位置
  設定、雀魂登入資料與錄影：~/Library/Application Support/MIA
  快取：~/Library/Caches/MIA
  紀錄：~/Library/Logs/MIA
  更新 MIA 時這些資料都會保留。

移除
  1. 想連資料一起刪的話，先在 MIA 的設定頁按「清除所有資料…」（MIA 會結束並刪掉上面三個資料夾）。
  2. 把「應用程式」裡的 MIA.app 丟到垃圾桶。
  3. 到「系統設定 → 隱私權與安全性 → 螢幕錄製」選 MIA，按「－」移除。


MIA Installation
================

Install
  Drag MIA.app onto the Applications folder next to it, then open it from Applications.
  Do not run it from inside this disk image; automatic updates will not work there.

First launch
  This is a preview build that has not been notarized by Apple. If macOS says the developer
  cannot be verified, open System Settings > Privacy & Security, scroll down to MIA and
  click "Open Anyway".
  When you turn on screen recognition or AI advice, allow MIA in System Settings >
  Privacy & Security > Screen Recording, then quit MIA completely and open it again.

Where your data is stored
  Settings, Mahjong Soul login and recordings: ~/Library/Application Support/MIA
  Cache: ~/Library/Caches/MIA
  Logs: ~/Library/Logs/MIA
  All of this is kept when MIA is updated.

Uninstall
  1. To delete your data as well, first click "Clear all data…" on MIA's settings page
     (MIA quits and deletes the three folders above).
  2. Move MIA.app from Applications to the Trash.
  3. In System Settings > Privacy & Security > Screen Recording, select MIA and click "-".
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已組裝的 MIA/ 目錄(裡面有 MIA.app)")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("DMG 必須在 Mac 建置")
    app = args.source / "MIA.app"
    if not (app / "Contents/MacOS/MIA").is_file():
        parser.error("缺少 MIA.app")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mia-dmg-", dir=args.output.parent) as temporary:
        root = Path(temporary)
        # 一般 Mac App 的裝法:把 MIA.app 拖到旁邊的「應用程式」捷徑上。
        # 資料不放在 App 旁邊(見 mia.distribution.layout),所以 DMG 裡也不需要資料夾。
        shutil.copytree(app, root / "MIA.app", symlinks=True)
        (root / "Applications").symlink_to("/Applications")
        (root / README_NAME).write_text(README, encoding="utf-8")
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
