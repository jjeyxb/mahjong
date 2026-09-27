#!/usr/bin/env python3
"""從天鳳公開牌譜存檔取得對局,轉成 Mortal 訓練要用的 mjai ``.json.gz``。

為什麼需要這個工具
------------------
M8(打法風格微調)的**第一個關卡不是 GRP 權重,是牌譜語料**。

README 的「唯一的前置關卡」寫的是「沒有公開的 GRP 權重,必須先用
``train_grp.py`` 自己訓一個 —— 這是動手的第一步」。但 ``train_grp.py`` 要的是
``cfg['dataset']['train_globs']`` 指向**一整批** ``.json.gz`` 牌譜;本專案自己
錄的只有個位數場次,而風格分群更需要大量玩家與場次才有統計意義。語料這件事
在任何文件裡都沒被列為阻擋項,``預期流程`` 第一步直接把它當成既有的東西。

三段路徑
--------
1. ``/sc/raw/list.cgi`` → 存檔檔案清單(JS 陣列,不是合法 JSON)
2. ``/sc/raw/dat/scc*.html.gz`` → 每小時一份的對局索引,含牌譜 ID 與規則字串
3. ``/5/mjlog2json.cgi?<id>`` → 天鳳**自己**把 mjlog 轉成 JSON
   (tenhou.net/6 格式)—— 所以不需要自己解析 mjlog XML

轉成 mjai 交給 ``mjai-reviewer --no-review --mjai-out``,那是 **Mortal 作者本人**
寫的 ``convlog``,格式相容性不必自己保證。見 ``--reviewer``。

用法::

    # 先看有哪些存檔、會挑到幾場,不下載
    python tools/fetch_tenhou.py --days 1 --dry-run

    # 實際抓 200 場(預設會跳過已經抓過的)
    python tools/fetch_tenhou.py --days 1 --limit 200

    # 只下載不轉換(轉換要先建好 mjai-reviewer)
    python tools/fetch_tenhou.py --days 1 --limit 200 --no-convert

取回來的東西放在 ``data/datasets/tenhou/``(``data/`` 在 gitignore)::

    raw/<log_id>.json                      天鳳原始格式
    mjai/<YYYY>/<MM>/<DD>/<log_id>.json.gz  訓練直接吃的

**mjai 依日期分層**是為了讓 train/val 用 glob 就切得開,與上游
``config.example.toml`` 的寫法一致(``'/path/to/dataset/2019/**/*.json.gz'``)。
日期直接取自牌譜 ID 的開頭(``2026091900gm-...``),不必另外問。

**原始格式留著**是刻意的:轉換器之後若修了 bug 要重轉,不必再跟天鳳要一次。
重轉時 ``raw/`` 已經有了,整批只花轉換的時間。
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.utils.logging import logger, setup_logging
from mia.utils.paths import PROJECT_ROOT

BASE = "https://tenhou.net"
LIST_URL = f"{BASE}/sc/raw/list.cgi"
ARCHIVE_URL = f"{BASE}/sc/raw/dat/{{name}}"
LOG_URL = f"{BASE}/5/mjlog2json.cgi?{{log_id}}"

#: 天鳳會擋沒有 Referer 的請求(``convlog/scripts/tenhou_dl.sh`` 也是這樣帶的)。
HEADERS = {
    "Referer": f"{BASE}/",
    "User-Agent": "Mozilla/5.0 (compatible; MIA/0.1; +https://github.com/jjeyxb/mahjong)",
    "Accept-Encoding": "gzip",
}

#: 每次請求之間睡多久。這是**別人的伺服器**,而且這批存檔是免費公開的 ——
#: 抓快一點省下的時間遠不值得被擋。
DEFAULT_DELAY = 1.0

#: 索引每一列長這樣(``scc`` 是段位戰鳳凰卓,只有它帶牌譜連結)::
#:
#:     00:01 | 22 | 四鳳南喰赤－ | <a href="...?log=2026091900gm-...">牌譜</a> | 名字(+40.5) ...
_ROW = re.compile(
    r"\|\s*(?P<rule>[^|]+?)\s*\|\s*<a href=\"[^\"]*?log=(?P<log_id>[^\"&]+)\""
)
_FILE = re.compile(r"file:'(?P<name>[^']+)'")


@dataclass(frozen=True, slots=True)
class Game:
    log_id: str
    rule: str

    @property
    def is_four_player(self) -> bool:
        return self.rule.startswith("四")

    @property
    def shard(self) -> tuple[str, str, str]:
        """從牌譜 ID 取出 ``(年, 月, 日)``。

        ID 長這樣:``2026091900gm-00a9-0000-740c25b8`` —— 前 8 碼就是日期,
        不必另外去問。分層是為了讓 train/val 用 glob 切得開。
        """
        return self.log_id[:4], self.log_id[4:6], self.log_id[6:8]

    @property
    def is_hanchan(self) -> bool:
        """南場(半莊)。

        **Mortal 只吃半莊** —— 上游 ``mjai-reviewer`` 對東風戰直接
        ``bail!("Mortal supports hanchan games only")``。不在這裡濾掉的話,
        會等到轉換那一步才一場一場失敗。
        """
        return "南" in self.rule


def fetch(url: str, *, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    # 帶了 Accept-Encoding: gzip,但只有部分路徑真的會壓
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def list_archives(days: int) -> list[str]:
    """最近 ``days`` 天的 ``scc`` 索引檔名,舊的在前。

    只要 ``scc`` —— 其他前綴(``sca`` / ``scb`` / ``scf``)不是段位戰鳳凰卓,
    或根本不帶牌譜連結。
    """
    listing = fetch(LIST_URL).decode("utf-8", errors="replace")
    names = [m.group("name") for m in _FILE.finditer(listing)]
    scc = sorted(n for n in names if n.startswith("scc") and n.endswith(".html.gz"))
    if not scc:
        raise SystemExit("清單裡沒有任何 scc 存檔 —— 天鳳的頁面格式可能改了")
    # 一天 24 份(每小時一份)
    return scc[-days * 24 :] if days > 0 else scc


def parse_archive(html: str) -> list[Game]:
    return [
        Game(log_id=m.group("log_id"), rule=m.group("rule"))
        for m in _ROW.finditer(html)
    ]


def collect_games(days: int, delay: float) -> list[Game]:
    games: list[Game] = []
    archives = list_archives(days)
    logger.info("索引檔 {} 份(最近 {} 天)", len(archives), days)
    for index, name in enumerate(archives, 1):
        try:
            html = fetch(ARCHIVE_URL.format(name=name)).decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("取 {} 失敗,跳過:{}", name, exc)
            continue
        found = parse_archive(html)
        games.extend(found)
        logger.debug("{} → {} 場", name, len(found))
        if index < len(archives):
            time.sleep(delay)
    return games


def convert(reviewer: Path, raw_path: Path, out_path: Path) -> bool:
    """天鳳 JSON → mjai ``.json.gz``。

    走 ``mjai-reviewer --no-review``,那條路只做轉換、不需要引擎也不需要權重。
    """
    try:
        result = subprocess.run(
            [
                str(reviewer),
                "--no-review",
                "--in-file",
                str(raw_path),
                "--mjai-out",
                "-",
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("轉換 {} 失敗:{}", raw_path.name, exc)
        return False

    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        logger.warning("轉換 {} 失敗:{}", raw_path.name, detail[-1] if detail else "沒有輸出")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb") as handle:
        handle.write(result.stdout)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--days", type=int, default=1, help="抓最近幾天的存檔,預設 1")
    parser.add_argument("--limit", type=int, default=200, help="最多抓幾場,預設 200")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "tenhou",
        help="輸出目錄",
    )
    parser.add_argument(
        "--reviewer",
        type=Path,
        default=PROJECT_ROOT
        / "engines/mortal/mjai-reviewer/target/release/mjai-reviewer.exe",
        help="mjai-reviewer 執行檔(轉換器)",
    )
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="每次請求間隔秒數")
    parser.add_argument("--no-convert", action="store_true", help="只下載,不轉 mjai")
    parser.add_argument("--dry-run", action="store_true", help="只看會挑到什麼,不下載")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    games = collect_games(args.days, args.delay)
    total = len(games)
    keep = [g for g in games if g.is_four_player and g.is_hanchan]
    logger.info(
        "索引到 {} 場,其中四人半莊 {} 場(濾掉三人與東風戰 {} 場)",
        total,
        len(keep),
        total - len(keep),
    )
    if not keep:
        raise SystemExit("沒有符合條件的對局")

    raw_dir = args.out / "raw"
    mjai_dir = args.out / "mjai"

    if args.dry_run:
        for game in keep[: args.limit]:
            print(f"  {game.rule}  {game.log_id}")
        logger.info("dry-run:會抓 {} 場到 {}", min(args.limit, len(keep)), args.out)
        return 0

    reviewer = args.reviewer
    if not args.no_convert and not reviewer.exists():
        raise SystemExit(
            f"找不到轉換器 {reviewer}\n"
            "  先建起來:\n"
            "    git clone --depth 1 https://github.com/Equim-chan/mjai-reviewer.git "
            "engines/mortal/mjai-reviewer\n"
            "    cd engines/mortal/mjai-reviewer && cargo build --release\n"
            "  或加 --no-convert 只下載原始牌譜。"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = converted = skipped = failed = 0
    #: ``--limit`` 算的是**處理過幾場**,不是下載幾場。
    #:
    #: 原本寫成 ``downloaded + skipped``,結果「raw 已經在快取裡、但還沒轉成
    #: mjai」那一類兩邊都不算,中止條件永遠碰不到 —— 實測 ``--limit 20``
    #: 轉出了 40 場。旗標說的是什麼就要算什麼。
    processed = 0

    for game in keep:
        if processed >= args.limit:
            break
        processed += 1
        raw_path = raw_dir / f"{game.log_id}.json"
        year, month, day = game.shard
        mjai_path = mjai_dir / year / month / day / f"{game.log_id}.json.gz"

        # 已經有了就不要再跟人家要一次
        if mjai_path.exists() or (args.no_convert and raw_path.exists()):
            skipped += 1
            continue

        if not raw_path.exists():
            try:
                raw = fetch(LOG_URL.format(log_id=game.log_id))
            except (urllib.error.URLError, OSError) as exc:
                logger.warning("下載 {} 失敗:{}", game.log_id, exc)
                failed += 1
                continue
            if not raw.strip():
                # 太舊的牌譜天鳳會回空的 —— 不是錯誤,就是沒了
                logger.debug("{} 沒有內容(牌譜可能已過期)", game.log_id)
                failed += 1
                continue
            raw_path.write_bytes(raw)
            downloaded += 1
            time.sleep(args.delay)

        if not args.no_convert and convert(reviewer, raw_path, mjai_path):
            converted += 1

        if processed % 25 == 0:
            logger.info("已處理 {} 場(下載 {}、轉換 {})", processed, downloaded, converted)

    logger.info(
        "完成:下載 {} 場、轉換 {} 場、跳過 {} 場(已存在)、失敗 {} 場",
        downloaded,
        converted,
        skipped,
        failed,
    )
    logger.info("原始牌譜:{}", raw_dir)
    if not args.no_convert:
        logger.info("mjai 語料:{}", mjai_dir)
        logger.info(
            "驗一下再拿去訓練:engines/mortal/Mortal/target/release/validate_logs.exe {}",
            mjai_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
