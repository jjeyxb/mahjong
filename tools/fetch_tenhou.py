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
1. ``/sc/raw/list.cgi`` → 存檔檔案清單(JS 陣列,不是合法 JSON)。
   加上 ``?old`` 換成**日檔**清單 —— 語料要放大就靠它,見 :data:`LIST_URL_OLD`。
2. ``/sc/raw/dat/scc*.html.gz`` → 對局索引,含牌譜 ID 與規則字串
3. ``/5/mjlog2json.cgi?<id>`` → 天鳳**自己**把 mjlog 轉成 JSON
   (tenhou.net/6 格式)—— 所以不需要自己解析 mjlog XML

轉成 mjai 交給 ``mjai-reviewer --no-review --mjai-out``,那是 **Mortal 作者本人**
寫的 ``convlog``,格式相容性不必自己保證。見 ``--reviewer``。

用法::

    # 先看有哪些存檔、會挑到幾場,不下載
    python tools/fetch_tenhou.py --days 1 --dry-run

    # 實際抓 200 場(預設會跳過已經抓過的)
    python tools/fetch_tenhou.py --days 1 --limit 200

    # **放大語料**:改吃日檔,可以回溯到 2026-01-01
    python tools/fetch_tenhou.py --since 20260101 --limit 20000

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mia.utils.logging import logger, setup_logging
from mia.utils.paths import PROJECT_ROOT

BASE = "https://tenhou.net"
LIST_URL = f"{BASE}/sc/raw/list.cgi"

#: 舊存檔清單。**這才是能把語料放大的那一條路。**
#:
#: 預設的 `list.cgi` 是一個滾動視窗:只有最近 10 天,而且是**每小時**一份
#: (230 份索引換到 5543 場四人南,那是天花板)。`?old` 給的是**每天**一份、
#: 按年份分目錄,實測回溯到 ``2026/scc20260101.html.gz`` —— 261 份日檔、
#: 每份 400~550 場四人南,合計約 13 萬場。索引請求數還從 6264 降到 261。
#:
#: 兩邊**不重疊**:日檔到 2026-09-18,小時檔從 2026-09-19 開始。
LIST_URL_OLD = f"{LIST_URL}?old"

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
    # 名字那一段是**選擇性**的,而且每一個字元類都排除換行。
    #
    # 兩個理由都被測試釘住了:設成必要的話,萬一哪天有一列沒帶名字,整場就被
    # 靜默丟掉;不排除換行的話,`[^|]*` 會跨行吃到**下一列**去,把別人的名字
    # 接到這一場上 —— 那個錯誤不會報,只會讓玩家頻率表算錯。
    r"(?:[^|\n]*\|(?P<names>[^<\n]*))?"
)

#: 索引列尾端的「名字(±分數)」。**這是一份免費的玩家頻率表** —— 不必下載任何
#: 一場牌譜就知道誰打得多,而風格分群缺的正是「每個人夠多場」。
#:
#: 名字本身可以含括號與空白,所以用非貪婪比對,把分數那一段當結束錨點。
_NAME = re.compile(r"(?P<name>.+?)\((?P<score>[-+][\d.]+)\)\s*")
_FILE = re.compile(r"file:'(?P<name>[^']+)'")


@dataclass(frozen=True, slots=True)
class Game:
    log_id: str
    rule: str
    #: 索引列上的四個名字。順序是**終局名次**而不是座位順序(天鳳的索引照分數排),
    #: 所以只拿來數「誰出現得多」,不當訓練資料 —— 座位要從 mjai 的
    #: ``start_game.names`` 拿。
    names: tuple[str, ...] = ()

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
        Game(
            log_id=m.group("log_id"),
            rule=m.group("rule"),
            names=tuple(
                n.group("name").strip() for n in _NAME.finditer(m.group("names") or "")
            ),
        )
        for m in _ROW.finditer(html)
    ]


def list_daily_archives(since: str | None, until: str | None) -> list[str]:
    """``?old`` 清單裡的 ``scc`` **日檔**,舊的在前。

    檔名長這樣:``2026/scc20260918.html.gz`` —— 前面帶年份目錄,而
    :data:`ARCHIVE_URL` 是直接把名字接在 ``dat/`` 後面,所以不需要特別處理。

    Args:
        since: 起始日 ``YYYYMMDD``(含)。``None`` 表示能拿多舊就多舊。
        until: 結束日 ``YYYYMMDD``(含)。
    """
    listing = fetch(LIST_URL_OLD).decode("utf-8", errors="replace")
    names = sorted(
        m.group("name") for m in _FILE.finditer(listing) if "/scc" in m.group("name")
    )
    if not names:
        raise SystemExit("?old 清單裡沒有 scc 日檔 —— 天鳳的頁面格式可能改了")

    def day_of(name: str) -> str:
        # '2026/scc20260918.html.gz' → '20260918'
        return name.split("/")[-1][3:11]

    if since:
        names = [n for n in names if day_of(n) >= since]
    if until:
        names = [n for n in names if day_of(n) <= until]
    if not names:
        raise SystemExit(f"{since or '(最舊)'} ~ {until or '(最新)'} 之間沒有日檔")
    return names


def collect_games(archives: list[str], delay: float, cache_dir: Path | None = None) -> list[Game]:
    """把每一份索引解析成對局清單。

    Args:
        cache_dir: 解析結果快取到哪。``None`` 表示不快取。

    快取的理由不是省時間,是**別重複跟人家要同一份東西**。261 份日檔就是 261 個
    請求、4 分半;而放大語料是一趟要跑好幾小時的事,中途被打斷(關機、斷網)
    重跑就再來一次。過去某一天的索引內容不會變,快取是安全的 ——
    只有「最近那幾天」還在長,但小時檔那條路的檔名本身就帶小時,同樣不會變。
    """
    games: list[Game] = []
    cached = 0
    logger.info("索引檔 {} 份", len(archives))
    for index, name in enumerate(archives, 1):
        # 檔名帶目錄(日檔是 '2026/scc...'),攤平成單一檔名才好放
        # 副檔名從 .txt 換成 .tsv 是刻意的:快取格式加上了玩家名字,
        # 舊檔案少那一段。換名字讓舊快取自然被忽略,不必寫版本判斷。
        path = (cache_dir / f"{name.replace('/', '_')}.tsv") if cache_dir else None
        if path is not None and path.exists():
            for line in path.read_text("utf-8").splitlines():
                if not line:
                    continue
                log_id, rule, *names = line.split("\t")
                games.append(Game(log_id=log_id, rule=rule, names=tuple(names)))
            cached += 1
            continue

        try:
            html = fetch(ARCHIVE_URL.format(name=name)).decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("取 {} 失敗,跳過:{}", name, exc)
            continue
        found = parse_archive(html)
        games.extend(found)
        logger.debug("{} → {} 場", name, len(found))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join("\t".join((g.log_id, g.rule, *g.names)) + "\n" for g in found),
                encoding="utf-8",
            )
        if index < len(archives):
            time.sleep(delay)

    if cached:
        logger.info("其中 {} 份來自快取,沒有再跟天鳳要", cached)
    return games


def player_counts(games: list[Game]) -> Counter[str]:
    """誰在這批索引裡出現過幾次。"""
    counter: Counter[str] = Counter()
    for game in games:
        counter.update(game.names)
    return counter


def report_index_stats(games: list[Game]) -> None:
    """只看索引就能回答「抓多少場才夠分群」。

    這份報表的存在理由:**總場數多不等於每個人夠多場。** 風格統計是 per-player 的,
    2 萬場均勻散在幾千個玩家身上,每人只有十幾場 —— 那種統計量撐不起分群。
    而要知道這件事**完全不必下載任何牌譜**,索引列上就寫著四個名字。
    """
    counts = player_counts(games)
    ranked = counts.most_common()
    print(f"索引裡的對局: {len(games)}    出現過的玩家: {len(counts)}\n")

    print(f"{'取前 N 名玩家':>14} {'涵蓋對局數':>10} {'佔全部':>7} {'第 N 名幾場':>11}")
    for top in (50, 100, 200, 500, 1000, 2000):
        if top > len(ranked):
            break
        focus = {name for name, _ in ranked[:top]}
        covered = sum(1 for g in games if focus & set(g.names))
        print(
            f"{top:>14} {covered:>10} {covered / len(games):>6.1%} "
            f"{ranked[top - 1][1]:>11}"
        )

    print("\n出現次數分位:")
    values = np.array([c for _, c in ranked])
    for label, value in (
        ("最多", values.max()),
        ("p99", np.percentile(values, 99)),
        ("p90", np.percentile(values, 90)),
        ("中位數", np.median(values)),
    ):
        print(f"  {label:>6} {value:>8.0f} 場")


def focus_filter(games: list[Game], top: int) -> list[Game]:
    """只留「出現次數前 ``top`` 名的玩家」參與的對局。

    **一個人就夠**,不需要四個都在名單上 —— `train.py` 會把 ``player_names``
    傳給 ``GameplayLoader``,只有名單上那些人的決策會變成訓練樣本,
    同一桌其他三家只是環境。所以一場只要有一個目標玩家就有價值。
    """
    counts = player_counts(games)
    keep = {name for name, _ in counts.most_common(top)}
    focused = [g for g in games if keep & set(g.names)]
    logger.info(
        "鎖定出現最多的 {} 個玩家 → {} 場(原本 {} 場)", top, len(focused), len(games)
    )
    return focused


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
    parser.add_argument(
        "--days", type=int, default=1, help="抓最近幾天的小時檔,預設 1(上限約 10 天)"
    )
    parser.add_argument(
        "--since",
        metavar="YYYYMMDD",
        help="改用**日檔**清單,從這一天開始。語料要放大就用這個 —— "
        "小時檔只有最近 10 天(約 2871 場四人南),日檔回溯到 2026-01-01(約 11 萬場)",
    )
    parser.add_argument("--until", metavar="YYYYMMDD", help="日檔的結束日(含)")
    parser.add_argument(
        "--index-stats",
        action="store_true",
        help="只解析索引、印出玩家出現次數分布就結束 —— 不下載任何牌譜。"
        "用來回答「抓多少場才夠分群」",
    )
    parser.add_argument(
        "--focus-players",
        type=int,
        metavar="N",
        help="只抓「出現次數前 N 名玩家」參與的對局。風格統計是 per-player 的,"
        "固定的下載預算集中在常打的人身上,每個人的場數才夠",
    )
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

    if args.since or args.until:
        archives = list_daily_archives(args.since, args.until)
        logger.info(
            "日檔 {} 份({} ~ {})",
            len(archives),
            args.since or "最舊",
            args.until or "最新",
        )
    else:
        archives = list_archives(args.days)
        logger.info("小時檔 {} 份(最近 {} 天)", len(archives), args.days)
    games = collect_games(archives, args.delay, args.out / "index")
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

    if args.index_stats:
        report_index_stats(keep)
        return 0

    if args.focus_players:
        keep = focus_filter(keep, args.focus_players)
        if not keep:
            raise SystemExit("鎖定玩家之後沒有對局了")

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
