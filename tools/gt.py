#!/usr/bin/env python3
"""Ground Truth 擷取與檢視。

GT 只用於**離線產生標註資料與量化 CV 準確率**,不參與線上決策。

三種擷取方式
------------
=========  ================  ==========  ==========  ================
子命令     機制              裝憑證?     導向?       能錄 Steam 版?
=========  ================  ==========  ==========  ================
``cdp``    瀏覽器除錯介面    不用        不用        否(僅網頁版)
``proxy``  MITM + 系統代理   要          要自己設    是(若客戶端遵循)
``local``  MITM + 行程重導   要          內建        是
=========  ================  ==========  ==========  ================

**預設用 ``cdp``。** 它不是中間人攔截 —— 瀏覽器自己解密,我們只是讀取結果,
所以完全不需要憑證信任、流量導向,也不怕客戶端做憑證固定。

``proxy`` / ``local`` 的存在意義是能錄 **Steam 桌面版**的對局。但雀魂是 Unity
原生客戶端,它的網路堆疊會不會查詢系統憑證信任清單,目前**尚未證實** ——
就算導向設定正確,仍可能因為憑證不被信任而連不上。

三者寫出的錄影檔格式完全相同,``inspect`` 不需要知道資料是怎麼來的。

用法::

    # 建議:用瀏覽器錄,零設定
    .venv/bin/python tools/gt.py cdp --out data/recordings/game1/ws.jsonl

    # 連到自己已開的 Chrome(需 --remote-debugging-port=9222)
    .venv/bin/python tools/gt.py cdp --connect http://localhost:9222 --out data/ws.jsonl

    # MITM:行程重導(mitmproxy 內建,不需要 Proxifier)
    .venv/bin/python tools/gt.py local --spec Jantama_MahjongSoul --out data/ws.jsonl

    # MITM:傳統代理(需自行把雀魂/瀏覽器指向 127.0.0.1:8080)
    .venv/bin/python tools/gt.py proxy --port 8080 --out data/ws.jsonl

    # 檢視錄影並轉成 MJAI 事件流
    .venv/bin/python tools/gt.py inspect data/ws.jsonl --actions --mjai-out data/game.mjai.jsonl
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mia.groundtruth import capture_addon
from mia.groundtruth.dump import DumpWriter
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.stream import MjaiDecoder
from mia.groundtruth.to_mjai import events_to_jsonl
from mia.mjai.events import MjaiEvent
from mia.utils.logging import setup_logging
from mia.utils.paths import DATA_DIR

DEFAULT_OUT = DATA_DIR / "recordings" / "ws_dump.jsonl"


# --------------------------------------------------------------------- cdp


def cmd_cdp(args: argparse.Namespace) -> int:
    from mia.calibration.canvas import Canvas
    from mia.groundtruth.cdp import (
        DEFAULT_MAJSOUL_URL,
        CdpCapture,
        PlaywrightMissingError,
    )
    from mia.live.control import ControlFile

    canvas = Canvas.parse(args.canvas)
    if args.canvas and canvas is None:
        print(f"看不懂的畫布尺寸 {args.canvas!r},格式是 1920x1080", file=sys.stderr)
        return 2
    if canvas is not None and args.connect:
        # 連到別人開的瀏覽器就不該去動它的視窗大小 —— 那個視窗不是我們的。
        print("--canvas 與 --connect 不能一起用:連過去的瀏覽器不歸我們管", file=sys.stderr)
        return 2

    try:
        with DumpWriter(args.out) as writer:
            capture = CdpCapture(writer, url_patterns=_patterns(args.filter))
            print(f"輸出檔案: {args.out}")
            if args.connect:
                stats = capture.run_connected(args.connect, duration=args.duration)
            else:
                stats = capture.run_launched(
                    url=args.url or DEFAULT_MAJSOUL_URL,
                    duration=args.duration,
                    headless=args.headless,
                    user_data_dir=args.user_data_dir,
                    canvas=canvas,
                    control=ControlFile(args.control) if args.control else None,
                )
    except PlaywrightMissingError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print()
    print(stats.summary())
    if stats.frames == 0:
        print("\n沒有錄到任何 frame。")
        if stats.sockets_seen == 0:
            print("  頁面沒有建立任何 WebSocket —— 是否真的進到對局畫面了?")
        else:
            print("  有 WebSocket 但都被網址過濾擋掉了,可用 --filter 放寬。")
    return 0


# --------------------------------------------------------------------- mitmproxy


def _run_mitmdump(out: Path, url_filter: str | None, extra: list[str]) -> int:
    """以子程序啟動 mitmdump。

    刻意用子程序而非 in-process:mitmproxy 自帶 asyncio 事件迴圈,
    塞進主程式只會讓兩邊的生命週期互相糾纏。
    """
    env = {**os.environ, capture_addon.ENV_OUTPUT: str(out)}
    if url_filter:
        env[capture_addon.ENV_URL_FILTER] = url_filter

    # 用 `-c "…mitmdump()"` 而不是 `-m mitmproxy.tools.main`:
    # 後者沒有 __main__ 區塊,執行後會立刻以 0 結束、什麼都不做。
    command = [
        sys.executable,
        "-c",
        "import sys; from mitmproxy.tools.main import mitmdump; sys.exit(mitmdump())",
        "-s",
        str(Path(capture_addon.__file__)),
        "--set",
        "termlog_verbosity=warn",
        *extra,
    ]
    print(f"輸出檔案: {out}")
    print("按 Ctrl-C 停止。\n")
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    print(f"啟動代理於 127.0.0.1:{args.port}")
    print("首次使用請到 http://mitm.it 安裝並信任 CA 憑證,")
    print("並自行把雀魂或瀏覽器的流量指向這個代理。")
    return _run_mitmdump(args.out, args.filter, ["--listen-port", str(args.port)])


def cmd_local(args: argparse.Namespace) -> int:
    """mitmproxy 的行程重導模式。

    底層在 macOS 是 Network Extension、Windows 是 WinDivert —— 等同於
    Proxifier 的功能,但 mitmproxy 內建,不需要額外安裝第三方軟體。
    """
    import mitmproxy_rs.local

    reason = mitmproxy_rs.local.LocalRedirector.unavailable_reason()
    if reason:
        print(f"這台機器無法使用 local 模式:{reason}", file=sys.stderr)
        return 1

    spec = args.spec or ""
    print(f"啟動行程重導{f'(僅 {spec})' if spec else '(全部流量)'}")
    print("首次執行時作業系統會要求核准系統延伸模組 —— 這一步無法自動化,")
    print("請到「系統設定 → 一般 → 登入項目與延伸功能」手動允許。")
    print("另外仍需安裝並信任 mitmproxy 的 CA 憑證(http://mitm.it)。")
    mode = f"local:{spec}" if spec else "local"
    return _run_mitmdump(args.out, args.filter, ["--mode", mode])


# --------------------------------------------------------------------- inspect


def cmd_inspect(args: argparse.Namespace) -> int:
    path: Path = args.file
    if not path.is_file():
        print(f"找不到錄影檔: {path}", file=sys.stderr)
        return 1

    schema = LiqiSchema.load()
    print(f"協定: {schema!r}\n")

    # 依連線分流的解析與轉換都在 MjaiDecoder 裡 —— 即時模式用的是同一個
    decoder = MjaiDecoder(schema)
    mjai_events: list[MjaiEvent] = []
    shown = 0

    for decoded in decoder.decode_file(path):
        mjai_events.extend(decoded.events)

        if args.actions and not decoded.message.is_action:
            continue
        if shown < args.limit:
            print(f"  {decoded.frame.timestamp:8.3f}s  {decoded.message}")
            for event in decoded.events:
                print(f"                → {event}")
            shown += 1

    stats = decoder.stats
    print()
    print(stats.summary())
    print(f"轉出 MJAI 事件 {len(mjai_events)} 個")
    if mjai_events:
        kinds = Counter(e.TYPE for e in mjai_events)
        print("  " + ", ".join(f"{k}×{v}" for k, v in kinds.most_common()))

    if args.mjai_out and mjai_events:
        args.mjai_out.parent.mkdir(parents=True, exist_ok=True)
        args.mjai_out.write_text(events_to_jsonl(mjai_events) + "\n", encoding="utf-8")
        print(f"MJAI 事件流已寫入 {args.mjai_out}")

    if stats.total == 0:
        print("\n錄影檔是空的 —— 沒有攔到任何 WebSocket frame。")
    elif stats.actions == 0:
        print("\n有攔到流量但沒有牌局動作 —— 可能只錄到大廳階段,對局中再錄一次。")
    return 0


# --------------------------------------------------------------------- CLI


def _patterns(value: str | None) -> tuple[str, ...]:
    from mia.groundtruth.cdp import DEFAULT_URL_PATTERNS

    if not value:
        return DEFAULT_URL_PATTERNS
    return tuple(p.strip() for p in value.split(",") if p.strip())


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:  # noqa: ARG001
    """把 SIGTERM 轉成 KeyboardInterrupt。

    ``cdp`` 的 ``_wait()`` 與 ``_run_mitmdump`` 都只認得 KeyboardInterrupt 才會
    走乾淨收尾(讓 Playwright 正常關閉瀏覽器 / 印出統計)。背景執行時終止訊號
    未必是 SIGINT,見 :mod:`tools.record` 同名函式的說明。
    """
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="雀魂 Ground Truth 擷取與檢視",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_capture_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="錄影檔輸出路徑")
        p.add_argument("--filter", help="逗號分隔的網址關鍵字;省略則用雀魂的預設網域")

    p_cdp = sub.add_parser("cdp", help="用瀏覽器除錯介面擷取(推薦,零前置設定)")
    add_capture_args(p_cdp)
    p_cdp.add_argument("--url", help="要開啟的網址,預設為網頁版雀魂")
    p_cdp.add_argument("--connect", metavar="ENDPOINT",
                       help="連到已在跑的瀏覽器,例如 http://localhost:9222")
    p_cdp.add_argument("--duration", type=float, metavar="SEC", help="錄製秒數")
    p_cdp.add_argument("--headless", action="store_true", help="無頭模式(通常不要,你得看畫面打牌)")
    p_cdp.add_argument("--user-data-dir", help="持久化設定檔目錄,可保留登入狀態")
    p_cdp.add_argument("--control", type=Path, metavar="FILE",
                       help="主程式用來即時改設定的 JSON 檔(見 mia.live.control)。"
                            "由 tools/ui.py 自動帶上,手動執行時不需要")
    p_cdp.add_argument("--canvas", metavar="WxH",
                       help="把視窗調到讓頁面 viewport 剛好是這個尺寸,例如 1920x1080。"
                            "畫面辨識就不必再猜牌桌邊界在哪")
    p_cdp.set_defaults(func=cmd_cdp)

    p_proxy = sub.add_parser("proxy", help="MITM 代理(需裝憑證,需自行導向)")
    add_capture_args(p_proxy)
    p_proxy.add_argument("--port", type=int, default=8080)
    p_proxy.set_defaults(func=cmd_proxy)

    p_local = sub.add_parser("local", help="MITM 行程重導(需裝憑證,導向內建)")
    add_capture_args(p_local)
    p_local.add_argument("--spec", help="行程名稱或 PID,例如 Jantama_MahjongSoul;省略則攔全部")
    p_local.set_defaults(func=cmd_local)

    p_inspect = sub.add_parser("inspect", help="檢視錄影並轉成 MJAI 事件流")
    p_inspect.add_argument("file", type=Path)
    p_inspect.add_argument("--actions", action="store_true", help="只列牌局動作")
    p_inspect.add_argument("--limit", type=int, default=40, help="最多列幾筆")
    p_inspect.add_argument("--mjai-out", type=Path, metavar="FILE",
                           help="把 MJAI 事件流寫成 JSON Lines")
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_dir=None)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
